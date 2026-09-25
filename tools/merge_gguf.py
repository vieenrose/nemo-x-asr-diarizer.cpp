#!/usr/bin/env python3
"""Merge several GGUF models into one file, namespacing each one's tensors and metadata.

Why this exists: the composite runs two models in one process, and "merge them" has two very different
meanings. This does the honest, mechanical one - ONE FILE, ONE MMAP, ONE DEPLOYMENT UNIT - and deliberately
does not pretend to be the other (one runtime, one thread pool), which is a graph-porting project, not a
container operation. See docs/pipeline-design.md.

Layout of the output:
  * every tensor of every input is copied, name-prefixed with that input's prefix
  * every KV pair is copied, key-prefixed the same way
  * the FIRST input keeps the unprefixed namespace, so a loader that has no notion of prefixes (here:
    audiocpp, whose tensors are already architecture-namespaced) can read the bundle unchanged
  * tensor data is copied verbatim and re-aligned to each file's own `general.alignment` (default 32)

Reading it back:
  * audiocpp: pass the bundle path, nothing to change
  * CrispASR: set CRISPASR_GGUF_PREFIX=asr. (the loader tries the prefixed name first, then the bare name,
    so single-model files keep working with the prefix set)

Usage:
  merge_gguf.py OUT.gguf --model native=DIA.gguf --model asr.=ASR.gguf [--model pfx=B.gguf ...]
  merge_gguf.py OUT.gguf --describe DIA.gguf ASR.gguf
"""
import struct
import sys

# GGUF metadata value types. 8=string, 9=array, everything else is a fixed-width scalar. This is the GGUF
# spec numbering, NOT ggml's tensor-type enum - mixing them up is how you end up parsing nonsense.
SCALARS = {0: ('<B', 1), 1: ('<b', 1), 2: ('<H', 2), 3: ('<h', 2), 4: ('<I', 4), 5: ('<i', 4),
           6: ('<f', 4), 7: ('<d', 8), 10: ('<Q', 8), 11: ('<q', 8)}


class Reader:
    def __init__(self, path):
        self.path = path
        self.buf = open(path, 'rb').read()
        self.pos = 0

    def take(self, n):
        out = self.buf[self.pos:self.pos + n]
        if len(out) != n:
            raise ValueError('%s: truncated at %d' % (self.path, self.pos))
        self.pos += n
        return out

    def u32(self):
        return struct.unpack('<I', self.take(4))[0]

    def u64(self):
        return struct.unpack('<Q', self.take(8))[0]

    def string(self):
        return self.take(self.u64()).decode('utf-8')

    def value(self, t):
        # Values are kept in a RE-ENCODABLE form (type tag + payload), because arrays of strings exist in
        # this metadata (audiocpp.tensor_names, audiocpp.embedded_files.names) and a reader that decodes them
        # to plain python loses the element type needed to write them back out.
        if t == 8:
            return (8, self.take(self.u64()))
        if t == 9:
            et = self.u32()
            n = self.u64()
            return (9, (et, [self.value(et)[1] for _ in range(n)]))
        fmt, size = SCALARS[t]
        return (t, struct.unpack(fmt, self.take(size))[0])


class Tensor:
    __slots__ = ('name', 'dims', 'dtype', 'src_offset', 'rel_offset', 'nbytes')


def read_model(path):
    r = Reader(path)
    if r.take(4) != b'GGUF':
        raise ValueError('%s: not a GGUF' % path)
    ver = r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()
    kv = []
    for _ in range(n_kv):
        key = r.string()
        vtype = r.u32()
        kv.append((key, vtype, r.value(vtype)))
    tensors = []
    for _ in range(n_tensors):
        t = Tensor()
        t.name = r.string()
        n_dims = r.u32()
        t.dims = [r.u64() for _ in range(n_dims)]
        t.dtype = r.u32()
        t.src_offset = r.u64()
        tensors.append(t)
    align = 32
    for key, _vtype, (_t, val) in kv:
        if key == 'general.alignment':
            align = int(val)
    # The data section starts at the current offset, padded up to the alignment.
    data_start = (r.pos + align - 1) // align * align
    for t in tensors:
        # nbytes is implied by the next tensor's offset (or the file size for the last one), which is the
        # only reliable way: ggml pads each tensor's blob to the alignment, so dtype-size math alone lies.
        t.rel_offset = t.src_offset
    return dict(path=path, ver=ver, kv=kv, tensors=tensors, align=align,
                data_start=data_start, size=len(r.buf), buf=r.buf)


def tensor_nbytes(model, idx):
    offs = sorted((t.src_offset, i) for i, t in enumerate(model['tensors']))
    pos = dict((i, o) for o, i in offs)
    my = pos[idx]
    nxt = [o for o, _ in offs if o > my]
    # Offsets are relative to the start of the DATA SECTION, so the last tensor ends at
    # (file size - data start), NOT at the file size. Using the file size inflates the last payload by the
    # header length, the running cursor drifts ahead of what has actually been written, `pad` goes
    # negative, and every tensor after it lands early - silently, because writing b'\0' * negative is a
    # no-op rather than an error.
    end = nxt[0] if nxt else (model['size'] - model['data_start'])
    return end - my


def build(out_path, models):
    """models: list of (prefix, model_dict). First entry keeps the unprefixed namespace."""
    merged_kv = []
    merged_tensors = []
    payloads = []  # (bytes,) in output order
    cursor = 0
    align = max(m['align'] for _, m in models)

    # KV first (header layout: magic, version, counts, KV block, tensor-info block, data).
    for mi, (prefix, m) in enumerate(models):
        for key, vtype, val in m['kv']:
            k = key if mi == 0 else prefix + key
            if any(existing == k for existing, _, _ in merged_kv):
                raise ValueError('KV key collision after prefixing: %s' % k)
            merged_kv.append((k, vtype, val))

    # Lay out tensor data now so the tensor-info block can carry the final offsets.
    plan = []
    for prefix, m in models:
        for i, t in enumerate(m['tensors']):
            nb = tensor_nbytes(m, i)
            pad = (align - (cursor % align)) % align
            cursor += pad
            plan.append((prefix, m, i, cursor, nb))
            cursor += nb
    data_bytes = cursor

    header = bytearray()
    header += b'GGUF' + struct.pack('<I', models[0][1]['ver'])
    header += struct.pack('<Q', len(plan))
    header += struct.pack('<Q', len(merged_kv))
    for key, vtype, (vtag, val) in merged_kv:
        header += struct.pack('<Q', len(key.encode())) + key.encode()
        header += _encode_value(vtype, vtag, val)

    # Tensor info block, with offsets relative to the start of the data section.
    for prefix, m, i, off, nb in plan:
        name = m['tensors'][i].name
        t = m['tensors'][i]
        full = name if not prefix else prefix + name
        nb_name = len(full.encode())
        header += struct.pack('<Q', nb_name) + full.encode()
        header += struct.pack('<I', len(t.dims))
        for d in t.dims:
            header += struct.pack('<Q', d)
        header += struct.pack('<I', t.dtype)
        header += struct.pack('<Q', off)

    data_start = (len(header) + align - 1) // align * align
    header += b'\0' * (data_start - len(header))

    with open(out_path, 'wb') as out:
        out.write(header)
        for prefix, m, i, off, nb in plan:
            pad = off - out.tell()
            if pad:
                out.write(b'\0' * pad)
            start = m['data_start'] + m['tensors'][i].src_offset
            out.write(m['buf'][start:start + nb])

    total = len(header) + data_bytes
    return dict(kv=len(merged_kv), tensors=len(plan), bytes=total,
                asr=sum(1 for p, _, _, _, _ in plan if p),
                native=sum(1 for p, _, _, _, _ in plan if not p))


def _encode_value(vtype, vtag, val):
    if vtag == 8:
        b = val if isinstance(val, bytes) else val.encode()
        return struct.pack('<I', vtype) + struct.pack('<Q', len(b)) + b
    if vtag == 9:
        et, items = val
        out = struct.pack('<I', vtype) + struct.pack('<I', et) + struct.pack('<Q', len(items))
        for v in items:
            # Elements carry NO type field: the array header declared the element type once. Writing one
            # per element adds 4 bytes each and desynchronises the whole KV block (the diar metadata has
            # arrays of strings, so this bit immediately).
            out += _encode_bare(et, v)
        return out
    fmt, _size = SCALARS[vtype]
    return struct.pack('<I', vtype) + struct.pack(fmt, val)


def _encode_bare(t, v):
    if t == 8:
        b = v if isinstance(v, bytes) else v.encode()
        return struct.pack('<Q', len(b)) + b
    fmt, _size = SCALARS[t]
    return struct.pack(fmt, v)


def describe(paths):
    for p in paths:
        m = read_model(p)
        print('%-46s %4d tensors %3d kv  %7.1f MB  align=%d  first tensors: %s' % (
            p, len(m['tensors']), len(m['kv']), m['size'] / 1048576.0, m['align'],
            ', '.join(t.name for t in m['tensors'][:3])))


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    out = argv[1]
    if out == '--describe':
        describe(argv[2:])
        return 0
    specs = argv[2:]
    models = []
    i = 0
    while i < len(specs):
        spec = specs[i]
        if not spec.startswith('--model'):
            print('expected --model PREFIX=FILE, got %r' % spec)
            return 2
        if spec.startswith('--model='):
            body = spec[len('--model='):]
        else:
            i += 1
            if i >= len(specs):
                print('--model needs PREFIX=FILE')
                return 2
            body = specs[i]
        prefix, path = body.split('=', 1)
        models.append((prefix, read_model(path)))
        i += 1
    info = build(out, models)
    print('wrote %s: %d tensors (%d native + %d prefixed), %d kv, %.1f MB' % (
        out, info['tensors'], info['native'], info['asr'], info['kv'], info['bytes'] / 1048576.0))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
