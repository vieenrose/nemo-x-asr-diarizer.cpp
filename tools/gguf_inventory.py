#!/usr/bin/env python3
"""Report a GGUF's tensor inventory by TYPE and by BITS PER ELEMENT.

Why this file exists: during this session several conclusions were built on dtype histograms produced by
reading GGUF type ids with the wrong enum. These two vendored ggml copies number the quantised types
`Q4_0=2, Q4_1=3, Q5_0=6, Q5_1=7, Q8_0=8`; the older llama.cpp numbering (where 8 means q5_0 and 7 means q8_0) is
easy to carry in from memory and produced a confident, entirely wrong story about both models' weights.

The fix is not to memorise an enum - it is to print a number that cannot be mislabelled. **Bits per element is
derived from the payload size and the element count alone**, so it needs no enum at all: a tensor declared
q5_0 that occupies 8.5 bits per element is visibly not a q5_0, whatever the type id says. ggml's own
`gguf_get_tensor_size` is the other half of the check, and where the library and an arithmetic expectation
disagree about a format, the library is right.

Usage: gguf_inventory.py MODEL.gguf [MODEL.gguf ...]
"""
import collections
import os
import sys

sys.path.insert(0, __file__.rsplit('/', 1)[0])
import merge_gguf as M   # noqa: E402

# Type ids are DERIVED from a ggml header, never hardcoded. This file exists because a hand-written enum
# produced a confident, wrong story twice in one session: first by reading type 8 as q5_0 when these trees
# number it Q8_0, and then by labelling type 2 BF16 when it is Q4_0 (BF16 is 30). Both trees in this repo carry
# the SAME enum - Q4_0=2, Q4_1=3, Q5_0=6, Q5_1=7, Q8_0=8, BF16=30 - but a map written from memory is exactly
# the thing that goes wrong, so parse it instead. The density column remains the check that needs no enum at all.
import re as _re

DEFAULT_HEADERS = [
    os.path.join(os.path.dirname(__file__), '..', '..', 'ref', 'crispasr', 'ggml', 'include', 'ggml.h'),
    os.path.join(os.path.dirname(__file__), '..', '..', 'ref', 'audiocpp', 'external', 'ggml', 'include', 'ggml.h'),
]


def load_type_names(headers=None):
    names = {}
    for h in (headers or DEFAULT_HEADERS):
        try:
            text = open(h, encoding='utf-8', errors='replace').read()
        except OSError:
            continue
        start = text.find('enum ggml_type')
        if start < 0:
            continue
        body = text[start:text.find('};', start)]
        for name, value in _re.findall(r'(GGML_TYPE_[A-Z0-9_]+)\s*=\s*(\d+)', body):
            names.setdefault(int(value), name.replace('GGML_TYPE_', ''))
        break
    return names or {0: 'F32', 1: 'F16', 2: 'Q4_0', 6: 'Q5_0', 8: 'Q8_0', 30: 'BF16'}


NAMES = load_type_names()

def payload_bytes(model, i):
    t = model['tensors'][i]
    n = 1
    for d in t.dims:
        n *= d
    if t.dtype in (6, 7, 8):        # block types: exact size from the element count
        return (n // 32) * {6: 22, 7: 34, 8: 34}[t.dtype]
    return M.tensor_nbytes(model, i)


def audit(path):
    model = M.read_model(path)
    per_type = collections.defaultdict(lambda: [0, 0, 0])   # tensors, elements, bytes
    print('== %s  (%d tensors, %.1f MB)' % (path, len(model['tensors']), model['size'] / 1048576.0))
    for i, t in enumerate(model['tensors']):
        n = 1
        for d in t.dims:
            n *= d
        nb = payload_bytes(model, i)
        e = per_type[t.dtype]
        e[0] += 1
        e[1] += n
        e[2] += nb
    print('   %-8s %5s %7s %10s %12s   %s' % ('type', 'n', 'nelem', 'MB', 'bits/elem', 'note'))
    for dtype, (count, nelem, nbytes) in sorted(per_type.items(), key=lambda kv: -kv[1][2]):
        bits = (nbytes * 8.0 / nelem) if nelem else 0.0
        name = NAMES.get(dtype, 'type%d' % dtype)
        note = ''
        # Cross-check the label against the density. These are the densities of the named types, so a
        # disagreement means the id was read with the wrong enum (or the file is unusual).
        expected = {'Q4_0': 4.5, 'Q4_1': 5.0, 'Q5_0': 5.5, 'Q5_1': 6.5, 'Q8_0': 8.5, 'Q8_1': 9.0,
                    'F16': 16.0, 'F32': 32.0, 'BF16': 16.0}
        # F16 and BF16 share a density, so bits/element cannot separate them - only the enum can, and
        # that is exactly why the label column exists next to the density column.
        if name in expected and abs(bits - expected[name]) > 0.6:
            note = 'DENSITY MISMATCH: %.1f bits/elem is not %s' % (bits, name)
        print('   %-8s %5d %7.1fM %10.1f %12.2f   %s' % (
            name, count, nelem / 1e6, nbytes / 1048576.0, bits, note))
    return per_type


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    for p in argv[1:]:
        audit(p)
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
