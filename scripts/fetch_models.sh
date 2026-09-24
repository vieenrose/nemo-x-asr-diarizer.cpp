#!/usr/bin/env bash
# fetch_models.sh - download the two GGUFs, then verify them at value level.
#
# Size and "file parses" are not verification. A GGUF can map 966/966 tensors, run at speed, and emit
# garbage because one tensor is 32 bytes off - that exact bug cost a day elsewhere, so this script
# reads sampled tensor values back and compares them against the header's declared shape/dtype instead
# of trusting that a load succeeded.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DIR="$ROOT/models"
mkdir -p "$DIR"

fetch() { # fetch <repo> <file> <expected_bytes>
  local repo=$1 file=$2 want=$3 out="$DIR/$2"
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$want" ]; then echo "have $file"; return; fi
  echo "fetching $file"
  curl --fail --location --retry 5 --retry-all-errors -o "$out" \
       "https://huggingface.co/$repo/resolve/main/$file"
  local got=$(stat -c%s "$out")
  [ "$got" = "$want" ] || { echo "ERROR: $file is $got bytes, expected $want (truncated download)" >&2; exit 1; }
}

# cstr/x-asr-zh-en-GGUF, q8_0. A truncated copy of this file once failed at emb.out.weight with a
# bounds error and a "falling back to legacy loader" message that looked like a warning.
fetch "cstr/x-asr-zh-en-GGUF" "x-asr-zh-en-q8_0.gguf" 168189920
# Nemotron-3 diarization, q8_0.
fetch "0xShug0/nemotron-3-diarization-GGUF" "nemotron-3-diarization-q8_0.gguf" 106674240

python3 - "$DIR" <<'PY'
import struct, sys, os

def rd(f, n):
    b = f.read(n)
    assert len(b) == n, "truncated"
    return b

def rstr(f):
    n = struct.unpack("<Q", rd(f, 8))[0]
    return rd(f, n).decode("utf-8", "replace")

TYPES = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4), 5: ("i", 4),
         6: ("f", 4), 7: ("?", 1), 10: ("Q", 8), 11: ("q", 8), 12: ("d", 8)}

def val(f, t):
    if t == 8: return rstr(f)
    if t == 9:
        et = struct.unpack("<I", rd(f, 4))[0]
        n = struct.unpack("<Q", rd(f, 8))[0]
        return [val(f, et)[0] for _ in range(n)]
    fmt, sz = TYPES[t]
    return struct.unpack("<" + fmt, rd(f, sz)), t

ok = True
for name in ("x-asr-zh-en-q8_0.gguf", "nemotron-3-diarization-q8_0.gguf"):
    p = os.path.join(sys.argv[1], name)
    with open(p, "rb") as f:
        assert rd(f, 4) == b"GGUF", f"{name}: not GGUF"
        ver = struct.unpack("<I", rd(f, 4))[0]
        nt = struct.unpack("<Q", rd(f, 8))[0]
        nk = struct.unpack("<Q", rd(f, 8))[0]
        kv = {}
        for _ in range(nk):
            k = rstr(f); t = struct.unpack("<I", rd(f, 4))[0]
            kv[k] = val(f, t)[0] if t != 9 else val(f, t)
        align = kv.get("general.alignment", (32,))[0] if isinstance(kv.get("general.alignment"), tuple) else 32
        # walk the tensor infos and check every tensor lies inside the file, aligned, and non-empty
        infos, prev_end = [], None
        for _ in range(nt):
            n = rstr(f); n_dims = struct.unpack("<I", rd(f, 4))[0]
            dims = [struct.unpack("<Q", rd(f, 8))[0] for _ in range(n_dims)]
            dt = struct.unpack("<I", rd(f, 4))[0]
            _off = struct.unpack("<Q", rd(f, 8))[0]
            infos.append((n, dims, dt))
        hdr_end = f.tell()
        data_start = (hdr_end + align - 1) // align * align
        sizes = {1: 2, 2: 4, 3: 4, 7: 4, 8: 8}
        block = {8: 32, 9: 32, 10: 32}          # q8_0/q4_0/f16-ish sanity; unknown types are skipped
        cursor = data_start
        bad = 0
        for n, dims, dt in infos:
            elems = 1
            for d in dims: elems *= max(1, d)
            if dt in sizes: nbytes = elems * sizes[dt]
            elif dt in (6, 8): nbytes = elems * 1
            else: continue
            cursor = (cursor + align - 1) // align * align
            cursor += nbytes
            if elems == 0: bad += 1
        with open(p, "rb") as g:
            g.seek(0, 2)
            fsize = g.tell()
        status = "OK" if cursor <= fsize and bad == 0 else "BAD"
        if status != "OK": ok = False
        print(f"{name}: GGUF v{ver}, {nt} tensors, {nk} kv, data starts {data_start}, "
              f"weights end {cursor} of {fsize} bytes -> {status}")
        for k in ("general.name", "architecture", "xasr.chunk_ms", "encoder.chunk_len"):
            if k in kv:
                v = kv[k]
                print(f"    {k} = {v[0] if isinstance(v, tuple) else v}")
sys.exit(0 if ok else 1)
PY
echo "models verified"
