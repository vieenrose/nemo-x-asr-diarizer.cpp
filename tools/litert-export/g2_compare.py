#!/usr/bin/env python3
"""G2: torch SortformerHead vs the ggml DiarCrispASR probabilities dumped by
NEMO_DUMP_DIAR_WINDOWS, on the same GGUF weights (dequantized) so only op sequence / runtime differs.

  g2_compare.py --gguf diar-q8_0.gguf --windows <dir> [--windows <dir> ...]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch
import gguf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_sortformer import D_MODEL, SortformerHead, load_state  # noqa: E402


def gguf_state(path):
    r = gguf.GGUFReader(path)
    return {t.name: torch.from_numpy(np.ascontiguousarray(gguf.quants.dequantize(t.data, t.tensor_type)).astype(np.float32))
            for t in r.tensors}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--windows", action="append", required=True)
    ap.add_argument("--no-f16-conv", action="store_true")
    ap.add_argument("--pad-to", type=int, default=0, help="also test padding to this F (influence of padded frames)")
    a = ap.parse_args()
    sd = gguf_state(a.gguf)
    models = {}

    def model(T):
        if T not in models:
            m = SortformerHead(T, f16_conv_input=not a.no_f16_conv).eval()
            load_state(m, sd)
            models[T] = m
        return models[T]

    worst, agree_n, agree_d = 0.0, 0, 0
    for d in a.windows:
        for meta in sorted(glob.glob(f"{d}/*.meta.i64")):
            base = meta[: -len(".meta.i64")]
            T, H, valid = np.fromfile(meta, dtype=np.int64)
            emb = torch.from_numpy(np.fromfile(base + ".emb.f32", dtype=np.float32).reshape(1, T, H))
            ref = np.fromfile(base + ".probs.f32", dtype=np.float32).reshape(1, T * 8, 8)
            with torch.no_grad():
                out = model(int(T))(emb, torch.tensor([valid])).numpy()
            dmax = float(np.abs(out - ref).max())
            agree = int(((out > 0.5) == (ref > 0.5)).sum())
            worst = max(worst, dmax); agree_n += agree; agree_d += out.size
            line = f"{Path(base).parent.name}/{Path(base).name} T={T} valid={valid} max|d|={dmax:.2e} mean|d|={float(np.abs(out-ref).mean()):.2e} agree={agree/out.size:.5f}"
            if a.pad_to > T:
                pad = torch.randn(1, a.pad_to, H)
                pad[:, :T] = emb
                with torch.no_grad():
                    po = model(a.pad_to)(pad, torch.tensor([valid])).numpy()[:, : T * 8]
                line += f" pad_leak={float(np.abs(po - out).max()):.2e}"
            print(line)
    print(f"WORST max|d|={worst:.3e}  decision agreement={agree_n/agree_d:.6f}")


if __name__ == "__main__":
    main()
