#!/usr/bin/env python3
"""G2 (graph): .tflite vs torch SortformerHead (same ckpt weights) on dumped real windows,
zero-padded/noise-padded to the model's fixed F. Optionally compare to the ggml dump too.

  g2_tflite.py --tflite m.tflite --ckpt model_weights.ckpt --windows <dir> [--windows <dir>]
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np, torch
from ai_edge_litert.interpreter import Interpreter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_sortformer import SortformerHead, load_weights  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--tflite", required=True); ap.add_argument("--ckpt", type=Path, required=True)
ap.add_argument("--windows", action="append", required=True)
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()

it = Interpreter(model_path=a.tflite, num_threads=a.threads); it.allocate_tensors()
ins = it.get_input_details(); outs = it.get_output_details()
F = [d for d in ins if len(d["shape"]) == 3][0]["shape"][1]
print("inputs", [(d["name"], d["shape"], d["dtype"].__name__) for d in ins], "F=", F)
ref_model = SortformerHead(F).eval(); load_weights(ref_model, a.ckpt)
tm = {}
worst = 0.0; agree_n = agree_d = 0; gworst = 0.0; gag = 0
rng = np.random.default_rng(0)
for d in a.windows:
    for meta in sorted(glob.glob(f"{d}/*.meta.i64")):
        base = meta[:-len(".meta.i64")]
        T, H, valid = np.fromfile(meta, dtype=np.int64)
        emb = np.fromfile(base + ".emb.f32", dtype=np.float32).reshape(T, H)
        pad = rng.standard_normal((1, F, H)).astype(np.float32); pad[0, :T] = emb
        v = np.array([valid], dtype=np.int64)
        with torch.no_grad():
            ref = ref_model(torch.from_numpy(pad), torch.from_numpy(v)).numpy()[:, : T * 8]
        for det in ins:
            it.set_tensor(det["index"], pad if len(det["shape"]) == 3 else v.astype(det["dtype"]))
        it.invoke()
        out = it.get_tensor(outs[0]["index"])[:, : T * 8]
        dmax = float(np.abs(out - ref).max()); worst = max(worst, dmax)
        ag = int(((out > .5) == (ref > .5)).sum()); agree_n += ag; agree_d += out.size
        gg = np.fromfile(base + ".probs.f32", dtype=np.float32).reshape(1, T * 8, 8)
        gd = float(np.abs(out - gg).max()); gworst = max(gworst, gd)
        gagr = int(((out > .5) == (gg > .5)).sum()); gag += gagr
        print(f"{Path(base).parent.name}/{Path(base).name} T={T} vs-torch max|d|={dmax:.2e} agree={ag/out.size:.5f} | vs-ggml max|d|={gd:.2e} mean|d|={float(np.abs(out-gg).mean()):.1e} agree={gagr/out.size:.5f}")
print(f"WORST tflite-vs-torch max|d|={worst:.3e} agreement={agree_n/agree_d:.6f}")
print(f"vs ggml(dump): WORST max|d|={gworst:.3e} agreement={gag/agree_d:.6f}")
