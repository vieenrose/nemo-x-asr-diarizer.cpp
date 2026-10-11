#!/usr/bin/env python3
"""Nemotron-3 Sortformer encoder + head as a fixed-shape torch module, for LiteRT export.

Mirrors the op sequence of src/diar_crispasr.cpp (encode() / build_layer / head), which is the
reference the GGUF/ggml path is checked against. Input is the post pre-encode embedding window
[1, F, 512] plus valid_frames; output is speaker probabilities [1, 8*F, 8] (head upsamples x8).
The pre-encode subsampling stays on the host, as in the C++ engine.

Usage:
  export_sortformer.py --ckpt <model_weights.ckpt> --frames 230 --out sortformer_f32.tflite
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 512
N_HEADS = 8
HEAD_DIM = D_MODEL // N_HEADS
N_LAYERS = 31
EPS = 1e-5
ROPE_THETA = 10000.0
UPSAMPLE = 8
HEAD_HIDDEN = 192
NUM_SPKS = 8
MASK_NEG = -10000.0  # same additive value as attention_mask() in diar_crispasr.cpp


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(D_MODEL, eps=EPS)
        self.w_qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.out_proj = nn.Linear(D_MODEL, D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL, eps=EPS)
        self.ffn_in = nn.Linear(D_MODEL, 2048)
        self.ffn_out = nn.Linear(2048, D_MODEL)

    def forward(self, x, cos, sin, mask):
        # x: [T, D]; cos/sin: [T, HALF]; mask: [T, T] additive (query x key)
        T = x.shape[0]
        n1 = self.norm1(x)
        qkv = self.w_qkv(n1)
        q, k, v = qkv.split(D_MODEL, dim=-1)
        q = q.view(T, N_HEADS, HEAD_DIM).transpose(0, 1)  # [H, T, HD]
        k = k.view(T, N_HEADS, HEAD_DIM).transpose(0, 1)
        v = v.view(T, N_HEADS, HEAD_DIM).transpose(0, 1)
        q = rope(q, cos, sin)
        k = rope(k, cos, sin)
        scores = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(HEAD_DIM)) + mask
        attn = torch.softmax(scores, dim=-1) @ v  # [H, T, HD]
        attn = attn.transpose(0, 1).reshape(T, D_MODEL)
        x1 = x + self.out_proj(attn)
        ff = F.gelu(self.ffn_in(self.norm2(x1)), approximate="none")
        return x1 + self.ffn_out(ff)


def rope(t, cos, sin):
    # t: [H, T, HD]; split each head into halves, rotate (x1*c - x2*s, x2*c + x1*s)
    half = HEAD_DIM // 2
    x1, x2 = t[..., :half], t[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class SortformerHead(nn.Module):
    def __init__(self, frames: int, f16_conv_input: bool = False):
        super().__init__()
        self.frames = frames
        # ggml's conv1d rounds its input through F16 (diar_crispasr.cpp:476); only for parity checks.
        self.f16_conv_input = f16_conv_input
        self.embed_norm = nn.LayerNorm(D_MODEL, eps=EPS)
        self.layers = nn.ModuleList([Layer() for _ in range(N_LAYERS)])
        self.final_norm = nn.LayerNorm(D_MODEL, eps=EPS)
        self.encoder_proj = nn.Linear(D_MODEL, HEAD_HIDDEN)
        self.upsample = nn.Conv1d(HEAD_HIDDEN, HEAD_HIDDEN * UPSAMPLE, kernel_size=3, padding=1)
        self.hidden = nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN)
        self.spks = nn.Linear(HEAD_HIDDEN, NUM_SPKS)
        # RoPE tables depend only on T: precomputed constants, so the graph has no trig ops.
        t = torch.arange(frames, dtype=torch.float64)[:, None]
        d = torch.arange(HEAD_DIM // 2, dtype=torch.float64)[None, :]
        freq = 1.0 / (ROPE_THETA ** (2.0 * d / HEAD_DIM))
        ang = (t * freq).float()
        self.register_buffer("cos", torch.cos(ang), persistent=False)
        self.register_buffer("sin", torch.sin(ang), persistent=False)

    def forward(self, emb, valid):
        # emb: [1, F, 512] fp32; valid: [1] int64 (valid frame count of the window)
        x = self.embed_norm(emb[0])
        T = self.frames
        keys = torch.arange(T, device=emb.device, dtype=valid.dtype if valid.is_floating_point() else torch.int64)
        mask = torch.where(keys[None, :] >= valid[0], MASK_NEG, 0.0).expand(T, T)
        for layer in self.layers:
            x = layer(x, self.cos, self.sin, mask)
        fn = self.final_norm(x)
        proj = self.encoder_proj(fn)              # [T, 192]
        # Rows past valid_frames must read as the conv's zero padding, as in an unpadded window.
        proj = proj * (keys < valid[0]).to(proj.dtype)[:, None]
        if self.f16_conv_input:
            proj = proj.half().float()
        conv = self.upsample(proj.t()[None])      # [1, 1536, T]
        # ggml lays the 1536 channels out as j*192 + d, frame s = 8t + j (reshape of [1536, T]).
        up = conv[0].view(UPSAMPLE, HEAD_HIDDEN, T).permute(2, 0, 1).reshape(T * UPSAMPLE, HEAD_HIDDEN)
        r1 = F.relu(up)
        r2 = F.relu(self.hidden(r1))
        return torch.sigmoid(self.spks(r2))[None]  # [1, 8T, 8]


def load_weights(model: SortformerHead, ckpt: Path) -> None:
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    load_state(model, sd.get("state_dict", sd))


def load_state(model: SortformerHead, sd: dict) -> None:
    sd = {k: torch.as_tensor(v).float() for k, v in sd.items()}
    m = {
        "embed_norm": "encoder.embed_norm",
        "final_norm": "encoder.final_norm",
        "encoder_proj": "sortformer_modules.encoder_proj",
        "upsample": "sortformer_modules.subpixel_upsample",
        "hidden": "sortformer_modules.first_hidden_to_hidden",
        "spks": "sortformer_modules.single_hidden_to_spks",
    }
    new = {}
    for ours, theirs in m.items():
        mod = getattr(model, ours)
        new[f"{ours}.weight"] = sd[f"{theirs}.weight"]
        new[f"{ours}.bias"] = sd[f"{theirs}.bias"]
    for i, layer in enumerate(model.layers):
        p = f"encoder.layers.{i}."
        new.update({
            f"layers.{i}.norm1.weight": sd[p + "norm1.weight"], f"layers.{i}.norm1.bias": sd[p + "norm1.bias"],
            f"layers.{i}.w_qkv.weight": sd[p + "attn.w_qkv.weight"],
            f"layers.{i}.out_proj.weight": sd[p + "attn.out_proj.weight"], f"layers.{i}.out_proj.bias": sd[p + "attn.out_proj.bias"],
            f"layers.{i}.norm2.weight": sd[p + "norm2.weight"], f"layers.{i}.norm2.bias": sd[p + "norm2.bias"],
            f"layers.{i}.ffn_in.weight": sd[p + "ffn.net.0.weight"], f"layers.{i}.ffn_in.bias": sd[p + "ffn.net.0.bias"],
            f"layers.{i}.ffn_out.weight": sd[p + "ffn.net.3.weight"], f"layers.{i}.ffn_out.bias": sd[p + "ffn.net.3.bias"],
        })
    missing, unexpected = model.load_state_dict(new, strict=False)
    missing = [k for k in missing if not k.startswith(("cos", "sin"))]
    if missing or unexpected:
        raise SystemExit(f"weight mismatch: missing={missing} unexpected={unexpected}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=190, help="fixed window F (largest window the scheduler emits)")
    ap.add_argument("--float-valid", action="store_true", help="valid_frames as float32 (no INT64 ops: GPU delegate)")
    ap.add_argument("--out", type=Path, help="write the fp32 .tflite here (needs litert_torch)")
    args = ap.parse_args()

    model = SortformerHead(args.frames).eval()
    load_weights(model, args.ckpt)
    emb = torch.randn(1, args.frames, D_MODEL)
    valid = torch.tensor([args.frames], dtype=torch.float32 if args.float_valid else torch.int64)
    with torch.no_grad():
        probs = model(emb, valid)
    print("probs shape:", tuple(probs.shape), "range:", float(probs.min()), float(probs.max()))

    if args.out is not None:
        import litert_torch
        edge = litert_torch.convert(model.eval(), (emb, valid))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        edge.export(str(args.out))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
