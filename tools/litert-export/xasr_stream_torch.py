"""Streaming X-ASR (GilgameshWind chunk-480ms) as a fixed-shape torch module.

Upstream-faithful: weights are xasr_native.pt (Luigi/x-asr-zh-en-icefall-trainable, extracted from
GilgameshWind/X-ASR-zh-en deployment/models/chunk-480ms-model). Forward config chunk_size=24,
left_context_frames=256 (480 ms). Wrapper follows icefall export-onnx-streaming.py OnnxEncoder.
Requires icefall on PYTHONPATH (egs/librispeech/ASR/zipformer) and k2 stubs (installed here).
"""
import logging
import sys
import types
import contextlib
from typing import List, Tuple

import torch
import torch.nn as nn

CHUNK = 24          # encoder frames per chunk (x2 pre-subsampling + 7 pad = 61 input frames)
LEFT = 256
PAD = 7 + 2 * 3
IN_FRAMES = CHUNK * 2 + PAD  # 61


def install_stubs():
    class _K2(types.ModuleType):
        # type annotations in icefall model.py reference k2 types; only the swoosh ops run here
        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)
            return type(name, (), {})
    k2 = _K2("k2")
    k2.swoosh_l = lambda x: torch.logaddexp(torch.zeros_like(x), x - 4.0) - 0.08 * x - 0.035
    k2.swoosh_r = lambda x: torch.logaddexp(torch.zeros_like(x), x - 1.0) - 0.08 * x - 0.313261687
    k2.swoosh_l_forward = k2.swoosh_l
    k2.swoosh_r_forward = k2.swoosh_r
    sys.modules.setdefault("k2", k2)
    ic = types.ModuleType("icefall"); icu = types.ModuleType("icefall.utils")

    @contextlib.contextmanager
    def torch_autocast(*a, **kw):
        with torch.amp.autocast("cpu", enabled=False):
            yield
    icu.torch_autocast = torch_autocast

    def make_pad_mask(lengths, max_len=None):
        max_len = int(lengths.max()) if max_len is None else max_len
        return torch.arange(max_len)[None] >= lengths[:, None]

    def _training_only(*a, **kw):
        raise RuntimeError("training-only icefall helper")
    icu.make_pad_mask = make_pad_mask
    icu.add_sos = icu.time_warp = _training_only
    ic.utils = icu
    sys.modules.setdefault("icefall", ic)
    sys.modules.setdefault("icefall.utils", icu)
    # model.py imports lhotse only for training-time SpecAugment; inference never uses it.
    lh = types.ModuleType("lhotse"); lhd = types.ModuleType("lhotse.dataset")
    lhd.SpecAugment = type("SpecAugment", (), {})
    lh.dataset = lhd
    sys.modules.setdefault("lhotse", lh)
    sys.modules.setdefault("lhotse.dataset", lhd)


def _patch_embed_streaming(cls):
    """Same math as icefall Conv2dSubsampling.streaming_forward minus the data-dependent
    `assert x.size(1) == x_lens.max().item()`, which torch.export cannot guard. Same recipe as
    tools/xasr-export/export_xasr.py (VoxSumDroid 66189d10)."""
    import warnings

    def streaming_forward(self, x, x_lens, cached_left_pad):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x, cached_left_pad = self.convnext.streaming_forward(x, cached_left_pad=cached_left_pad)
        b, c, t, f = x.size()
        x = x.transpose(1, 2).reshape(b, t, c * f)
        x = self.out(x)
        x = self.out_norm(x)
        x_lens = (x_lens - 7) // 2 - 3
        return x, x_lens, cached_left_pad

    cls.streaming_forward = streaming_forward


def build(icefall_dir: str, ckpt: str):
    sys.path.insert(0, f"{icefall_dir}/egs/librispeech/ASR/zipformer")
    install_stubs()
    from zipformer import Zipformer2
    from subsampling import Conv2dSubsampling
    from model import AsrModel
    from decoder import Decoder
    from joiner import Joiner
    _patch_embed_streaming(Conv2dSubsampling)
    # scaling._no_op does x.chunk(1, dim=-1)[0]; litert_torch 0.9.x fails to lower the single-chunk
    # split_with_sizes. It is an identity, so replace it (same patch as tools/xasr-export).
    import scaling, zipformer as _zf, subsampling as _ss
    for mod in (scaling, _zf, _ss):
        if hasattr(mod, "_no_op"):
            mod._no_op = lambda x: x

    ED = (192, 256, 512, 768, 512, 256)
    enc = Zipformer2(
        output_downsampling_factor=2, downsampling_factor=(1, 2, 4, 8, 4, 2),
        encoder_dim=ED, num_encoder_layers=(2, 2, 4, 5, 4, 2),
        encoder_unmasked_dim=(192, 192, 256, 320, 256, 192),
        query_head_dim=32, pos_head_dim=4, value_head_dim=12, num_heads=(4, 4, 4, 8, 4, 4),
        feedforward_dim=(512, 768, 1536, 2048, 1536, 768),
        cnn_module_kernel=(31, 31, 15, 15, 15, 31), pos_dim=48, causal=True,
        chunk_size=(CHUNK,), left_context_frames=(LEFT,),
    )
    m = AsrModel(
        encoder_embed=Conv2dSubsampling(80, ED[0]), encoder=enc,
        decoder=Decoder(vocab_size=5000, decoder_dim=512, blank_id=0, context_size=2),
        joiner=Joiner(encoder_dim=max(ED), decoder_dim=512, joiner_dim=512, vocab_size=5000),
        encoder_dim=max(ED), decoder_dim=512, vocab_size=5000, use_transducer=True, use_ctc=False,
    )
    sd = torch.load(ckpt, map_location="cpu")
    m.load_state_dict(sd["model"], strict=True)
    return m.eval()


class StreamEncoder(nn.Module):
    """(x [1,61,80], 116 states) -> (encoder_out [1,12,512], 116 new states).

    Same as icefall export-onnx-streaming.OnnxEncoder with the state list flattened.
    Input states order: encoder states (6 per layer), then cached_embed_left_pad, then processed_lens.
    """

    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.encoder_embed = model.encoder_embed
        self.encoder_proj = model.joiner.encoder_proj
        self.chunk_size = CHUNK
        self.left_context_len = LEFT

    def forward(self, x: torch.Tensor, *states: torch.Tensor):
        states = list(states)
        N = x.size(0)
        x_lens = torch.full((N,), IN_FRAMES, dtype=torch.int64)
        x, x_lens, new_left = self.encoder_embed.streaming_forward(
            x=x, x_lens=x_lens, cached_left_pad=states[-2])
        src_mask = torch.zeros(N, self.chunk_size, dtype=torch.int32)
        proc_mask = torch.arange(self.left_context_len).expand(N, self.left_context_len)
        processed_lens = states[-1]
        proc_mask = (processed_lens.unsqueeze(1) <= proc_mask).to(torch.int32).flip(1)
        new_processed = processed_lens + x_lens.to(processed_lens.dtype)
        src_mask = torch.cat([proc_mask, src_mask], dim=1)
        x = x.permute(1, 0, 2)
        enc_out, _, new_enc_states = self.encoder.streaming_forward(
            x=x, x_lens=x_lens, states=states[:-2], src_key_padding_mask=src_mask)
        enc_out = self.encoder_proj(enc_out.permute(1, 0, 2))
        return (enc_out, *new_enc_states, new_left, new_processed.to(states[-1].dtype))


def init_states(model, batch: int = 1):
    """Flat list of the 116 initial states, same order as icefall OnnxEncoder.get_init_states."""
    states = list(model.encoder.get_init_states(batch_size=batch, device="cpu"))
    states.append(model.encoder_embed.get_init_states(batch, device="cpu"))
    states.append(torch.zeros(batch, dtype=torch.int64))
    return states


def validate_vs_onnx(model, onnx_path: str, n_chunks: int = 10, seed: int = 0):
    """Stream n_chunks random chunks through torch and through the chunk-480ms ONNX (onnxruntime).

    Prints max|Δ| of encoder_out per chunk; states are carried along on both sides.
    """
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    assert in_names[0] == "x" and len(in_names) == 117, in_names[:3]
    wrap = StreamEncoder(model).eval()
    st_t = init_states(model)
    st_o = [np.zeros([d if isinstance(d, int) else 1 for d in i.shape], dtype=np.float32)
            if i.type == "tensor(float)" else np.zeros([1], dtype=np.int64)
            for i in sess.get_inputs()[1:]]
    rng = np.random.default_rng(seed)
    worst = 0.0
    with torch.no_grad():
        for c in range(n_chunks):
            x = rng.standard_normal((1, IN_FRAMES, 80)).astype(np.float32)
            out_t = wrap(torch.from_numpy(x), *st_t)
            out_o = sess.run(None, {"x": x, **dict(zip(in_names[1:], st_o))})
            d = float(np.abs(out_t[0].numpy() - out_o[0]).max())
            worst = max(worst, d)
            print(f"chunk {c}: enc_out max|d| = {d:.3e}", flush=True)
            st_t = list(out_t[1:])
            st_o = list(out_o[1:])
    print(f"worst max|d| over {n_chunks} chunks: {worst:.3e}")
    return worst


def export_fp32(model, out_path: str):
    """litert_torch fp32 conversion, single signature 'chunk': (x, 116 states) -> (enc_out, 116 states)."""
    import litert_torch
    wrap = StreamEncoder(model).eval()
    states = init_states(model)
    args = (torch.zeros(1, IN_FRAMES, 80), *states)
    edge = litert_torch.convert(wrap, args)
    edge.export(out_path)
    print(f"exported {out_path}", flush=True)


def validate_tflite(model, tflite_path: str, n_chunks: int = 10, seed: int = 1):
    """Stream random chunks through torch and the tflite (ai_edge_litert), states carried."""
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=tflite_path)
    it.allocate_tensors()
    inp = it.get_input_details()
    outd = it.get_output_details()
    wrap = StreamEncoder(model).eval()
    st_t = init_states(model)
    st_l = [s.numpy() for s in st_t]
    rng = np.random.default_rng(seed)
    worst = 0.0
    with torch.no_grad():
        for c in range(n_chunks):
            x = rng.standard_normal((1, IN_FRAMES, 80)).astype(np.float32)
            out_t = wrap(torch.from_numpy(x), *st_t)
            # map by shape/order: inputs[0]=x, rest states in same order as init_states
            order = sorted(range(1, len(inp)), key=lambda k: inp[k]["index"])
            it.set_tensor(inp[0]["index"], x)
            for k, v in zip(order, st_l):
                it.set_tensor(inp[k]["index"], v)
            it.invoke()
            outs = [it.get_tensor(d["index"]) for d in outd]
            d = float(np.abs(out_t[0].numpy() - outs[0]).max())
            worst = max(worst, d)
            print(f"chunk {c}: tflite enc_out max|d| vs torch = {d:.3e}", flush=True)
            st_t = list(out_t[1:])
            st_l = outs[1:]
    print(f"worst max|d| over {n_chunks} chunks: {worst:.3e}")
    return worst


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--icefall", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--onnx")
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--export", help="write fp32 tflite to this path")
    ap.add_argument("--validate-tflite", help="validate this tflite against torch")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    model = build(a.icefall, a.ckpt)
    if a.onnx:
        validate_vs_onnx(model, a.onnx, a.chunks)
    if a.export:
        export_fp32(model, a.export)
    if a.validate_tflite:
        validate_tflite(model, a.validate_tflite, a.chunks)
