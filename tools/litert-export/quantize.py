#!/usr/bin/env python3
"""Dynamic int8 weights / fp32 activations (the q8_0 equivalent) for a litert_torch fp32 .tflite.
OCTAV is the weight-range algorithm validated on X-ASR in the earlier LiteRT work.

  quantize.py in_f32.tflite out_int8.tflite [--algo octav|minmax]
"""
import argparse
from ai_edge_quantizer import quantizer, recipe, algorithm_manager

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--algo", choices=["octav", "minmax"], default="octav")
a = ap.parse_args()
algo = {"octav": algorithm_manager.AlgorithmName.OCTAV, "minmax": algorithm_manager.AlgorithmName.MIN_MAX_UNIFORM_QUANT}[a.algo]
qt = quantizer.Quantizer(a.src)
qt.load_quantization_recipe(recipe.dynamic_wi8_afp32(algo))
res = qt.quantize()
res.export_model(a.dst, overwrite=True)
print("wrote", a.dst)
