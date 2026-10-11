"""dynamic_wi8_afp32 with OCTAV for the streaming X-ASR encoder (same recipe as the diarizer)."""
import argparse, os
from ai_edge_quantizer import quantizer, recipe
from ai_edge_quantizer.recipe import AlgorithmName

ap = argparse.ArgumentParser()
ap.add_argument("fp32")
ap.add_argument("out")
a = ap.parse_args()
r = recipe.dynamic_wi8_afp32()
for entry in r:
    entry["algorithm_key"] = AlgorithmName.OCTAV
q = quantizer.Quantizer(float_model=a.fp32)
q.load_quantization_recipe(r)
with open(a.out, "wb") as f:
    f.write(q.quantize().quantized_model)
print(f"q8-octav: {a.out} {os.path.getsize(a.out)/1e6:.0f} MB")
