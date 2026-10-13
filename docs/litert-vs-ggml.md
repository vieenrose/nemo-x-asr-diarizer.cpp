# LiteRT (CPU) vs ggml: parity report

Host x86_64, 4 threads. Reference = ggml Q8_0 GGUF. LiteRT = X-ASR dynamic int8 (OCTAV), Sortformer dynamic int8 (F_max 190). Status: CPU only, host only; Android/iOS device parity not yet measured for this report.

## Results

| Check | Clip(s) | Result | Gate |
|---|---|---|---|
| ASR text, int8 vs ggml | golden_en_5min (ASR only) | 99.24 % word similarity | info |
| ASR text, fp32 vs ggml | golden_en_5min (ASR only) | 99.12 % | info |
| ASR text, int8 vs ggml (zh) | golden_zh_5min (ASR only, char level) | 99.55 % (780 vs 779 chars) | info |
| Diarizer DER, int8 vs ggml | 12 AMI clips (EN2002a-d, ES2004a-d, IS1009a-d) | mean 19.8 % vs 19.9 % (Δ −0.1 pt) | |ΔDER| < 1 pt: passes on mean, 10/12 clips within 1 pt |
| Speaker count | same 12 clips | |Δk| ≤ 1 on all 12; exact on 11/12 (int8) vs 10/12 (ggml) | |
| Sortformer graph, tflite fp32 vs torch | 16 real windows | max |Δ| 5.7e-6 | ≤ 1e-5 |
| Sortformer, tflite int8 vs ggml q8, per window | 16 windows | 99.95 % of frame×speaker decisions agree | info |
| X-ASR graph, fp32 tflite vs torch | 10 chunks | max |Δ| 4.1e-6 | ≤ 1e-3 |

## Decisions

- **Diarizer: int8** (101 MB). fp32 (395 MB) would move frame agreement from 98.1 % to 99.8 %, but it gives no DER gain on the 12-clip set (not measured on fp32 there; decision taken on the window-level result).
- **X-ASR: int8.** fp32 does not reduce the ASR gap to ggml (99.12 % vs 99.24 %), so the residual ~0.8 % is not caused by the int8 weights. Likely source: ggml's Q8_0 activation quantization in mul_mat (not verified; would need a GGUF F16 comparison).

## Method notes

- DER: `score.py` (0.25 s collar, reference clipped to the excerpt window 120–720 s, speakers with ≥ 5 s). The excerpts are cut from the meeting at 120 s (`stage.sh`); an earlier 0 s alignment gave meaningless DER > 100 %.
- DER is absolute-scale check only: `score_turns.py` (project scorer) not found in this checkout, so absolute values are not comparable to published numbers.

## Open

- Device parity (G4): Note10+ and iPhone, CPU vs GPU, not yet run for this report.
- WER gate: the official `validate.sh` set (eval-bilingual, score_stream.py) is not on this machine. Validated by the user on the substitute measures above (ASR text similarity, en and zh).
- Device RTF / memory, litert_host int8 on Note10+: RTF 0.146–0.165 on a 12.7 s clip (one run, `NemoStreamEngineTest`), RSS 548 MB. Earlier golden-clip runs (5 min, 4 clips): RTF 0.143–0.224. ggml RTF on device not measurable (the NO_GGML APK has no ggml), so no device ggml comparison.
- iPhone 14 Pro Max, litert_host int8, sample 123 s: RTF 0.123, 2 speakers / 66 turns (host: 72 turns), footprint 371 MB (peak 442 MB). `litert_cpu` on the ggml-free iOS lib returned 0 turns because the engine skipped the diarizer silently (the ggml branch is compiled out); it now fails with an explicit error.
