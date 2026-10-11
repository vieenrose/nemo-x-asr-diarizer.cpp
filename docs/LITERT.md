# LiteRT backend: status

Branch `litert`. Runs the two neural graphs (X-ASR streaming encoder and Nemotron-3 Sortformer diarizer)
on LiteRT, dynamic int8, behind a dev switch. Everything else (fbank, decoder, joiner, scheduler, fusion)
is shared with the ggml build.

## Usage (host)

    LITERT_SO_DIR=<dir with libLiteRt.so> LITERT_INC_DIR=<litert headers> ALLOW_DEPS_DRIFT=1 bash scripts/build_host.sh
    ./build/nemo-x-asr-diarizer --audio clip.wav --backend litert_cpu \
        --xasr-tflite xasr_stream_q8_octav.tflite --diar-tflite sortformer_int8_f190.tflite \
        --xasr-model models/x-asr-zh-en-q8_0.gguf

`--backend ggml` (default) is unchanged.

## Results (host CPU, 600 s excerpts of AMI EN2002a/b/c)

DER (md-eval-22, collar 0.25, full 600 s; each excerpt starts 120 s into its meeting):

| Excerpt | ggml q8 | LiteRT int8 | LiteRT fp32 diarizer |
|---|---|---|---|
| EN2002a | 17.33 % | 17.72 % | 16.82 % |
| EN2002b | 22.41 % | 22.43 % | not run |
| EN2002c | 11.13 % | 10.90 % | not run |

Speed and memory (EN2002a, ASR + diarizer):

| Config | Peak RSS | RTF |
|---|---|---|
| ggml q8 | 517 MB | 0.25 |
| LiteRT int8 | 728 MB | 0.11 |
| LiteRT int8 ASR + fp32 diarizer | 1289 MB | 0.15 |

- ASR-only, 60 s clip: transcripts identical word for word.
- ASR-only, 600 s: 98.1 % word similarity (int8 encoder: small deletions and substitutions).
- Diarizer, per window (16 windows, G2): int8 vs ggml q8 agree 99.95 % of decisions.
- DER: LiteRT int8 minus ggml is +0.39, +0.02, -0.23 point on the three excerpts (mean +0.06): equivalent.
  The fp32 diarizer is 0.5 point better on EN2002a only.
- Earlier numbers (18.66 % / 19.08 % on EN2002a) were scored over the last 480 s of the excerpt only, because
  of a window shift in the scoring script; they are superseded by the table above.

## Android (Note10+, CPU)

Build: `scripts/build_android.sh` with `LITERT_LIB_DIR` (Android `libLiteRt.so`) and `LITERT_INC_DIR` set
(compiles with `-DNEMO_WITH_LITERT -DNEMO_LITERT_ENV_API`; the chunk hook patch must be applied to CrispASR).
Galaxy Note10+, 4 cores (`taskset f0`), EN2002a excerpt, 600 s, DER md-eval collar 0.25:

| | ggml q8 | LiteRT CPU int8 |
|---|---|---|
| DER | 17.64 % | 17.20 % |
| RTF | 0.314 | 0.265 |
| Peak RSS | 498 MB | 753 MB |

On 60 s, CPU int8 RTF is 0.175 vs 0.246 for ggml (RSS 651 vs 397 MB) and the transcripts are identical.

GPU was measured and dropped (target is CPU only). Findings, for the record:

- The int8 X-ASR graph delegates only 177 of 3517 ops to LiteRT CL; partial delegation fails to compile. The ASR graph is CPU only in any case.
- The diarizer needs `export_sortformer.py --float-valid` (no INT64 ops) to delegate fully (2186/2186 ops).
- Diarizer on GPU, int8: DER 18.10 %, RTF 0.287, RSS 831 MB. fp32 (395 MB): DER 16.82 %, RTF 0.290, RSS 1400 MB.
  Neither beats CPU int8 on speed or memory.

## Not done

- iOS (iPhone 14 Pro Max) integration and benchmarks, CPU only.
- Speed and memory on EN2002b/c and a non-English clip (AISHELL-4).
