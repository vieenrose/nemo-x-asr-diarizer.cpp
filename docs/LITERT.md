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

## Not done

- GPU backend (ML Drift) on host and devices.
- Android (Note10+) and iOS (iPhone 14 Pro Max) integration and benchmarks.
- Speed and memory on EN2002b/c and a non-English clip (AISHELL-4).
- Full report with latency and RSS per device.
