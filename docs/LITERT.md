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

## Results (EN2002a excerpt, 600 s, host CPU)

| Config | Peak RSS | RTF | DER (collar 0.25, [120,720) s) |
|---|---|---|---|
| ggml q8 (ASR + diar) | 517 MB | 0.25 | 18.66 % |
| LiteRT int8 (ASR + diar) | 728 MB | 0.11 | 19.08 % |

- ASR-only, 60 s clip: transcripts identical word for word.
- ASR-only, 600 s: 98.1 % word similarity (int8 encoder: small deletions and substitutions).
- Diarizer, per window (16 windows, G2): int8 vs ggml q8 agree 99.95 % of decisions.
- DER gap: most likely the int8 diarizer weights; not isolated yet (fp32 LiteRT run pending).

## Not done

- GPU backend (ML Drift) on host and devices.
- Android (Note10+) and iOS (iPhone 14 Pro Max) integration and benchmarks.
- Second evaluation clip (AMI / AISHELL-4) to confirm DER parity.
- Full report with latency and RSS per device.
