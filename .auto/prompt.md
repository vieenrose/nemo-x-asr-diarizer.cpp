# Autoresearch: composite phone RTF (x-asr streaming + Nemotron-3 diarization, CPU only)

## Objective

Cut phone RTF of the **composite** streaming ASR+diarization stack (`src/engine.cpp`: x-asr streaming via
CrispASR's `xasr_stream_*`, Nemotron-3 diarization via `libaudiocpp.so`'s `audiocpp_stream_*`, one process,
one audio timeline, CPU inference only) while keeping:

- **accuracy**: transcripts byte-identical to the blessed reference (`.auto/checks.sh`),
- **true streaming**: per-piece cached incremental decoding. `xasr_stream_accept` on 100 ms pieces with
  `chunk_ms` caches, and `audiocpp_stream_push` into a `streaming` session. A rolling-window re-decode, a
  batch decode, or buffering the file and decoding once is NOT streaming and is NOT a valid result.
- **the behavioral contract**: silence / noise / tone produce zero turns and zero text; speaker tags stay
  consistent; first-output latency does not get worse.

Baseline anchor: **phone_rtf ≈ 0.46** on the baseline's own pinning (`taskset C0`, `-t 2`, armed), against
the 1.5B VibeASR streaming baseline at **1.2231** with 2198 MB peak RSS. Composite uses ~400 MB.

## Metrics

- **Primary**: `phone_rtf` (ratio, lower is better) — aggregate wall/audio over `chat69.wav` ×2 +
  `gate_ms_v2.wav` ×2 inside one armed window.
- **Secondary** (monitors, not targets): `asr_s`, `diar_s`, `other_s` (wall − asr − diar: this loop's own
  bookkeeping and waits), `first_partial_s`, `p95_piece_ms`, `peak_rss_mb`, `witness_mhz`,
  `deliv2400_pct`, `rtf_chat69`, plus `HASH` on the device transcripts.

`other_s` is the diagnostic that tells you where the time is: asr and diar are charged to their own calls
inside the run loop, so anything else — attribution, windowing, allocation, page touching — lands there.

## How to Run

```
./.auto/measure.sh     # builds arm64 if sources changed, pushes, arms, measures, witnesses, emits METRIC lines
./.auto/checks.sh      # host-side accuracy gate: byte-identity + WER ceilings + attribution/DER info
```

`measure.sh` exits non-zero if the device never reached an armed window (`witness_mhz < 1950`). That is a
`crash`, not a data point. Do not "average" an unarmed row: an unarmed device parks near 1.3 GHz and inflates
RTF 20-60 % while printing something that looks fine.

## Files in Scope

| file | note |
|---|---|
| `src/engine.cpp/.h` | the streaming loop: diar push → asr accept → attribution. Where RTF is actually spent. |
| `src/fusion.cpp/.h` | attribution of text deltas to diar turns; char timeline; `push_token` exact-times path |
| `src/main.cpp` | CLI, windowed `[k/N]` output contract |
| `src/wav.h` | wav load + in-process Lanczos3 resample to 16 kHz |
| `scripts/build_android.sh` | arm64 build recipe (validated; refuses to link if ggml symbols leak) |
| `.auto/*` | this harness |

## Off Limits

- `patches/crispasr-token-times.patch` semantics (the 40 ms token timeline is relied on by the validator).
- The two-ggml isolation: CrispASR ggml static in the binary, audio.cpp ggml hidden inside
  `libaudiocpp.so`. `build_android.sh` gates on 0 leaked `ggml_*/gguf_*` symbols; do not "unify" the runtimes
  (tried: links, then `GGML_ASSERT(*cur_backend_id != -1)`).
- The archive (`../VibeASR.cpp/.auto/device_state.tsv`) is another session's ledger. Never write to it.
- Diarization detection knobs (`speaker_threshold`, `speaker_pad_frames`) stay at the shipped defaults
  (`0.3` / `45`). They were tuned on the bilingual clips once; re-tuning them against `gate_ms_v2` would be
  optimizing on the accuracy gate.
- No new dependencies, no network at run time, no GPU path (`p.use_gpu = false` — see below).
- Accuracy may only move with the full paired evidence `.auto/validate.sh` produces (11 clips / ~1720 s, WER
  per clip + micro aggregate + discordance + sign test) and a deliberate re-bless. Reference table captured
  2026-09-25: gate_ms 0.2371, gate_ms_g100 0.2165, gate_ms_g1000 0.2062, gate_ms_v2 0.1765, control_ls 0.0422,
  holdout_en 0.2364, holdout_en_aligned 0.2500, holdout_zh 0.0791, holdout_zh2 0.0433, holdout_zh_aligned
  0.0662, holdout_zh_ph35200 0.0641.

## Constraints

1. **Phone-only measurement.** ARM gains do not transfer from x86; host numbers are for correctness only.
2. **Byte-identity or bust.** Any change to kernels, quantization, chunk geometry, resampling, or decode
   order must keep transcripts byte-identical, or come with a full paired accuracy validation (gate40 +
   holdout_en + holdout_zh, McNemar/discordant tokens) and a deliberate re-bless of `.auto/bless.txt`.
   Sub-1 % RTF moves are inside run-to-run noise unless proven by a paired A/B in one armed window.
3. **Watchdog sets are read-only.** `holdout_en` / `holdout_zh` exist to detect generalization loss. Do not
   tune against them, do not "fix" a single clip there.
4. **No benchmark gaming.** No caching outputs, no skipping audio, no shortening the protocol clips, no
   special-casing clip names, no suppressing output to look fast, no lowering accuracy thresholds to pass.
5. **Streaming integrity check**: `piecewise == one-shot` must stay true on any clip used, and first partial
   latency should improve or stay flat — a "faster" run that delays first text is trading the wrong thing.
6. Threads ≤ cores in the mask, always `-t 2` with `taskset C0`. `-t 4` under that mask read RTF 66 once.

## What's Been Tried (and the two traps this session inherits)

**Confirmed wins already in the tree**

- `p.use_gpu = false` — `xasr_context_default_params()` defaults it to **true**, and `xasr_init` then calls
  `crispasr_init_gpu_backend()`. On a GPU-less phone that is not a clean CPU fallback: ASR leg 33.7 s →
  9.8 s per 45 s of audio (RTF 0.74 → 0.218, matching the standalone probe) and the 2-CPU "stall" vanished.
  **Trap: the number was perfectly reproducible and still wrong.** Two published explanations died here
  ("two ggml spin pools livelock", "the composite needs four cpus"). Reproducible ≠ correct mechanism.
- In-process Lanczos3 resample so 24 kHz inputs work (`Wav::to_16k`), measured accuracy-neutral.
- Diar `speaker_pad_frames=45` + `threshold=0.3`: DER-lite 40.9 → 30.3 (tuned clip) and 45.8 → 29.3
  (held-out clip). Not to be re-tuned here.
- Model token timestamps (40 ms grid) with `--token-offset-ms 300` for the transducer's decision lag.
- Window output that never splits a word: identical text, WER 0.3136 → 0.2455, because the scorer tokenises
  per line. **Second trap: score a formatter, never diff it.**

**Known dead ends**

- Rolling-window re-decode (not streaming, disqualifies the result).
- One shared ggml between the two engines; `objcopy --localize-symbols` to hide ggml (silently wrong links).
- RapidSpeech's x-asr ggml path (loads, emits nothing; encoder dumps NaN).
- Forcing `--threads 4` under the 2-cpu mask.
- Reducing the ASR leg by touching `n_threads`: CrispASR's Android ggml has **no OpenMP** and creates no
  worker threads — the probe runs with exactly 1 thread. Any "add threads to x-asr" idea must first explain
  that observation.

**Where the time actually goes (measured, armed, screened — do not re-litigate)**

`phone_rtf 0.4651` is the arithmetic sum of two compute-bound model legs. On chat69, armed and screened:
ASR alone (`--no-diar`) 15.43 s = 0.224 s/audio-s, diar alone (`--no-asr`) 16.89 s = 0.245, composite
31.83 s. The composite is ~1.5% *faster* than the sum of its isolated legs, so scheduling has nothing to win.

- `other_s` is **not** bookkeeping. Timed with `NEMO_PROF=1`: `push_delta` 0.00 s and the final `attribute()`
  pass 0.00 s per run. It is the diarizer's last encoder window landing outside the per-piece timers.
- The diar encoder runs a **fixed-size 188-frame window** = `chunk_len 340` × 80 ms ≈ 27 s of audio, costing
  ~7-8.5 s per window on the two A78s. Short clips pay a whole window (3 s clip: 7.27 s at 1.3 GHz), which is
  what made it look like a constant overhead in `audiocpp_stream_finish`.
- Both legs are **weight-format insensitive**: q4_k (89 MB) ties the shipped q5_0 (160 MB) within 2%, all-q8_0
  ties it, and q6_k/iq4_nl are 7-9% *worse*. Compute-bound, not bandwidth-bound. Quantisation cannot help.
- `x-asr-zh-en-q8_0.gguf` is not q8_0: 298 tensors q5_0, 619 f32, 49 f16, arch `xasr` (zipformer-style
  `z.0..z.5`, `emb.conv*`, `dec.*`, `join.*`). Its metadata declares supported `chunk_ms` = [160, 480, 960,
  1920]; other values are rejected, and 960 changes the transcript.
- `Engine::init` is 0.20 s on the phone. There is no load cost to hide, so "prefault the models" is a shift
  between `load_s` and `wall_s`, not a win (measured: load +0.12 vs wall −0.13).

**Levers measured out (all byte-identical, so all fair tests)**

- Running the two legs concurrently (diar on a worker thread): 37-47% *slower*, and both legs slow ~2.8×
  (asr 24.9 → 70.5 s per 69 s). Retried at 1 ASR thread and with per-leg core pinning: still 14% worse.
- Thread counts (1 vs 2 both scale), piece cadence (100/200/400 ms identical to the byte), `graph_arena_mb`,
  `weight_context_mb`, dotprod (already on: `-mcpu=cortex-a78` implies `+dotprod`; ggml's `check_cxx_source_runs`
  probe can never pass when cross-compiling, so the CMake log misleads, not the binary), KleidiAI (kernels
  not vendored → needs network).
- Diar streaming geometry in both directions: `latency_profile` low/very_low/ultra_low are 1.5-5× worse
  (per-pass overhead), and 510/1020-frame chunks are a wash that *changes speaker tags* on chat69.
- `weight_type` f16/bf16/f32: slower and changes the transcript.
- Skipping `stream_finish` (saves 8.5 s but loses the turn still open at end-of-audio), and diar-only trailing
  silence (no effect; side fact: 2-4 s of trailing silence leaves the transcript byte-identical).

**Probe hygiene — this bit me twice**

Ad-hoc probes must arm and screen, not just run: use `.auto/arm.sh`, then `.auto/witness.sh <before> <after>`
and refuse the row if it says UNARMED. Two bugs lived here: probes called an `arm.sh` that did not exist
(arming was a function inside `measure.sh`, so the failure was invisible behind `>/dev/null`), and `arm.sh`
did not clear `arm.sh`'s own disarm sentinel, so the wake stream exited at once. Unarmed means ~1.3 GHz and
20-60% inflated numbers that look completely reasonable — that is how "ASR is 0.36 s/audio-s" turned out to
be a frequency artifact, not a code path.
