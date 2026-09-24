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

**Open observations worth attacking**

- `other_s` is a large slice of wall time (once ~8 s per 35 s). Attribution and window assembly re-scan the
  whole timeline; `attribute()` runs on every diar turn update, which is O(chars × turns) per call.
- audio.cpp creates ~8 threads lazily during streaming regardless of `bc.threads`.
- The diarizer commits turns in batches (first batch ~30 s of audio in these clips), so attribution leans on
  proximity fill early on.
- Diar leg is ~6-10 s per 45 s of audio while the diar model is 106 MB and q8; its encoder may be the cheaper
  half to attack with cache/geometry work.
