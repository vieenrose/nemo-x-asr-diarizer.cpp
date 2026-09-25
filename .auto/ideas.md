# Ideas backlog — composite phone RTF

Everything in the "measured out" list below was tested, not guessed. `prompt.md` carries the numbers; this
file is the queue. Prune a line when it is tried or when its assumption dies.

## Floor, established 2026-09-25

`phone_rtf = 0.4651` = ASR 0.224 + diar 0.245 (s/audio-s), armed and screened, verified against the isolated
legs (`--no-diar` 15.43 s, `--no-asr` 16.89 s, composite 31.83 s on chat69). The composite already beats the
sum of its parts by ~1.5%. Both legs are compute-bound on two A78s and insensitive to weight format.

## Open — needs something outside this repo

- **KleidiAI matmul backend for CrispASR's ggml** (`GGML_CPU_KLEIDIAI=ON`). The glue is vendored
  (`ggml/src/ggml-cpu/kleidiai/`) but the `arm_llama` kernels are not, so it needs a network fetch. If a
  machine with the kernels appears, this is the only untried kernel-level lever. Accumulation order will
  probably change, so expect to need `.auto/validate.sh` and a re-bless decision.
- **A cheaper model pair.** Out of scope for the contract (byte-identity names these two models), but the
  honest way to beat 0.465 is fewer FLOPs per audio-second, not better scheduling. Note the diar leg is the
  *bigger* one (0.245 vs 0.224), and its cost is ~7-8.5 s per 27 s window because the window is a fixed 188
  encoder frames — a model trained for smaller windows would be the win.
- **`spkcache_len` on the diarizer** (session option, 264 frames of speaker memory). It is plausibly the
  dominant term in that fixed per-window cost. Untouched because the speaker cache is what determines
  *which* speaker a frame gets: expect a DER move, so it belongs with a full re-validation, and it edges
  close to the "do not tune diarization against the accuracy gate" rule.

## Measured out — do not retry without a changed assumption

- Leg concurrency / diar on a worker thread (#1082, #1083): 37-47% slower; also provably pointless given the
  sum-of-parts floor.
- Attribution / window-assembly overhead: timed at 0.00 s. `other_s` is the last encoder window, not bookkeeping.
- Piece cadence (100/200/400 ms), thread counts (1 vs 2), `graph_arena_mb`, `weight_context_mb`, prefault,
  dotprod/build flags.
- Requantising x-asr (q4_k / q4_0 / iq4_nl / q6_k / all-q8_0): no win, some worse. Compute-bound.
- Diar `latency_profile` and chunk geometry in both directions; `weight_type`.
- Skipping `stream_finish`; diar-only trailing silence.
- ASR `chunk_ms` 960: ~12% faster on the leg but changes the transcript — would need the full accuracy path,
  and the model's supported set is [160, 480, 960, 1920] so there is nothing in between to tune.
