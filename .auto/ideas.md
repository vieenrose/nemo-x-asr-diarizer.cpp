# Ideas backlog — composite phone RTF

Everything under "measured out" was measured, not guessed. `prompt.md` and `docs/pipeline-design.md` carry
the numbers; this file is the queue.

## Where the composite stands (2026-09-25, end of session)

`phone_rtf` 0.4651 -> **0.307-0.314**, byte-identical throughout, from four changes:

| change | where | effect | numerics |
|---|---|---|---|
| diar `spkcache_len` 264 -> 128 | engine default | -9% | re-blessed with 11-clip evidence |
| stop padding diar encoder windows to capacity | audiocpp | -1.2% | identical |
| mel filterbank: skip exact zeros, drop `long double` | audiocpp | -16.6% | identical |
| build the x-asr chunk graph once (fixes a UAF) | crispasr | -1.8% | identical |

Plus one merged GGUF (`--models-bundle`), measured neutral by design.

## Open, ranked

- **f16 / q8_0 joint weights.** The transducer joint is a [5000x512] matvec per encoder frame in host C++:
  3.87 s per 69 s clip, 31% of the ASR leg, invisible to the ggml profiler. Streamed at ~4.5 GB/s with one
  use per element, so bytes are the lever: f16 halves it, q8_0 quarters it. NEON accumulators did NOT help
  (3.87 -> 3.84 s) and neither did sharing weight passes across frames (6.43 s). Changes logits -> validate.sh
  + deliberate re-bless. **Largest identified item.**
- **One-runtime merge** (port the diar encoder into crispasr's ggml so one scheduler, one pool, one arena
  serve both). Ceiling is the 1.6-of-2-cores utilisation. The two-pool form is measured dead (37-47% slower).
  The container merge is already done and committed, which is the prerequisite.
- **KleidiAI** (`GGML_CPU_KLEIDIAI=ON`): the only untried kernel-level lever; needs a network fetch for the
  `arm_llama` kernels. Expect accumulation-order changes -> validation.
- **ggml-native streaming caches** on the ASR side: 114 cache tensors per step currently go
  graph -> host vector -> graph (outputs 0.563 s + inputs 0.278 s per 69 s clip). Bit-identical in principle;
  every other host-side phase has been partly hidden under the matmuls, so measure before believing it.
- **Re-profile the diar encoder** now that its frontend is 18x faster. It is ~6.6 s per 69 s clip and was
  never phase-split after the filterbank fix - the filterbank find says check before assuming.

## Measured out — do not retry without a changed assumption

- Leg concurrency / diar on a worker thread: 37-47% slower; also pointless given the sum-of-parts floor.
- Requantising x-asr (q4_k / q4_0 / iq4_nl / q6_k / all-q8_0): no win, some worse. Every quantised type
  declares `vec_dot_type = Q8_0`, so activations are quantised and dotted with int8 SDOT regardless of the
  weight format - which is *why* the weight format cannot matter.
- F16 weights for the ASR: crispasr deliberately patches F16 to an f32 dot (upstream's f32->f16 cast
  saturates above 65504 and produced NaN matmuls). The A78's fp16 pipe is unreachable without solving that.
- ASR `chunk_ms` 960: ~12% faster on the leg, but WER +5.9 pts on gate_ms_v2 and past the holdout_en ceiling.
- Elementwise micro-optimisation of the ASR leg: copies, thread-partition threshold, per-element integer
  division. Up to -28% profiled CPU, 0.0% wall - it is overlapped under the matmuls (cores_used 1.6-1.7/2).
- Diar streaming geometry: `latency_profile` variants 1.5-5x worse; chunk_len 510 faster on the 69 s clip but
  slower on the 45 s one and pushes the first turn 30.5 s -> 44.1 s; chunk_len 680 is worse and just moves
  work into the drain. The drain is the price of bounded turn latency, not waste.
- Diar `spkcache_update_period`: compute-neutral by construction (capacity is fixed by `spkcache_len`).
- `spkcache_len` below 128: output is byte-identical down to 64, but the cost differences sit inside
  run-to-run variance, and 64 is slower (more compression passes). 128 stays.
- Attribution / window assembly: timed at 0.00 s. `other_s` is the final encoder window, instrumented.
- Prefault, graph arena / weight context sizing, piece cadence, thread counts, dotprod/build flags.
- KleidiAI without kernels; one shared ggml between the two engines (dies at
  `GGML_ASSERT(*cur_backend_id != -1)`).

## Method notes worth keeping

- The protocol's row-to-row spread is +/-10% on a hot phone and ~0.5% after a 180 s cooldown. Cool before
  measuring, or interleave candidate and baseline in one armed window. Three "wins" in this session's early
  log were inside that band.
- Arm + witness every device number. Unarmed rows are 20-60% slow and look perfectly reasonable.
- A percentage of runtime is not a diagnosis: print call counts and bytes per op, or a phase split, before
  choosing between "fix the layout" and "fix the dispatch".
- The XASR_PROF eval callback is itself distorting (~2x, and it races on a shared map from both worker
  threads) - treat only large, directionally consistent deltas as signal.
- `long double` on aarch64 is quad precision in software. Grep for it before believing a numeric loop is
  "just some accumulation".
