# Ideas backlog — composite phone RTF

Everything under "measured out" was measured, not guessed. `prompt.md` and `docs/pipeline-design.md` carry
the numbers; this file is the queue.

## Where the composite stands (2026-09-25, end of session)

`phone_rtf` 0.4651 -> **0.269**, byte-identical on the gate throughout, from five changes:

| change | where | effect | numerics |
|---|---|---|---|
| diar `spkcache_len` 264 -> 128 | engine default | -9% | re-blessed with 11-clip evidence |
| stop padding diar encoder windows to capacity | audiocpp | -1.2% | identical |
| mel filterbank: skip exact zeros, drop `long double` | audiocpp | -16.6% | identical |
| build the x-asr chunk graph once (fixes a UAF) | crispasr | -1.8% | identical |
| transducer joint matrix in f16 | crispasr | -12% (and -19% on the ASR leg) | gate + all 11 WERs identical |

Plus one merged GGUF (`--models-bundle`), measured neutral by design.

## Open, ranked

- **q8_0 joint weights.** f16 took the joint from 3.87 s to 1.11 s per 69 s clip; the phase is now small
  enough that a further 2x is worth ~1 s per clip (~2% composite) and would need the validation path again.
  Low priority.
- **One-runtime merge** (port the diar encoder into crispasr's ggml so one scheduler, one pool, one arena
  serve both). Ceiling is the 1.6-of-2-cores utilisation. The two-pool form is measured dead (37-47% slower).
  The container merge is already done and committed, which is the prerequisite.
- **KleidiAI** (`GGML_CPU_KLEIDIAI=ON`): the only untried kernel-level lever; needs a network fetch for the
  `arm_llama` kernels. Expect accumulation-order changes -> validation.
- **ggml-native streaming caches** on the ASR side: 114 cache tensors per step currently go
  graph -> host vector -> graph (outputs 0.563 s + inputs 0.278 s per 69 s clip). Bit-identical in principle;
  every other host-side phase has been partly hidden under the matmuls, so measure before believing it.
- **Re-profile both encoders** now that the bandwidth pressure is gone: the ASR encoder compute was measured
  at ~19 s per 69 s clip (profiled) with the joint competing for memory. Both are already on int8 SDOT
  (x-asr 298 q5_0 tensors, diar 130 q5_0 = 99.5 MB), so there is no precision lever left - only KleidiAI.
- **DONE, and the answer is no:** the diar encoder's carried speaker state cannot be cached across windows.
  Its attention mask is indexed by VALIDITY, not POSITION (it masks only keys past the valid length, and
  `build_encoder_layer` passes that tensor and nothing else), so attention over [state | fifo | chunk] is
  bidirectional: each state frame's representation depends on the newest chunk frames. Caching their K/V
  would convert bidirectional chunk attention into causal attention and change the output. An earlier entry in
  this file claimed the opposite, from misreading the same mask; the correction is in docs/pipeline-design.md
  §11. The encoder is linear in packed frames (4.85-5.46 ms/frame measured at 380/548/351 frames), so the
  state term is only reducible by a smaller `spkcache_len`, which moves speaker decisions.
- **KleidiAI** is the only remaining kernel-level lever and needs a network fetch. Everything else measured
  out below.

## Measured out — do not retry without a changed assumption

- Leg concurrency / diar on a worker thread: 37-47% slower; also pointless given the sum-of-parts floor.
- Requantising x-asr (q4_k / q4_0 / iq4_nl / q6_k / all-q8_0). Re-measured properly at 0.4% noise with an
  accuracy check: **Q4_0 is 3-10% faster and 72 MB smaller, and costs 32 WER points** (gate 0.1765 -> 0.4941
  against a 0.18 ceiling). The old "it ties" was measured in the +/-10% noise era with no WER check - wrong on
  both axes at once. Closed on evidence, and the arithmetic (Q8_0 is already the int8 SDOT path, the core's
  fastest) explains why nothing above Q4_0 can help. Every quantised type declares `vec_dot_type = Q8_0`, so the dot is
  int8 SDOT whatever the weights are; on A78 (2x128-bit NEON, ARMv8.2) int8 is ~4x fp32 and fp16 FMLA ~2x,
  so both models already sit on the fastest arithmetic the core has. f16 weights would move the diar encoder
  from the 4x path to the 2x path and roughly double its matmul time. Fewer FLOPs (different model) or a
  better int8 GEMM (KleidiAI) are the only remaining arithmetic levers.
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
  "just some accumulation" - that one line was 19% of the composite.
- **If a change moves the clock by exactly 0%, suspect the build.** Both build scripts used to swallow cmake
  exit codes, so a dependency that failed to compile was measured as a stale archive and read as a clean
  negative result. Both scripts now fail loudly. A build that cannot fail is a measurement that cannot be
  believed.
- Look for work *outside* the engine: the ggml op profile cannot see host C++ (the joint was 31% of the ASR
  leg and invisible), and host phases can matter even when they are not on the critical path, because they
  consume the same memory bandwidth everything else needs.
