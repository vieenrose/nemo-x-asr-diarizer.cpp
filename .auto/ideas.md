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

## 2026-09-26: `scripts/build_android.sh` did not build from a clean checkout - fixed, and it was hiding a real ISA win

Discovered while profiling the one-runtime-merge performance gap (candidate (a), docs/one-runtime-merge.md
§16): `echo | clang++ --target=aarch64-linux-android33 -dM -E -` on this NDK shows no
`__ARM_FEATURE_DOTPROD`, and the tracked build's binaries had zero `sdot` instructions (`llvm-objdump -d`).
Every quantised dot product on the phone was running the scalar fallback, both engines, despite §12/§14 of
pipeline-design.md asserting the int8 SDOT path was already active - that claim was never checked against the
actual compiled binary. `rm -rf ref/*/build-android && bash scripts/build_android.sh` from a clean checkout
does not build at all: 4 independent breaks (GGML_NATIVE forced ON overriding the correct cross-compile
default and crashing on `-mcpu=native`; CrispASR's cmake missing `-DBUILD_SHARED_LIBS=OFF`, producing `.so`
where the link step wants `.a`; GGML_OPENMP defaulting ON and pulling in a `libomp.so` the device does not
have; no `-march`/`-mcpu` at all, hence no dotprod), every one masked for who knows how long by a stale
`CMakeCache.txt` nobody had reconfigured from zero. Fixed all four, plus added explicit
`-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod` (confirmed via the phone's own `/proc/cpuinfo`: `asimddp` present,
`i8mm` absent - Cortex-A78 has dotprod, not matmul-int8). Two independent on-device A/B pairs (measure.sh,
armed, chat69+gate_ms_v2): clean/no-tuning phone_rtf 0.3124/0.3315 -> with dotprod 0.2617/0.2914 (**-12% to
-16%**, almost all of it the diar leg: **-30% to -34%**), and the transcript HASH is **byte-identical across
all four runs** (`d64b8863ba8d`) - SDOT sums the same int32 accumulator a scalar loop does, so this clears the
project's own byte-identity gate by construction, no WER re-check needed. Unresolved: the historical 0.269
composite baseline this whole session cites sits inside the dotprod range, not the newly-discovered clean
range - most likely that number's build was *also* riding a stale cache with dotprod already on by accident,
but this is not verified, only flagged. Full writeup: docs/pipeline-design.md §15.

Re-measured `--diar-native` (docs/one-runtime-merge.md §16-17) on the dotprod build: the RTF gap is unchanged
(~11% slower diar leg, same as before dotprod existed - both paths share the same kernel, so this closes
candidate (a) as "real bug, fixed, but not the `--diar-native` differentiator"). Re-measuring surfaced a new
number instead: `--diar-native` peak RSS is 1568 MB vs the default path's 396 MB on a 45 s clip. Found and
fixed a real bug behind it (`src/diar_crispasr.cpp::encode()` allocated the new per-shape activation buffer
before releasing the old one, so both were resident during every rebuild - packed frame counts almost never
repeat, so this was nearly every call) - WER-neutral, verified. **The fix did not move peak RSS at all**
(1568 MB, unchanged, 3 runs across 2 binaries) - meaning the order was not the actual constraint; most likely
a single large deterministic allocation that glibc's allocator retains after `free()` rather than returning
to the OS. Kept the fix (free and strictly correct) but the memory gap is open, unresolved, and points at the
redesign candidate (b) already named in §16 (a fixed-max-capacity graph reused via views instead of rebuilt)
as the next real step. `--diar-native` stays default-off. Full writeup: docs/one-runtime-merge.md §18.

**Correction, same tick:** candidate (b)'s own premise ("packed frame counts almost never repeat") was never
actually checked at `DiarCrispASR::encode()`'s call granularity - added a one-line env-gated counter
(`DIARCRISPASR_DEBUG_T`) and found it doesn't hold there: `gate_ms_v2.wav` makes only 2 total `encode()`
calls in 45 s, `holdout_en.wav` only 6 in 139.56 s, and half of those 6 reused the cached graph with zero
rebuild (3 distinct T out of 6 calls). `encode()` fires roughly once per ~20-25 s of audio, not once per
streaming chunk. The fixed-max-capacity redesign is very unlikely to move the ~11% RTF gap (there are only
2-3 rebuilds per clip to save on) - deprioritised, not attempted. With all three SS16 candidates now
addressed, the ~11% gap's real cause is unidentified - likely a small genuine per-call kernel/scheduling
difference, not overhead from graph churn. Caught before spending the redesign effort, not after.

**Correction, next session ("continue toward the goal"):** the ~11% figure itself was wrong - not the
comparison, the *metric*. `[stats]`'s `diar` figure structurally excludes the clip's LAST diarization window
(its flush happens inside `audiocpp_stream_finish`'s drain phase, timed separately) - true for BOTH paths
equally (confirmed by re-running default with the same new diagnostic), so it isn't a native-specific bug,
but it does mean the diar-only comparison SS16-18 used wasn't measuring the full cost on either side.
`wall`/`rtf` are unaffected (drain IS included in final wall time) - only the diar/asr/other breakdown was
off. Re-measured on wall time instead: default 18.92-18.94s, native 19.84-19.90s on `gate_ms_v2.wav` -
**~5% slower, not ~11%**. Confirmed the per-call compute itself is real and large (~3.1-3.6s/call, isolated
via new `tools/diar_crispasr_bench.cpp` with zero engine/ASR interference) - not an artifact - just smaller
in aggregate than previously stated. Still default-off, still an unresolved (smaller) gap, but this is a
materially more accurate number for anyone deciding whether closing it is worth the effort. Full writeup,
and the two new permanent diagnostics (`DIARCRISPASR_PROF`, `ENGINE_DIAR_PUSH_PROF`) kept for next time:
docs/one-runtime-merge.md §19.

**Continued, same thread:** diffed this port's attention block against `ref/audiocpp`'s own
`GroupedQueryAttentionModule` and found a real, fixable difference - audio.cpp's `FlashGroupedViewKV`
lowering skips materialising K/V into contiguous memory before flash attention (only Q gets copied); this
port copied all three. Fixed (K/V now a single strided `ggml_view_4d`, zero copies), verified byte-identical
on host and device before measuring anything. Peak RSS dropped a real 47 MB (1564 -> 1517 MB) but wall time
did not move outside noise (~5.3% slower, same as SS19's figure) - a correct, worthwhile fix that is not
where the gap comes from. Four candidates now addressed (ISA tuning, graph churn, thread contention, this
copy), the gap held every time - it's diffuse, not one fixable thing findable without a real op-level
profiler. Stopped here. Full writeup: docs/one-runtime-merge.md §20.

## Open, ranked

- **One-runtime merge** (port the diar encoder into crispasr's ggml so one scheduler, one pool, one arena
  serve both). Ceiling is the 1.6-of-2-cores utilisation. The two-pool form is measured dead (37-47% slower).
  The container merge is already done and committed, which is the prerequisite. 2026-09-26: the "attention
  is 100% non-finite" blocker in docs/one-runtime-merge.md §10-11 was a broken oracle, not a port bug -
  `attention_mask` was never pinned as a graph output (audiocpp `deps.lock` now 58e8496, gated on
  `AUDIOCPP_DUMP_LAYER0` after an unconditional version measurably moved production confidence scores) and
  the layer0_port harness separately had a Q8_0-row-size offset bug on the Q/K/V slice (this repo,
  `tools/layer0_port.cpp`). Both fixed; attention is finite now but the layer's final output is still wrong
  (max delta ~30). Full writeup and next steps: docs/one-runtime-merge.md §12.
  2026-09-26 (same session, continued): built `run_layer0_isolated` (audiocpp `deps.lock` now 3d3d2e1) - a
  sound single-layer oracle, byte-identical final output to production, 7/12 stages now trustworthy (up from
  0). Diffing the port's `03_q`/`04_k` against it showed a permuted-but-valid rotation (same per-head RMS,
  ~0.1% elements matching) - `tools/layer0_port.cpp`'s RoPE cos/sin feed applied an unnecessary reindex loop;
  the dump's own `.ne` file already showed the target layout. Deleted the loop: **layer 0 port is now
  byte-identical to audio.cpp**, verified on 3 windows (339 and 679 frames). Stage 2 of the one-runtime merge
  is done. Full writeup: docs/one-runtime-merge.md §13. Generalized the dump/port tools to any layer index
  (audiocpp `deps.lock` now 0babdf9, `PORT_LAYER_INDEX` in layer0_port.cpp) and spot-checked layers 0, 15, 30
  (first/middle/last of 31) - all byte-identical with their own real weights.
  Then closed it out for real: `run_encoder_isolated` (audiocpp `deps.lock` now be2af81) chains all 31 layers
  as a sound whole-stack oracle, and new `tools/encoder_port.cpp` loops the proven per-layer sequence over
  every real layer (auto-detects layer count from the GGUF). **`ENCODER PORT (31 layers): BYTE-IDENTICAL to
  audio.cpp`** on two independent clips (339 and 527 frames) - the entire encoder, not a spot check. Stage 2
  is fully done: docs/one-runtime-merge.md §14.
  Scoped the rest (§14 update): the AOS state machine (streaming.cpp) needs no port at all - every function
  is plain float*/std::vector<float>, indifferent to which runtime produced the numbers. Ported the head
  instead (`run_head_isolated`, audiocpp deps.lock now a9c032a; `tools/head_port.cpp`): byte-identical
  through every op except the subpixel-upsample conv1d, which lands ~1e-3 off near unit scale - traced to a
  CrispASR-local ggml patch that forces F32 im2col (audio.cpp's unpatched ggml hardcodes F16; forcing F16 in
  the port crashes with `GGML_ASSERT(src1->type==F32)`, confirming CrispASR's ggml cannot take that path at
  all). Permanent, well-understood, not a bug - CrispASR's path is higher precision, not lower. Full
  writeup: docs/one-runtime-merge.md §15.
  2026-09-26, wired the actual merge: `Session::set_external_encoder()` (audiocpp `deps.lock` now 4d3de79)
  redirects JUST encode() to a caller-supplied ggml runtime, everything else (mel, scheduler, AOS state, turn
  decoding) stays audio.cpp's own code - deliberately not reimplemented (`AoscState::compress()` is fragile
  score logic with no gate to catch a transcription bug). `DiarCrispASR` (new, `src/diar_crispasr.h`/`.cpp`)
  is the callback, gated behind `--diar-native` (default off). Runs end to end, correct (same transcript
  text, small turn/speaker-label shifts matching the known ~1e-3 head gap), **not yet faster**: diar leg
  ~10-13% WORSE on both gate clips after fixing two real bugs (rebuilding all 31 layers' weights on every
  call - fixed by splitting persistent weights from the per-shape activation graph; reopening the GGUF ~350
  times per call - fixed with one open). The "one shared scheduler is faster" hypothesis is not confirmed by
  this first working version - full writeup, measured numbers, and next steps (DER check, then profile
  instead of guessing among 3 candidates): docs/one-runtime-merge.md §16.
  2026-09-26, item (1) closed: `.auto/validate.sh --compare default native` across all 11 gate/holdout clips -
  WER delta `+0.0000` on every clip, micro S/D/I/H identical to four figures (245/69/11/2865 both tags),
  0 worse/0 better/11 equal. `--diar-native` is measured WER-neutral, not just assumed so. A stricter
  non-WER text-diff check (labels/line-splits stripped) shows nonzero word-level edits on 5/11 clips
  (up to 29 on `control_ls`), concentrated in the already-higher-WER `holdout_en*`/`control_ls` clips -
  consistent with the ~1e-3 conv1d head gap occasionally relocating an error rather than adding one, since
  two different hypotheses can tie on edit count against the same reference. No ground-truth DER labels
  exist for these clips, so this is a WER-neutral finding, not a DER-neutral one; the label-churn column
  remains the only (proxy) diarization-accuracy signal, and it stayed bounded. Full writeup:
  docs/one-runtime-merge.md §17. Next: item (2), profile the 3 performance candidates instead of guessing.
- **ggml-native streaming caches** on the ASR side: 114 cache tensors per step currently go
  graph -> host vector -> graph (outputs 0.563 s + inputs 0.278 s per 69 s clip). Bit-identical in principle;
  every other host-side phase has been partly hidden under the matmuls, so measure before believing it.
- **Re-profile both encoders** now that the bandwidth pressure is gone: the ASR encoder compute was measured
  at ~19 s per 69 s clip (profiled) with the joint competing for memory. **Correction, 2026-09-26:** "already
  on int8 SDOT" was false when this was written - `scripts/build_android.sh` never actually compiled dotprod
  in (docs/pipeline-design.md §15); that ~19 s number predates the fix and may no longer be accurate now that
  x-asr's Q8_0 kernels genuinely dispatch to SDOT. Measured instead, at the leg level: dotprod left the ASR
  leg flat (36.3-38.1s pre-fix range vs 36.7-41.1s post, noisy, no clear direction) while the diar leg fell
  30-34% - consistent with the ASR leg's dominant cost being memory-bandwidth-bound (§9: the joint's own win
  came from cutting bytes streamed per frame, not from faster arithmetic), so a purely-arithmetic lever like
  dotprod has less to work with there. Not confirmed with a phase-level profile (the XASR_PROF callback is
  itself ~2x-distorting per this file's own method notes) - flagged as the honest next step if anyone wants
  a real answer rather than a leg-level proxy. No untried kernel-level lever remains (KleidiAI tried and
  closed below, measured worse).
- **DONE, and the answer is no:** the diar encoder's carried speaker state cannot be cached across windows.
  Its attention mask is indexed by VALIDITY, not POSITION (it masks only keys past the valid length, and
  `build_encoder_layer` passes that tensor and nothing else), so attention over [state | fifo | chunk] is
  bidirectional: each state frame's representation depends on the newest chunk frames. Caching their K/V
  would convert bidirectional chunk attention into causal attention and change the output. An earlier entry in
  this file claimed the opposite, from misreading the same mask; the correction is in docs/pipeline-design.md
  §11. The encoder is linear in packed frames (4.85-5.46 ms/frame measured at 380/548/351 frames), so the
  state term is only reducible by a smaller `spkcache_len`, which moves speaker decisions.
- **DONE and shipped, opt-in: q8_0 joint weights.** Implemented (`ref/crispasr` 86bf8fe, reusing ggml's own
  `quantize_row_q8_0`/`ggml_vec_dot_q8_0_q8_0`, not hand-rolled), and found the same class of bug as SS15
  fixing it: the `xasr` CMake target never inherited `GGML_CPU_ARM_ARCH`'s `-march` flag (only `ggml-cpu`
  did), so the new path first compiled to dead code - caught by measuring, not assuming. Once actually
  running: ASR leg -5 to -7% on all 11 validation clips (real, consistent). But unlike the f16 conversion
  (WER-identical on all 11 clips), q8_0 moves WER both directions - better on 9 clips, worse on 2
  (`gate_ms_g100` +0.05), and `holdout_en`'s WER exceeds this repo's own blessed ceiling. Net micro-WER across
  all 11 is better (0.1073 -> 0.1022) but "better on average, worse on some clips you can name" needed a
  decision, not just a measurement - shipped behind `XASR_JOINT_Q8=1`, off by default, confirmed byte-
  identical to the existing f16 path when unset. Full writeup: docs/pipeline-design.md §17.
- No open kernel-level lever remains - see KleidiAI in "Measured out" below. Everything else measured out too.

## Measured out — do not retry without a changed assumption

- Leg concurrency / diar on a worker thread: 37-47% slower; also pointless given the sum-of-parts floor.
- Requantising x-asr (q4_k / q4_0 / iq4_nl / q6_k / all-q8_0). Re-measured properly at 0.4% noise with an
  accuracy check: **Q4_0 is 3-10% faster and 72 MB smaller, and costs 32 WER points** (gate 0.1765 -> 0.4941
  against a 0.18 ceiling). The old "it ties" was measured in the +/-10% noise era with no WER check - wrong on
  both axes at once. Closed on evidence, and the arithmetic (Q8_0 is already the int8 SDOT path, the core's
  fastest) explains why nothing above Q4_0 can help. Every quantised type declares `vec_dot_type = Q8_0`, so the dot is
  int8 SDOT whatever the weights are; on A78 (2x128-bit NEON, ARMv8.2) int8 is ~4x fp32 and fp16 FMLA ~2x,
  so both models already sit on the fastest arithmetic the core has. f16 weights would move the diar encoder
  from the 4x path to the 2x path and roughly double its matmul time. Fewer FLOPs (different model) is the
  only remaining arithmetic lever - KleidiAI (a better int8 GEMM) was tried 2026-09-26 and measured worse,
  see below.
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
- KleidiAI without kernels (an earlier attempt, before the network was re-checked - did nothing, unsurprisingly).
- **KleidiAI, actually tried, 2026-09-26.** The "verified network-blocked" premise was stale: `curl` to
  github.com's release asset succeeds in this environment, and `-DGGML_CPU_KLEIDIAI=ON` fetches, configures,
  and builds cleanly for both vendored ggmls, `aarch64-linux-android33`. Ran on-device: no crash (self-selects
  the DOTPROD-only kernel variant, correctly skipping `i8mm` this chip lacks), transcript byte-identical to
  the dotprod-only baseline. Performance: **diar leg ~62% SLOWER** (3.39s -> 5.49s/5.50s, reproduced twice,
  same clip/mask), ASR leg flat. Full writeup: docs/pipeline-design.md §16. Not wired into
  `scripts/build_android.sh` - closed, not an open lever.
- One shared ggml between the two engines (dies at `GGML_ASSERT(*cur_backend_id != -1)`).

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
