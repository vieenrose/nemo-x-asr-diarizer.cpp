# Kernel brief: what a faster GEMM has to beat on this device

Companion to `pipeline-design.md`. That document says *what* the composite costs and *why* each lever is
closed. This one is the engineering brief for the only lever that is still open — the int8 GEMM inside the two
encoders — written from measurements rather than from the shape of the models.

## The target, in one line

Both encoders run their GEMMs through ggml's `vec_dot_q5_0_q8_0` path: activations are quantised to Q8_0 and
multiplied with the dequantised weights using int8 SDOT, accumulating in int32. On Cortex-A78 (ARMv8.2-A, two
128-bit NEON pipes) SDOT is roughly **4x the fp32 FMA rate**, so this is already the fastest arithmetic the
core offers — and the measured efficiency is **13-20% of that peak**.

| leg | useful MACs per emitted frame | measured cost per frame | effective rate |
|---|---|---|---|
| x-asr encoder | 12.4 M (19 layers: dims 192/256/512/768/512/256, ffn 512/768/1536/2048/1536/768) | ~4.9 ms | ~12.7 G MAC/s |
| diar encoder | 97.5 M (31 layers, hidden 512, ffn 2048, qkv 1536) | ~4.9 ms | ~20 G MAC/s |

The x-asr figure is *useful* work; the graph actually evaluates a 61-row window to emit 12 encoder frames, so
its real MAC count is roughly 5x higher and its efficiency is correspondingly lower. The two legs landing on
the same ~4.9 ms per frame is a coincidence of size, not a shared bottleneck - but it does mean a kernel win
applies to both legs almost equally.

## Why the current path leaves 4-7x on the table

Three mechanisms, in descending order of how much they are worth knowing:

1. **Activation requantisation is per matmul call.** Every `mul_mat` quantises its F32 activations to Q8_0
   before the dot. In a transformer block the *same* activation feeds three or four matmuls (q/k/v, the FFN's
   two projections, attention output), so the same tensor is requantised repeatedly, and always on the critical
   path of the call. A kernel that keeps activations in Q8_0 for the whole block removes that work without
   changing any arithmetic - the quantised values are identical, only computed once instead of three times.
2. **N is small.** The streaming window emits 12-48 columns per matmul. A GEMM tuned for large N amortises the
   weight stream and the requantisation across many columns; at N=12-48 the weight stream and the per-call
   setup dominate, which is exactly the regime where a well-blocked int8 kernel beats a generic one.
3. **Generic tiling.** ggml's q5_0/q8_0 dot is written for generality. A kernel that knows k (512/2048) and the
   block structure at graph-build time can pick tile sizes, keep accumulators in registers across the k loop,
   and overlap dequantisation with the dot instead of treating it as a separate pass.

## What to build, in order

1. **A block-scoped Q8_0 activation cache** (mechanism 1). Smallest change, no new arithmetic, bit-identical
   output: quantise once per tensor per step rather than once per use. This is the first thing to try because it
   is a scheduling win inside the existing kernels, not a rewrite. Expected: single-digit percent, and it is
   cheap enough to measure in one iteration.
2. **A specialised Q5_0 x Q8_0 GEMM for the shapes these two models actually use** (mechanisms 2-3). The brief
   for the kernel: k in {512, 1536, 2048}, n in {12, 24, 48, 380, 508}, int8 accumulate into int32, output f32,
   two threads on two A78s. This is what KleidiAI's `ai_micro_kernels` provides, and it is the reason that
   dependency is the only open lever: both vendored ggml copies ship the glue (`kleidiai.h`, `kernels.h`, the
   CMake option) but no `ai_micro_kernels` exists anywhere on this machine, so the build cannot be completed
   offline. KleidiAI accumulation order differs from ggml's, so it needs the paired validation and a
   deliberate re-bless - not the byte-identity gate.
3. **Only if 1 and 2 land:** fusing the surrounding elementwise chain. Measured worthless on its own (four
   separate 0% results: the ASR leg is matmul-bound and its elementwise work is already overlapped underneath),
   but it stops being separate work once the GEMM itself is fast.

## The ceiling, stated honestly

If the encoders reached 40-60% of int8 peak - what a tuned kernel achieves on shapes like these - the two
encoder legs (37.9 s + 15.0 s of a 60.9 s protocol wall) would fall by roughly 2-3x, i.e. **25-35% of the
composite**. That is an order of magnitude more than anything left in configuration, and it is the only reason
this brief exists. It is also the number most likely to be wrong: it extrapolates from arithmetic intensity and
measured efficiency, not from a prototype, and a kernel that only reaches 30% of peak would give half of it.

## Before you start: fix the profiler

`XASR_PROF=1` prints a per-op time mix, and it is **not trustworthy as it stands**: the eval callback keeps one
`std::unordered_map` that both worker threads touch without a lock, so it races, and the callback's own
per-node cost inflates the leg roughly 2x. It pointed at real problems (it is how the per-element division was
found) but it cannot be used to decide whether a new kernel helped. Give each thread its own accumulator, or
aggregate under a mutex, and re-baseline before trusting any number a kernel change produces. The per-phase
timers around the engine calls (`NEMO_PROF`, `AUDIOCPP_PROF`) are sound and are what established the
~4.9 ms/frame figures above.

## What "done" looks like

- ms per encoder frame on both legs, from the phase timers, at a fixed clip and a pinned frequency.
- The byte-identity gate for mechanism 1 (it should be bit-identical: same quantised values, computed once).
- The full 11-clip paired validation for mechanism 2, plus first-partial and p95 per clip - the new joint
  numerics can move an early argmax, and that is a contract, not a metric.
- A protocol row that is not thermally confounded: cool ~180 s first, or interleave candidate and baseline in
  one armed window. The row-to-row band is 0.4% cool and ~10% hot, and sub-1% claims are meaningless without
  that discipline.
