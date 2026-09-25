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
| x-asr encoder | 12.4 M useful (19 layers: dims 192/256/512/768/512/256, ffn 512/768/1536/2048/1536/768), run at **n = 3-24** | ~4.9 ms | ~12.7 G MAC/s |
| diar encoder | 97.5 M (31 layers, hidden 512, ffn 2048, qkv 1536), run at **n = 351-548** | ~4.9 ms | ~20 G MAC/s |

The x-asr figure is *useful* work; the graph actually evaluates a 61-row window to emit 12 encoder frames, so
its real MAC count is roughly 5x higher and its efficiency is correspondingly lower. The two legs landing on
the same ~4.9 ms per frame is a coincidence of size, not a shared bottleneck - and the shape measurement below
shows they are not even the same problem.

## Why: the shapes are skinny, and that is now measured

The race-free profiler (`XASR_PROF`, per-thread accumulators) with a MUL_MAT histogram answers the question
this brief previously said was inferable. On chat69 (`--no-diar`, armed):

| k x n | calls | us/call | share of accounted time |
|-------|-------|---------|------------------------|
| 512 x 6 | 9,344 | 166 | 7.9% |
| 768 x 3 | 5,840 | 230 | 6.9% |
| 1536 x 3 | 2,190 | 303 | 3.4% |
| 256 x 12 | 4,672 | 101 | 2.4% |
| 1152 x 6 | 2,336 | 187 | 2.2% |

Essentially **all** of the encoder's matmul time is in calls with **n between 3 and 24 columns**, and the top 12
of 54 distinct shapes cover about a third of it. The cause is structural: the encoder downsamples time by
[1, 2, 4, 8, 4, 2] across its six stages, so a 24-frame chunk arrives at the deep stages as 6, 3 and 3
columns - and those deep, wide stages (dims 512-768, ffn 1536-2048) hold most of the parameters. A GEMM with
n=3 gets single-digit percent of peak no matter how good the kernel is: there is no reuse of the weight matrix
across columns to amortise the stream, and the per-call setup and thread barrier dominate.

**The diar leg is the opposite case, and it is the better target.** Measured structurally on the same clip
(AUDIOCPP_PROF, chat69): its MUL_MAT shapes are **k = 512 / 1536 / 2048 with n = 351, 380 and 548** - 20 to 30
nodes per shape per window, 3.7-11.6 M elements each. Those are *conventional* GEMM shapes: hundreds of columns,
plenty of cross-column reuse, the regime any tuned int8 kernel is designed for. It also explains the measured
efficiency gap between the legs (diar ~20 G MAC/s versus x-asr ~12.7) without appealing to kernel quality at
all: same library, same instruction set, different shapes.

So the two legs are **different kernel problems**, and the ranking inverts:

| leg | n | what a tuned kernel can do | difficulty |
|-----|---|--------------------------|-----------|
| diar encoder | 351-548 | the ordinary case: 2-3x is plausible, no structural obstacle | moderate - it is a drop-in GEMM replacement |
| x-asr encoder | 3-24 | only a dedicated small-N path helps; large-N work is irrelevant | high - and batching is impossible (below) |

**Prioritise the diar.** It is roughly a quarter of the composite (14.9 s of 60.9 s), it is already feeding
kernels the right shape, and it is the leg this project has spent the least time on. If the small-N kernel work
ever becomes possible, it is a second project - not the first.

**This has a consequence that rules out the usual remedy for the ASR.** Batching skinny GEMMs is the standard fix, and it
is unavailable here: the number of columns IS the number of time frames after downsampling, which the model's
streaming chunk fixes. Enlarging the chunk is exactly the change measured at +5.9 WER on the primary gate clip,
and halving the step count while keeping the same rows would recompute rows the model already emitted. So the
skinny shape cannot be engineered away in-graph - it has to be handled by a kernel that is good at n=3, which
is precisely what a tuned small-N GEMM path is for. That is a sharper requirement than "make the GEMM faster":
**the kernel must be fast at n = 3-24, k = 512-2560.**

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

0. **Nothing to convert.** An earlier draft proposed converting the diar's weights to q8_0 to reach ggml's
   `.nrows = 2` path, on the belief that they were q5_0 (which is hard-coded to `.nrows = 1`). They are already
   q8_0, so the two-row path is already in use on both legs and there is nothing to win here. Recorded because
   the idea is superficially very convincing and the belief underneath it was wrong.

1. **A block-scoped Q8_0 activation cache** (mechanism 1) - **MEASURED, AND NOT WORTH BUILDING.** I counted the
   volume rather than assuming it (ggml's `g_ggml_quant_elems`, behind `GGML_COUNT_QUANT`): **180.8 M activation
   elements are quantised per 69 s clip** (118.8 M for the 45 s clip). That sounds like a lot until you divide by
   what each element is worth: a quantised activation element feeds one MAC per OUTPUT ROW, so with
   `n_out` in 512-2048 it is reused roughly a thousand times. The conversion is therefore ~1/1000 of the
   arithmetic, about 56 ms per clip at 1-2 cycles/element across two threads - **0.7% of the ASR leg, below the
   0.4% noise band once you account for run-to-run spread.** The mechanism is real; the prize is not. Do not
   build it. (The measurement is also a useful cross-check on the rest of this brief: it is what reconciled the
   apparent 100x gap between '12.4 M MACs per frame' and '180 M quantised elements per clip' - the ratio is
   simply the average number of output rows per activation element.)
2. **A specialised Q5_0 x Q8_0 GEMM for the shapes these two models actually use** (mechanisms 2-3), now with
   the shape requirement measured rather than assumed: **k in {192..2560}, n in {3, 6, 12, 24}** for the x-asr
   encoder, plus n in {380, 508} for the diar. int8 accumulate into int32, output f32, two threads on two A78s.
   The small-N path is the one that matters; a kernel tuned for large N would not help this workload at all. This is what KleidiAI's `ai_micro_kernels` provides, and it is the reason that
   dependency is the only open lever: both vendored ggml copies ship the glue (`kleidiai.h`, `kernels.h`, the
   CMake option) but no `ai_micro_kernels` exists anywhere on this machine, so the build cannot be completed
   offline. KleidiAI accumulation order differs from ggml's, so it needs the paired validation and a
   deliberate re-bless - not the byte-identity gate.
2b. **If 2 lands,** re-measure mechanism 1 before dismissing it: a 2-3x faster dot makes the conversion a
   correspondingly larger share, though still small.

3. **Only if 2 lands:** fusing the surrounding elementwise chain. Measured worthless on its own (four
   separate 0% results: the ASR leg is matmul-bound and its elementwise work is already overlapped underneath),
   but it stops being separate work once the GEMM itself is fast.

## The ceiling, MEASURED (and it is much lower than the first estimate)

`tools/gemm_shape_bench.cpp` runs ggml's own q8_0 x q8_0 path on the device across the shapes the models
actually use, sweeping k, columns n, and rows m. Two results matter:

1. **Rows do not matter.** m=4096 (eight times the weight bytes) gives the same ~19-20 GMAC/s as m=512, so the
   kernel is not weight-bandwidth-bound: the weight stream is fully hidden behind arithmetic.
2. **Small n costs a fixed ~2x.** At n = 3-12 the same kernel delivers ~9-11 GMAC/s regardless of k or m, and
   the knee is sharp between n=12 and n=24, after which it plateaus at ~20 GMAC/s.

Putting those next to the in-situ measurements is what changes the picture:

* The **diar encoder already runs at the plateau** - ~20 GMAC/s measured in the model, 20.6 GMAC/s in the
  benchmark at n >= 24. There is no headroom for a better GEMM to collect on that leg unless the kernel can
  raise the plateau itself.
* The **x-asr encoder is already at the small-n rate**: ~12.7 GMAC/s in the model against 10.4 GMAC/s in the
  benchmark at n=3. A small-N-specialised kernel could at most chase the ~1.9x gap to the plateau, and would
  have to beat a kernel that is already there.

So the first estimate in this brief - 25-35% of the composite from a tuned GEMM - was too high, and it was
high because it compared the models against a theoretical SDOT peak rather than against what this library
achieves on any shape. The defensible statement is narrower: **a shape-specialised kernel is worth at most
about 1.9x on the ASR's skinny GEMMs and roughly nothing on the diar's fat ones**, and whether it can beat
20 GMAC/s at all is the question a KleidiAI evaluation has to answer first. Measure that before budgeting a
kernel project.

## The ceiling as first estimated

(Superseded - see the measured ceiling above. Kept because the error is instructive: 40-60% of int8 peak is
what a tuned kernel reaches on a desktop-class core with a fat GEMM, and this workload has neither. The
measurement above is the number to plan against.)

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


## 13. Correction: the GGUF type ids were being read with the wrong enum

An earlier draft of this brief (and several ledger rows) described both models as shipping `q5_0` weights, and
brief proposed converting the diar's `q5_0` tensors to `q8_0` so the matmuls would reach ggml's `.nrows = 2`
dotprod path, which `q5_0` does not have. **Both claims were wrong, and the error was a measurement error
rather than a modelling one.**

These two vendored ggml copies number the quantised types `Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8`.
The histograms that produced the "q5_0" story were read with the older llama.cpp numbering
(`q8_0 = 7, q5_0 = 8`), in which type 8 means q5_0. In these trees type 8 is **q8_0**, so:

* x-asr is 619 F32, 49 F16 and **298 Q8_0** - not q5_0.
* the Nemotron-3 diarizer is **130 Q8_0** - which is what its filename always said - plus 229 tensors of an
  audio.cpp-specific type id, which `tools/gguf_inventory.py` resolves by density to **16 bits per element**
  (F16/BF16-class storage: the norms and biases, 1.0 M elements, 2.0 MB).

Two independent checks confirm the reading, and both were available before the claim was written:

1. **Payload density.** 278,528 bytes for a 512x512 tensor is 1.0625 bytes/element - 8.5 bits - which is
   q8_0's density exactly (34 bytes per 32 values) and not q5_0's (5.5).
2. **ggml's own answer.** `gguf_get_tensor_size` reported the same 278,528 where my q5_0 arithmetic expected
   180,224. When the library and your arithmetic disagree about a file format, the library is right; the useful
   move is to ask it, which is what `gsize.c` in this session now does.

What survives, and what does not:

* **Does not survive:** "the diar's weights are q5_0", "the diar is missing the two-row dotprod path", and any
  conversion motivated by either. The tools/requant_gguf.py written to do that conversion has been deleted
  rather than left behind as a plausible-looking tool for a job that does not exist.
* **Survives, and is reinforced:** weight format cannot help this composite. Every quantised ggml type declares
  `vec_dot_type = Q8_0`, the shipped weights are already Q8_0, and Q8_0 is the type that gets `.nrows = 2` under
  dotprod. The conclusion was right and the reasoning behind it was wrong, which is worth recording separately -
  a right answer for the wrong reason will not survive the next person who looks at the evidence.
* **Worth noting about the requantisation sweep:** its "all q8_0" arm was probably vacuous, because the model was
  already q8_0. The other arms (q4_k, q4_0, iq4_nl, q6_k) were genuine conversions and tied or regressed, so the
  sweep's conclusion stands; one arm of it just was not a test of anything.
* **Unaffected:** every shape and timing measurement in this brief. Those came from the profilers, not from the
  dtype histograms.


## 14. The audit that would have caught it, and what it says

`tools/gguf_inventory.py` reports every tensor by type **and by bits per element**, where the density comes
from the payload size and the element count alone - no enum involved. Run on both models:

| model | type | tensors | elements | bits/elem | verdict |
|-------|------|---------|----------|-----------|---------|
| x-asr | Q8_0 | 298 | 151.0 M | 8.50 | matches Q8_0 |
| x-asr | F32 | 619 | 1.6 M | 32.00 | matches F32 |
| x-asr | F16 | 49 | 0.6 M | 16.00 | matches F16 |
| diar | Q8_0 | 130 | 98.2 M | 8.50 | matches Q8_0 |
| diar | type30 | 229 | 1.0 M | 16.00 | F16/BF16-class: the norms and biases |
| diar | F32 / F16 | 1 / 2 | - | 32 / 16 | matches |

A tensor labelled `q5_0` that occupies 8.5 bits per element is visibly not a `q5_0`, whatever the type id
says - so this table is self-checking, and the cross-check against `gguf_get_tensor_size` is the second half.
Both halves are cheap; the error they prevent was not.
