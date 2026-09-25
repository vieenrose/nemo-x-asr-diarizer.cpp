# The real merge: one runtime, one thread pool, one allocator

A study, in three parts: what "real" means here, what the merge is worth (measured, not estimated), and what
it would take. Numbers are from the phone at `taskset C0`, armed and screened, at the current tree
(phone_rtf 0.267, byte-identical on the gate clips).

## 1. Three different things get called "merging two models"

| level | what it means | status |
|-------|---------------|--------|
| **container** | one GGUF file holding both models; one mmap, one artifact, one deploy step | **done** — `tools/merge_gguf.py`, `--models-bundle`; reproduces the blessed transcript hash exactly, measured neutral |
| **shared process** | one loader, one weight buffer, one allocator/arena for both models | partly there (one process, two `.so`-side runtimes) |
| **one runtime** | both graphs execute in ONE ggml: one `ggml_backend_sched`, one thread pool, one arena — so their ops can overlap | **this document** |

Only the third is usually meant by "really merged", because it is the only one that changes what the hardware
can do. This study finds that it is feasible, and that it is worth less than the earlier estimate said.

## 2. What it is worth: at most ~10%, and that is the ceiling before any risk

The composite is sequential by construction: per 100 ms piece, push to the diarizer, then accept on the ASR,
then attribute. The two graphs are independent for a piece (only *attribution* needs the diar turns, and that
is cheap host work). So in principle they could overlap — if there were idle capacity to overlap into.

**There is very little of it.** From the current protocol row (wall 61.35 s = asr 38.50 + diar 14.96 + drain 7.89):

* **100% of the wall is engine compute.** There is no unattributed loop overhead to reclaim: `other_s` is not
  slack, it is the final diar window (instrumented directly inside `Session::finalize`).
* **`cores_used` is 1.81 of 2**, so the mask is ~9.5% idle on average.
* The single-threaded moments that *explain* that idle are 1.87 s of a 19.1 s per-clip wall (9.8%): the ASR's
  host phases (0.58 s: cache copies and input writes), the joint/greedy pass (1.03 s), the diar's mel frontend
  (0.24 s) and pre-encode (0.02 s). Note these and the idle figure are **the same opportunity counted twice** —
  the moments when only one core is busy ARE the idle mask — so the honest bound is **~10%, not ~20%**.

And the drain (12.9% of wall) cannot overlap with anything: it is the last thing the run does, and its output is
needed for the final attribution.

So the *performance* case is: **≤10% of the composite, in the best case where overlap is perfect and adds no
contention.** An earlier version of the design note estimated ~20% from the 1.61-of-2 utilisation of the time;
utilisation is now 1.81, and that estimate is superseded.

## 3. Why the naive version failed, and what the single pool changes

The obvious attempt — diar on a worker thread — was measured at **37-47% slower**, with *both* legs slowing
~2.8x. That is not "concurrency is bad"; it is two independent ggml thread pools (2 for the ASR, ~8 created
lazily by the diarizer) plus two ~100-170 MB working sets contending for two A78s. A single pool removes the
double-pool pathology by construction, and it is the only form worth attempting — but the prize from §2 is
unchanged by that, because §2 is about idle capacity, not about who owns the threads.

## 4. Feasibility: better than expected, because the two ggml trees are nearly the same

The port means reimplementing the diar encoder's graph in CrispASR's ggml and moving the streaming scheduler
and AOS state machine with it. What makes that less alarming than it sounds:

* **The two runtimes share 43 of 45 tensor types.** They differ only by `Q2_0` (CrispASR only) and `I2_S`/`I8_S`
  (audio.cpp scalars). No meaningful op-set divergence.
* **The module vocabulary is small and ordinary**: Linear, LayerNorm, Gelu, Sigmoid, Add, Slice, Transpose,
  SplitRoPE, GroupedQueryAttention.
* The weights are already in the shared format (Q8_0, 98.2 M elements, verified by density — see
  kernel-brief §13), so parameters can be loaded into the other context unchanged.

The hard part is not the API, it is **exactness**. The port must reproduce the *op sequence* — not merely an
equivalent computation — or the output moves: any change in accumulation order, any fused-vs-unfused
substitute, any different norm implementation changes the bytes. And the AOS state machine (speaker cache,
FIFO, compression) is audio.cpp logic that has to move as C++ regardless of which runtime draws the graph.

## 5. Staged plan, cheapest test first, with a kill criterion at every step

| stage | what | cost | kill criterion |
|-------|------|------|----------------|
| **0. Measure the prize** | serial fraction and idle mask | done (§2) | if overlap can only reach <5%, stop — **result: ~10%, continue but with low expectations** |
| **1. Cross-runtime bit-identity, one layer** | build the diar encoder's layer 0 in CrispASR's ggml; compare its output against audio.cpp's, bit for bit | small | **if the two runtimes do not agree bit-for-bit on one layer, stop** — the entire premise is byte-identity, and this tests it for the price of one layer |
| 2. Full encoder graph | all 31 layers + pre-encode + head | large | any divergence → revert; the port is not viable bit-identically |
| 3. Streaming + AOS | move the scheduler, speaker cache, FIFO, drain | large | if the drain's output changes, the accuracy evidence must be re-run from scratch |
| 4. One sched, both graphs | engine issues both graphs to one scheduler | moderate | **the real test**: if wall does not improve by ≥5%, the project has spent everything for nothing — say so at this point rather than stage 5 |
| 5. Overlap | schedule ASR and diar ops concurrently | moderate | — |

Stages 0 and 1 are cheap and together they decide whether stages 2-5 are worth starting. **Stage 1 is the one
to do next**: it is the only experiment that can kill the whole idea for the price of one encoder layer, and
the 37-47% two-pool result means the idea needs a real test rather than another argument.

## 6. The case that is not about speed

If the GEMM work ever starts (kernel-brief: diar first, ASR second, and the measured ceiling is ~20 GMAC/s
rather than a theoretical peak), the one-runtime merge becomes considerably more valuable for a non-obvious
reason: **both models would share one tuned kernel set.** Today a kernel improvement has to be made twice, in
two vendored ggml trees, and a bug fixed in one does not fix the other — the session hit exactly that when it
found the same per-element division and thread-partition issue in both copies separately. One runtime is
also one weight buffer, one allocator and one place where a memory or numerics bug can hide.

So the recommendation is: **do stage 1 now** (cheap, decisive), and treat stages 2-5 as justified only if
stage 1 passes *and* either the overlap prize survives a more careful bound or the kernel work has started.
