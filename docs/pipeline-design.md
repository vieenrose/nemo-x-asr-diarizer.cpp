# Composite phone pipeline: where the time goes, and what a merged/fused design has to beat

Written after the 2026-09-25 measurement round, from numbers taken on the phone (taskset C0, armed and
screened). Every claim below is either a measurement from this repo or a quote from the source it refers to.
Where something is a design proposal rather than a measurement, it says so.

## 1. Can the two models be merged into one GGUF?

**Yes for the container, and it buys ~0% on its own. The interesting merge is one runtime, not one file.**

GGUF is a header, a K/V metadata block, a tensor table and a blob. Merging is mechanical: namespace the
tensors (`asr.*`, `diar.*`), union the metadata under per-model prefixes, concatenate the data section. The
x-asr GGUF has been parsed and rewritten in this session already (298 `q5_0` + 619 `f32` + 49 `f16` tensors,
architecture `xasr`, stages `z.0..z.5`, `emb.conv*`, `dec.*`, `join.*`).

The reason that is not the win: `Engine::init` is **0.20 s** on the phone, both files are already resident
after the first run, and the two graphs share no tensors. A single file is *plumbing* for the work below,
not a speedup. It does buy one mmap and one loader, which matters only if the graphs end up in one context.

## 2. What the composite actually costs

Armed, screened, per 69 s of chat69 (witness 2.3 GHz, 92-97% of time at the 2.4 GHz step):

| leg | time | note |
|---|---|---|
| ASR (`--no-diar`) | 15.4 s | 0.223 s per audio-second |
| diar (`--no-asr`) | 16.9 s → **7.3 s/run** now | 0.245 → ~0.11 after `spkcache_len` 264→128 |
| diar drain (`other_s`) | 2.6-3.1 s | instrumented: it is **one final encoder window**; `decode_turns` is 0.000 s |
| composite | 31.8 s → **26.5 s** | sequential; the composite is ~1.5% *faster* than the sum of its legs |

So there is no scheduling fat: the loop already runs at the sum of its parts, and the two legs are
compute-bound model inference.

## 3. Why the obvious kernel levers are already closed

This is the part that matters for any "write better kernels" plan, because three of the usual targets are
already known-dead on this device:

**Weight format does not matter, and now we know why.** Five requantisations (q4_k 89 MB, q4_0, iq4_nl, q6_k,
all-q8_0) all tied the shipped q5_0 within 2%, with the k-quants and iq4_nl *worse*. The reason is in
`ggml/src/ggml-cpu/ggml-cpu.c`: every quantised type declares `vec_dot_type = GGML_TYPE_Q8_0`. Activations are
quantised to Q8_0 and dotted with int8×int8 → int32, which on A78 is `SDOT`. The weight format selects how
the weights are *prepared*, not which hardware unit runs the dot. There is no bandwidth win to collect.

**fp16 FMLA is unreachable in this tree, by a deliberate patch.** `GGML_TYPE_F16` is overridden to
`vec_dot_type = GGML_TYPE_F32` with a local `ggml_vec_dot_f16_f32` (CrispASR issue #38): upstream's path casts
F32 activations to F16, which saturates above 65504 and produced NaN matmuls for models whose activations
exceed that. An f16 checkpoint would therefore still dot in f32 - which is exactly what the earlier
"weight_type f16 is slower" observation was. To use the A78's fp16 pipe you must either solve the saturation
(scale before the cast, or a custom kernel with an explicit range) or accept f32 dots.

**Elementwise and copy work is not on the critical path.** The op mix for the ASR leg (with the `XASR_PROF=1`
eval-callback hook, which is itself distorting - see §6) is MUL_MAT 33-52%, ADD 22%, CONT 18-22%, UNARY 6.7%,
MUL 4.5%, CONCAT 3.9%, IM2COL 3.4%. Three separate bit-identical optimisations of that 60% - dropping
`ggml_cont` around elementwise slices, single-threading small binary ops past a work threshold, and removing
a 64-bit integer division from the strided-elementwise index arithmetic - cut profiled CPU time by up to 28%
(26.7 s → 19.2 s accounted, CONT 58 → 13 µs/call) and moved the **wall clock by 0.0%**. With `cores_used`
at 1.61 of 2, ggml's scheduler was already overlapping that work under the matmuls. The ASR leg is
MUL_MAT-throughput-bound. Micro-optimising the elementwise chain is a dead end on this device.

## 4. What a merged, fused pipeline would actually have to change

Ranked by measured headroom, with the honest ceiling for each:

**(a) Fewer matmul FLOPs per audio-second.** This is the whole game. The only in-contract ways: a smaller
model (out of scope - the contract names this model pair) or kernel-level wins that reduce the *work* rather
than the copies. Nothing else in this list is worth more than 2-3%.

**(b) One thread pool instead of two - ceiling ~0.38, probably much less.** Composite = ASR 0.223 + diar
~0.11 s/audio-s; the sequential composite is already at the sum, so overlap cannot beat it outright. It can
only recover the 1.61/2 core utilisation, i.e. up to ~20% *if* a single pool avoids the contention that killed
the naive attempt. That attempt (diar on a worker thread) was 37-47% slower with both legs slowing ~2.8× - two
ggml runtimes, two thread pools, two 100-170 MB working sets. A *single* pool with one barrier and one
interleaved work queue is a different experiment, and the only version of it that can pay. It means porting
one model's forward pass into the other's ggml (the diar graph is the smaller, simpler one), because the two
runtimes are deliberately isolated - unifying them by linking was tried and died at
`GGML_ASSERT(*cur_backend_id != -1)`.

**(c) A shared frontend.** Both models consume the same 16 kHz stream. Today: one in-process Lanczos3
resample, then *two* independent mel frontends. A fused pipeline computes the filterbank once and fans out.
The measured prize is small - the frontend is inside the per-piece `diar`/`asr` timers and neither dominates -
but it is free once the graphs are in one process, and it removes a whole pass over the audio.

**(d) Graph-native streaming state (no host round-trip).** Every 480 ms step, each of the 19 zipformer layers
writes 6 caches (`ck`, `cn`, `cv1`, `cv2`, `cc1`, `cc2`) as `out_tensor(last_cols(...))`, the host reads them
with `get_output`, and `set_input` copies them back into the next graph. That is graph→host→graph traffic
that no fusion removes. llama.cpp's KV-cache pattern (keep the tensors in the graph, double-buffer) applies
directly, and it is bit-identical: the values never change, only who holds them. This is the one structural
change I would actually attempt next on the ASR leg.

**(e) Fused elementwise kernels (bias_norm, bypass, swoosh).** `bias_norm` is 6 nodes (sub/sqr/mean/sqrt/div/
scale), `bypass` is 3, `swoosh` is 3-4, per layer, per step. A single NEON kernel each would cut node count
~10% and is bit-identical if the reduction order matches ggml's sequential MEAN. But §3 says the elementwise
chain is not on the critical path, so the expected wall-clock gain is ~0. This is a *portability* and
*battery* win, not an RTF win, and should be labelled as such.

**(f) KleidiAI (`GGML_CPU_KLEIDIAI=ON`).** The only untried kernel-level lever with real headroom: Arm's
optimised GEMM/SDOT kernels for exactly the int8 path this model already uses. The glue is vendored
(`ggml/src/ggml-cpu/kleidiai/`); the `arm_llama` kernels are not, so it needs a network fetch. Accumulation
order will change → the full paired validation path, not byte-identity.

## 5. Streaming integrity, whatever gets built

Nothing above relaxes the contract. A fused pipeline must still be per-piece and incremental: `xasr_stream_*`
with its chunk caches on 100 ms pieces, `audiocpp_stream_*` into a `streaming` session, no re-decode, no
buffer-and-decode-once. The two constraints that bit during this round are worth keeping as design rules:
the diar drain is the *price of bounded turn latency* (chunk_len 510 is faster on the 69 s clip and slower on
the 45 s one, and pushes the first turn from 30.5 s to 44.1 s), and first-output latency is a gate, not a
metric to trade against.

## 6. Measurement hygiene (the part that keeps producing false positives)

- The `XASR_PROF` eval callback is an observer, but it is not free: it makes the ASR leg run ~28 s instead of
  ~12.5 s, because its per-node bookkeeping runs on both worker threads and touches one shared
  `std::unordered_map` with no lock. Its absolute numbers are unreliable; only large, directionally
  consistent deltas mean anything. A correct profiler needs per-thread accumulators.
- Every device number must be armed and screened (`arm.sh` + `witness.sh`). Unarmed rows are 20-60% slow and
  look completely reasonable.
- A percentage of runtime is not a diagnosis. "CONT is 18%" could be twenty huge tensors (fix the layout) or
  two hundred thousand tiny ones (fix the dispatch) - and those need opposite changes. Print call counts and
  bytes per op before deciding anything.
- Optimisations that are bit-identical by construction (layout, thread partitioning, index arithmetic) are
  the right first probes, but "no accuracy risk" is not "no risk of being pointless": three of them moved
  profiled CPU time a lot and wall time not at all.


## 7. Status: the container merge is DONE (and it is neutral, as predicted)

`tools/merge_gguf.py` builds one GGUF carrying both models: the diar keeps the native namespace (its tensors
are already architecture-prefixed, so it needs no code change beyond tolerating a name list that is a prefix
of the file's tensors), and the ASR is stored under `asr.` with crispasr's loader taught to try the prefixed
name first. `--models-bundle FILE` points both engines at it.

Verified: all 1328 tensor payloads byte-identical to the two source files, 40 KV entries intact, and the
four gate clips reproduce the blessed transcript hash `192184ebcd54977e` exactly - same output as two files.
On the phone (armed, 2352 MHz, 95.9% at 2.4 GHz): chat69 26.05 s vs 26.55 s, gate 16.59 s vs 16.58 s, peak RSS
404 MB either way, load 0.15 s vs 0.12 s. **Neutral, exactly as the arithmetic predicted** - and that is the
point: it buys one file, one mmap and one deployable artifact, not speed. Treat it as infrastructure for the
one-runtime merge, never as an optimization.

Three bugs were found and fixed while building it, all of the same species - silent, not loud:
1. GGUF array elements must not repeat the type field; writing one per element desynchronised the whole KV
   block (the diar metadata has arrays of strings, so it showed up immediately).
2. Tensor offsets are relative to the data section, not the file. Using the file size for the last tensor
   inflated it by the header length, and because `b'\0' * negative` is a no-op rather than an error, every
   later tensor landed early and the file still parsed.
3. A loader that returns a *default* when a key is missing (here: `vocab_size -> 0`) turns "wrong namespace"
   into "corrupt model". Any bundle-capable loader must try the prefixed name in its metadata helpers too,
   not only in its tensor lookups.


## 8. What the instrumentation found AFTER the design was written (read this first)

The section above was written from the state of the tree at 0.4651. Four measurements then changed the
picture, and they are the reusable lessons from this round:

**1. A "model-bound" leg was 40% signal processing.** `AUDIOCPP_PROF` phase timers (features / pre_encode /
encode / AOS) showed the diar leg spending 4.52 s of an 11.1 s budget in the mel frontend against 6.61 s in
the transformer. The cause was `MelFilterbank::compute_custom`: it walked all 128x257 filterbank weights per
frame and accumulated in `long double`, which on ARM64 is **software-emulated IEEE quad precision** - a libgcc
call per term, not an FPU op. The diarizer's bank has 504 nonzero weights out of 32,896. Skipping exact zeros
in the same frequency order is bit-identical (adding +0 to a finite partial sum is exact; the only possible
difference is a sum of exactly zero, where ±0 both feed the same log). Frontend 4.518 s -> 0.249 s, an 18x
improvement on that phase, and the single largest win of the session: **phone_rtf 0.47 -> 0.3142**.

**2. The ASR leg is ~93% ggml compute, and that is now measured, not assumed.** The same treatment on the
x-asr side (host phase timers around the chunk loop) gives, per 69 s clip: compute ~19 s profiled vs 1.3 s of
host work (build 0.196, alloc 0.349, inputs 0.223, outputs 0.551, pos-enc 0.034, fbank 0.000). The encoder is
MUL_MAT-bound on int8 SDOT; the elementwise chain is already overlapped under it. That is why three
bit-identical micro-optimisations of the elementwise path (copy removal, thread-partition threshold, index
arithmetic) cut profiled CPU time by up to 28% and moved the wall clock by 0.0%.

**3. Two pieces of per-step bookkeeping were pure waste, and one of them was a latent use-after-free.**
`build_chunk_graph` rebuilt and re-allocated the whole ~4,000-node graph on every decode step (143 builds for
145 steps on a 69 s clip) - and it returned a graph whose `ggml_context` it had already freed, so the graph
lived in freed memory through alloc/compute/get_output and only survived on allocation luck. Owning the
context and reusing the graph across steps (llama.cpp's pattern for streaming models) is bit-identical and
worth ~3% of the composite, with p95 piece latency *improving* 101 -> 97 ms.

**4. The transducer joint is a fourth cost centre that no profiler was showing.** It is a [5000 x 512] matvec
per encoder frame in plain host C++ - 4.4 GMAC and 3.87 s per 69 s clip, 31% of the ASR leg - and the ggml
eval-callback profile cannot see it at all. Two attacks failed and are documented in the source: four
independent NEON accumulators (3.87 -> 3.84 s, with NEON genuinely enabled) and sharing one pass over the
10.2 MB matrix across frames that share the decoder state (3.87 -> 6.43 s). The matrix is streamed at
~4.5 GB/s with exactly one use per element, so the remaining lever is **bytes, not instructions**: f16 or
q8_0 storage for the joint weights. That moves the logits, so it is a validation-and-re-bless change, not a
byte-identity one.

### The method that actually paid

Both big wins came from the same move: **measure the phases of a "model-bound" leg before theorising about
it.** Three separate times this session, a phase timer or an op/byte histogram found something that no
amount of reasoning about the model would have: 40% of the diar leg in a scalar filterbank, a 4,000-node
graph rebuilt per step, a use-after-free, and a 3.87 s matvec outside ggml entirely. The pattern to carry
forward: for any leg someone calls "compute-bound", split it into (a) host phases around the engine call,
(b) engine-internal phases, and (c) work that happens outside the engine entirely.

### Current state and what is left

Composite phone RTF is **0.307-0.314** (hot row) against a 0.4651-0.47 baseline, byte-identical throughout.
The two files are now merged into one working GGUF (`--models-bundle`, neutral by design). Both legs are
genuinely compute-bound in their kernels. The ranked remainder:

1. **f16/q8_0 joint weights** - est. -6 to -10% composite, numerics-changing, validation path ready.
2. **One-runtime merge** - the single pool is the only overlap form that can pay (the two-pool form lost
   37-47%); ceiling is the 1.6-of-2-cores utilisation; the container merge is already in place to support it.
3. **KleidiAI** - the only untried kernel-level lever, blocked on a network fetch for the `arm_llama` kernels.
4. **ASR cache round-trip** (outputs 0.563 + inputs 0.278 s per clip) - ggml-native double-buffered caches;
   bit-identical in principle, but expect it to be largely hidden under the matmuls like the other host work.


## 9. Addendum: the joint was bandwidth-bound after all, and the harness was lying

Two late findings, both about measurement rather than modelling.

**The transducer joint is now f16, and it is the second-largest win of the session.** §8.4 recorded the joint
as a bounded negative (3.87 s/clip, unattackable by vectorising or by sharing weight passes). That conclusion
was reached with a *stale binary*: the f16 and NEON variants had both failed to compile, and the build script
was swallowing the error. With the build fixed and the variant actually running, storing the matrix as f16
takes the phase from 3.87 s to **1.11 s**, and the whole ASR leg from 47.7 s to 38.6 s per protocol run
(-19%) - more than the joint's own saving, because the 10.2 MB it was streaming per frame had been starving
every other phase on the same two cores. p95 piece latency improved 97 -> 79 ms and first partial 0.306 ->
0.263 s at the same time.

The evidence bar was met without a re-bless: the four gate clips stay byte-identical, and all eleven
validation clips keep the reference WER exactly. Seven of the eleven raw transcripts move, but only in where
a segment splits and which speaker tag it carries - after stripping labels and segment joins the Chinese
clips are character-identical and the English clips differ by one space at a segment boundary. **No
recognised word changes anywhere.** Worth stating explicitly: an f16 weight change that leaves the gate and
the whole validation set at the reference WER is a *free* 2.76 s/clip here, which is not the general rule -
the same class of change (f16 joint weights) on the *encoder* would be a different story.

**The harness bug that made the above invisible.** `scripts/build_android.sh` and `scripts/build_host.sh`
both ran `cmake --build ... >/dev/null` with no exit-status check. A compile error in a dependency therefore
did not fail the build: the previous archive was linked, pushed, measured and reported as though it were the
change. Two experiments in this session produced confident, entirely meaningless numbers that way, and both
looked like clean negative results. Both scripts now fail loudly and print the compiler's errors.

The generalisable rule, and the one this session kept relearning: **if a change produces exactly no timing
difference, suspect the build before believing the physics.** A real optimisation on this hardware moved the
clock by 0% three separate times (#1091 copies, #1092 thread partition, the index-arithmetic fix), and a
broken build looks identical from the outside.


## 10. Why the model-format lever is closed by arithmetic, not by measurement

Five requantisation experiments tied or regressed, and the usual reading was "compute-bound, quantisation
cannot help". That is right but it was an inference. The mechanism is in the ggml type table plus one fact
about the core:

* Every quantised ggml type declares `vec_dot_type = GGML_TYPE_Q8_0` - q4_0, q5_0, q4_K, q6_k, q8_0, all of
  them. So the activation side is quantised to int8 and the dot is an integer MAC. The *weight* format only
  decides how the weights are prepared for that one int8 path; it cannot change which path runs.
* Cortex-A78 is ARMv8.2-A with two 128-bit NEON pipes, so per cycle: SDOT (int8) is ~4x the fp32 FMA rate,
  fp16 FMLA is ~2x, and fp32 is 1x. **The fastest arithmetic this core has is the int8 dot both models are
  already using.**

That closes the lever in the strong sense. Converting the diar encoder from `q5_0` to `f16` - which looked
attractive for 6.6 s/clip of encoder time - would move it from the 4x path to the 2x path and roughly double
the matmul time. The same argument retires the idea of f16 activations on either leg, and it explains every
earlier null result at once: q4_k tying q5_0, all-q8_0 tying, q6_k and iq4_nl regressing, and CrispASR's F16
weights measuring slower (that copy patches F16 to an f32 dot, so it drops all the way to 1x).

What is left on the arithmetic axis is therefore only: fewer FLOPs (a different or smaller model, outside the
contract) or a better int8 GEMM (KleidiAI, blocked on a network fetch). Both encoders are on the right path.


## 11. Correction: the diar encoder's attention is NOT causal, so its state cannot be cached

An earlier version of this note recorded a ~7% opportunity - cache the carried speaker state's per-layer K/V
across windows - on the grounds that the attention mask is causal. **That was wrong, and the opportunity does
not exist.** The mask built in `audiocpp/src/models/nemotron_3_diar/encoder.cpp` is
`[batch, 1, frames, frames]` and masks exactly one region:

```cpp
for (int64_t query = 0; query < frames; ++query)
    for (int64_t key = lengths[batch]; key < frames; ++key)   // keys at/past the VALID LENGTH
        values[(query * frames) + key] = -10000.0F;
```

There is no position-dependent mask anywhere - not in `build_encoder_layer`, not in the attention module,
which receives this tensor and nothing else. So every query attends to **every key in the valid region**:
attention over `[speaker cache | fifo | chunk]` is bidirectional, exactly as Sortformer-with-AOS intends.

Two consequences, and the second is the one that matters:

1. The encoder cost is quadratic in the packed length *for the attention term only*, and the measured
   per-window cost is linear (4.85 / 5.46 / 4.89 ms per frame at 380 / 548 / 351 frames) - so attention is a
   small share and the encoder is dominated by per-frame FFN/conv work.
2. **The carried state's representations depend on the newest chunk frames.** Reusing their K/V from the
   previous window would make those frames attend only to their older context - silently turning bidirectional
   chunk attention into causal attention and changing the model's output. The state frames are irreducibly
   part of every window, and the only way to shrink that term is a smaller `spkcache_len`, which moves speaker
   decisions and has already been taken as far as the evidence allows (264 -> 128, re-blessed).

The mistake was reading a mask that exists as a causal mask. It is worth stating as a rule: **before building a
cache on top of an attention mask, check whether the mask is indexed by POSITION or only by VALIDITY.** A
validity mask says "ignore the padding"; a causal mask says "ignore the future". They look similar in code and
mean opposite things, and only one of them makes incremental decoding legal.
