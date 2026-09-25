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
