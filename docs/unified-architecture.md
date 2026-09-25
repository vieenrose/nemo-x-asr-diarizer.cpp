# Unified architecture: absorbing Nemotron-3 diarization and x-asr into one model

A proposal, built from this project's measurements rather than from the model cards. Every number below was
taken on the phone (taskset C0, armed) or derived arithmetically from those measurements, and the derivations
are shown so they can be checked.

## 0. Two corrections this proposal depends on

* **The x-asr leg runs at 9.0 GMAC/s, not 12.7.** Recomputing properly: 19 layers of (3·k² + k² + 2·k·f) gives
  **51.8 MMAC per 40 ms encoder frame**, i.e. 89.4 GMAC per 69 s clip; at the measured ~9.9 s of encoder time
  that is **9.0 GMAC/s**. The earlier figure assumed a ~5x window amplification, but the six per-layer streaming
  caches mean left context is *not* recomputed. The correction makes the opportunity below **larger**, not
  smaller: 9.0 against a measured plateau of 20.6 is 44%.
* **The diarizer's attachable part is 1.9 MB, not 101.6.** `sortformer_modules.*` (the AOS head) is 1.8 MB and
  `preprocessor.*` is 0.1 MB; the Sortformer-style encoder is 99.7 MB. This matters enormously for what
  "recycling weights" can mean.

## 1. What the measurements say a redesign can and cannot buy

| fact (measured) | consequence for a redesign |
|---|---|
| Both models are Q8_0 and dot through int8 SDOT; the library's plateau on these cores is **20.6 GMAC/s** | A unified model cannot win by "sharing matmul work". There is no spare arithmetic to pool. |
| x-asr's GEMMs run at **n = 24, 12, 6, 3, 6, 12** across its six stages — the widest stage (k=768, ffn 2048) gets the *thinnest* GEMM (3 columns) | This is the single largest structural inefficiency in the system: the most expensive layers get the least efficient shape. It is a *shape* problem created by a 24-frame chunk divided by [1,2,4,8,4,2] time downsampling. |
| The same kernel at n≥24 reaches 20.6 GMAC/s vs 10.4 at n=3 | Fixing the column count is worth up to **~2.3x on the ASR leg's matmul-bound work** — bigger than anything configuration can do. |
| The diar encoder already runs at **n = 351–548** and ~20 GMAC/s, i.e. **at the plateau** | Nothing can make its arithmetic cheaper. The only way to make the diar leg cheaper is to **not run it**, by reusing frames another model already computed. |
| The composite is sequential per piece; `cores_used` 1.81/2; 100% of wall is engine compute | Overlapping the two legs is worth **≤10%**, measured. Not the prize. |
| The two vendored ggmLs produce **byte-identical** results for the whole op vocabulary the encoder uses, including `flash_attn_ext` F32, `gelu_erf`, SplitRoPE, the layout conts, and the model's real Q8_0 weights | One runtime is safe. It is worth ~10%, not more — but it is the *precondition* for everything below. |

So the honest summary: **a unified model is justified by exactly one thing — the ASR encoder's column
starvation — and the only way to fix that in a redesign is to change the time base of the shared trunk.** Not by
fusing attention, not by sharing weights, not by a faster kernel.

## 2. The three tiers, and how much of each model's weight survives

### Tier A — one runtime, both encoders intact (no training, 309 MB recycled as-is)

One ggml context, one thread pool, one arena, one mel pass over the audio, both encoder graphs issued to the
same scheduler. Every weight is recycled unchanged. This is the merge study's conclusion: **≤10%**, and it is
worth doing for consolidation (one kernel set, one bug surface) rather than speed.

### Tier B — ASR trunk, diarization by attachment (~1.9 MB transplanted, 99.7 MB discarded)

This is the tier where **weight recycling genuinely works**, and it is the most interesting proposal here.

The observation: the AOS head consumes 512-dim features on an 80 ms grid. The x-asr trunk *already has* a
512-dim stage (stage 2, and stage 4) and *already has* the audio, already fronted and already encoded. So:

```
16 kHz ─► one mel/fbank pass ─► x-asr encoder (unchanged weights, unchanged 10 ms stack)
                                   │
                                   ├─ ASR head (unchanged): tokens on its 40 ms grid
                                   │
                                   └─ pool pairs of 40 ms frames ─► 80 ms, 512-dim
                                                        │
                                     graft the diar's AOS head here (1.8 MB, initialised
                                     from encoder.layers.0..30's trained values where shapes allow)
```

What is recycled, literally: **the entire ASR** (160 MB, unchanged — it is the accuracy-critical component and
its WER is the thing we must not break) and **the diar's AOS head + preprocessor** (1.9 MB). What is
discarded: the diar's 99.7 MB Sortformer encoder.

Why this is worth doing: the diar leg is **24% of the wall** (14.9 s of 61.4 s) and is already at the
arithmetic plateau, so the *only* way to shrink it is to not run it. The pooled x-asr representation is close to
the right shape and arrives for free as a by-product of work the ASR does anyway. A plausible prize is
**15–20% of the composite**, which is larger than the entire measured overlap prize.

The honest risk: the AOS head was trained on Sortformer features, and speaker-discriminative information may
not be linearly available in a zipformer trunk. **That is an empirical question** — train a linear/MLP adapter on
frozen pooled x-asr features against the diar targets and measure DER. It is a day of work, it reuses the
existing validation harness, and it answers the question before anyone commits to the architecture. If the
adapter fails, tier B dies cheaply.

### Tier C — one coarse trunk, two heads (retrain; the shape fix)

The motivation is the measured 2.3x: a trunk on an **80 ms grid** (the diar's native rate, and 8x coarser than
the ASR's 10 ms frames) would give the deep stages n ≥ 24 instead of 3, moving the ASR leg from 44% of the
plateau toward it. Both heads then read the same trunk: the AOS head natively, and an ASR head with a learned
×2 upsampling back to the 40 ms token grid the decoder expects.

```
16 kHz ─► mel (80 ms frames) ─► single encoder, coarse time axis
                                      ├─ AOS head (80 ms, native)
                                      └─ ASR head: learned upsample ×2 ─► transducer decoder (40 ms)
```

What recycles: the AOS head and the ASR's decoder/embedding as *initialisation*; the two encoders' stacks do
**not** survive, because changing the frame rate changes every convolution and attention geometry they were
trained for. A plausible prize is **composite ~0.20–0.22** (the ASR leg's matmul-bound work up ~2.3x), at the
cost of a full training run and a re-bless of the whole accuracy contract.

**The tension tier C cannot dodge, stated with numbers:** the measured throughput curve is flat from n=3 to
n=12 and only rises past n≈24. At 80 ms per frame, n=24 means **~1.9 s of audio per pass** — so a trunk fat
enough to be efficient emits on a ~2 s cadence, while the current first-partial latency is **0.25 s**. Tier C
therefore has to solve early emission (an intermediate read-out, or a causal cache that lets the ASR head
predict before the pass completes) or it trades the latency contract for the shape win. This is the central
design problem of the whole proposal, and it is visible only because the shape curve was measured.

## 3. Answering "recycle their weights" directly

| what | can it be recycled? |
|------|---------------------|
| Both encoders, unchanged | **Yes**, in tier A only. They are architecturally unrelated — 19 layers at 10 ms with causal chunked attention and KV caches, versus 31 layers at 80 ms with full bidirectional attention over a 27 s window — so no tensor is shape-compatible across the boundary and no cross-loading is meaningful. |
| The diar's AOS head + preprocessor (1.9 MB) | **Yes, and this is the one genuinely transplantable piece**: 512-dim, 80 ms, and the ASR trunk has a 512-dim stage and a 40 ms grid that pools to exactly that. |
| The ASR's transducer decoder, embeddings, joint | **Yes**, as initialisation for a retrained head; the joint's f16 trick and its 1.03 s cost are properties of the runtime, not of the weights. |
| The encoders' *stacks* | **No**, for tier C. A changed frame rate or chunk geometry invalidates every convolution and attention projection. |
| Everything that is not weights | **Yes, and this is where most of the durable value already is**: one ggml (the two trees agree bit-for-bit, verified), one thread pool, graph reuse across steps, the Q8_0/SDOT path, the measured shape profile, the phase timers, the parity harness, the corrected GGUF inventory. None of that is tied to either architecture. |

## 4. What I would do next, in order

1. **Tier B feasibility spike** (a day, no architecture commitment): freeze the x-asr trunk, pool its
   stage-2/4 512-dim output to 80 ms, train a linear-then-MLP adapter against the diar's own per-frame speaker
   targets, and score DER on the bilingual gate. This answers "is speaker information linearly present in a
   zipformer trunk?" — the single question on which tier B stands.
2. **Tier A** if consolidation is wanted independently of speed: it is already scoped, priced at ≤10%, and its
   numerical safety is proven.
3. **Tier C only if** the latency question has an answer. It is a research problem, and the measurements say
   what the answer has to achieve: fat GEMMs need n≥24, which at 80 ms is ~1.9 s per pass, against a 0.25 s
   first-partial budget.

**The one-sentence version:** recycle the ASR wholesale and the diar's 1.9 MB head, discard the diar's
99.7 MB encoder, and do it only if a frozen-trunk adapter can recover speaker separation — because the
measurements say the win is in *not running a second encoder*, not in running two more cheaply.
