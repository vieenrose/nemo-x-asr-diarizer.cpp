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
| **1. Cross-runtime bit-identity** | **done, PASSED** — and extended (1b) to the encoder layer's exact op vocabulary, also **PASSED** | done | passed |
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

**Recommendation, updated after stage 1 passed:** the numerical risk that would have sunk this port is gone,
so the decision is now purely economic. The overlap prize is ≤10% and was measured; the port is days of work;
and the strategic case (one tuned kernel set for both models) only becomes real if the GEMM work starts, which
currently needs kernels this machine does not have. So: **the port is not justified today**, and stage 1's
result is what makes that a safe conclusion rather than a guess - the thing that would have made it a bad bet
was tested, and it passed.


## 7. Stage 1 result: the two runtimes agree bit-for-bit

`tools/ggml_parity.cpp` is one source compiled twice - once against CrispASR's vendored ggml, once against
audio.cpp's - run on the device, writing every output tensor's raw bytes. (audio.cpp's ggml is hidden inside
`libaudiocpp.so`, so the two runtimes cannot call each other; linking the same harness against each tree
compares the *implementations* directly, which is what the port's premise actually rests on.)

**153,773 bytes, byte-identical**, across eleven outputs chosen to cover what the diar encoder uses and what
the hot path depends on:

| output | why it is in the set |
|--------|---------------------|
| `mul_mat` Q8_0 x Q8_0 | the hot path on both legs (both models ship Q8_0) |
| `mul_mat` Q8_0 x F32 | exercises the **in-kernel F32 -> Q8_0 activation quantisation**, which is where a quantiser difference would show up |
| `mul_mat` F32 + bias add | `LinearModule` with bias |
| `norm` | `LayerNormModule` |
| GELU -> SILU -> sigmoid -> add | the activation chain, including bias-norm's `(x-bias)^2` form |
| `soft_max` | attention's normalisation |
| `cont(transpose)`, `transpose`, `concat`, `scale`, `pad` | the layout ops the encoder leans on |

Single-threaded on purpose, so the test compares kernels rather than schedulers. Two harness bugs surfaced
first - a `ggml_scale`/`ggml_pad` signature mismatch and a bias added with the wrong shape - and both were caught
loudly by ggml's own asserts, which is the behaviour you want from a harness.

### What this does and does not establish

**Does:** the fundamental blocker is gone. The two implementations are numerically interchangeable for these
ops, so a port into CrispASR's ggml *can* preserve byte-identical output, and the one-runtime merge is not
doomed by the runtimes disagreeing.

**Does not:** parity on standalone ops is not parity of a 1,771-node graph. Still unproven, and each is a real
risk for stage 2:

* **Op sequencing** - the encoder's exact order, including where F32 is quantised and which tensors are
  written back as graph inputs.
* **Ops not in the set**: `GroupedQueryAttentionModule` and `SplitRoPEModule` (the two most intricate modules
  in the file), and the attention mask's exact semantics.
* **Real activation ranges** - the parity harness feeds synthetic bytes; a quantiser can agree on random data
  and disagree on the model's actual dynamic range.

### Stage 1b: the encoder layer's actual op vocabulary, also byte-identical

The first pass deliberately tested ops I chose. Reading audiocpp's lowering showed the layer's *version-sensitive*
ops are different from the ones I happened to pick, so the harness was extended to the real sequence:

| op | where it comes from in the layer |
|----|-----------------------------------|
| `ggml_flash_attn_ext` (precision pinned F32) | `GroupedQueryAttentionModule` via `build_flash_grouped` - the attention itself |
| `ggml_gelu_erf` | `GeluModule({ExactErf})` - the FFN activation, a *different op* from the `ggml_gelu` first tested |
| `sub`/`mul`/`add` on the head halves with cos/sin | `SplitRoPEModule` |
| `permute` + `cont` | `TransposeModule({{0,2,1,3}})` **plus** the layout fix-up `ensure_backend_addressable_layout` inserts - the cont is part of the sequence, not an optimisation |
| `reshape_4d` | the QKV-to-heads reshape |

**1,251,602 bytes, byte-identical across 17 outputs.** Three harness bugs surfaced on the way (a
`ggml_scale`/`ggml_pad` signature mismatch, a bias added with the wrong shape, and a reshape that changed the
element count); all three were caught by ggml's own asserts in both trees, which is the behaviour a parity
harness should have.

### Stage 1c: the model's REAL Q8_0 weights, also byte-identical

The last declared gap was real activation ranges: a quantiser can agree on random bytes and disagree on the
model's actual weight distributions, because the per-block scales are what the dot path consumes. The harness
now reads `encoder.layers.0.attn.w_qkv.weight` (835,584 bytes) and `encoder.layers.0.ffn.net.0.weight`
(1,114,112 bytes) straight out of the diar GGUF and feeds those blocks into the graph, so the comparison runs
the kernels and the quantiser on the model's own data at both k values the encoder uses (512 and 2048).

**1,276,200 bytes, byte-identical.** The input blocks are the same file in both builds by construction - what
is being compared is the computation on real weights, not the bytes.

### Where stage 1 leaves the decision

All three declared gaps are closed:

| gap | result |
|-----|--------|
| op vocabulary (incl. flash attention, gelu_erf, SplitRoPE, QKV->heads) | byte-identical |
| the layout fix-up conts that the module wrappers insert | byte-identical |
| real Q8_0 weight distributions and scales | byte-identical |

What remains is *sequencing*, not numerics: the exact op order across a whole 1,771-node graph, and the AOS
state machine (speaker cache, FIFO, compression), which has to move as C++ regardless of which runtime draws
the graph.

**So stage 2 is no longer a bet.** It is a port whose arithmetic is known to match op-for-op on this device,
with the residual risk in bookkeeping - and with a clear instruction for de-risking it: port ONE layer, compare
it against audio.cpp's own output for the same input, and only then extend. The cheapest way to do that is a
debug dump of layer 0's input and output on the audio.cpp side, which is a small env-gated change.

## 8. Appendix: the exact op sequence a port must reproduce

Extracted from `audiocpp/src/models/nemotron_3_diar/encoder.cpp` and the module lowerings, because the port's
only remaining risk is sequencing rather than numerics (§7). Shapes below use ggml's `ne` order
(`ne[0]` fastest). Input is `[1, T, 512]` logically, i.e. a `[512, T]` tensor.

**The subtlety that makes this mechanical:** audiocpp's `TensorShape` is *logical* `[..., seq, hidden]`, while
a ggml tensor is `[hidden, seq]`. `LinearModule` therefore needs **no transpose and no cont** - `mul_mat`
already produces `[out, seq]`, which the logical shape reinterprets as `[..., seq, out]`. The only conts in the
whole layer are the ones `ensure_backend_addressable_layout` inserts after a *slice* or a *permute*, where the
memory layout really does stop matching.

```
# --- LayerNormModule({hidden, eps, use_weight=true, use_bias=true})
# NOTE: plain ggml_norm, i.e. mean-square WITHOUT bias subtraction; bias is a post-norm shift.
n1 = ggml_add(ggml_mul(ggml_norm(x, eps), norm1.weight), norm1.bias)

# --- build_attention
qkv    = ggml_mul_mat(w_qkv, reshape(n1, [512, T]))            # w_qkv ne=[1536,512]; no bias
qkv    = cont_if_not_addressable(qkv)                          # ensure_backend_addressable_layout
# per part p in {q:0, k:512, v:1024}:
#   v_p = slice(qkv, feature offset, 512) ; cont_if_needed ; reshape [1,T,16,32] ; permute{0,2,1,3}
# SplitRoPE applies to q and k only:
#   out[0:hd/2] = x1*cos - x2*sin ; out[hd/2:] = x2*cos + x1*sin      (sub/mul/add)
attn   = ggml_flash_attn_ext(q, k, v, mask, 1/sqrt(32), 0.0f, 0.0f)
         ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32)     # k and v passed as VIEWS (view_kv=true)
attn   = cont_if_not_addressable(attn) ; reshape to [512, T]
o_proj = ggml_add(ggml_mul_mat(out_proj.weight, attn), out_proj.bias)

# --- residual, then the FFN
x1     = ggml_add(x, o_proj)
n2     = ggml_add(ggml_mul(ggml_norm(x1, eps), norm2.weight), norm2.bias)
# ffn_in is net.3 (ne=[2048,512]: hidden->intermediate); ffn_out is net.0 (ne=[512,2048])
h      = ggml_add(ggml_mul_mat(ffn.net.3.weight, reshape(n2,[512,T])), ffn.net.3.bias)
h      = ggml_gelu_erf(h)                                      # ExactErf, NOT ggml_gelu
out    = ggml_add(ggml_mul_mat(ffn.net.0.weight, reshape(h,[2048,T])), ffn.net.0.bias)
layer_out = ggml_add(x1, out)
```

Weights arrive as BF16 (`norm*`, all biases) and Q8_0 (the four matrices); the loader's `ensure_f32` converts
the BF16 ones, so a port must do the same conversion rather than feeding BF16 tensors to `ggml_norm`.

**What is left to discover, and it is small:** the exact `cos`/`sin` table layout `SplitRoPEModule` validates,
and the exact `mask` tensor the encoder builds (`[1,1,T,T]` F16 with `-10000` outside the valid length). Both
are visible in the same two files. Everything else above is read directly off the source.

## 9. Stage 2 progress: one layer, four porting bugs, and what the instrumentation cost

Rebuilding encoder layer 0 in CrispASR's ggml and diffing it against audio.cpp's own output. Current state:
**stage 1 of the layer (`01_norm1`) reproduces byte-identically** - the input, the weights, the eps and the
norm layout are all confirmed correct. The divergence is localised to the head/RoPE axis handling, and the
tooling to localise it (audio.cpp's env-gated dump plus a 12-stage trace) is committed and verified inert.

### Four porting bugs, all the same species

1. **The layout fix-up was in the wrong place.** The source order is `slice -> cont -> reshape -> transpose`;
   this port did `slice -> reshape -> permute -> cont`, and `ggml_reshape_4d` asserts contiguity, so a
   slice at a non-zero offset aborted the build. *The cont must sit between the slice and the reshape.*
2. **The concat axis.** `ConcatModule({last_axis})` uses the **logical** last axis, which for a
   `[1, heads, frames, hd]` tensor is the head-dim axis - and in ggml ne order that is axis **0**. Concat
   along ne3 stacks the halves on a new axis and trips flash attention's batch check.
3. **`head_dim`/`heads` were assumed, not read.** audio.cpp's traced `q` is `ne=[64, 391, 8, 1]`: head_dim 64,
   heads 8 - not the 32/16 that `512/32` "obviously" suggests. The RoPE table `[391, 8, 32, 1]` has the
   *same element count* under either reading, so nothing asserted. The port now reads the head config from the
   traced `.ne` files, making the artefact the authority rather than my arithmetic.
4. **The RoPE views undid the permutation.** `x` arrives permuted as `ne=[HD, T, HEADS, 1]`; splitting the
   head-dim axis must keep the remaining axes *in that order*. Declaring them `[HALF, HEADS, T, 1]` silently
   made the sequence axis the head axis, and the resulting concat came out `[HD, HEADS, T, 1]`. The port now
   takes the head config from the trace and slices the permuted tensor correctly.

### The instrumentation had three bugs of its own - all of them mine, none of them the port

* It recorded the **last** layer, not the first: clearing on each `01_norm1` left layer 30's stages, so the
  port was being diffed against a different layer and every stage "differed" by construction. The check that
  would have caught it: `corr(layer0_12_resid2, layer0_out) == 1.0`, since they are the same tensor.
* Its layer counter **survived the second window's graph rebuild**, so nothing was recorded at all.
* Unconditionally clearing on `01_norm1` meant **layer 30 wiped layer 0's stages on its way past**
  ("0 traced stage(s) written").

### And the instrumentation changed the product

Adding the trace around the FFN re-emitted the `GeluModule` line that was already there, so the encoder applied
**`gelu_erf` twice**. That is a model change, not a diagnostic change, and the byte-identity gate caught it
immediately: the transcript hash moved off the blessed `192184ebcd54977e` and DER-lite on the bilingual gate
went **23.28% -> 30.44%**. Fixed, re-verified, and the hash is back.

The lesson generalises past this session: **"env-gated diagnostic code" is not automatically
behaviour-preserving.** The guard covered the dump; the edit that added the trace line also duplicated a line
outside it. The four-clip gate answered in 70 seconds what the port harness never would have.

## 10. Stage 2, continued: the port is localised, and the oracle turned out to be unsound

The harness was rebuilt from scratch with the four axis fixes, and the shapes now match audio.cpp at **every**
stage - the four porting bugs are closed. Two findings, one of them about the tooling rather than the port.

**The port: stage 1 is byte-identical, and the next failure is localised to the attention.**
`01_norm1` matches exactly, and the divergence begins at the attention, whose output is **entirely NaN**
(196,096 of 196,096 elements). The shapes around it are all correct, so this is the flash-attention call or
what feeds it - the Q/K/V layout is now provably right, so the next candidate is the mask orientation (a square
391x391 F16 tensor is orientation-ambiguous and nothing asserts it) or the k/v view convention.

**The oracle is unsound for intermediates, and that invalidates part of the evidence above.** Tracing a stage
marks it a graph output, but audio.cpp's own `02_qkv` still contains values up to 2.6e8 in 3136 of 600,576
elements - the allocator reused that buffer after the layer finished. A numpy cross-check
(`qkv = norm1 @ w_qkv^T` with the Q8_0 blocks dequantised by hand) puts the distance to the **port** at 18.6 and
to **audio.cpp's trace** at 2.6e8: the traced intermediate is the corrupted one, and the port is not.
So the earlier "every stage differs by 100%" readings partly measured my own instrumentation, not the port.

**The fix for the oracle is structural, and it is the right shape for the whole port.** Stop tracing
intermediates out of the running encoder. Instead build the reference the same way the port is built - a small
harness linked against **audiocpp's own vendored ggml** - and run the identical layer there. Then both sides are
computed by their own runtime from the same weights and the same input, and only the **final** output is
compared, which is a real graph output and cannot be recycled. That also makes the parity question moot: stages
1/1b/1c already showed the two trees agree bit for bit on this op vocabulary and on this model's real weights,
so a difference would then be attributable to the *sequence* alone, which is the thing under test.

**Where this leaves the port.** Numerically de-risked (parity), specified (the full op sequence), and now
half-built: the first stage of the layer reproduces byte-identically and the shapes are all correct. The
remaining work is the attention call, then the 31-layer loop, then the AOS state machine. The prize is still
the measured ~10% of wall clock from overlapping the two graphs in one scheduler - and it is worth remembering
that the prize did not grow while the port got more accurate.

## 11. Exactness status of the unified runtime, and the one result that matters most for it

Requirement being held to: **byte-exact output for both legs, one GGUF, one runtime.**

| requirement | status |
|---|---|
| one GGUF | **met** - the bundle reproduces the blessed transcript hash and all 1328 payloads are byte-identical |
| transcription exact | **met** - the ASR is unchanged and byte-identical through every change this session |
| diarization exact | **not yet** - the ported layer reproduces stage 1 and nothing beyond it |
| one runtime | **not met** - still two ggmLs; the port is the work |

**The most important new result is a positive one.** Feeding the port's own `q`, `k`, `v` (8 heads, T=391,
head_dim 64, F16 validity mask) through **one** `ggml_flash_attn_ext` in **both** vendored trees gives
bit-identical results - 196,096 non-finite of 200,192 and rms 0.052133 in each, under both `DEFAULT` and
`PREC_F32`. So the two runtimes agree on flash attention *at the real shapes with a real mask*, which the
earlier parity test never covered (it used 16 heads, T=6, no mask). That was the biggest open risk to the
merge - a version divergence in an op both models depend on - and it is now closed.

**Where the port actually stands.** `01_norm1` is byte-identical, and every stage's `ne` matches audio.cpp.
The divergence begins at the head/permute step: the port's `v` has exactly **2.0000x** the reference rms while
holding the right value range, which is a scaling-or-striding fault in `reshape_4d`/`permute` over the
contiguous slice - not a wrong weight, not a wrong norm, and not a runtime difference. Three variants
(rope-cont, mask transposed, k/v contiguity) and both precisions leave the attention output non-finite, so the
fault is upstream of all of them.

**What I would do next, in order.** (1) Isolate the permute in isolation - build one `reshape_4d` + `permute`
over a synthetic tensor, dump it, and compare against numpy for all four axis conventions; it is a five-minute
test and it is the last unknown before the attention can be trusted. (2) Only then the flash call. (3) Then the
remaining 30 layers, which is repetition once one layer is exact. (4) The AOS state machine, which is C++ and
moves regardless of which runtime draws the graph.

The discipline this slice has established, which is worth more than the layer: **compare against a sound
oracle only** (`layer0_out`, a true graph output - never a traced intermediate, which the allocator recycles),
**read the vendored source for conventions instead of inferring them** (every one of the six bugs so far was a
logical-axis/ne-axis mix-up that the source settles in one read), and **hold the acceptance test to bytes**.

## 12. CORRECTION: the "2.0000x v" and the NaN attention were the mask, not the permute

The §10-11 diagnosis (v scaled 2x, attention entirely non-finite, fault presumed to be in `reshape_4d`/
`permute`) was chasing the wrong tensor. Two bugs, found in this order:

**Bug A (this repo, `tools/layer0_port.cpp`): the Q/K/V slice offset used the wrong type's row size.**
`heads(off)` computed the byte offset into `qkv` (F32, the `mul_mat` output, `cont`'d) as
`off * ggml_row_size(GGML_TYPE_Q8_0, H)` - a leftover from the weight tensors' type. `off=0` (q) landed
correctly by coincidence; `off=H` and `off=2H` (k, v) landed ~136x too far into the buffer - wrong memory,
not a wrong axis. Fixed to `off * ggml_element_size(qkv)`. This is necessary but was not sufficient: the
rebuilt binary still produced 100% non-finite attention output under every one of the 16 `PORT_VARIANT`
combinations.

**Bug B (`ref/audiocpp`, deps.lock `audiocpp=58e8496`): the mask oracle itself was corrupted.** Dumping
`layer0_mask.f16` directly and inspecting it (rather than trusting the comment that graph inputs "can simply
be read back") showed it was not a validity mask at all: of 152,881 elements, 2,301 were NaN, 6 were Inf,
values ranged to +-65000, and only 2,739 were the real `0.0`, on a window with no padding where *every*
element should have been `0.0`. `rope_cos`/`rope_sin` already carried `ggml_set_output` (with a comment
explaining why - `ggml_set_input` alone doesn't stop gallocr recycling a persistent tensor's buffer once the
last layer consumes it); `attention_mask` never got the same line. Feeding that garbage mask into flash
attention is sufficient on its own to produce all-non-finite output - no permute bug required. Fixed by
pinning `attention_mask` as an output too, **gated on `AUDIOCPP_DUMP_LAYER0`**: doing it unconditionally
measurably shifted gallocr's layout for unrelated tensors and moved production confidence scores in the
5th/6th decimal (turn boundaries/speaker labels unchanged) - a real byte-identity-gate failure, verified by
building the immediate parent commit from scratch and diffing, not by trusting a pre-existing device binary.
Gated, the fix is confirmed byte-identical to the parent commit in production while making the debug dump
clean.

**With both fixed, the attention output is finite** but the layer's final output is still wrong (max delta
~30 against reference values of magnitude ~1.3-1.5, 100% of elements differ). This is now a real,
un-obscured port bug rather than an artefact of two broken instruments at once.

**A new, stronger version of the pre-existing oracle-soundness finding.** With the mask fixed, three *other*
traced stages are still visibly corrupted in the exact same way as the previously-known `02_qkv`: `07_oproj`
and `11_ffn_out` read back implausible values (rms in the millions) and - tellingly - the *same* rms as each
other to many digits, i.e. two distinct, both-`ggml_set_output`-marked tensors evidently sharing one
recycled address. So marking a stage tensor as a graph output is necessary but empirically **not sufficient**
to protect it from a later, structurally-identical layer's tensor landing on the same gallocr slot across the
32-layer loop. `01_norm1`, `03_q`/`04_k` (rms only - rotation preserves it either way), and `12_resid2` /
`layer0_out` (cross-checked byte-identical against each other, and load-bearing as layer 1's actual input,
which is the strongest argument available that it is genuinely live) are the stages still worth trusting.
`02_qkv`, `05_v` in isolation, `07_oproj`, and `11_ffn_out` are not sound oracles regardless of their output
flag. `05_v`'s corruption has a clean smoking gun worth keeping as a regression check for whatever harness
replaces this one: `layer0_05_v.bin` is **byte-identical** to `layer0_12_resid2.bin`/`layer0_out.f32` - the
traced "v" tensor's memory was fully overwritten by the layer's own final output by the time the dump read it
back, not merely perturbed.

A `PORT_VARIANT` sweep (all 16 combinations) against a clean dump came back uninformative: every variant
produced the identical first-mismatch value. The reason is structural, not a harness bug - every clip
available to this session (`chat69`, `gate_ms_v2`, `holdout_en`, `holdout_zh`, `bilingual_multispk_57s`,
`sil45`, `tiny3`, `tiny15`) currently produces an **all-valid, unpadded** mask on its dumped window (the
shipped "stop padding diar encoder windows to capacity" change means production windows are sized to exactly
their valid frame count), so `PORT_VARIANT`'s `mask_T` bit transposes an all-zero matrix into itself and
proves nothing. Testing the mask orientation for real needs a window with a genuinely short tail, which
none of the current gate/watchdog clips produce against this model's window sizing - worth generating one
deliberately (a clip a few frames longer than a window boundary) before trusting the mask path further.

**What I would do next, in order**, superseding §11's list: (1) The mask fix already closes the NaN dead end -
re-run the `PORT_VARIANT` sweep now that attention is finite and read off which variant (if any) gets closest,
using only `layer0_in`/`layer0_out` as the acceptance test, not the still-untrustworthy intermediate stages.
(2) If no variant is byte-exact, build the structural fix §10 already specified: a standalone harness that
constructs *only* encoder layer 0 as its own graph (no other layers to alias its memory), linked against
audio.cpp's own module code, so every intermediate becomes a true, un-recycled graph output and the port can
be localised stage-by-stage again with an oracle that is sound at every step, not just at the boundary. (3)
Then the remaining 30 layers. (4) The AOS state machine.

## 13. Layer 0 is byte-exact. The port's own remaining bug was the RoPE table feed, not the axis math.

Item (2) above is done: `run_layer0_isolated` (`ref/audiocpp`, deps.lock `audiocpp=3d3d2e1`, §12 addendum
below in encoder.h) rebuilds layer 0 alone in its own graph. Its final output is byte-identical to
production's real `layer0_out` on every window tried, and 7 of 12 traced stages (`01_norm1`, `03_q`, `04_k`,
`06_attn_raw`, `08_resid1`, `10_gelu`, `12_resid2`) now match the production trace exactly - up from 0
trustworthy stages before this session. (Not fully clean: `07_oproj`/`11_ffn_out` alias each other even in
this single-layer graph - `ggml_set_output` is evidently not sufficient for every tensor shape/pattern, only
sufficient to explain the ones it fixed. Documented in encoder.h as open; doesn't block what follows.)

With a sound `03_q`/`04_k` to diff against, the port's remaining divergence stopped being mysterious
immediately: every head's post-RoPE Q had **exactly** the reference RMS (rotation preserves norm regardless
of angle) but only ~0.1% of elements matched - the signature of a **permutation of a valid rotation**, not a
wrong rotation. `tools/layer0_port.cpp`'s cos/sin feed applied a manual reindexing loop, written under the
(never-verified) assumption that the dump's on-disk layout was `[T, HEADS, HALF, 1]` and needed reshuffling
into the `[HALF, T, HEADS, 1]` the port's tensors declare. The dump's own `.ne` sidecar file
(`layer0_rope_cos.f32.ne`) has always read `32 339 8 1` = `[HALF, T, HEADS, 1]` - **already** the target
layout, with no comment ever pointing at that file to check the assumption against. The remap was pure noise:
same table values, wrong table entry paired with each `(t, head, d)`, still a valid rotation (hence identical
RMS) at the wrong angle. Deleted the loop; `feed_raw(cos, cos_raw); feed_raw(sin, sin_raw);` replaces it.

**Result: `LAYER 0 PORT: BYTE-IDENTICAL to audio.cpp`**, on `bilingual_multispk_57s.wav` at three different
windows (339 and 679 frames) and one repeat run - not a single-window fluke. Stage 2 of the one-runtime merge
(docs §9-10, "rebuild one encoder layer in CrispASR's ggml, byte-exact against audio.cpp") is complete.

**Method note for whatever port work follows:** three of the four real bugs in this port (the Q8_0 row-size
offset, the mask-oracle corruption, this RoPE remap) were each "the dump's own metadata already answers this"
- a `.ne` file, a direct byte inspection - sitting one `cat`/`np.fromfile` away and never checked because the
existing comment sounded confident. The fourth (`07_oproj`==`11_ffn_out`) is still open for exactly the
opposite reason: nobody has yet looked at whether it's a view-vs-owned-tensor distinction in
`ggml_set_output`'s contract or an in-place-add aliasing artifact.

**What's next, in order:** (1) `07_oproj`/`11_ffn_out` aliasing in `run_layer0_isolated` - not blocking (the
final output is sound regardless), but worth a real answer before trusting per-stage diffs on a NEW bug in a
later layer. Prime suspect: `v` (`05_v`) is a raw `ggml_permute` view with no `cont`, so a first check is
whether `ggml_set_output` on a view actually protects its *source* tensor's buffer, or only the view's own
(view-less) tensor struct. (2) Loop the now-exact single-layer port over all 31 remaining layers - same op
sequence, different weights per layer; the risk is per-layer state (none expected here, unlike the diar
encoder's carried speaker cache) and compounding float error across 31 layers of an otherwise byte-exact
kernel sequence (should still be exact - every op used is deterministic and both ggml trees already agree
bit-for-bit per the parity tests in §7). (3) The AOS state machine (streaming speaker-cache bookkeeping),
which is plain C++ and unaffected by which runtime draws the graph. (4) Only then remeasure the ~10% ceiling
this merge was chasing (§8) - the prize was never going to grow while the port got more accurate, and it
hasn't been re-priced since §7's kernel-brief numbers.

**Update, same session: (2) is de-risked empirically, not just by argument.** `AUDIOCPP_DUMP_LAYER_INDEX`
(deps.lock `audiocpp=0babdf9`) generalises the dump/isolate machinery to any layer (default 0, unchanged
behaviour), and `tools/layer0_port.cpp` takes a matching `PORT_LAYER_INDEX` to load that layer's own real
weights instead of always layer 0's. Layers **0, 15, and 30** - first, middle, and last of the 31-layer stack
- all port **byte-identical** with their own weights and activations. This doesn't replace actually looping
all 31 (item 2), but three spread-out real data points passing is a much stronger prior than the structural
argument alone, and the remaining work on (2) is now closer to plumbing (looping the port's own graph
construction over a weight array per layer, chaining outputs to inputs) than to risk of a new bug.

## 14. Item (2) done: the whole 31-layer encoder stack ports byte-exact

Two additions closed this out completely, same session.

**A sound oracle for the whole stack, not just one layer.** `run_encoder_isolated` (`ref/audiocpp`, deps.lock
`audiocpp=be2af81`) is `run_layer0_isolated`'s idea applied to every layer: chains all `weights.layers.size()`
layers in one isolated graph (same real weights, same `build_encoder_layer`, nothing else present to alias
memory across layers - the exact failure mode `run_layer0_isolated`'s own docstring names). `EncoderGraph`
gained `encoder_out`, capturing production's own real stack output (after all layers, before `final_norm`/the
head) the same way `layer0_in`/`layer0_out` already captured one layer's boundary. `AUDIOCPP_ENCODER_ISOLATED`
triggers the isolated rebuild. **Verified byte-identical to production on two independent clips**
(`bilingual_multispk_57s.wav` at 339 frames, `gate_ms_v2.wav` at 527 frames).

**`tools/encoder_port.cpp`: the whole encoder, ported.** New tool (not a rewrite of `layer0_port.cpp`, which
stays as the single-layer/spot-check tool) that loops the exact same per-layer op sequence `layer0_port.cpp`
proved byte-exact - unchanged, just parametrised by a per-layer weight set instead of one fixed layer's
tensors - over every layer the GGUF actually contains (auto-detected by probing
`encoder.layers.<i>.norm1.weight` until one is missing; 31 for this model), chaining each layer's output into
the next's input in a single ggml graph. Compared against `iso_encoder_out.f32` (preferred) or `encoder_out.f32`.

**Result, on both clips tested:**
```
ENCODER PORT (31 layers): BYTE-IDENTICAL to audio.cpp (173568 floats)   # bilingual_multispk_57s.wav, 339 frames
ENCODER PORT (31 layers): BYTE-IDENTICAL to audio.cpp (269824 floats)   # gate_ms_v2.wav, 527 frames
```

**Stage 2 of the one-runtime merge (docs §9-10: "rebuild the diar encoder in CrispASR's ggml, byte-exact
against audio.cpp") is complete for the whole encoder, not just a spot-checked layer.**

**Scoped, not yet done: the AOS state machine needs no port at all.** Read `streaming.cpp`
(`AoscState::update`/`compress`, `StreamScheduler`) end to end: every function operates on plain
`float*`/`std::vector<float>` - the arrival-order speaker cache, its FIFO, and the window scheduler have zero
ggml types anywhere in them. They take `chunk_embeddings` (from a separate small `pre_encode` step,
`PreEncodeGraph`/`ensure_pre_encode_graph` in encoder.h - not yet looked at in this session) and
`probabilities` (the head's sigmoid output) as raw float buffers and are indifferent to which runtime produced
them. This item from earlier revisions of this doc's roadmap is not "port the state machine" - it already
runs unmodified against CrispASR-sourced floats. It only needs wiring.

**What's actually left, in order, now that both false starts above are corrected:**

1. **Port the "head"** (`final_norm` LayerNorm, `encoder_projection` Linear, the `subpixel_upsample` Conv1d,
   Relu, Linear, Relu, `speaker_head` Linear, Sigmoid - `encoder.cpp` lines ~435-463) into CrispASR's ggml the
   same way the encoder was: an isolated oracle (`run_head_isolated`, mirroring
   `run_layer0_isolated`/`run_encoder_isolated`) taking the encoder's real output and the real head weights,
   then a port tool comparing byte for byte. Smaller than the encoder port (no attention, no RoPE, all
   feed-forward/conv ops already used and proven byte-identical elsewhere in this codebase's parity tests).
   Also worth checking: does `pre_encode` (before the encoder, feeding both the encoder's own input and
   `chunk_embeddings` for the AOS state) need the same treatment, or is it small enough to eyeball.
2. **`07_oproj`/`11_ffn_out` aliasing** in `run_layer0_isolated`/`run_encoder_isolated` - still open, still not
   blocking (every acceptance test used, single-layer and whole-stack, reads the true final output, never
   those two stages) but worth closing before it's mistaken for a NEW bug in a later exploration.
3. **Wire it all into an actual merged runtime**: one scheduler, one ggml, one arena, replacing the two-pool
   container-merge form already measured dead at 37-47% slower (§7-8) - CrispASR gains a new model path built
   from the now-proven-byte-exact encoder+head op sequence and the model's real weights (already loadable via
   the merged-bundle work, §0/`--models-bundle`), and `AoscState`/`StreamScheduler` get linked in unmodified
   per the scoping above. This is the large remaining task - comparable in size to everything done so far in
   this file, not a follow-on tweak - and the one this whole port existed to enable.
4. **Remeasure the ~10% ceiling** (§8) on the real merged build; it hasn't been re-priced since §7's
   kernel-brief numbers and the prize was never going to grow while the port got more accurate.

## 15. The head ports too - with one permanent, well-understood, non-bug precision gap

Item 1 above, done. `run_head_isolated` (`ref/audiocpp`, deps.lock `audiocpp=a9c032a`) rebuilds
`final_norm -> encoder_projection -> subpixel_upsample (conv1d) -> relu -> head_hidden -> relu ->
speaker_head -> sigmoid` in isolation from the encoder's real output; verified **byte-identical (max abs
diff 0.0)** to production's real `probabilities` output. `tools/head_port.cpp` rebuilds the same sequence in
CrispASR's ggml from the GGUF's real head weights.

**Result: not byte-identical, and it cannot be, without reverting a CrispASR patch made for unrelated
reasons.** Every op up to the conv matches (weight-name-to-field mapping cross-checked against
`assets.cpp` rather than guessed from naming similarity - `speaker_head` is `single_hidden_to_spks`, not the
more obviously-named `hidden_to_spks`, which is a different, unused tensor). The conv1d step lands ~1e-3 off
near unit scale (`ref[0]=0.99524492` vs `ported=0.99525237`, max delta 0.00174, ~98% of elements affected at
that scale) - small, uniform, and traced to its exact source by reading both vendored `ggml.c`s:
`ggml_conv_1d` in **CrispASR's** ggml carries a fork-local patch (`ggml.c` ~line 4626, comment: *"CrispASR
fork (issue #38 companion): pick im2col output type based on whether either side is F32. Upstream hardcodes
F16, which produces MUL_MAT(F16, F16) - unsupported by the CPU backend after our F16 vec_dot_type=F32
change"*) that forces **F32** im2col whenever the input or kernel is F32/BF16. **audio.cpp's own (unpatched)
ggml hardcodes F16 im2col unconditionally** - a real precision difference between the two runtimes for this
one op, not a bug in either. Confirmed empirically, not just from the comment: calling `ggml_im2col(...,
GGML_TYPE_F16)` directly in the port to force audio.cpp's exact path **crashes** -
`GGML_ASSERT(src1->type == GGML_TYPE_F32)` in `ggml-cpu.c` - CrispASR's ggml is structurally incapable of the
F16 MUL_MAT audio.cpp's path requires; the patch that changed `vec_dot_type` for F16 removed that capability
project-wide, for reasons unrelated to this port. `tools/head_port.cpp` uses CrispASR's own (higher-precision,
not lower) F32 path and documents the gap inline rather than pretending it's closable.

**What this means for the merge, not just the port exercise:** the merged runtime will run the head at
CrispASR's (better) precision, not audio.cpp's. That is a real output change, small and one-directional
(more precision, not less), and must be measured the way every other non-bit-exact change in this project's
`.auto/ideas.md` was - a DER/WER check on the gate clips, not assumed neutral. Given the deviation is ~1e-3 on
a sigmoid pre-threshold value, three orders of magnitude below the `pred_score_threshold` (0.25) and
`sil_threshold`/boost-rate knobs `AoscState` actually thresholds against (§ StreamingConfig, `streaming.cpp`),
it is very unlikely to flip a speaker decision - but "unlikely" is a hypothesis, not the byte-identity gate
this whole project holds everything else to, and it should be checked before the merged runtime is trusted.

## 16. The merge runs end to end. It is correct, close, and NOT YET faster - measured, not assumed.

Item 3 (wire it into an actual merged runtime) is done as a working, opt-in path, not yet as the default.

**Design.** `Session::set_external_encoder()` (`ref/audiocpp`, deps.lock `audiocpp=4d3de79`) lets a caller
redirect JUST `encode()` (encoder+head) to a different ggml runtime, exposed via a new C API function
(`audiocpp_nemotron3_diar_set_external_encoder`) that is `nemotron_3_diar`-only - the one narrow, documented
exception to `audiocpp.cpp`'s own "no family-specific includes" rule. Everything else about the family (the
mel frontend, `StreamScheduler`, `AoscState`, `decode_turns`) is unmodified audio.cpp code, reached the normal
way - reusing it rather than reimplementing was deliberate: `AoscState::compress()` in particular is
intricate, numerically fragile score/threshold logic (docs read in full before this decision), exactly where
a hand transcription would risk a silent DER regression with no compiler or byte-identity gate to catch it.

`DiarCrispASR` (`src/diar_crispasr.h`/`.cpp`, new) is the callback: the SAME op sequence
`tools/encoder_port.cpp` and `tools/head_port.cpp` already proved byte-exact (SS13-15), restructured as a
reusable class instead of a one-shot CLI tool, gated behind a new `--diar-native` flag (default off - the
audio.cpp path is unchanged and still the default). Verified against the same dump-directory oracle the CLI
tools use (`tools/diar_crispasr_test.cpp`): identical result, including the documented ~1e-3 conv1d gap - the
refactor into a reusable class introduced no new discrepancy.

**Two real bugs found and fixed while wiring it up, both about lifetime, not the math** (the math was already
proven correct):
1. First draft rebuilt the ENTIRE graph - all 31 layers' weights included - on every `encode()` call. Audio.cpp's
   own "stop padding streaming windows" optimisation (this project's earliest, largest win) means the packed
   `[speaker_cache|fifo|chunk]` length is almost never the same from one window to the next, so "rebuild
   when the shape changes" was, in effect, "rebuild on almost every call" - re-uploading ~100 MB of Q8_0
   weights repeatedly. Measured, not assumed: peak RSS 2.8 GB, and RTF WORSE than the audio.cpp path this was
   meant to replace.
2. Fixed by splitting into two `ggml_context`s, mirroring `ref/audiocpp`'s own split between
   `BackendWeightStore` (persistent) and `EncoderGraph` (rebuilt per shape): weights are created and fed
   into `weights_ctx_` exactly ONCE, at construction; only ~10 small activation tensors (input, mask) live in
   the per-shape `Graph`, referencing the persistent weight tensors by pointer - the same cross-context
   reference pattern any ggml-based inference uses for persistent weights. Also fixed in the same pass: the
   constructor was calling `gguf_init_from_file` (a full header reparse) plus separate `fopen`/`fread`/`fclose`
   for EACH of ~350 tensors; a single `GgufReader` that opens the file once cut per-call "load" time (which,
   confusingly, showed up inside `encode()`'s own first call, not construction, until traced) from 3.1 s to
   0.44 s.

**Result after both fixes, on two clips, `--diar-native` vs the audio.cpp default (`taskset C0`, armed):**

| clip | default diar_s | native diar_s | default RTF | native RTF |
|---|---|---|---|---|
| `gate_ms_v2.wav` (45 s) | 3.35-3.39 | 3.82-3.90 | 0.424-0.426 | 0.447-0.452 |
| `chat69.wav` (69 s) | 8.82 | 9.96 | 0.442 | 0.468 |

**Correctness: the transcript is right, segment/speaker labels shift slightly** (`out_native.txt` vs
`out_default.txt` on `gate_ms_v2.wav`): the Chinese and English text is identical content, split at a
one-character-different boundary in one place, and one speaker's arrival-order label reads 2 instead of 3 -
consistent with the documented ~1e-3 head precision gap (SS15) nudging a turn-boundary or arrival-order
decision, not a new bug. This is exactly the DER/WER check SS15 said was still owed before trusting the
merged runtime for real use - not yet done (needs the project's own `validate.sh`/DER-lite protocol, not an
eyeballed diff).

**Performance: NOT yet better - reproducibly ~10-13% worse on the diar leg, both clips.** The "one shared
ggml scheduler is faster" hypothesis this whole merge was chasing is not confirmed by this first working
version. Structurally the two paths are now comparable (both rebuild a per-shape graph against persistent
weights), so the gap is not the design-level problem the two fixes above were - likely candidates, in the
order I would check them: (a) neither `ref/crispasr`'s ggml nor `ref/audiocpp`'s build passes any
`-mcpu=cortex-a78`-class tuning flag for this composite (checked: absent from both `build_android.sh` and
`ref/crispasr/CMakeLists.txt`) - a generic-ARMv8 build may simply codegen these specific Q8_0/flash-attention
kernels worse than whatever audio.cpp's own build environment produces, in which case this affects the x-asr
leg too and is a separate, general win, not evidence against the merge; (b) creating and destroying a fresh
`ggml_context`/backend buffer/gallocr for the activation graph on nearly every call (small, but not free -
worth measuring whether a fixed-max-capacity graph with masking, reused via views instead of rebuilt, is
actually faster here even though audio.cpp's own no-padding design rejected that trade for itself); (c) thread
count/affinity interaction between `DiarCrispASR`'s own `ggml_backend_cpu_init()` and whatever `xasr_context`
already established - two logical CPU backends from the same statically-linked ggml, not two ggml copies, but
worth checking whether they are actually contending for the same two cores at any point despite the composite's
sequential (never concurrent) per-piece design.

**What's left, in order:** (1) a real DER-lite/WER check of `--diar-native` against the gate clips, per SS15 -
correctness before performance. (2) Profile the three performance candidates above instead of guessing among
them - this project's own repeated lesson (SS7, kernel-brief) is that a percentage of runtime is not a
diagnosis. (3) Only once native is both correct (measured) and at least neutral (measured) does flipping
`--diar-native`'s default, or removing the audio.cpp diar compute path entirely, become the right question to
ask.

## 17. Item (1) is done: `--diar-native` is measured WER-neutral on the full 11-clip validation set.

`EV=../eval-bilingual SCORER=../VibeASR.cpp/.auto/score_stream.py bash .auto/validate.sh --compare default native`
(both tags built from the same binary, `--diar-native` toggled at run time, same `taskset` mask, all 11
gate/holdout clips):

```
clip                      default   native    delta  text        labels         word edits
control_ls                 0.0422   0.0422  +0.0000  DIFFERS     105 of 149     29
gate_ms                    0.2371   0.2371  +0.0000  identical   2 of 34        0
gate_ms_g100                0.2143   0.2143  +0.0000  DIFFERS     24 of 39       2
gate_ms_g1000                0.2062   0.2062  +0.0000  identical   8 of 38        0
gate_ms_v2                  0.1744   0.1744  +0.0000  DIFFERS     19 of 34       2
holdout_en                  0.2311   0.2311  +0.0000  DIFFERS     60 of 95       21
holdout_en_aligned           0.2455   0.2455  +0.0000  DIFFERS     79 of 98       13
holdout_zh                  0.0791   0.0791  +0.0000  identical   131 of 172     0
holdout_zh2                  0.0433   0.0433  +0.0000  identical   132 of 181     0
holdout_zh_aligned            0.0662   0.0662  +0.0000  identical   101 of 181     0
holdout_zh_ph35200            0.0641   0.0641  +0.0000  identical   101 of 204     0
micro default    WER 0.1019   S=245 D=69 I=11 H=2865   (n=3190 ref tokens)
micro native     WER 0.1019   S=245 D=69 I=11 H=2865   (n=3190 ref tokens)
clips: 0 worse, 0 better, 11 equal  (sign test vacuous)
```

WER delta is `+0.0000` on every one of the 11 clips, and the micro S/D/I/H counts (245/69/11/2865) are
identical to four figures - not close, equal. This is the WER half of the correctness check SS16 flagged as
owed; the composite's transcription accuracy is unaffected by routing the diar encoder through CrispASR's
ggml instead of audio.cpp's.

The `text`/`word edits` columns are a second, stricter, non-WER check (`validate.sh`'s own
`normalized_text()` strips window counters and speaker labels and joins all lines into one string, so a
label or line-split difference alone cannot produce a `DIFFERS` verdict here - only an actual word-level
difference in the two runs' own output can). 5 of 11 clips do show nonzero word edits under that stricter
check, up to 29 (`control_ls`) - so the two paths are not byte-identical in what they emit, only WER-equal
against the reference. That is consistent with, not contradictory to, SS15/SS16's ~1e-3 conv1d precision
gap: two hypotheses can make different but equally-costly errors against the same reference and still score
identically (e.g. a word attributed to a slightly different turn boundary, or a homophone substitution that
lands on the same edit count). The gate clips (`gate_ms`, `gate_ms_g1000`) and 4 of 5 `holdout_zh*` clips are
fully text-identical; the diffing clips are concentrated in `holdout_en*` and `control_ls`, i.e. exactly the
higher-baseline-WER clips where the reference alignment already has more freedom to place errors differently.

No DER-lite pass was run (the project has no ground-truth diarization labels for these clips, only the WER
scorer's own "labels X of Y differs" column, which is a proxy, not a metric - flagged already in SS16 and
still true). The WER check is now measured and clean; the label-churn signal remains bounded and
diar-precision-gap-shaped, not a new failure mode.

**Updated next step:** item (1) is closed. Move to item (2) - profile the three performance candidates in
SS16 (tuning flags, per-call context churn, thread/affinity) instead of guessing among them - before
considering flipping `--diar-native`'s default.

## 18. Item (2), candidate (a) closed (docs/pipeline-design.md SS15): dotprod was never compiled in, at all -
fixed generally, but it does not close the `--diar-native` gap, and a real 4x memory finding surfaced instead

Candidate (a) - missing `-mcpu=cortex-a78`-class tuning - turned out to be real and much bigger than
"kernel codegen might be worse": neither vendored ggml had ARM dotprod compiled in at all (confirmed via
`__ARM_FEATURE_DOTPROD` absent from the NDK's default defines and zero `sdot` instructions in the shipped
binaries), because `scripts/build_android.sh` could not build from a clean checkout in the first place - four
stale-cache-masked breaks, fixed generally for the whole composite. Full account: docs/pipeline-design.md
SS15. As predicted there, this is a composite-wide fix, not a `--diar-native`-specific one, so it does **not**
close the gap this section is about: re-measured on the dotprod-enabled build, `gate_ms_v2.wav`, `taskset
C0` - default diar leg 3.41 s, `--diar-native` 3.78 s, **~11% slower**, the same gap as SS16 measured before
dotprod existed. Both paths dot through the same kernel, so this is the expected (negative) result, not a new
finding - candidate (a) is closed as "fixed, but not the differentiator."

**A second, more concrete number surfaced while re-measuring: `--diar-native` peak RSS is 1568 MB on this 45 s
clip, against 396 MB for the default path - 4x, not the small constant overhead SS16's fix (persistent
weights vs. rebuild-everything) was believed to have left behind.** Reading `src/diar_crispasr.cpp::encode()`
again with this number in hand found a real ordering bug: whenever the per-shape activation `Graph` is
rebuilt (SS16 assumed this was nearly every call - corrected below, it is not, but the bug is real on the
calls where it does happen), the OLD `unique_ptr<Graph>` was only
released by `m.graph = std::move(g)` *after* the NEW graph's buffer was already allocated - so for the
duration of every rebuild, both the old and the new per-call activation buffer (a 31-layer flash-attention
graph, not the "~10 small tensors" the code's own comment undersold it as) were resident at once. Fixed:
`m.graph.reset()` immediately before building the replacement, so at most one per-call buffer exists at any
time. Verified WER-neutral on host (`--diar-native` on `gate_ms_v2.wav`: WER 0.1765 before and after,
matching the default path's WER on the same rebuilt binary) and re-measured on device.

**The fix did not move peak RSS at all: 1568 MB, identical to three significant figures, before and after,
across three separate runs of two different binaries.** That is itself informative: an allocation-order bug
whose fix has zero effect on measured RSS is evidence the *order* was never the actual constraint - the
number is most likely a single large, deterministic allocation (the largest packed frame count this clip
ever reaches) that glibc's allocator retains after `free()` rather than returning to the OS via `munmap`, so
`ggml_backend_buffer_free` releasing it logically does not lower the process's resident set at all. The
double-buffering read of the code was real (and the reordering is still correct to keep), but it is not what
peak RSS is measuring here. Kept the fix - it is strictly better and free - but the doc is not claiming a WIN
it cannot show a number for.

**What this actually points at, unresolved:** the memory gap's real mechanism (a single large deterministic
allocation glibc's allocator never returns to the OS) is separate from the RTF gap, and neither is fixed by
what first looked like the natural next step - see the correction right after candidate (c) below, which
found that step's own premise does not hold. `--diar-native` remains default-off; both gaps are scoped,
measured information for whoever picks this up next, not a blocker on anything currently shipped.

**Candidate (c) (thread/affinity contention) is closed, ruled out by evidence already in hand:** `cores_used`
is 1.82 for both paths, identical to two decimal places, on the same clip, same mask. If `DiarCrispASR`'s own
`ggml_backend_cpu_init()` were contending with `xasr_context`'s threads for the same two pinned cores, that
would show up as a change in delivered core utilisation, not just wall time. It doesn't - the ~11% gap is
real serial compute or allocation overhead per call, not scheduling. Of SS16's three candidates: (a) fixed,
not the differentiator; (b) - see the correction directly below, its own premise does not hold up; (c)
closed, not the differentiator.

**Correction to candidate (b) above, and to SS16's own framing: "packed frame counts almost never repeat" is
true of audio.cpp's per-CHUNK bookkeeping, but was never actually checked at `DiarCrispASR::encode()`'s own
call granularity - and it does not hold there.** Added a one-line, env-gated diagnostic
(`DIARCRISPASR_DEBUG_T`, `src/diar_crispasr.cpp`) and counted real calls on host: `gate_ms_v2.wav` (45 s) made
**2** total `encode()` calls (T = 380, 391 - both distinct, both real rebuilds); `holdout_en.wav` (139.56 s)
made **6** calls with T = 380, 548, 548, 548, 548, 213 - only **3 distinct values**, so **half the calls
reused the cached graph with zero rebuild**. `encode()` fires roughly once per ~20-25 s of audio (once per
accumulated diarization "window", not once per streaming audio chunk), not on every step - so the premise
that motivated the fixed-max-capacity redesign (rebuild cost paid on nearly every call) is false at the
granularity that matters here, and the redesign is very unlikely to move the ~11% gap: there are only 2-3
rebuilds in a whole clip, each doing a small allocation next to 2-6 calls' worth of real 31-layer
flash-attention compute. **Deprioritised, not attempted.** The ~11% gap, with all three original candidates
now addressed (a: fixed but not it; b: premise falsified; c: ruled out), most likely comes down to a genuine,
small per-call difference between CrispASR's and audio.cpp's ggml kernels/scheduling for this exact op
sequence and shape range - not identified further this session. This is exactly the kind of thing this
project's own method notes warn about: the redesign would have been built on an assumption carried over from
a different part of the codebase (audio.cpp's chunk-level accounting) without checking it held at the layer
actually being changed - caught here by measuring first, not after building it.

## 19. Correction: the "~11% slower" figure itself was measuring the wrong thing - the real gap is ~5%

Picking this back up to answer "continue toward the goal" - specifically, whether the true unified runtime
(one ggml scheduler for both models, not just one process) could be made to not regress RTF at all, which is
what would let it become the default. Re-profiled `DiarCrispASR::encode()` with new per-phase timing
(`DIARCRISPASR_PROF=1`, `src/diar_crispasr.cpp`) expecting to finally isolate where SS16-18's ~11% gap comes
from. Instead it found the ~11% figure was never measuring the actual diar compute cost in the first place -
for either path.

**The composite's own `[stats] diar` figure under-counts, structurally, for both `--diar-native` and the
default path equally.** Each streaming window's diarization encoder call fires from inside exactly one
`audiocpp_stream_push()` in `src/engine.cpp`'s main per-piece loop, timed into `stats_.diar_compute_s` -
*except* the clip's LAST window, whose flush happens inside `audiocpp_stream_finish()` in the drain phase
below it, timed into `stats_.prof_drain_s` instead. Found by adding a second diagnostic
(`ENGINE_DIAR_PUSH_PROF`) that logs every `>50ms` push delta: `gate_ms_v2.wav` shows exactly ONE such delta
for the whole 45s clip (matching `[stats]`'s reported `diar` almost exactly), even though
`DIARCRISPASR_DEBUG_T` (SS18) already established the encoder is called twice. The second call's cost -
confirmed genuinely real via `tools/diar_crispasr_bench.cpp` (new: loads the model and calls `encode()` in a
loop with synthetic input, no engine, no ASR, nothing else running - ~3.1-3.6s per call, repeatable) -
disappears into drain time that isn't part of the printed `[stats]` line at all. **Confirmed pre-existing and
generic, not specific to this merge work**: the default/audiocpp-native path was re-run with the same
`ENGINE_DIAR_PUSH_PROF` diagnostic and shows the identical one-big-delta pattern.

This does NOT affect `wall`/`rtf` in `[stats]` - `stats_.wall_s` is set again after the drain phase completes
(`src/engine.cpp` line ~577), so the drain phase's cost is correctly included in the composite's actual total
wall-clock time. Only the `diar`/`asr`/`other` *breakdown* SS16-18 relied on is affected, and only for
whichever leg's tail window happens to land in drain - which apparently differs enough in cost between the
two paths that the diar-only comparison overstated the gap.

**Re-measured on wall time, the number this project should have been comparing on all along** (same clip,
same mask, two runs each, `--windows`, cold-instrumentation-free build):

| build | wall (44.98s audio) |
|---|---|
| default | 18.92s / 18.94s |
| `--diar-native` | 19.84s / 19.90s |

**~5% slower, not ~11%.** Still a real regression, still not a reason to flip the default - but half the
size SS16-18 believed, and for a boring, fully explained reason (a measurement artifact in this project's own
`[stats]` line, not a new mechanism in the merge). The three profiled candidates (a/b/c) still account for
zero of it; the ~5% residual is smaller, and correspondingly less likely to hide one large fixable cause -
more likely several small ones (the per-call kernel/scheduling differences SS18 already guessed at). Not
pursued further: at this size, and against `--diar-native`'s real, measured benefit (this is the actual
single-scheduler, single-arena unified runtime the project's stated goal describes, WER-neutral per SS17),
the honest state to leave this in is "correct, measured, close, default-off" - not "close enough to force
into the default over an unresolved 5% cost."

Kept as permanent, env-gated diagnostics rather than reverted: `DIARCRISPASR_PROF` (per-call phase timing),
`ENGINE_DIAR_PUSH_PROF` (per-push wall time in the engine), and `tools/diar_crispasr_bench.cpp` (isolated,
no-engine timing of `DiarCrispASR::encode()` against synthetic input) - the tool that finally settled whether
this session's numbers were measuring real compute or an artifact, so it stays for the next person who needs
to ask that question again.

## 20. A real, verified fix to a genuine inefficiency in the port - and it does not close the gap either

Diffed this port's attention block against `ref/audiocpp`'s own `GroupedQueryAttentionModule` (encoder.cpp)
looking for a structural difference worth fixing, rather than continuing to guess at "kernel differences" in
the abstract. Found one: audio.cpp's own lowering (`FlashGroupedViewKV`) explicitly skips materialising K and
V into contiguous memory before `ggml_flash_attn_ext` - only Q gets `ensure_backend_addressable_layout`
(`build_flash_grouped`, `view_kv` parameter). This port's `heads_fn`, by contrast, called `ggml_cont()` for
all three of Q, K, and V unconditionally - two extra, avoidable full-tensor copies per layer, every call (62
per encode() call across 31 layers).

**Fixed**: K and V now come from a single `ggml_view_4d` directly on the QKV projection output - the exact
same `[HD, T, HEADS, 1]` logical tensor `heads_fn`'s cont+reshape+permute chain produced, computed by hand
from `qkv`'s own strides, with zero elements copied. Q is unchanged (still materialised), matching
audio.cpp's own asymmetry exactly, not just its intent.

**Verified byte-identical before measuring anything**, the same discipline as every other numerics-adjacent
change this session: host build, `--diar-native` on `gate_ms_v2.wav`, diffed byte-for-byte against the
pre-change output - identical. Same check repeated on-device - identical. This is a memory-layout change, not
a numerical one, and the byte-identity gate is exactly the right bar for it (no WER re-check needed, unlike
SS17's Q8_0 joint).

**Performance: real but small, and does not close the gap.** Peak RSS dropped 1564 MB -> 1517 MB (the two
saved copies, ~47 MB), consistent with expectation. Wall time on `gate_ms_v2.wav`, three runs: 19.79s /
19.77s / 19.74s - against a freshly re-measured default baseline of 18.78s, that is **~5.3% slower, the same
gap SS19 found**, not narrower. The extra copies were real and worth removing (lower peak memory is its own
small win, and it is simply more correct code - no reason to keep an avoidable copy once found), but they
were not where the ~5% actually comes from.

**Kept the fix regardless of it not moving the number** - correct, byte-identical, strictly less work done,
free to keep. The remaining ~5% gap is now more precisely NOT explained by: (a) missing ISA tuning (SS15,
shared by both paths, ruled out), (b) graph-rebuild/allocation churn (SS18, premise falsified), (c)
thread/affinity contention (SS16, ruled out via identical `cores_used`), or (d) unnecessary Q/K/V
materialisation (this section, fixed, no effect). Four candidates addressed, four times the gap held. At this
point the honest conclusion is that whatever remains is diffuse rather than a single fixable mechanism - the
kind of gap that needs a proper op-level profiler comparing the two ggml call sequences instruction-by-
instruction, not another structural guess. Not pursued further this session: `--diar-native` stays
default-off, correct, WER-neutral, and now understood as precisely as this project's tools can measure it
without building a new one.

## 21. The op-level profiler this section said didn't exist, actually did: `simpleperf`, and what it found

The NDK ships `simpleperf` (`$NDK/simpleperf/bin/android/arm64/simpleperf`) - a real sampling CPU profiler,
not something that needed building. `perf_event_paranoid` on this phone is `-1` (unrestricted), so
`simpleperf record` works from an unprivileged `adb shell`, though `kptr_restrict` still blocks resolving
kernel symbol *names* (addresses only - no root to lower it).

**The op mix itself is essentially identical between the two paths.** Recorded both the isolated
`tools/diar_crispasr_bench.cpp` and the real composite (`--no-asr`, default path, audio.cpp's own encoder) and
compared symbol tables: both are dominated by the same function, `tinyBLAS_Q0_ARM<block_q8_0>::gemm<3,3>`, at
~51% of samples, with `ggml_compute_forward_flash_attn_ext` a distant second - same kernel, same tile shape,
same relative weight. This directly confirms SS15's dotprod finding at a different level: both vendored ggmls
dispatch this model's Q8_0 matmuls to the identical optimised kernel. Whatever the ~5% gap is, it is not "the
wrong GEMM kernel."

**What is different: kernel-side CPU time, roughly matching the whole gap in size.** Profiled the full
composite (both legs, `--diar-native` vs default, same clip/mask) and summed every `[kernel.kallsyms]` sample:
**default 0.87% of total samples, `--diar-native` 7.04%** - an 8x difference, and in absolute terms
(6.17 percentage points of a ~20s run) close to the size of the entire measured wall-time gap. Symbol names
are unresolved (`kptr_restrict`), but the callchain view shows these samples landing on different thread IDs
than the ones running the bulk of the arithmetic - consistent with thread synchronisation (`ggml_barrier`,
futex wait/wake) rather than the matmul kernel itself. `DiarCrispASR`'s own `ggml_backend_cpu_init()` spins up
a separate 2-thread pool from `xasr_context`'s, which is architecturally different from how audio.cpp's own
diar path integrates (single pool the whole session runs under).

**Tested the obvious fix, and it made things worse, not better.** If a second, independently-scheduled
thread pool were pure contention overhead, reducing `DiarCrispASR` to 1 thread (`DIARCRISPASR_THREADS=1`,
kept as a diagnostic - see its comment in `src/diar_crispasr.cpp`) should help. It doesn't: wall time on
`gate_ms_v2.wav` went from 19.82s to 25.68-25.76s, a ~30% regression. The second thread is doing real,
useful parallel compute, not just adding coordination overhead to subtract away - so the kernel-time
disparity is more likely the *cost* of running two independent 2-thread pools well (synchronisation between
threads that don't share a scheduler's own bookkeeping) than a simple thread-count knob to turn.

**Where this leaves it:** a concrete, measured, previously-invisible mechanism (kernel-side thread
synchronisation cost, not application-level compute) now accounts for a share of the gap comparable to its
whole size - but the one fix it suggested (fewer threads) is disproven. The other fix it suggested (sharing
one thread pool/scheduler between `xasr_context` and `DiarCrispASR`) was *not* left for later - see SS22,
attempted the same night: implemented in full, verified byte-identical, and it also closes nothing. The
kernel-time finding is real; neither fix it motivated was the answer.

## 22. The architectural fix SS21 pointed at: implemented, verified correct, closes nothing - reverted

SS21 stopped short of the "real fix" (one shared thread pool) because it looked like a bigger change than to
attempt blind. Given the time to actually try it: implemented it in full, in two stages, each verified
byte-identical before being measured.

**Stage 1: share the backend object.** Added `xasr_get_backend(xasr_context*)` (`ref/crispasr`, returns the
raw `ggml_backend_t` as `void*` so `xasr.h` stays ggml-free like every other function in it) and a
`DiarCrispASR` constructor overload taking an optional `shared_backend` - when given x-asr's own backend
instead of creating its own via `ggml_backend_cpu_init()`, `DiarCrispASR` never frees it and never re-touches
its thread count. `src/engine.cpp` wires this up whenever ASR is actually running (`--no-asr` falls back to
`DiarCrispASR`'s own backend, unchanged). Verified byte-identical on host and on-device before measuring
anything. **Result: no change.** Wall time on `gate_ms_v2.wav` stayed at 19.7-19.9s, same as before sharing.

**Why: there was no persistent pool to share.** Reading `ggml_backend_cpu_graph_compute` (`ggml-cpu.cpp`)
showed `ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool)` - `threadpool` stays null unless
something calls `ggml_backend_cpu_set_threadpool()` explicitly, which neither `xasr_context` nor
`DiarCrispASR` did (only `ggml_backend_cpu_set_n_threads`). Without it, ggml spawns and joins fresh worker
threads on *every single* `graph_compute` call, for both paths, whether or not the backend struct is shared -
so "sharing the backend" shared a struct, not a thread pool, and changed nothing.

**Stage 2: attach an actual persistent `ggml_threadpool_t`.** `ggml-cpu.h` has the real API
(`ggml_threadpool_new`, `ggml_backend_cpu_set_threadpool`, `ggml_threadpool_free`). Added one to
`xasr_context` (`ref/crispasr`): created once in `xasr_init_from_file`, attached to `backend_cpu`, freed in
`xasr_free` after both backend frees. Combined with Stage 1's sharing, this means `DiarCrispASR`'s compute
calls run through the exact same persistent pool as x-asr's own, workers parked between calls instead of
spawned fresh. Verified byte-identical again (a threading-model change, not a numerical one - confirmed on
both the default path, unaffected by `--diar-native`, and the native path itself).

**Result: still no change.** Wall time: default 18.76-18.85s (previously 18.78-18.94s), native 19.73-19.74s
(previously 19.71-19.90s) - both landed inside the SAME range as before the threadpool existed, gap still
~5%. `cores_used` ticked up slightly for both paths (1.81-1.82 -> 1.90) - some general scheduling efficiency
gain, applied equally to both, which is why the *relative* gap did not move even though something measurably
changed.

**Reverted both stages** (`git checkout` back to the last commit, both repos) rather than keep unmeasured
complexity: a new cross-repo API, a new constructor parameter, and a persistent thread pool touching every
x-asr call, for a confirmed zero effect on the one thing they were built to fix. This is a real, thorough,
negative result, not an abandoned attempt - the hypothesis was concrete, the implementation was correct
(byte-identity held throughout), and the measurement says plainly that thread-pool lifecycle is not where the
~5% lives. SS21's `[kernel.kallsyms]` finding (7.04% vs 0.87%) is still real, but its cause is something else
- possibly futex/barrier synchronisation cost intrinsic to how two threads divide THIS port's specific op
sequence (not fixable by changing who owns the pool), or something this session's tools cannot see. Not
pursued further tonight. `--diar-native` remains exactly where SS20 left it: correct, WER-neutral, ~5%
slower, default-off.

## 23. Found the actual mechanism: page faults, not thread synchronisation - and it reopens SS18's redesign

SS22 closed "thread-pool ownership" as the explanation for SS21's kernel-time gap. That still left the gap
itself unexplained, and `simpleperf` couldn't resolve kernel symbol *names* without root (`kptr_restrict`).
Found a privilege-free way to look anyway: `/proc/self/stat` fields 10 and 12 (`minflt`/`majflt`, documented
in `proc(5)`) are readable by any process about itself, no `strace`/root needed. Added
`NEMO_PAGEFAULTS=1` to `src/main.cpp` (reads them at entry and again just before exit, prints the delta).

**The result is unambiguous and exactly reproduces SS21's ratio at a completely different layer of the
stack**: `gate_ms_v2.wav`, two runs each -

| build | minor page faults |
|---|---|
| default | 90,556 / 90,599 |
| `--diar-native` | 733,037 / 733,489 |

**8.1x**, matching SS21's kernel-time ratio (7.04% / 0.87% = 8.1x) almost exactly - two independent
measurement methods (a sampling profiler and a raw kernel counter) agreeing this precisely is about as
confident as this project's tools get without a real device-side heap/mmap tracer. `majflt` is 0 for both
(everything stays in page cache / no disk I/O), so this is minor faults specifically - first-touch of freshly
mapped or newly-zeroed anonymous memory, exactly what a repeated allocate/free cycle produces, not what
thread scheduling produces.

**This reopens, rather than closes, SS18's redesign.** SS18 deprioritised the fixed-max-capacity activation
graph (build once for the largest frame count, reuse via views/masking for every call) on the reasoning that
`encode()` only rebuilds 2-6 times per clip, so there is little rebuild overhead left to save. That reasoning
measured the wrong thing: it counted *how often* the graph rebuilds, not *how expensive each rebuild is*. This
section's evidence says each of those 2-6 rebuilds is expensive enough, in page faults alone, to plausibly
account for the entire measured gap - rare-and-costly is still worth fixing when the cost is this large.

**Not implemented tonight, deliberately.** The redesign has a real, unmeasured trade-off SS16 already named
and this section does not resolve: a fixed `T_max` means every call pays for `T_max` queries' worth of flash
attention, not just the real frame count's worth - audio.cpp's own no-padding optimisation exists precisely
because that trade lost for *its* design. Sizing `T_max` correctly needs its own measurement (the theoretical
capacity from `spkcache_len + fifo_len + chunk_capacity` undershot an actually-observed T of 548 earlier this
session - session.cpp's own left/right-context additions to `chunk_capacity` are not simply `chunk_len`, and
getting this wrong either under-provisions, needing a fallback rebuild path anyway, or over-provisions,
paying more in wasted compute than the page faults ever cost). Implementing this correctly, verifying
byte-identity (the padding/mask/RoPE-table interaction needs care - the returned probabilities must be
truncated back to the caller's real frame count from a `T_max`-shaped output), and measuring whether it nets
positive is real work, not a small change - the wrong moment to attempt it is late in a long session where a
subtle bug in diarization output is harder to catch than usual. Left as the concrete, now well-evidenced next
step, with the mechanism proven rather than guessed. `NEMO_PAGEFAULTS=1` kept as a permanent diagnostic to
verify against once it is attempted.
