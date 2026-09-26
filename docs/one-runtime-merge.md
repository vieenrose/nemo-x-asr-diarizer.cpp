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
