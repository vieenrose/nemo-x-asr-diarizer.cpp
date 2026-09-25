// Cross-runtime parity harness: does CrispASR's ggml and audio.cpp's ggml compute the SAME BYTES?
//
// This is stage 1 of docs/one-runtime-merge.md. The one-runtime merge is only worth attempting if the two
// vendored ggml trees agree bit-for-bit, because the port's entire premise is byte-identical output. Porting a
// whole encoder layer to find that out would be expensive and would entangle the answer with the port's own
// mistakes; compiling THIS file against each tree and comparing the raw output bytes isolates the question.
//
// audio.cpp's ggml is hidden inside libaudiocpp.so (ggml symbols are not exported), so the two runtimes cannot
// call each other. Linking this same source against each vendored tree and comparing outputs is the equivalent
// test, and it is stricter: it compares the implementations, not the wrappers.
//
// Built twice:
//   clang++ --target=aarch64-linux-android26 -O2 -mcpu=cortex-a78 -std=c++17 \
//       -I<ref/crispasr/ggml/include> -I<ref/crispasr/ggml/src> tools/ggml_parity.cpp -o /tmp/parity_crisp \
//       -L<ref/crispasr/build-android/ggml/src> -l:libggml.a -l:libggml-cpu.a -l:libggml-base.a -lm
//   (same, with ref/audiocpp/external/ggml)
//
// Single-threaded on purpose: parallel reductions may order their adds differently, and this test is about the
// kernels agreeing, not about the scheduler. Run both, then `cmp` the outputs.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>

// Deterministic bytes: same inputs in both builds, no RNG, no time, no environment.
static void fill_pattern(uint8_t * p, size_t n, uint32_t seed) {
    uint32_t x = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; i++) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        p[i] = (uint8_t)(x >> 24);
    }
}

// The op sequence mirrors what the Nemotron-3 diarizer's encoder actually uses (see encoder.cpp: Linear,
// LayerNorm, Gelu, Sigmoid, Add, Slice, Transpose, SplitRoPE/RoPE, GroupedQueryAttention, plus the q8_0
// matmul that is 52% of the x-asr leg and a large share of the diar leg).
static void run_case(ggml_backend_t backend, ggml_context * ctx_alloc, const char * name,
                     std::vector<uint8_t> & out) {
    const int64_t K = 512, M = 256, Ncols = 6;    // K and M from the models; Ncols deliberately tiny
    out.clear();

    ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * wq  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, M);   // weights, q8_0 as both models ship
    ggml_tensor * wq2 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, M);
    ggml_tensor * wf  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * xq  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, Ncols);
    ggml_tensor * xf  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, Ncols);
    ggml_tensor * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, M);   // a real Linear's bias broadcasts over rows

    // 1. q8_0 x q8_0 matmul - the hot path on both legs
    ggml_tensor * mm = ggml_mul_mat(ctx, wq, xq);
    // 2. f32 x q8_0 (mixed, what a norm feeding a quantised projection looks like)
    ggml_tensor * mmf = ggml_mul_mat(ctx, wq, xf);
    // 3. f32 matmul + bias add ([M] broadcasts over the [M, Ncols] result)
    ggml_tensor * mm2 = ggml_add(ctx, ggml_mul_mat(ctx, wf, xf), bias);
    // 4. norm (LayerNormModule is a LayerNorm with bias in this model)
    ggml_tensor * nrm = ggml_norm(ctx, xf, 1e-5f);
    // 5. elementwise chain: gelu, silu, sigmoid, add
    ggml_tensor * act = ggml_gelu(ctx, xf);
    ggml_tensor * act2 = ggml_silu(ctx, act);
    ggml_tensor * sig = ggml_sigmoid(ctx, xf);
    ggml_tensor * adds = ggml_add(ctx, act2, sig);
    // 6. layout ops the encoder leans on
    ggml_tensor * tr = ggml_transpose(ctx, xf);
    ggml_tensor * ct = ggml_cont(ctx, ggml_transpose(ctx, xf));
    ggml_tensor * cc = ggml_concat(ctx, xf, xf, 1);
    // 7. softmax over the small axis (what attention does before the weighted sum)
    ggml_tensor * sm = ggml_soft_max(ctx, xf);
    // 8. scale and pad
    ggml_tensor * sc = ggml_scale(ctx, xf, 0.125f);          // scalar overload in this ggml
    ggml_tensor * pd = ggml_pad(ctx, xf, 0, 0, 0, 2);        // pads ne0..ne3

    // --- the parts of a real encoder layer that the first pass did NOT cover, and which are the most
    // version-sensitive ops in the whole port. Taken from audiocpp's encoder.cpp / module lowerings:
    //   * GELU ExactErf  -> ggml_gelu_erf, NOT the tanh ggml_gelu tested above
    //   * attention      -> ggml_flash_attn_ext with k and v passed as VIEWS (view_kv=true in
    //                       build_flash_grouped) and precision pinned to F32
    //   * SplitRoPE      -> pure sub/mul/add on the two head halves with cos/sin tables
    //   * the 3D reshape + transpose that turns the fused QKV projection into heads
    const int64_t HD = 32, HEADS = 16, SEQ = 6;
    ggml_tensor * ge = ggml_gelu_erf(ctx, xf);
    ggml_tensor * q4 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HD, SEQ, HEADS, 1);
    ggml_tensor * k4 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HD, SEQ, HEADS, 1);
    ggml_tensor * v4 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HD, SEQ, HEADS, 1);
    ggml_tensor * msk = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, SEQ, SEQ, 1, 1);
    ggml_tensor * flash = ggml_flash_attn_ext(ctx, q4, k4, v4, msk, 1.0f / 32.0f, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
    // SplitRoPE's arithmetic: out = x1*cos - x2*sin for the first half, x2*cos + x1*sin for the second.
    ggml_tensor * x2 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HD / 2, SEQ, HEADS, 1);
    ggml_tensor * x1 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HD / 2, SEQ, HEADS, 1);
    ggml_tensor * cs = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HD / 2, 1, 1, 1);
    ggml_tensor * sn = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HD / 2, 1, 1, 1);
    ggml_tensor * rope_a = ggml_sub(ctx, ggml_mul(ctx, x1, cs), ggml_mul(ctx, x2, sn));
    ggml_tensor * rope_b = ggml_add(ctx, ggml_mul(ctx, x2, cs), ggml_mul(ctx, x1, sn));
    // The QKV -> heads path. TransposeModule({{0,2,1,3},4}) is ggml_permute (a view) followed by the
    // layout fix-up ensure_backend_addressable_layout inserts - the cont is part of the sequence, not an
    // optimisation, so both appear here. Shapes are kept self-consistent (a reshape must preserve the
    // element count or ggml asserts, which is how the first draft of this line was caught).
    ggml_tensor * qkv4 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, K, 1, HEADS, HD);
    ggml_tensor * heads = ggml_cont(ctx, ggml_permute(ctx, qkv4, 0, 2, 1, 3));
    ggml_tensor * resh = ggml_reshape_4d(ctx, ggml_cont(ctx, xf), K, Ncols, 1, 1);   // legal: same count

    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (ggml_tensor * t : {mm, mmf, mm2, nrm, adds, ct, cc, sm, sc, pd, tr,
                            ge, flash, rope_a, rope_b, heads, resh}) ggml_build_forward_expand(gf, t);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "%s: alloc failed\n", name); ggml_free(ctx); return; }
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(galloc, gf);
    ggml_gallocr_free(galloc);

    struct { ggml_tensor * t; uint32_t seed; } feeds[] = {
        {wq, 1}, {wq2, 2}, {wf, 3}, {xq, 4}, {xf, 5}, {bias, 6},
        {q4, 7}, {k4, 8}, {v4, 9}, {msk, 10}, {x1, 11}, {x2, 12}, {cs, 13}, {sn, 14}, {qkv4, 15},
    };
    std::vector<uint8_t> tmp;
    for (auto & f : feeds) {
        tmp.resize(ggml_nbytes(f.t));
        fill_pattern(tmp.data(), tmp.size(), f.seed);
        ggml_backend_tensor_set(f.t, tmp.data(), 0, tmp.size());
    }
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: compute failed\n", name);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return;
    }

    // Dump every distinct output, with a small header naming it, so a mismatch can be localised.
    struct { const char * label; ggml_tensor * t; } outs[] = {
        {"mul_mat_q8xq8", mm}, {"mul_mat_q8xf32", mmf}, {"mul_mat_f32_plus_bias", mm2},
        {"norm", nrm}, {"gelu_silu_sigmoid_add", adds}, {"cont_transpose", ct},
        {"concat", cc}, {"soft_max", sm}, {"scale", sc}, {"pad", pd}, {"transpose", tr},
        // the encoder layer's version-sensitive ops
        {"gelu_erf", ge}, {"flash_attn_ext_f32", flash},
        {"split_rope_a", rope_a}, {"split_rope_b", rope_b}, {"qkv_to_heads", heads}, {"reshape4d", resh},
    };
    uint32_t count = 0;
    for (auto & o : outs) {
        const size_t n = (size_t) ggml_nelements(o.t) * ggml_element_size(o.t);
        out.insert(out.end(), o.label, o.label + strlen(o.label));
        out.push_back(':');
        uint32_t nn = (uint32_t) n;
        for (int b = 0; b < 4; b++) out.push_back((uint8_t)(nn >> (8 * b)));
        out.resize(out.size() + n);
        ggml_backend_tensor_get(o.t, out.data() + out.size() - n, 0, n);
        count++;
    }
    printf("%s: %u tensors, %zu bytes\n", name, count, out.size());
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    (void) ctx_alloc;
}

int main(int argc, char ** argv) {
    const char * tag = argc > 1 ? argv[1] : "unknown";
    const char * path = argc > 2 ? argv[2] : "/data/local/tmp/nemo_x/parity.bin";
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { fprintf(stderr, "no cpu backend\n"); return 1; }
    ggml_backend_cpu_set_n_threads(backend, 1);   // compare kernels, not schedulers

    ggml_init_params ip = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context * scratch = ggml_init(ip);

    std::vector<uint8_t> out;
    run_case(backend, scratch, "q8xq8", out);

    FILE * fh = fopen(path, "wb");
    if (!fh) { fprintf(stderr, "cannot write %s\n", path); return 1; }
    fwrite(out.data(), 1, out.size(), fh);
    fclose(fh);
    printf("%s: wrote %zu bytes to %s\n", tag, out.size(), path);
    ggml_backend_free(backend);
    ggml_free(scratch);
    return 0;
}
