// One encoder layer of the Nemotron-3 diarizer, rebuilt in CrispASR's ggml, diffed against audio.cpp's own.
//
// Stage 2, first slice, of docs/one-runtime-merge.md. audio.cpp can dump its own layer 0 with
// AUDIOCPP_DUMP_LAYER0=<dir> (and trace its twelve stages with AUDIOCPP_TRACE_LAYER0=1); this program reads
// that dump plus the real layer-0 weights and rebuilds the layer, then compares stage by stage.
//
// The arithmetic is not the risk - parity tests already showed the two runtimes agree bit for bit on every op
// here, including on this model's real Q8_0 weights. The risk is the SEQUENCE, and specifically the gap
// between audio.cpp's LOGICAL tensor shapes [.., seq, hidden] and ggml's ne order [hidden, seq]. Four bugs came
// from that gap and each is commented where it bit:
//
//   1. the layout fix-up (ggml_cont) belongs BETWEEN the slice and the reshape - ggml_reshape_4d asserts
//      contiguity and a slice at a non-zero offset is a strided view;
//   2. ConcatModule({last_axis}) concatenates on the LOGICAL last axis, which is ggml ne axis 0;
//   3. head_dim is 64 and heads is 8 - not the 32/16 that hidden/32 suggests. Read from the traced .ne, never
//      assumed: the RoPE table has the same element count under either reading, so nothing asserts;
//   4. the RoPE halves are views of the PERMUTED tensor, so they must keep its axis order [HD, T, HEADS, 1];
//      declaring [HALF, HEADS, T, 1] silently makes the head axis the sequence axis.
//
// Usage: layer0_port <dump-dir> <diar.gguf> [LAYER0_STAGE_DIR=<dir>]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<uint8_t> slurp(const std::string & path, bool * ok = nullptr) {
    std::vector<uint8_t> data;
    FILE * fh = std::fopen(path.c_str(), "rb");
    if (fh == nullptr) { if (ok) *ok = false; return data; }
    fseek(fh, 0, SEEK_END);
    const long n = ftell(fh);
    fseek(fh, 0, SEEK_SET);
    data.resize((size_t) n);
    const bool good = n > 0 && fread(data.data(), 1, (size_t) n, fh) == (size_t) n;
    fclose(fh);
    if (ok) *ok = good;
    return data;
}

static bool gguf_tensor(const std::string & path, const char * name, std::vector<uint8_t> & out,
                        int64_t ne[4]) {
    struct ggml_context * ctx = nullptr;
    struct gguf_context * g = gguf_init_from_file(path.c_str(), (struct gguf_init_params){ true, &ctx });
    if (g == nullptr) return false;
    bool found = false;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
        if (strcmp(gguf_get_tensor_name(g, i), name) != 0) continue;
        const size_t n = gguf_get_tensor_size(g, i);
        out.resize(n);
        FILE * fh = fopen(path.c_str(), "rb");
        if (fh != nullptr) {
            fseek(fh, (long) (gguf_get_data_offset(g) + gguf_get_tensor_offset(g, i)), SEEK_SET);
            found = fread(out.data(), 1, n, fh) == n;
            fclose(fh);
        }
        const struct ggml_tensor * t = ggml_get_tensor(ctx, name);
        for (int d = 0; d < 4; d++) ne[d] = t ? t->ne[d] : 0;
        break;
    }
    gguf_free(g);
    if (ctx) ggml_free(ctx);
    return found;
}

// audio.cpp stores norms and biases as BF16 and its loader widens them to F32, so a port must too.
static std::vector<float> bf16_to_f32(const std::vector<uint8_t> & raw) {
    std::vector<float> out(raw.size() / 2);
    for (size_t i = 0; i < out.size(); i++) {
        const uint16_t bits = (uint16_t) ((uint16_t) raw[2 * i] | ((uint16_t) raw[2 * i + 1] << 8));
        uint32_t u = ((uint32_t) bits) << 16;
        float f;
        memcpy(&f, &u, 4);
        out[i] = f;
    }
    return out;
}

static void feed_raw(ggml_tensor * t, const std::vector<uint8_t> & b) { ggml_backend_tensor_set(t, b.data(), 0, b.size()); }
static void feed_f32(ggml_tensor * t, const std::vector<float> & v) {
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        printf("usage: %s <dump-dir> <diar.gguf> [LAYER0_STAGE_DIR=<dir>]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    const std::string model = argv[2];
    const char * stage_dir = std::getenv("LAYER0_STAGE_DIR");
    // Variants for the conventions that remain unproven. Each is scored against layer0_out - a true graph
    // output, and therefore the only sound oracle: the traced intermediates are not (audio.cpp's own 02_qkv was
    // found holding 2.6e8 values, because the allocator recycles an intermediate's buffer once the layer is
    // done with it). Bit 0: cont after the RoPE concat. Bit 1: mask transposed. Bit 2: k/v cont before flash.
    int variant = 0;
    if (const char * v = std::getenv("PORT_VARIANT")) variant = atoi(v);
    printf("variant %d (rope_cont=%d mask_T=%d kv_cont=%d f32prec=%d)\n", variant,
           variant & 1, (variant >> 1) & 1, (variant >> 2) & 1, (variant & 8) ? 0 : 1);

    bool ok = false;
    const std::vector<uint8_t> in_raw = slurp(dir + "/layer0_in.f32", &ok);
    if (!ok) { printf("missing layer0_in.f32\n"); return 1; }
    const std::vector<uint8_t> ref_raw = slurp(dir + "/layer0_out.f32", &ok);
    if (!ok) { printf("missing layer0_out.f32\n"); return 1; }
    const std::vector<uint8_t> mask_raw = slurp(dir + "/layer0_mask.f16", &ok);
    const std::vector<uint8_t> cos_raw = slurp(dir + "/layer0_rope_cos.f32", &ok);
    const std::vector<uint8_t> sin_raw = slurp(dir + "/layer0_rope_sin.f32", &ok);
    long hidden = 0, frames = 0;
    if (FILE * sh = std::fopen((dir + "/layer0_in.shape").c_str(), "rb")) {
        if (fscanf(sh, "%ld %ld", &hidden, &frames) != 2) { fclose(sh); return 1; }
        fclose(sh);
    }
    const int64_t H = hidden, T = frames;

    // (3) head config from the trace, not from hidden/32.
    int64_t HD = 64, HEADS = 0;
    if (FILE * qne = std::fopen((dir + "/layer0_03_q.bin.ne").c_str(), "rb")) {
        long a0 = 0, a1 = 0, a2 = 0;
        if (fscanf(qne, "%ld %ld %ld", &a0, &a1, &a2) == 3 && a0 > 0) { HD = a0; HEADS = a2; }
        fclose(qne);
    }
    if (HEADS == 0) HEADS = H / HD;
    const int64_t HALF = HD / 2;
    printf("hidden=%lld frames=%lld head_dim=%lld heads=%lld (head config read from the trace)\n",
           (long long) H, (long long) T, (long long) HD, (long long) HEADS);

    struct Loaded { std::vector<uint8_t> raw; std::vector<float> f32; int64_t ne[4]; bool bf16; };
    const char * q8_names[] = { "encoder.layers.0.attn.w_qkv.weight",
                                "encoder.layers.0.attn.out_proj.weight",
                                "encoder.layers.0.ffn.net.0.weight",   // ffn_in : hidden -> inter
                                "encoder.layers.0.ffn.net.3.weight" }; // ffn_out: inter  -> hidden
    const char * f1_names[] = { "encoder.layers.0.norm1.weight", "encoder.layers.0.norm1.bias",
                                "encoder.layers.0.norm2.weight", "encoder.layers.0.norm2.bias",
                                "encoder.layers.0.attn.out_proj.bias",
                                "encoder.layers.0.ffn.net.0.bias", "encoder.layers.0.ffn.net.3.bias" };
    Loaded W[11];
    for (int i = 0; i < 4; i++) {
        if (!gguf_tensor(model, q8_names[i], W[i].raw, W[i].ne)) { printf("missing %s\n", q8_names[i]); return 1; }
        W[i].bf16 = false;
    }
    for (int i = 0; i < 7; i++) {
        if (!gguf_tensor(model, f1_names[i], W[4 + i].raw, W[4 + i].ne)) { printf("missing %s\n", f1_names[i]); return 1; }
        W[4 + i].bf16 = true;
        W[4 + i].f32 = bf16_to_f32(W[4 + i].raw);
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 2);
    ggml_init_params ip = { ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(512, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    const float eps = 1e-5f;
    ggml_tensor * in   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, T, T, 1, 1);
    // (5) the RoPE table must broadcast against x1, which is [HALF, T, HEADS, 1]; the dump is [T, HEADS, HALF, 1].
    ggml_tensor * cos = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HALF, T, HEADS, 1);
    ggml_tensor * sin = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HALF, T, HEADS, 1);

    ggml_tensor * w_qkv    = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, W[0].ne[0], W[0].ne[1]);
    ggml_tensor * w_out    = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, W[1].ne[0], W[1].ne[1]);
    ggml_tensor * w_ffn_in = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, W[2].ne[0], W[2].ne[1]);
    ggml_tensor * w_ffn_out= ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, W[3].ne[0], W[3].ne[1]);
    ggml_tensor * n1w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, W[4].ne[0]);
    ggml_tensor * n1b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, W[5].ne[0]);
    ggml_tensor * n2w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, W[6].ne[0]);
    ggml_tensor * n2b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, W[7].ne[0]);
    ggml_tensor * ob  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, W[8].ne[0]);
    ggml_tensor * fib = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, W[9].ne[0]);
    ggml_tensor * fob = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, W[10].ne[0]);

    // --- the layer, in the source's order -----------------------------------
    ggml_tensor * n1 = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, in, eps), n1w), n1b);
    ggml_tensor * qkv_raw = ggml_mul_mat(ctx, w_qkv, n1);
    ggml_tensor * qkv = ggml_cont(ctx, qkv_raw);                      // ensure_backend_addressable_layout

    // (1) cont between slice and reshape; (3)+(4) the permuted head layout.
    const auto heads = [&](int64_t off) {
        ggml_tensor * sl = ggml_view_2d(ctx, qkv, H, T, qkv->nb[1], (size_t) off * ggml_row_size(GGML_TYPE_Q8_0, H));
        return ggml_permute(ctx, ggml_reshape_4d(ctx, ggml_cont(ctx, sl), HD, HEADS, T, 1), 0, 2, 1, 3);
    };
    ggml_tensor * q0 = heads(0), * k0 = heads(H), * v = heads(2 * H);

    const auto rope = [&](ggml_tensor * x) {
        // (4) x is PERMUTED, ne=[HD, T, HEADS, 1]: split ne0 and keep the rest in that order.
        ggml_tensor * x1 = ggml_view_4d(ctx, x, HALF, T, HEADS, 1, x->nb[1], x->nb[2], x->nb[3], 0);
        ggml_tensor * x2 = ggml_view_4d(ctx, x, HALF, T, HEADS, 1, x->nb[1], x->nb[2], x->nb[3], HALF * x->nb[0]);
        ggml_tensor * c = ggml_repeat(ctx, cos, x1), * s = ggml_repeat(ctx, sin, x1);
        // (2) the concat is on the LOGICAL last axis = ggml ne axis 0. No cont: audio.cpp's SplitRoPE output
        // keeps the permuted ne [HD, T, HEADS, 1], and conting collapses the axes to [HD, HEADS, T, 1].
        ggml_tensor * r = ggml_concat(ctx, ggml_sub(ctx, ggml_mul(ctx, x1, c), ggml_mul(ctx, x2, s)),
                                     ggml_add(ctx, ggml_mul(ctx, x2, c), ggml_mul(ctx, x1, s)), 0);
        return (variant & 1) ? ggml_cont(ctx, r) : r;
    };
    ggml_tensor * q = rope(q0), * k = rope(k0);

    if (variant & 4) { k = ggml_cont(ctx, k); v = ggml_cont(ctx, v); }
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f / std::sqrt((float) HD), 0.0f, 0.0f);
    // audio.cpp pins GGML_PREC_F32 here. These two vendored ggml trees are DIFFERENT versions, and if
    // CrispASR's does not implement the F32 flash path on CPU it produces NaN rather than failing loudly - which
    // is exactly the symptom. Bit 3 drops the pin so the default path can be tested.
    if (!(variant & 8)) ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    ggml_tensor * attn_raw = attn;
    attn = ggml_reshape_2d(ctx, ggml_cont(ctx, attn), H, T);
    ggml_tensor * o = ggml_add(ctx, ggml_mul_mat(ctx, w_out, attn), ob);

    ggml_tensor * x1 = ggml_add(ctx, in, o);
    ggml_tensor * n2 = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x1, eps), n2w), n2b);
    ggml_tensor * ff_in  = ggml_add(ctx, ggml_mul_mat(ctx, w_ffn_in, n2), fib);
    ggml_tensor * ff_gel = ggml_gelu_erf(ctx, ff_in);
    ggml_tensor * ff_out = ggml_add(ctx, ggml_mul_mat(ctx, w_ffn_out, ff_gel), fob);
    ggml_tensor * out = ggml_add(ctx, x1, ff_out);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) { printf("alloc failed\n"); return 1; }
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_gallocr_free(ga);

    // --- feed --------------------------------------------------------------
    feed_raw(in, in_raw);
    if (variant & 2) {   // a square T x T F16 mask is orientation-ambiguous and nothing asserts it
        const size_t n = mask_raw.size() / sizeof(uint16_t);
        std::vector<uint16_t> t(n);
        for (int64_t r = 0; r < T; r++)
            for (int64_t c = 0; c < T; c++)
                t[r * T + c] = (uint16_t) (((const uint16_t *) mask_raw.data())[c * T + r]);
        feed_raw(mask, std::vector<uint8_t>((const uint8_t *) t.data(),
                                           (const uint8_t *) t.data() + t.size() * sizeof(uint16_t)));
    } else {
        feed_raw(mask, mask_raw);
    }
    {
        const size_t n = cos_raw.size() / sizeof(float);
        std::vector<float> c(n), s(n), rc(n), rs(n);
        memcpy(rc.data(), cos_raw.data(), n * sizeof(float));
        memcpy(rs.data(), sin_raw.data(), n * sizeof(float));
        for (int64_t t = 0; t < T; t++)
            for (int64_t hI = 0; hI < HEADS; hI++)
                for (int64_t d = 0; d < HALF; d++) {
                    c[(d * T + t) * HEADS + hI] = rc[(t * HEADS + hI) * HALF + d];
                    s[(d * T + t) * HEADS + hI] = rs[(t * HEADS + hI) * HALF + d];
                }
        feed_f32(cos, c);
        feed_f32(sin, s);
    }
    feed_raw(w_qkv, W[0].raw);     feed_raw(w_out, W[1].raw);
    feed_raw(w_ffn_in, W[2].raw);  feed_raw(w_ffn_out, W[3].raw);
    ggml_tensor * f1t[7] = { n1w, n1b, n2w, n2b, ob, fib, fob };
    for (int i = 0; i < 7; i++) feed_f32(f1t[i], W[4 + i].f32);

    // Mark the tensors of interest as graph outputs so the allocator keeps their buffers alive: reading a
    // dead intermediate after the graph has run is unsound (ggml recycles it - that is how audio.cpp's traced
    // 02_qkv came to hold 2.6e8 values). ggml_set_output extends liveness in a gallocr-allocated graph, so the
    // stage dump below is now a real observation rather than a guess.
    for (ggml_tensor * t : { qkv_raw, q, k, v, attn_raw, o }) ggml_set_output(t);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { printf("compute failed\n"); return 1; }


    // --- stage dump + verdict ---------------------------------------------
    const auto stage = [&](const char * name, ggml_tensor * t) {
        if (stage_dir == nullptr) return;
        const std::string f = std::string(stage_dir) + "/port_" + name + ".bin";
        const size_t n = ggml_nbytes(t);
        std::vector<uint8_t> b(n);
        ggml_backend_tensor_get(t, b.data(), 0, n);
        if (FILE * fh = std::fopen(f.c_str(), "wb")) { std::fwrite(b.data(), 1, n, fh); std::fclose(fh); }
        if (FILE * nh = std::fopen((f + ".ne").c_str(), "wb")) {
            for (int64_t d = 0; d < GGML_MAX_DIMS; ++d)
                std::fprintf(nh, "%lld%c", (long long) t->ne[d], d + 1 == GGML_MAX_DIMS ? '\n' : ' ');
            std::fclose(nh);
        }
    };
    stage("01_norm1", n1);   stage("02_qkv", qkv_raw); stage("03_q", q);      stage("04_k", k);
    stage("05_v", v);       stage("06_attn_raw", attn_raw); stage("07_oproj", o);
    stage("08_resid1", x1); stage("09_ffn_in", ff_in); stage("10_gelu", ff_gel);
    stage("11_ffn_out", ff_out); stage("12_resid2", out);

    std::vector<float> got((size_t) ggml_nelements(out));
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));
    const std::vector<float> ref((const float *) ref_raw.data(), (const float *) ref_raw.data() + ref_raw.size() / 4);
    if (ref.size() != got.size()) { printf("SIZE MISMATCH %zu vs %zu\n", ref.size(), got.size()); return 1; }
    size_t diff = 0, first = 0;
    double worst = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
        if (memcmp(&ref[i], &got[i], sizeof(float)) != 0) {
            if (diff == 0) first = i;
            diff++;
            worst = std::max(worst, (double) fabs(ref[i] - got[i]));
        }
    }
    if (diff == 0) {
        printf("LAYER 0 PORT: BYTE-IDENTICAL to audio.cpp (%zu floats)\n", ref.size());
    } else {
        printf("LAYER 0 PORT: %zu of %zu floats differ (%.4f%%), first at %zu, max |delta| %.6g\n",
               diff, ref.size(), 100.0 * diff / ref.size(), first, worst);
        printf("  ref[%.0f]=%.9g ported=%.9g\n", (double) first, ref[first], got[first]);
    }
    return diff == 0 ? 0 : 1;
}
