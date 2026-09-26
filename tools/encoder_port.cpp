// The WHOLE Nemotron-3 diarizer encoder stack (all layers), rebuilt in CrispASR's ggml, diffed against
// audio.cpp's own. Stage 2 completion of docs/one-runtime-merge.md: tools/layer0_port.cpp proved the
// per-layer op sequence byte-exact on layers 0, 15, and 30 individually; this loops that same sequence over
// every real layer's weights, chaining outputs to inputs in one graph, and compares against
// AUDIOCPP_ENCODER_ISOLATED's encoder_out.f32 / iso_encoder_out.f32 (run_encoder_isolated in ref/audiocpp) -
// the whole-stack analogue of layer0_port.cpp's layer0_out.f32.
//
// Usage: encoder_port <dump-dir> <diar.gguf> [ENCODER_STAGE_DIR=<dir>]
//   <dump-dir> must contain layer0_in.f32 (+ .shape), layer0_mask.f16, layer0_rope_cos/sin.f32 (from
//   AUDIOCPP_DUMP_LAYER0 with AUDIOCPP_DUMP_LAYER_INDEX=0, so layer0_in is the embed-norm output - the whole
//   stack's real input) and encoder_out.f32 (the whole stack's real output, dumped unconditionally alongside
//   layer0_*) or iso_encoder_out.f32 (AUDIOCPP_ENCODER_ISOLATED's sound oracle - prefer this one if present).

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

static bool gguf_tensor(const std::string & path, const char * name, std::vector<uint8_t> & out, int64_t ne[4]) {
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

static bool gguf_has_tensor(const std::string & path, const char * name) {
    struct ggml_context * ctx = nullptr;
    struct gguf_context * g = gguf_init_from_file(path.c_str(), (struct gguf_init_params){ true, &ctx });
    if (g == nullptr) return false;
    bool found = false;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
        if (strcmp(gguf_get_tensor_name(g, i), name) == 0) { found = true; break; }
    }
    gguf_free(g);
    if (ctx) ggml_free(ctx);
    return found;
}

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
static void feed_f32(ggml_tensor * t, const std::vector<float> & v) { ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float)); }

struct LayerWeights {
    std::vector<uint8_t> w_qkv, w_out, w_ffn_in, w_ffn_out;   // Q8_0, raw bytes
    int64_t ne_qkv[4], ne_out[4], ne_ffn_in[4], ne_ffn_out[4];
    std::vector<float> n1w, n1b, n2w, n2b, ob, fib, fob;      // BF16 widened to F32
};

static bool load_layer(const std::string & model, int layer_index, LayerWeights & w) {
    const std::string lp = "encoder.layers." + std::to_string(layer_index) + ".";
    if (!gguf_tensor(model, (lp + "attn.w_qkv.weight").c_str(), w.w_qkv, w.ne_qkv)) return false;
    if (!gguf_tensor(model, (lp + "attn.out_proj.weight").c_str(), w.w_out, w.ne_out)) return false;
    if (!gguf_tensor(model, (lp + "ffn.net.0.weight").c_str(), w.w_ffn_in, w.ne_ffn_in)) return false;
    if (!gguf_tensor(model, (lp + "ffn.net.3.weight").c_str(), w.w_ffn_out, w.ne_ffn_out)) return false;
    std::vector<uint8_t> raw; int64_t ne[4];
    auto bf16 = [&](const char * suffix, std::vector<float> & out) {
        if (!gguf_tensor(model, (lp + suffix).c_str(), raw, ne)) return false;
        out = bf16_to_f32(raw);
        return true;
    };
    if (!bf16("norm1.weight", w.n1w)) return false;
    if (!bf16("norm1.bias", w.n1b)) return false;
    if (!bf16("norm2.weight", w.n2w)) return false;
    if (!bf16("norm2.bias", w.n2b)) return false;
    if (!bf16("attn.out_proj.bias", w.ob)) return false;
    if (!bf16("ffn.net.0.bias", w.fib)) return false;
    if (!bf16("ffn.net.3.bias", w.fob)) return false;
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        printf("usage: %s <dump-dir> <diar.gguf> [ENCODER_STAGE_DIR=<dir>]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    const std::string model = argv[2];

    bool ok = false;
    const std::vector<uint8_t> in_raw = slurp(dir + "/layer0_in.f32", &ok);
    if (!ok) { printf("missing layer0_in.f32 (dump with AUDIOCPP_DUMP_LAYER_INDEX=0)\n"); return 1; }
    // Prefer the sound isolated oracle if present; fall back to the (also-sound - encoder_out is a real
    // graph output, not a recycled intermediate) production dump.
    std::vector<uint8_t> ref_raw = slurp(dir + "/iso_encoder_out.f32", &ok);
    if (!ok) ref_raw = slurp(dir + "/encoder_out.f32", &ok);
    if (!ok) { printf("missing iso_encoder_out.f32 / encoder_out.f32\n"); return 1; }
    const std::vector<uint8_t> mask_raw = slurp(dir + "/layer0_mask.f16", &ok);
    const std::vector<uint8_t> cos_raw = slurp(dir + "/layer0_rope_cos.f32", &ok);
    const std::vector<uint8_t> sin_raw = slurp(dir + "/layer0_rope_sin.f32", &ok);
    long hidden = 0, frames = 0;
    if (FILE * sh = std::fopen((dir + "/layer0_in.shape").c_str(), "rb")) {
        if (fscanf(sh, "%ld %ld", &hidden, &frames) != 2) { fclose(sh); return 1; }
        fclose(sh);
    }
    const int64_t H = hidden, T = frames;

    int64_t HD = 64, HEADS = 0;
    if (FILE * qne = std::fopen((dir + "/layer0_03_q.bin.ne").c_str(), "rb")) {
        long a0 = 0, a1 = 0, a2 = 0;
        if (fscanf(qne, "%ld %ld %ld", &a0, &a1, &a2) == 3 && a0 > 0) { HD = a0; HEADS = a2; }
        fclose(qne);
    }
    if (HEADS == 0) HEADS = H / HD;
    const int64_t HALF = HD / 2;

    int n_layers = 0;
    if (const char * nl = std::getenv("PORT_N_LAYERS")) {
        n_layers = atoi(nl);
    } else {
        while (gguf_has_tensor(model, ("encoder.layers." + std::to_string(n_layers) + ".norm1.weight").c_str())) n_layers++;
    }
    printf("hidden=%lld frames=%lld head_dim=%lld heads=%lld n_layers=%d\n",
           (long long) H, (long long) T, (long long) HD, (long long) HEADS, n_layers);
    if (n_layers == 0) { printf("no layers found (checked encoder.layers.0.norm1.weight)\n"); return 1; }

    std::vector<LayerWeights> layers(n_layers);
    for (int i = 0; i < n_layers; i++) {
        if (!load_layer(model, i, layers[i])) { printf("failed to load layer %d weights\n", i); return 1; }
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 2);
    // ~20 tensors/layer * n_layers, plus the four shared inputs and per-layer weight tensors (11 each).
    ggml_init_params ip = {
        ggml_tensor_overhead() * (size_t) (120 * n_layers + 64) +
            ggml_graph_overhead_custom((size_t) (96 * n_layers + 64), false),
        nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    const float eps = 1e-5f;
    ggml_tensor * in   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, T, T, 1, 1);
    ggml_tensor * cos = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HALF, T, HEADS, 1);
    ggml_tensor * sin = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HALF, T, HEADS, 1);

    // Per-layer weight tensor handles, created now (fed after allocation, same as the single-layer port).
    std::vector<ggml_tensor *> w_qkv(n_layers), w_out(n_layers), w_ffn_in(n_layers), w_ffn_out(n_layers);
    std::vector<ggml_tensor *> n1w(n_layers), n1b(n_layers), n2w(n_layers), n2b(n_layers);
    std::vector<ggml_tensor *> ob(n_layers), fib(n_layers), fob(n_layers);
    for (int i = 0; i < n_layers; i++) {
        w_qkv[i]    = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, layers[i].ne_qkv[0], layers[i].ne_qkv[1]);
        w_out[i]    = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, layers[i].ne_out[0], layers[i].ne_out[1]);
        w_ffn_in[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, layers[i].ne_ffn_in[0], layers[i].ne_ffn_in[1]);
        w_ffn_out[i]= ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, layers[i].ne_ffn_out[0], layers[i].ne_ffn_out[1]);
        n1w[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) layers[i].n1w.size());
        n1b[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) layers[i].n1b.size());
        n2w[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) layers[i].n2w.size());
        n2b[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) layers[i].n2b.size());
        ob[i]  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) layers[i].ob.size());
        fib[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) layers[i].fib.size());
        fob[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) layers[i].fob.size());
    }

    // --- the layer, exactly tools/layer0_port.cpp's proven op sequence, parametrised by weight set ---------
    const auto build_layer = [&](ggml_tensor * x, int i) {
        ggml_tensor * n1 = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), n1w[i]), n1b[i]);
        ggml_tensor * qkv = ggml_cont(ctx, ggml_mul_mat(ctx, w_qkv[i], n1));
        const auto heads = [&](int64_t off) {
            ggml_tensor * sl = ggml_view_2d(ctx, qkv, H, T, qkv->nb[1], (size_t) off * ggml_element_size(qkv));
            return ggml_permute(ctx, ggml_reshape_4d(ctx, ggml_cont(ctx, sl), HD, HEADS, T, 1), 0, 2, 1, 3);
        };
        ggml_tensor * q0 = heads(0), * k0 = heads(H), * v = heads(2 * H);
        const auto rope = [&](ggml_tensor * t) {
            ggml_tensor * x1 = ggml_view_4d(ctx, t, HALF, T, HEADS, 1, t->nb[1], t->nb[2], t->nb[3], 0);
            ggml_tensor * x2 = ggml_view_4d(ctx, t, HALF, T, HEADS, 1, t->nb[1], t->nb[2], t->nb[3], HALF * t->nb[0]);
            ggml_tensor * c = ggml_repeat(ctx, cos, x1), * s = ggml_repeat(ctx, sin, x1);
            return ggml_concat(ctx, ggml_sub(ctx, ggml_mul(ctx, x1, c), ggml_mul(ctx, x2, s)),
                                ggml_add(ctx, ggml_mul(ctx, x2, c), ggml_mul(ctx, x1, s)), 0);
        };
        ggml_tensor * q = rope(q0), * k = rope(k0);
        ggml_tensor * attn = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f / std::sqrt((float) HD), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        attn = ggml_reshape_2d(ctx, ggml_cont(ctx, attn), H, T);
        ggml_tensor * o = ggml_add(ctx, ggml_mul_mat(ctx, w_out[i], attn), ob[i]);
        ggml_tensor * x1 = ggml_add(ctx, x, o);
        ggml_tensor * n2 = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x1, eps), n2w[i]), n2b[i]);
        ggml_tensor * ff_in  = ggml_add(ctx, ggml_mul_mat(ctx, w_ffn_in[i], n2), fib[i]);
        ggml_tensor * ff_gel = ggml_gelu_erf(ctx, ff_in);
        ggml_tensor * ff_out = ggml_add(ctx, ggml_mul_mat(ctx, w_ffn_out[i], ff_gel), fob[i]);
        return ggml_add(ctx, x1, ff_out);
    };

    ggml_tensor * stage_output = in;
    for (int i = 0; i < n_layers; i++) stage_output = build_layer(stage_output, i);
    ggml_tensor * out = stage_output;

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, (size_t) (96 * n_layers + 64), false);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) { printf("alloc failed\n"); return 1; }
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_gallocr_free(ga);

    feed_raw(in, in_raw);
    feed_raw(mask, mask_raw);
    feed_raw(cos, cos_raw);
    feed_raw(sin, sin_raw);
    for (int i = 0; i < n_layers; i++) {
        feed_raw(w_qkv[i], layers[i].w_qkv);       feed_raw(w_out[i], layers[i].w_out);
        feed_raw(w_ffn_in[i], layers[i].w_ffn_in); feed_raw(w_ffn_out[i], layers[i].w_ffn_out);
        feed_f32(n1w[i], layers[i].n1w); feed_f32(n1b[i], layers[i].n1b);
        feed_f32(n2w[i], layers[i].n2w); feed_f32(n2b[i], layers[i].n2b);
        feed_f32(ob[i], layers[i].ob);   feed_f32(fib[i], layers[i].fib); feed_f32(fob[i], layers[i].fob);
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { printf("compute failed\n"); return 1; }

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
        printf("ENCODER PORT (%d layers): BYTE-IDENTICAL to audio.cpp (%zu floats)\n", n_layers, ref.size());
    } else {
        printf("ENCODER PORT (%d layers): %zu of %zu floats differ (%.4f%%), first at %zu, max |delta| %.6g\n",
               n_layers, diff, ref.size(), 100.0 * diff / ref.size(), first, worst);
        printf("  ref[%.0f]=%.9g ported=%.9g\n", (double) first, ref[first], got[first]);
    }
    return diff == 0 ? 0 : 1;
}
