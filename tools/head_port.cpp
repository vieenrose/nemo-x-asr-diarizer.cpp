// The Nemotron-3 diarizer's "head" (final_norm -> encoder_projection -> subpixel_upsample conv1d -> relu ->
// head_hidden -> relu -> speaker_head -> sigmoid), rebuilt in CrispASR's ggml, diffed against audio.cpp's own.
// Completes the port alongside tools/layer0_port.cpp (one encoder layer) and tools/encoder_port.cpp (all
// encoder layers): together they cover input -> probabilities end to end.
//
// Usage: head_port <dump-dir> <diar.gguf>
//   <dump-dir> needs encoder_out.f32 (+ layer0_in.shape for hidden; frames comes from encoder_out's own
//   size / hidden) and probabilities.f32 (or iso_probabilities.f32, preferred) from a run with
//   AUDIOCPP_DUMP_LAYER0=<dir> (any AUDIOCPP_DUMP_LAYER_INDEX) and, for the sound isolated oracle,
//   AUDIOCPP_HEAD_ISOLATED=<dir>.

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

// Reads a tensor's raw bytes AND the ne[] ggml's own gguf reader assigns it - never trust a python reader's
// or a call site's guessed shape order for this; the port's job is to match ggml's own interpretation.
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
// regular_conv_weight (conv_modules.cpp) casts a BF16 conv weight to F16, not F32, before ggml_conv_1d - a
// real precision difference (F16 has ~3 decimal digits) this port must reproduce exactly, not approximate.
static void feed_f32_as_f16(ggml_tensor * t, const std::vector<float> & v) {
    std::vector<ggml_fp16_t> f16(v.size());
    ggml_fp32_to_fp16_row(v.data(), f16.data(), (int64_t) v.size());
    ggml_backend_tensor_set(t, f16.data(), 0, f16.size() * sizeof(ggml_fp16_t));
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        printf("usage: %s <dump-dir> <diar.gguf>\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    const std::string model = argv[2];

    bool ok = false;
    const std::vector<uint8_t> in_raw = slurp(dir + "/encoder_out.f32", &ok);
    if (!ok) { printf("missing encoder_out.f32\n"); return 1; }
    std::vector<uint8_t> ref_raw = slurp(dir + "/iso_probabilities.f32", &ok);
    if (!ok) ref_raw = slurp(dir + "/probabilities.f32", &ok);
    if (!ok) { printf("missing iso_probabilities.f32 / probabilities.f32\n"); return 1; }
    long hidden = 0, frames_hint = 0;
    if (FILE * sh = std::fopen((dir + "/layer0_in.shape").c_str(), "rb")) {
        if (fscanf(sh, "%ld %ld", &hidden, &frames_hint) != 2) { fclose(sh); return 1; }
        fclose(sh);
    }
    const int64_t H = hidden;
    const int64_t T = (int64_t) (in_raw.size() / sizeof(float)) / H;
    if (T * H * (int64_t) sizeof(float) != (int64_t) in_raw.size()) {
        printf("encoder_out.f32 size %zu is not a multiple of hidden=%lld\n", in_raw.size(), (long long) H);
        return 1;
    }

    struct Loaded { std::vector<uint8_t> raw; std::vector<float> f32; int64_t ne[4]; };
    Loaded final_norm_w, final_norm_b, enc_proj_w, enc_proj_b, upsample_w, upsample_b,
           head_hidden_w, head_hidden_b, speaker_head_w, speaker_head_b;
    auto load_bf16 = [&](const char * name, Loaded & dst) {
        if (!gguf_tensor(model, name, dst.raw, dst.ne)) { printf("missing %s\n", name); return false; }
        dst.f32 = bf16_to_f32(dst.raw);
        return true;
    };
    auto load_q8 = [&](const char * name, Loaded & dst) {
        if (!gguf_tensor(model, name, dst.raw, dst.ne)) { printf("missing %s\n", name); return false; }
        return true;
    };
    if (!load_bf16("encoder.final_norm.weight", final_norm_w)) return 1;
    if (!load_bf16("encoder.final_norm.bias", final_norm_b)) return 1;
    if (!load_q8("sortformer_modules.encoder_proj.weight", enc_proj_w)) return 1;
    if (!load_bf16("sortformer_modules.encoder_proj.bias", enc_proj_b)) return 1;
    if (!load_bf16("sortformer_modules.subpixel_upsample.weight", upsample_w)) return 1;   // conv weight: F32/BF16, not quantised
    if (!load_bf16("sortformer_modules.subpixel_upsample.bias", upsample_b)) return 1;
    if (!load_q8("sortformer_modules.first_hidden_to_hidden.weight", head_hidden_w)) return 1;
    if (!load_bf16("sortformer_modules.first_hidden_to_hidden.bias", head_hidden_b)) return 1;
    if (!load_q8("sortformer_modules.single_hidden_to_spks.weight", speaker_head_w)) return 1;
    if (!load_bf16("sortformer_modules.single_hidden_to_spks.bias", speaker_head_b)) return 1;

    const int64_t head_hidden_size = enc_proj_w.ne[1];              // encoder_proj: [H, head_hidden] Q8_0 (mul_mat weight ne=[in,out])
    const int64_t upsample_factor_x_hidden = upsample_b.ne[0];      // subpixel_upsample bias: [hidden*upsample_factor]
    const int64_t kernel_size = upsample_w.ne[0];
    const int64_t upsample_in_ch = upsample_w.ne[1];
    const int64_t upsample_out_ch = upsample_w.ne[2];
    const int64_t upsample_factor = upsample_out_ch / upsample_in_ch;
    printf("hidden=%lld frames=%lld head_hidden=%lld upsample: kernel=%lld in_ch=%lld out_ch=%lld factor=%lld (bias %lld)\n",
           (long long) H, (long long) T, (long long) head_hidden_size, (long long) kernel_size,
           (long long) upsample_in_ch, (long long) upsample_out_ch, (long long) upsample_factor,
           (long long) upsample_factor_x_hidden);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 2);
    ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    const float eps = 1e-5f;
    ggml_tensor * in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
    ggml_tensor * fnw = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, final_norm_w.ne[0]);
    ggml_tensor * fnb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, final_norm_b.ne[0]);
    ggml_tensor * epw = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, enc_proj_w.ne[0], enc_proj_w.ne[1]);
    ggml_tensor * epb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, enc_proj_b.ne[0]);
    ggml_tensor * usw = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, upsample_w.ne[0], upsample_w.ne[1], upsample_w.ne[2]);
    ggml_tensor * usb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, upsample_b.ne[0]);
    ggml_tensor * hhw = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, head_hidden_w.ne[0], head_hidden_w.ne[1]);
    ggml_tensor * hhb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_hidden_b.ne[0]);
    ggml_tensor * shw = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, speaker_head_w.ne[0], speaker_head_w.ne[1]);
    ggml_tensor * shb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, speaker_head_b.ne[0]);

    // --- final_norm ----------------------------------------------------------------------------------------
    ggml_tensor * n = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, in, eps), fnw), fnb);
    // --- encoder_projection (Linear: hidden -> head_hidden) -------------------------------------------------
    ggml_tensor * proj = ggml_add(ctx, ggml_mul_mat(ctx, epw, n), epb);   // [head_hidden, T]
    // --- transpose to [T, head_hidden] (channels-first for conv1d), conv1d, transpose back -------------------
    // ggml_conv_1d(ctx, kernel[K,IC,OC], input[L,IC,N]) -> [OL,OC,N]. proj is [head_hidden, T] = logically
    // [T, head_hidden] row-major; conv1d wants [L, IC] i.e. length fastest is WRONG - it wants IC as ne1, L as
    // ne0? No: ggml's conv_1d input convention is ne=[L, IC, N] (L fastest). `proj` has ne=[head_hidden, T],
    // i.e. head_hidden(=IC) fastest - the OPPOSITE of what conv_1d's `b` wants, hence the module's own
    // transpose before conv. Transpose proj to ne=[T, head_hidden, 1] (L fastest) via ggml_cont(ggml_transpose).
    ggml_tensor * proj_t = ggml_cont(ctx, ggml_transpose(ctx, proj));      // ne=[T, head_hidden]
    // KNOWN, PERMANENT GAP (not a bug - see docs/one-runtime-merge.md SS15): CrispASR's ggml_conv_1d is
    // patched to force F32 im2col whenever either side is F32/BF16 ("Upstream hardcodes F16, which produces
    // MUL_MAT(F16,F16) - unsupported by the CPU backend after our F16 vec_dot_type=F32 change"). audio.cpp's
    // OWN (unpatched) ggml_conv_1d hardcodes F16 im2col unconditionally, which is measurably less precise
    // (~1e-3 near unit scale after the sigmoid) - this port's one residual, structural difference. Forcing
    // ggml_im2col(..., GGML_TYPE_F16) directly to match audio.cpp exactly was TRIED and CRASHES
    // (GGML_ASSERT(src1->type == GGML_TYPE_F32) in ggml-cpu.c, exactly as the patch comment predicts) -
    // CrispASR's ggml is structurally incapable of the F16 path audio.cpp took. Using CrispASR's own
    // (higher-precision) F32 path here.
    ggml_tensor * conv = ggml_conv_1d(ctx, usw, proj_t, /*s0*/1, /*p0*/(int) (kernel_size / 2), /*d0*/1);
    // conv: ne=[T, out_ch=head_hidden*factor]. Add bias broadcast over ne0 (per output channel = ne1).
    ggml_tensor * bias_bc = ggml_reshape_2d(ctx, usb, 1, upsample_factor_x_hidden);
    conv = ggml_add(ctx, conv, bias_bc);
    // transpose back to ne=[out_ch, T] (channel-fastest, matching everything else), then reshape the
    // upsample-factor axis into the sequence axis: [out_ch, T] logically [T, head_hidden*factor] ->
    // [head_hidden, T*factor] (each frame's out_ch splits into `factor` consecutive new frames of
    // head_hidden each - PyTorch's PixelShuffle-style subpixel upsample along the sequence axis).
    ggml_tensor * conv_t = ggml_cont(ctx, ggml_transpose(ctx, conv));      // ne=[head_hidden*factor, T]
    ggml_tensor * up = ggml_reshape_2d(ctx, conv_t, head_hidden_size, T * upsample_factor);
    // --- relu, head_hidden linear, relu, speaker_head linear, sigmoid --------------------------------------
    ggml_tensor * r1 = ggml_relu(ctx, up);
    ggml_tensor * hh = ggml_add(ctx, ggml_mul_mat(ctx, hhw, r1), hhb);
    ggml_tensor * r2 = ggml_relu(ctx, hh);
    ggml_tensor * sh = ggml_add(ctx, ggml_mul_mat(ctx, shw, r2), shb);
    ggml_tensor * out = ggml_sigmoid(ctx, sh);

    for (ggml_tensor * t : { n, proj, proj_t, conv, conv_t, up, r1, hh, r2, sh }) ggml_set_output(t);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) { printf("alloc failed\n"); return 1; }
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_gallocr_free(ga);

    feed_raw(in, in_raw);
    feed_f32(fnw, final_norm_w.f32);   feed_f32(fnb, final_norm_b.f32);
    feed_raw(epw, enc_proj_w.raw);     feed_f32(epb, enc_proj_b.f32);
    feed_f32_as_f16(usw, upsample_w.f32); feed_f32(usb, upsample_b.f32);
    feed_raw(hhw, head_hidden_w.raw);  feed_f32(hhb, head_hidden_b.f32);
    feed_raw(shw, speaker_head_w.raw); feed_f32(shb, speaker_head_b.f32);

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { printf("compute failed\n"); return 1; }

    auto dump_stat = [&](const char * name, ggml_tensor * t) {
        std::vector<float> v((size_t) ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
        double s2 = 0; for (float x : v) s2 += (double) x * x;
        printf("  stage %-8s ne=[%lld,%lld,%lld,%lld] rms=%.6f [0..2]=%.6f %.6f %.6f\n", name,
               (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
               std::sqrt(s2 / v.size()), v.size() > 0 ? v[0] : 0, v.size() > 1 ? v[1] : 0, v.size() > 2 ? v[2] : 0);
    };
    if (std::getenv("HEAD_PORT_STAGES")) {
        dump_stat("n", n); dump_stat("proj", proj); dump_stat("proj_t", proj_t); dump_stat("conv", conv);
        dump_stat("conv_t", conv_t); dump_stat("up", up); dump_stat("r1", r1); dump_stat("hh", hh);
        dump_stat("r2", r2); dump_stat("sh", sh);
    }

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
        printf("HEAD PORT: BYTE-IDENTICAL to audio.cpp (%zu floats)\n", ref.size());
    } else {
        printf("HEAD PORT: %zu of %zu floats differ (%.4f%%), first at %zu, max |delta| %.6g\n",
               diff, ref.size(), 100.0 * diff / ref.size(), first, worst);
        printf("  ref[%.0f]=%.9g ported=%.9g\n", (double) first, ref[first], got[first]);
    }
    return diff == 0 ? 0 : 1;
}
