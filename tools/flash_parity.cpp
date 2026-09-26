// Do CrispASR's ggml and audio.cpp's ggml agree on ggml_flash_attn_ext with the REAL shapes?
//
// This is the load-bearing question for the one-runtime merge, and the parity tests did not answer it: they
// used small synthetic tensors (16 heads, T=6, no mask). The port reaches flash attention with 8 heads,
// T=391, head_dim 64 and an F16 validity mask, and CrispASR's tree returns an all-NaN attention output from
// finite q/k/v. Either the port's tensors are wrong, or the two vendored ggml versions behave differently on
// this op at this shape.
//
// This harness removes the port from the question entirely: it reads the port's own dumped q, k and v plus
// audio.cpp's mask, runs ONE op, and reports rms / non-finite count. Compiled twice - once against each
// vendored ggml - so any difference is attributable to the runtime and nothing else.
//
// Usage: flash_parity <dump-dir> [LAYER0_VARIANT]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<uint8_t> slurp(const std::string & p, bool * ok = nullptr) {
    std::vector<uint8_t> d;
    FILE * fh = std::fopen(p.c_str(), "rb");
    if (!fh == false && fh == nullptr) { if (ok) *ok = false; return d; }
    if (fh == nullptr) { if (ok) *ok = false; return d; }
    fseek(fh, 0, SEEK_END);
    const long n = ftell(fh);
    fseek(fh, 0, SEEK_SET);
    d.resize((size_t) n);
    const bool good = n > 0 && fread(d.data(), 1, (size_t) n, fh) == (size_t) n;
    fclose(fh);
    if (ok) *ok = good;
    return d;
}

int main(int argc, char ** argv) {
    if (argc < 2) { printf("usage: %s <dump-dir>\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    bool ok = false;
    const std::vector<uint8_t> q = slurp(dir + "/port_03_q.bin", &ok);
    if (!ok) { printf("missing port_03_q.bin (run layer0_port with LAYER0_STAGE_DIR first)\n"); return 1; }
    const std::vector<uint8_t> k = slurp(dir + "/port_04_k.bin", &ok);
    const std::vector<uint8_t> v = slurp(dir + "/port_05_v.bin", &ok);
    const std::vector<uint8_t> m = slurp(dir + "/layer0_mask.f16", &ok);
    const int64_t HD = 64, T = 391, HEADS = 8;   // read from the traced ne earlier
    const size_t n = (size_t) HD * T * HEADS;
    if (q.size() != n * 4 || k.size() != n * 4 || v.size() != n * 4) {
        printf("unexpected sizes q=%zu k=%zu v=%zu (expected %zu)\n", q.size(), k.size(), v.size(), n * 4);
        return 1;
    }
    printf("shapes from the port: q/k/v [%lld, %lld, %lld, 1], mask %zu bytes\n",
           (long long) HD, (long long) T, (long long) HEADS, m.size());

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 2);
    ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * tq = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HD, T, HEADS, 1);
    ggml_tensor * tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HD, T, HEADS, 1);
    ggml_tensor * tv = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HD, T, HEADS, 1);
    ggml_tensor * tm = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, T, T, 1, 1);
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, tq, tk, tv, tm, 1.0f / std::sqrt((float) HD), 0.0f, 0.0f);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, attn);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { printf("alloc failed\n"); return 1; }
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_gallocr_free(ga);
    ggml_backend_tensor_set(tq, q.data(), 0, q.size());
    ggml_backend_tensor_set(tk, k.data(), 0, k.size());
    ggml_backend_tensor_set(tv, v.data(), 0, v.size());
    ggml_backend_tensor_set(tm, m.data(), 0, m.size());

    for (int pass = 0; pass < 2; pass++) {
        if (pass == 1) ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { printf("compute failed\n"); return 1; }
        const size_t no = ggml_nelements(attn);
        std::vector<float> o(no);
        ggml_backend_tensor_get(attn, o.data(), 0, no * sizeof(float));
        size_t bad = 0;
        double ss = 0.0;
        for (float x : o) { if (std::isnan(x) || std::isinf(x)) bad++; else ss += (double) x * x; }
        printf("  %-8s : ne=[%lld %lld %lld %lld]  non-finite %zu / %zu  rms %.6g\n",
               pass == 0 ? "DEFAULT" : "PREC_F32", (long long) attn->ne[0], (long long) attn->ne[1],
               (long long) attn->ne[2], (long long) attn->ne[3], bad, no, no ? std::sqrt(ss / no) : 0.0);
    }
    ggml_backend_free(backend);
    return 0;
}
