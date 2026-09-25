// Shape-response microbenchmark for the q8_0 x q8_0 GEMM path, on the device.
//
// Why: the kernel brief claims a specialised small-N GEMM could give 2-3x on the x-asr encoder, which runs its
// matmuls at 3-24 columns. That claim is currently an extrapolation from "the encoders reach 13-20% of int8
// peak". This measures the thing that bounds it: the SAME ggml kernel's throughput as a function of column
// count, at the k values the models actually use. If ggml's own large-N throughput is far above what it
// achieves at n=3, a small-N-specialised kernel has that much headroom and no more. If the curve is flat, the
// whole GEMM direction is worth nothing and the brief should say so.
//
// Build (from the composite repo, with the deps guard satisfied):
//   $NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-linux-android26 -O3 \
//       -mcpu=cortex-a78 -std=c++17 -I<ref/crispasr/ggml/include> -I<ref/crispasr/ggml/src> \
//       tools/gemm_shape_bench.cpp -o /tmp/bench \
//       -L<ref/crispasr/build-android/ggml/src> -l:libggml.a -l:libggml-cpu.a -l:libggml-base.a \
//       -lm -lpthread
// Run: adb push /tmp/bench /data/local/tmp/nemo_x/ && adb shell "cd /data/local/tmp/nemo_x &&
//       LD_LIBRARY_PATH=. taskset C0 ./bench"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct shape_result { int k, n, m; double gmac_s; double us_per_call; };

static shape_result run_one(ggml_backend_t backend, int k, int n, int m, int reps) {
    ggml_init_params ip = { /*.mem_size=*/ ggml_tensor_overhead() * 16 + ggml_graph_overhead(),
                            /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, m);   // weights
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, n);   // activations
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); return {k, n, m, 0.0, 0.0}; }

    // Fill both operands with real block data (random-ish but deterministic) so nothing is denormal-trapped.
    std::vector<uint8_t> ba(ggml_nbytes(a)), bb(ggml_nbytes(b));
    for (size_t i = 0; i < ba.size(); i++) ba[i] = (uint8_t)(i * 37 + 11);
    for (size_t i = 0; i < bb.size(); i++) bb[i] = (uint8_t)(i * 17 + 5);
    ggml_backend_tensor_set(a, ba.data(), 0, ba.size());
    ggml_backend_tensor_set(b, bb.data(), 0, bb.size());

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);
    ggml_gallocr_free(alloc);

    // warm up, then time
    for (int i = 0; i < 3; i++) ggml_backend_graph_compute(backend, gf);
    const int64_t t0 = ggml_time_us();
    for (int i = 0; i < reps; i++) ggml_backend_graph_compute(backend, gf);
    const int64_t dt = ggml_time_us() - t0;
    const double per_call_us = (double) dt / reps;
    const double gmac_s = (double) k * n * m * reps / (double) dt;   // MAC/us == GMAC/s
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return {k, n, m, gmac_s, per_call_us};
}

int main() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { fprintf(stderr, "no cpu backend\n"); return 1; }
    const int n_threads = 2;
    ggml_backend_cpu_set_n_threads(backend, n_threads);
    printf("# ggml %s, %d threads, q8_0 x q8_0 -> int32 -> f32\n", ggml_version(), n_threads);
    printf("# rows m = 512 (the models' hidden size), k and n from their real shapes\n");
    printf("%6s %6s %6s %12s %14s %10s\n", "k", "n", "m", "GMAC/s", "us/call", "vs n=512");

    const int ks[] = {512, 2048};
    const int ns[] = {3, 6, 12, 24, 48, 128, 351, 512};
    std::vector<shape_result> rows;
    // Sweep the ROW count as well. Rows are what the weight matrix contributes: with m=512 the weight stream
    // is short and the kernel may be limited by having only 512 rows to split across two threads, rather than
    // by arithmetic. If a much larger m does not raise throughput, then ~20 GMAC/s is this machine's practical
    // int8 GEMM ceiling and the whole 'a tuned kernel gives 2-3x' claim collapses.
    for (int m : {512, 4096}) {
        for (int k : ks) {
            for (int n : ns) {
                const int reps = n <= 24 ? (m > 512 ? 40 : 200) : (m > 512 ? 10 : 40);
                shape_result r = run_one(backend, k, n, m, reps);
                if (r.gmac_s <= 0.0) continue;
                rows.push_back(r);
                printf("%6d %6d %6d %12.2f %14.1f\n", r.k, r.n, r.m, r.gmac_s, r.us_per_call);
            }
        }
    }
    // second pass now that the n=512 reference is known
    printf("\n# throughput relative to the same (k, m) at n=512 - the ceiling a small-N kernel could chase\n");
    for (size_t i = 0; i < rows.size(); i++) {
        double ref = 0.0;
        for (auto & o : rows) if (o.k == rows[i].k && o.m == rows[i].m && o.n == 512) ref = o.gmac_s;
        if (ref > 0.0)
            printf("k=%-5d m=%-5d n=%-4d %8.2f GMAC/s  %5.1f%% of n=512\n", rows[i].k, rows[i].m, rows[i].n,
                   rows[i].gmac_s, 100.0 * rows[i].gmac_s / ref);
    }
    ggml_backend_free(backend);
    return 0;
}
