// diar_crispasr_bench.cpp - isolated timing of DiarCrispASR::encode(), no engine.cpp, no ASR running
// concurrently, synthetic input. Answers: is ~3.5s/call for T~380-390 frames real compute cost on this
// device, independent of anything the full composite's bookkeeping might be doing.
#include "../src/diar_crispasr.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: %s <diar.gguf> <T> <reps>\n", argv[0]); return 2; }
    const std::string model = argv[1];
    const int64_t T = atoll(argv[2]);
    const int reps = atoi(argv[3]);

    nemo::DiarCrispASR diar(model, 2);
    const int64_t H = diar.hidden_size();
    printf("loaded: hidden=%lld num_speakers=%lld T=%lld reps=%d\n",
           (long long) H, (long long) diar.num_speakers(), (long long) T, reps);

    std::vector<float> embeddings((size_t)(H * T));
    for (size_t i = 0; i < embeddings.size(); i++) embeddings[i] = (float)((i % 997) - 498) / 500.0f;

    for (int r = 0; r < reps; r++) {
        const auto t0 = std::chrono::steady_clock::now();
        auto probs = diar.encode(embeddings, 1, T, {T});
        const auto t1 = std::chrono::steady_clock::now();
        printf("call %d: %.2f ms (out size %zu)\n", r,
               std::chrono::duration<double, std::milli>(t1 - t0).count(), probs.size());
    }
    return 0;
}
