// diar_crispasr_test.cpp - exercises the DiarCrispASR class (src/diar_crispasr.cpp) directly against the
// same dump-directory acceptance test tools/encoder_port.cpp and tools/head_port.cpp use, before wiring it
// into the composite via the external-encoder hook. Confirms the reusable, weights-load-once class gives the
// same answer as the one-shot CLI tools that validated the op sequence.
//
// Usage: diar_crispasr_test <dump-dir> <diar.gguf>
//   Needs layer0_in.f32 (+ .shape, from AUDIOCPP_DUMP_LAYER_INDEX=0) and iso_probabilities.f32 (preferred) or
//   probabilities.f32.
#include "../src/diar_crispasr.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

static std::vector<uint8_t> slurp(const std::string & path, bool * ok) {
    std::vector<uint8_t> data;
    FILE * fh = std::fopen(path.c_str(), "rb");
    if (fh == nullptr) { *ok = false; return data; }
    fseek(fh, 0, SEEK_END);
    const long n = ftell(fh);
    fseek(fh, 0, SEEK_SET);
    data.resize((size_t) n);
    *ok = n > 0 && fread(data.data(), 1, (size_t) n, fh) == (size_t) n;
    fclose(fh);
    return data;
}

int main(int argc, char ** argv) {
    if (argc < 3) { printf("usage: %s <dump-dir> <diar.gguf>\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    const std::string model = argv[2];

    bool ok = false;
    const auto in_raw = slurp(dir + "/layer0_in.f32", &ok);
    if (!ok) { printf("missing layer0_in.f32\n"); return 1; }
    std::vector<uint8_t> ref_raw = slurp(dir + "/iso_probabilities.f32", &ok);
    if (!ok) ref_raw = slurp(dir + "/probabilities.f32", &ok);
    if (!ok) { printf("missing iso_probabilities.f32 / probabilities.f32\n"); return 1; }

    long hidden = 0, frames = 0;
    if (FILE * sh = std::fopen((dir + "/layer0_in.shape").c_str(), "rb")) {
        if (fscanf(sh, "%ld %ld", &hidden, &frames) != 2) { fclose(sh); return 1; }
        fclose(sh);
    }
    std::vector<float> embeddings(in_raw.size() / sizeof(float));
    memcpy(embeddings.data(), in_raw.data(), in_raw.size());

    try {
        nemo::DiarCrispASR diar(model, 2);
        printf("loaded: hidden=%lld num_speakers=%lld\n", (long long) diar.hidden_size(), (long long) diar.num_speakers());
        auto probs = diar.encode(embeddings, 1, frames, {frames});
        const std::vector<float> ref((const float *) ref_raw.data(), (const float *) ref_raw.data() + ref_raw.size() / 4);
        if (ref.size() != probs.size()) { printf("SIZE MISMATCH %zu vs %zu\n", ref.size(), probs.size()); return 1; }
        size_t diff = 0, first = 0;
        double worst = 0.0;
        for (size_t i = 0; i < ref.size(); i++) {
            if (memcmp(&ref[i], &probs[i], sizeof(float)) != 0) {
                if (diff == 0) first = i;
                diff++;
                worst = std::max(worst, (double) fabs(ref[i] - probs[i]));
            }
        }
        if (diff == 0) {
            printf("DIAR_CRISPASR: BYTE-IDENTICAL to audio.cpp (%zu floats)\n", ref.size());
        } else {
            printf("DIAR_CRISPASR: %zu of %zu floats differ (%.4f%%), first at %zu, max |delta| %.6g\n",
                   diff, ref.size(), 100.0 * diff / ref.size(), first, worst);
            printf("  ref[%zu]=%.9g got=%.9g\n", first, ref[first], probs[first]);
        }
        return 0;
    } catch (const std::exception & e) {
        printf("EXCEPTION: %s\n", e.what());
        return 1;
    }
}
