// unit.cpp - correctness tests for the two pieces that can be wrong in silence: the WAV reader and the
// attributor. Both fail by producing PLAUSIBLE output, which is why they need tests rather than eyeballs.
//
// Run through tests/selftest.sh, which also builds the planted-fault variants and asserts they FAIL.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "fusion.h"
#include "wav.h"

static int failures = 0;

#define CHECK(cond, msg)                                                            \
    do {                                                                            \
        if (cond) { std::printf("  ok   %s\n", msg); }                              \
        else { std::printf("  FAIL %s   (%s:%d)\n", msg, __FILE__, __LINE__); failures++; } \
    } while (0)

static void wr16(std::ostream& o, uint16_t v) { o.write((char*)&v, 2); }
static void wr32(std::ostream& o, uint32_t v) { o.write((char*)&v, 4); }
static void wr64(std::ostream& o, uint64_t v) { o.write((char*)&v, 8); }

// A WAV with a LIST/INFO chunk between fmt and data, which is what ffmpeg actually writes and what the
// first version of this reader could not walk.
static void make_wav(const std::string& path, uint16_t bits, bool with_list, bool truncate) {
    std::ofstream f(path, std::ios::binary);
    const uint32_t frames = 3200;
    const uint32_t data_bytes = frames * 2 * (bits / 16);
    const uint32_t list_bytes = with_list ? 26 : 0;
    f.write("RIFF", 4);
    wr32(f, 36 + data_bytes + list_bytes);
    f.write("WAVE", 4);
    f.write("fmt ", 4); wr32(f, 16);
    wr16(f, 1);                       // PCM
    wr16(f, 1);                       // mono
    wr32(f, 16000); wr32(f, 16000 * 2 * (bits / 16)); wr16(f, 2 * (bits / 16)); wr16(f, bits);
    if (with_list) {
        f.write("LIST", 4); wr32(f, list_bytes);
        f.write("INFOISFT", 8); wr32(f, 10); f.write("Lavf62.3.100\0\0", 14);
    }
    f.write("data", 4); wr32(f, data_bytes);
    for (uint32_t i = 0; i < (truncate ? frames / 3 : frames); i++) wr16(f, (uint16_t)(i * 7));
}

static void test_wav() {
    std::printf("[wav]\n");
    nemo::Wav w; std::string err;

    make_wav("/tmp/nemo_test_ok.wav", 16, true, false);
    CHECK(nemo::Wav::load("/tmp/nemo_test_ok.wav", w, err), "parses a real ffmpeg WAV (LIST chunk before data)");
    CHECK(w.rate == 16000, "sample rate read from fmt, not guessed");
    CHECK(w.pcm.size() == 3200, "frame count matches the data chunk");
    CHECK(err.empty(), "clean parse reports no error");

    make_wav("/tmp/nemo_test_32.wav", 32, true, false);
    CHECK(!nemo::Wav::load("/tmp/nemo_test_32.wav", w, err), "refuses PCM32 instead of misreading it");
    CHECK(err.find("PCM16") != std::string::npos, "says why it refused");

    make_wav("/tmp/nemo_test_trunc.wav", 16, true, true);
    CHECK(!nemo::Wav::load("/tmp/nemo_test_trunc.wav", w, err), "refuses a truncated data chunk");
}

// Per-COMPONENT mark, repeated per character: "0011" means the first two characters landed on
// speaker_0 and the next two on speaker_1. Grouping is the attributor's business; the test wants to see
// every character's verdict, so a piece contributes as many marks as it has codepoints.
static std::string speak(const std::vector<nemo::TaggedPiece>& pieces) {
    std::string s;
    for (const auto& p : pieces) {
        const char mark = p.speaker.empty() ? '?' : p.speaker.substr(8)[0];
        for (size_t i = 0; i < nemo::codepoints(p.text).size(); i++) s += mark;
    }
    return s;
}

static void test_fusion() {
    std::printf("[fusion]\n");
    const int R = 16000;
    std::vector<nemo::Turn> ts = { {0, 2 * R, "speaker_0", 0.9f}, {24 * R / 10, 4 * R, "speaker_1", 0.9f} };

    nemo::Fusion f(R, 0.0);
    f.set_char_dur_ms(100);
    f.set_gap_snap_ms(400);
    f.update_turns(ts);
    CHECK(f.turns().size() == 2, "two turns registered");

    // A delta that describes 0-1 s belongs to speaker_0; one that describes 3-4 s to speaker_1.
    auto a = f.on_delta("hello", 0, 1 * R);
    auto b = f.on_delta("world", 3 * R, 4 * R);
    CHECK(speak(a) == "00000", "text inside turn 0 attributes to speaker_0");
    CHECK(speak(b) == "11111", "text inside turn 1 attributes to speaker_1");

    // A turn re-stated with a longer end refines in place rather than duplicating.
    f.update_turns({ {0, 3 * R, "speaker_0", 0.95f} });
    CHECK(f.turns().size() == 2, "restated turn is refined, not appended");
    CHECK(f.turns()[0].end == 3 * R, "the later end sample wins");
    CHECK(std::abs(f.turns()[0].confidence - 0.95f) < 1e-6, "confidence follows the newest snapshot");

    // Inside a 2.0-2.4 s gap: proximity fill, marked as such.
    nemo::Fusion g(R, 0.0);
    g.set_char_dur_ms(100); g.set_gap_snap_ms(400);
    g.update_turns(ts);
    auto c = g.on_delta("x", (int64_t)(2.05 * R), (int64_t)(2.10 * R));
    CHECK(c.size() == 1 && c[0].snapped, "text in a short between-turns pause is proximity-filled and marked");

    // A gap wider than the snap tolerance stays unattributed - the tool must be able to say "unknown".
    nemo::Fusion h(R, 0.0);
    h.set_char_dur_ms(100); h.set_gap_snap_ms(100);
    h.update_turns({ {0, R, "speaker_0", 0.9f}, {10 * R, 11 * R, "speaker_1", 0.9f} });
    auto d = h.on_delta("x", (int64_t)(5.0 * R), (int64_t)(5.05 * R));
    CHECK(d.size() == 1 && d[0].speaker.empty(), "text far from any turn stays unattributed");

    // The latency term must be load-bearing: the same delta, attributed with and without the encoder's
    // lag, has to disagree across a boundary. If this ever passes silently, the placement code is dead.
    nemo::Fusion late(R, 0.0), early(R, 0.0);
    late.update_turns(ts); early.update_turns(ts);
    auto dl = late.on_delta("abcd", 19000, 40000);       // horizon past the boundary, so it crosses turns
    auto de = early.on_delta("abcd", 19000, 20000);      // same words, horizon inside turn 0
    CHECK(speak(dl) != speak(de), "the attribution horizon changes the answer (placement is not decorative)");

    auto cps = nemo::codepoints("a视b");
    CHECK(cps.size() == 3, "codepoints, not bytes: 'a视b' is three characters");
    std::string back;
    for (auto& [off, len] : cps) back += std::string("a视b").substr(off, len);
    CHECK(back == "a视b", "codepoint slicing reproduces the bytes exactly");
}

int main() {
    test_wav();
    test_fusion();
    std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
