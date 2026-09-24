// engine.h - the composite: one audio timeline, two streaming models, one speaker-tagged transcript.
//
// Written for this repo. Both models run in-process, in the same loop, on the same pieces:
//
//   piece (100 ms)  ->  Nemotron-3 diar stream  ->  turn intervals (refined as we go)
//                 ->  x-asr stream              ->  text delta
//                 ->  Fusion                    ->  characters tagged with a speaker
//
// Feeding the diarizer BEFORE the transcriber per piece is deliberate: attribution needs the turn
// that covers the audio, and a turn that arrives one piece late shifts the first characters of
// every speaker change into the previous speaker.
#pragma once
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "fusion.h"

namespace nemo {

struct Config {
    std::string xasr_model;         // x-asr-zh-en-q8_0.gguf
    std::string diar_model;         // nemotron-3-diarization-q8_0.gguf
    std::string audio;              // 16 kHz mono PCM16 wav
    int  threads      = 2;
    int  piece_ms     = 100;
    int  chunk_ms     = 480;        // x-asr chunk; also the default ASR latency estimate
    double asr_latency_ms = -1;     // <0 = chunk_ms (the encoder's own window)
    double char_dur_ms    = 90;     // per-character speaking rate used to place a delta in time
    double gap_snap_ms    = 400;    // attribution tolerance across a between-turns pause
    bool paced        = false;      // wall-clock 1x, for the realtime demonstration
    bool live_provisional = false;  // re-attribute on every turn update, print revisions to stderr
    bool skip_asr     = false;
    bool skip_diar    = false;
};

struct Segment {
    int    index = 0;
    int    speaker = -1;            // -1 = unattributed audio
    std::string speaker_id;         // diarizer's label, e.g. "speaker_2"
    std::string text;
    double start_s = 0, end_s = 0;
    float  min_confidence = 0.0f;
    std::string asr_text_so_far;    // cumulative transcript at close time (for diffing)
};

// One ASR delta pinned to the audio it decodes. Attribution runs over these, NOT over live text:
// the Nemotron stream computes in chunks but COMMITS turns in batches - on the bilingual clip the first
// turn arrived at 30.5 s of audio - so a diarizer-in front of the text at push time can only tag
// everything "unknown". Deltas are kept, and tagged against the turn timeline whenever it updates.
struct Delta {
    int64_t     start = 0, end = 0;   // samples
    std::string text;
};

struct Stats {
    double audio_s = 0, wall_s = 0, load_s = 0, asr_compute_s = 0, diar_compute_s = 0;
    double first_partial_s = -1, piece_p95_ms = 0, piece_max_ms = 0;
    double peak_rss_mb = 0;
    double first_turn_audio_s = -1;   // when the diarizer committed its FIRST turn, in audio time.
                                      // This is the attribution floor: no word can be tagged before it.
    size_t turns = 0, segments = 0, speakers = 0, deltas = 0, unattributed_chars = 0, snapped_chars = 0;
    std::vector<double> pass_rtf;
};

class Engine {
public:
    Engine(Config cfg) : cfg_(std::move(cfg)), fusion_(16000, 0.0) {}
    ~Engine();

    bool init(std::string& err);
    // Called whenever a segment closes (speaker change or end of input).
    bool run(const std::function<void(const Segment&)>& on_segment, std::string& err);

    const Stats& stats() const { return stats_; }
    const std::vector<Turn>& turns() const { return fusion_.turns(); }
    const std::string& transcript() const { return asr_text_; }

private:
    // Tag every buffered delta against the current turn timeline and emit the segments.
    void attribute(const std::function<void(const Segment&)>& on_segment, bool final_pass);

    Config cfg_;
    Fusion fusion_;
    Stats  stats_;
    std::string asr_text_;
    std::vector<Delta> deltas_;
    std::map<std::string, int> spk_id_;
    double first_turn_audio_ = -1;
    void* asr_ctx_ = nullptr;
    void* asr_stream_ = nullptr;
    void* registry_ = nullptr;
    void* model_ = nullptr;
    void* session_ = nullptr;
    std::vector<double> piece_ms_;
};

double rss_mb();      // VmHWM, peak resident set of this process
double now_s();

}  // namespace nemo
