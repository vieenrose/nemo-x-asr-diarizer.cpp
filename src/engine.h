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
    double token_offset_ms = 300;   // a transducer emits a token at the frame where it DECIDED the token,
                                    // not at the audio that produced it, so raw frame times run late. 300 ms
                                    // was derived from the audio - it is the shift that puts 0 of 119 tokens
                                    // inside ground-truth silence on the sequential bilingual clip (3 at
                                    // 0 ms, 2 at 450 ms) - never from a WER or attribution score. See
                                    // tools/validate_timestamps.py; re-derive per model, do not copy it.
    int  gap_fill       = 0;        // 0 nearest, 1 previous-speaker continuation
    int  timing         = 0;        // 0 auto, 1 force model timestamps, 2 force inferred placement
    double gap_snap_ms    = 0;      // attribution tolerance across a between-turns pause; 0 = nearest.
                                    // Not a timid default: the archive scorer that these numbers come
                                    // from drops "Speaker -1" lines from WER entirely, so leaving text
                                    // untagged costs WER (0.2235 vs 0.1765, identical transcript). The
                                    // honest counter is snapped_chars, which is reported either way.
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

// The model's own token timeline, with the speaker each token landed on. Only available when the model
// reports times; the validator (tools/validate_timestamps.py) reads this to check the times against
// ground truth WITHOUT trusting the attributor that consumed them.
struct TokenInfo {
    std::string text;
    double      t_s = 0;
    std::string speaker_id;      // "" = unattributed
    bool        snapped = false;
};

struct Stats {
    double audio_s = 0, wall_s = 0, load_s = 0, asr_compute_s = 0, diar_compute_s = 0;
    double first_partial_s = -1, piece_p95_ms = 0, piece_max_ms = 0;
    double peak_rss_mb = 0;
    double first_turn_audio_s = -1;
    int timing_mode = 0;            // 1 = model-given token timestamps, 0 = inferred placement
    size_t tokens = 0;   // when the diarizer committed its FIRST turn, in audio time.
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
    std::vector<TokenInfo> token_table() const;
    const std::vector<Turn>& turns() const { return fusion_.turns(); }
    const std::string& transcript() const { return asr_text_; }

private:
    // Tag every buffered delta against the current turn timeline and emit the segments.
    void attribute(const std::function<void(const Segment&)>& on_segment, bool final_pass);

    Config cfg_;
    Fusion fusion_;
    Stats  stats_;
    std::string asr_text_;
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
