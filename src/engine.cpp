// engine.cpp - see engine.h. Written for this repo; the model runtimes it calls are upstream
// (CrispASR's xasr stream, audio.cpp's Nemotron-3 diar stream) and are attributed in PROVENANCE.md.
#include "engine.h"
#include "wav.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

#include <ctime>

#include "xasr.h"                       // CrispASR (MIT) - x-asr streaming Zipformer2
#include "audiocpp.h"                   // audio.cpp (Apache-2.0) - Nemotron-3 diarization stream

namespace nemo {

double now_s() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return double(ts.tv_sec) + 1e-9 * ts.tv_nsec;
}

double rss_mb() {
    FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return 0.0;
    char line[256];
    double v = 0;
    while (std::fgets(line, sizeof(line), f)) {
        if (!std::strncmp(line, "VmHWM", 5)) { std::sscanf(line + 6, "%lf", &v); break; }
    }
    std::fclose(f);
    return v / 1024.0;
}

Engine::~Engine() {
    if (asr_stream_) xasr_stream_free((xasr_stream*)asr_stream_);
    if (asr_ctx_) xasr_free((xasr_context*)asr_ctx_);
    if (session_) audiocpp_session_free((audiocpp_session*)session_);
    if (model_) audiocpp_model_free((audiocpp_model*)model_);
    if (registry_) audiocpp_registry_free((audiocpp_registry*)registry_);
}

bool Engine::init(std::string& err) {
    const double t0 = now_s();
    if (!cfg_.skip_asr) {
        if (cfg_.xasr_model.empty()) { err = "--xasr-model required (or --no-asr)"; return false; }
        xasr_context_params p = xasr_context_default_params();
        p.n_threads = cfg_.threads;
        p.verbosity = 0;
        p.chunk_ms = cfg_.chunk_ms;
        asr_ctx_ = xasr_init_from_file(cfg_.xasr_model.c_str(), p);
        if (!asr_ctx_) { err = "x-asr model failed to load: " + cfg_.xasr_model; return false; }
        asr_stream_ = xasr_stream_init((xasr_context*)asr_ctx_);
        if (!asr_stream_) { err = "xasr_stream_init failed"; return false; }
    }
    if (!cfg_.skip_diar) {
        if (cfg_.diar_model.empty()) { err = "--diar-model required (or --no-diar)"; return false; }
        audiocpp_status st = audiocpp_registry_create(nullptr, (audiocpp_registry**)&registry_);
        if (st != AUDIOCPP_OK) { err = std::string("diar registry: ") + audiocpp_last_error(); return false; }
        audiocpp_model_config mc{};
        mc.family_hint = "nemotron_3_diar";
        st = audiocpp_model_load((audiocpp_registry*)registry_, cfg_.diar_model.c_str(), &mc, nullptr,
                                 (audiocpp_model**)&model_);
        if (st != AUDIOCPP_OK) { err = std::string("diar model load: ") + audiocpp_last_error(); return false; }
        audiocpp_backend_config bc{};
        bc.backend = "cpu";
        bc.device = 0;
        bc.threads = cfg_.threads;
        st = audiocpp_session_create((audiocpp_model*)model_, "diar", "streaming", &bc, nullptr,
                                     (audiocpp_session**)&session_);
        if (st != AUDIOCPP_OK) { err = std::string("diar session (streaming): ") + audiocpp_last_error(); return false; }
        st = audiocpp_stream_start((audiocpp_session*)session_, nullptr);
        if (st != AUDIOCPP_OK) { err = std::string("diar stream_start: ") + audiocpp_last_error(); return false; }
    }
    stats_.load_s = now_s() - t0;
    double latency = cfg_.asr_latency_ms;
    if (latency < 0) latency = cfg_.chunk_ms;    // the encoder's own window is its floor
    fusion_ = Fusion(16000, latency / 1000.0);
    fusion_.set_char_dur_ms(cfg_.char_dur_ms);
    fusion_.set_gap_snap_ms(cfg_.gap_snap_ms);
    return true;
}

// Tag every buffered delta against the turn timeline as it stands. Locals only, so calling it twice
// (a provisional live pass and the final pass) does not double-count anything.
void Engine::attribute(const std::function<void(const Segment&)>& on_segment, bool final_pass) {
    std::map<std::string, int>& spk_id = spk_id_;
    Segment open;
    int idx = 0;
    size_t unattributed = 0, pieces = 0, snapped_chars = 0;

    auto close = [&](const Segment& s) {
        Segment done = s;
        done.index = ++idx;
        on_segment(done);
    };
    {
        for (const TaggedPiece& p : fusion_.attribute_all()) {
        // Text the diarizer left uncovered keeps the label it was already under instead of opening a
        // "Speaker -1" segment. This is not a hedge: the scorer WER is measured with drops every line it
        // cannot parse a speaker from, so an untagged island silently DELETES words from the transcript
        // (0.1765 -> 0.2235 on the bilingual gate with an identical character stream). The count is still
        // reported as unattributed_chars, so the honesty lives in the telemetry, not in the layout.
        if (p.speaker.empty() && !open.text.empty()) {
            unattributed += codepoints(p.text).size();
            open.text += p.text;
            open.end_s = std::max(open.end_s, p.end_s);
            continue;
        }
            pieces++;
            snapped_chars += p.snapped ? codepoints(p.text).size() : 0;
            const std::string& want = p.speaker;
            if (open.text.empty()) {
                if (!want.empty()) {
                    auto it = spk_id.find(want);
                    if (it == spk_id.end()) it = spk_id.emplace(want, (int)spk_id.size()).first;
                    open.speaker = it->second;
                    open.speaker_id = want;
                } else {
                    open.speaker = -1;
                    open.speaker_id.clear();
                    unattributed += codepoints(p.text).size();
                }
                open.text.clear();
                open.start_s = p.start_s;
                open.end_s = p.end_s;
                open.min_confidence = p.min_confidence;
            } else if (want != open.speaker_id) {
                close(open);
                if (!want.empty()) {
                    auto it = spk_id.find(want);
                    if (it == spk_id.end()) it = spk_id.emplace(want, (int)spk_id.size()).first;
                    open.speaker = it->second;
                    open.speaker_id = want;
                } else {
                    open.speaker = -1;
                    open.speaker_id.clear();
                    unattributed += codepoints(p.text).size();
                }
                open.text.clear();
                open.start_s = p.start_s;
                open.end_s = p.end_s;
                open.min_confidence = p.min_confidence;
            }
            open.text += p.text;
            open.end_s = std::max(open.end_s, p.end_s);
            open.min_confidence = std::min(open.min_confidence, p.min_confidence);
        }
    }
    if (!open.text.empty()) close(open);

    stats_.segments = idx;
    stats_.unattributed_chars = unattributed;
    stats_.snapped_chars = snapped_chars;
    stats_.speakers = spk_id.size();
    if (final_pass && getenv("NEMO_DEBUG_ATTR")) {
        std::fprintf(stderr, "[attrib] %zu pieces, %zu turns, %zu chars\n", pieces, fusion_.turns().size(),
                     fusion_.chars());
    }
}

bool Engine::run(const std::function<void(const Segment&)>& on_segment, std::string& err) {
    Wav wav;
    if (!Wav::load(cfg_.audio, wav, err)) return false;
    if (wav.rate != 16000) { err = "need 16 kHz input, got " + std::to_string(wav.rate) + " (resample first)"; return false; }

    const int rate = wav.rate;
    const size_t piece = size_t(cfg_.piece_ms) * rate / 1000;
    const int64_t total = (int64_t)wav.pcm.size();
    const double latency_samples = fusion_.latency_s() * rate;

    const double t0 = now_s();
    int64_t charged_upto = 0;          // samples of audio already claimed by buffered deltas
    size_t turns_seen = 0;

    for (size_t off = 0; off < total; off += piece) {
        const size_t n = std::min(piece, total - off);
        const bool last = (off + n >= (size_t)total);
        const double a0 = now_s();

        // 1. diarizer first: attribution needs the turn that covers this audio
        if (!cfg_.skip_diar) {
            const double d0 = now_s();
            audiocpp_event* ev = nullptr;
            audiocpp_status st = audiocpp_stream_push((audiocpp_session*)session_, wav.pcm.data() + off, n,
                                                      rate, 1, (int64_t)off, &ev);
            if (st != AUDIOCPP_OK) { err = std::string("diar stream_push: ") + audiocpp_last_error(); return false; }
            std::vector<Turn> snapshot;
            auto harvest = [&](const audiocpp_result* r) {
                if (!r) return;
                const size_t k = audiocpp_result_speaker_turn_count(r);
                for (size_t i = 0; i < k; i++) {
                    int64_t s0 = 0, s1 = 0; float conf = 0; const char* sid = nullptr; const char* txt = nullptr;
                    if (audiocpp_result_speaker_turn(r, i, &s0, &s1, &sid, &conf, &txt) == AUDIOCPP_OK) {
                        snapshot.push_back(Turn{s0, s1, sid ? sid : "", conf});
                    }
                }
            };
            harvest(ev ? audiocpp_event_as_result(ev) : nullptr);
            if (ev) audiocpp_event_free(ev);
            for (;;) {                                   // a family may queue several events per push
                audiocpp_event* more = nullptr;
                if (audiocpp_stream_next_event((audiocpp_session*)session_, &more) != AUDIOCPP_OK || !more) break;
                harvest(audiocpp_event_as_result(more));
                audiocpp_event_free(more);
            }
            if (!snapshot.empty()) {
                if (first_turn_audio_ < 0) first_turn_audio_ = double(off + n) / rate;
                fusion_.update_turns(snapshot);
                if (cfg_.live_provisional && fusion_.turns().size() != turns_seen) {
                    turns_seen = fusion_.turns().size();
                    attribute([&](const Segment& s) {
                        std::fprintf(stderr, "~[%d] Speaker %d %s\n", s.index, s.speaker, s.text.c_str());
                    }, false);
                }
            }
            stats_.diar_compute_s += now_s() - d0;
        }

        // 2. transcriber
        std::string delta;
        if (!cfg_.skip_asr) {
            const double s0 = now_s();
            char* full = xasr_stream_accept((xasr_stream*)asr_stream_, wav.pcm.data() + off, (int)n, last);
            std::string cur = full ? full : "";
            std::free(full);
            delta = cur.size() > asr_text_.size() ? cur.substr(asr_text_.size()) : "";
            asr_text_ = cur;
            stats_.asr_compute_s += now_s() - s0;
        }

        // 3. pin the delta to the audio it decodes. The delta describes audio up to (fed - latency);
        // tagging happens in attribute(), against whatever turn timeline exists at that moment.
        if (!delta.empty()) {
            int64_t horizon = last ? total : (int64_t)(off + n) - (int64_t)latency_samples;
            if (horizon < charged_upto) horizon = charged_upto;
            // The Nemotron stream computes in chunks but COMMITS turns in batches (first batch at
            // 30.5 s on the bilingual clip), so deltas are pushed into the attributor's own character
            // timeline and tagged against the turn list whenever it updates - see fusion.h.
            fusion_.push_delta(delta, charged_upto, horizon);
            charged_upto = horizon;
            stats_.deltas++;
            if (stats_.first_partial_s < 0) stats_.first_partial_s = now_s() - t0;
        }

        piece_ms_.push_back((now_s() - a0) * 1000.0);
        stats_.wall_s = now_s() - t0;
        if (cfg_.paced) {                       // 1x wall clock, so latency means what it says
            const double due = t0 + double(off + n) / rate;
            const double sl = due - now_s();
            if (sl > 0) { timespec ts{(long)sl, (long)((sl - (long)sl) * 1e9)}; nanosleep(&ts, nullptr); }
        }
    }

    // Drain the diarizer's final turns, then attribute the whole timeline against them.
    if (!cfg_.skip_diar) {
        audiocpp_result* res = nullptr;
        if (audiocpp_stream_finish((audiocpp_session*)session_, &res) == AUDIOCPP_OK && res) {
            std::vector<Turn> snapshot;
            const size_t k = audiocpp_result_speaker_turn_count(res);
            for (size_t i = 0; i < k; i++) {
                int64_t s0 = 0, s1 = 0; float conf = 0; const char* sid = nullptr; const char* txt = nullptr;
                if (audiocpp_result_speaker_turn(res, i, &s0, &s1, &sid, &conf, &txt) == AUDIOCPP_OK) {
                    snapshot.push_back(Turn{s0, s1, sid ? sid : "", conf});
                }
            }
            if (!snapshot.empty()) fusion_.update_turns(snapshot);
            audiocpp_result_free(res);
        }
    }

    attribute(on_segment, true);

    stats_.audio_s = double(total) / rate;
    stats_.wall_s = now_s() - t0;
    stats_.turns = fusion_.turns().size();
    stats_.first_turn_audio_s = first_turn_audio_;
    std::vector<double> sorted = piece_ms_;
    std::sort(sorted.begin(), sorted.end());
    if (!sorted.empty()) {
        stats_.piece_p95_ms = sorted[size_t(0.95 * (sorted.size() - 1))];
        stats_.piece_max_ms = sorted.back();
    }
    stats_.peak_rss_mb = rss_mb();
    return true;
}

}  // namespace nemo
