// engine.cpp - see engine.h. Written for this repo; the model runtimes it calls are upstream
// (CrispASR's xasr stream, audio.cpp's Nemotron-3 diar stream) and are attributed in PROVENANCE.md.
#include "engine.h"
#include "wav.h"

#include <algorithm>
#include <cstdio>
#include <map>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/resource.h>
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
    if (diar_request_) audiocpp_request_free((audiocpp_request*)diar_request_);
    if (session_) audiocpp_session_free((audiocpp_session*)session_);
    if (model_) audiocpp_model_free((audiocpp_model*)model_);
    if (registry_) audiocpp_registry_free((audiocpp_registry*)registry_);
}

// Apply cpu affinity per thread. Affinity is per-thread on Linux, so taskset on the process applies to
// every thread and cannot separate the two engines - which is exactly the problem here.
static int apply_affinity(long main_mask, long engine_mask) {
    const pid_t self = getpid();
    DIR* d = opendir("/proc/self/task");
    if (!d) return -1;
    int moved = 0, kept = 0;
    struct dirent* e;
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.') continue;
        const pid_t tid = (pid_t)std::atoi(e->d_name);
        const long m = (tid == self) ? main_mask : engine_mask;
        if (m < 0) { kept++; continue; }
        cpu_set_t cs;
        CPU_ZERO(&cs);
        for (int b = 0; b < 64; b++) if (m & (1L << b)) CPU_SET(b, &cs);
        if (sched_setaffinity(tid, sizeof(cs), &cs) == 0) moved++; else kept++;
    }
    closedir(d);
    return moved;
}

// Turns carried by a diar result, in the shape fusion_ wants. Shared by the per-piece drain and the tail push.
void harvest_turns(const audiocpp_result* r, std::vector<Turn>& out) {
    if (!r) return;
    const size_t k = audiocpp_result_speaker_turn_count(r);
    for (size_t i = 0; i < k; i++) {
        int64_t s0 = 0, s1 = 0; float conf = 0; const char* sid = nullptr; const char* txt = nullptr;
        if (audiocpp_result_speaker_turn(r, i, &s0, &s1, &sid, &conf, &txt) == AUDIOCPP_OK) {
            out.push_back(Turn{s0, s1, sid ? sid : "", conf});
        }
    }
}

namespace {
// Read the file in strides so the page cache holds it. A plain read() of the whole file into a buffer would
// cost an allocation the size of the model; 64 KB at a time is enough to fault every 4 KB page in.
void prefault_file(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return;
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::rewind(f);
    char buf[65536];
    volatile long got = 0;
    for (long off = 0; off < sz; off += (long)sizeof buf) {
        const size_t n = std::fread(buf, 1, sizeof buf, f);
        if (!n) break;
        got += (long)buf[0];           // touch, or the compiler may drop the read
    }
    (void)got;
    std::fclose(f);
}
}  // namespace

bool Engine::init(std::string& err) {
    const double init_t0 = now_s();
    {                                    // thread-count probe, its own scope on purpose
        DIR* d0 = opendir("/proc/self/task");
        int n0 = 0; if (d0) { while (readdir(d0)) n0++; closedir(d0); }
        if (getenv("NEMO_DEBUG_THREADS"))
            std::fprintf(stderr, "[threads] at Engine::init entry: %d\n", n0 - 2);
    }

    auto nthr = []{ DIR* d = opendir("/proc/self/task"); int n = 0; if (d) { while (readdir(d)) n++; closedir(d); } return n - 2; };
    const bool dbg_thr = getenv("NEMO_DEBUG_THREADS") != nullptr;

    const bool prof_on = getenv("NEMO_PROF") != nullptr;   // timers stay off unless asked for
    const double t0 = now_s();
    if (!cfg_.skip_asr) {
        if (cfg_.xasr_model.empty()) { err = "--xasr-model required (or --no-asr)"; return false; }
        xasr_context_params p = xasr_context_default_params();
        p.n_threads = cfg_.threads;
        p.verbosity = 0;
        // xasr_context_default_params() sets use_gpu = TRUE, and xasr_init then calls
        // crispasr_init_gpu_backend() unconditionally when it is set. On a GPU-less phone that is not a
        // clean "fall back to cpu": it cost ~3.5x on the ASR leg (rtf 0.748 vs the standalone probe's
        // 0.389-0.215 on the same model and params), which is exactly the kind of stable, mask-independent
        // penalty I spent a long time misattributing to thread pools and page faults. Always say cpu.
        p.use_gpu = false;
        p.chunk_ms = cfg_.chunk_ms;
        // Bundle support: the loader reads this once from the environment. Set it before init, and always
        // set it (empty included) so a previous bundle run in the same process cannot leak its prefix.
        ::setenv("CRISPASR_GGUF_PREFIX", cfg_.asr_gguf_prefix.c_str(), 1);
        asr_ctx_ = xasr_init_from_file(cfg_.xasr_model.c_str(), p);
        if (!asr_ctx_) { err = "x-asr model failed to load: " + cfg_.xasr_model; return false; }
        asr_stream_ = xasr_stream_init((xasr_context*)asr_ctx_);
        if (!asr_stream_) { err = "xasr_stream_init failed"; return false; }
    }
    if (!cfg_.skip_diar) {
        if (cfg_.diar_model.empty()) { err = "--diar-model required (or --no-diar)"; return false; }
        audiocpp_status st = audiocpp_registry_create(nullptr, (audiocpp_registry**)&registry_);
    if (dbg_thr) std::fprintf(stderr, "[threads] after registry: %d\n", nthr());
        if (st != AUDIOCPP_OK) { err = std::string("diar registry: ") + audiocpp_last_error(); return false; }
        audiocpp_model_config mc{};
        mc.family_hint = "nemotron_3_diar";
        st = audiocpp_model_load((audiocpp_registry*)registry_, cfg_.diar_model.c_str(), &mc, nullptr,
                                 (audiocpp_model**)&model_);
    if (dbg_thr) std::fprintf(stderr, "[threads] after model_load: %d\n", nthr());
        if (st != AUDIOCPP_OK) { err = std::string("diar model load: ") + audiocpp_last_error(); return false; }
        audiocpp_backend_config bc{};
        bc.backend = "cpu";
        bc.device = 0;
        bc.threads = cfg_.threads;
        // Defaults first, CLI last, collapsed into a map so an explicit --diar-session-opt always wins.
        // Appending the CLI pairs to a vector leaves "who wins" up to the library's internals, and an
        // A/B that silently keeps the default when the flag says otherwise is worse than no flag at all.
        std::map<std::string, std::string> session_map;
        for (const auto& kv : cfg_.diar_session_opts) session_map[kv.first] = kv.second;
        audiocpp_options* sopts = audiocpp_options_create();
        if (!sopts) { err = "audiocpp_options_create failed"; return false; }
        for (const auto& kv : session_map) {
            if (audiocpp_options_set(sopts, kv.first.c_str(), kv.second.c_str()) != AUDIOCPP_OK) {
                err = "diar session option " + kv.first + "=" + kv.second + ": " + audiocpp_last_error();
                audiocpp_options_free(sopts);
                return false;
            }
        }
        st = audiocpp_session_create((audiocpp_model*)model_, "diar", "streaming", &bc, sopts,
                                     (audiocpp_session**)&session_);
        if (sopts) audiocpp_options_free(sopts);   // the API copies the map into the session
    if (dbg_thr) std::fprintf(stderr, "[threads] after session_create: %d\n", nthr());
        if (st != AUDIOCPP_OK) { err = std::string("diar session (streaming): ") + audiocpp_last_error(); return false; }
        // The decode knobs live on the REQUEST (session.cpp reads decode_config(stream_request_.options)),
        // so build one instead of passing NULL. It stays alive for the run: freeing it after stream_start
        // would leave the family reading a dangling map, or quietly fall back to the defaults, and a
        // threshold sweep that silently used 0.5 everywhere is the kind of result I do not want to report.
        if (!cfg_.diar_opts.empty()) {
            audiocpp_request* req = audiocpp_request_create();
            if (!req) { err = "audiocpp_request_create failed"; return false; }
            for (const auto& kv : cfg_.diar_opts) {
                audiocpp_status os = audiocpp_request_set_option(req, kv.first.c_str(), kv.second.c_str());
                if (os != AUDIOCPP_OK) {
                    err = "diar option " + kv.first + "=" + kv.second + ": " + audiocpp_last_error();
                    audiocpp_request_free(req);
                    return false;
                }
            }
            diar_request_ = req;
            st = audiocpp_stream_start((audiocpp_session*)session_, req);
    if (dbg_thr) std::fprintf(stderr, "[threads] after stream_start: %d\n", nthr());
        } else {
            st = audiocpp_stream_start((audiocpp_session*)session_, nullptr);
        }
        if (st != AUDIOCPP_OK) { err = std::string("diar stream_start: ") + audiocpp_last_error(); return false; }
    }
    if (cfg_.prefault) prefault_file(cfg_.xasr_model), prefault_file(cfg_.diar_model);
    stats_.load_s = now_s() - t0;   // the prefault is deliberately INSIDE this: the point is to charge it
    double latency = cfg_.asr_latency_ms;
    if (latency < 0) latency = cfg_.chunk_ms;    // the encoder's own window is its floor
    fusion_ = Fusion(16000, latency / 1000.0);
    fusion_.set_char_dur_ms(cfg_.char_dur_ms);
    fusion_.set_gap_snap_ms(cfg_.gap_snap_ms);
    fusion_.set_gap_fill(cfg_.gap_fill == 1 ? Fusion::GapFill::PREVIOUS : Fusion::GapFill::NEAREST);
    return true;
}

// Tag every buffered delta against the turn timeline as it stands. Locals only, so calling it twice
// (a provisional live pass and the final pass) does not double-count anything.
// Rebuild the character timeline from the model's own token timestamps. Returns true only when the
// result reproduces the streamed transcript byte for byte, which is what makes this path safe to prefer:
// a token-time table that disagrees with the text means one of the two is broken, and silently attributing
// against the wrong timeline is exactly the "plausible output, wrong numbers" failure this project keeps
// hitting. Frame k covers audio from k * 40 ms (10 ms fbank hop, encoder downsampling 4; measured 25.6 Hz
// including the tail padding, so 40 ms is the mapping and not a guess).
static bool push_timed_tokens(Fusion& out, xasr_context* ctx, xasr_stream* stream, const std::string& expect,
                              double offset_ms) {
    const int32_t* ids = nullptr;
    const int64_t* frames = nullptr;
    int n = 0;
    if (!stream || xasr_stream_token_times(stream, &ids, &frames, &n) != 0 || n <= 0) return false;
    out.clear_text();
    std::string prev;
    for (int i = 0; i < n; i++) {
        char* all = xasr_tokens_to_text(ctx, ids, i + 1);      // cumulative decode with the SAME function
        std::string cur = all ? all : "";                      // that produced the streamed text, so the
        std::free(all);                                        // two can be compared instead of assumed equal
        if (cur.size() < prev.size()) return false;            // not append-only -> times cannot be mapped
        std::string piece = cur.substr(prev.size());
        prev = cur;
        // 40 ms per encoder frame, minus the decision lag the greedy loop introduces (see engine.h).
        const int64_t at = frames[i] * 640 - (int64_t)(offset_ms * 16.0);
        const int64_t next = (i + 1 < n) ? frames[i + 1] * 640 - (int64_t)(offset_ms * 16.0) : at + 40 * 16;
        out.push_token(piece, std::max<int64_t>(0, at), std::max(next, at + 1));
    }
    return out.text() == expect;
}

std::vector<TokenInfo> Engine::token_table() const {
    std::vector<TokenInfo> out;
#ifdef NEMO_HAVE_TOKEN_TIMES
    const int32_t* ids = nullptr;
    const int64_t* frames = nullptr;
    int n = 0;
    if (!asr_stream_ || xasr_stream_token_times((xasr_stream*)asr_stream_, &ids, &frames, &n) != 0 || n <= 0)
        return out;
    std::string prev;
    for (int i = 0; i < n; i++) {
        char* all = xasr_tokens_to_text((xasr_context*)asr_ctx_, ids, i + 1);
        std::string cur = all ? all : "";
        std::free(all);
        if (cur.size() < prev.size()) break;
        TokenInfo ti;
        ti.text = cur.substr(prev.size());
        ti.t_s = double(frames[i]) * 0.040 - cfg_.token_offset_ms / 1000.0;
        bool sn = false;
        const Turn* t = fusion_.covering((int64_t)(ti.t_s * 16000.0), 640, &sn);
        ti.speaker_id = t ? t->speaker : std::string();
        ti.snapped = sn;
        prev = cur;
        out.push_back(std::move(ti));
    }
#endif
    return out;
}

void Engine::attribute(const std::function<void(const Segment&)>& on_segment, bool final_pass) {
    const Fusion* use = &fusion_;
    Fusion timed = fusion_;
    // The two paths are both live at runtime (not compile-time), because the useful comparison is the same
    // model, same audio, same turns, with only the character timeline coming from a different place.
    bool want_tokens = cfg_.timing != 2;
#ifdef NEMO_HAVE_TOKEN_TIMES
    if (want_tokens && asr_stream_ &&
        push_timed_tokens(timed, (xasr_context*)asr_ctx_, (xasr_stream*)asr_stream_, asr_text_,
                          cfg_.token_offset_ms)) {
        use = &timed;
        stats_.timing_mode = 1;
    } else
#endif
    {
        stats_.timing_mode = 0;
    }

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
        for (const TaggedPiece& p : use->attribute_all()) {
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

    if (final_pass) {
        if (const char* dump = getenv("NEMO_DUMP_TIMELINE")) {
            FILE* f = std::fopen(dump, "w");
            if (f) {
                const auto cps = codepoints(use->text());
                const auto& sp = use->spans();
                std::fprintf(f, "[\n");
                for (size_t i = 0; i < cps.size() && i < sp.size(); i++)
                    std::fprintf(f, "%s  {\"c\": \"%s\", \"t\": %.3f}\n", i ? ",\n" : "",
                                 use->text().substr(cps[i].first, cps[i].second).c_str(),
                                 double(sp[i].start) / 16000.0);
                std::fprintf(f, "]\n");
                std::fclose(f);
            }
        }
    }
    stats_.segments = idx;
    stats_.unattributed_chars = unattributed;
    stats_.snapped_chars = snapped_chars;
    stats_.speakers = spk_id.size();
    if (final_pass && getenv("NEMO_DEBUG_ATTR")) {
        std::fprintf(stderr, "[attrib] %zu pieces, %zu turns, %zu chars, timing=%s\n", pieces,
                     use->turns().size(), use->chars(), stats_.timing_mode ? "tokens" : "inferred");
    }
}

bool Engine::run(const std::function<void(const Segment&)>& on_segment, std::string& err) {    // Set the calling thread's mask now, and every other thread's mask as they appear: the diarizer creates
    // its worker pool on FIRST COMPUTE, so a single pre-loop pass would find nothing to move, and threads
    // created later inherit the creating thread's mask - which is the ASR thread's. Re-applying on a small
    // period is what makes this stick, and it costs one opendir plus a few syscalls every few pieces.
    const bool aff_cfg = cfg_.main_affinity >= 0 || cfg_.engine_affinity >= 0;
    if (aff_cfg) apply_affinity(cfg_.main_affinity, -1);

    Wav wav;
    if (!Wav::load(cfg_.audio, wav, err)) return false;
    // The baseline is fed 24 kHz files; so must this. Resample instead of refusing.
    if (wav.rate != 16000) {
        const int was = wav.rate;
        wav.to_16k();
        std::fprintf(stderr, "[input]   resampled %d Hz -> 16000 Hz (Lanczos3) so the baseline's own\n"
                             "[input]   inputs are accepted without an external conversion step\n", was);
    }

    const int rate = wav.rate;
    const size_t piece = size_t(cfg_.piece_ms) * rate / 1000;
    const int64_t total = (int64_t)wav.pcm.size();
    const double latency_samples = fusion_.latency_s() * rate;

    const bool prof_on = getenv("NEMO_PROF") != nullptr;   // timers stay off unless asked for
    const double t0 = now_s();
    int64_t charged_upto = 0;          // samples of audio already claimed by buffered deltas
    size_t turns_seen = 0;

    for (size_t off = 0; off < total; off += piece) {
        if (aff_cfg && (off / piece) % 4 == 0) {
            const int moved = apply_affinity(cfg_.main_affinity, cfg_.engine_affinity);
            if (getenv("NEMO_DEBUG_THREADS") && off / piece < 12)
                std::fprintf(stderr, "[affinity] piece %zu moved=%d threads=%d\n", off / piece, moved,
                             []{ DIR* d = opendir("/proc/self/task"); int n = 0; if (d) { while (readdir(d)) n++; closedir(d); } return n - 2; }());
        }
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
            auto harvest = [&](const audiocpp_result* r) { harvest_turns(r, snapshot); };
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
            const double pd0 = prof_on ? now_s() : 0.0;
            int64_t horizon = last ? total : (int64_t)(off + n) - (int64_t)latency_samples;
            if (horizon < charged_upto) horizon = charged_upto;
            // The Nemotron stream computes in chunks but COMMITS turns in batches (first batch at
            // 30.5 s on the bilingual clip), so deltas are pushed into the attributor's own character
            // timeline and tagged against the turn list whenever it updates - see fusion.h.
            if (cfg_.timing != 1) fusion_.push_delta(delta, charged_upto, horizon);
            if (prof_on) stats_.prof_pushdelta_s += now_s() - pd0;
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

    // Diarizer-only tail: the ASR never sees it, so text cannot shift from this side. Whether the DIAR side
    // stays identical (turn padding near the end) is exactly what the byte-identity gate decides.
    if (cfg_.diar_tail_ms > 0 && !cfg_.skip_diar) {
        std::vector<float> zeros(size_t(cfg_.diar_tail_ms) * rate / 1000, 0.0f);
        const double d0 = now_s();
        audiocpp_event* ev = nullptr;
        audiocpp_status st = audiocpp_stream_push((audiocpp_session*)session_, zeros.data(), (int)zeros.size(),
                                                 rate, 1, (int64_t)total, &ev);
        if (st != AUDIOCPP_OK) { err = std::string("diar tail push: ") + audiocpp_last_error(); return false; }
        std::vector<Turn> snapshot;
        harvest_turns(ev ? audiocpp_event_as_result(ev) : nullptr, snapshot);
        if (ev) audiocpp_event_free(ev);
        for (;;) {
            audiocpp_event* more = nullptr;
            if (audiocpp_stream_next_event((audiocpp_session*)session_, &more) != AUDIOCPP_OK || !more) break;
            harvest_turns(audiocpp_event_as_result(more), snapshot);
            audiocpp_event_free(more);
        }
        if (!snapshot.empty()) fusion_.update_turns(snapshot);   // turns are real; keep them
        stats_.diar_compute_s += now_s() - d0;
    }

    const double drain0 = prof_on ? now_s() : 0.0;
    // Drain the diarizer's final turns, then attribute the whole timeline against them.
    if (!cfg_.skip_diar) {
        audiocpp_result* res = nullptr;
        // The final decode also closes the turn that is still open at end-of-audio, so skipping it is a
        // diarisation change, not a pure speedup - the accuracy gate decides, not the clock.
        const bool do_finish = !cfg_.diar_no_finish;
        if (do_finish && audiocpp_stream_finish((audiocpp_session*)session_, &res) == AUDIOCPP_OK && res) {
            std::vector<Turn> snapshot;
            harvest_turns(res, snapshot);
            if (!snapshot.empty()) fusion_.update_turns(snapshot);
            audiocpp_result_free(res);
        }
    }

    if (prof_on) stats_.prof_drain_s = now_s() - drain0;
    const double attr0 = prof_on ? now_s() : 0.0;
    attribute(on_segment, true);
    if (prof_on) {
        stats_.prof_attrfinal_s = now_s() - attr0;
        std::fprintf(stderr, "[prof] push_delta %.2f  final_drain %.2f  final_attribute %.2f  (wall %.2f,"
                     " asr %.2f, diar %.2f) s\n", stats_.prof_pushdelta_s, stats_.prof_drain_s,
                     stats_.prof_attrfinal_s, stats_.wall_s, stats_.asr_compute_s, stats_.diar_compute_s);
    }

    stats_.audio_s = double(total) / rate;
    stats_.wall_s = now_s() - t0;
    {   // getrusage(RUSAGE_SELF) counts every thread, so cores = cpu/wall is honest.
        struct rusage ru;
        if (getrusage(RUSAGE_SELF, &ru) == 0)
            stats_.cpu_s = double(ru.ru_utime.tv_sec) + ru.ru_utime.tv_usec / 1e6
                         + double(ru.ru_stime.tv_sec) + ru.ru_stime.tv_usec / 1e6;
    }
    stats_.turns = fusion_.turns().size();
    stats_.first_turn_audio_s = first_turn_audio_;
#ifdef NEMO_HAVE_TOKEN_TIMES
    { const int32_t* id; const int64_t* fr; int nn = 0;
      if (asr_stream_ && xasr_stream_token_times((xasr_stream*)asr_stream_, &id, &fr, &nn) == 0) stats_.tokens = (size_t)nn; }
#endif
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
