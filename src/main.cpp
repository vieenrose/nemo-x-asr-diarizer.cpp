// main.cpp - CLI for the composite: streaming x-asr + Nemotron-3 diarization, speaker-tagged output.
//
// Output format matches the multilingual archive's streaming convention so the existing scorers
// (WER + speaker attribution) score this without a converter:
//
//   [3/7]
//    Speaker 1:text text text
//
// --json prints the same telemetry the phone harness collects, so a run here is comparable to the
// rows in the comparison table.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <map>
#include <string>
#include <vector>

#include "engine.h"

using namespace nemo;

static void usage(const char* p) {
    std::printf(
        "nemo-x-asr-diarizer - streaming ASR (x-asr) + streaming diarization (Nemotron-3), one timeline\n"
        "usage: %s --audio clip16k.wav [--xasr-model M] [--diar-model M] [options]\n"
        "  --xasr-model PATH     x-asr GGUF (default models/x-asr-zh-en-q8_0.gguf)\n"
        "  --diar-model PATH     Nemotron-3 diarization GGUF (default models/nemotron-3-diarization-q8_0.gguf)\n"
        "  --audio PATH          16 kHz mono PCM16; resample first, this tool refuses anything else\n"
        "  -t, --threads N       worker threads, keep equal to the core count you pin to (default 2)\n"
        "  --piece-ms N          audio fed per step (default 100)\n"
        "  --chunk-ms N          x-asr chunk: 160/480/960 (default 480)\n"
        "  --asr-latency-ms N    attribution lag; <0 = chunk-ms (see fusion.h)\n"
        "  --char-dur-ms N       per-character duration for time placement (default 90)\n"
        "  --gap-snap-ms N       attribution tolerance across between-turns pauses (default 400)\n"
        "  --paced               sleep to 1x wall clock, so reported latency is real-time latency\n"
        "  --diar-threshold F    diarizer detection threshold (family default 0.5; lower = more speech found)\n"
        "  --diar-opt K=V        any diarizer request option, e.g. speaker_min_frames=2 (repeatable)\n"
        "  --no-asr / --no-diar  run one half (diar-only still emits turn timeline; asr-only tags nothing)\n"
        "  --live                print segments as they close, to stderr (latency demonstration)\n"
        "  --main-affinity HEX     cpu mask for the ASR thread (e.g. c0 = cpu6-7, the A78 primes)\n"
        "  --engine-affinity HEX   cpu mask for the diarizer's worker pool (e.g. f = cpu0-3, the A55s)\n"
        "  --windows             emit the baseline's window format: [k/N] blocks every 2.93 s of\n"
        "                        audio (HOP_S = 70400/24000) instead of one block per speaker change\n"
        "  --window-ms F         window length, default 2933.333\n"
        "  --json                print telemetry JSON to stdout\n"
        "  --turns-out PATH      dump the diar turn timeline as JSON\n"
        "  --out PATH            write the tagged transcript here\n", p);
}

int main(int argc, char** argv) {
    Config cfg;
    bool json = false, live = false, turns_out = false, tokens_out = false, windowed = false;
    double window_s = 70400.0 / 24000.0;   // the archive's HOP_S, in seconds
    std::string turns_path, out_path, tokens_path;
    cfg.xasr_model = "models/x-asr-zh-en-q8_0.gguf";
    cfg.diar_model = "models/nemotron-3-diarization-q8_0.gguf";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "ERROR: %s needs a value\n", name); std::exit(2); }
            return argv[++i];
        };
        if (a == "--audio") cfg.audio = next("--audio");
        else if (a == "--xasr-model") cfg.xasr_model = next("--xasr-model");
        else if (a == "--diar-model") cfg.diar_model = next("--diar-model");
        else if (a == "-t" || a == "--threads") cfg.threads = atoi(next("--threads"));
        else if (a == "--piece-ms") cfg.piece_ms = atoi(next("--piece-ms"));
        else if (a == "--chunk-ms") cfg.chunk_ms = atoi(next("--chunk-ms"));
        else if (a == "--asr-latency-ms") cfg.asr_latency_ms = atof(next("--asr-latency-ms"));
        else if (a == "--gap-fill") {
            std::string v = next("--gap-fill");
            if (v == "nearest") cfg.gap_fill = 0; else if (v == "prev") cfg.gap_fill = 1;
            else { std::fprintf(stderr, "ERROR: --gap-fill takes nearest|prev\n"); return 2; }
        }
        else if (a == "--token-offset-ms") cfg.token_offset_ms = atof(next("--token-offset-ms"));
        else if (a == "--timing") {
            std::string v = next("--timing");
            cfg.timing = v == "tokens" ? 1 : v == "inferred" ? 2 : v == "auto" ? 0 : (std::fprintf(stderr, "ERROR: --timing takes auto|tokens|inferred\n"), 9);
            if (cfg.timing == 9) return 2;
        }
        else if (a == "--tokens-out") tokens_path = next("--tokens-out"), tokens_out = true;
        else if (a == "--main-affinity") cfg.main_affinity = std::strtol(next("--main-affinity"), nullptr, 16);
        else if (a == "--engine-affinity") cfg.engine_affinity = std::strtol(next("--engine-affinity"), nullptr, 16);
        else if (a == "--windows") windowed = true;
        else if (a == "--diar-no-finish") cfg.diar_no_finish = true;
        else if (a == "--diar-async") cfg.diar_async = true;
        else if (a == "--diar-tail-ms") cfg.diar_tail_ms = atoi(next("--diar-tail-ms"));
        else if (a == "--window-ms") window_s = atof(next("--window-ms")) / 1000.0;
        else if (a == "--char-dur-ms") cfg.char_dur_ms = atof(next("--char-dur-ms"));
        else if (a == "--gap-snap-ms") cfg.gap_snap_ms = atof(next("--gap-snap-ms"));
        else if (a == "--paced") cfg.paced = true;
        else if (a == "--diar-threshold") cfg.diar_opts.emplace_back("speaker_threshold", next("--diar-threshold"));
        else if (a == "--diar-opt") {
            std::string kv = next("--diar-opt");
            const size_t eq = kv.find('=');
            if (eq == std::string::npos) { std::fprintf(stderr, "ERROR: --diar-opt wants key=value\n"); return 2; }
            cfg.diar_opts.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
        }
        else if (a == "--no-asr") cfg.skip_asr = true;
        else if (a == "--no-diar") cfg.skip_diar = true;
        else if (a == "--live") live = true;
        else if (a == "--json") json = true;
        else if (a == "--turns-out") { turns_path = next("--turns-out"); turns_out = true; }
        else if (a == "--out") { out_path = next("--out"); }
        else { usage(argv[0]); std::fprintf(stderr, "ERROR: unknown argument '%s'\n", a.c_str()); return 2; }
    }
    if (cfg.audio.empty()) { usage(argv[0]); std::fprintf(stderr, "ERROR: --audio is required\n"); return 2; }
    if (cfg.skip_asr && cfg.skip_diar) { std::fprintf(stderr, "ERROR: --no-asr and --no-diar together runs nothing\n"); return 2; }

    cfg.live_provisional = live;      // set BEFORE the Engine copies cfg; revisions print on turn updates
    Engine eng(cfg);
    std::string err;
    if (!eng.init(err)) { std::fprintf(stderr, "ERROR: init: %s\n", err.c_str()); return 1; }

    std::vector<Segment> segs;
    if (!eng.run([&](const Segment& s) {
            segs.push_back(s);
        }, err)) {
        std::fprintf(stderr, "ERROR: run: %s\n", err.c_str());
        return 1;
    }

    // stdout holds ONE thing: the transcript, or the JSON. With --json and no --out the transcript goes
    // to stderr, because a telemetry flag that produces unparsable JSON is worse than no flag.
    FILE* sink = !out_path.empty() ? std::fopen(out_path.c_str(), "w") : (json ? stderr : stdout);
    if (!sink) { std::fprintf(stderr, "ERROR: cannot write %s\n", out_path.c_str()); return 1; }
    // Two output shapes over the same attribution. Default = one block per speaker change (what a product
    // wants). --windows = one block per HOP_S of audio, tagged with the speaker(s) inside it, which is what
    // the autoresearch harness and score_stream.py consume - so "drop-in" is checkable by running the
    // archive's own tools against this binary, not by trusting a claim in a README.
    //
    // Window boundaries come from the model's token times when they exist. Otherwise text is spread
    // uniformly across the segment's span, which is coarse (a segment can run tens of seconds) and is
    // reported as such by the window-onset diagnostic - it does not silently pretend to be exact.
    if (windowed) {
        std::map<std::string, int> spk_label;   // diarizer label -> the small int the segments already use
        for (const Segment& sg : segs) spk_label[sg.speaker_id] = sg.speaker;
        auto label_of = [&](const std::string& id) -> int {
            const auto it = spk_label.find(id);
            return it == spk_label.end() ? -1 : it->second;
        };
        const double hop = window_s > 0 ? window_s : 2.933333;
        int nwin = 1;
        for (const Segment& s : segs) nwin = std::max(nwin, (int)(s.end_s / hop) + 1);
        std::vector<std::vector<std::pair<int, std::string>>> win(nwin);
        auto add = [&](int w, int lbl, const std::string& t) {
            if (w < 0) w = 0; if (w >= nwin) w = nwin - 1;
            if (!win[w].empty() && win[w].back().first == lbl) win[w].back().second += t;
            else if (!t.empty()) win[w].push_back({lbl, t});
        };

        // NEVER CUT A WORD AT A WINDOW BOUNDARY. This cost 7 points of WER before it was fixed: with the
        // same characters and the same manifest, segment output scored 0.2455 and window output 0.3136 on
        // the English holdout, because 27 of 55 window lines began mid-word ("that end" | "s well") and
        // the scorer tokenises per line, so one reference word became a substitution plus an insertion.
        // Concatenated text is IDENTICAL either way, which is exactly why a text-equality check cannot see
        // this - the check has to be the score, or an explicit mid-word test.
        //
        // So characters are grouped into words first, and a word goes to the window its FIRST character
        // falls in, whole. Non-ASCII (CJK) codepoints are their own unit - Chinese has no spaces to split
        // on. Punctuation and spaces attach to the word they follow.
        struct Chr { std::string c; double t; int label; };
        std::vector<Chr> chars;
        const auto toks = eng.token_table();
        if (!toks.empty()) {
            for (const TokenInfo& ti : toks) {
                const int lab = ti.speaker_id.empty() ? -1 : label_of(ti.speaker_id);
                const auto cps = codepoints(ti.text);
                const double dt = cps.empty() ? 0.0 : 0.040 / (double)cps.size();   // token spans <= 40 ms
                for (size_t i = 0; i < cps.size(); i++)
                    chars.push_back({ti.text.substr(cps[i].first, cps[i].second), ti.t_s + dt * (double)i, lab});
            }
        } else {
            // No model times: spread each segment's text over its span. Coarse, but it must still produce
            // TEXT - an output shape that silently emits nothing when timings are missing is the worst kind
            // of bug, because it looks like a working run.
            for (const Segment& sg : segs) {
                const auto cps = codepoints(sg.text);
                const double span = std::max(1e-6, sg.end_s - sg.start_s);
                for (size_t i = 0; i < cps.size(); i++)
                    chars.push_back({sg.text.substr(cps[i].first, cps[i].second),
                                     sg.start_s + span * (double)i / (double)std::max<size_t>(1, cps.size()),
                                     sg.speaker});
            }
        }

        auto is_word_char = [](unsigned char c) { return std::isalnum(c) || c == '\'' || c == '-'; };
        std::string buf;
        double buf_at = 0;
        int buf_label = -1;
        auto flush = [&]() { if (!buf.empty()) { add((int)(buf_at / hop), buf_label, buf); buf.clear(); } };
        for (const Chr& ch : chars) {
            const unsigned char u0 = (unsigned char)(ch.c.empty() ? 0 : ch.c[0]);
            if (u0 >= 0x80) { flush(); add((int)(ch.t / hop), ch.label, ch.c); continue; }   // CJK: own unit
            if (buf.empty()) { buf = ch.c; buf_at = ch.t; buf_label = ch.label; if (u0 != ' ' && !is_word_char(u0)) flush(); }
            // The separator belongs to the word it follows. Dropping it here merged every pair of words
            // into one token ("the cat" -> "thecat") and took WER from 0.25 to 0.95, which is a good reminder
            // that a formatter has to be scored, not eyeballed.
            else if (u0 == ' ') { buf += " "; flush(); }
            else if (is_word_char(u0) || is_word_char((unsigned char)buf[buf.size() - 1])) { buf += ch.c; if (!is_word_char(u0)) flush(); }
            else { flush(); buf = ch.c; buf_at = ch.t; buf_label = ch.label; }
        }
        flush();
        for (int w = 0; w < nwin; w++) {
            if (win[w].empty()) continue;                       // silent window emits nothing
            std::fprintf(sink, "[%d/%d]\n", w + 1, nwin);
            for (const auto& r : win[w]) std::fprintf(sink, " Speaker %d:%s\n", r.first, r.second.c_str());
        }
    } else {
        for (const Segment& s : segs) {
            std::fprintf(sink, "[%d/%zu]\n Speaker %d:%s\n", s.index, segs.size(), s.speaker, s.text.c_str());
        }
    }
    if (!out_path.empty()) std::fclose(sink); else std::fflush(sink);

    if (turns_out) {
        FILE* f = std::fopen(turns_path.c_str(), "w");
        if (!f) { std::fprintf(stderr, "ERROR: cannot write %s\n", turns_path.c_str()); return 1; }
        std::fprintf(f, "[");
        const auto& ts = eng.turns();
        for (size_t i = 0; i < ts.size(); i++) {
            std::fprintf(f, "%s{\"start_sample\":%lld,\"end_sample\":%lld,\"speaker_id\":\"%s\",\"confidence\":%.4f}",
                         i ? "," : "", (long long)ts[i].start, (long long)ts[i].end, ts[i].speaker.c_str(),
                         ts[i].confidence);
        }
        std::fprintf(f, "]\n");
        std::fclose(f);
    }

    if (tokens_out) {
        FILE* f = std::fopen(tokens_path.c_str(), "w");
        if (!f) { std::fprintf(stderr, "ERROR: cannot write %s\n", tokens_path.c_str()); return 1; }
        const auto toks = eng.token_table();
        std::fprintf(f, "[\n");
        for (size_t i = 0; i < toks.size(); i++) {
            std::fprintf(f, "%s  {\"i\": %zu, \"text\": \"%s\", \"t_s\": %.3f, \"speaker\": \"%s\", \"snapped\": %s}",
                         i ? ",\n" : "\n", i, toks[i].text.c_str(), toks[i].t_s, toks[i].speaker_id.c_str(),
                         toks[i].snapped ? "true" : "false");
        }
        std::fprintf(f, "\n]\n");
        std::fclose(f);
    }

    const Stats& st = eng.stats();
    if (json) {
        std::printf("{\n");
        std::printf(" \"audio\": \"%s\", \"xasr\": \"%s\", \"diar\": \"%s\",\n", cfg.audio.c_str(),
                    cfg.skip_asr ? "-" : cfg.xasr_model.c_str(), cfg.skip_diar ? "-" : cfg.diar_model.c_str());
        std::printf(" \"threads\": %d, \"piece_ms\": %d, \"chunk_ms\": %d, \"asr_latency_ms\": %.0f,\n",
                    cfg.threads, cfg.piece_ms, cfg.chunk_ms,
                    cfg.asr_latency_ms < 0 ? double(cfg.chunk_ms) : cfg.asr_latency_ms);
        std::printf(" \"audio_s\": %.3f, \"wall_s\": %.3f, \"load_s\": %.3f, \"rtf\": %.4f,\n", st.audio_s,
                    st.wall_s, st.load_s, st.wall_s / st.audio_s);
        std::printf(" \"asr_compute_s\": %.3f, \"diar_compute_s\": %.3f,\n", st.asr_compute_s, st.diar_compute_s);
        std::printf(" \"first_partial_s\": %.3f, \"first_turn_audio_s\": %.2f, \"piece_ms_p95\": %.2f, \"peak_rss_mb\": %.1f,\n",
                    st.first_partial_s, st.first_turn_audio_s, st.piece_p95_ms, st.peak_rss_mb);
        std::printf(" \"timing\": \"%s\", \"tokens\": %zu, \"token_offset_ms\": %.0f,\n",
                    st.timing_mode == 1 ? "model-timestamps" : "inferred-placement", st.tokens,
                    cfg.token_offset_ms);
        std::printf(" \"segments\": %zu, \"turns\": %zu, \"speakers\": %zu, \"unattributed_chars\": %zu,\n",
                    st.segments, st.turns, st.speakers, st.unattributed_chars);
        std::printf(" \"snapped_chars\": %zu,\n", st.snapped_chars);
        std::printf(" \"text\": \"%s\"\n}\n", eng.transcript().c_str());
    } else {
        std::printf("\n[stats] audio %.2fs wall %.2fs rtf %.4f  asr %.2fs + diar %.2fs  first partial %.3fs  "
                    "p95 piece %.1fms  peak RSS %.0fMB\n",
                    st.audio_s, st.wall_s, st.wall_s / st.audio_s, st.asr_compute_s, st.diar_compute_s,
                    st.first_partial_s, st.piece_p95_ms, st.peak_rss_mb);
        std::printf("[timing]  %s over %zu tokens\n",
                    st.timing_mode == 1 ? "model token timestamps (40 ms grid)" : "inferred placement", st.tokens);
        std::printf("[diar]    %zu turns, %zu speakers; %zu segments, %zu unattributed, %zu by proximity fill; first turn at "
                    "audio %.2fs (attribution floor)\n",
                    st.turns, st.speakers, st.segments, st.unattributed_chars, st.snapped_chars,
                    st.first_turn_audio_s);
    }
    return 0;
}
