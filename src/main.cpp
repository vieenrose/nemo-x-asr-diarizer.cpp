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
        "  --no-asr / --no-diar  run one half (diar-only still emits turn timeline; asr-only tags nothing)\n"
        "  --live                print segments as they close, to stderr (latency demonstration)\n"
        "  --json                print telemetry JSON to stdout\n"
        "  --turns-out PATH      dump the diar turn timeline as JSON\n"
        "  --out PATH            write the tagged transcript here\n", p);
}

int main(int argc, char** argv) {
    Config cfg;
    bool json = false, live = false, turns_out = false;
    std::string turns_path, out_path;
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
        else if (a == "--char-dur-ms") cfg.char_dur_ms = atof(next("--char-dur-ms"));
        else if (a == "--gap-snap-ms") cfg.gap_snap_ms = atof(next("--gap-snap-ms"));
        else if (a == "--paced") cfg.paced = true;
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
    for (const Segment& s : segs) {
        std::fprintf(sink, "[%d/%zu]\n Speaker %d:%s\n", s.index, segs.size(), s.speaker, s.text.c_str());
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
        std::printf(" \"segments\": %zu, \"turns\": %zu, \"speakers\": %zu, \"unattributed_chars\": %zu,\n",
                    st.segments, st.turns, st.speakers, st.unattributed_chars);
        std::printf(" \"snapped_chars\": %zu,\n", st.snapped_chars);
        std::printf(" \"text\": \"%s\"\n}\n", eng.transcript().c_str());
    } else {
        std::printf("\n[stats] audio %.2fs wall %.2fs rtf %.4f  asr %.2fs + diar %.2fs  first partial %.3fs  "
                    "p95 piece %.1fms  peak RSS %.0fMB\n",
                    st.audio_s, st.wall_s, st.wall_s / st.audio_s, st.asr_compute_s, st.diar_compute_s,
                    st.first_partial_s, st.piece_p95_ms, st.peak_rss_mb);
        std::printf("[diar]    %zu turns, %zu speakers; %zu segments, %zu unattributed, %zu by proximity fill; first turn at "
                    "audio %.2fs (attribution floor)\n",
                    st.turns, st.speakers, st.segments, st.unattributed_chars, st.snapped_chars,
                    st.first_turn_audio_s);
    }
    return 0;
}
