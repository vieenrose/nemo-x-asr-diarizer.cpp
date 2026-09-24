// fusion.cpp - see fusion.h. Written for this repo.
#include "fusion.h"

#include <algorithm>
#include <cmath>

namespace nemo {

std::vector<std::pair<size_t, size_t>> codepoints(const std::string& s) {
    std::vector<std::pair<size_t, size_t>> out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        size_t len = 1;
        if (c >= 0xF0) len = 4;
        else if (c >= 0xE0) len = 3;
        else if (c >= 0xC0) len = 2;
        if (i + len > s.size()) len = 1;      // truncated tail: treat as one unit, never drop bytes
        out.emplace_back(i, len);
        i += len;
    }
    return out;
}

void Fusion::update_turns(const std::vector<Turn>& snapshot) {
    // Match tolerance: two diar frames. A streaming turn is re-stated with the same start and a
    // later end; anything else (a new turn, a re-segmentation) is a new interval.
    const int64_t tol = rate_ / 100;
    for (const Turn& t : snapshot) {
        auto it = std::find_if(turns_.begin(), turns_.end(),
                               [&](const Turn& e) { return std::abs(e.start - t.start) <= tol; });
        if (it == turns_.end()) {
            turns_.push_back(t);
        } else {
            it->end = std::max(it->end, t.end);
            it->confidence = t.confidence;
            if (!t.speaker.empty()) it->speaker = t.speaker;
        }
    }
    std::sort(turns_.begin(), turns_.end(), [](const Turn& a, const Turn& b) { return a.start < b.start; });
}

const Turn* Fusion::covering(int64_t t, int64_t dur, bool* snapped) const {
    if (snapped) *snapped = false;
#ifdef NEMO_PLANT_FAULT
    // Test-fault hook: proves tests/selftest.sh can tell a working attributor from one that always
    // answers with the first turn. Compiled only with -DNEMO_PLANT_FAULT; the self-test asserts that
    // this build FAILS, so an assertion that never fires cannot pass as coverage.
    return turns_.empty() ? nullptr : &turns_.front();
#endif
    const Turn* best = nullptr;
    int64_t best_ov = 0;
    for (const Turn& e : turns_) {
        const int64_t lo = std::max<int64_t>(e.start, t);
        const int64_t hi = std::min<int64_t>(e.end, t + dur);
        const int64_t ov = hi - lo;
        if (ov > best_ov) { best_ov = ov; best = &e; }
    }
    if (best) return best;

    // No overlap: inside a between-turns gap. Snap to the nearer end of the closest turn rather than
    // inventing an unattributed island, but only for gaps short enough that a word really could not
    // have been spoken there.
    const bool infinite = gap_snap_s_ <= 0.0;      // 0 = always nearest, never leave text untagged
    const int64_t snap = infinite ? (int64_t)1e18 : (int64_t)(gap_snap_s_ * rate_);
    int64_t best_d = snap + 1;
    for (const Turn& e : turns_) {
        const int64_t d = std::min(std::abs(t - e.end), std::abs(e.start - (t + dur)));
        if (d < best_d) { best_d = d; best = &e; }
    }
    if (best_d <= snap && best) { if (snapped) *snapped = true; return best; }
    return nullptr;
}

std::vector<TaggedPiece> Fusion::on_delta(const std::string& delta, int64_t span_start, int64_t span_end) {
    std::vector<TaggedPiece> pieces;
    const auto cps = codepoints(delta);
    if (cps.empty()) return pieces;

    // The delta describes audio up to (now - latency). Clamp: early in the stream that horizon is
    // still negative, and the first characters have to be attributed to something - they go to the
    // first turn, which is what the diarizer has committed to by then.
    if (span_end <= span_start) span_end = span_start + 1;
    // Right-align: the words came out at the END of that span (see fusion.h).
    const int64_t est = std::min<int64_t>(span_end - span_start,
                                          (int64_t)(cps.size() * char_dur_s_ * rate_));
    const int64_t base = span_end - est;
    const double step = double(est) / double(cps.size());

    for (size_t i = 0; i < cps.size(); i++) {
        const int64_t at = base + (int64_t)std::llround(i * step);
        const int64_t dur = std::max<int64_t>(1, (int64_t)std::llround(step));
        bool snapped = false;
        const Turn* tn = covering(at, dur, &snapped);
        TaggedPiece* cur = pieces.empty() ? nullptr : &pieces.back();
        if (cur && (cur->speaker == (tn ? tn->speaker : std::string())) && cur->snapped == snapped) {
            cur->text.append(delta, cps[i].first, cps[i].second);
            cur->end_s = double(at + dur) / rate_;
            if (tn) cur->min_confidence = std::min(cur->min_confidence, tn->confidence);
        } else {
            TaggedPiece p;
            p.speaker = tn ? tn->speaker : std::string();
            p.text = delta.substr(cps[i].first, cps[i].second);
            p.start_s = double(at) / rate_;
            p.end_s = double(at + dur) / rate_;
            p.min_confidence = tn ? tn->confidence : 0.0f;
            p.snapped = snapped;
            pieces.push_back(std::move(p));
        }
    }
    return pieces;
}

}  // namespace nemo
