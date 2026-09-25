#!/bin/bash
# Paired accuracy validation for changes that legitimately move the output (quantisation, weight type,
# decode geometry). checks.sh is the byte-identity GATE; this is the evidence you bring when the gate is
# allowed to move. Two modes:
#
#   .auto/validate.sh --xasr ../models/x-asr-zh-en-q8_0.gguf --tag ref     # run a variant, save transcripts
#   .auto/validate.sh --compare ref cand                                  # WER deltas + discordance + McNemar
#
# The clip set is everything in eval-bilingual that has a manifest (~1720 s, both languages, two sample
# rates, one control clip). The watchdog sets (holdout_*) are scored but must never be tuned against.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT" || exit 1
EV=${EV:-../eval-bilingual}
SCORER=${SCORER:-../VibeASR.cpp/.auto/score_stream.py}
BIN=build/nemo-x-asr-diarizer
DM=../models/nemotron-3-diarization-q8_0.gguf
DIR=/tmp/val
XM="" TAG="" CMP1="" CMP2=""
while [ $# -gt 0 ]; do case $1 in
  --xasr) XM=$2; shift 2;; --diar) DM=$2; shift 2;; --tag) TAG=$2; shift 2;;
  --compare) CMP1=$2; CMP2=$3; shift 3;; *) echo "unknown $1"; exit 2;; esac; done

CLIPS="gate_ms gate_ms_g100 gate_ms_g1000 gate_ms_v2 control_ls holdout_en holdout_en_aligned holdout_zh holdout_zh2 holdout_zh_aligned holdout_zh_ph35200"
MAN() { case $1 in
  gate_ms) echo "$EV/manifest_ms.json";; gate_ms_g100) echo "$EV/manifest_ms_g100.json";;
  gate_ms_g1000) echo "$EV/manifest_ms_g1000.json";; gate_ms_v2) echo "$EV/manifest_ms_v2.json";;
  control_ls) echo "$EV/manifest_control_ls.json";; holdout_en) echo "$EV/manifest_holdout_en.json";;
  holdout_en_aligned) echo "$EV/manifest_holdout_en_aligned.json";; holdout_zh) echo "$EV/manifest_holdout_zh.json";;
  holdout_zh2) echo "$EV/manifest_holdout_zh2.json";; holdout_zh_aligned) echo "$EV/manifest_holdout_zh_aligned.json";;
  holdout_zh_ph35200) echo "$EV/manifest_holdout_zh_ph35200.json";; esac; }
WAV() { case $1 in *aligned|*ph35200) echo "$EV/$1.wav";; gate_ms*) echo "$EV/$1.wav";; *) echo "$EV/$1.wav";; esac; }

if [ -n "$CMP1" ]; then
  python3 - "$CMP1" "$CMP2" "$DIR" "$SCORER" <<'PY'
import sys, os, math, re, subprocess

tag1, tag2, d, scorer = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

# index.tsv is tag/clip/manifest; take the union of the two tags' clips, in a stable order.
index = {}
with open(os.path.join(d, "index.tsv")) as fh:
    for line in fh:
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 3:
            index[(parts[0], parts[1])] = parts[2]
clips = sorted({c for (t, c) in index if t in (tag1, tag2)})

# The scorer prints "WER  0.1765 (S=6 D=8 I=1 H=71) over 85 ref tokens". The counts are what a micro-average
# needs. An earlier version of this script looked for "N <digits>", never matched, swallowed the exception
# in its except branch and reported every clip as "(missing)" - which is why two comparisons in this session
# had to be redone by hand.
def score(path, manifest):
    if not os.path.exists(path):
        return None
    try:
        out = subprocess.run(["python3", scorer, path, manifest],
                             capture_output=True, text=True, timeout=600).stdout
    except Exception:
        return None
    w = re.search(r"WER\s+([0-9.]+)", out)
    e = re.search(r"\(S=(\d+)\s+D=(\d+)\s+I=(\d+)\s+H=(\d+)\)", out)
    if not w or not e:
        return None
    s, de, i, h = (int(x) for x in e.groups())
    return s, de, i, h

def wer_of(counts):
    s, de, i, h = counts
    return (s + de + i) / float(s + de + i + h) if (s + de + i + h) else 0.0

# Text with the window counters and speaker labels removed, and with segment joins collapsed, so a change of
# SPEAKER TAG or of where a line splits is not counted as a change of words. The raw windowed output
# interleaves both; an earlier version compared it raw and reported 74 "edits" on a clip whose text was
# character-identical.
LABEL = re.compile(r"^\s*(?:\[\d+/\d+\]\s*)?(?:speaker\s+)?[A-Za-z]?\d*\s*[:：]\s*", re.I)
def normalized_text(path):
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        line = re.sub(r"^\s*\[\d+/\d+\]\s*", "", line)
        out.append(LABEL.sub("", line).strip())
    return "".join(out)

def labels(path):
    got = []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        m = re.match(r"^\s*(?:\[\d+/\d+\]\s*)?(?:speaker\s+)?([A-Za-z]?\d*)\s*[:：]", line, re.I)
        got.append(m.group(1) if m else "?")
    return got

def word_edits(a, b):
    wa, wb = a.split(), b.split()
    if len(wa) * len(wb) > 4000000:
        return -1
    prev = list(range(len(wb) + 1))
    for i, x in enumerate(wa, 1):
        cur = [i] + [0] * len(wb)
        for j, y in enumerate(wb, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]

print("# WER columns are recomputed from the scorer's S/D/I/H, so they are internally consistent but may")
print("# differ in the third decimal from the scorer's own printed WER and from .auto/bless.txt. Judge the")
print("# DELTA and the text column, not the absolute value.")
print("%-24s %8s %8s %8s  %-11s %-14s %s" % ("clip", tag1, tag2, "delta", "text", "labels", "word edits"))
acc = {tag1: [0, 0, 0, 0], tag2: [0, 0, 0, 0]}
worse = better = same = text_same = scored = 0
for clip in clips:
    a = score(os.path.join(d, tag1, clip + ".txt"), index.get((tag1, clip), ""))
    b = score(os.path.join(d, tag2, clip + ".txt"), index.get((tag2, clip), ""))
    if not a or not b:
        print("%-24s   (unscored)" % clip)
        continue
    scored += 1
    for j in range(4):
        acc[tag1][j] += a[j]
        acc[tag2][j] += b[j]
    pa = os.path.join(d, tag1, clip + ".txt")
    pb = os.path.join(d, tag2, clip + ".txt")
    ta, tb = normalized_text(pa), normalized_text(pb)
    same_text = ta == tb
    text_same += 1 if same_text else 0
    la, lb = labels(pa), labels(pb)
    churn = sum(1 for x, y in zip(la, lb) if x != y) + abs(len(la) - len(lb))
    we = word_edits(ta, tb)
    wa_, wb_ = wer_of(a), wer_of(b)
    print("%-24s %8.4f %8.4f %+8.4f  %-11s %-14s %s" % (
        clip, wa_, wb_, wb_ - wa_,
        "identical" if same_text else "DIFFERS",
        "%d of %d" % (churn, max(len(la), len(lb))),
        we if we >= 0 else "n/a"))
    if wb_ > wa_ + 1e-9:
        worse += 1
    elif wb_ < wa_ - 1e-9:
        better += 1
    else:
        same += 1

for tag in (tag1, tag2):
    s, de, i, h = acc[tag]
    print("micro %-10s WER %.4f   S=%d D=%d I=%d H=%d   (n=%d ref tokens)" % (tag, wer_of(acc[tag]), s, de, i, h, s + de + i + h))
print("text identical (labels and segment joins removed) on %d of %d scored clips" % (text_same, scored))
# Sign test over clips, discarding ties. With no clip moving in either direction there are no discordant
# pairs and the test is vacuous: it must report p=1.0, not the p=0.001 an earlier version printed for an
# all-equal result by including the ties in n.
n = worse + better
if n:
    k = min(worse, better)
    p = min(1.0, 2.0 * sum(math.comb(n, j) for j in range(k + 1)) / float(2 ** n))
    verdict = "two-sided sign test p=%.3f" % p
else:
    verdict = "no clip moved in either direction (sign test vacuous)"
print("clips: %d worse, %d better, %d equal  (%s)" % (worse, better, same, verdict))
print()
print("Reading this: 'text identical' with label churn means the candidate moved turn boundaries or speaker")
print("tags, not words. Only 'DIFFERS' in the text column, or a WER delta, is an accuracy change - and a")
print("candidate that fails the gate's hash but lands here as text-identical needs a deliberate re-bless,")
print("not a widened threshold.")
PY
  exit 0
fi

[ -n "$XM" ] && [ -n "$TAG" ] || { echo "usage: validate.sh --xasr PATH --tag NAME   |   --compare A B"; exit 2; }
[ -x "$BIN" ] || bash scripts/build_host.sh >/tmp/ar_host.log 2>&1
mkdir -p "$DIR/$TAG"; : > "$DIR/index.tsv"
for c in $CLIPS; do
  m=$(MAN "$c"); w=$(WAV "$c")
  [ -f "$w" ] || { echo "skip $c (no wav)"; continue; }
  [ -f "$m" ] || { echo "skip $c (no manifest)"; continue; }
  printf '%-24s ' "$c"
  timeout 900 "$BIN" --audio "$w" --windows ${EXTRA:-} --xasr-model "$XM" --diar-model "$DM" --out "$DIR/$TAG/$c.txt" >/dev/null 2>&1 \
    || { echo "FAIL"; continue; }
  python3 "$SCORER" "$DIR/$TAG/$c.txt" "$m" 2>/dev/null | grep -oE "WER +[0-9.]+" | head -1 | tr -d '\n'
  echo "   $(basename "$w")"
done
# keep every tag's clip->manifest mapping so --compare can resolve them
for t in $(ls "$DIR" 2>/dev/null); do for c in $(ls "$DIR/$t" 2>/dev/null); do printf '%s\t%s\t%s\n' "$t" "${c%.txt}" "$(MAN "${c%.txt}")" >> "$DIR/index.tsv"; done; done
sort -u -o "$DIR/index.tsv" "$DIR/index.tsv"
echo "tagged $TAG into $DIR/$TAG"
