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
import sys, os, math, subprocess, re
tag1, tag2, d, scorer = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
clips = os.environ.get("CLIPS","").split() or []
def wer_of(t, man):
    if not os.path.exists(t): return None
    try:
        out = subprocess.run(["python3", scorer, t, man], capture_output=True, text=True, timeout=300).stdout
        m = re.search(r"WER\s+([0-9.]+)", out);  n = re.search(r"N\s+(\d+)", out)
        return (float(m.group(1)), int(n.group(1))) if m else (None, None)
    except Exception: return (None, None)
rows=[]; tot_w=[0.0,0.0]; b0=b1=0
print("%-24s %8s %8s %8s %10s" % ("clip","WER(ref)","WER(cand)","delta","discordant"))
for line in open(os.path.join(d, "index.tsv")):
    tag, clip, man = line.rstrip("\n").split("\t")
    if tag not in (tag1, tag2): continue
    a = wer_of(os.path.join(d, tag1, clip+".txt"), man); b = wer_of(os.path.join(d, tag2, clip+".txt"), man)
    if a[0] is None or b[0] is None: print("%-24s   (missing)" % clip); continue
    ha = open(os.path.join(d, tag1, clip+".txt")).read(); hb = open(os.path.join(d, tag2, clip+".txt")).read()
    wa, wb = ha.split(), hb.split()
    # word-level DP over the two hypotheses: count positions where they disagree with each other
    n, m = len(wa), len(wb)
    if n*m < 4_000_000:
        D = [[0]*(m+1) for _ in range(n+1)]
        for i in range(n+1): D[i][0]=i
        for j in range(m+1): D[0][j]=j
        for i in range(1, n+1):
            Di, Dp = D[i], D[i-1]
            for j in range(1, m+1):
                Di[j] = min(Dp[j]+1, Di[j-1]+1, Dp[j-1] + (wa[i-1] != wb[j-1]))
        disc = D[n][m]
    else: disc = -1
    rows.append((clip, a[0], b[0], a[1], b[1], disc))
    tot_w[0] += a[0]*a[1]; tot_w[1] += b[0]*b[1]
    b0 += a[1]; b1 += b[1]
    print("%-24s %8.4f %8.4f %+8.4f %10s" % (clip, a[0], b[0], b[0]-a[0], disc if disc>=0 else "n/a"))
W0 = tot_w[0]/b0 if b0 else 0; W1 = tot_w[1]/b1 if b1 else 0
print("\nmicro- aggregate WER  %s=%.4f  %s=%.4f  delta=%+.4f  (n=%d words)" % (tag1, W0, tag2, W1, W1-W0, b0))
# McNemar needs per-item correctness, which the scorer does not emit; report the sign test over clips and
# the absolute WER movement as the honest floor of what can be claimed.
worse = sum(1 for r in rows if r[2] > r[1] + 1e-9); better = sum(1 for r in rows if r[2] < r[1] - 1e-9)
print("clips worse %d, better %d, equal %d  (sign test p=%.3f)" % (
    worse, better, len(rows)-worse-better,
    sum(math.comb(len(rows),k) for k in range(min(worse,better), max(worse,better)+1))/2**len(rows)*2 if rows else 1))
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
  timeout 900 "$BIN" --audio "$w" --windows --xasr-model "$XM" --diar-model "$DM" --out "$DIR/$TAG/$c.txt" >/dev/null 2>&1 \
    || { echo "FAIL"; continue; }
  python3 "$SCORER" "$DIR/$TAG/$c.txt" "$m" 2>/dev/null | grep -oE "WER +[0-9.]+" | head -1 | tr -d '\n'
  echo "   $(basename "$w")"
done
# keep every tag's clip->manifest mapping so --compare can resolve them
for t in $(ls "$DIR" 2>/dev/null); do for c in $(ls "$DIR/$t" 2>/dev/null); do printf '%s\t%s\t%s\n' "$t" "${c%.txt}" "$(MAN "${c%.txt}")" >> "$DIR/index.tsv"; done; done
sort -u -o "$DIR/index.tsv" "$DIR/index.tsv"
echo "tagged $TAG into $DIR/$TAG"
