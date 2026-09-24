#!/usr/bin/env python3
"""Check token timestamps against ground-truth SILENCE, without consulting any score.

Why silence: an attribution score can be moved by the attributor, the scorer's mapping, or luck - on a
76-token gate one wrong token is 1.3 pp, so score-driven "improvements" here are usually noise. Silence is
not. If a token's timestamp is right, the token cannot land in a stretch of audio where the reference says
nobody is talking.

Two uses:
  * verify a build's timestamps (the `outside speech` count should be ~0)
  * derive --token-offset-ms (the decision lag): sweep --offsets and take the minimum

A clip whose reference turns OVERLAP is useless for this - the intervals then cover the whole audio and
every timestamp passes. The tool says so instead of reporting a flattering zero.
"""
import argparse, json, sys

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", required=True, help="JSON from --tokens-out")
ap.add_argument("--gt", required=True, help="manifest JSON with table[] (start_s/end_s), or turns.tsv")
ap.add_argument("--offsets", default="0,150,300,450")
a = ap.parse_args()

tok = json.load(open(a.tokens))
if a.gt.endswith(".tsv"):
    rows = [l.split("\t") for l in open(a.gt).read().splitlines()[1:]]
    iv = sorted((float(r[4]), float(r[5])) for r in rows)
else:
    man = json.load(open(a.gt))
    iv = sorted((float(t["start_s"]), float(t["end_s"])) for t in man["table"])
speech = sum(e - s for s, e in iv)
span = max(e for _, e in iv) - min(s for s, _ in iv)
gaps = [(iv[i][1], iv[i + 1][0]) for i in range(len(iv) - 1) if iv[i + 1][0] > iv[i][1]]
print(f"{len(tok)} tokens | reference speech {speech:.1f}s over {span:.1f}s audio | {len(gaps)} silent gaps")
if speech > 1.05 * span:
    print("INCONCLUSIVE: reference turns overlap (speech > audio), silence cannot discriminate anything.")
    sys.exit(0)

for off in [float(x) for x in a.offsets.split(",")]:
    o = off / 1000.0
    bad = [t for t in tok if not any(s - 0.06 <= t["t_s"] - o <= e + 0.06 for s, e in iv)]
    ingap = sum(1 for t in bad if any(s <= t["t_s"] - o <= e for s, e in gaps))
    print(f"  offset {off:6.0f} ms: {len(bad):3d}/{len(tok)} tokens in silence "
          f"({100*len(bad)/len(tok):4.1f}%), {ingap} inside a real gap")
print("Pick the offset that minimises the count. That number is a property of the audio, not of a score.")
