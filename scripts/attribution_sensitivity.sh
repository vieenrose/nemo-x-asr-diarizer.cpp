#!/usr/bin/env bash
# attribution_sensitivity.sh - how much does attribution depend on the INFERRED character timing?
#
# x-asr returns text, not tokens with timestamps, so fusion.h places each delta's characters inside the
# audio span it decodes (right-aligned, one character per --char-dur-ms) and then tags them. The question
# this answers, with numbers instead of an argument: if that placement were better - real token times -
# how much could attribution actually improve?
#
# Method: the ASR text is identical across these runs (same model, same audio, greedy decode), so the only
# thing a knob changes is WHERE the characters are placed in time. Sweeping the placement model over a
# 10x range and watching the attribution column therefore bounds what exact timestamps could buy. Flat
# column => timestamps are not the binding constraint. Moving column => they are, and the best cell is
# close to what a token-time path would reach.
#
# Requires: the composite built, eval-bilingual next to the repo, and score_stream.py from the archive.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BIN=${BIN:-$ROOT/build/nemo-x-asr-diarizer}
SCORE=${SCORE:-$ROOT/../VibeASR.cpp/.auto/score_stream.py}
AUDIO=${AUDIO:-$ROOT/../eval-bilingual/gate_ms_v2.wav}
MANIFEST=${MANIFEST:-$ROOT/../eval-bilingual/manifest_ms_v2.json}
XASR=${XASR:-$ROOT/models/x-asr-zh-en-q8_0.gguf}
DIAR=${DIAR:-$ROOT/models/nemotron-3-diarization-q8_0.gguf}

[ -x "$BIN" ] || { echo "ERROR: no binary at $BIN (run scripts/build_host.sh)" >&2; exit 1; }
[ -f "$AUDIO" ] || { echo "ERROR: $AUDIO missing" >&2; exit 1; }
W16=${W16:-/tmp/probe_16k.wav}
[ -f "$W16" ] || ffmpeg -v error -y -i "$AUDIO" -ar 16000 -ac 1 -c:a pcm_s16le "$W16"

printf '%-10s %-12s %-10s | %-8s %-8s %-9s %-8s %s\n' char_dur gap_snap latency wer attrib cover consist snapped
for cd in 30 90 300; do
  for gs in 400 0; do
    for lat in 160 480 960; do
      out=/tmp/probe_tagged.txt
      j=$("$BIN" --audio "$W16" --xasr-model "$XASR" --diar-model "$DIAR" \
              --char-dur-ms "$cd" --gap-snap-ms "$gs" --asr-latency-ms "$lat" --json --out "$out" 2>/dev/null)
      sc=$(python3 "$SCORE" "$out" "$MANIFEST" 2>/dev/null || true)
      parse() { printf '%s' "$sc" | grep -oE "$1 +[0-9.]+" | head -1 | awk '{print $2}'; }
      wer=$(parse "WER"); att=$(parse "attribution"); cov=$(parse "coverage"); con=$(parse "consistency")
      sn=$(printf '%s' "$j" | python3 -c "import json,sys;print(json.load(sys.stdin)['snapped_chars'])")
      printf '%-10s %-12s %-10s | %-8s %-8s %-9s %-8s %s\n' "${cd}ms" "${gs}ms" "${lat}ms" "$wer" "$att" "$cov" "$con" "$sn"
    done
  done
done
echo
echo "Read it as: attribution is the answer to 'did the right voice get the word', coverage is the share of"
echo "reference tokens carrying any tag, consistency is one voice keeping one label, snapped counts the"
echo "characters tagged by proximity because no turn covered their audio at all."
