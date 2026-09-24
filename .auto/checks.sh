#!/bin/bash
# checks.sh - the accuracy contract. RTF gains that change what the model says are not gains, they are a
# different product, so this runs after every passing benchmark and blocks the keep.
#
# Criterion 1 is BYTE-IDENTITY of transcripts on four clips, against .auto/bless.txt. Byte-identity is
# strictly stronger than "WER within tolerance": a kernel, a quantization, a resampler or a decode-order
# change can hold average WER while swapping which words come out. When a change legitimately alters output
# (it will: int8 kernels, chunk geometry), the loop must run the full paired validation (gate40 + the two
# holdout sets + discordant-token counts) and re-bless by editing .auto/bless.txt - NOT loosen a threshold.
#
# Criterion 2 is WER ceilings per clip, as a backstop for when output legitimately changes.
#
# Criterion 3 prints attribution and diarization DER as information. Detection knobs (speaker_threshold,
# speaker_pad_frames) are FROZEN at the shipped defaults - tuning them against these clips would turn the
# accuracy gate into an optimization target, which is exactly the overfit this session must avoid.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT" || exit 1
ARCH=${ARCH:-../VibeASR.cpp}
EV=${EV:-../eval-bilingual}
SCORER=$ARCH/.auto/score_stream.py
TURN_SCORER=$ARCH/.auto/nemo/score_turns.py
XM=../models/x-asr-zh-en-q8_0.gguf
DM=../models/nemotron-3-diarization-q8_0.gguf
BIN=build/nemo-x-asr-diarizer

[ -x "$BIN" ] || [ -n "$(find src -newer "$BIN" 2>/dev/null | head -1)" ] && { bash scripts/build_host.sh > /tmp/ar_host.log 2>&1 || { echo "HOST BUILD FAILED"; tail -15 /tmp/ar_host.log; exit 1; }; }
[ -x "$BIN" ] || { echo "no host binary"; exit 1; }

run() { # run <path> <label> <manifest>
  local out="/tmp/ar_chk_$2.txt"
  "$BIN" --audio "$1" --windows --xasr-model "$XM" --diar-model "$DM" --out "$out" > /dev/null 2>&1 \
    || { echo "FAIL: run on $2"; return 1; }
  local wer="nan"
  [ -f "$SCORER" ] && [ -n "$3" ] && wer=$(python3 "$SCORER" "$out" "$3" 2>/dev/null | grep -oE "WER +[0-9.]+" | grep -oE "[0-9.]+" | head -1)
  [ -z "$wer" ] && wer="nan"
  echo "WER $2=$wer"
  echo "$2=$wer" >> /tmp/ar_chk_wer.txt
  echo "$out" >> /tmp/ar_chk_list
  return 0
}

: > /tmp/ar_chk_list; : > /tmp/ar_chk_wer.txt
# 24 kHz files from eval-bilingual on purpose: same bytes the 1.5B baseline is fed, so the in-process
# resampler is part of what this gate protects.
run "$EV/gate_ms_v2.wav"        gate_ms_v2 "$EV/manifest_ms_v2.json"      || exit 1
run "$EV/holdout_en.wav"        holdout_en "$EV/manifest_holdout_en.json" || exit 1
run "$EV/holdout_zh.wav"        holdout_zh "$EV/manifest_holdout_zh.json" || exit 1
run "$EV/publish/bilingual_multispk_57s.wav" multispk57 ""                || exit 1

NOW=$(cat $(cat /tmp/ar_chk_list) | sha256sum | cut -c1-16)
echo "TRANSCRIPT_HASH $NOW"

# Attribution on the multispeaker gate, as information (the scorer's speaker tag is a separate axis from WER).
if [ -f "$SCORER" ]; then
  ATT=$(python3 "$SCORER" /tmp/ar_chk_gate_ms_v2.txt "$EV/manifest_ms_v2.json" 2>/dev/null | grep -oE "attribution +[0-9.]+ of [0-9]+" | head -1)
  echo "ATTRIBUTION $ATT"
fi
if [ -f "$TURN_SCORER" ]; then
  "$BIN" --audio "$EV/publish/bilingual_multispk_57s.wav" --turns-out /tmp/ar_turns.json \
         --xasr-model "$XM" --diar-model "$DM" --out /dev/null > /dev/null 2>&1
  echo "DER $(python3 "$TURN_SCORER" --gt "$EV/publish/turns.tsv" --turns run=/tmp/ar_turns.json 2>/dev/null | tail -1 | grep -oE "DER-lite +[0-9.]+%" | head -1)"
fi

# --- blessed criteria: byte-identity first, then per-clip WER ceilings
if [ -f .auto/bless.txt ]; then
  BH=$(grep -oE '^hash=[0-9a-f]+' .auto/bless.txt | cut -d= -f2)
  if [ -n "$BH" ] && [ "$BH" != "$NOW" ]; then
    echo "FAIL: transcript bytes changed (blessed $BH, now $NOW)."
    echo "      Byte-identity IS the accuracy contract here. A kernel, quantization, resampler or decode-order"
    echo "      change can hold average WER while swapping which words come out. If this change is meant to"
    echo "      alter output: run the full paired validation (gate40 + holdout_en + holdout_zh with"
    echo "      discordant-token counts), then re-bless .auto/bless.txt deliberately. Never widen a threshold"
    echo "      to make this check pass."
    exit 1
  fi
  fail=0
  while IFS='=' read -r name lim; do
    case "$name" in wer_*) ;; *) continue;; esac
    clip=${name#wer_}
    got=$(grep -oE "^$clip=[0-9.]+" /tmp/ar_chk_wer.txt | cut -d= -f2)
    [ -z "$got" ] || [ "$got" = "nan" ] && continue          # clip absent from this run: hash gate covers it
    if python3 -c "import sys;sys.exit(0 if float('$got') > float('$lim') else 1)"; then
      echo "FAIL: WER $clip=$got exceeds blessed ceiling $lim"; fail=1
    fi
  done < .auto/bless.txt
  [ $fail = 1 ] && exit 1
fi
exit 0
