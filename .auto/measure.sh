#!/bin/bash
# measure.sh - phone RTF for the composite (x-asr streaming + Nemotron-3 diarization, one process, CPU only).
#
# Phone-only by design: ARM gains do not transfer from x86, and this session's whole lesson so far is that a
# stable number can still come from the wrong mechanism (see prompt.md). So nothing here is decided on host.
#
# Protocol, inherited from the VibeASR autoresearch loop and non-negotiable:
#   * taskset C0 = hex mask 0xc0 = cpu6-7, the two A78 primes. Same pinning as the 1.5B baseline.
#   * ARMED. The Dimensity 1300 governor parks near 1.3 GHz unless something keeps waking the UI; an unarmed
#     run inflates RTF 20-60 % and still prints a plausible number.
#   * WITNESS. cpufreq time_in_state is sampled before/after; a mean delivered below ~1950 MHz means the run
#     was NOT armed -> the row is retried, never averaged, and after 3 attempts the script FAILS.
#   * Short clips run REPS passes back-to-back inside ONE armed window, so the governor state matches the
#     baseline's instead of measuring a cold start.
#
# Threads: 2. Never more than cores in the mask - a 4-thread run under a 2-cpu mask once read RTF 66.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT" || exit 1
DEV=${DEV:-$(adb devices | sed -n 2p | awk '{print $1}')}
D=${D:-/data/local/tmp/nemo_x}
MASK=${MASK:-C0}
THREADS=${THREADS:-2}
ARM_MIN=${ARM_MIN:-1950}
REPS=${REPS:-2}
XM=x-asr-zh-en-q8_0.gguf
DM=nemotron-3-diarization-q8_0.gguf
BIN=build-android/nemo-x-asr-diarizer
DEV_BIN=./nemo-x-asr-diarizer   # what it is called ON THE DEVICE - pushing to $D flattens the path, so the
                                # local build-android/ prefix must not leak into the remote command

[ -n "$DEV" ] || { echo "no device"; exit 1; }

# ---------------------------------------------------------------- build (incremental)
if [ ! -x "$BIN" ] || [ -n "$(find src patches -newer "$BIN" 2>/dev/null | head -1)" ]; then
  if ! bash scripts/build_android.sh > /tmp/ar_build.log 2>&1; then
    echo "BUILD FAILED:"; tail -18 /tmp/ar_build.log; exit 1
  fi
  echo "built"
fi
adb -s "$DEV" push "$BIN" "$D"/ > /dev/null 2>&1 || { echo "push failed"; exit 1; }
adb -s "$DEV" shell "chmod 777 $D/nemo-x-asr-diarizer; mkdir -p $D/out" > /dev/null 2>&1

# ---------------------------------------------------------------- device state
n=$(adb -s "$DEV" shell "ps -A -o NAME" 2>/dev/null | tr -d '\r' | grep -cE 'nemo-x-asr|asr_streaming|xasr_stream_probe|audiocpp_cli' || true)
[ "${n:-0}" != "0" ] && { echo "ERROR: $n benchmark process(es) already on device - times inflated"; exit 1; }
LOAD=$(adb -s "$DEV" shell "cat /proc/loadavg | cut -d' ' -f1" 2>/dev/null | tr -d '\r')
BATT=$(adb -s "$DEV" shell "for z in /sys/class/power_supply/battery/temp /sys/class/thermal/thermal_zone*/temp; do v=\$(cat \$z 2>/dev/null); [ -n \"\$v\" ] && echo \$v && break; done" 2>/dev/null | tr -d '\r')

ARM_PID=""; ARM_S=""
arm_start() {
  adb -s "$DEV" shell "input keyevent KEYCODE_WAKEUP" >/dev/null 2>&1 || true
  for _i in 1 2 3; do
    adb -s "$DEV" shell "input keyevent KEYCODE_VOLUME_DOWN" >/dev/null 2>&1 || true
    adb -s "$DEV" shell "input keyevent KEYCODE_VOLUME_UP"   >/dev/null 2>&1 || true
    sleep 1
  done
  ARM_S=$(mktemp /tmp/ar_arm.XXXXXX 2>/dev/null || echo "")
  if [ -n "$ARM_S" ]; then
    ( while [ -f "$ARM_S" ]; do adb -s "$DEV" shell "input keyevent KEYCODE_WAKEUP" >/dev/null 2>&1 || true; sleep 3; done ) >/dev/null 2>&1 &
    ARM_PID=$!; sleep 1
  fi
}
arm_stop() { [ -n "$ARM_S" ] && rm -f "$ARM_S"; [ -n "$ARM_PID" ] && wait "$ARM_PID" 2>/dev/null; ARM_S=""; ARM_PID=""; }
tis() { adb -s "$DEV" shell "cat /sys/devices/system/cpu/cpu7/cpufreq/stats/time_in_state" 2>/dev/null | tr -d '\r'; }

# mean delivered MHz + share of time at the 2.4 GHz step, from the time_in_state delta. Inline so this repo
# does not depend on the archive's deliv_share.py - same maths, 15 lines.
witness() {
  python3 - "$1" "$2" <<'PY'
import re, sys
def parse(t):
    out = {}
    for ln in t.splitlines():
        m = re.match(r'\s*(\d+)\s+(\d+)', ln)
        if m: out[int(m.group(1))] = int(m.group(2))
    return out
a, b = parse(open(sys.argv[1]).read()), parse(open(sys.argv[2]).read())
tot = dt = 0.0
for f in sorted(set(a) & set(b)):
    d = b[f] - a[f]
    if d > 0: tot += d * f; dt += d
# time_in_state lists FREQUENCIES IN kHz, so tot/dt is kHz. ARM_MIN is MHz, hence /1000. Getting this wrong
# is not a cosmetic bug: printed in Hz (2321926) the "mean >= 1950" test passes even for an unarmed device
# parked at 1.3 GHz (1300000 >= 1950), i.e. the guard silently stops guarding. Verified against both cases:
# armed -> ~2320 (pass), unarmed -> ~1300 (reject).
print("%.1f %.1f" % (tot / dt / 1000 if dt else 0.0, (100.0 * sum(b[f] - a[f] for f in set(a) & set(b) if f >= 2400000 and b[f] > a[f]) / dt) if dt else 0.0))
PY
}

# ---------------------------------------------------------------- one cell
AUDIT_TOT=0; WALL_TOT=0; ASR_TOT=0; DIAR_TOT=0; FP=""; P95=0; RSS=0
declare -A CLIP_RTF
run_cell() { # run_cell <wav> <reps>  -- accumulates into the totals; per-clip RTF recorded separately
  local wav=$1 reps=$2 line
  local ca=0 cw=0 casr=0 cdi=0 fp="" 
  for i in $(seq 1 "$reps"); do
    line=$(adb -s "$DEV" shell "cd $D && LD_LIBRARY_PATH=. taskset $MASK $DEV_BIN --audio wav/$wav --threads $THREADS \
            --windows --xasr-model $XM --diar-model $DM --out out/ar_${wav}_$i.txt 2>&1" | tr -d '\r' | grep '^\[stats\]')
    [ -z "$line" ] && { echo "ERROR: no [stats] from $wav pass $i"; return 1; }
    local a w asr di f p r cores
    a=$(echo "$line"   | grep -oE 'audio [0-9.]+s'         | grep -oE '[0-9.]+')
    w=$(echo "$line"   | grep -oE 'wall [0-9.]+s'          | grep -oE '[0-9.]+')
    asr=$(echo "$line" | grep -oE 'asr [0-9.]+s'           | grep -oE '[0-9.]+')
    di=$(echo "$line"  | grep -oE 'diar [0-9.]+s'          | grep -oE '[0-9.]+')
    f=$(echo "$line"   | grep -oE 'first partial [0-9.]+s' | grep -oE '[0-9.]+')
    # sed -E, not a second grep -oE: the label "p95" contains digits, so grep-over-label returns 95 AND the
    # value, and max(0, 95\n343.4) is a python SyntaxError that silently became an empty metric.
    p=$(echo "$line"   | sed -nE 's/.*p95 piece ([0-9.]+).*/\1/p')
    r=$(echo "$line"   | grep -oE 'peak RSS [0-9.]+MB'     | grep -oE '[0-9.]+')
    cores=$(echo "$line" | sed -nE 's/.*\(([0-9.]+) cores\).*/\1/p')
    for v in a w asr di f p r cores; do [ -z "${!v}" ] && { echo "ERROR: unparsed field in $line"; return 1; }; done
    ca=$(python3 -c "print($ca+$a)"); cw=$(python3 -c "print($cw+$w)")
    casr=$(python3 -c "print($casr+$asr)"); cdi=$(python3 -c "print($cdi+$di)")
    fp=$f; P95=$(python3 -c "print(max($P95,$p))"); RSS=$(python3 -c "print(max($RSS,$r))")
    CORES=$(python3 -c "print(max($CORES,$cores))")
  done
  AUDIT_TOT=$(python3 -c "print($AUDIT_TOT+$ca)"); WALL_TOT=$(python3 -c "print($WALL_TOT+$cw)")
  ASR_TOT=$(python3 -c "print($ASR_TOT+$casr)");   DIAR_TOT=$(python3 -c "print($DIAR_TOT+$cdi)")
  FP=$fp
  CLIP_RTF[$wav]=$(python3 -c "print(round($cw/$ca,4))")
  adb -s "$DEV" pull "$D/out/ar_${wav}_$reps.txt" "/tmp/ar_$wav.txt" >/dev/null 2>&1
  return 0
}

# ---------------------------------------------------------------- measure with arm screening
for attempt in 1 2 3; do
  AUDIT_TOT=0; WALL_TOT=0; ASR_TOT=0; DIAR_TOT=0; P95=0; RSS=0; CORES=0; FP=""; CLIP_RTF=()
  arm_start; T0=$(tis)
  ok=1
  run_cell chat69.wav "$REPS" || ok=0
  [ $ok = 1 ] && { run_cell gate_ms_v2.wav "$REPS" || ok=0; }
  arm_stop; T1=$(tis)
  [ $ok = 0 ] && { echo "run failed on attempt $attempt"; sleep 20; continue; }
  W=$(witness <(printf '%s\n' "$T0") <(printf '%s\n' "$T1") 2>/dev/null)
  MHZ=$(echo "$W" | awk '{print $1}'); DELIV=$(echo "$W" | awk '{print $2}')
  if python3 -c "import sys;sys.exit(0 if float('$MHZ')>=$ARM_MIN else 1)" 2>/dev/null; then
    RTF=$(python3 -c "print(round($WALL_TOT/$AUDIT_TOT,4))")
    OTHER=$(python3 -c "print(round($WALL_TOT-$ASR_TOT-$DIAR_TOT,2))")
    echo "METRIC phone_rtf=$RTF"
    echo "METRIC asr_s=$(python3 -c "print(round($ASR_TOT,2))")"
    echo "METRIC diar_s=$(python3 -c "print(round($DIAR_TOT,2))")"
    echo "METRIC other_s=$OTHER"
    echo "METRIC first_partial_s=${FP:--1}"
    echo "METRIC p95_piece_ms=${P95:-0}"
    echo "METRIC cores_used=${CORES}"
    echo "METRIC peak_rss_mb=$RSS"
    echo "METRIC witness_mhz=$MHZ"
    echo "METRIC deliv2400_pct=$DELIV"
    echo "METRIC audio_s=$(python3 -c "print(round($AUDIT_TOT,2))")"
    echo "METRIC rtf_chat69=${CLIP_RTF[chat69.wav]:-0}"
    echo "HASH $(cat /tmp/ar_chat69.wav.txt /tmp/ar_gate_ms_v2.wav.txt 2>/dev/null | sha256sum | cut -c1-12)"
    echo "note loadavg=$LOAD batt=$BATT mask=$MASK t=$THREADS r=$REPS attempt=$attempt"
    exit 0
  fi
  echo "attempt $attempt partial-arm (mean ${MHZ} MHz < ${ARM_MIN}) - cooling 30 s"
  sleep 30
done
echo "ERROR: never reached an armed window - refusing to log an unarmed number (an unarmed run is 20-60% slow and looks fine)"
exit 1
