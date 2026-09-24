#!/usr/bin/env bash
# selftest.sh - prove the tests, not just the code. A green test that cannot fail is worse than none,
# so each unit test also runs against a build with a fault deliberately planted in the code under test;
# those runs MUST fail. A planted fault that still passes is a hole in the test, and this script says so.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BIN=${BIN:-/tmp/nemo-unit}
rc=0

build() { g++ -O1 -std=c++17 -I"$ROOT/src" "$@" "$ROOT/tests/unit.cpp" "$ROOT/src/fusion.cpp" -o "$BIN"; }

echo "[1/3] unit tests"
build || { echo "  FAIL compile"; exit 1; }
if "$BIN"; then echo "  PASS"; else echo "  FAIL unit tests failed"; rc=1; fi

echo "[2/3] planted fault: attributor always answers the first turn"
build -DNEMO_PLANT_FAULT || { echo "  FAIL compile"; exit 1; }
if "$BIN" >/dev/null 2>&1; then echo "  FAIL the fusion tests never noticed a broken attributor"; rc=1
else echo "  PASS (test caught it)"; fi

echo "[3/3] planted fault: WAV chunk walk measures the offset after the payload"
build -DNEMO_PLANT_FAULT_WAV || { echo "  FAIL compile"; exit 1; }
if "$BIN" >/dev/null 2>&1; then echo "  FAIL the wav tests never noticed a broken chunk walk"; rc=1
else echo "  PASS (test caught it)"; fi

[ $rc = 0 ] && echo "selftest: green" || echo "selftest: RED"
exit $rc
