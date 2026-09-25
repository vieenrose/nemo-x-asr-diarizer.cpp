#!/bin/bash
# Compare the dependency worktrees against deps.lock and FAIL on drift.
#
# This exists because of a measured failure mode, not a hypothetical one: both build scripts used to pipe
# cmake to /dev/null without checking the exit status, so a dependency that failed to compile was linked
# stale, measured, and reported as if the change had been tested. Two experiments produced confident,
# meaningless numbers that way. A dependency that is not what you think it is should be just as loud.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOCK="$ROOT/deps.lock"
[ -f "$LOCK" ] || { echo "deps.lock missing - cannot verify the dependency state"; exit 1; }

want() { grep -E "^$1=" "$LOCK" | head -1 | cut -d= -f2; }

drift=0
report() { # report <label> <dir>
  local label=$1 dir=$2 locked actual
  locked=$(want "$label")
  actual=$(git -C "$dir" rev-parse --short HEAD 2>/dev/null)
  local dirty; dirty=$(git -C "$dir" status --porcelain 2>/dev/null | grep -v '^ M ggml$' | wc -l)
  if [ "$actual" != "$locked" ]; then
    echo "DEPS DRIFT  $label: locked $locked, worktree has ${actual:-<none>}"
    drift=1
  elif [ "$dirty" != "0" ]; then
    echo "DEPS DIRTY  $label: at $locked but $dirty uncommitted change(s) in the worktree"
    drift=1
  else
    echo "deps ok     $label $locked"
  fi
}
report crispasr     "$ROOT/../ref/crispasr"
report crispasr_ggml "$ROOT/../ref/crispasr/ggml"
report audiocpp     "$ROOT/../ref/audiocpp"

if [ "$drift" = "1" ]; then
  cat >&2 <<'MSG'

The dependency worktrees are not the state this project was measured against. Either:
  * you are intentionally changing a dependency - re-measure, then update deps.lock with the new hashes, or
  * something moved them (a git checkout, a submodule update, a discarded experiment) - restore with
    `git -C ../ref/<repo> checkout <hash from deps.lock>`.
Set ALLOW_DEPS_DRIFT=1 to build anyway; the result will not correspond to the recorded numbers.
MSG
  [ "${ALLOW_DEPS_DRIFT:-0}" = "1" ] || exit 1
fi
exit 0
