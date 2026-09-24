#!/usr/bin/env bash
# build_host.sh - build the composite against local checkouts of the two upstreams.
#
# The two model stacks each carry their own ggml. That is not a preference, it is the only arrangement
# that works: see PROVENANCE.md, "Two ggml runtimes in one process" - sharing one ggml compiles, links,
# and then asserts inside ggml_backend_sched. So the diarizer comes in through libaudiocpp.so, whose
# version script keeps its ggml out of the global symbol table, and the ASR comes in as static CrispASR.
#
# Layout expected (override with DEPS_ROOT):
#   $DEPS_ROOT/crispasr     https://github.com/CrispStrobe/CrispASR        (measured at cb6171b)
#   $DEPS_ROOT/audiocpp     https://github.com/0xShug0/audio.cpp           (measured at fc24c99)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DEPS_ROOT=${DEPS_ROOT:-$(for c in ../ref ../../ref ../deps; do [ -d "$ROOT/$c/crispasr" ] && { echo "$(cd "$ROOT/$c" && pwd)"; break; }; done)}
[ -n "${DEPS_ROOT:-}" ] || { echo "ERROR: no upstream checkouts found - set DEPS_ROOT to a dir containing crispasr/ and audiocpp/" >&2; exit 1; }
C="$DEPS_ROOT/crispasr"; A="$DEPS_ROOT/audiocpp"
THREADS=${THREADS:-$(nproc)}
[ -d "$C" ] || { echo "ERROR: $C missing" >&2; exit 1; }
[ -d "$A" ] || { echo "ERROR: $A missing" >&2; exit 1; }

echo "== audio.cpp C API (shared, ggml hidden inside)"
# AUDIOCPP_BUILD_C_API is OFF by default, so the audiocpp target does not even exist without it -
# a plain "cmake --build --target audiocpp" fails with "No rule to make target", which reads like a
# missing checkout rather than a switched-off option.
cmake -S "$A" -B "$A/build-host" -DCMAKE_BUILD_TYPE=Release -DAUDIOCPP_BUILD_C_API=ON \
      -DENGINE_ENABLE_NATIVE_CPU="${NATIVE_CPU:-ON}" -DENGINE_ENABLE_OPENMP=ON >/dev/null
cmake --build "$A/build-host" --target audiocpp -j"$THREADS" >/dev/null
LIB_DIAG="$A/build-host/bin/libaudiocpp.so"
[ -f "$LIB_DIAG" ] || { echo "ERROR: $LIB_DIAG not produced" >&2; exit 1; }
if nm -D --defined-only "$LIB_DIAG" | grep -qE " T ggml_| T gguf_"; then
  echo "ERROR: $LIB_DIAG leaks ggml symbols - its version script did not apply, and the ASR half will"
  echo "       bind to the wrong ggml and assert in ggml_backend_sched." >&2
  exit 1
fi

echo "== CrispASR x-asr (static, own ggml)"
# Token timestamps need four lines in their greedy loop (one frame counter, one push_back). Shipped as a
# patch rather than a fork: it is model-agnostic, it cannot change any decoding decision, and if it fails
# to apply the build still works - it just falls back to inferred placement and says so at run time.
if [ -f "$ROOT/patches/crispasr-token-times.patch" ]; then
  if git -C "$C" apply -R --check "$ROOT/patches/crispasr-token-times.patch" 2>/dev/null; then
    echo "   token-times patch already applied"
  elif git -C "$C" apply --check "$ROOT/patches/crispasr-token-times.patch" 2>/dev/null; then
    git -C "$C" apply "$ROOT/patches/crispasr-token-times.patch" && echo "   applied token-times patch"
  else
    echo "   NOTE: token-times patch does not apply to this CrispASR revision - attribution will infer timing"
  fi
fi
cmake -S "$C" -B "$C/build-host" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON >/dev/null
for t in xasr crispasr-core ggml ggml-base ggml-cpu; do
  cmake --build "$C/build-host" --target "$t" -j"$THREADS" >/dev/null
done

echo "== composite"
mkdir -p "$ROOT/build"
# Always compile from scratch. Three files, ~10 s: a stale object linked against a changed struct layout
# (Config grew fields) does not fail to link, it corrupts memory at run time and segfaults somewhere that
# points at nothing relevant. That cost a debugging detour once; the cheapest fix is to never have it.
rm -f "$BUILD"/*.o "$BUILD"/nemo-x-asr-diarizer
INC="-I$ROOT/src -I$C/src -I$A/include"
# Compile-time availability only; the choice between exact and inferred timing happens at run time.
if nm --defined-only "$C/build-host/src/libxasr.a" 2>/dev/null | grep -q xasr_stream_token_times; then
  INC="$INC -DNEMO_HAVE_TOKEN_TIMES"; echo "   exact token timestamps: available"
else
  echo "   exact token timestamps: NOT available (inferred placement will be used)"
fi
g++ -O2 -std=c++17 $INC -c "$ROOT/src/engine.cpp" -o "$ROOT/build/engine.o"
g++ -O2 -std=c++17 $INC -c "$ROOT/src/fusion.cpp" -o "$ROOT/build/fusion.o"
g++ -O2 -std=c++17 $INC -c "$ROOT/src/main.cpp"  -o "$ROOT/build/main.o"
g++ "$ROOT/build/engine.o" "$ROOT/build/fusion.o" "$ROOT/build/main.o" \
    -o "$ROOT/build/nemo-x-asr-diarizer" \
    -L "$A/build-host/bin" -l:libaudiocpp.so \
    "$C/build-host/src/libxasr.a" "$C/build-host/src/libcrispasr-core.a" \
    "$C/build-host/ggml/src/libggml.a" "$C/build-host/ggml/src/libggml-base.a" \
    "$C/build-host/ggml/src/libggml-cpu.a" \
    -fopenmp -lpthread -ldl -lm \
    -Wl,-rpath,"$A/build-host/bin"
echo "built $ROOT/build/nemo-x-asr-diarizer"
