#!/bin/bash
# build_android.sh - arm64 build of the composite. THIS RECIPE IS THE ONE THAT PRODUCED THE BINARY THAT RAN
# ON THE OPPO (it ran chat69 at rtf 0.771 with model token timestamps); the earlier version of this file was
# a guess and was never executed.
#
# Design, and the one thing that must not be "simplified": the two engines carry their own ggml.
#   * x-asr links CrispASR's ggml STATICALLY into this binary.
#   * Nemotron diarization comes from libaudiocpp.so, which keeps ggml hidden inside itself
#     (AUDIOCPP_BUILD_C_API=ON -> audiocpp SHARED + hidden visibility + src/capi/audiocpp.map).
# Forcing one shared ggml was tried and fails after linking (GGML_ASSERT(*cur_backend_id != -1) in
# ggml-backend.cpp). objcopy --localize-symbols looked cheaper and was worse: archive members with localized
# definitions silently stopped being pulled in, so part of the binary called the wrong ggml. Verify with
#   llvm-readelf --dyn-syms libaudiocpp.so | grep -cE ' (ggml|gguf)_'      # must be 0
# - that is a value-level check on the linkage, not a "it loaded" claim.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)

# Refuse to build against dependency worktrees that are not the state this project was measured
# against: a silently stale dependency is indistinguishable from a failed optimisation.
bash scripts/check_deps.sh || exit 1
NDK=${NDK:-/tmp/ndk/android-ndk-r26d}
ABI=${ABI:-arm64-v8a}
API=${API:-33}
C=${CRISPASR:-$ROOT/../ref/crispasr}
A=${AUDIOCPP:-$ROOT/../ref/audiocpp}
OUT=$ROOT/build-android
CLANG=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++
STRIP=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip
READER=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf
mkdir -p "$OUT"

echo "== audio.cpp C API for Android (shared, ggml hidden inside)"
# AUDIOCPP_BUILD_C_API defaults to OFF, so an Android tree configured without it has NO C API at all - just
# audiocpp_cli. That is how this branch's first audio.cpp Android build looked.
# -llog: the C-API target calls __android_log_write and does not link liblog, so the link fails with
# "undefined symbol: __android_log_write" unless CMAKE_SHARED_LINKER_FLAGS carries it.
cmake -S "$A" -B "$A/build-android" \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=$ABI -DANDROID_PLATFORM=android-$API -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF -DAUDIOCPP_BUILD_C_API=ON \
  -DCMAKE_SHARED_LINKER_FLAGS="-llog" >/dev/null
cmake --build "$A/build-android" --target audiocpp -j"$(nproc)" > /tmp/ar_audiocpp_build.log 2>&1 || {
  echo "audiocpp build FAILED (a stale library would be linked and measured as if it were the change):"
  grep -E "error|Error" /tmp/ar_audiocpp_build.log | head -10; exit 1; }
SO=$A/build-android/bin/libaudiocpp.so
[ -f "$SO" ] || { echo "no $SO"; exit 1; }
LEAK=$($READER --dyn-syms "$SO" 2>/dev/null | grep -cE ' (ggml|gguf)_' || true)
echo "   ggml symbols leaked from libaudiocpp.so: $LEAK (must be 0)"
[ "$LEAK" = 0 ] || { echo "refusing to build a composite whose two ggmls can collide"; exit 1; }
# 453 MB unstripped vs 29 MB stripped; the phone does not need DWARF.
"$STRIP" -g "$SO" -o "$OUT/libaudiocpp.so"

echo "== CrispASR x-asr for Android (static)"
# The whole CrispASR tree does NOT build for Android here (piper-tts fails, unrelated). Build the targets the
# composite needs and check the token-times symbol actually exists in the arm64 archive.
cmake -S "$C" -B "$C/build-android" \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=$ABI -DANDROID_PLATFORM=android-$API -DCMAKE_BUILD_TYPE=Release >/dev/null
if [ -f "$ROOT/patches/crispasr-token-times.patch" ] &&
   git -C "$C" apply --check "$ROOT/patches/crispasr-token-times.patch" 2>/dev/null; then
  git -C "$C" apply "$ROOT/patches/crispasr-token-times.patch" && echo "   applied token-times patch"
fi
cmake --build "$C/build-android" --target xasr -j"$(nproc)" > /tmp/ar_crispasr_build.log 2>&1 || {
  echo "CrispASR build FAILED (a stale archive would be linked and measured as if it were the change):"
  grep -E "error|Error" /tmp/ar_crispasr_build.log | head -10; exit 1; }
if $NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-nm --defined-only "$C/build-android/src/libxasr.a" 2>/dev/null \
     | grep -q xasr_stream_token_times; then
  TIMES="-DNEMO_HAVE_TOKEN_TIMES"; echo "   exact token timestamps: available"
else
  TIMES=""; echo "   exact token timestamps: NOT available (timing will be inferred)"
fi

echo "== composite"
# -g0 is not cosmetic: with debug info, ld.lld died with "unable to execute command: Bus error" on this
# box while linking the same inputs. The binary needs no symbols - it is stripped for the device anyway.
# $ORIGIN in RUNPATH means libaudiocpp.so sits next to the binary in the device dir, no LD_LIBRARY_PATH
# needed for the common case (the harnesses still set it because adb does not always honour RUNPATH).
rm -f "$OUT"/*.o "$OUT"/nemo-x-asr-diarizer
# diar_crispasr.cpp needs CrispASR's ggml headers directly (docs/one-runtime-merge.md SS16: it builds the
# diar encoder+head on CrispASR's own ggml, the actual one-runtime merge, behind --diar-native) - the rest of
# this composite only ever sees ggml through xasr.h/audiocpp.h's own C APIs.
"$CLANG" --target=aarch64-linux-android$API -O2 -g0 -std=c++17 $TIMES \
  -I"$ROOT/src" -I"$C/src" -I"$A/include" \
  -I"$C/ggml/include" -I"$C/ggml/src" -I"$C/ggml/src/ggml-cpu" -w \
  "$ROOT/src/fusion.cpp" "$ROOT/src/engine.cpp" "$ROOT/src/main.cpp" "$ROOT/src/diar_crispasr.cpp" \
  -o "$OUT/nemo-x-asr-diarizer" \
  -L"$A/build-android/bin" -l:libaudiocpp.so \
  "$C/build-android/src/libxasr.a" "$C/build-android/src/libcrispasr-core.a" \
  "$C/build-android/ggml/src/libggml.a" "$C/build-android/ggml/src/libggml-base.a" \
  "$C/build-android/ggml/src/libggml-cpu.a" \
  -static-libstdc++ -Wl,-rpath,'$ORIGIN' -Wl,--build-id=none -ldl -lm -llog
ls -la "$OUT" | awk 'NR>1{print "  ", $5, $9}'
cat <<'EOF'

On the device: put nemo-x-asr-diarizer, libaudiocpp.so and both .gguf files in one directory.
Run it on four cpus (taskset f0) for now. On the baseline's 2-cpu mask (C0 = cpu6-7) it is >3x slower per
audio-second and a 45 s clip may not finish in 150 s; on f0/ff it runs at rtf ~0.77. Each engine alone is
fine on C0. The spin-pool explanation written here earlier was WRONG - see README "On the phone" for what
was actually measured. The diarizer does create ~8 threads, but lazily, during streaming, so it cannot
explain a --no-diar hang.
EOF
