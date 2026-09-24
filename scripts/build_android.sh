#!/usr/bin/env bash
# build_android.sh - cross-build for arm64-v8a.
#
# STATUS: the two upstreams each cross-build for Android on their own (that is how the phone numbers in
# the README were measured), but this script's LINK of the composite for Android has not been validated
# end to end yet. Treat it as a recipe that is one verification run away from working, not as a working
# build. The host path in build_host.sh IS validated. Expect to spend the gap on: PIC flags for the
# static ggml, -llog for __android_log_write, and keeping OpenMP off on a device with no libomp.so.
#
# Known trap, learned the expensive way: threads must not exceed the cores in the pinning mask. The
# baseline runs `taskset C0` (cpu6-7) with -t 2; running the composite with -t 4 under that same mask
# measured RTF 66 instead of 0.23 - four spin-waiting workers on two cpus.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DEPS_ROOT=${DEPS_ROOT:?set DEPS_ROOT to a dir containing crispasr/ and audiocpp/}
NDK=${NDK:-/opt/android-ndk-r26d}
ABI=${ABI:-arm64-v8a}
API=${API:-android-33}
CPU=${CPU:--mcpu=cortex-a78}      # Dimensity 1300 / X60 class; A78c on other SoCs, adjust per device
[ -x "$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++" ] || { echo "ERROR: NDK clang++ missing at $NDK (if you unpacked with a tool that drops symlinks, extract with symlink support - a 'clang' that is a 5-byte file is the symptom)" >&2; exit 1; }
CLANG="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin"
TOOL="$CLANG/clang++ --target=aarch64-linux-android$(sed 's/android-//' <<<"$API") -O2 -std=c++17 $CPU"
A="$DEPS_ROOT/audiocpp"; C="$DEPS_ROOT/crispasr"

echo "== audio.cpp for Android (static, no OpenMP on device)"
cmake -S "$A" -B "$A/build-android" -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
      -DANDROID_ABI=$ABI -DANDROID_PLATFORM=$API -DCMAKE_BUILD_TYPE=Release \
      -DENGINE_ENABLE_NATIVE_CPU=OFF -DGGML_NATIVE=OFF -DENGINE_ENABLE_OPENMP=OFF \
      -DAUDIOCPP_BUILD_C_API=ON \
      -DCMAKE_EXE_LINKER_FLAGS="-llog" >/dev/null
cmake --build "$A/build-android" --target audiocpp engine_runtime ggml ggml-base ggml-cpu -j"$(nproc)" >/dev/null

echo "== CrispASR x-asr for Android"
cmake -S "$C" -B "$C/build-android" -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
      -DANDROID_ABI=$ABI -DANDROID_PLATFORM=$API -DCMAKE_BUILD_TYPE=Release \
      -DGGML_ARM_DOTPROD=ON -DGGML_OPENMP=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON >/dev/null
cmake --build "$C/build-android" --target xasr crispasr-core ggml ggml-base ggml-cpu -j"$(nproc)" >/dev/null

echo "== composite"
mkdir -p "$ROOT/build-android"
for f in engine fusion main; do
  $TOOL -I"$ROOT/src" -I"$C/src" -I"$A/include" -c "$ROOT/src/$f.cpp" -o "$ROOT/build-android/$f.o"
done
# Static on Android: libaudiocpp.so would need an rpath that Android ignores outside /data, so the C API
# object is linked in directly and its ggml comes from audio.cpp's archives. This is the part that still
# needs verification - if it asserts in ggml_backend_sched, the two ggml copies are colliding, and the fix
# is to keep audio.cpp's side in a .so shipped next to the binary with LD_LIBRARY_PATH=..
$TOOL "$ROOT/build-android"/*.o -o "$ROOT/build-android/nemo-x-asr-diarizer" \
  -Wl,--start-group "$A/build-android/libaudiocpp.so" "$A/build-android/libengine_runtime.a" \
  "$A/build-android/ggml/src/libggml.a" "$A/build-android/ggml/src/libggml-base.a" \
  "$A/build-android/ggml/src/libggml-cpu.a" \
  "$C/build-android/src/libxasr.a" "$C/build-android/src/libcrispasr-core.a" \
  "$C/build-android/ggml/src/libggml.a" "$C/build-android/ggml/src/libggml-base.a" \
  "$C/build-android/ggml/src/libggml-cpu.a" -Wl,--end-group \
  -fopenmp -static-libstdc++ -llog -ldl
echo "built $ROOT/build-android/nemo-x-asr-diarizer (UNVERIFIED on device - run tests before quoting numbers)"
