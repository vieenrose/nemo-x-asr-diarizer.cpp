# Provenance

What came from where, and which revision each measurement used. Everything not listed below as upstream
is written in this repository.

## Engines and libraries (not vendored; built as separate artifacts)

| upstream | license | revision measured | used for | how it is linked |
|---|---|---|---|---|
| [CrispASR](https://github.com/CrispStrobe/CrispASR) | MIT | `cb6171b` | x-asr streaming Zipformer2 transducer: `src/xasr.cpp` (`xasr_stream_*`), `src/core/gguf_loader.cpp`, Kaldi fbank | static libs (`libxasr.a`, `libcrispasr-core.a`) + its ggml fork, linked into the binary |
| [audio.cpp](https://github.com/0xShug0/audio.cpp) | Apache-2.0 | `fc24c99` | Nemotron-3 diarization: `src/models/nemotron_3_diar/{session,streaming,encoder,frontend}.cpp` behind the C API `src/capi/audiocpp.cpp` | `libaudiocpp.so` built with `AUDIOCPP_BUILD_C_API=ON`, so its version script keeps its ggml/cJSON/sentencepiece out of the global symbol table |
| [ggml](https://github.com/ggml-org/ggml) (CrispASR fork `CrispStrobe/ggml`) | MIT | `2f5a80d`, plus audio.cpp's bundled ggml | tensors/backends for both models | see "two ggml runtimes" below |

Nothing was copied from [RapidSpeech.cpp](https://github.com/RapidAI/RapidSpeech.cpp): its repository
ships no license file, and its x-asr path produced no output during this evaluation (`stack0_out` came out
all-NaN with a value-verified model and external reference features). Its documentation informed the choice
of the 480 ms chunk and the chunked-cache streaming design. That is a debt we acknowledge rather than hide.

Model weights are downloaded, not shipped, and are licensed by their own release terms
(`scripts/fetch_models.sh` prints what each one claims).

## Written here

| file | what it is |
|---|---|
| `src/fusion.{h,cpp}` | the attribution algorithm: codepoint placement inside a delta's audio span, turn overlap with proximity fill, `snapped` marking |
| `src/engine.{h,cpp}` | the composite: one timeline, two streams, delta buffering, telemetry (RTF split by model, first partial, p95 piece, peak RSS, first-turn audio) |
| `src/wav.h` | PCM16 WAV reader that refuses anything it cannot read exactly |
| `src/main.cpp` | CLI, the `[i/N] Speaker k:text` output form, JSON telemetry, turn dump |
| `scripts/build_host.sh` | the build recipe, including why the two ggml copies must stay separate |
| `scripts/build_android.sh` | NDK cross-build for `arm64-v8a` - **not validated end to end yet**; the two upstreams each cross-build on their own |
| `scripts/fetch_models.sh` | downloads plus size and GGUF value-level verification (a structurally valid GGUF with a 32-byte alignment slip is still a broken model - that bug was found the hard way elsewhere) |
| `tests/unit.cpp`, `tests/selftest.sh` | WAV parsing and attribution correctness, including planted faults to prove the assertions bite |

## Two ggml runtimes in one process

Tried and rejected: compiling CrispASR's `xasr.cpp` against audio.cpp's ggml. It compiles, it links, and
then `ggml_backend_sched` asserts on an op the other fork does not implement. Localizing the colliding
`ggml_*`/`gguf_*` symbols with `objcopy --localize-symbols` also fails in a quieter way - archive members
whose definitions became local are never pulled in, so the ASR half silently ends up calling the *other*
ggml and asserts in the same place.

What works: keep them apart. audio.cpp already ships the isolation for its side (`src/capi/audiocpp.map`
- its own comment explains that static libs linked in were compiled without `-fvisibility=hidden` and would
otherwise be re-exported), and CrispASR's side is plain static linking. Two ggml copies, two heaps, no
shared objects: total peak RSS costs about 3 MB of code for that freedom.
