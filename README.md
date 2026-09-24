# nemo-x-asr-diarizer.cpp

Streaming speech recognition **and** speaker diarization in one process on one audio timeline:
[X-ASR](https://huggingface.co/cstr/x-asr-zh-en-GGUF) streaming Zipformer2 transducer for the words,
Nemotron-3 Diarization for *who*, and a small attribution layer that puts the second onto the first.
C++17 + ggml, CPU only, no GPU, no Python at runtime.

It exists because the two jobs are usually done by two systems that do not know about each other, and
because one model that does both (a 1.5B ASR+diarize LLM) costs 9x the memory and 5x the compute of
this pair. Measured numbers below, on a phone.

## What it does

```
 piece (100 ms of 16 kHz PCM)
   |
   +--> Nemotron-3 diar stream  -->  turns: (start, end, speaker, confidence)     [audio.cpp, C API]
   |
   +--> x-asr stream            -->  text delta (append-only transcript)          [CrispASR, xasr_stream]
   |
   +--> Fusion                  -->  characters tagged with a speaker  -->  [i/N] Speaker k:text
```

Output format is the streaming convention used by the sibling ASR archive, so its WER + speaker-attribution
scorers run against this output without a converter:

```
[1/12]
 Speaker 0:一次次存取不同的视角
[2/12]
 Speaker 1:。黄灯的十字路口
```

## Measured

Phone: Oppo CPH2371 (Dimensity 1300, 2x A78 prime), `taskset C0`, `-t 2`, governor armed and witnessed.
Host: x86-64, 2 threads. Models: `x-asr-zh-en-q8_0.gguf` (168 MB) + `nemotron-3-diarization-q8_0.gguf` (107 MB).

| | RTF | peak RSS | first text |
|---|---|---|---|
| composite, phone | 0.14-0.23 | ~420 MB | ~0.2-0.4 s |
| composite, host | 0.13 | 420 MB | 0.12 s |
| x-asr alone, phone | 0.215 (hard audio) | 248 MB | 0.22 s |
| Nemotron-3 diar alone, phone | 0.279 | 171 MB | turns commit at 30.5 s - see Known limits |

Accuracy, same 40-utterance LibriSpeech gate set, paired token-level test:

| system | WER | vs baseline |
|---|---|---|
| 1.5B ASR+diarize baseline | 4.51 % | - |
| x-asr (this repo's ASR half), phone | **4.27 %** | +0.27 pp [-1.09, +1.64], McNemar p=1.0 (b=13, c=12) - indistinguishable |

Hard consumer audio (bilingual, 4 speakers, code-switching; single clips, so read these as observations):

| set | WER baseline | WER composite | attribution baseline | attribution composite |
|---|---|---|---|---|
| `gate_ms_v2` | 0.1765 | 0.1765 | 0.4235 err / cov 1.000 / consist 0.622 | **0.0132 err** / cov 0.894 / **consist 0.993** |
| `holdout_en` | 0.2636 | 0.2455 | | |
| `holdout_zh` | 0.1538 | 0.0791 | | |

Attribution is an error rate over the tokens that carry a tag, under an optimal one-to-one tag mapping,
so it must be read next to **coverage**: the composite gets 89.4 % of tokens tagged at 1.3 % error
(~88 % correct speaker tags) where the baseline gets 100 % tagged at 42 % error (~58 % correct).

## Build

Needs the three upstreams (see [PROVENANCE.md](PROVENANCE.md)); `scripts/build_host.sh` wires them
together and explains the one non-obvious requirement - the two model stacks each want their own ggml.

```bash
scripts/fetch_models.sh          # two GGUFs, size- and value-verified
scripts/build_host.sh            # -> build/nemo-x-asr-diarizer
./build/nemo-x-asr-diarizer --audio clip16k.wav
./build/nemo-x-asr-diarizer --audio clip16k.wav --json --turns-out turns.json --live
```

Android (`arm64-v8a`, NDK r26): `scripts/build_android.sh`. On device, pin to the two big cores and
keep `--threads` equal to the number of cores in the mask - `-t 4` under a 2-core mask measured RTF 66
instead of 0.23, because the worker threads spin-wait on each other.

```bash
adb shell "cd /data/local/tmp/nemo && taskset C0 ./nemo-x-asr-diarizer --audio wav/clip.wav -t 2"
```

## How attribution works (and why it is not a timestamp lookup)

`src/fusion.h` is the whole idea. A text delta is not *at* a moment; it is a claim about an interval of
audio, delayed by the encoder's own window. So:

1. each delta is pinned to the audio span it decodes (`charged_upto -> fed - asr_latency`);
2. its **codepoints** (not bytes - half the output is Chinese) are placed inside that span, right-aligned,
   at `--char-dur-ms` each, because a delta arrives when words *came out*, not when the silence before them
   started;
3. each character is attributed to the turn overlapping it; text inside a between-turns pause goes to the
   nearer turn and is marked `snapped`, because the diarizer misses ~41 % of speech frames on this material
   and untaged text is a worse report than proximity-attributed text;
4. attribution runs over the buffered deltas **against the turn timeline as it stands**, which is the only
   way it can work at all - see the next section.

Tunables: `--asr-latency-ms` (default = `--chunk-ms`), `--char-dur-ms` (90), `--gap-snap-ms` (400; 0 =
always nearest-speaker).

## Known limits

* **The diarizer commits turns late.** It computes in chunks with caches - genuinely streaming - but the
  first turn batch on the bilingual gate clip arrives at **30.5 s of audio**. So live output is provisional
  (`--live` prints revisions as `~[n] ...`) and only stabilises after roughly half a clip. Attribution
  latency is reported as `first_turn_audio_s` and is a property of the model, not of this code.
* **Diarization recall needs tuning.** Against the 12-turn ground truth, both the host and the phone build
  miss ~41 % of speech frames (0 false alarms, 0 speaker-error where it does fire). That is a threshold
  question in the upstream decode config (`speaker_threshold`, `speaker_min_frames`), which this tool does
  not yet expose. Every attribution number above is quoted with that hole in it.
* **No word timestamps from x-asr.** The transducer API returns text, not tokens-with-times, so character
  timing is inferred (step 2 above) rather than measured. A token-time path through the encoder would make
  attribution exact; that is the highest-value change here.
* **Two ggml runtimes in one process, deliberately.** x-asr's runtime is CrispASR's, against its ggml fork;
  Nemotron-3 lives in audio.cpp's C API, which is version-scripted to hide its own ggml. Building the two
  against a single ggml compiles and links - and then asserts in `ggml_backend_sched` because the fork has
  ops the other does not. If you are reading this to wire the libraries up yourself: keep them separate,
  `libaudiocpp.so` for the diarizer, static CrispASR for the ASR.

## Provenance

Not much here is invented. Engines, model runtimes and ideas are upstream, and
[PROVENANCE.md](PROVENANCE.md) lists them file by file with licenses and the commit each measurement used.
In short: [CrispASR](https://github.com/CrispStrobe/CrispASR) (MIT) for the x-asr streaming runtime,
[audio.cpp](https://github.com/0xShug0/audio.cpp) (Apache-2.0) for Nemotron-3 diarization and its C API,
[ggml](https://github.com/ggml-org/ggml) (MIT) underneath both, and
[RapidSpeech.cpp](https://github.com/RapidAI/RapidSpeech.cpp) - whose x-asr path did not run, and whose
repo carries no license file, so nothing was copied from it and only its documentation informed this one.
Model weights carry their own licenses (`fetch_models.sh` prints them); this repository's license covers
the code here only.
