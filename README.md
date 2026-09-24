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

## The window format has to be scored, not eyeballed

Two bugs of the same shape showed up while building `--windows`, and both were invisible to the obvious
check ("is the text the same?"):

1. Cutting text at window boundaries split English words (`"that end"` | `"s well"`, 27 of 55 lines on the
   English holdout). Concatenated text was identical to segment output; scored, WER went 0.2455 -> 0.3136,
   because the scorer tokenises per line, so one reference word became a substitution plus an insertion.
2. "Fixing" the grouping by flushing on spaces dropped the separators. Words merged (`the cat` -> `thecat`),
   WER 0.9455. Still identical when concatenated.

Now text is grouped into words first - a word goes whole into the window its first character falls in, CJK
stays character-granular, punctuation and spaces attach to the word they follow - and the two output shapes
are equivalent where it counts: on host and on the phone, gate_ms_v2 scores WER 0.1765 / attribution 0.0658
of 76 either way, and holdout_en scores 0.2455 / 0.6124 of 209 either way. Consistency differs slightly
(0.963 vs 0.957) only because boundaries split speaker runs differently.

The transferable rule: **a formatter is verified by scoring its output, never by comparing its characters.**
Character equality was true in both broken cases.

## Token timestamps (added, and what they turned out to be worth)

x-asr never returned times, so the engine used to *infer* where each character was spoken. It can now use
the model's own timeline: `patches/crispasr-token-times.patch` adds a frame counter to the greedy loop
(CrispASR consumes one encoder frame per greedy step and emits at most one symbol, so a token's timestamp
is a loop counter - 40 ms grid, no alignment model, no extra compute). Verify the claim yourself with
`tools/validate_timestamps.py`, which checks timestamps against reference **silence** - a score can be
moved by the attributor or by luck, silence cannot.

Three measurements, in order of how much they surprised me:

1. **A transducer's frame index is a decision time, not a sound time.** Raw frame timestamps run ~0.3 s late:
   measured against an independent timeline they are +0.295 s median (p10 +0.08, p90 +0.48), and against
   ground-truth silence the shift that puts **0 of 119** tokens in silence is +300 ms (3 at 0 ms, 2 at
   450 ms). So `--token-offset-ms 300` is the default, derived from the audio and never from a score. This
   one generalises: anyone wiring up transducer timestamps will hit it.
2. **The timestamps are real.** 114 tokens over 44.98 s, monotonic, last one at 44.92 s - which also proves
   the 40 ms mapping rather than a fitted 39.0 ms (a 2.5 % scale error would have put the tail at 46.2 s).
   The exact path is used only if it reproduces the streamed transcript byte for byte; otherwise the engine
   falls back and says so (`[timing] inferred-placement`).
3. **Attribution did not get better, and the honest reading is that it could not have.** Attribution error
   moved between 0.0132 and 0.0658 across timing/offset/gap-fill combinations - that is **1 to 5 wrong tokens
   out of 76**, i.e. noise on a single clip. Meanwhile the 10x placement sweep above showed a flat 0.0132.
   Both point the same way: with a diarizer that misses 41 % of speech frames, *recall* is the binding
   constraint on speaker attribution, not clocks. Timestamps bought exact word alignment (subtitles, forced
   alignment, A/V sync, boundary reporting), not a better who.

So `--timing auto` prefers model timestamps, `inferred` reproduces the previous behaviour, and neither is
allowed to claim an accuracy win on this evidence. What is solid: a 40 ms token timeline that verifies
against silence, and a documented 300 ms decision lag.

## Drop-in output contract

`--windows` re-emits the same attribution in the shape the VibeASR autoresearch harness consumes: one
`[k/N]` block per `HOP_S = 70400/24000 = 2.933 s` of audio, text inside tagged with the speaker(s) that
spoke it, silent windows emitting nothing. Window cuts use the model's token times when available, and
otherwise spread each segment's text uniformly over its span (coarse, and the window-onset diagnostic says
so rather than pretending). Both shapes score identically through the archive's `score_stream.py` - WER
0.1765 and attribution 0.0658 either way on the bilingual gate - which is the point: "drop-in" is checkable
by running the archive's own tools against this binary.

## Diarization detection knobs

The diarizer's decode config is read from the REQUEST (`nemotron_3_diar/session.cpp` reads
`stream_request_.options`), so session options do nothing and a NULL request silently means
`threshold=0.5, min_frames=0, pad_frames=0`. Measured DER-lite (12-14 ground-truth turns per clip):

| speaker_pad_frames | 57 s clip (tuned on) | 45 s v2 clip (never tuned on) |
|---|---|---|
| 0 (upstream) | 37.8 | 45.8 |
| 20 | 30.3 | 37.5 |
| 45 (**default**) | 30.3 -> 27.6 miss | 29.3 (FA 1.1, spk 1.9) |
| 90 | ~20 | 22.3 (FA 3.7) |

`speaker_threshold` barely matters between 0.25-0.35; **pad_frames dominates**, because most of the miss
was the diarizer dropping below threshold in the middle of a turn rather than never detecting it. Defaults
are `threshold=0.3, pad_frames=45`, chosen in the middle of the curve rather than at its best point on the
eval clips: past that, `pad` starts merging genuine turn-taking, and both clips here are presentation-style
audio with few rapid exchanges. At these settings silence, pink-ish noise and a 440 Hz tone all produce
**zero** turns and zero text, which is the contract that matters for a streaming product.

Attribution on the 85-token gate moved between 0.0263 and 0.0658 across these settings - 2 to 5 wrong tokens
out of 76 - so no attribution claim is made from it; DER-lite above, on ~50 s of ground-truth turns per clip,
is the metric that actually resolves this change.

## Build note

`scripts/build_host.sh` deletes objects before compiling. A stale object linked against a changed struct
layout does not fail to link - it corrupts memory at run time and segfaults somewhere unhelpful.

## On the phone (Oppo CPH2371, Dimensity 1300, Android 13)

The arm64 build in `scripts/build_android.sh` is the one that ran on the device, armed and witness-checked
through the same protocol as the VibeASR streaming-1.5B baseline (`taskset`, `--threads 2`, wake-stream
arming, `time_in_state` witness, repeats inside one armed window). Mask `f0`, 2 threads:

| clip | RTF | first partial | p95 piece | peak RSS | witness mean |
|---|---|---|---|---|---|
| chat69 (protocol, 69 s) | **0.771** | 0.601 s | 95 ms | ~396 MB | 2175 MHz |
| gate_ms_v2 (hard, 4 speakers) | 0.775 | 0.752 s | 95 ms | 396 MB | 2176 MHz |
| bilingual_multispk_57s | 0.805 | 0.765 s | 95 ms | 408 MB | 2245 MHz |

Against the baseline's own anchor on the same device (RTF 1.2231, peak RSS 2198 MB, first text at one 2.93 s
window): **1.6x lower RTF, 5.5x less memory, ~4x lower first-output latency**, and it also emits speaker
turns, which the baseline does not. Token timestamps work on arm64 too (388 tokens on chat69, 40 ms grid).

Two findings that only showed up on the device:

1. **On a 2-CPU mask this composite is far slower than its parts predict, and I do not yet know why.** With
   the baseline's `taskset C0` (cpu6-7) a 45 s clip did not finish in 150 s (RTF > 3.3), while on `f0`/`ff` it
   finishes at RTF 0.77. Each engine alone is fine on `C0` (x-asr 0.2151, diarizer 0.279) and the config is
   identical to the standalone probe (`piece_ms 100`, `chunk_ms 480`, 1 ggml thread).

   I first explained this as two ggml spin-wait pools livelocking. **That explanation was wrong and is
   retracted.** The measurements that kill it: the standalone x-asr probe runs on that same mask with
   **exactly 1 thread** (neither build links OpenMP, and neither ggml creates worker threads), so there is no
   spin pool in the ASR at all; and `libaudiocpp.so` creates its ~8-thread pool *during* streaming (1 thread
   through `init`, 10 threads mid-run), so a pool-contention story cannot explain a `--no-diar` hang either.
   What is left, untested: cache/TLB interference between two resident models, and the diar pool's threads
   landing on 2 cpus. `--main-affinity`/`--engine-affinity` exist to separate those hypotheses; note that the
   pool is created lazily, so the current implementation moves only threads that already exist (measured:
   `moved=1`, `threads=1` at pieces 0-8) - it is a probe, not yet a fix.
2. **The composite costs more than the sum of its parts**: 0.77 vs 0.215 + 0.279 = 0.49. The two engines
   contend for the same cores with spin-wait barriers, so the ASR leg alone stretches from 0.215 to ~0.75
   inside the composite. Closing that gap means one shared pool (or a non-spinning build), not more tuning.

Invariant worth having: with and without `--no-diar` the ASR text is byte-identical (1185 chars on chat69),
so attribution is a layer over the transcript, never a rewrite of it.

## Known limits

* **The diarizer commits turns late.** It computes in chunks with caches - genuinely streaming - but the
  first turn batch on the bilingual gate clip arrives at **30.5 s of audio**. So live output is provisional
  (`--live` prints revisions as `~[n] ...`) and only stabilises after roughly half a clip. Attribution
  latency is reported as `first_turn_audio_s` and is a property of the model, not of this code.
* **Diarization recall needs tuning.** Against the 12-turn ground truth, both the host and the phone build
  miss ~41 % of speech frames (0 false alarms, 0 speaker-error where it does fire). That is a threshold
  question in the upstream decode config (`speaker_threshold`, `speaker_min_frames`), which this tool does
  not yet expose. Every attribution number above is quoted with that hole in it.
* **No word timestamps from x-asr - measured, and it costs less than expected.** See the next section:
  attribution is flat across a 10x range of the placement model, so exact token times would buy almost
  nothing for *who said it*. What they would buy is *when the speaker changed* - boundaries are placed at
  word granularity, so a label change carries about one word of timing uncertainty.
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

## Does the missing timestamp data actually hurt? (measured, `scripts/attribution_sensitivity.sh`)

x-asr returns text, not tokens with times, so `Fusion` places each delta's characters inside the audio it
decodes. Two knobs control that placement (`--char-dur-ms`, `--asr-latency-ms`) and one controls what happens
in gaps (`--gap-snap-ms`). The transcript is byte-identical across every cell below - same model, same audio,
greedy decode - so whatever moves is the placement doing its work. Bilingual gate clip, 18 runs:

| placement sweep | attribution error | coverage | consistency | WER |
|---|---|---|---|---|
| per-char duration 30 / 90 / 300 ms (10x range) | 0.0132 | 0.894 | 0.993 | 0.1765 |
| lag correct (480 ms = `--chunk-ms`) | 0.0132 | 0.894 | 0.994→0.993 | 0.1765 |
| lag wrong by 3x low (160 ms) | 0.0526 | 0.894 | 0.958 | 0.1765 |
| lag wrong by 2x high (960 ms) | 0.0526-0.0658 | 0.894 | 0.950-0.971 | 0.1765 |

So:

1. **Per-character precision is nearly free.** A 10x change in the character-time model moves attribution
   by 0.0000. Real token timestamps would not make "who said this word" meaningfully better on this material.
2. **The aggregate lag is not free.** Getting it wrong by 3x costs 4-5x the attribution error - and that lag
   is *known analytically* (the encoder's chunk length), no timestamps needed.
3. **The damage timestamps did cause was indirect, and it was in the cut, not the choice.** With
   character-granular cuts, a boundary landing inside a word printed `...subsequently dro` / `f`. The scorer
   then reads two tokens where the reference has one: the same transcript scored **WER 0.1765 with clean cuts
   and 0.3176 with cuts through words**, while attribution barely noticed. Attribution is now word-granular
   where words exist (CJK stays per-character), and WER is invariant across all 18 cells.

Two honest caveats about timing measurement:

* Boundary *time* is still inferred at word granularity, and the diarizer's own turn onsets are median 0.80 s
  off the reference (p90 1.30 s) on this clip. That is the floor under any boundary claim here.
* The archive scorer's `turn-onset` diagnostic reports 7.78 s for this output format. Ignore it: it multiplies
  the segment counter by its 2.93 s window hop, which is only meaningful for a windowed streaming transcript.
  It is a diagnostic in that tool and is not gated - do not gate it against this format.
