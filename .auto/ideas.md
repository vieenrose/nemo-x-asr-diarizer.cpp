# Ideas backlog — composite phone RTF

Deferred but promising. Prune when tried or stale.

- ~~**Make attribution incremental.**~~ DEAD (run #1081): timed with NEMO_PROF - push_delta 0.00 s and the
  final attribute pass 0.00 s per run. `other_s` was never bookkeeping; it is the diarizer's last encoder
  window landing outside the per-piece timers. Do not re-litigate this.
- **Overlap the two legs.** DEAD (measured, run #1082 follow-up): diar on a worker thread makes the composite
  37-47% SLOWER (gate RTF 0.785→1.074, chat69 0.776→1.145) and slows BOTH legs ~2.8x (asr 24.9→70.5 s),
  total CPU unchanged, peak RSS unchanged. 8 diar threads + 1 ASR thread on 2 A78s is not the mechanism -
  the loss is superlinear in the number of concurrent big-model working sets (168 MB + 106 MB streaming
  through the same caches). Any "run the legs concurrently" variant is predicted to fail the same way.
  Note for the archive: this is the closest thing to the old "two ggml spin pools" folklore, but it is a
  throughput/cache effect, not a livelock, and it happens with the GPU path already fixed.
- **Stop rebuilding the window list per output pass.** `main.cpp`'s `--windows` path builds a char vector and
  a per-window run map after the run; on long clips that is a second full pass over the transcript. Could be
  emitted incrementally as windows close during the loop.
- **Charge the loop honestly.** `other_s` = wall − asr − diar. Print a `now_s()` histogram of the three
  regions per piece to see whether the residue is attribution, the diar `next_event` drain, or allocation.
- **Diar push cadence.** We push 100 ms pieces. The scheduler has `chunk_len`/`chunk_left_context`/
  `chunk_right_context`; pushing 200-400 ms per call may amortise per-call overhead (byte-identity check
  required — geometry changes alter output).
- **Reuse buffers.** `std::vector<float>` copies per piece and `codepoints()` allocations in the hot path;
  a persistent arena for the char timeline could matter on ARM more than x86.
- **Prefault the models.** 168 MB + 106 MB mmapped: touch pages once during init and measure whether pass 1
  of a cell stops costing more than pass 2 (the archive measured a first-window premium of ~130 ms on the
  baseline for exactly this reason).
- **Check whether `audiocpp_request_set_option` can shrink the diar pool.** It creates ~8 threads lazily
  during streaming regardless of `bc.threads=2`; if an option exists to cap it, the 2-cpu contention margin
  gets wider. (Earlier thread-count probing found the pool appears mid-run, not at session_create.)

## Still open (post-#1082)
- **Is the ASR leg single-threaded on device?** The archive says CrispASR's Android ggml has no OpenMP and
  creates no worker threads. If true, asr (49.7 s of 106 s wall) occupies ONE core while the second core
  idles - the biggest structural waste left, since the legs cannot overlap. Verify with NEMO_DEBUG_THREADS
  (it prints a thread count per piece) before designing anything around it.
- **Diar `--diar-batch` > 1 in streaming mode** - if the family accepts a batched window it processes the
  same windows with fewer graph entries. Unverified; may be offline-only.
- **Prefault the models** (still untested).
