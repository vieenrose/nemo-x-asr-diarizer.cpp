# Ideas backlog — composite phone RTF

Deferred but promising. Prune when tried or stale.

- **Make attribution incremental.** `attribute()` re-scans the entire char timeline and re-tags every piece
  on every diar turn update → O(chars × turns) per update. On long clips this is most of `other_s`. Fix:
  keep a "tagged up to char index" cursor and only re-tag the tail, or re-tag only pieces whose covering turn
  actually changed (diff the turn list by (speaker, start)). Expected: biggest `other_s` win available.
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
