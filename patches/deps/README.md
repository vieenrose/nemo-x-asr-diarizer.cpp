# Dependency patches

Every performance change in this session lives in the two **dependency worktrees**, not in this repository:

* `../ref/crispasr` — CrispASR (static, with its own vendored ggml): the transducer joint in f16, the
  per-step graph rebuild (and the latent use-after-free that came with it), cache buffer reuse, the
  thread-partition threshold, the measurement hooks, and the GGUF tensor-name prefix that makes the merged
  bundle loadable.
* `ref/crispasr/ggml` — the ggml submodule inside CrispASR: the per-element integer division in the strided
  elementwise path, the work-size threshold for small copy/elementwise ops, and the skinny-matmul threshold.
* `../ref/audiocpp` — audio.cpp: the mel filterbank that was multiplying 98.5% zeros in software quad
  precision (the session's largest win), the removal of the streaming window padding, the bundle's
  name-list tolerance, and the phase/shape instrumentation.

`deps.lock` records the three commit heads this project was measured against, and
`scripts/check_deps.sh` refuses to build against anything else. But those heads live in *local clones* of
other people's repositories, so a clone of this project alone cannot build. These patch series close that gap:
they are `git format-patch` output of exactly the commits `deps.lock` points at.

## Apply order (order matters)

The CrispASR series contains commits that move the `ggml` submodule pointer, so the submodule must be patched
first, or those gitlink commits will not apply.

```bash
DEPS=../ref
# 1. CrispASR's vendored ggml FIRST - CrispASR's own series moves this submodule's pointer
git -C "$DEPS/crispasr/ggml" am ../../nemo-x-asr-diarizer.cpp/patches/deps/crispasr-ggml/*.patch
# 2. CrispASR itself (base cb6171b)
git -C "$DEPS/crispasr" am ../../nemo-x-asr-diarizer.cpp/patches/deps/crispasr/*.patch
# 3. audio.cpp (base fc24c99)
git -C "$DEPS/audiocpp" am ../../nemo-x-asr-diarizer.cpp/patches/deps/audiocpp/*.patch
bash scripts/check_deps.sh      # will report drift until the resulting heads match deps.lock
```

If the resulting heads differ from `deps.lock` (different `git am` metadata, rebases), either update
`deps.lock` and re-measure, or build with `ALLOW_DEPS_DRIFT=1` — but then the numbers are not the ones in
`.auto/`.

## Why the patches live here rather than as forks

These are upstream projects with their own release histories. Keeping the changes as a reviewable patch
series in this repository means the diff against upstream is one `git am` away, and the claims in
`docs/` and `.auto/` can be checked against exactly the code they describe.
