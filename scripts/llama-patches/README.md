<!-- SPDX-License-Identifier: MIT -->

# llama.cpp patches

`scripts/build_llama.sh` merges a fixed upstream master commit into the pinned
PR #28243 branch. When that merge conflicts, the script applies a patch from
this directory instead of stopping.

- `qwen4exp-merge-38a5b42d9.patch` — resolves `src/models/qwen4exp.cpp` for the
  merge of master `38a5b42d9` into pin `d1a92352c`. Master PR #28896 reshaped
  the qwen4exp norm gammas to `{ n_embd, hc }` and fused `rms_norm` with `mul`.
  The pin still creates those tensors as `{ hc_dim }`. The patch keeps master's
  shapes and fusion, and applies the same reshape to `layer.nextn.hc_head_norm`,
  which the MTP graph needs.

The patch applies to the pin's file, not to a conflicted file, so the result is
deterministic. The script checks the result against a SHA-256 value and stops if
it differs.

Each patch holds llama.cpp source lines. llama.cpp is MIT licensed.
Copyright (c) 2023-2026 The ggml authors.
