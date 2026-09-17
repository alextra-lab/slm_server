#!/usr/bin/env bash
# Build the llama-server binary that slm_server runs.
#
# Why this exists (2026-09-03): Homebrew's llama.cpp v0.3.0 (build 10621) predates
# the qwen4exp architecture merge, so it cannot load Qwen3.8-Flash-Next at all --
# it fails with "unknown model architecture: 'qwen4exp'". PR #28243 adds the MTP
# draft-head support that Flash-Next needs for speculative decoding. That PR is an
# open draft upstream, so it is pinned by commit here rather than tracked.
#
# The checkout lives beside this repo, not inside it: the build tree is ~360 MB and
# has no business in slm_server's history.
#
# Usage:
#   scripts/build_llama.sh          # build at the pinned commit
#   LLAMA_PIN=<sha> scripts/build_llama.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${LLAMA_SRC:-$(dirname "$REPO_DIR")/llama.cpp}"
PIN="${LLAMA_PIN:-$(cat "$REPO_DIR/config/llama.cpp.pin")}"
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

echo "==> source:  $SRC"
echo "==> pin:     $PIN"

if [ ! -d "$SRC/.git" ]; then
    echo "==> cloning llama.cpp"
    git clone --filter=blob:none https://github.com/ggml-org/llama.cpp "$SRC"
fi

cd "$SRC"
if ! git cat-file -e "$PIN^{commit}" 2>/dev/null; then
    echo "==> fetching $PIN"
    # Draft PR commits are not on any branch; fetch the PR head that carries it.
    git fetch origin "pull/28243/head:pr28243" --force || git fetch origin
fi
git merge --abort 2>/dev/null || true
git checkout --detach "$PIN"

# Upstream master merged on top of the pinned PR branch (2026-09-15).
# The pin (#28243) is an unmerged draft, so upstream fixes cannot arrive by moving
# the pin. They are merged in instead, at a fixed master commit so the build stays
# reproducible. Key contents: #28330 (skip the unused indexer V cache on qwen4exp),
# #28390 (single-device drafter skips the meta backend wrapper), #28302 (context-
# checkpoint eviction only when the list is full), #28896 (qwen4exp rms_norm+mul
# fusion). Measured against the earlier 311d4211 merge: decode +7.8% single stream,
# +29.8% at 3 concurrent requests; MTP acceptance unchanged.
MASTER_PIN="${LLAMA_MASTER_PIN:-38a5b42d9a3e82e0a586bcd1caed121f36c87a73}"
if ! git cat-file -e "$MASTER_PIN^{commit}" 2>/dev/null; then
    echo "==> fetching master for $MASTER_PIN"
    git fetch origin master
fi
echo "==> merging master $MASTER_PIN"
if ! git -c user.name=build -c user.email=build@local merge --no-commit --no-ff "$MASTER_PIN"; then
    if [ -z "$(git diff --name-only --diff-filter=U)" ]; then
        echo "merge failed, and not because of a conflict" >&2
        exit 1
    fi
fi

# #28896 reshaped the qwen4exp norm gammas to { n_embd, hc } and fused rms_norm+mul,
# while the pin still creates them as { hc_dim }, so src/models/qwen4exp.cpp conflicts.
# The resolution keeps master's shapes and fusion, and applies the same reshape to
# layer.nextn.hc_head_norm, which git does not flag but build_hc_mix needs in the MTP
# graph. Applying it to the pin's file is deterministic, so the result is checksummed.
RESOLVED_SHA256="41a0ca03b2a3684c9948f728b2957bd59df78b168505006830f234fb3d175707"
UNMERGED="$(git diff --name-only --diff-filter=U)"
if [ -n "$UNMERGED" ]; then
    if [ "$UNMERGED" != "src/models/qwen4exp.cpp" ]; then
        echo "unexpected merge conflicts, resolve by hand:" >&2
        echo "$UNMERGED" >&2
        exit 1
    fi
    echo "==> resolving the qwen4exp.cpp conflict"
    git show "$PIN:src/models/qwen4exp.cpp" > src/models/qwen4exp.cpp
    git apply "$REPO_DIR/scripts/llama-patches/qwen4exp-merge-38a5b42d9.patch"
    GOT_SHA256="$(shasum -a 256 src/models/qwen4exp.cpp | cut -d' ' -f1)"
    if [ "$GOT_SHA256" != "$RESOLVED_SHA256" ]; then
        echo "resolved qwen4exp.cpp does not match the tested build" >&2
        echo "  expected $RESOLVED_SHA256" >&2
        echo "  got      $GOT_SHA256" >&2
        exit 1
    fi
    git add src/models/qwen4exp.cpp
fi
git -c user.name=build -c user.email=build@local commit --no-edit \
    -m "merge upstream master $MASTER_PIN into qwen4exp/mtp pin"

# The point of this script is to rebuild what production runs, so assert it. The
# checks above cover the conflict path only, and a future MASTER_PIN can merge
# this file cleanly but differently. The tree hash covers every file and every path.
EXPECTED_TREE="91123c7e3b211337fcdbb9a0ecc033b5b279184b"
GOT_TREE="$(git rev-parse HEAD^{tree})"
if [ "$GOT_TREE" != "$EXPECTED_TREE" ]; then
    echo "merged tree does not match the tested build" >&2
    echo "  expected $EXPECTED_TREE" >&2
    echo "  got      $GOT_TREE" >&2
    echo "  update EXPECTED_TREE only after you benchmark the new binary" >&2
    exit 1
fi

echo "==> HEAD: $(git log -1 --format='%h %ad %s' --date=short)"

# A moved build tree keeps absolute @rpath entries and the binary will not start,
# so configure from scratch whenever the cache points somewhere else.
if [ -f build/CMakeCache.txt ] && ! grep -q "CMAKE_HOME_DIRECTORY:INTERNAL=$SRC\$" build/CMakeCache.txt; then
    echo "==> stale CMake cache (tree moved); removing build/"
    rm -rf build
fi

cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON \
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --config Release -j"$JOBS" --target llama-server

BIN="$SRC/build/bin/llama-server"
echo "==> built: $("$BIN" --version 2>&1 | head -1)"
echo
echo "Point slm_server at it by setting this in .env:"
echo "  SLM_LLAMA_SERVER_BIN=$BIN"
