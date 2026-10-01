#!/usr/bin/env bash
# Build the llama-server binary that slm_server runs.
#
# Why this exists (2026-09-03): Homebrew's llama.cpp v0.3.0 (build 10621) predates
# the qwen4exp architecture merge, so it cannot load Qwen3.8-Flash-Next at all --
# it fails with "unknown model architecture: 'qwen4exp'". PR #28243 adds the MTP
# draft-head support that Flash-Next needs for speculative decoding. That PR is open
# and unmerged upstream, so it is pinned by commit here rather than tracked.
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

# No master merge happens here any more (2026-10-01). Upstream rebased PR #28243 and
# merged master into it at 2026-09-21, and resolved the src/models/qwen4exp.cpp
# conflict this script used to patch by hand, the same way: master's { n_embd, hc }
# norm-gamma shapes plus the matching reshape of layer.nextn.hc_head_norm. The pin is
# that merge, so the checkout alone is the tested source. Earlier pins needed a fixed
# MASTER_PIN merged on top; if upstream stalls again, re-add that step rather than
# moving the pin to a master commit, which has no qwen4exp MTP at all.

# The point of this script is to rebuild what production runs, so assert it. The tree
# hash covers every file, so a pin that is not the tested commit stops the build.
EXPECTED_TREE="1056b367d35e2b030d0b48fe089229880f5dde3a"
GOT_TREE="$(git rev-parse HEAD^{tree})"
if [ "$GOT_TREE" != "$EXPECTED_TREE" ]; then
    echo "checked-out tree does not match the tested build" >&2
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
