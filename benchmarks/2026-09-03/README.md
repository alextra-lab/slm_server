# Benchmark data, 2026-09-03

Kept for the A/B of `qwen3.6-35-A3B` against `qwen3.8-flash-next` with MTP.

## Stats

| File | Server / model | Records |
|------|----------------|---------|
| `stats/mtp_matrix.jsonl` | **both models, MTP depth 0-3, build 10770** | 32 |
| `stats/short_matrix.jsonl` | **both models, 4 prompts, depth 1, build 10770** | 8 |
| `stats/edges_lcpp.jsonl` | llama.cpp qwen3.6 Q6 + Q4 subagent, build 10621 | 46 |
| `stats/edges_flash.jsonl` | llama.cpp flash-next, build 10770 | 23 |
| `stats/edges.jsonl` | mtplx qwen3.8-flash-next | 21 |
| `stats/edges36.jsonl` | mtplx qwen3.6-35b-a3b | 18 |
| `stats/edges38.jsonl` | mtplx qwen3.8-27b | 24 |
| `stats/concurrency.jsonl` | **flash-next concurrency 1/2/3, cold start, over-subscription** | 11 |

The first two are the like-for-like comparison: same binary, same prompts,
temperature 0. The rest predate that and mix builds and effort levels — read
them with that caveat.

## Harness

- `harness/mtp_matrix.py` — launches each model at MTP depth 0/1/2/3, records
  decode tok/s and wall time. Edit `MODELS` for paths. **Importing this module
  runs the matrix**; the loop is at module level.
- `harness/short_matrix.py` — the 4-prompt comparison at a fixed depth.
- `harness/edges*.py` — budget, difficulty and reasoning-effort sweeps.
- `harness/concurrency.py` — concurrency scaling, MTP accept rate under load,
  cold-start penalty, over-subscription. Fixed token count per request and a
  unique prefix per prompt, so the aggregate figure is not inflated by shared KV.

## Findings that shape the A/B

- **Per token, qwen3.6 is ~2.1x faster**: 77-93 tok/s against 36-45.
- **Per answer, flash-next emits 2-6x fewer tokens**, so wall-clock favours it
  on short prompts (2.1-2.3x) and ties on a 600-word essay.
- **MTP depth 1 is optimal for both.** Accept rates fall from 92% at depth 1 to
  62% at depth 3, and net speed drops. Unsloth's README suggests depth 2 for
  flash-next; depth 1 measured better here.
- **Flash-next needs the pinned build.** Homebrew's llama.cpp cannot load it.
  See `scripts/build_llama.sh` and `config/llama.cpp.pin`.
- **Set `max_tokens` >= 4000 with reasoning on.** qwen3.6 on :8502 returns
  either 1788 or 3330 reasoning tokens for an identical request depending on
  what ran before it. Too small a budget returns empty content.
- **`reasoning_effort` works on flash-next, not on qwen3.6** — the difference is
  the chat template, not the server.

## Concurrency findings (2026-09-04)

- **Aggregate throughput rises only 19% from concurrency 1 to 3** (34.2 -> 40.6
  tok/s), while per-request throughput falls 2.57x (37.08 -> 14.42). The GPU is
  memory-bandwidth bound. Three parallel requests finish 26.3s of sequential
  work in 22.2s — a 15.8% saving, not 3x.
- **MTP accept rate holds under load**: 80-82% at 1 concurrent, 77-89% at 3.
  Depth 1 stays the right operating point when batching.
- **Cold start costs ~2s, in prefill only**: 73.5 tok/s prefill on the first
  request after idle against ~275 warm. Decode speed is unaffected.
- **4 concurrent against 3 slots wedged the server.** All four returned HTTP 504
  after 600s and the backend needed a forced restart. Cap clients at 3.
- **An invalid `reasoning_effort` can restart the backend.** The template raises,
  llama-server returns 500, and the watchdog treats 5xx as a backend failure.
  Observed once in `logs/watchdog.jsonl`.

## Reproducing

Both models need `SLM_LLAMA_SERVER_BIN` pointing at the pinned build, and
flash-next needs `spec_model_path` for its sidecar MTP head.
