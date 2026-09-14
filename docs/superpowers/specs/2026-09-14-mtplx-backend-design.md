# MTPLX backend for slm_server — design

- **Date:** 2026-09-14
- **Status:** Draft for owner review. No code exists.
- **Context:** FRE-1502 planner probe and FRE-1517 primary-model study. Both served MTPLX packs with a
  throwaway script that stops the whole slm_server stack. This design replaces that script with a
  managed backend.
- **MTPLX version examined:** 2.11.2. Facts marked *source* come from reading its code. Facts marked
  *live* come from real requests on 2026-09-14.

## Goal

Let slm_server start, route to, and restart an MTPLX server exactly as it does a llama.cpp server,
selected per entry in `config/models.yaml`. llama.cpp stays the default backend.

## Non-goals

- Running `mtplx tune` from slm_server. The operator tunes separately and writes the result into config.
- A generation-aware health probe. That work is tracked separately (FRE-1474) and applies to every
  backend.
- Batched MTP for concurrent requests. MTPLX serves one request at a time by default; batching stays an
  opt-in field.
- Changing FRE-1517. The study keeps the manual script until it ends.

## 1. Config and launcher

`backend: "mtplx"` joins `mlx`, `llamacpp` and `mlx-rerank`. `start_model_server` gets one new branch that
calls a new `build_mtplx_command()`.

### Reused fields

| Field | MTPLX flag |
|---|---|
| `id` | `--model-id` |
| `port` | `--port`, always with `--host 127.0.0.1 --no-auth` |
| `model_path` | `--model` |
| `context_length` | `--context-window` |
| `reasoning_parser` | `--reasoning-parser` (allowlist: `qwen3`, `step3p5`, `gemma4`, `poolside_v1`, `none`) |
| `temp`, `top_p`, `top_k`, `presence_penalty` | `--default-temperature`, `--default-top-p`, `--default-top-k`, `--default-presence-penalty` |

### New optional fields (MTPLX entries only, except `peak_memory_gib`)

| Field | Values | Flag | When unset |
|---|---|---|---|
| `mtp_depth` | integer 1–6 | `--depth` | Validation error. `mtplx serve` ignores saved tuning (*source*). |
| `reasoning_effort` | `low`, `medium`, `high`, `xhigh`, `auto` | `--reasoning-effort` | Not passed |
| `preserve_thinking` | `off`, `on`, `auto`, `scoped` | `--preserve-thinking` | `off`. The MTPLX default `auto` resolved to on (*live*). |
| `mtplx_profile` | `stable`, `performance-cold`, `sustained`, `turbo`, `exact`, `max-diagnostic` | `--profile` | Not passed |
| `mtplx_batching_preset` | `solo`, `latency`, `agent`, `throughput` | `--batching-preset` | Not passed (serial, one request) |
| `mtplx_fan_mode` | `default`, `smart`, `max` | `--fan-mode` | Not passed (Apple automatic fan control) |
| `peak_memory_gib` | number > 0 | none | See section 4. Applies to every backend. |

### Rules

- `model_path` must be a local directory that contains `config.json` and `mtplx_runtime.json`. A Hugging
  Face id is rejected, so a start never downloads.
- `reasoning_effort`, `preserve_thinking`, `mtp_depth` and the `mtplx_*` fields apply only to MTPLX
  entries. On any other backend, validation warns that the field is ignored. llama.cpp entries keep
  setting effort through `chat_template_kwargs`.
- `mtp_depth` is checked against 1–6 only. slm_server does not read the pack's `mtplx_runtime.json` limit
  (`mtp_depth_max`, 3 for both Optimized-Speed packs), so `mtplx serve` may reject a higher depth at
  startup. That failure reaches the log through the existing early-exit check.
- Binary lookup, in order: `SLM_MTPLX_BIN`, `mtplx` on `PATH`, `~/.mtplx/bin/mtplx`.
- stderr goes to `logs/mtplx-<id>-<port>.log`, the same pattern as the other backends.
- The watchdog, `stop.sh` and restarts need no change. They work by process and port, and the `mtplx`
  wrapper `exec`s its Python process, so the launcher's PID is the server's PID.

## 2. Router request translation

MTPLX reads `chat_template_kwargs.enable_thinking` but not `chat_template_kwargs.reasoning_effort`, which
it accepts only as a top-level field (*source*: `_thinking_enabled_for_request`,
`_reasoning_effort_for_state`).

For entries with `backend == "mtplx"` only, one helper runs in the three places that handle
`chat_template_kwargs` today (`router.py`: chat completions and both `/v1/responses` fallbacks):

1. Merge the entry's config `chat_template_kwargs` with the request's `chat_template_kwargs`, per key,
   request keys winning. This matches llama-server's merge.
2. If the merged dict holds `reasoning_effort` and the request has no top-level `reasoning_effort`, set the
   top-level field from it.
3. Forward the merged `chat_template_kwargs` unchanged, so MTPLX still reads `enable_thinking`.
4. Everything else passes through: `model`, sampling fields, tools, `response_format`, `stream_options`,
   and every request header except `content-length`, `host`, `connection` and `transfer-encoding`
   (already the router's rule). `X-Session-Id`, `X-Trace-Id`, `X-Span-Id` and `traceparent` therefore
   reach MTPLX. MTPLX keys its session cache on `X-Session-Id` (*live*).

The launch flags from section 1 set effort and preserve-thinking defaults for every request, so they apply
even when a request carries its own kwargs without those keys.

Unchanged behaviour:

- Routing by `id`. An unknown or disabled id gets the router's existing error, so MTPLX's habit of
  answering any model name never shows.
- Streaming bytes pass through as they arrive: MTPLX's `: keep-alive` comments before the first token
  (every 5 s by default, *source*), `finish_reason` values including `length` with `tool_calls`, and the
  final usage chunk with `prompt_tokens_details.cached_tokens` and
  `completion_tokens_details.reasoning_tokens` (*live*).
- Client disconnect: the router already closes the backend stream (`response.aclose()`), and MTPLX cancels
  on disconnect (*source*). This path is untested and is part of live verification.
- Telemetry fields read from llama.cpp's `timings` block stay empty for MTPLX.

## 3. Errors and startup

1. **Readiness wait (all backends).** MTPLX opens its port only after the model load and warm-up
   (*source*: `ServerState(args)` runs before `uvicorn.run`). `start.sh` waits about 40 s today. The fixed
   loop becomes `SLM_BACKEND_READY_TIMEOUT`, default 180 s. A healthy start takes the same time as today.
2. **HTTP 507 is a caller fault (all backends).** MTPLX returns 507 when a request does not fit in
   memory. `watchdog.classify_status` counts every status from 500 up as a backend failure, so repeated
   refusals could restart a healthy backend. 507 joins `CALLER_FAULT_STATUSES`. MTPLX's OpenAI-style error
   body passes through unchanged.
3. **Missing binary.** The launcher logs `mtplx_binary_not_found` with the searched paths and skips that
   backend. Other backends still start.
4. **Invalid config.** A missing or out-of-range `mtp_depth`, a bad `model_path`, or an unknown enum value
   is reported by config validation and raised as `ValueError` by the builder. The launcher already catches
   that and logs `invalid_config_parameters`.
5. **Crash during load.** The existing 0.5 s early-exit check catches immediate crashes. A later crash
   reaches the watchdog after the 90 s startup grace, with its limit of 5 restarts in 10 minutes.
6. **Host.** The builder always binds `127.0.0.1` with `--no-auth`. A `host` value other than `0.0.0.0` or
   `127.0.0.1` produces a warning, because MTPLX would require an API key.
7. **Stalls.** The router's 300 s no-bytes rule works unchanged because of the keep-alive comments.
   MTPLX's own 300 s stream-stall deadline stays at its default. The launcher never sets
   `MTPLX_SSE_HEARTBEAT=0`.
8. **Fan mode without the daemon.** `smart` and `max` need the privileged thermalforge daemon
   (`/tmp/thermalforge.sock`). Without it MTPLX logs a warning and serves normally (*source*). Validation
   warns when `mtplx_fan_mode` is `smart` or `max` and the socket is missing.

## 4. Memory guard (all backends)

One heavy engine at a time is the intended operating mode: memory pressure and heat change decode speed,
and Seshat's `slm_local` provider shares one concurrency pool across every local model.

- New optional entry field `peak_memory_gib` (declared peak resident memory, number > 0).
- New environment variable `SLM_MEMORY_BUDGET_GIB`, default `100`.
- New function `check_memory_budget(config)`. It sums `peak_memory_gib` over enabled entries. If the sum
  exceeds the budget, it returns an error message that names each entry and its value.
- The `backends` launcher calls it before starting any backend and exits non-zero on an error. The router
  is not affected.
- Enabled entries without `peak_memory_gib` get a warning, and they count as zero. Existing configs
  therefore keep starting.

Reference values measured on 2026-09-14: llama.cpp Flash-Next UD-IQ4_XS about 87 GiB; MTPLX Flash-Next
Optimized-Speed peak 84.2 GB; MTPLX 27B Optimized-Speed peak 23.0 GB.

## 5. Seshat requirements (from the FRE-1517 review)

| Requirement | Where it is met |
|---|---|
| Unknown id gets a clear 4xx | Section 2, existing router routing; pinned by a test |
| Session and trace headers pass through | Section 2 |
| Configured effort and preserve-thinking survive request kwargs | Section 1 launch flags, section 2 merge |
| Keep-alives, `finish_reason` with `tool_calls`, usage details pass through | Section 2; pinned by tests |
| A 507 that litellm does not retry three times | Section 3 item 2, plus a live check. If litellm retries a 507, a follow-up maps MTPLX's 507 to a 400 `context_length_exceeded` shape. |
| Client-disconnect cancel | Live verification |
| No model-specific timeouts | Nothing to do |

## 6. Tests

pytest with monkeypatching (no `unittest.mock`) and `tmp_path` fake packs, as in the existing suite.

- `tests/test_start_backends_mtplx.py`: every flag of `build_mtplx_command`; the `preserve_thinking`
  default; rejection of a missing or out-of-range depth, a pack without its two files, and a Hugging Face
  id; binary lookup order; the `mtplx` branch of `start_model_server`; the log file name.
- `tests/test_config_mtplx.py`: warnings for llama.cpp-only fields on an MTPLX entry, `max_concurrency`
  above 1 with a serial preset, an unusual `host`, and fan mode without the daemon socket.
- `tests/test_memory_budget.py`: under budget, over budget (error names entries), entries without
  `peak_memory_gib` (warning), budget from the environment, launcher exits before starting a backend.
- `tests/test_router_mtplx.py`: the kwargs merge and top-level effort mapping; the forwarded body on chat
  completions and both `/v1/responses` fallbacks; header pass-through; keep-alive comment lines,
  `finish_reason: length` with `tool_calls`, and the usage chunk passed through unchanged; non-MTPLX
  bodies unchanged; unknown id status code.
- `tests/test_watchdog.py` (extended): 507 is a caller fault.
- `start.sh` has no automated tests. The `SLM_BACKEND_READY_TIMEOUT` change is checked by hand in live
  verification (section 8, step 7) and by `bash -n`.

Before each commit: `uv run pytest`, `uv run ruff check src/ tests/`, `uv run ruff format --check src/
tests/`, `uv run mypy src/`.

## 7. Documentation

- README: an MTPLX backend section covering the fields, the separate tune step, the fan daemon, the memory
  guard, and the one-heavy-engine rule.
- `config/models.yaml.example`: one disabled MTPLX example entry with placeholder paths.
- CLAUDE.md: the module descriptions for `start_backends.py`, `router.py`, `config.py` and `watchdog.py`.

## 8. Rollout

1. **Code on a branch, one tested task per commit.** No push without the owner's request. No change to the
   running server or to the live `config/models.yaml`.
2. **Live verification after FRE-1517 arm 4 ends,** in a window the owner approves (about 30 minutes, with
   engine swaps). Steps:
   1. Add disabled MTPLX entries with `peak_memory_gib`, enable the 27B entry, run `./start.sh`.
   2. `/v1/models` lists the MTPLX id; a real completion returns 200.
   3. A request with `chat_template_kwargs.reasoning_effort` resolves to that effort in the MTPLX request
      log.
   4. A streamed request ends with a usage chunk that keeps `cached_tokens` and `reasoning_tokens`.
   5. Killing the MTPLX process leads to a watchdog restart.
   6. A client disconnect mid-stream cancels the generation (MTPLX request log shows it cancelled), and the
      next request is not delayed.
   7. A cold start finishes within `SLM_BACKEND_READY_TIMEOUT`.
   8. An oversize request is refused without a watchdog restart. Record the status MTPLX returns.
   9. Send that oversize request through litellm with `num_retries` 3 and count how many requests reach
      MTPLX. If more than one, open the 507-mapping follow-up.
   10. Enabling a second heavy entry makes the launcher refuse to start.
   11. Restore the pre-verification config and confirm llama.cpp Flash-Next serves.

## Known limits found during testing

- Prefix reuse inside a turn is complete on MTPLX; reuse across turns was partial in a synthetic test and
  is being measured in FRE-1517 from live traffic.
- `mtplx tune` accepts depths 1–3 only.
- MTPLX treats a request with `max_tokens` of 48 or less and a `[system, user]` shape as background work
  and skips its session cache (*source*: `is_background_request`). Health probes and tests must avoid that
  shape when they measure caching.
