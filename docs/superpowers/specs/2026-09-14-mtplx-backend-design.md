# MTPLX backend for slm_server — design

- **Date:** 2026-09-14 (revision 2, after the cc-master review)
- **Status:** Draft for owner review. No code exists.
- **Context:** FRE-1502 planner probe and FRE-1517 primary-model study. Both served MTPLX packs with a
  throwaway script that stops the whole slm_server stack. This design replaces that script with a
  managed backend.
- **Base branch:** `mtplx-backend` builds on `llamacpp-pinned-build`, which is not yet on `main`.
  Citations refer to that branch.
- **MTPLX version examined:** 2.11.2. Facts marked *source* come from reading its code. Facts marked
  *live* come from real requests on 2026-09-14.

## Goal

Let slm_server start, route to, and restart an MTPLX server exactly as it does a llama.cpp server,
selected per entry in `config/models.yaml`. llama.cpp stays the default backend.

## Non-goals

- Running `mtplx tune` from slm_server. The operator tunes separately and writes the result into config.
- Generation-aware health. The mechanism is an owner decision spanning FRE-1474 (Seshat's health probe)
  and slm_server's watchdog, and it applies to every backend.
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

### New optional fields

| Field | Values | Flag | When unset |
|---|---|---|---|
| `mtp_depth` | integer 1 to the pack's `mtp_depth_max` | `--depth` | Validation error. `mtplx serve` ignores saved tuning (*source*). |
| `reasoning_effort` | `low`, `medium`, `high`, `xhigh`, `auto` | `--reasoning-effort` | Not passed |
| `preserve_thinking` | `off`, `on`, `auto`, `scoped` | `--preserve-thinking` | `off`. The MTPLX default `auto` resolved to on (*live*). |
| `mtplx_profile` | `stable`, `performance-cold`, `sustained`, `turbo`, `exact`, `max-diagnostic` | `--profile` | Not passed |
| `mtplx_batching_preset` | `solo`, `latency`, `agent`, `throughput` | `--batching-preset` | Not passed (serial, one request) |
| `mtplx_fan_mode` | `default`, `smart`, `max` | `--fan-mode` | Not passed (Apple automatic fan control) |
| `peak_memory_gib` | number > 0 | none | See section 4. Applies to every backend. |

### Rules

- `model_path` must be a local directory that contains `config.json` and `mtplx_runtime.json`. A Hugging
  Face id is rejected, so a start never downloads.
- Validation reads `mtp_depth_max` from the pack's `mtplx_runtime.json` (3 for both Optimized-Speed packs,
  6 for 27B Bare-Speed) and rejects a larger `mtp_depth`. A bad depth then fails in config, not in a
  startup log. If the file has no `mtp_depth_max`, the range is 1–6.
- `reasoning_effort`, `preserve_thinking`, `mtp_depth` and the `mtplx_*` fields apply only to MTPLX
  entries. On any other backend, validation warns that the field is ignored. llama.cpp entries keep
  setting effort through `chat_template_kwargs`.
- On an MTPLX entry, `reasoning_effort` or `preserve_thinking` inside the config `chat_template_kwargs` is
  a validation error. The top-level fields are the only config source, so they cannot disagree with the
  launch flags.
- Binary lookup, in order: `SLM_MTPLX_BIN`, `mtplx` on `PATH`, `~/.mtplx/bin/mtplx`.
- stderr goes to `logs/mtplx-<id>-<port>.log`, the same pattern as the other backends.
- The watchdog, `stop.sh` and restarts need no change. They work by process and port, and the `mtplx`
  wrapper `exec`s its Python process, so the launcher's PID is the server's PID.

## 2. Router request translation

MTPLX reads `chat_template_kwargs.enable_thinking` but not `chat_template_kwargs.reasoning_effort`, which
it accepts only as a top-level field (*source*: `_thinking_enabled_for_request`,
`_reasoning_effort_for_state`).

For entries with `backend == "mtplx"` only, one helper runs in the three places that handle
`chat_template_kwargs` today (`router.py` chat completions and both `/v1/responses` fallbacks):

1. Merge the entry's config `chat_template_kwargs` with the request's `chat_template_kwargs`, per key,
   request keys winning. This matches llama-server's merge.
2. If the **request's own** `chat_template_kwargs` holds `reasoning_effort` and the request has no
   top-level `reasoning_effort`, set the top-level field from it. Config kwargs never hold that key on an
   MTPLX entry (section 1 rules), so the configured effort comes only from the launch flag.
3. Forward the merged `chat_template_kwargs`, so MTPLX still reads `enable_thinking`.
4. Everything else passes through: `model`, sampling fields, tools, `response_format`, `stream_options`,
   and every request header except `content-length`, `host`, `connection` and `transfer-encoding`
   (`_filtered_forward_headers`). `X-Session-Id`, `X-Trace-Id`, `X-Span-Id` and `traceparent` therefore
   reach MTPLX. MTPLX keys its session cache on `X-Session-Id` (*live*).

### Model ids

- An unknown id returns 404 (existing behaviour).
- A configured but disabled id returns **404** with the detail "configured but currently disabled".
  Today it returns 503 (`_get_model_definition`), which clients retry as a server error. In a
  one-heavy-engine setup, a request for the engine that is not loaded is normal during swaps.
- Router-raised errors are never scored against a backend. The outcome middleware scores a response only
  after an endpoint sets `request.state.backend_port`, and a failed model lookup never sets it
  (`router.py` outcome middleware). A test pins this.

### Streaming and non-streaming

- Streaming requests pass bytes through as they arrive (`_iter_with_heartbeat`): MTPLX's `: keep-alive`
  comments before the first token (every 5 s by default, *source*), `finish_reason` values including
  `length` with `tool_calls`, and the final usage chunk with `prompt_tokens_details.cached_tokens` and
  `completion_tokens_details.reasoning_tokens` (*live*).
- Non-streaming requests have no keep-alive. The router buffers the whole response (`client.post`), so
  the client sees no bytes until MTPLX finishes, and a Cloudflare 120 s limit on the client path still
  applies. Seshat always streams.

### Client disconnect

- The router closes the backend stream when its client disconnects (`response.aclose()`, verified in
  code).
- MTPLX polls `is_disconnected()` every 0.25 s and cancels generation when the connection closes
  (*source*: `_monitor_request_disconnect`).
- Observed limit: on 2026-09-14 13:01 UTC a stream that ran cloudflared → MTPLX directly, without the
  router, kept generating to a normal stop after the client lost it (*live*). The upstream break did not
  close cloudflared's local connection, so MTPLX never saw a disconnect. Behind the router the local
  connection closes explicitly, but that path is untested. Live verification must confirm the cancel.

### Telemetry

For every backend, the request span takes `slm.cache_reuse` from `timings.cache_n` when llama.cpp
`timings` are present, otherwise from `usage.prompt_tokens_details.cached_tokens` (MTPLX's case), and
adds `usage.completion_tokens_details.reasoning_tokens`. llama.cpp spans are unchanged where timings
exist.

## 3. Errors and startup

1. **Readiness wait (all backends).** MTPLX opens its port after the model load and the foreground
   warm-up (*source*: `ServerState` runs `_run_startup_warmup` before `uvicorn.run`). An extended warm-up
   then continues in the background after the port opens (*source*, `[6/6] Extended warmup continues
   silently in the background`). It yields to real requests (*live*: step states `yielded`,
   `waiting_idle`), so it does not block or fail them. On 2026-09-14 the port answered 10 s after start and
   the background warm-up finished at 111 s (*live*).
   - `start.sh` waits about 40 s for ports today. The fixed loop becomes `SLM_BACKEND_READY_TIMEOUT`,
     default 180 s, to cover a cold load. A healthy start takes the same time as today.
   - The watchdog's 90 s startup grace is unchanged. A slow request during the background warm-up is not a
     watchdog failure (only 5xx, 408/425/429/504 and a 300 s first-byte stall count). The open risk is a
     cold load longer than 90 s; live verification measures it, and `SLM_WATCHDOG_STARTUP_GRACE_SECONDS`
     covers it if needed.
   - Clients that measure timing must wait for `/health` `warmup.background.state` to leave `running`
     before they start.
2. **HTTP 507 is ignored by the watchdog (all backends).** MTPLX returns 507 when a request does not fit in
   memory. Today `classify_status` counts every status from 500 up as a failure. 507 becomes an `ignore`
   verdict: it neither trips a restart nor resets the failure streak, so a backend that refuses everything
   under memory pressure does not look healthy. MTPLX's OpenAI-style error body passes through unchanged.
3. **Missing binary.** The launcher logs `mtplx_binary_not_found` with the searched paths and skips that
   backend. Other backends still start.
4. **Invalid config.** A bad `mtp_depth`, a bad `model_path`, an unknown enum value, or effort keys in an
   MTPLX entry's config kwargs is reported by config validation and raised as `ValueError` by the builder.
   The launcher already catches that and logs `invalid_config_parameters`.
5. **Crash during load.** The existing 0.5 s early-exit check catches immediate crashes. A later crash
   reaches the watchdog after the 90 s startup grace, with its limit of 5 restarts in 10 minutes.
6. **Host.** The builder always binds `127.0.0.1` with `--no-auth`. A `host` value other than `0.0.0.0` or
   `127.0.0.1` produces a warning, because MTPLX would require an API key.
7. **Stalls.** The router's 300 s first-byte stall rule does not detect a wedged MTPLX. In the streaming
   loop, `in_flight.first_byte()` fires on every backend chunk except the router's own heartbeat
   sentinel, and MTPLX's `: keep-alive` comments are backend chunks. A wedged MTPLX that keeps emitting
   keep-alives is therefore never flagged as a stall. llama.cpp's 30 s SSE ping has the same effect, so
   this gap is not MTPLX-specific.
   - The keep-alives still matter: without them the rule would kill MTPLX during a long cold prefill.
     The launcher never sets `MTPLX_SSE_HEARTBEAT=0`.
   - MTPLX's own stream-stall deadline (300 s by default, *source*) fails a stream whose model owner makes
     no progress. The router then sees a broken stream and `in_flight.failed()` records a backend failure.
     Stall detection for MTPLX therefore depends on that deadline staying enabled.
   - Closing the gap needs generation-aware health (the mechanism is an owner decision spanning FRE-1474
     and slm_server's watchdog). FRE-1474 as written recommends a Seshat-side generation probe, which
     detects a dead backend for monitoring but cannot restart one. Restarting a wedged backend that
     still emits keep-alives needs wedge detection in slm_server's watchdog, for example a periodic short
     completion per enabled backend that feeds `record_failure`. The urgency of that decision rises with
     this backend.
8. **Fan mode without the daemon.** `smart` and `max` need the privileged thermalforge daemon
   (`/tmp/thermalforge.sock`). Without it MTPLX logs a warning and serves normally (*source*). Validation
   warns when `mtplx_fan_mode` is `smart` or `max` and the socket is missing.

## 4. Memory guard (all backends)

One heavy engine at a time is the intended operating mode: memory pressure and heat change decode speed,
and Seshat's `slm_local` provider shares one concurrency pool across every local model.

- New optional entry field `peak_memory_gib` (declared peak resident memory, number > 0).
- New environment variable `SLM_MEMORY_BUDGET_GIB`, default `100`.
- A heavy entry is an enabled entry with `model_type` `lm` or `multimodal`. Rerank and embeddings entries
  are not heavy.
- New function `check_memory_budget(config)` returns an error when:
  - two or more heavy entries are enabled and any of them lacks `peak_memory_gib`, or
  - the sum of `peak_memory_gib` over enabled entries exceeds the budget.
  The message names each entry and its value.
- With a single heavy entry enabled and no `peak_memory_gib`, it returns a warning only, so today's
  single-model configs keep starting.
- The `backends` launcher calls it before starting any backend and exits non-zero on an error. The router
  is not affected.

Reference values measured on 2026-09-14: llama.cpp Flash-Next UD-IQ4_XS about 87 GiB; MTPLX Flash-Next
Optimized-Speed peak 84.2 GB; MTPLX 27B Optimized-Speed peak 23.0 GB.

## 5. Seshat requirements (from the FRE-1517 review)

| Requirement | Where it is met |
|---|---|
| Unknown or disabled id gets a 4xx, never scored against a backend | Section 2 "Model ids"; pinned by tests |
| Session and trace headers pass through | Section 2 |
| Configured effort and preserve-thinking cannot be overridden by config kwargs | Section 1 rules; section 2 step 2 |
| Keep-alives, `finish_reason` with `tool_calls`, usage details pass through | Section 2; pinned by tests |
| A 507 that litellm does not retry three times | Section 3 item 2, plus a live check. If litellm retries a 507, a follow-up maps MTPLX's 507 to a 400 `context_length_exceeded` shape. |
| Client-disconnect cancel | Section 2; live verification |
| Cache telemetry for MTPLX | Section 2 "Telemetry" |
| No model-specific timeouts | Nothing to do |

## 6. Tests

pytest with monkeypatching (no `unittest.mock`) and `tmp_path` fake packs, as in the existing suite.

- `tests/test_start_backends_mtplx.py`: every flag of `build_mtplx_command`; the `preserve_thinking`
  default; rejection of a missing depth, a depth above the pack's `mtp_depth_max`, a pack without its two
  files, and a Hugging Face id; binary lookup order; the `mtplx` branch of `start_model_server`; the log
  file name.
- `tests/test_config_mtplx.py`: warnings for llama.cpp-only fields on an MTPLX entry and for MTPLX-only
  fields on other backends; the error for effort keys in an MTPLX entry's config kwargs;
  `max_concurrency` above 1 with a serial preset; an unusual `host`; fan mode without the daemon socket.
- `tests/test_memory_budget.py`: one heavy entry without `peak_memory_gib` (warning); two heavy entries
  with one missing the field (error); under and over budget; rerank entries not counted as heavy; budget
  from the environment; launcher exits before starting any backend.
- `tests/test_router_mtplx.py`: the kwargs merge; the effort lift from request kwargs only; the forwarded
  body on chat completions and both `/v1/responses` fallbacks; header pass-through; keep-alive comment
  lines, `finish_reason: length` with `tool_calls`, and the usage chunk passed through unchanged; the
  telemetry fields; non-MTPLX bodies unchanged.
- `tests/test_router_model_selection.py` (changed): a disabled id returns 404 with the disabled detail
  (was 503); an unknown or disabled id is not recorded by the watchdog.
- `tests/test_watchdog.py` (extended): 507 returns the `ignore` verdict.
- `start.sh` has no automated tests. The `SLM_BACKEND_READY_TIMEOUT` change is checked by hand in live
  verification (section 8, step 7) and by `bash -n`.

Before each commit: `uv run pytest`, `uv run ruff check src/ tests/`, `uv run ruff format --check src/
tests/`, `uv run mypy src/`.

## 7. Documentation

- README: an MTPLX backend section covering the fields, the separate tune step, the fan daemon, the memory
  guard, the one-heavy-engine rule, the background warm-up, and the non-streaming limit.
- `config/models.yaml.example`: one disabled MTPLX example entry with placeholder paths.
- CLAUDE.md: the module descriptions for `start_backends.py`, `router.py`, `config.py` and `watchdog.py`.

## 8. Rollout

1. **Merge order (owner decision).** `llamacpp-pinned-build` is 11 commits ahead of `main` with no pull
   request, and it holds the watchdog request-error changes this design relies on. It needs its own
   review and merge before `mtplx-backend`.
2. **Code on `mtplx-backend`, one tested task per commit.** No push without the owner's request. No change
   to the running server or to the live `config/models.yaml`.
3. **Live verification after FRE-1517 arm 4 ends,** in a window the owner approves (about 45 minutes, with
   engine swaps). Steps:
   1. Add disabled MTPLX entries with `peak_memory_gib`, enable the 27B entry, run `./start.sh`.
   2. `/v1/models` lists the MTPLX id; a real completion returns 200. A request for a disabled id returns
      404 and does not appear in the watchdog log.
   3. A request with `chat_template_kwargs.reasoning_effort` resolves to that effort in the MTPLX request
      log.
   4. A streamed request ends with a usage chunk that keeps `cached_tokens` and `reasoning_tokens`, and the
      slm_server span carries both.
   5. One Seshat eval turn through the integrated router (Seshat side run by cc-master): served-id check;
      `cached_tokens > 0` on the second tool round of one turn with `X-Session-Id`; a worker landing call
      with tools, `tool_choice: "none"` and a `json_schema` response format; a tool call cut at length
      reaching Seshat with `finish_reason: "length"`.
   6. Killing the MTPLX process leads to a watchdog restart.
   7. A cold start (after `purge` or a reboot) finishes within `SLM_BACKEND_READY_TIMEOUT`; record the time
      to port and the time to `warmup.background.state` done, and compare the port time with the 90 s
      startup grace.
   8. A client disconnect mid-stream through the router cancels the generation (MTPLX request log shows it
      cancelled), and the next request is not delayed.
   9. An oversize request is refused without a watchdog restart. Record the status MTPLX returns.
   10. Send that oversize request through litellm with `num_retries` 3 and count how many requests reach
       MTPLX. If more than one, open the 507-mapping follow-up.
   11. Enabling a second heavy entry makes the launcher refuse to start.
   12. Restore the pre-verification config and confirm llama.cpp Flash-Next serves.

## Known limits found during testing

- Prefix reuse inside a turn is complete on MTPLX; reuse across turns was partial in a synthetic test and
  is being measured in FRE-1517 from live traffic.
- `mtplx tune` accepts depths 1–3 only.
- MTPLX treats a request with `max_tokens` of 48 or less and a `[system, user]` shape as background work
  and skips its session cache (*source*: `is_background_request`). Health probes and tests must avoid that
  shape when they measure caching.
