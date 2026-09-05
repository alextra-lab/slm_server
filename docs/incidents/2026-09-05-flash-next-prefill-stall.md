# Flash-Next prefill stall, 2026-09-05

## Summary

A multi-turn conversation against `unsloth/qwen3.8-flash-next` on port 8502 hung
on its eighth turn. The backend produced no first byte for 300 seconds, ignored
SIGTERM, and had to be SIGKILLed by the watchdog.

The cause was almost certainly a llama.cpp bug in the speculative-decoding path
for qwen4exp drafts that borrow the target model's embeddings. Upstream fixed it
in commit `d1a9235` on PR #28243, committed 69 minutes before the incident. The
pin moved from `2857e5114` (build 10770) to `d1a92352c` (build 10802).

## Timeline (local time; watchdog logs UTC)

| Time | Event |
|------|-------|
| 07:15:35 - 07:17:46 | Seven turns complete normally, 35-40 tok/s |
| 07:17:47 | Turn 8 routed. No response ever returned |
| 07:19:52, 07:21:57 | Client retries. Both also hang, consuming two more slots |
| 07:22:51 | Watchdog: `stall`, no first byte within 300s |
| 07:23:06 | `watchdog_sigterm_ignored_killing` pid 65830 -> SIGKILL |
| 07:23:06 | Three `RemoteProtocolError` as in-flight requests are torn down |
| 07:23:17 | `restart_succeeded`, new pid 44047 |

Full record: `logs/incidents/2026-09-05-stall.log`.

## The failing session

One session, `cf72ea24`, requests issued **serially**. Prompt grew turn by turn
with cache reuse tracking it at roughly 94%:

| Turn | prompt_tok | cache_reuse | completion | decode |
|------|-----------|-------------|-----------|--------|
| 1 | 2758 | 2401 | 393 | 10.3s |
| 2 | 613 | 125 | 648 | 17.3s |
| 3 | 1458 | 1260 | 662 | 17.1s |
| 4 | 2251 | 2120 | 628 | 16.6s |
| 5 | 3071 | 2879 | 666 | 18.1s |
| 6 | 3955 | 3737 | 684 | 19.3s |
| 7 | 4855 | 4639 | 532 | 14.9s |
| 8 | ~5400 est. | — | **none** | **hung** |

## Cause

Upstream commit `d1a9235`, `common/speculative.cpp`, +3/-1:

> A qwen4exp draft that borrows the target's embeddings sets `ctx_other` but
> keeps its own memory, so it must be caught up and rolled back like any other
> draft. Treating it as memory-shared skipped the catch-up decode and placed
> every draft token at the same position, which the M-RoPE position check
> rejects.

This matches the incident on every axis:

- The architecture is qwen4exp.
- The configured draft head is the **shared** variant,
  `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, which borrows the target's
  embeddings and is therefore the affected case.
- A skipped catch-up decode degrades cumulatively, matching a failure on turn 8
  rather than turn 1.
- The hang was in prefill, before any output token.

**Not proven.** The commit describes a position-check rejection, not a deadlock
that ignores SIGTERM. The link is circumstantial, though strong.

## Ruled out

| Candidate | Evidence against |
|-----------|------------------|
| Over-subscription (>3 slots) | Requests were serial, one at a time |
| Memory pressure | RSS 69.8 GB against a 107.5 GB Metal ceiling; swap unused |
| Invalid `reasoning_effort` | The 500s came after the stall, from the SIGKILL |
| Cold start | Seven turns had just completed |
| Decode slowdown | Previous turn ran at 35.7 tok/s |

The `server_error` watchdog entries at 07:23:06 are a **consequence** of the
restart, not the trigger. The restart decision was already made on `stall`.

## Fix and verification

Pin moved to `d1a92352cbd417fd840b4e765c0b82f5fe3d1d89`, rebuilt with
`scripts/build_llama.sh`.

`benchmarks/2026-09-03/harness/multiturn_stall.py` replays the failing shape.
On the new build it ran **12 of 12 turns** with no stall, reaching 7135 prompt
tokens and 6413 cached — past the point of the original failure:

| Turn | prompt | cached | decode | MTP draft/accepted |
|------|--------|--------|--------|--------------------|
| 8 | 4484 | 3805 | 35.1 tok/s | 331/283 |
| 12 | 7135 | 6413 | 29.9 tok/s | 305/255 |

A subsequent real workload also passed the failure point, reaching
`cache_reuse=4511` with no watchdog events.

One clean run is not proof: the old build managed seven good turns before
failing.

## If it recurs

1. Switch to the self-contained draft head,
   `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` (4.14 GB, not yet downloaded). It carries
   its own embeddings, does not set `ctx_other`, and avoids the affected path.
   Costs ~1.35 GB more memory.
2. Failing that, drop `spec_model_path`, `spec_type` and `spec_draft_n_max` to
   disable MTP. Costs roughly 40% of decode speed.

## Client-side follow-up

Three retries into a hung backend consumed all three slots and made recovery
harder. Cap retries at one and back off beyond the 300s stall window.

## Standing caveat

PR #28243 is an **open draft**. It received a correctness fix for this exact
configuration on the day of the incident. Treat Flash-Next stability results
taken before build 10802 as provisional.
