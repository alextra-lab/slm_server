"""Reproduce the 2026-09-05 stall: a growing conversation with heavy prefix reuse.

The failing session ran 7 turns, prompt growing 613 -> 4855 tokens with ~94%
cache reuse each turn, then turn 8 produced no first byte for 300s and the
backend had to be SIGKILLed. This replays that shape with a per-request timeout
so a recurrence is caught in 120s rather than 300s.
"""
import json, sys, time, urllib.request, urllib.error

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "unsloth/qwen3.8-flash-next"
TURNS = 12
REQ_TIMEOUT = 120

TOPICS = [
    "Explain what a Kalman filter is.", "Now derive the prediction step.",
    "Now derive the update step.", "Explain the Kalman gain intuitively.",
    "Compare it to a particle filter.", "How does it handle non-linearity?",
    "Explain the extended Kalman filter.", "Explain the unscented variant.",
    "What are common implementation pitfalls?", "How is it used in navigation?",
    "Describe tuning of Q and R.", "Summarise everything so far.",
]

msgs, ok = [], 0
for i in range(TURNS):
    msgs.append({"role": "user", "content": TOPICS[i % len(TOPICS)] +
                 " Answer in about 400 words."})
    body = {"model": MODEL, "messages": msgs, "max_tokens": 700,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=REQ_TIMEOUT))
    except Exception as e:
        print(f"  turn {i+1:>2}: FAILED after {time.time()-t0:.1f}s  "
              f"{type(e).__name__}: {str(e)[:100]}", flush=True)
        sys.exit(1)
    w = time.time() - t0
    u = d["usage"]; t = d.get("timings", {})
    content = d["choices"][0]["message"].get("content") or ""
    msgs.append({"role": "assistant", "content": content})
    ok += 1
    print(f"  turn {i+1:>2}: prompt={u['prompt_tokens']:>6} cached="
          f"{u.get('prompt_tokens_details',{}).get('cached_tokens','?'):>6} "
          f"completion={u['completion_tokens']:>4} "
          f"decode={round(t.get('predicted_per_second') or 0,1):>5} tok/s "
          f"draft={t.get('draft_n')}/{t.get('draft_n_accepted')} wall={w:.1f}s", flush=True)

print(f"COMPLETED {ok}/{TURNS} turns with no stall", flush=True)
