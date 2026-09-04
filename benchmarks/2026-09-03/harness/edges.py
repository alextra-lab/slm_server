import json, time, urllib.request, sys

URL = "http://127.0.0.1:8042/v1/chat/completions"
MODEL = "mtplx-qwen38-27b-optimized-quality"
OUT = "/private/tmp/claude-501/-Users-Alex-Dev-slm-server/9b8aaa05-c340-4d30-b0e2-8d8bbaa07db4/scratchpad/edges38.jsonl"

def call(tag, prompt, max_tokens, **extra):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0}
    body.update(extra)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=600))
    except Exception as e:
        rec = {"tag": tag, "max_tokens": max_tokens, "error": str(e)[:200]}
        log(rec); return rec
    wall = time.time() - t0
    u = d["usage"]; s = d.get("mtplx_stats", {})
    content = d["choices"][0]["message"].get("content") or ""
    det = u.get("completion_tokens_details", {})
    rec = {"tag": tag, "max_tokens": max_tokens, "extra": extra,
           "finish": d["choices"][0]["finish_reason"],
           "prompt_tok": u["prompt_tokens"],
           "completion_tok": u["completion_tokens"],
           "reasoning_tok": det.get("reasoning_tokens", 0),
           "answer_tok": s.get("answer_tokens"),
           "answer_chars": len(content),
           "answer_head": content[:80].replace("\n", " "),
           "wall_s": round(wall, 2),
           "decode_tok_s": round(s.get("decode_tok_s") or 0, 1)}
    log(rec); return rec

def log(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)

MED = "Explain how a Kalman filter works, in about 600 words."

MODE = sys.argv[1] if len(sys.argv) > 1 else "none"
if MODE == "budget":
    for mt in [64, 128, 256, 512, 1024, 2048, 4096]:
        call("budget", MED, mt)
elif MODE == "difficulty":
    prompts = {
      "trivial": "What is 2+2? Answer with the number only.",
      "easy": "Name the capital of France.",
      "medium": MED,
      "hard": "A train leaves A at 60 km/h. Two hours later a second train leaves A at 90 km/h on the same track. How far from A do they meet? Show the algebra.",
    }
    for k, p in prompts.items():
        call("difficulty:" + k, p, 2048)
elif MODE == "effort":
    for e in ["low", "medium", "high", "xhigh"]:
        call("effort:" + e, MED, 4096, reasoning_effort=e)
    call("effort:off", MED, 4096, enable_thinking=False)
