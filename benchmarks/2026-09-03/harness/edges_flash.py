import json, time, urllib.request, sys

ROUTER = "http://127.0.0.1:8502/v1/chat/completions"
SCRATCH = "/private/tmp/claude-501/-Users-Alex-Dev-slm-server/9b8aaa05-c340-4d30-b0e2-8d8bbaa07db4/scratchpad"

TARGETS = {
    "flashnext": ("unsloth/qwen3.8-flash-next", 8502),
    
}

def post(url, payload, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def ntok(port, text):
    if not text:
        return 0
    try:
        r = post(f"http://127.0.0.1:{port}/tokenize", {"content": text}, timeout=60)
        return len(r.get("tokens", []))
    except Exception:
        return -1

def call(target, tag, prompt, max_tokens, **extra):
    model, port = TARGETS[target]
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0}
    body.update(extra)
    t0 = time.time()
    try:
        d = post(ROUTER, body)
    except Exception as e:
        rec = {"target": target, "tag": tag, "max_tokens": max_tokens, "error": str(e)[:200]}
        log(rec); return rec
    wall = time.time() - t0
    ch = d["choices"][0]; m = ch["message"]; u = d["usage"]; t = d.get("timings", {})
    content = m.get("content") or ""
    reasoning = m.get("reasoning_content") or ""
    rec = {"target": target, "tag": tag, "max_tokens": max_tokens, "extra": extra,
           "finish": ch["finish_reason"],
           "prompt_tok": u["prompt_tokens"],
           "completion_tok": u["completion_tokens"],
           "reasoning_tok": ntok(port, reasoning),
           "answer_tok": ntok(port, content),
           "answer_chars": len(content),
           "answer_head": content[:80].replace("\n", " "),
           "tool_calls": bool(m.get("tool_calls")),
           "wall_s": round(wall, 2),
           "prefill_tok_s": round(t.get("prompt_per_second") or 0, 1),
           "decode_tok_s": round(t.get("predicted_per_second") or 0, 1),
           "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted")}
    log(rec); return rec

def log(rec):
    with open(f"{SCRATCH}/edges_flash.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)

MED = "Explain how a Kalman filter works, in about 600 words."
PROMPTS = {
    "trivial": "What is 2+2? Answer with the number only.",
    "easy": "Name the capital of France.",
    "medium": MED,
    "hard": "A train leaves A at 60 km/h. Two hours later a second train leaves A at 90 km/h on the same track. How far from A do they meet? Show the algebra.",
}

if __name__ == "__main__":
    target, mode = sys.argv[1], sys.argv[2]
    if mode == "budget":
        for mt in [64, 128, 256, 512, 1024, 2048, 4096, 6144]:
            call(target, "budget", MED, mt)
    elif mode == "difficulty":
        for k, p in PROMPTS.items():
            call(target, "difficulty:" + k, p, 2048)
    elif mode == "effort":
        for e in ["low", "medium", "high", "xhigh"]:
            call(target, "effort:" + e, MED, 4096, reasoning_effort=e)
        call(target, "effort:kwargs_off", MED, 4096,
             chat_template_kwargs={"enable_thinking": False})
        call(target, "effort:default", MED, 4096)
