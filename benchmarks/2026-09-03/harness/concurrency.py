"""Concurrency scaling, MTP under load, cold-start penalty, over-subscription.

Method notes that matter for reading the numbers:
  * Thinking is OFF and max_tokens is fixed, and the prompt is chosen to overrun
    that budget, so every request emits EXACTLY the same token count. tok/s is
    then directly comparable and the aggregate is meaningful.
  * Every prompt carries a unique filler so no two requests share a cached
    prefix. cache_prompt + kv_unified would otherwise let later requests ride on
    an earlier one's KV and inflate the parallel numbers.
  * A warmup request precedes the scaling runs, so cold-start is measured
    separately rather than polluting level 1.
"""
import json, statistics as st, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "unsloth/qwen3.8-flash-next"
MAX_TOKENS = 300
OUT = "/Users/Alex/Dev/slm_server/benchmarks/2026-09-03/stats/concurrency.jsonl"

FILLER = ("Background material follows. " * 30).strip()   # ~200 prompt tokens

def body(tag):
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content":
            f"[case {tag}] {FILLER}\nWrite a long, detailed explanation of how "
            f"gradient descent works. Do not stop early."}],
        "max_tokens": MAX_TOKENS, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }

def one(tag):
    req = urllib.request.Request(URL, data=json.dumps(body(tag)).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=900))
    except urllib.error.HTTPError as e:
        return {"tag": tag, "http": e.code, "err": e.read().decode()[:120], "wall": time.time()-t0}
    except Exception as e:
        return {"tag": tag, "err": f"{type(e).__name__}: {e}"[:140], "wall": time.time()-t0}
    w = time.time() - t0
    t = d.get("timings", {}); u = d["usage"]
    dn, da = t.get("draft_n"), t.get("draft_n_accepted")
    return {"tag": tag, "wall": round(w, 2), "tok": u["completion_tokens"],
            "prompt_tok": u["prompt_tokens"],
            "decode_tok_s": round(t.get("predicted_per_second") or 0, 2),
            "prompt_tok_s": round(t.get("prompt_per_second") or 0, 2),
            "draft_n": dn, "draft_acc": da,
            "accept_pct": round(da/dn*100, 1) if dn else None,
            "finish": d["choices"][0]["finish_reason"]}

def batch(n, label):
    t0 = time.time()
    with ThreadPoolExecutor(n) as ex:
        res = list(ex.map(one, [f"{label}-{i}" for i in range(n)]))
    wall = time.time() - t0
    ok = [r for r in res if "tok" in r]
    total = sum(r["tok"] for r in ok)
    rec = {"level": n, "label": label, "batch_wall_s": round(wall, 2),
           "aggregate_tok_s": round(total/wall, 2) if wall else None,
           "total_tokens": total, "n_ok": len(ok), "n_err": len(res)-len(ok),
           "per_request": res}
    with open(OUT, "a") as f: f.write(json.dumps(rec) + "\n")
    return rec

def show(rec):
    print(f"  level {rec['level']}: batch_wall={rec['batch_wall_s']}s  "
          f"AGGREGATE={rec['aggregate_tok_s']} tok/s  total_tok={rec['total_tokens']}"
          f"{'  ERRORS=' + str(rec['n_err']) if rec['n_err'] else ''}")
    for r in rec["per_request"]:
        if "tok" in r:
            print(f"      {r['tag']:10} {r['decode_tok_s']:>6} tok/s  wall={r['wall']:>6}s  "
                  f"tok={r['tok']}  accept={r['accept_pct']}%  prefill={r['prompt_tok_s']} tok/s")
        else:
            print(f"      {r['tag']:10} ERROR http={r.get('http')} {r.get('err','')[:90]}")

if __name__ == "__main__":
    print("=== 3. COLD START (server idle before this) ===")
    c1 = one("cold-1"); print(f"  first : {c1.get('decode_tok_s')} tok/s wall={c1.get('wall')}s prefill={c1.get('prompt_tok_s')}")
    c2 = one("warm-1"); print(f"  second: {c2.get('decode_tok_s')} tok/s wall={c2.get('wall')}s prefill={c2.get('prompt_tok_s')}")
    c3 = one("warm-2"); print(f"  third : {c3.get('decode_tok_s')} tok/s wall={c3.get('wall')}s prefill={c3.get('prompt_tok_s')}")
    with open(OUT, "a") as f:
        f.write(json.dumps({"test": "cold_start", "runs": [c1, c2, c3]}) + "\n")

    print("=== 1 & 2. CONCURRENCY SCALING + MTP UNDER LOAD (3 reps each) ===")
    for rep in range(3):
        for n in (1, 2, 3):
            show(batch(n, f"r{rep}n{n}"))

    print("=== 4. OVER-SUBSCRIPTION: 4 concurrent against 3 slots ===")
    show(batch(4, "oversub"))
    print("DONE")
