import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtp_matrix as MM

PROMPTS = {
 "trivial": "What is 2+2? Answer with the number only.",
 "easy": "Name the capital of France.",
 "hard": "A train leaves A at 60 km/h. Two hours later a second train leaves A at 90 km/h on the same track. How far from A do they meet? Show the algebra.",
 "essay": MM.MED,
}
OUT = os.path.join(os.environ.get("BENCH_OUT_DIR", "."), "short_matrix.jsonl")

def rec(r):
    with open(OUT, "a") as f: f.write(json.dumps(r)+"\n")
    print(json.dumps(r), flush=True)

for name, cfg in MM.MODELS.items():
    p = MM.launch(cfg, 1)          # depth 1 = the measured optimum for both
    try:
        for k, prompt in PROMPTS.items():
            d, w = MM.ask({"messages":[{"role":"user","content":prompt}],
                           "max_tokens":8192, "temperature":0})
            t = d.get("timings",{}); m = d["choices"][0]["message"]
            rec({"model":name, "prompt":k,
                 "completion_tok": d["usage"]["completion_tokens"],
                 "answer_chars": len(m.get("content") or ""),
                 "reasoning_chars": len(m.get("reasoning_content") or ""),
                 "finish": d["choices"][0]["finish_reason"],
                 "decode_tok_s": round(t.get("predicted_per_second") or 0,2),
                 "wall_s": round(w,2)})
    finally:
        MM.stop(p)
print("SHORT DONE", flush=True)
