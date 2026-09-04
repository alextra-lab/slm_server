import json, os, signal, subprocess, time, urllib.request

S = "/private/tmp/claude-501/-Users-Alex-Dev-slm-server/9b8aaa05-c340-4d30-b0e2-8d8bbaa07db4/scratchpad"
BIN = f"{S}/llama.cpp/build/bin/llama-server"
M = "/Volumes/EnvoyUltra/lm-studio/models"
PORT = 8597
OUT = f"{S}/mtp_matrix.jsonl"
REPO = "/Users/Alex/Dev/slm_server"

MODELS = {
  "qwen36": dict(
    model=f"{M}/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf",
    mmproj=f"{M}/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/mmproj-F32.gguf",
    md=None,
    tmpl=f"{REPO}/config/templates/qwen3.6-unsloth.jinja",
    kw={"enable_thinking": True, "preserve_thinking": False},
    ctx=131072, temp=0.6, top_p=0.85, top_k=10),
  "flash": dict(
    model=f"{M}/unsloth/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf",
    mmproj=f"{M}/unsloth/qwen3.8-flash-next/mmproj-F16.gguf",
    md=f"{M}/unsloth/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf",
    tmpl=f"{REPO}/config/templates/qwen3.8-flash-next.jinja",
    kw={"enable_thinking": True, "preserve_thinking": False, "reasoning_effort": "medium"},
    ctx=131072, temp=1.0, top_p=0.95, top_k=20),
}

MED = "Explain how a Kalman filter works, in about 600 words."

def launch(cfg, depth):
    cmd = [BIN, "-m", cfg["model"], "--mmproj", cfg["mmproj"],
           "--port", str(PORT), "--host", "127.0.0.1", "--parallel", "1",
           "-c", str(cfg["ctx"]), "--jinja", "--chat-template-file", cfg["tmpl"],
           "--chat-template-kwargs", json.dumps(cfg["kw"]),
           "--temp", str(cfg["temp"]), "--top-p", str(cfg["top_p"]),
           "--top-k", str(cfg["top_k"]), "--min-p", "0.0",
           "--n-predict", "49152", "--kv-unified",
           "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
           "--flash-attn", "on", "--fit", "on", "--cont-batching"]
    if depth:
        if cfg["md"]:
            cmd += ["-md", cfg["md"]]
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(depth)]
    env = dict(os.environ, LLAMA_ARG_REASONING_PRESERVE="0")
    log = open(f"{S}/matrix_server.log", "a")
    p = subprocess.Popen(cmd, stdout=log, stderr=log, env=env, preexec_fn=os.setsid)
    for _ in range(90):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5).status == 200:
                return p
        except Exception:
            pass
        if p.poll() is not None:
            raise RuntimeError("server exited during load")
        time.sleep(5)
    raise RuntimeError("server never became healthy")

def stop(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        p.wait(timeout=60)
    except Exception:
        pass
    time.sleep(5)

def ask(payload, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    return d, time.time() - t0

def record(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)

for name, cfg in MODELS.items():
    for depth in [0, 1, 2, 3]:
        try:
            p = launch(cfg, depth)
        except Exception as e:
            record({"model": name, "depth": depth, "error": str(e)[:200]}); continue
        try:
            # A: raw decode speed, thinking off, fixed token count
            kw = dict(cfg["kw"]); kw["enable_thinking"] = False
            d, w = ask({"messages": [{"role": "user", "content": MED}],
                        "max_tokens": 400, "temperature": 0,
                        "chat_template_kwargs": kw})
            t = d.get("timings", {})
            record({"model": name, "depth": depth, "test": "raw_speed",
                    "completion_tok": d["usage"]["completion_tokens"],
                    "decode_tok_s": round(t.get("predicted_per_second") or 0, 2),
                    "prefill_tok_s": round(t.get("prompt_per_second") or 0, 2),
                    "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"),
                    "wall_s": round(w, 2)})
            # B: full answer with reasoning on
            d, w = ask({"messages": [{"role": "user", "content": MED}],
                        "max_tokens": 8192, "temperature": 0})
            t = d.get("timings", {}); m = d["choices"][0]["message"]
            record({"model": name, "depth": depth, "test": "reasoning_on",
                    "completion_tok": d["usage"]["completion_tokens"],
                    "answer_chars": len(m.get("content") or ""),
                    "reasoning_chars": len(m.get("reasoning_content") or ""),
                    "finish": d["choices"][0]["finish_reason"],
                    "decode_tok_s": round(t.get("predicted_per_second") or 0, 2),
                    "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"),
                    "wall_s": round(w, 2)})
        except Exception as e:
            record({"model": name, "depth": depth, "error": str(e)[:200]})
        finally:
            stop(p)
print("MATRIX DONE", flush=True)
