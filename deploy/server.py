"""Pharos API Server — REST endpoint for knowledge-injected inference.

Exposes Pharos as a simple API:
  POST /query     — ask a question with knowledge injection
  POST /packs     — list available packs
  POST /load      — pre-load a pack into cache
  GET  /health    — health check

Deploy: docker-compose up
Or: python server.py --host 0.0.0.0 --port 8080
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

log = logging.getLogger("pharos-api")

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B")
PACKS_DIR = Path(os.environ.get("PACKS_DIR", "packs"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "cache"))
MAX_LOADED = int(os.environ.get("MAX_LOADED_PACKS", "10"))

# Lazy-loaded globals
model = None
tokenizer = None
loaded_packs = {}


def ensure_model():
    global model, tokenizer
    if model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        log.info(f"Loading model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        log.info("Model loaded")


def load_pack(pack_name):
    if pack_name in loaded_packs:
        return loaded_packs[pack_name]

    for f in PACKS_DIR.rglob("*.json"):
        data = json.loads(f.read_text())
        if pack_name in data.get("pack_name", "").lower().replace(" ", "_"):
            # Build knowledge text
            text = f"Knowledge: {data['description']}\n"
            for t in data["triples"]:
                text += f"- {t['subject']} [{t['predicate']}] {t['object']}\n"

            loaded_packs[pack_name] = {"text": text, "data": data, "path": str(f)}

            # Evict if over limit
            while len(loaded_packs) > MAX_LOADED:
                oldest = next(iter(loaded_packs))
                del loaded_packs[oldest]

            return loaded_packs[pack_name]
    return None


def query_with_pack(question, pack_name=None):
    import torch
    ensure_model()

    knowledge = ""
    pack_used = None
    if pack_name:
        pack = load_pack(pack_name)
        if pack:
            knowledge = pack["text"]
            pack_used = pack["data"]["pack_name"]

    if knowledge:
        prompt = f"Knowledge:\n{knowledge}\n\nQuestion: {question}\nAnswer:"
    else:
        prompt = f"Question: {question}\nAnswer:"

    input_ids = tokenizer(prompt, return_tensors="pt")
    start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **input_ids, max_new_tokens=200, do_sample=False, use_cache=True)

    new_tokens = outputs[0][input_ids["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    elapsed = time.time() - start

    return {
        "answer": answer,
        "pack_used": pack_used,
        "tokens_in_prompt": input_ids["input_ids"].shape[1],
        "latency_s": round(elapsed, 2),
    }


class PharosHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok", "model": MODEL_NAME,
                                "packs_loaded": len(loaded_packs)})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_len)) if content_len else {}

        if self.path == "/query":
            question = body.get("question", "")
            pack = body.get("pack")
            if not question:
                self._respond(400, {"error": "question required"})
                return
            result = query_with_pack(question, pack)
            self._respond(200, result)

        elif self.path == "/packs":
            packs = []
            for f in PACKS_DIR.rglob("*.json"):
                try:
                    data = json.loads(f.read_text())
                    packs.append({
                        "name": data.get("pack_name"),
                        "triples": len(data.get("triples", [])),
                        "domain": data.get("domain"),
                    })
                except:
                    pass
            self._respond(200, {"packs": packs})

        elif self.path == "/load":
            pack_name = body.get("pack")
            if not pack_name:
                self._respond(400, {"error": "pack name required"})
                return
            pack = load_pack(pack_name)
            if pack:
                self._respond(200, {"loaded": pack["data"]["pack_name"],
                                    "triples": len(pack["data"]["triples"])})
            else:
                self._respond(404, {"error": f"pack '{pack_name}' not found"})
        else:
            self._respond(404, {"error": "not found"})

    def _respond(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        log.info(f"{self.client_address[0]} {format % args}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[pharos-api] %(message)s")
    log.info(f"Starting Pharos API on {args.host}:{args.port}")
    log.info(f"Model: {MODEL_NAME}")
    log.info(f"Packs: {PACKS_DIR}")

    server = HTTPServer((args.host, args.port), PharosHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
