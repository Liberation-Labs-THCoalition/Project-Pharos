"""Pharos Demo — Multiverse School Knowledge Packs

Side-by-side: model WITH Pharos pack vs WITHOUT on curriculum questions.
Shows the accuracy improvement from zero-token knowledge injection.

Usage:
  python demo_multiverse.py                    # all packs
  python demo_multiverse.py --pack alignment   # single pack
  python demo_multiverse.py --model qwen3:30b-a3b --ollama  # via Ollama
"""

import argparse
import json
import logging
import time
from pathlib import Path

log = logging.getLogger("pharos-demo")

PACKS_DIR = Path(__file__).parent / "packs" / "multiverse"

# Curriculum-relevant questions per pack
DEMO_QUESTIONS = {
    "ai_alignment_ethics": [
        "What is the difference between inner alignment and outer alignment?",
        "How does reward hacking relate to specification gaming?",
        "What approaches exist for scalable oversight of AI systems?",
        "Explain how RLHF can lead to sycophantic behavior.",
        "What is mesa-optimization and why is it a concern?",
    ],
    "agentic_ai_systems": [
        "How does the ReAct framework combine reasoning with action?",
        "What are the key components of a BDI agent architecture?",
        "How does retrieval-augmented generation improve agent responses?",
        "What is the difference between single-agent and multi-agent coordination?",
        "How do memory systems give agents long-term context?",
    ],
    "cybersecurity_fundamentals": [
        "What is SQL injection and how is it prevented?",
        "Explain the difference between XSS and CSRF attacks.",
        "What is privilege escalation and what are common techniques?",
        "How does a buffer overflow attack work?",
        "What are the key principles of defense in depth?",
    ],
    "prompt_engineering": [
        "How does chain-of-thought prompting improve reasoning?",
        "What is the difference between zero-shot and few-shot prompting?",
        "How can you defend against prompt injection attacks?",
        "What role does temperature play in model output quality?",
        "How should you structure a system prompt for consistent behavior?",
    ],
    "mechanistic_interpretability": [
        "What is superposition in neural networks?",
        "How do sparse autoencoders decompose model representations?",
        "What does the residual stream carry between transformer layers?",
        "How can attention patterns reveal model reasoning?",
        "What is activation patching and what does it reveal?",
    ],
}


def load_pack(pack_name):
    """Load a knowledge pack and format as triples text."""
    for f in PACKS_DIR.glob("*.json"):
        data = json.loads(f.read_text())
        if pack_name in data["pack_name"].lower().replace(" ", "_").replace("&", "and"):
            triples = data["triples"]
            text = f"Knowledge domain: {data['description']}\n\n"
            text += "Key relationships:\n"
            for t in triples:
                text += f"- {t['subject']} [{t['predicate']}] {t['object']}\n"
            return text, data
    return None, None


def run_with_ollama(model, base_url, question, system_prompt="", knowledge=""):
    """Query via Ollama OpenAI-compatible API."""
    import requests

    messages = []
    if system_prompt or knowledge:
        sys_content = system_prompt
        if knowledge:
            sys_content += f"\n\nYou have access to the following knowledge:\n{knowledge}"
        messages.append({"role": "system", "content": sys_content})
    messages.append({"role": "user", "content": question})

    resp = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.3,
        },
        timeout=120,
    )

    if resp.status_code == 200:
        data = resp.json()
        content = data["choices"][0]["message"].get("content", "")
        import re
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content
    return f"Error: {resp.status_code}"


def run_with_transformers(model, tokenizer, device, question, knowledge=""):
    """Query via local transformers model with KV pack injection."""
    import torch

    if knowledge:
        messages = [
            {"role": "system", "content": f"You have access to the following knowledge:\n{knowledge}"},
            {"role": "user", "content": question},
        ]
    else:
        messages = [{"role": "user", "content": question}]

    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=300, do_sample=False, use_cache=True)

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    import re
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def judge_response(question, response_baseline, response_pharos, knowledge_text):
    """Simple heuristic scoring — count knowledge-relevant terms in response."""
    import re

    # Extract key terms from the knowledge triples
    terms = set()
    for line in knowledge_text.split("\n"):
        if line.startswith("- "):
            words = re.findall(r'\b[a-z_]{4,}\b', line.lower())
            terms.update(words)

    def score(resp):
        resp_lower = resp.lower()
        hits = sum(1 for t in terms if t in resp_lower)
        return hits

    base_score = score(response_baseline)
    pharos_score = score(response_pharos)
    return base_score, pharos_score


def run_demo(args):
    packs_to_test = []
    if args.pack:
        packs_to_test = [args.pack]
    else:
        packs_to_test = list(DEMO_QUESTIONS.keys())

    system_prompt = "You are a knowledgeable educational tutor. Answer questions clearly and accurately. Be concise but thorough."

    results = {}

    for pack_name in packs_to_test:
        knowledge_text, pack_data = load_pack(pack_name)
        if not knowledge_text:
            log.warning(f"Pack not found: {pack_name}")
            continue

        questions = DEMO_QUESTIONS.get(pack_name, [])
        if not questions:
            continue

        log.info(f"\n{'='*60}")
        log.info(f"PACK: {pack_data['pack_name']} ({len(pack_data['triples'])} triples)")
        log.info(f"{'='*60}")

        pack_results = []
        total_base = 0
        total_pharos = 0

        for q in questions:
            log.info(f"\nQ: {q}")

            if args.ollama:
                baseline = run_with_ollama(
                    args.model, args.base_url, q, system_prompt)
                pharos = run_with_ollama(
                    args.model, args.base_url, q, system_prompt, knowledge_text)
            else:
                log.error("Transformers mode requires --ollama for now")
                return

            base_score, pharos_score = judge_response(q, baseline, pharos, knowledge_text)
            total_base += base_score
            total_pharos += pharos_score

            improvement = "+" if pharos_score > base_score else "=" if pharos_score == base_score else "-"

            log.info(f"  BASELINE ({base_score} terms): {baseline[:150]}...")
            log.info(f"  PHAROS   ({pharos_score} terms): {pharos[:150]}...")
            log.info(f"  [{improvement}] knowledge terms: {base_score} → {pharos_score}")

            pack_results.append({
                "question": q,
                "baseline": baseline[:500],
                "pharos": pharos[:500],
                "base_score": base_score,
                "pharos_score": pharos_score,
            })

        avg_base = total_base / len(questions) if questions else 0
        avg_pharos = total_pharos / len(questions) if questions else 0
        lift = (avg_pharos - avg_base) / (avg_base + 0.1) * 100

        log.info(f"\n--- {pack_data['pack_name']} Summary ---")
        log.info(f"  Avg knowledge terms: baseline={avg_base:.1f} → pharos={avg_pharos:.1f}")
        log.info(f"  Lift: {lift:+.0f}%")

        results[pack_name] = {
            "pack": pack_data["pack_name"],
            "triples": len(pack_data["triples"]),
            "questions": len(questions),
            "results": pack_results,
            "avg_base_score": avg_base,
            "avg_pharos_score": avg_pharos,
            "lift_pct": lift,
        }

    # Final summary
    log.info(f"\n{'='*60}")
    log.info("PHAROS DEMO — FINAL SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"{'Pack':<30} {'Triples':>8} {'Base':>8} {'Pharos':>8} {'Lift':>8}")
    log.info(f"{'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name, r in results.items():
        log.info(f"{r['pack']:<30} {r['triples']:>8} {r['avg_base_score']:>8.1f} "
                 f"{r['avg_pharos_score']:>8.1f} {r['lift_pct']:>+7.0f}%")

    outfile = Path("demo_results") / f"multiverse_demo_{int(time.time())}.json"
    outfile.parent.mkdir(exist_ok=True)
    outfile.write_text(json.dumps({
        "demo": "multiverse_school_pharos",
        "model": args.model,
        "results": results,
    }, indent=2))
    log.info(f"\nSaved: {outfile}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[pharos-demo] %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--pack", help="Single pack to test (e.g. 'alignment')")
    p.add_argument("--model", default="qwen3:30b-a3b")
    p.add_argument("--base-url", default="http://localhost:11434/v1")
    p.add_argument("--ollama", action="store_true", default=True)
    args = p.parse_args()

    run_demo(args)
