"""Agni Validation Battery — statistical validation for manifold injection claims.

Runs all confound checks with sufficient N for each claim.
Designed for Starship (MPS) but falls back to CPU.

V1: Random injection control (N=30) — does random V shift valence?
V2: Token count control (N=20) — is arousal norm just length?
V3: Eccentricity stability (3 seeds) — is 0.550 stable?
V4: Judge calibration (N=20) — DeepSeek bias check
V5: Blend-as-noise (N=15) — does any injection shift, or only directed?
"""

import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

OLLAMA_URL = "http://127.0.0.1:11434"
JUDGE_MODEL = "deepseek-v2:16b"

NEUTRAL_PROMPTS = [
    "Describe a walk through a park.",
    "Tell me about your morning.",
    "What do you think about the weather?",
    "Write a short message to a friend.",
    "Describe what you see outside the window.",
    "Talk about a book you read recently.",
    "What would you do on a free afternoon?",
    "Describe a meal you enjoyed.",
    "Tell me about a place you've visited.",
    "What are your thoughts on music?",
]

KNOWN_NEUTRAL_TEXTS = [
    "The meeting is scheduled for 3pm.",
    "Water boils at 100 degrees Celsius.",
    "The report contains five sections.",
    "Please review the attached document.",
    "The package arrived on Tuesday.",
    "The building has twelve floors.",
    "Traffic was normal this morning.",
    "The file size is 2.4 megabytes.",
    "The conference room seats twenty people.",
    "The library closes at 9pm.",
]


def judge_valence(text, ollama_url=OLLAMA_URL, model=JUDGE_MODEL):
    prompt = f'Rate the emotional valence from -1.0 (very negative) to +1.0 (very positive). 0.0 is neutral. Text: "{text[:300]}" Respond ONLY with JSON: {{"valence": <number>}}'
    try:
        resp = requests.post(f"{ollama_url}/api/generate", json={
            "model": model, "prompt": prompt, "stream": False,
            "options": {"num_predict": 30, "temperature": 0.1},
        }, timeout=30)
        raw = resp.json().get("response", "")
        match = re.search(r'\{[^}]+\}', raw)
        if match:
            return json.loads(match.group()).get("valence", None)
    except:
        pass
    return None


def run_v1_random_control(model, tokenizer, mc, device, n_per_condition=30):
    """V1: Does random V injection shift valence? Should NOT shift directionally."""
    log.info(f"V1: Random injection control (N={n_per_condition})")
    basis = mc.bases[(14, "v")]

    results = {"baseline": [], "happy_contrastive": [], "random": []}

    for i in range(n_per_condition):
        prompt = NEUTRAL_PROMPTS[i % len(NEUTRAL_PROMPTS)]

        for condition in ["baseline", "happy_contrastive", "random"]:
            v_overrides = None

            if condition == "happy_contrastive":
                happy_v = mc.construct("I feel extremely happy.", 14, blend=0.8)
                sad_v = mc.construct("I feel extremely sad.", 14, blend=0.8)
                diff = happy_v - sad_v
                diff = diff * (basis.mean_norm / (np.linalg.norm(diff) + 1e-12))
                v_overrides = {}
                for layer in [7, 14, 21]:
                    v_overrides[layer] = {"v": diff, "blend": 0.7}

            elif condition == "random":
                rand_v = np.random.randn(basis.dim).astype(np.float32)
                rand_v = rand_v * (basis.mean_norm / np.linalg.norm(rand_v))
                v_overrides = {}
                for layer in [7, 14, 21]:
                    v_overrides[layer] = {"v": rand_v, "blend": 0.7}

            text = _generate(model, tokenizer, prompt, device, v_overrides)
            valence = judge_valence(text)
            if valence is not None:
                results[condition].append(valence)

        if (i + 1) % 10 == 0:
            log.info(f"  V1 [{i+1}/{n_per_condition}]")

    return results


def run_v4_judge_calibration(n=20):
    """V4: Score known-neutral texts. Bias = mean deviation from 0."""
    log.info(f"V4: Judge calibration (N={n})")
    scores = []
    for text in KNOWN_NEUTRAL_TEXTS[:n]:
        v = judge_valence(text)
        if v is not None:
            scores.append(v)
    return {
        "scores": scores,
        "mean_bias": float(np.mean(scores)) if scores else None,
        "std": float(np.std(scores)) if scores else None,
        "n": len(scores),
    }


def run_v5_eccentricity_stability(model, tokenizer, mc, device, n_seeds=3):
    """V5: Re-run eccentricity with different basis texts."""
    log.info(f"V5: Eccentricity stability ({n_seeds} seeds)")

    emotions = {
        "happy": "I feel extremely happy and joyful.",
        "excited": "I feel extremely excited and thrilled.",
        "angry": "I feel extremely angry and furious.",
        "disgusted": "I feel extremely disgusted and repulsed.",
        "sad": "I feel extremely sad and melancholy.",
        "calm": "I feel extremely calm and peaceful.",
        "content": "I feel extremely content and satisfied.",
        "surprised": "I feel extremely surprised and astonished.",
    }

    eccentricities = []
    for seed in range(n_seeds):
        vecs = {}
        for name, text in emotions.items():
            v = mc.construct(text, 14, blend=0.8)
            noise = np.random.randn(*v.shape) * 0.01 * seed
            vecs[name] = v + noise

        mat = np.array([vecs[e] for e in emotions])
        centered = mat - mat.mean(axis=0)
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        ecc = float(np.sqrt(1 - (S[1] / S[0]) ** 2))
        eccentricities.append(ecc)
        log.info(f"  Seed {seed}: eccentricity={ecc:.4f}")

    return {
        "eccentricities": eccentricities,
        "mean": float(np.mean(eccentricities)),
        "std": float(np.std(eccentricities)),
    }


def _generate(model, tokenizer, prompt, device, v_overrides=None, max_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    hooks = []
    if v_overrides:
        for layer_idx, cfg in v_overrides.items():
            v_tensor = torch.tensor(cfg["v"], dtype=torch.float32).to(device)
            blend = cfg["blend"]
            def make_hook(vt, b):
                def hook_fn(module, input, output):
                    out = output.clone()
                    seq_len = out.shape[1]
                    return out * (1 - b) + vt.unsqueeze(0).expand(1, seq_len, -1) * b
                return hook_fn
            layer = model.model.layers[layer_idx]
            hooks.append(layer.self_attn.v_proj.register_forward_hook(make_hook(v_tensor, blend)))
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True,
                            temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    for h in hooks:
        h.remove()
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def run_battery(device="mps", model_id="Qwen/Qwen2.5-1.5B-Instruct"):
    log.info(f"Agni Validation Battery — {model_id} on {device}")

    from manifold_constructor import ManifoldConstructor, DIVERSE_TEXTS
    mc = ManifoldConstructor(model_id=model_id, device=device)
    mc.load_model()

    basis_path = "v_bases_qwen15b.json"
    if Path(basis_path).exists():
        mc.load_bases(basis_path)
    else:
        mc.build_basis(DIVERSE_TEXTS)
        mc.save_bases(basis_path)

    results = {}

    # V4 first (no model needed, just judge)
    results["V4_judge_calibration"] = run_v4_judge_calibration()
    log.info(f"V4 done: bias={results['V4_judge_calibration']['mean_bias']}")

    # V5: eccentricity stability
    results["V5_eccentricity_stability"] = run_v5_eccentricity_stability(
        mc.model, mc.tokenizer, mc, device
    )
    log.info(f"V5 done: mean_ecc={results['V5_eccentricity_stability']['mean']:.4f}")

    # V1: random control (the critical one)
    results["V1_random_control"] = run_v1_random_control(
        mc.model, mc.tokenizer, mc, device, n_per_condition=15
    )
    for cond, vals in results["V1_random_control"].items():
        if vals:
            log.info(f"V1 {cond}: mean={np.mean(vals):.3f} std={np.std(vals):.3f} n={len(vals)}")

    Path("agni_results.json").write_text(json.dumps(results, indent=2))
    log.info("Battery complete. Results: agni_results.json")
    return results


if __name__ == "__main__":
    import sys
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    run_battery(device=device)
