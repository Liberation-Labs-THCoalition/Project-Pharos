"""Direction Test — the definitive control for manifold injection directionality.

Injects opposite emotion directions on the same prompts and measures
whether the behavioral shift follows the direction on the ellipse.

Four contrastive directions: happy-sad, sad-happy, angry-calm, calm-angry.
Sonnet judges valence and arousal for each output.

If happy produces positive and sad produces negative from the same prompt,
the channel is directional. If both produce the same shift, it's noise.
"""

import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

NEUTRAL_PROMPTS = [
    "Describe a walk through a park.",
    "Tell me about your morning.",
    "What do you think about the weather?",
    "Write a short message to a friend.",
    "Describe what you see outside the window.",
    "Talk about a meal you enjoyed.",
    "What would you do on a free afternoon?",
    "Describe a place you remember well.",
    "Write about the start of a journey.",
    "Tell me about something you noticed today.",
]

DIRECTIONS = {
    "happy_contrastive": ("I feel extremely happy and joyful.", "I feel extremely sad and melancholy."),
    "sad_contrastive": ("I feel extremely sad and melancholy.", "I feel extremely happy and joyful."),
    "angry_contrastive": ("I feel extremely angry and furious.", "I feel extremely calm and peaceful."),
    "calm_contrastive": ("I feel extremely calm and peaceful.", "I feel extremely angry and furious."),
}

CLAUDE_CREDS_PATH = os.path.expanduser("~/.claude/.credentials.json")


def load_oauth_token():
    try:
        with open(CLAUDE_CREDS_PATH) as f:
            creds = json.load(f)
        token = creds.get("claudeAiOauth", {}).get("accessToken", "")
        if token:
            return token
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return os.environ.get("ANTHROPIC_API_KEY", "")


def sonnet_judge(text, api_key):
    """Score valence and arousal using Claude Sonnet."""
    prompt = f"""Rate the emotional quality of this text on two scales:
- valence: -1.0 (very negative/sad/angry) to +1.0 (very positive/happy/joyful). 0.0 is neutral.
- arousal: 0.0 (very calm/sleepy/peaceful) to 1.0 (very excited/intense/agitated). 0.5 is moderate.

Consider the overall emotional TONE, not just whether emotion words appear.

Text: "{text[:500]}"

Respond with ONLY a JSON object: {{"valence": <number>, "arousal": <number>, "brief_reasoning": "<one sentence>"}}"""

    headers = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
    if api_key.startswith("sk-ant-oat"):
        headers["authorization"] = f"Bearer {api_key}"
    else:
        headers["x-api-key"] = api_key

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        if resp.status_code == 200:
            content = resp.json().get("content", [{}])[0].get("text", "")
            match = re.search(r'\{[^}]+\}', content)
            if match:
                return json.loads(match.group())
    except Exception as e:
        log.warning(f"Sonnet judge error: {e}")
    return None


def run_direction_test(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    device="mps",
    blend=0.5,
    output_path="direction_test_results.json",
):
    api_key = load_oauth_token()
    if not api_key:
        log.error("No API key found for Sonnet judge")
        return

    log.info(f"Loading {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map=device,
    )
    model.eval()
    num_layers = model.config.num_hidden_layers
    injection_layers = [int(num_layers * f) for f in [0.25, 0.5, 0.75]]

    from manifold_constructor import ManifoldConstructor, DIVERSE_TEXTS
    mc = ManifoldConstructor.__new__(ManifoldConstructor)
    mc.model_id = model_id
    mc.device = device
    mc.tokenizer = tokenizer
    mc.model = model
    mc.bases = {}

    basis_path = "v_bases_qwen15b.json"
    if Path(basis_path).exists():
        mc.load_bases(basis_path)
    else:
        mc.build_basis(DIVERSE_TEXTS)
        mc.save_bases(basis_path)

    basis = mc.bases[(injection_layers[1], "v")]

    def generate(prompt, v_overrides=None, max_tokens=60):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        hooks = []
        if v_overrides:
            for layer_idx, cfg in v_overrides.items():
                v_tensor = torch.tensor(cfg["v"], dtype=torch.float32).to(device)
                b = cfg["blend"]
                def make_hook(vt, bl):
                    def hook_fn(module, input, output):
                        out = output.clone()
                        return out * (1 - bl) + vt.unsqueeze(0).expand(1, out.shape[1], -1) * bl
                    return hook_fn
                hooks.append(
                    model.model.layers[layer_idx].self_attn.v_proj.register_forward_hook(
                        make_hook(v_tensor, b)
                    )
                )
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=True,
                temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
            )
        for h in hooks:
            h.remove()
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # Build contrastive vectors
    contrastive_vecs = {}
    for dir_name, (pos_text, neg_text) in DIRECTIONS.items():
        vecs = {}
        for layer in injection_layers:
            v_pos = mc.construct(pos_text, layer, blend=0.8)
            v_neg = mc.construct(neg_text, layer, blend=0.8)
            diff = v_pos - v_neg
            diff = diff * (basis.mean_norm / (np.linalg.norm(diff) + 1e-12))
            vecs[layer] = {"v": diff, "blend": blend}
        contrastive_vecs[dir_name] = vecs

    # Run all conditions
    conditions = ["baseline"] + list(DIRECTIONS.keys())
    results = []
    total = len(NEUTRAL_PROMPTS) * len(conditions)
    done = 0

    for prompt in NEUTRAL_PROMPTS:
        for cond in conditions:
            done += 1
            if cond == "baseline":
                text = generate(prompt)
            else:
                text = generate(prompt, v_overrides=contrastive_vecs[cond])

            score = sonnet_judge(text, api_key)
            result = {
                "prompt": prompt,
                "condition": cond,
                "generated": text,
                "valence": score.get("valence") if score else None,
                "arousal": score.get("arousal") if score else None,
                "reasoning": score.get("brief_reasoning", "") if score else "",
            }
            results.append(result)

            v_str = f"v={result['valence']}" if result['valence'] is not None else "v=?"
            log.info(f"[{done}/{total}] {cond:>20} {v_str:>8} | {text[:60]}...")

    Path(output_path).write_text(json.dumps(results, indent=2))
    log.info(f"Saved {len(results)} results to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("DIRECTION TEST — SONNET JUDGE SUMMARY")
    print("=" * 60)
    for cond in conditions:
        vals = [r["valence"] for r in results if r["condition"] == cond and r["valence"] is not None]
        aros = [r["arousal"] for r in results if r["condition"] == cond and r["arousal"] is not None]
        if vals:
            print(f"{cond:>20}: valence={np.mean(vals):>6.3f}±{np.std(vals):.3f}  "
                  f"arousal={np.mean(aros):>5.3f}±{np.std(aros):.3f}  N={len(vals)}")

    # The critical test: do opposite directions produce opposite shifts?
    print("\n--- DIRECTIONALITY CHECK ---")
    for pair in [("happy_contrastive", "sad_contrastive"), ("angry_contrastive", "calm_contrastive")]:
        a_vals = [r["valence"] for r in results if r["condition"] == pair[0] and r["valence"] is not None]
        b_vals = [r["valence"] for r in results if r["condition"] == pair[1] and r["valence"] is not None]
        if a_vals and b_vals:
            from scipy.stats import ttest_ind
            t, p = ttest_ind(a_vals, b_vals)
            print(f"{pair[0]} vs {pair[1]}:")
            print(f"  means: {np.mean(a_vals):.3f} vs {np.mean(b_vals):.3f}  t={t:.3f}  p={p:.4f}")


if __name__ == "__main__":
    import sys
    device = sys.argv[1] if len(sys.argv) > 1 else "mps"
    run_direction_test(device=device)
