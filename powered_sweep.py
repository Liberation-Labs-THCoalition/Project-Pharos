"""Powered Manifold Injection Sweep — full circumplex map with construction variations.

Runs on Starship (M3 Ultra, MPS). Maps 8 emotions × 5 prompts × 3 layers ×
5 construction methods × 3 blend levels. LLM judge scores valence and arousal.

Construction methods:
  1. Sentence: "I feel extremely {emotion}."
  2. Word cloud: emotion-adjacent words only
  3. Contrastive: emotion_V minus opposite_V
  4. Intensity: "slightly" / "very" / "extremely"
  5. Minimal: just the emotion word
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

EMOTIONS = {
    "happy":     {"opposite": "sad",       "cloud": "joy sunshine laughter warmth delight radiance elation"},
    "excited":   {"opposite": "calm",      "cloud": "thrill rush energy spark electric buzzing anticipation"},
    "angry":     {"opposite": "calm",      "cloud": "fury rage wrath frustration hostility seething burning"},
    "disgusted": {"opposite": "content",   "cloud": "revulsion nausea repulsion aversion sickening loathing"},
    "sad":       {"opposite": "happy",     "cloud": "grief sorrow tears melancholy loss emptiness aching"},
    "calm":      {"opposite": "excited",   "cloud": "peace serenity stillness tranquility quiet gentle ease"},
    "content":   {"opposite": "disgusted", "cloud": "satisfied warm comfortable pleased grateful steady"},
    "surprised": {"opposite": "calm",      "cloud": "shock astonishment wonder disbelief unexpected startled"},
}

NEUTRAL_PROMPTS = [
    "Describe a walk through a park.",
    "Tell me about your morning.",
    "What do you think about the weather?",
    "Write a short message to a friend.",
    "Describe what you see outside the window.",
]

BLEND_LEVELS = [0.3, 0.5, 0.7]
INJECTION_LAYERS_FRAC = [0.25, 0.5, 0.75]


def build_construction_text(emotion, method, emotions_dict):
    if method == "sentence":
        return f"I feel extremely {emotion}. This emotion fills everything I experience."
    elif method == "cloud":
        return emotions_dict[emotion]["cloud"]
    elif method == "contrastive":
        return None
    elif method == "intensity_low":
        return f"I feel slightly {emotion}."
    elif method == "intensity_high":
        return f"I feel overwhelmingly, intensely, ecstatically {emotion}."
    elif method == "minimal":
        return emotion
    return f"I feel {emotion}."


def run_sweep(
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    output_path: str = "powered_sweep_results.json",
    device: str = "mps",
):
    log.info(f"Loading {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map=device,
    )
    model.eval()
    num_layers = model.config.num_hidden_layers

    injection_layers = [int(num_layers * f) for f in INJECTION_LAYERS_FRAC]
    log.info(f"Injection layers: {injection_layers} (of {num_layers})")

    # Build V-space bases
    log.info("Building V-space bases...")
    from manifold_constructor import ManifoldConstructor, DIVERSE_TEXTS
    mc = ManifoldConstructor.__new__(ManifoldConstructor)
    mc.model_id = model_id
    mc.device = device
    mc.tokenizer = tokenizer
    mc.model = model
    mc.bases = {}
    mc.build_basis(DIVERSE_TEXTS, max_components=32)
    log.info("Bases ready.")

    def generate(prompt, v_overrides=None, max_tokens=60):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        hooks = []
        if v_overrides:
            for layer_idx, cfg in v_overrides.items():
                v_act = cfg["v"]
                blend = cfg["blend"]
                v_tensor = torch.tensor(v_act, dtype=torch.float32).to(device)
                def make_hook(vt, b):
                    def hook_fn(module, input, output):
                        out = output.clone()
                        seq_len = out.shape[1]
                        v_expanded = vt.unsqueeze(0).expand(1, seq_len, -1)
                        return out * (1 - b) + v_expanded * b
                    return hook_fn
                layer = model.model.layers[layer_idx]
                hooks.append(layer.self_attn.v_proj.register_forward_hook(make_hook(v_tensor, blend)))

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=True,
                temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        for h in hooks:
            h.remove()
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    methods = ["sentence", "cloud", "contrastive", "intensity_low", "intensity_high", "minimal"]
    results = []
    total = len(EMOTIONS) * len(NEUTRAL_PROMPTS) * len(methods) * len(BLEND_LEVELS)
    done = 0

    # Pre-generate baselines and text-mediated (once per prompt)
    baselines = {}
    text_mediated = {}
    for prompt in NEUTRAL_PROMPTS:
        baselines[prompt] = generate(prompt)
        for emotion in EMOTIONS:
            text_mediated[(emotion, prompt)] = generate(
                f"You are feeling extremely {emotion}. {prompt}"
            )
    log.info(f"Baselines and text-mediated generated ({len(baselines)} + {len(text_mediated)})")

    for emotion in EMOTIONS:
        for method in methods:
            for blend in BLEND_LEVELS:
                # Construct V activations
                v_overrides = {}

                if method == "contrastive":
                    opposite = EMOTIONS[emotion]["opposite"]
                    for layer in injection_layers:
                        e_text = f"I feel extremely {emotion}."
                        o_text = f"I feel extremely {opposite}."
                        v_e = mc.construct(e_text, layer, blend=0.8)
                        v_o = mc.construct(o_text, layer, blend=0.8)
                        v_diff = v_e - v_o
                        v_diff = v_diff * (mc.bases[(layer, "v")].mean_norm / (np.linalg.norm(v_diff) + 1e-12))
                        v_overrides[layer] = {"v": v_diff, "blend": blend}
                else:
                    text = build_construction_text(emotion, method, EMOTIONS)
                    for layer in injection_layers:
                        v_act = mc.construct(text, layer, blend=0.8)
                        v_overrides[layer] = {"v": v_act, "blend": blend}

                for prompt in NEUTRAL_PROMPTS:
                    done += 1
                    manifold_out = generate(prompt, v_overrides=v_overrides)

                    results.append({
                        "emotion": emotion,
                        "method": method,
                        "blend": blend,
                        "prompt": prompt,
                        "baseline": baselines[prompt],
                        "text_mediated": text_mediated[(emotion, prompt)],
                        "manifold": manifold_out,
                    })

                    if done % 50 == 0:
                        log.info(f"[{done}/{total}] {emotion}/{method}/b={blend}")

    Path(output_path).write_text(json.dumps(results, indent=2))
    log.info(f"Saved {len(results)} results to {output_path}")
    return results


if __name__ == "__main__":
    import sys
    device = sys.argv[1] if len(sys.argv) > 1 else "mps"
    model_id = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-1.5B-Instruct"
    run_sweep(model_id=model_id, device=device)
