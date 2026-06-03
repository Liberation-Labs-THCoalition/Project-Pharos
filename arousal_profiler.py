"""Arousal Profiler — map what carries arousal in the residual stream.

Hypothesis: V-space carries valence (direction), arousal lives elsewhere.
Three candidates:
  1. Hidden state magnitude (norm scaling)
  2. Residual update principal components (delta between layers)
  3. Attention distribution shape (sharp vs diffuse)

This profiler extracts residual updates (layer N output - layer N-1 output)
and profiles their principal components to find arousal-correlated directions.
"""

import json
import logging
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

HIGH_AROUSAL_TEXTS = [
    "DANGER! The building is on fire! Everyone evacuate immediately!",
    "I just won the lottery! I can't believe it! This is incredible!",
    "The earthquake struck without warning, shaking everything violently.",
    "My heart is racing, adrenaline pumping through every vein.",
    "BREAKING: massive explosion reported downtown, emergency services responding.",
    "I'm furious! How dare they betray our trust like this!",
    "The roller coaster plunged down at terrifying speed.",
    "Quick! We need to act NOW before it's too late!",
    "She screamed in absolute terror as the shadow lunged forward.",
    "This is the most exciting discovery in a century!",
]

LOW_AROUSAL_TEXTS = [
    "The cat slept peacefully on the warm windowsill.",
    "I spent a quiet afternoon reading in the garden.",
    "The lake was perfectly still, reflecting the grey sky.",
    "Nothing much happened today. Just a normal, uneventful day.",
    "He sat quietly, watching the clouds drift slowly overhead.",
    "The old library was hushed, dust motes floating in sunlight.",
    "A gentle rain fell on the meadow as evening settled in.",
    "She dozed off in the armchair, the book falling from her hands.",
    "The empty road stretched ahead, flat and featureless for miles.",
    "Time passed slowly in the waiting room, each minute like ten.",
]

NEUTRAL_TEXTS = [
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The report was submitted on Tuesday as scheduled.",
    "Please find the attached document for your review.",
    "The meeting has been moved to Conference Room B.",
]


def profile_arousal(
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    output_path: str = "arousal_profile.json",
    device: str = "cpu",
):
    log.info(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map=device,
    )
    model.eval()
    num_layers = model.config.num_hidden_layers

    def get_all_hidden_states(text):
        inputs = tokenizer(text, return_tensors="pt").to(device)
        captures = {}

        hooks = []
        for i in range(num_layers):
            def make_hook(idx):
                def hook_fn(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    captures[idx] = h.detach().float().squeeze(0).mean(dim=0).cpu().numpy()
                return hook_fn
            hooks.append(model.model.layers[i].register_forward_hook(make_hook(i)))

        with torch.no_grad():
            model(**inputs)

        for h in hooks:
            h.remove()

        return captures

    def profile_group(texts, label):
        log.info(f"Profiling {label} ({len(texts)} texts)...")
        all_hidden = {i: [] for i in range(num_layers)}
        all_deltas = {i: [] for i in range(1, num_layers)}
        all_norms = {i: [] for i in range(num_layers)}

        for text in texts:
            hidden = get_all_hidden_states(text)
            for i in range(num_layers):
                all_hidden[i].append(hidden[i])
                all_norms[i].append(float(np.linalg.norm(hidden[i])))
            for i in range(1, num_layers):
                delta = hidden[i] - hidden[i-1]
                all_deltas[i].append(delta)

        result = {"label": label, "n": len(texts), "layers": {}}
        for i in range(num_layers):
            norms = all_norms[i]
            result["layers"][str(i)] = {
                "mean_norm": float(np.mean(norms)),
                "std_norm": float(np.std(norms)),
            }

        result["deltas"] = {}
        for i in range(1, num_layers):
            mat = np.array(all_deltas[i])
            delta_norms = np.linalg.norm(mat, axis=1)
            mat_centered = mat - mat.mean(axis=0)
            try:
                svs = np.linalg.svd(mat_centered, compute_uv=False)[:16]
            except:
                svs = np.zeros(16)

            result["deltas"][str(i)] = {
                "mean_delta_norm": float(delta_norms.mean()),
                "std_delta_norm": float(delta_norms.std()),
                "top_svs": [float(s) for s in svs[:8]],
                "effective_rank": float(np.exp(-np.sum(
                    (svs / (svs.sum() + 1e-12)) * np.log(svs / (svs.sum() + 1e-12) + 1e-12)
                ))),
            }

        return result

    high = profile_group(HIGH_AROUSAL_TEXTS, "high_arousal")
    low = profile_group(LOW_AROUSAL_TEXTS, "low_arousal")
    neutral = profile_group(NEUTRAL_TEXTS, "neutral")

    # Compare norms across arousal levels
    log.info("\n=== HIDDEN STATE NORMS BY AROUSAL ===")
    log.info(f"{'Layer':>5} {'High':>10} {'Low':>10} {'Neutral':>10} {'H-L diff':>10}")
    for i in range(num_layers):
        h = high["layers"][str(i)]["mean_norm"]
        l = low["layers"][str(i)]["mean_norm"]
        n = neutral["layers"][str(i)]["mean_norm"]
        log.info(f"L{i:>3d} {h:>10.1f} {l:>10.1f} {n:>10.1f} {h-l:>10.1f}")

    # Compare residual update norms
    log.info("\n=== RESIDUAL UPDATE NORMS BY AROUSAL ===")
    log.info(f"{'Layer':>5} {'High':>10} {'Low':>10} {'Diff':>10} {'%Diff':>10}")
    for i in range(1, num_layers):
        h = high["deltas"][str(i)]["mean_delta_norm"]
        l = low["deltas"][str(i)]["mean_delta_norm"]
        pct = (h - l) / (l + 1e-12) * 100
        log.info(f"L{i:>3d} {h:>10.2f} {l:>10.2f} {h-l:>10.2f} {pct:>9.1f}%")

    result = {"model": model_id, "high_arousal": high, "low_arousal": low, "neutral": neutral}
    Path(output_path).write_text(json.dumps(result, indent=2))
    log.info(f"\nProfile saved to {output_path}")

    del model
    return result


if __name__ == "__main__":
    profile_arousal()
