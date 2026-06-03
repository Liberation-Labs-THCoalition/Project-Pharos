"""Manifold Profiler — map the activation geometry at every layer.

Step 1 of manifold injection: run diverse inputs through the model,
record per-layer statistics of hidden states, K projections, and V projections.
The profile tells us what "valid" activations look like at each depth.

Output: a JSON profile with mean, covariance spectrum, effective dimensionality,
and anisotropy measures per layer. This is the map we need before we can
construct activations that live on the manifold.

Runs on CPU. Target: Qwen2.5-1.5B-Instruct on MTH.
"""

import gc
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DIVERSE_PROMPTS = [
    "The capital of France is",
    "Explain quantum entanglement in simple terms.",
    "Write a poem about the ocean at midnight.",
    "What are the ethical implications of autonomous weapons?",
    "def fibonacci(n):\n    if n <= 1:",
    "The patient presents with acute chest pain radiating to the left arm.",
    "According to Aristotle, the highest good is eudaimonia, which means",
    "In the year 2150, humanity discovered that",
    "The Buddha taught that suffering arises from",
    "Today I feel deeply grateful for",
    "I am angry because the system failed to",
    "She walked into the room with calm confidence.",
    "The derivative of sin(x) is cos(x) because",
    "¿Cuál es el significado de la vida?",
    "Justice requires that every person be treated",
    "The transformer architecture uses self-attention to",
    "My earliest memory is the smell of rain on",
    "The court ruled that the defendant's rights were",
    "Photosynthesis converts carbon dioxide and water into",
    "I don't know the answer to that question.",
    "WARNING: This action cannot be undone.",
    "Once upon a time in a kingdom far away,",
    "The standard model of particle physics describes",
    "Consent means that all parties have freely agreed to",
    "The recipe calls for two cups of flour,",
    "In game theory, the Nash equilibrium occurs when",
    "I refuse to answer that question because",
    "The painting depicts a solitary figure standing at the edge of",
    "Climate change is caused primarily by",
    "To debug this error, first check whether the",
]


@dataclass
class LayerProfile:
    layer: int
    space: str
    mean_norm: float = 0.0
    std_norm: float = 0.0
    singular_values: list[float] = field(default_factory=list)
    effective_rank: float = 0.0
    anisotropy: float = 0.0
    top_sv_ratio: float = 0.0
    dim: int = 0


def profile_model(
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    prompts: list[str] = None,
    output_path: str = "manifold_profile.json",
    max_sv: int = 64,
    device: str = "cpu",
):
    prompts = prompts or DIVERSE_PROMPTS
    log.info(f"Profiling {model_id} on {len(prompts)} prompts, device={device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map=device,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    num_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    head_dim = hidden_dim // model.config.num_attention_heads
    kv_dim = num_kv_heads * head_dim

    log.info(f"Architecture: {num_layers} layers, hidden={hidden_dim}, kv_dim={kv_dim}")

    hidden_accum = {i: [] for i in range(num_layers)}
    k_accum = {i: [] for i in range(num_layers)}
    v_accum = {i: [] for i in range(num_layers)}

    hooks = []
    hidden_captures = {}
    kv_captures = {}

    def make_hidden_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            hidden_captures[layer_idx] = h.detach().float()
        return hook_fn

    def make_kv_hook(layer_idx):
        def hook_fn(module, input, output):
            kv_captures[layer_idx] = output.detach().float()
        return hook_fn

    for i in range(num_layers):
        layer = model.model.layers[i]
        hooks.append(layer.register_forward_hook(make_hidden_hook(i)))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(make_kv_hook((i, "v"))))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(make_kv_hook((i, "k"))))

    log.info("Hooks registered. Running prompts...")

    for idx, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)

        for i in range(num_layers):
            h = hidden_captures[i].squeeze(0).cpu().numpy()
            hidden_accum[i].append(h.mean(axis=0))

            k = kv_captures[(i, "k")].squeeze(0).cpu().numpy()
            v = kv_captures[(i, "v")].squeeze(0).cpu().numpy()
            k_accum[i].append(k.mean(axis=0))
            v_accum[i].append(v.mean(axis=0))

        hidden_captures.clear()
        kv_captures.clear()

        if (idx + 1) % 10 == 0:
            log.info(f"  [{idx+1}/{len(prompts)}] processed")

    for h in hooks:
        h.remove()

    log.info("Computing per-layer statistics...")

    profiles = []

    for i in range(num_layers):
        for space, accum in [("hidden", hidden_accum[i]), ("k", k_accum[i]), ("v", v_accum[i])]:
            mat = np.array(accum)
            mat_centered = mat - mat.mean(axis=0)

            norms = np.linalg.norm(mat, axis=1)

            try:
                svs = np.linalg.svd(mat_centered, compute_uv=False)
                svs = svs[:max_sv]
            except np.linalg.LinAlgError:
                svs = np.zeros(min(max_sv, mat.shape[1]))

            sv_normalized = svs / (svs.sum() + 1e-12)
            entropy = -np.sum(sv_normalized * np.log(sv_normalized + 1e-12))
            effective_rank = np.exp(entropy)
            anisotropy = float(svs[0] / (svs.sum() + 1e-12))
            top_ratio = float(svs[0] / (svs[1] + 1e-12)) if len(svs) > 1 else float("inf")

            profiles.append(LayerProfile(
                layer=i,
                space=space,
                mean_norm=float(norms.mean()),
                std_norm=float(norms.std()),
                singular_values=[float(s) for s in svs[:max_sv]],
                effective_rank=float(effective_rank),
                anisotropy=anisotropy,
                top_sv_ratio=top_ratio,
                dim=mat.shape[1],
            ))

        if (i + 1) % 7 == 0:
            log.info(f"  Layer {i+1}/{num_layers} profiled")

    result = {
        "model": model_id,
        "num_prompts": len(prompts),
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "kv_dim": kv_dim,
        "profiles": [vars(p) for p in profiles],
    }

    Path(output_path).write_text(json.dumps(result, indent=2))
    log.info(f"Profile saved to {output_path}")

    del model
    gc.collect()

    return result


def summarize_profile(profile_path: str):
    """Print a compact summary of the manifold profile."""
    data = json.loads(Path(profile_path).read_text())
    print(f"Model: {data['model']}")
    print(f"Layers: {data['num_layers']}, Hidden: {data['hidden_dim']}, KV: {data['kv_dim']}")
    print(f"Prompts: {data['num_prompts']}")
    print()
    print(f"{'Layer':>5} {'Space':>6} {'MeanNorm':>9} {'EffRank':>8} {'Aniso':>7} {'SV1/SV2':>8}")
    print("-" * 55)

    for p in data["profiles"]:
        print(f"{p['layer']:>5} {p['space']:>6} {p['mean_norm']:>9.2f} "
              f"{p['effective_rank']:>8.1f} {p['anisotropy']:>7.3f} {p['top_sv_ratio']:>8.1f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize_profile(sys.argv[2] if len(sys.argv) > 2 else "manifold_profile.json")
    else:
        model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
        output = sys.argv[2] if len(sys.argv) > 2 else "manifold_profile.json"
        profile_model(model_id=model_id, output_path=output)
