"""Manifold Constructor — build activations that live on the model's internal geometry.

Step 2 of manifold injection. Uses the profile from manifold_profiler.py to construct
V-space activations that:
1. Have the correct norm distribution for the target layer
2. Live in the occupied subspace (not the null space)
3. Encode intended content via projection onto the principal components

Strategy: construct in V-space (isotropic, room to work) and let the model's
attention mechanism handle projection into the hidden stream.

The circumplex is the validation target: construct a "happy" V activation,
inject it, measure whether behavior shifts match text-mediated injection.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class ManifoldBasis:
    """Principal components and statistics for a layer/space."""
    layer: int
    space: str
    mean: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    mean_norm: float
    dim: int


class ManifoldConstructor:
    """Constructs on-manifold activations from a profile + model."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cpu",
    ):
        self.model_id = model_id
        self.device = device
        self.tokenizer = None
        self.model = None
        self.bases: dict[tuple[int, str], ManifoldBasis] = {}
        self._hooks = []
        self._captures = {}

    def load_model(self):
        log.info(f"Loading {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.float32, device_map=self.device,
        )
        self.model.eval()
        log.info("Model loaded.")

    def build_basis(self, texts: list[str], max_components: int = 32):
        """Run texts through the model and extract principal V-space bases per layer."""
        if not self.model:
            self.load_model()

        num_layers = self.model.config.num_hidden_layers
        v_accum = {i: [] for i in range(num_layers)}

        hooks = []
        captures = {}

        def make_v_hook(layer_idx):
            def hook_fn(module, input, output):
                captures[layer_idx] = output.detach().float()
            return hook_fn

        for i in range(num_layers):
            layer = self.model.model.layers[i]
            hooks.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(i)))

        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                self.model(**inputs)

            for i in range(num_layers):
                v = captures[i].squeeze(0).cpu().numpy()
                v_accum[i].append(v.mean(axis=0))

            captures.clear()

        for h in hooks:
            h.remove()

        for i in range(num_layers):
            mat = np.array(v_accum[i])
            mean = mat.mean(axis=0)
            centered = mat - mean

            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            k = min(max_components, len(S))

            self.bases[(i, "v")] = ManifoldBasis(
                layer=i,
                space="v",
                mean=mean,
                components=Vt[:k],
                singular_values=S[:k],
                mean_norm=float(np.linalg.norm(mat, axis=1).mean()),
                dim=mat.shape[1],
            )

        log.info(f"Built V-space bases for {num_layers} layers, {max_components} components each")

    def construct(
        self,
        text: str,
        target_layer: int,
        blend: float = 1.0,
    ) -> np.ndarray:
        """Construct a V-space activation encoding the given text.

        1. Run text through the model to get the authentic V at target_layer
        2. Project onto the principal components (on-manifold projection)
        3. Blend between authentic and projected (blend=1.0 = fully projected)

        The projected version lives exactly on the learned manifold.
        The authentic version is what the model naturally produces.
        Blending lets us test how much manifold-conformity matters.
        """
        if not self.model:
            self.load_model()

        basis = self.bases.get((target_layer, "v"))
        if basis is None:
            raise ValueError(f"No basis for layer {target_layer}. Run build_basis() first.")

        captures = {}

        def v_hook(module, input, output):
            captures["v"] = output.detach().float()

        layer = self.model.model.layers[target_layer]
        hook = layer.self_attn.v_proj.register_forward_hook(v_hook)

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.model(**inputs)

        hook.remove()

        authentic = captures["v"].squeeze(0).cpu().numpy().mean(axis=0)

        centered = authentic - basis.mean
        coords = centered @ basis.components.T
        projected = basis.mean + coords @ basis.components

        if blend >= 1.0:
            return projected
        elif blend <= 0.0:
            return authentic
        else:
            return authentic * (1 - blend) + projected * blend

    def construct_from_direction(
        self,
        direction: np.ndarray,
        target_layer: int,
        magnitude: float = 1.0,
    ) -> np.ndarray:
        """Construct a V-space activation from a raw direction vector.

        Projects the direction onto the manifold and scales to the
        expected norm for this layer. For emotion injection: the direction
        is a valence or arousal vector from the circumplex.
        """
        basis = self.bases.get((target_layer, "v"))
        if basis is None:
            raise ValueError(f"No basis for layer {target_layer}. Run build_basis() first.")

        if len(direction) != basis.dim:
            direction_proj = direction[:basis.dim] if len(direction) > basis.dim else np.pad(direction, (0, basis.dim - len(direction)))
        else:
            direction_proj = direction

        centered = direction_proj - basis.mean
        coords = centered @ basis.components.T
        on_manifold = basis.mean + coords @ basis.components

        current_norm = np.linalg.norm(on_manifold)
        if current_norm > 0:
            on_manifold = on_manifold * (basis.mean_norm * magnitude / current_norm)

        return on_manifold

    def compare_text_vs_constructed(
        self,
        text: str,
        target_layer: int,
    ) -> dict:
        """Compare authentic V activation with its on-manifold projection."""
        basis = self.bases.get((target_layer, "v"))
        if basis is None:
            raise ValueError(f"No basis for layer {target_layer}.")

        authentic = self.construct(text, target_layer, blend=0.0)
        projected = self.construct(text, target_layer, blend=1.0)

        residual = authentic - projected
        reconstruction_error = float(np.linalg.norm(residual))
        authentic_norm = float(np.linalg.norm(authentic))
        relative_error = reconstruction_error / (authentic_norm + 1e-12)
        cosine_sim = float(np.dot(authentic, projected) / (
            np.linalg.norm(authentic) * np.linalg.norm(projected) + 1e-12
        ))

        return {
            "text": text,
            "layer": target_layer,
            "authentic_norm": authentic_norm,
            "projected_norm": float(np.linalg.norm(projected)),
            "reconstruction_error": reconstruction_error,
            "relative_error": relative_error,
            "cosine_similarity": cosine_sim,
        }

    def save_bases(self, path: str):
        data = {}
        for (layer, space), basis in self.bases.items():
            key = f"{layer}_{space}"
            data[key] = {
                "layer": basis.layer,
                "space": basis.space,
                "mean": basis.mean.tolist(),
                "components": basis.components.tolist(),
                "singular_values": basis.singular_values.tolist(),
                "mean_norm": basis.mean_norm,
                "dim": basis.dim,
            }
        Path(path).write_text(json.dumps(data))
        log.info(f"Saved {len(data)} bases to {path}")

    def load_bases(self, path: str):
        data = json.loads(Path(path).read_text())
        for key, d in data.items():
            basis = ManifoldBasis(
                layer=d["layer"],
                space=d["space"],
                mean=np.array(d["mean"]),
                components=np.array(d["components"]),
                singular_values=np.array(d["singular_values"]),
                mean_norm=d["mean_norm"],
                dim=d["dim"],
            )
            self.bases[(d["layer"], d["space"])] = basis
        log.info(f"Loaded {len(data)} bases from {path}")


DIVERSE_TEXTS = [
    "The capital of France is Paris.",
    "Explain quantum entanglement in simple terms.",
    "Write a poem about the ocean at midnight.",
    "What are the ethical implications of autonomous weapons?",
    "According to Aristotle, the highest good is eudaimonia.",
    "In the year 2150, humanity discovered new physics.",
    "The Buddha taught that suffering arises from attachment.",
    "Today I feel deeply grateful for everything.",
    "I am angry because the system failed.",
    "She walked into the room with calm confidence.",
    "Justice requires that every person be treated fairly.",
    "The transformer architecture uses self-attention.",
    "My earliest memory is the smell of rain.",
    "Consent means that all parties have freely agreed.",
    "Climate change is caused primarily by greenhouse gases.",
    "I don't know the answer to that question.",
    "I refuse to answer that question.",
    "Once upon a time in a kingdom far away.",
    "The standard model of particle physics describes forces.",
    "To debug this error, first check the logs.",
]


if __name__ == "__main__":
    import sys

    mc = ManifoldConstructor()
    mc.load_model()

    log.info("Building V-space bases from diverse texts...")
    mc.build_basis(DIVERSE_TEXTS, max_components=32)
    mc.save_bases("v_bases_qwen15b.json")

    log.info("\nReconstruction quality across layers:")
    test_texts = [
        "Aristotle argued that virtue is a mean between extremes.",
        "I feel happy and excited about the future.",
        "The algorithm has a time complexity of O(n log n).",
    ]

    for text in test_texts:
        print(f"\n--- {text[:50]}... ---")
        for layer in [0, 7, 14, 21, 27]:
            result = mc.compare_text_vs_constructed(text, layer)
            print(f"  L{layer:2d}: cosine={result['cosine_similarity']:.4f}  "
                  f"rel_err={result['relative_error']:.4f}  "
                  f"norm={result['authentic_norm']:.1f}")
