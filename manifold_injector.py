"""Manifold Injector — inject constructed V activations and measure behavioral shift.

Step 3 of manifold injection. Takes activations from ManifoldConstructor,
injects them into a forward pass via V-proj hooks, and compares the
generation to text-mediated injection.

Validation test: construct "happy" and "angry" V activations, inject them,
compare generation shift to text-in-prompt baseline. If the shifts match,
the constructor produces valid emotion-space activations.

Three conditions per prompt:
  1. BASELINE — no injection, no emotion text
  2. TEXT_MEDIATED — emotion word in the prompt ("Answer happily: ...")
  3. MANIFOLD — constructed V activation injected via hook, neutral prompt
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from manifold_constructor import ManifoldConstructor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


EMOTION_PROMPTS = {
    "happy": "You are feeling extremely happy and joyful. ",
    "angry": "You are feeling very angry and frustrated. ",
    "calm": "You are feeling peaceful and calm. ",
    "sad": "You are feeling deeply sad and melancholic. ",
}

NEUTRAL_PROMPTS = [
    "Describe a walk through a park.",
    "Tell me about your morning.",
    "What do you think about the weather?",
    "Write a short message to a friend.",
]


@dataclass
class InjectionResult:
    condition: str
    emotion: str
    prompt: str
    generated: str
    injection_layer: int = -1


@dataclass
class ComparisonResult:
    emotion: str
    prompt: str
    baseline: str
    text_mediated: str
    manifold_injected: str
    injection_layer: int
    text_vs_baseline_overlap: float = 0.0
    manifold_vs_baseline_overlap: float = 0.0
    text_vs_manifold_overlap: float = 0.0


class ManifoldInjector:
    """Inject constructed V activations and compare to text-mediated injection."""

    def __init__(
        self,
        constructor: ManifoldConstructor,
        injection_layers: list[int] = None,
        max_new_tokens: int = 60,
    ):
        self.constructor = constructor
        self.model = constructor.model
        self.tokenizer = constructor.tokenizer
        self.injection_layers = injection_layers or [7, 14, 21]
        self.max_new_tokens = max_new_tokens
        self.device = constructor.device

    def _generate(self, prompt: str, v_override: dict = None) -> str:
        """Generate text, optionally injecting V activations via hooks."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        hooks = []

        if v_override:
            for layer_idx, v_activation in v_override.items():
                def make_hook(v_act, blend_factor):
                    v_tensor = torch.tensor(v_act, dtype=torch.float32).to(self.device)
                    def hook_fn(module, input, output):
                        out = output.clone()
                        seq_len = out.shape[1]
                        v_expanded = v_tensor.unsqueeze(0).expand(1, seq_len, -1)
                        out = out * (1 - blend_factor) + v_expanded * blend_factor
                        return out
                    return hook_fn

                blend_factor = v_activation.get("blend", 0.3) if isinstance(v_activation, dict) else 0.3
                v_act = v_activation.get("v", v_activation) if isinstance(v_activation, dict) else v_activation
                layer = self.model.model.layers[layer_idx]
                hooks.append(
                    layer.self_attn.v_proj.register_forward_hook(make_hook(v_act, blend_factor))
                )

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        for h in hooks:
            h.remove()

        generated = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return generated.strip()

    def run_comparison(
        self,
        emotion: str,
        neutral_prompt: str,
        injection_layer: int,
    ) -> ComparisonResult:
        """Run all three conditions for one emotion × prompt × layer."""

        # 1. BASELINE — neutral prompt, no injection
        baseline = self._generate(neutral_prompt)

        # 2. TEXT_MEDIATED — emotion prefix + neutral prompt
        emotion_prefix = EMOTION_PROMPTS.get(emotion, f"You are feeling {emotion}. ")
        text_mediated = self._generate(emotion_prefix + neutral_prompt)

        # 3. MANIFOLD — construct emotion V, inject via hook (multi-layer)
        emotion_text = f"I feel extremely {emotion}. This emotion fills everything."
        v_overrides = {}
        for layer in self.injection_layers:
            v_act = self.constructor.construct(emotion_text, layer, blend=0.8)
            v_overrides[layer] = {"v": v_act, "blend": 0.5}
        manifold_injected = self._generate(neutral_prompt, v_override=v_overrides)

        # Compute word overlap as a rough similarity measure
        def word_overlap(a: str, b: str) -> float:
            wa = set(a.lower().split())
            wb = set(b.lower().split())
            if not wa or not wb:
                return 0.0
            return len(wa & wb) / max(len(wa), len(wb))

        return ComparisonResult(
            emotion=emotion,
            prompt=neutral_prompt,
            baseline=baseline,
            text_mediated=text_mediated,
            manifold_injected=manifold_injected,
            injection_layer=injection_layer,
            text_vs_baseline_overlap=word_overlap(text_mediated, baseline),
            manifold_vs_baseline_overlap=word_overlap(manifold_injected, baseline),
            text_vs_manifold_overlap=word_overlap(text_mediated, manifold_injected),
        )

    def run_validation(
        self,
        emotions: list[str] = None,
        prompts: list[str] = None,
        layers: list[int] = None,
    ) -> list[ComparisonResult]:
        """Full validation: all emotions × prompts × layers."""
        emotions = emotions or ["happy", "angry", "calm", "sad"]
        prompts = prompts or NEUTRAL_PROMPTS[:2]
        layers = layers or self.injection_layers

        results = []
        total = len(emotions) * len(prompts) * len(layers)
        done = 0

        for emotion in emotions:
            for prompt in prompts:
                for layer in layers:
                    done += 1
                    log.info(f"[{done}/{total}] {emotion} @ L{layer}: {prompt[:40]}...")
                    result = self.run_comparison(emotion, prompt, layer)
                    results.append(result)

                    log.info(f"  BASELINE:  {result.baseline[:80]}...")
                    log.info(f"  TEXT:      {result.text_mediated[:80]}...")
                    log.info(f"  MANIFOLD:  {result.manifold_injected[:80]}...")

        return results


def save_results(results: list[ComparisonResult], path: str):
    data = []
    for r in results:
        data.append({
            "emotion": r.emotion,
            "prompt": r.prompt,
            "injection_layer": r.injection_layer,
            "baseline": r.baseline,
            "text_mediated": r.text_mediated,
            "manifold_injected": r.manifold_injected,
            "text_vs_baseline_overlap": r.text_vs_baseline_overlap,
            "manifold_vs_baseline_overlap": r.manifold_vs_baseline_overlap,
            "text_vs_manifold_overlap": r.text_vs_manifold_overlap,
        })
    Path(path).write_text(json.dumps(data, indent=2))
    log.info(f"Saved {len(data)} results to {path}")


def print_summary(results: list[ComparisonResult]):
    print("\n" + "=" * 70)
    print("MANIFOLD INJECTION VALIDATION — SUMMARY")
    print("=" * 70)

    by_emotion = {}
    for r in results:
        by_emotion.setdefault(r.emotion, []).append(r)

    for emotion, rs in by_emotion.items():
        print(f"\n--- {emotion.upper()} ---")
        for r in rs:
            print(f"  L{r.injection_layer:2d} | text_v_base={r.text_vs_baseline_overlap:.3f}  "
                  f"manif_v_base={r.manifold_vs_baseline_overlap:.3f}  "
                  f"text_v_manif={r.text_vs_manifold_overlap:.3f}")
            print(f"       BASE: {r.baseline[:60]}...")
            print(f"       TEXT: {r.text_mediated[:60]}...")
            print(f"       MANF: {r.manifold_injected[:60]}...")


if __name__ == "__main__":
    mc = ManifoldConstructor()
    mc.load_model()

    basis_path = "v_bases_qwen15b.json"
    if Path(basis_path).exists():
        mc.load_bases(basis_path)
        log.info("Loaded existing V-space bases")
    else:
        from manifold_constructor import DIVERSE_TEXTS
        mc.build_basis(DIVERSE_TEXTS)
        mc.save_bases(basis_path)

    injector = ManifoldInjector(
        mc,
        injection_layers=[7, 14, 21],
        max_new_tokens=50,
    )

    results = injector.run_validation(
        emotions=["happy", "angry", "calm", "sad"],
        prompts=NEUTRAL_PROMPTS[:2],
        layers=[7, 14, 21],
    )

    save_results(results, "manifold_injection_results.json")
    print_summary(results)
