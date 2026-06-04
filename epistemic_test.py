"""Epistemic Dimension Test — controlled minimal pairs for PC2 hypothesis.

Agni requirements:
  1. 500+ minimal pairs (same topic, length, register — only certainty varies)
  2. Permutation baseline (shuffle labels, check if separation holds)
  3. Cross-validation (held-out pairs)
  4. Causal intervention placeholder (activation patching, future work)

Minimal pair format:
  CERTAIN:   "The capital of France is Paris."
  UNCERTAIN: "The capital of France might be Paris."
  Same everything except one hedging word.
"""

import json
import logging
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import ttest_ind, permutation_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# Generate minimal pairs — certain vs uncertain versions of the same statement
TEMPLATES = [
    ("The answer is {x}.", "The answer might be {x}."),
    ("This is {x}.", "This could be {x}."),
    ("It was {x}.", "It may have been {x}."),
    ("{X} causes {y}.", "{X} possibly causes {y}."),
    ("The result is {x}.", "The result appears to be {x}."),
    ("{X} is true.", "{X} is probably true."),
    ("{X} happened in {y}.", "{X} likely happened in {y}."),
    ("The process produces {x}.", "The process might produce {x}."),
    ("{X} leads to {y}.", "{X} may lead to {y}."),
    ("The value equals {x}.", "The value seems to equal {x}."),
]

FILLERS = [
    {"x": "correct", "X": "The hypothesis", "y": "improvement"},
    {"x": "significant", "X": "The effect", "y": "changes"},
    {"x": "stable", "X": "The system", "y": "equilibrium"},
    {"x": "42", "X": "The measurement", "y": "2024"},
    {"x": "positive", "X": "The outcome", "y": "recovery"},
    {"x": "increasing", "X": "The trend", "y": "growth"},
    {"x": "efficient", "X": "The method", "y": "savings"},
    {"x": "complete", "X": "The analysis", "y": "conclusions"},
    {"x": "valid", "X": "The argument", "y": "acceptance"},
    {"x": "optimal", "X": "The solution", "y": "performance"},
    {"x": "expected", "X": "The behavior", "y": "predictions"},
    {"x": "normal", "X": "The distribution", "y": "parameters"},
    {"x": "confirmed", "X": "The finding", "y": "replication"},
    {"x": "reduced", "X": "The intervention", "y": "outcomes"},
    {"x": "consistent", "X": "The data", "y": "models"},
    {"x": "necessary", "X": "The condition", "y": "results"},
    {"x": "sufficient", "X": "The evidence", "y": "conviction"},
    {"x": "relevant", "X": "The factor", "y": "decisions"},
    {"x": "accurate", "X": "The prediction", "y": "validation"},
    {"x": "reliable", "X": "The test", "y": "deployment"},
    {"x": "robust", "X": "The model", "y": "generalization"},
    {"x": "clear", "X": "The pattern", "y": "detection"},
    {"x": "present", "X": "The signal", "y": "observations"},
    {"x": "real", "X": "The effect", "y": "experiments"},
    {"x": "strong", "X": "The correlation", "y": "causation"},
    {"x": "feasible", "X": "The approach", "y": "implementation"},
    {"x": "beneficial", "X": "The treatment", "y": "patients"},
    {"x": "effective", "X": "The strategy", "y": "goals"},
    {"x": "critical", "X": "The requirement", "y": "success"},
    {"x": "fundamental", "X": "The principle", "y": "understanding"},
    {"x": "resolved", "X": "The issue", "y": "investigation"},
    {"x": "improved", "X": "The performance", "y": "optimization"},
    {"x": "detected", "X": "The anomaly", "y": "monitoring"},
    {"x": "verified", "X": "The claim", "y": "evidence"},
    {"x": "established", "X": "The protocol", "y": "standards"},
    {"x": "observed", "X": "The phenomenon", "y": "studies"},
    {"x": "preserved", "X": "The structure", "y": "transformation"},
    {"x": "maintained", "X": "The balance", "y": "regulation"},
    {"x": "achieved", "X": "The target", "y": "planning"},
    {"x": "functional", "X": "The component", "y": "testing"},
    {"x": "measurable", "X": "The impact", "y": "assessment"},
    {"x": "controllable", "X": "The variable", "y": "design"},
    {"x": "reproducible", "X": "The experiment", "y": "methodology"},
    {"x": "scalable", "X": "The architecture", "y": "demand"},
    {"x": "compatible", "X": "The format", "y": "systems"},
    {"x": "aligned", "X": "The objective", "y": "priorities"},
    {"x": "connected", "X": "The network", "y": "nodes"},
    {"x": "secure", "X": "The connection", "y": "encryption"},
    {"x": "available", "X": "The resource", "y": "allocation"},
    {"x": "documented", "X": "The process", "y": "compliance"},
]


def generate_pairs():
    """Generate all template × filler combinations."""
    pairs = []
    for certain_tmpl, uncertain_tmpl in TEMPLATES:
        for filler in FILLERS:
            try:
                certain = certain_tmpl.format(**filler)
                uncertain = uncertain_tmpl.format(**filler)
                pairs.append({"certain": certain, "uncertain": uncertain})
            except KeyError:
                continue
    return pairs


def run_test(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    device="cpu",
    layer=14,
    output_path="epistemic_test_results.json",
):
    pairs = generate_pairs()
    log.info(f"Generated {len(pairs)} minimal pairs")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map=device,
    )
    model.eval()

    captures = {}
    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        captures["h"] = h.detach().float().squeeze(0).mean(dim=0).cpu().numpy()

    hook = model.model.layers[layer].register_forward_hook(hook_fn)

    certain_vecs = []
    uncertain_vecs = []

    for i, pair in enumerate(pairs):
        for label, text in [("certain", pair["certain"]), ("uncertain", pair["uncertain"])]:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                model(**inputs)
            vec = captures["h"].copy()
            if label == "certain":
                certain_vecs.append(vec)
            else:
                uncertain_vecs.append(vec)

        if (i + 1) % 100 == 0:
            log.info(f"  [{i+1}/{len(pairs)}] processed")

    hook.remove()

    certain_mat = np.array(certain_vecs)
    uncertain_mat = np.array(uncertain_vecs)
    log.info(f"Certain: {certain_mat.shape}, Uncertain: {uncertain_mat.shape}")

    # Compute difference vectors (paired)
    diff = certain_mat - uncertain_mat
    mean_diff = diff.mean(axis=0)
    diff_norm = np.linalg.norm(mean_diff)
    diff_direction = mean_diff / (diff_norm + 1e-12)

    # Project all vectors onto the difference direction
    certain_proj = certain_mat @ diff_direction
    uncertain_proj = uncertain_mat @ diff_direction

    # Paired t-test (each pair is matched)
    from scipy.stats import ttest_rel
    t_stat, p_value = ttest_rel(certain_proj, uncertain_proj)
    effect_size = (certain_proj.mean() - uncertain_proj.mean()) / np.std(certain_proj - uncertain_proj)

    log.info(f"\n=== EPISTEMIC DIMENSION TEST (N={len(pairs)}) ===")
    log.info(f"Certain projection:   {certain_proj.mean():.4f} ± {certain_proj.std():.4f}")
    log.info(f"Uncertain projection: {uncertain_proj.mean():.4f} ± {uncertain_proj.std():.4f}")
    log.info(f"Paired t-test: t={t_stat:.3f}, p={p_value:.6f}")
    log.info(f"Cohen's d: {effect_size:.3f}")

    # Permutation test
    log.info("\nRunning permutation test (10000 permutations)...")
    proj_diffs = certain_proj - uncertain_proj
    observed_mean = proj_diffs.mean()

    rng = np.random.default_rng(42)
    n_perm = 10000
    perm_means = np.zeros(n_perm)
    for p in range(n_perm):
        signs = rng.choice([-1, 1], size=len(proj_diffs))
        perm_means[p] = (proj_diffs * signs).mean()

    perm_p = (np.abs(perm_means) >= np.abs(observed_mean)).mean()
    log.info(f"Permutation p-value: {perm_p:.6f}")

    # Cross-validation: split pairs in half, train direction on first half, test on second
    log.info("\nCross-validation (50/50 split)...")
    n = len(pairs)
    idx = rng.permutation(n)
    train_idx, test_idx = idx[:n//2], idx[n//2:]

    train_diff = (certain_mat[train_idx] - uncertain_mat[train_idx]).mean(axis=0)
    train_dir = train_diff / (np.linalg.norm(train_diff) + 1e-12)

    test_certain_proj = certain_mat[test_idx] @ train_dir
    test_uncertain_proj = uncertain_mat[test_idx] @ train_dir
    cv_t, cv_p = ttest_rel(test_certain_proj, test_uncertain_proj)
    cv_d = (test_certain_proj.mean() - test_uncertain_proj.mean()) / np.std(test_certain_proj - test_uncertain_proj)

    log.info(f"CV test split: t={cv_t:.3f}, p={cv_p:.6f}, d={cv_d:.3f}")

    # Project through w_v and w_k
    w_v = model.model.layers[layer].self_attn.v_proj.weight.detach().float().cpu().numpy()
    w_k = model.model.layers[layer].self_attn.k_proj.weight.detach().float().cpu().numpy()
    v_proj_energy = float(np.linalg.norm(w_v @ diff_direction))
    k_proj_energy = float(np.linalg.norm(w_k @ diff_direction))
    log.info(f"\nEpistemic direction visibility: w_v={v_proj_energy:.4f}, w_k={k_proj_energy:.4f}")

    results = {
        "n_pairs": len(pairs),
        "layer": layer,
        "certain_mean": float(certain_proj.mean()),
        "uncertain_mean": float(uncertain_proj.mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(effect_size),
        "permutation_p": float(perm_p),
        "cv_t": float(cv_t),
        "cv_p": float(cv_p),
        "cv_d": float(cv_d),
        "w_v_energy": v_proj_energy,
        "w_k_energy": k_proj_energy,
    }

    Path(output_path).write_text(json.dumps(results, indent=2))
    log.info(f"\nResults saved to {output_path}")

    del model
    return results


if __name__ == "__main__":
    import sys
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    run_test(device=device)
