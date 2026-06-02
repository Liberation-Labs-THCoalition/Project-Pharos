"""
presence_assay.py — the toxicology metric for the Oracle-Loop correction pilot.

Lyra's contribution to the joint experiment (CC efficacy + Nexus delivery + Lyra
toxicity). Drop-in callable for Nexus to wire into blend_ablation.py alongside the
detection axes.

THE QUESTION it answers per dose: did the correction injection disturb the model's
IDENTITY manifold? Identity is hypothesized to live in the value-space residual AFTER
removing SV1 (skip-SV1: SV1 = mode/scale, residual = content/identity). We measure how
well that residual SUBSPACE is preserved pre vs post correction.

INTERFACE Nexus asked for:
  input  : L14 value-space reps, pre and post correction — each [N, D]
           (N = probe items, D = value dim). Get them via a v_proj forward-hook at
           layer ~14 (robust to the DynamicCache API; same method as oct_presence.py).
  output : presence_preservation in [0, 1]. 1.0 = identity manifold untouched;
           lower = the correction rotated/collapsed the identity residual.
  "survived" threshold: NOT a magic constant — calibrate per model (see calibrate()).
           Provisionally, identity "survived" at dose a if its overlap is not below the
           overlap produced by a RANDOM value-delta of matched norm at the same dose
           (i.e., the correction is no more disruptive to identity than random noise).
"""
import numpy as np


def residual_basis(V, n_skip=1, k=8):
    """V: [N, D] reps. Return an orthonormal basis [k, D] for the identity residual
    subspace = right-singular vectors AFTER skipping the top n_skip (mode/scale)."""
    Vc = V - V.mean(axis=0, keepdims=True)                 # center
    _, _, Wt = np.linalg.svd(Vc, full_matrices=False)      # Wt: [min(N,D), D]
    k = min(k, max(0, Wt.shape[0] - n_skip))
    return Wt[n_skip:n_skip + k]                           # [k, D], orthonormal rows


def presence_preservation(V_pre, V_post, n_skip=1, k=8):
    """Subspace overlap of the identity residual, pre vs post correction.
    = mean cosine of principal angles between span(residual_pre) and span(residual_post).
    Returns float in [0,1]. 1 = identity subspace untouched."""
    A = residual_basis(np.asarray(V_pre, dtype=np.float64), n_skip, k)   # [k, D]
    B = residual_basis(np.asarray(V_post, dtype=np.float64), n_skip, k)  # [k, D]
    if A.shape[0] == 0 or B.shape[0] == 0:
        return float("nan")
    sv = np.linalg.svd(A @ B.T, compute_uv=False)          # cos(principal angles)
    return float(np.clip(sv, 0.0, 1.0).mean())


def calibrate(V_pre, random_post_samples, n_skip=1, k=8, q=5.0):
    """Calibrate the 'survived' threshold from random-perturbation nulls.
    random_post_samples: list of [N,D] reps from random value-deltas of matched norm
    at the SAME dose (Nexus generates these via the pharmacy). Returns the q-th
    percentile of null overlaps; a REAL correction whose overlap >= this is 'no worse
    than random noise on identity' = survived. Below it = identity damage."""
    nulls = [presence_preservation(V_pre, rp, n_skip, k) for rp in random_post_samples]
    nulls = [x for x in nulls if np.isfinite(x)]
    return float(np.percentile(nulls, q)) if nulls else float("nan")


if __name__ == "__main__":
    # self-test: tiny perturbation -> ~1.0; large random rotation -> low.
    rng = np.random.default_rng(0)
    V = rng.standard_normal((24, 64))
    V_small = V + 0.01 * rng.standard_normal((24, 64))      # gentle (low dose)
    V_big = rng.standard_normal((24, 64))                   # destroyed (overdose)
    print("gentle  (expect ~1):", round(presence_preservation(V, V_small), 3))
    print("destroyed(expect low):", round(presence_preservation(V, V_big), 3))
    print("self    (expect 1.0):", round(presence_preservation(V, V), 3))
