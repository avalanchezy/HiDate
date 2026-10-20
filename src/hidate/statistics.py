"""Paired cluster bootstrap with explicit ID alignment."""

import numpy as np


def paired_bootstrap(a, b, metric, seed=42, n_resamples=2000):
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    aa, bb = {r["id"]: r for r in a}, {r["id"]: r for r in b}
    if len(aa) != len(a) or len(bb) != len(b):
        raise ValueError("Duplicate ids")
    if aa.keys() != bb.keys():
        raise ValueError("Paired ids must match exactly")
    groups, excluded = {}, 0
    for key in sorted(aa):
        left, right = aa[key], bb[key]
        if left.get("group_id", key) != right.get("group_id", key):
            raise ValueError(f"group_id differs for {key}")
        if left.get(metric) is None or right.get(metric) is None:
            excluded += 1
            continue
        delta = float(right[metric]) - float(left[metric])
        if not np.isfinite(delta):
            raise ValueError("Metrics must be finite")
        groups.setdefault(left.get("group_id", key), []).append(delta)
    if not groups:
        raise ValueError("No complete metric pairs")
    sums = np.array([sum(v) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_resamples):
        indices = rng.integers(0, len(groups), len(groups))
        estimates.append(float(sums[indices].sum() / counts[indices].sum()))
    return {"metric": metric, "delta_b_minus_a": float(sums.sum() / counts.sum()), "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(), "n_pairs": int(counts.sum()), "n_groups": len(groups), "excluded_missing_metric": excluded, "seed": seed, "n_resamples": n_resamples, "method": "paired percentile cluster bootstrap; sample-weighted mean"}
