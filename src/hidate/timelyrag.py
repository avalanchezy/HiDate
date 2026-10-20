"""Independent TimelyRAG auto-alpha scoring with observed as-of date proxies.

The supplied pool is the frozen history-dense top50, before any target binding.
QET is the requested day; QIT and document event times remain unknown. The
earliest visible as-of anchor is an explicit observation proxy for missing DIT,
not a publication timestamp or an inferred legal-validity interval.
"""

from datetime import date

import numpy as np


ALPHA_GRID = tuple(i / 10 for i in range(11))
DAY_GRANULARITY_SIGNAL = 2 / 3  # retrieval.py encodes day as gran_id=2.
DAY_GAP_CAP = 60.0


def _auto_alpha(base, penalty, known_ratio, gap_avg, gap_cap):
    text_variance = float(np.clip(np.std(base) / .35, 0., 1.))
    time_variance = float(np.clip(np.std(penalty) / .35, 0., 1.))
    valid_mean = known_ratio / 2  # QET_DET* and QET_DIT known; both QIT terms absent.
    gap_signal = float(np.clip(gap_avg / gap_cap, 0., 1.))
    importance = float(np.clip(.32 * gap_signal + .18 + .08 * DAY_GRANULARITY_SIGNAL
        + .22 * time_variance + .18 * valid_mean + .02 * known_ratio - .20 * text_variance, 0., 1.))
    continuous = 1. - importance
    alpha = min(ALPHA_GRID, key=lambda value: abs(value - continuous))
    gate = .35 + .10 * DAY_GRANULARITY_SIGNAL + .35 * valid_mean + .20 * time_variance
    if time_variance < .10:
        gate *= .25
    return alpha, float(np.clip(gate, 0., 1.)), {
        "text_std_normalized": text_variance, "time_std_normalized": time_variance,
        "valid_mean": valid_mean, "gap_normalized": gap_signal,
        "time_importance": importance, "alpha_continuous": continuous}


def score_observed_pool(base_scores, observation_dates, target_asof):
    """Score one already retrieved pool, with no gold labels or alpha search.

    Scores and observation_dates are in the same original candidate order.
    Unknown observations are None. Inputs use ISO calendar days. Day distances
    and GAP_MAX['day']=60 use consistent units; all remaining numeric constants
    follow the public auto-alpha recipe. The upstream48-day mapping is recorded
    only as a CPU diagnostic and never selected from relevance outcomes.
    """
    if len(base_scores) != len(observation_dates):
        raise ValueError("Semantic scores and observation dates must align")
    target = date.fromisoformat(target_asof)
    observed = [date.fromisoformat(value) if value is not None else None for value in observation_dates]
    if any(value is not None and value > target for value in observed):
        raise ValueError("The historical pool contains a future observation date")
    raw = np.asarray(base_scores, dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("Semantic retrieval scores must be finite")
    if not len(raw):
        return {"alpha": 1., "gate": 0., "semantic_scores": [], "distance_days": [],
            "time_affinity": [], "final_scores": [], "known_observation_ratio": 0.,
            "time_informative": False, "qet": target_asof, "qit": None, "det": None,
            "gap_avg_days": 0., "valid_ratios": {k: 0. for k in ("QET_DET", "QIT_DIT", "QIT_DET", "QET_DIT")},
            "gap_mapping_correction": {"granularity": "day", "gap_cap_days": DAY_GAP_CAP,
                "upstream_mapped_cap": 48., "upstream_alpha": 1., "alpha_changed": False,
                "max_absolute_score_difference": 0., "ranking_changed": False}, "signals": {}}
    base = (raw - raw.min()) / (np.ptp(raw) + 1e-9)
    # With QIT/DET unknown, the main QET-DIT proxy plus the QET-DIT auxiliary
    # term gives 1.1 times the day distance. No interval end is inferred.
    delta = np.array([1.1 * (target - value).days if value is not None else np.nan for value in observed])
    finite = np.isfinite(delta)
    normalized = delta.copy()
    if finite.any():
        spread = np.ptp(delta[finite])
        normalized[finite] = (delta[finite] - delta[finite].min()) / spread if spread else 0.
    affinity = np.full(len(raw), .5)
    affinity[finite] = np.exp(-2. * normalized[finite])
    # Ordinal distance ranks and deterministic pool-index perturbation retain
    # the public recipe; equal distances are ordered by their original position.
    penalty = np.full(len(raw), 1.05)
    if finite.any():
        positions = np.flatnonzero(finite)
        order = np.argsort(normalized[finite], kind="stable")
        ranks = np.empty(len(positions), dtype=float)
        ranks[order] = np.arange(len(positions))
        penalty[finite] = ranks / (len(positions) - 1) if len(positions) > 1 else 0.
        penalty = np.clip(penalty + 1e-6 * ((np.arange(len(raw)) % 997) / 997.), 0., 1.05)
    known = float(finite.mean())
    sorted_dates = sorted(value for value in observed if value is not None)
    gap_avg = float(np.mean([(right - left).days for left, right in zip(sorted_dates[:-1], sorted_dates[1:])])) if len(sorted_dates) > 1 else 0.
    alpha, gate, signals = _auto_alpha(base, penalty, known, gap_avg, DAY_GAP_CAP)
    margin = np.clip(1. - (base.max() - base) / max(1e-6, float(np.ptp(base))), 0., 1.)

    def combine(value):
        weights = (1. - value) * gate * margin
        return (1. - weights) * base + weights * affinity

    final = combine(alpha)
    upstream_alpha, _, _ = _auto_alpha(base, penalty, known, gap_avg, 48.)
    upstream_final = combine(upstream_alpha)
    return {"alpha": float(alpha), "gate": gate, "semantic_scores": base.tolist(),
        "distance_days": [float(value) if valid else None for value, valid in zip(delta, finite)],
        "time_affinity": affinity.tolist(), "final_scores": final.tolist(),
        "known_observation_ratio": known, "time_informative": bool(np.ptp(affinity) > 0.),
        "qet": target_asof, "qit": None, "det": None, "gap_avg_days": gap_avg,
        "valid_ratios": {"QET_DET": known, "QIT_DIT": 0., "QIT_DET": 0., "QET_DIT": known},
        "signals": signals,
        "gap_mapping_correction": {"granularity": "day", "gap_cap_days": DAY_GAP_CAP,
            "upstream_mapped_cap": 48., "upstream_alpha": float(upstream_alpha),
            "alpha_changed": bool(alpha != upstream_alpha),
            "max_absolute_score_difference": float(np.max(np.abs(final - upstream_final))),
            "ranking_changed": bool(not np.array_equal(np.argsort(final, kind="stable"), np.argsort(upstream_final, kind="stable")))}}


def rerank_observed_hits(hits, chunks, target_asof):
    """Rescore up to50 frozen history-dense chunk hits, preserving source rows.

    The returned score is for source-chunk preselection. Address aggregation,
    shared binding, at most10 CE pairs and final4 evidence remain the caller's
    existing task adapter. This function does not hydrate or read target labels.
    """
    selected, dates, excluded = [], [], 0
    for rank, hit in enumerate(hits[:50], 1):
        chunk = chunks[hit["chunk_id"]]
        anchors = sorted(anchor for anchor in chunk.get("asof_dates", []) if anchor <= target_asof)
        if not anchors and chunk["source_kind"] != "guidance":
            excluded += 1
            continue
        observation = anchors[0] if anchors else None
        selected.append({**hit, "original_candidate_rank": rank, "observation_anchor": observation})
        dates.append(observation)
    scored = score_observed_pool([hit["score"] for hit in selected], dates, target_asof)
    for hit, value, affinity in zip(selected, scored["final_scores"], scored["time_affinity"]):
        hit.update(timelyrag_score=value, time_affinity=affinity)
    selected.sort(key=lambda hit: (-hit["timelyrag_score"], hit["original_candidate_rank"], hit["chunk_id"]))
    return {"method": "TimelyRAG_auto_observed_anchor_day", "date_proxy": "first_visible_requested_asof_anchor",
        "input_hit_count": len(hits), "candidate_count": len(selected),
        "excluded_future_or_unobserved_statute_hits": excluded, "candidates": selected, "scoring": scored}
