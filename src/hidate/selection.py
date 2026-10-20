"""Frozen-grid DEV evidence mixing and empirical address-support threshold."""

from collections import defaultdict


def selected_evidence(condition, alpha=0., max_passages=4):
    def score(row):
        return (alpha * row["learned_probability"] + (1 - alpha) * row["reranker_score"]
                if alpha else row["reranker_score"])
    ranked = sorted(condition["reranked_candidates"],
                    key=lambda row: (-score(row), row["selection_rank"], row["chunk_id"]))
    selected, parents = [], set()
    for row in ranked:
        if row["parent_passage_id"] not in parents:
            parents.add(row["parent_passage_id"])
            selected.append({**row, "final_score": float(score(row))})
            if len(selected) == max_passages:
                break
    return selected


def target_mrr(query, evidence):
    if query["target_status"] != "available":
        return 0.
    return next((1. / rank for rank, row in enumerate(evidence, 1)
                 if row["parent_passage_id"] == query["target_passage_id"]), 0.)


def support_score(evidence, address_diagnostics):
    if not evidence:
        return 0.
    confidence = float(evidence[0]["learned_probability"])
    unavailable = max((row["learned_probability"] for row in address_diagnostics
                       if row["status"] != "available"), default=-1.)
    return 0. if unavailable >= confidence else confidence


def baseline_support_score(method, evidence, address_diagnostics):
    if not evidence:
        return 0.
    if method in {"history_bound", "history_current"} and address_diagnostics and address_diagnostics[0]["status"] != "available":
        return 0.
    return float(evidence[0]["reranker_score"])


def precision_threshold(scored_labels, target_precision=.90):
    choices = []
    for threshold in sorted({score for score, _ in scored_labels if score > 0}):
        accepted = [label for score, label in scored_labels if score >= threshold]
        precision = sum(accepted) / len(accepted)
        if precision >= target_precision:
            choices.append({"threshold": float(threshold), "accepted": len(accepted),
                "coverage": len(accepted) / len(scored_labels), "precision": precision})
    return max(choices, key=lambda row: (row["accepted"], -row["threshold"])) if choices else {
        "threshold": 1.000001, "accepted": 0, "coverage": 0., "precision": None}


def choose_policy(rows, queries, alpha_grid, target_precision=.90):
    grid = []
    for alpha in alpha_grid:
        groups = defaultdict(list)
        for row in rows:
            query = queries[row["query_id"]]
            if query["split"] != "valid":
                raise ValueError("Policy selection requires only VALID queries")
            if query["target_status"] == "available":
                evidence = selected_evidence(row["conditions"]["history_learned"], alpha=alpha)
                groups[(query["query_variant"], query["stratum"])].append(target_mrr(query, evidence))
        means = {"|".join(key): sum(values) / len(values) for key, values in groups.items()}
        grid.append({"alpha": alpha, "macro_mrr": sum(means.values()) / len(means), "cell_means": means})
    chosen = max(grid, key=lambda row: (row["macro_mrr"], -row["alpha"]))
    method_labels = defaultdict(list)
    for row in rows:
        query = queries[row["query_id"]]
        for method, condition in row["conditions"].items():
            evidence = selected_evidence(condition, alpha=chosen["alpha"] if method == "history_learned" else 0.)
            confidence = (support_score(evidence, condition["address_diagnostics"]) if method == "history_learned"
                          else baseline_support_score(method, evidence, condition["address_diagnostics"]))
            correct = int(bool(evidence) and query["target_status"] == "available"
                          and evidence[0]["parent_passage_id"] == query["target_passage_id"])
            method_labels[method].append((confidence, correct))
    thresholds = {method: precision_threshold(labels, target_precision) for method, labels in method_labels.items()}
    return {"alpha": chosen["alpha"], "alpha_grid_results": grid,
        "threshold_target_precision": target_precision, "support_threshold": thresholds["history_learned"],
        "method_thresholds": thresholds,
        "support_definition": "top selected parent equals source-derived available target parent",
        "support_unavailable_rule": "withhold when a ranked unavailable address has learned probability at least the selected top parent's",
        "baseline_support_rule": "own top reranker sigmoid; bound methods also withhold when their top ordered address is unavailable; no learned scores",
        "queries": len(rows), "training_runs": 0, "model_forward_calls": 0}
