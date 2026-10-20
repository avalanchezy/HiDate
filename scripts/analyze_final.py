"""Analyze complete frozen HiDate outputs without fitting or selecting a policy."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
import json
import os
from pathlib import Path
import time

import numpy as np

from hidate.io import write_json


METHODS = ("asof_bm25", "asof_dense", "asof_hybrid", "history_unbound", "history_bound", "history_current", "history_learned")
REFERENCES = ("closed_book", "released_oracle")
BOOTSTRAP_SEED = 42
BOOTSTRAP_DRAWS = 2000


def json_rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def interval(values):
    valid = values[np.isfinite(values)]
    return {"ci95": np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
            "n_valid_draws": int(len(valid))}


def cluster_statistics(samples):
    """Resample paired clusters, then form sample-weighted numerator/denominator ratios.

    For a mean this is the repository paired_bootstrap estimator. Keeping separate
    denominators also supports selective precision when accepted sets differ.
    """
    ids = [row["id"] for row in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Paired analysis IDs must be unique within each panel")
    groups = sorted({row["group_id"] for row in samples})
    group_rows = {group: i for i, group in enumerate(groups)}
    methods = list(samples[0]["values"])
    metrics = list(samples[0]["values"][methods[0]])
    numerator = np.zeros((len(groups), len(methods), len(metrics)), dtype=np.float64)
    denominator = np.zeros_like(numerator)
    for row in samples:
        group = group_rows[row["group_id"]]
        for m, method in enumerate(methods):
            for k, metric in enumerate(metrics):
                num, den = row["values"][method][metric]
                numerator[group, m, k] += num
                denominator[group, m, k] += den
    draw = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(groups), (BOOTSTRAP_DRAWS, len(groups)))
    resampled_num, resampled_den = numerator[draw].sum(axis=1), denominator[draw].sum(axis=1)
    draws = np.full_like(resampled_num, np.nan)
    np.divide(resampled_num, resampled_den, out=draws, where=resampled_den > 0)
    num, den = numerator.sum(axis=0), denominator.sum(axis=0)
    estimates = {}
    for m, method in enumerate(methods):
        estimates[method] = {metric: {"mean": float(num[m, k] / den[m, k]) if den[m, k] else None,
            "numerator": float(num[m, k]), "denominator": int(den[m, k]),
            "denominator_groups": int((denominator[:, m, k] > 0).sum()),
            **interval(draws[:, m, k])} for k, metric in enumerate(metrics)}
    contrasts = []
    for a, b in combinations(range(len(methods)), 2):
        for k, metric in enumerate(metrics):
            left, right = estimates[methods[a]][metric], estimates[methods[b]][metric]
            contrasts.append({"left": methods[a], "right": methods[b], "metric": metric,
                "delta_right_minus_left": right["mean"] - left["mean"] if left["mean"] is not None and right["mean"] is not None else None,
                "left_denominator": left["denominator"], "right_denominator": right["denominator"],
                "n_pairs": len(samples), "n_groups": len(groups), **interval(draws[:, b, k] - draws[:, a, k])})
    return {"n_pairs": len(samples), "n_groups": len(groups), "cluster_ids": groups,
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_draws": BOOTSTRAP_DRAWS,
        "estimator": "sample-weighted ratio of cluster sums; paired percentile cluster bootstrap",
        "estimates": estimates, "contrasts": contrasts}


def risk_curve(samples, cutoff):
    """One point per tied positive-confidence group; zero confidence stays withheld."""
    tied = defaultdict(list)
    for row in samples:
        if row["score"] > 0:
            tied[row["score"]].append(row["correct"])
    points = [{"threshold": None, "accepted": 0, "coverage": 0., "precision": None, "risk": None}]
    accepted = correct = 0
    for threshold, labels in sorted(tied.items(), reverse=True):
        accepted += len(labels)
        correct += sum(labels)
        points.append({"threshold": threshold, "accepted": accepted, "coverage": accepted / len(samples),
                       "precision": correct / accepted, "risk": 1. - correct / accepted})
    labels = [row["correct"] for row in samples if row["accepted"]]
    return {"queries": len(samples), "zero_score_withheld": sum(row["score"] <= 0 for row in samples),
        "frozen_cutoff": cutoff, "frozen_point": {"accepted": len(labels), "coverage": len(labels) / len(samples),
            "precision": sum(labels) / len(labels) if labels else None,
            "risk": 1. - sum(labels) / len(labels) if labels else None}, "points": points}


def facets(records):
    """Complete variant/stratum/date panels plus the original source-quote subset."""
    panels = defaultdict(list)
    for record in records:
        variant, stratum, date, absent = record["query_variant"], record["stratum"], record["target_asof"], record["source_quote_absent_from_target"]
        for s, d in (("all", "all"), (stratum, "all"), ("all", date), (stratum, date)):
            panels[(variant, s, d, None)].append(record)
            if absent is True:
                panels[(variant, s, d, True)].append(record)
    return panels


def natural_values(condition, available, raw):
    labels, accepted = condition["structural"], int(condition["accepted"])
    correct = int(labels["top_parent_correct"])
    values = {"top_address_accuracy": [int(labels["top_address_correct"]), 1],
        "target_revision_accuracy": [int(labels["top_revision_correct"]), int(available)],
        "target_parent_accuracy": [correct, int(available)],
        "mrr_at4": [labels["target_mrr"], int(available)],
        "recall_at4": [int(labels["target_recall_at_four"]), int(available)],
        "coverage": [accepted, 1], "support_precision": [accepted * correct, accepted],
        "support_risk": [accepted * (1 - correct), accepted],
        "unavailable_rejection_rate": [int(not available) * (1 - accepted), int(not available)],
        "unavailable_false_support_rate": [int(not available) * accepted, int(not available)]}
    values.update(raw_mrr_at50=[raw["mrr"], int(available)], raw_recall_at50=[raw["recall"], int(available)])
    return values


def controlled_values(condition, target_address):
    accepted = int(condition["accepted"])
    found = any(tuple(row["address"]) == target_address for row in condition["address_diagnostics"])
    return {"rejection_rate": [1 - accepted, 1], "false_support_rate": [accepted, 1],
        "nonempty_evidence_rate": [int(bool(condition["evidence_parent_ids"])), 1],
        "address_discovery_rate": [int(found), 1], "support_precision": [0, accepted]}


def raw_discovery(discovery_root, chunks_path, queries):
    parents = {row["chunk_id"]: row["parent_passage_id"] for row in json_rows(chunks_path)}
    raw = {}
    for name, variant in (("test-natural-exact", "exact_quote"), ("test-natural-paraphrase", "generated_paraphrase")):
        directory = Path(discovery_root) / name
        if not read_json(directory / "discovery.json")["complete"]:
            raise ValueError("Raw final discovery must be complete for Recall/MRR@50")
        seen = set()
        for row in json_rows(directory / "candidates.jsonl"):
            identity = row["query_id"]
            if identity in seen:
                raise ValueError("Raw discovery query IDs must occur exactly once")
            seen.add(identity)
            query = queries[identity]
            for method in METHODS[:4]:
                hits = row["history"]["hybrid"] if method == "history_unbound" else row["asof"][method[5:]]
                rank = next((i for i, hit in enumerate(hits[:50], 1)
                    if query["target_status"] == "available" and parents[hit["chunk_id"]] == query["target_passage_id"]), None)
                raw[(identity, method)] = {"mrr": 1. / rank if rank else 0., "recall": int(rank is not None)}
        expected = {identity for identity, q in queries.items() if q["query_variant"] == variant}
        if seen != expected:
            raise ValueError("Raw discovery coverage must match all natural TEST query IDs")
    return raw


def analyze_final(final_dir, prepared_dir, controls_dir, discovery_root, chunks_path, output_dir, report_path,
                  *, protocol_path=Path("configs/retrieval-protocol.json")):
    started = time.perf_counter()
    final, prepared, controls, output = map(Path, (final_dir, prepared_dir, controls_dir, output_dir))
    evaluation = read_json(final / "evaluation.json")
    if not evaluation["complete"] or evaluation["phase"] != "complete":
        raise ValueError("Final model output must be complete before saved-output analysis")
    policy, protocol = evaluation["frozen_policy"], read_json(protocol_path)
    if not policy["complete"] or set(policy["method_thresholds"]) != set(METHODS):
        raise ValueError("Final analysis requires the already frozen seven-method policy")
    exact = {q["query_id"]: q for q in json_rows(prepared / "queries-exact.jsonl") if q["split"] == "test"}
    queries = dict(exact)
    for q in json_rows(prepared / "queries-paraphrase.jsonl"):
        if q["split"] == "test":
            queries[q["query_id"]] = {**q, "source_quote_absent_from_target": exact[q["base_query_id"]]["source_quote_absent_from_target"]}
    external = {q["query_id"]: q for q in json_rows(prepared / "lawqa-retrieval-queries.jsonl")}
    questions = {q["ファイル名"]: q for q in json_rows(prepared / "lawqa-external.jsonl")}
    dates = protocol["LawQA_corpus_conditions"]
    if len(questions) != evaluation["external_questions"] or {(q["original_question_id"], q["target_asof"]) for q in external.values()} != {
            (identity, date) for identity in questions for date in dates}:
        raise ValueError("External analysis requires all original questions at every frozen corpus condition")
    control_manifest = read_json(controls / "controls.json")
    if not control_manifest["complete"]:
        raise ValueError("Controlled-removal audit must be complete")
    controlled_base = set()
    for date in control_manifest["counts_by_date"]:
        controlled_base.update(read_json(controls / (date + ".json"))["eligible_base_query_ids"])
    if controlled_base != {identity for identity, q in exact.items() if q["target_status"] == "available"}:
        raise ValueError("Controlled analysis coverage must match the frozen available TEST base IDs")
    expected = {"natural": set(queries), "controlled": {identity for identity, q in queries.items()
        if q.get("base_query_id", identity) in controlled_base}, "lawqa": set(external)}
    observed, records = defaultdict(set), {scope: [] for scope in expected}
    external_conditions = {}
    for row in json_rows(final / "retrieval.jsonl"):
        scope, identity = row["evaluation_scope"], row["query_id"]
        if identity not in expected[scope] or identity in observed[scope] or set(row["conditions"]) != set(METHODS):
            raise ValueError("Complete retrieval coverage requires every saved query and all seven methods")
        observed[scope].add(identity)
        for method, condition in row["conditions"].items():
            threshold = policy["method_thresholds"][method]["threshold"]
            if condition["support_threshold"] != threshold or bool(condition["accepted"]) != (condition["support_confidence"] > 0 and condition["support_confidence"] >= threshold):
                raise ValueError("Saved acceptance must use the frozen DEV cutoff")
        if scope == "lawqa":
            q = external[identity]
            for method, condition in row["conditions"].items():
                external_conditions[(q["original_question_id"], q["target_asof"], method)] = {
                    key: condition[key] for key in ("accepted", "support_confidence")}
        else:
            q = queries[identity]
            compact = {}
            for method, condition in row["conditions"].items():
                compact[method] = {key: condition[key] for key in
                    ("accepted", "support_confidence", "support_threshold", "evidence_parent_ids", "structural")}
                compact[method]["address_diagnostics"] = [{"address": address["address"], "status": address["status"],
                    "parent_passage_ids": address.get("parent_passage_ids", [])} for address in condition["address_diagnostics"]]
            records[scope].append({"query_id": identity, "query_variant": q["query_variant"], "conditions": compact,
                "query": q, "stratum": q["stratum"], "target_asof": q["target_asof"],
                "source_quote_absent_from_target": q["source_quote_absent_from_target"]})
    if any(observed[scope] != identities for scope, identities in expected.items()) or {
            scope: len(ids) for scope, ids in observed.items()} != evaluation["retrieval_rows_by_scope"]:
        raise ValueError("Complete retrieval coverage disagrees with the query maps or final receipt")
    answers = {}
    for row in json_rows(final / "answers.jsonl"):
        identity, method = row["original_question_id"], row["method"]
        date = row["target_asof"] if row["evaluation_scope"] == "lawqa" else "reference"
        key = (identity, date, method)
        if key in answers or identity not in questions:
            raise ValueError("Saved answer coverage requires unique original-question/date/method keys")
        gold = questions[identity]["output"].strip().lower()
        if row["released_answer"] != gold:
            raise ValueError("Saved answer labels must match the original released LawQA labels")
        answers[key] = {"correct": int(row["prediction"] == gold), "accepted": row["accepted"]}
    answer_keys = {(identity, date, method) for identity in questions for date in dates for method in METHODS} | {
        (identity, "reference", method) for identity in questions for method in REFERENCES}
    if set(answers) != answer_keys or len(answers) != evaluation["reader_calls"]:
        raise ValueError("Saved answer coverage must include every method/date and both once-per-question references")

    raw = raw_discovery(discovery_root, chunks_path, queries)
    panels, contrasts, curves = {"natural": [], "controlled": [], "lawqa": []}, [], []
    for scope in ("natural", "controlled"):
        for (variant, stratum, date, absent), rows in sorted(facets(records[scope]).items(), key=lambda item: str(item[0])):
            metadata = {"evaluation_scope": scope, "query_variant": variant, "stratum": stratum,
                "target_asof": date, "source_quote_absent_from_target": absent, "cluster_unit": "source family"}
            samples = []
            scored = defaultdict(list)
            for row in rows:
                q, values = row["query"], {}
                for method in METHODS:
                    condition = row["conditions"][method]
                    if scope == "natural":
                        available = q["target_status"] == "available"
                        if method in METHODS[4:]:
                            rank = next((i for i, address in enumerate(condition["address_diagnostics"][:50], 1)
                                if available and q["target_passage_id"] in address.get("parent_passage_ids", [])), None)
                            raw_metric = {"mrr": 1. / rank if rank else 0., "recall": int(rank is not None)}
                        else:
                            raw_metric = raw[(q["query_id"], method)]
                        values[method] = natural_values(condition, available, raw_metric)
                        correct = int(condition["structural"]["top_parent_correct"])
                    else:
                        target = q["target_key"]
                        values[method] = controlled_values(condition, (target[0], target[2], target[3]))
                        correct = 0
                    scored[method].append({"id": q["query_id"], "score": condition["support_confidence"],
                        "correct": correct, "accepted": condition["accepted"]})
                samples.append({"id": q["query_id"], "group_id": q["family_id"], "values": values})
            stats = cluster_statistics(samples)
            panels[scope].append({**metadata, **{key: value for key, value in stats.items() if key != "contrasts"}})
            contrasts.extend({**metadata, **contrast} for contrast in stats["contrasts"])
            for method in METHODS:
                curves.append({**metadata, "method": method, "label": "source-derived available target-parent support",
                    **risk_curve(scored[method], policy["method_thresholds"][method]["threshold"])})

    reference_samples = []
    for identity in sorted(questions):
        values = {method: {"accuracy": [answers[(identity, "reference", method)]["correct"], 1]} for method in REFERENCES}
        reference_samples.append({"id": identity, "group_id": identity, "values": values})
    references = cluster_statistics(reference_samples)
    contrasts.extend({"evaluation_scope": "lawqa_reference", "target_asof": "once_per_question",
        "cluster_unit": "original question", **contrast} for contrast in references.pop("contrasts"))
    for date in dates:
        samples, scored = [], defaultdict(list)
        for identity in sorted(questions):
            values = {}
            for method in REFERENCES + METHODS:
                answer = answers[(identity, "reference" if method in REFERENCES else date, method)]
                correct, accepted = answer["correct"], int(answer["accepted"])
                values[method] = {"accuracy": [correct, 1], "coverage": [accepted, 1],
                    "selective_accuracy": [accepted * correct, accepted], "selective_risk": [accepted * (1 - correct), accepted]}
                if method in METHODS:
                    condition = external_conditions[(identity, date, method)]
                    if bool(answer["accepted"]) != bool(condition["accepted"]):
                        raise ValueError("External reader acceptance must match frozen retrieval acceptance")
                    scored[method].append({"id": identity, "score": condition["support_confidence"],
                        "correct": correct, "accepted": accepted})
            samples.append({"id": identity, "group_id": identity, "values": values})
        stats = cluster_statistics(samples)
        metadata = {"evaluation_scope": "lawqa", "target_asof": date, "cluster_unit": "original question"}
        panels["lawqa"].append({**metadata, **{key: value for key, value in stats.items() if key != "contrasts"}})
        contrasts.extend({**metadata, **contrast} for contrast in stats["contrasts"])
        for method in METHODS:
            curves.append({**metadata, "method": method, "label": "released LawQA choice correctness",
                **risk_curve(scored[method], policy["method_thresholds"][method]["threshold"])})

    work_fields = ("devices", "model_pins", "stack_model_pins", "logical_pair_requests", "physical_pair_count",
        "exact_pair_cache_reuses", "reranker_model_call_count", "reranker_encoded_input_tokens", "reranker_padded_token_slots",
        "reranker_wall_seconds_with_cuda_sync", "reranker_sdpa_backend", "scope_variant_condition_pair_requests",
        "scope_variant_condition_requested_input_tokens", "condition_pair_requests", "condition_requested_input_tokens",
        "reader_calls", "reader_work", "answer_counts", "wall_seconds")
    analysis = {"complete": False, "job_id": os.environ.get("PBS_JOBID"),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "final_evaluation_path": str((final / "evaluation.json").resolve()),
        "sample_counts": {**{scope: len(ids) for scope, ids in observed.items()}, "answers": len(answers)},
        "methods": list(METHODS), "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_draws": BOOTSTRAP_DRAWS,
        "natural": panels["natural"], "controlled": panels["controlled"], "lawqa": panels["lawqa"],
        "lawqa_references": references, "frozen_policy": policy, "protocol": protocol,
        "controlled_removal": control_manifest, "work": {key: evaluation[key] for key in work_fields if key in evaluation},
        "exposures": {"preparation": read_json(prepared / "preparation.json"), "ranker": read_json(final / "ranker-used.json"),
            "development_scoring": read_json(policy["scoring_path"])},
        "raw_discovery_rank_unit": "first target-parent chunk rank for asof/unbound; saved full address ordering for bound/current/learned",
        "source_quote_subset": "original exact source_quote_absent_from_target, also used for generated queries",
        "runtime_accounting": "shared measured model work; per-method pair/token requests are logical, not independent runtimes",
        "training_runs": 0, "model_forward_calls": 0, "dev_updates": 0, "wall_seconds": time.perf_counter() - started}
    write_json(output / "analysis.json", analysis)
    write_json(output / "paired-intervals.json", {"complete": True, "seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS,
        "direction": "right minus left; all method pairs retained", "contrasts": contrasts})
    write_json(output / "risk-coverage.json", {"complete": True, "threshold_source": "frozen DEV method_thresholds",
        "test_threshold_updates": 0, "curves": curves})
    write_report(analysis, contrasts, report_path)
    analysis.update(complete=True, wall_seconds=time.perf_counter() - started)
    write_json(output / "analysis.json", analysis)
    return analysis


def formatted(estimate):
    if estimate["mean"] is None:
        return "undefined (n=0)"
    lo, hi = estimate["ci95"]
    return f"{estimate['mean']:.4f} [{lo:.4f}, {hi:.4f}] (n={estimate['denominator']})"


def write_report(analysis, contrasts, report_path):
    lines = ["# HiDate final saved-output results", "", f"Final receipt: `{analysis['final_evaluation_path']}`.", "",
        f"Bootstrap: {BOOTSTRAP_DRAWS} paired cluster draws with fixed RNG seed {BOOTSTRAP_SEED}; structural clusters are observed source families and LawQA clusters are original questions within each corpus date.", "",
        "Structural parent/revision/MRR/Recall estimates use available-target denominators; unavailable strata retain address and rejection metrics. Selective ratios use accepted denominators and remain undefined when all rows are withheld.", "",
        "The source quote subset uses the original exact source quote flag for both query variants. LawQA corpus dates are index conditions; structural support labels and released choice correctness are separate endpoints.", "",
        "## Complete sample counts", "", "| Scope | Saved rows |", "|---|---:|"]
    lines.extend(f"| {scope} | {count} |" for scope, count in analysis["sample_counts"].items())
    for scope in ("natural", "controlled"):
        lines.extend(["", f"## {scope.title()} structural panels", ""])
        for panel in analysis[scope]:
            label = f"{panel['query_variant']} / {panel['stratum']} / {panel['target_asof']}"
            if panel["source_quote_absent_from_target"] is True:
                label += " / original source quote absent"
            lines.extend([f"### {label}", "", f"Queries: {panel['n_pairs']}; observed families: {panel['n_groups']}.", ""])
            metrics = ("mrr_at4", "recall_at4", "target_parent_accuracy", "target_revision_accuracy", "coverage", "support_precision") if scope == "natural" else (
                "rejection_rate", "false_support_rate", "address_discovery_rate", "nonempty_evidence_rate")
            if panel["stratum"] == "unavailable_target" and scope == "natural":
                metrics = ("top_address_accuracy", "unavailable_rejection_rate", "unavailable_false_support_rate", "coverage", "support_precision")
            lines.extend(["| Method | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)])
            lines.extend("| " + method + " | " + " | ".join(formatted(panel["estimates"][method][metric]) for metric in metrics) + " |" for method in METHODS)
            if scope == "natural" and panel["stratum"] != "unavailable_target":
                lines.extend(["", "| Method | Raw MRR@50 | Raw Recall@50 |", "|---|---|---|"])
                lines.extend("| " + method + " | " + " | ".join(formatted(panel["estimates"][method][metric])
                    for metric in ("raw_mrr_at50", "raw_recall_at50")) + " |" for method in METHODS)
    lines.extend(["", "## LawQA answers by corpus condition", ""])
    for panel in analysis["lawqa"]:
        lines.extend([f"### Corpus {panel['target_asof']}", "", f"Original questions: {panel['n_pairs']}; paired clusters: {panel['n_groups']}.", "",
            "| Method | Forced-choice accuracy | Coverage | Accepted accuracy | Accepted risk |", "|---|---|---|---|---|"])
        lines.extend("| " + method + " | " + " | ".join(formatted(values[metric]) for metric in
            ("accuracy", "coverage", "selective_accuracy", "selective_risk")) + " |" for method, values in panel["estimates"].items())
    lines.extend(["", "Closed-book and released-context references were executed once per original question; their same saved outputs are joined into each corpus-date comparison.", "",
        "## Paired effects in complete overall panels", "", "All values are right minus left; negative effects are retained.", "",
        "| Scope / variant / date | Metric | Left | Right | Delta [95% CI] | Rows / clusters |", "|---|---|---|---|---|---|"])
    for row in contrasts:
        if row["delta_right_minus_left"] is None:
            continue
        if row["evaluation_scope"] in ("natural", "controlled") and (row["stratum"] != "all" or row["target_asof"] != "all" or row["source_quote_absent_from_target"] is not None):
            continue
        metric = row["metric"]
        if metric not in ("mrr_at4", "recall_at4", "target_parent_accuracy", "rejection_rate", "false_support_rate", "accuracy"):
            continue
        label = " / ".join(str(row.get(key, "")) for key in ("evaluation_scope", "query_variant", "target_asof"))
        lo, hi = row["ci95"]
        lines.append(f"| {label} | {metric} | {row['left']} | {row['right']} | {row['delta_right_minus_left']:+.4f} [{lo:+.4f}, {hi:+.4f}] | {row['n_pairs']} / {row['n_groups']} |")
    lines.extend(["", "Raw rank units: " + analysis["raw_discovery_rank_unit"] + ".", "",
        "## Saved work and exposure records", "", "Shared measured execution and hardware:", "", "```json",
        json.dumps(analysis["work"], ensure_ascii=False, indent=2), "```", "",
        "Logical pair/token requests are reported by scope, variant and method above; the shared cached reranker time is not an independently measured latency for each method.", "",
        f"TRAIN fit runs recorded: {analysis['exposures']['ranker']['fit_runs']}; TRAIN statistics: `{json.dumps(analysis['exposures']['ranker']['training_statistics'], ensure_ascii=False)}`.", "",
        f"DEV scoring rows: {analysis['exposures']['development_scoring']['query_rows']}; frozen alpha: {analysis['frozen_policy']['alpha']}.", "",
        f"Preparation families by split: `{json.dumps(analysis['exposures']['preparation']['statistics']['families_by_split'], ensure_ascii=False)}`; exact-query counts by split: `{json.dumps(analysis['exposures']['preparation']['statistics']['queries_by_split'], ensure_ascii=False)}`.", "",
        f"DEV variant rows: `{json.dumps(analysis['exposures']['development_scoring'].get('variants', {}), ensure_ascii=False)}`; frozen method cutoffs: `{json.dumps(analysis['frozen_policy']['method_thresholds'], ensure_ascii=False)}`.", "",
        f"Frozen budgets: discovery {analysis['protocol']['discovery_top_k']}, reranker {analysis['protocol']['reranker_top_k']}, reader {analysis['protocol']['reader_max_passages']} passages / {analysis['protocol']['reader_prompt_tokens']} full prompt tokens.", "",
        "Full subgroup estimates, every paired contrast, tied risk/coverage points and frozen-cutoff results are saved in analysis.json, paired-intervals.json and risk-coverage.json.", ""])
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-dir", type=Path, default=Path("runs/hidate/final"))
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/processed/hidate"))
    parser.add_argument("--controls-dir", type=Path, default=Path("runs/hidate/controlled-removal"))
    parser.add_argument("--discovery-root", type=Path, default=Path("runs/hidate/final-discovery"))
    parser.add_argument("--chunks", type=Path, default=Path("data/index/hidate/chunks.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/hidate/analysis"))
    parser.add_argument("--report", type=Path, default=Path("reports/hidate-final-results.md"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/retrieval-protocol.json"))
    args = parser.parse_args()
    result = analyze_final(args.final_dir, args.prepared_dir, args.controls_dir, args.discovery_root,
        args.chunks, args.output_dir, args.report, protocol_path=args.protocol)
    print(json.dumps({"complete": result["complete"], "sample_counts": result["sample_counts"],
                      "bootstrap_draws": BOOTSTRAP_DRAWS, "training_runs": 0, "model_forward_calls": 0}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
