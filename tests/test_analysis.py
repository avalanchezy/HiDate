"""Saved-output analysis pairs families/questions and retains negative effects."""

import json
from pathlib import Path
import runpy

import pytest

from hidate.io import load_jsonl, write_json, write_jsonl
from hidate.statistics import paired_bootstrap


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_final.py"
METHODS = ("asof_bm25", "asof_dense", "asof_hybrid", "history_unbound", "history_bound", "history_current", "history_learned")
DATES = ("2024-09-30", "2025-11-04", "2026-09-30")


def functions():
    assert SCRIPT.exists(), "Saved-output HiDate final analysis is not implemented"
    return runpy.run_path(str(SCRIPT))


def test_family_bootstrap_is_sample_weighted_paired_and_retains_negative_effect():
    samples = [{"id": str(i), "group_id": "A" if i < 3 else "B", "values": {
        "asof_hybrid": {"mrr_at4": [float(i < 3), 1]},
        "history_learned": {"mrr_at4": [0., 1]}}} for i in range(4)]
    result = functions()["cluster_statistics"](samples)
    left = result["estimates"]["asof_hybrid"]["mrr_at4"]
    contrast = result["contrasts"][0]
    reference = paired_bootstrap(
        [{"id": str(i), "group_id": "A" if i < 3 else "B", "score": float(i < 3)} for i in range(4)],
        [{"id": str(i), "group_id": "A" if i < 3 else "B", "score": 0.} for i in range(4)], "score")
    assert left["mean"] == .75 and left["denominator"] == 4
    assert result["n_pairs"] == 4 and result["n_groups"] == 2
    assert contrast["delta_right_minus_left"] == -.75
    assert contrast["ci95"] == pytest.approx(reference["ci95"])
    assert contrast["n_valid_draws"] == 2000 and result["bootstrap_seed"] == 42


def test_all_withheld_precision_is_undefined_and_tied_curve_never_splits_a_group():
    sample = [{"id": "a", "group_id": "F", "values": {
        "asof_hybrid": {"support_precision": [0, 0]}, "history_learned": {"support_precision": [0, 0]}}}]
    result = functions()["cluster_statistics"](sample)
    estimate = result["estimates"]["asof_hybrid"]["support_precision"]
    assert estimate["mean"] is None and estimate["ci95"] is None and estimate["n_valid_draws"] == 0
    curve = functions()["risk_curve"]([
        {"id": "a", "score": .8, "correct": 1, "accepted": True},
        {"id": "b", "score": .8, "correct": 0, "accepted": True},
        {"id": "c", "score": .3, "correct": 1, "accepted": False},
        {"id": "d", "score": 0., "correct": 0, "accepted": False}], .7)
    assert [point["accepted"] for point in curve["points"]] == [0, 2, 3]
    assert curve["points"][1]["coverage"] == .5 and curve["points"][1]["risk"] == .5
    assert curve["zero_score_withheld"] == 1 and curve["frozen_point"]["accepted"] == 2


@pytest.fixture
def saved_outputs(tmp_path):
    final, prepared, controls, discovery, output = [tmp_path / name for name in
        ("final", "prepared", "controls", "discovery", "analysis")]
    chunks = tmp_path / "chunks.jsonl"
    report = tmp_path / "report.md"
    policy = {"complete": True, "alpha": .5, "queries": 8, "method_thresholds": {
        method: {"threshold": .7, "accepted": 2, "coverage": .25, "precision": 1.} for method in METHODS}}
    dev = tmp_path / "dev-scoring.json"
    write_json(dev, {"complete": True, "query_rows": 8, "variants": {"exact_quote": 4, "generated_paraphrase": 4}})
    policy["scoring_path"] = str(dev)
    write_json(final / "evaluation.json", {"complete": True, "phase": "complete", "job_id": "fixture",
        "conditions": list(METHODS), "frozen_policy": policy, "external_questions": 2,
        "retrieval_rows_by_scope": {"natural": 6, "controlled": 4, "lawqa": 6},
        "expected_retrieval_rows_by_scope": {"natural": 6, "controlled": 4, "lawqa": 6},
        "reader_calls": 46, "expected_reader_calls": 46, "training_runs": 0, "dev_updates": 0,
        "logical_pair_requests": 112, "physical_pair_count": 21, "reranker_encoded_input_tokens": 63,
        "reranker_padded_token_slots": 84, "reranker_wall_seconds_with_cuda_sync": .5,
        "scope_variant_condition_pair_requests": {"natural|exact_quote|asof_hybrid": 3},
        "scope_variant_condition_requested_input_tokens": {"natural|exact_quote|asof_hybrid": 9},
        "devices": {"reranker": {"name": "fixture GPU", "dtype": "torch.bfloat16"}},
        "reader_work": {"input_tokens": 460, "padded_token_slots": 460, "wall_seconds_with_cuda_sync": 2.},
        "model_pins": {"reader": {"model_id": "Qwen", "revision": "frozen"}}})
    write_json(final / "ranker-used.json", {"complete": True, "fit_runs": 1, "random_state": 42,
        "training_statistics": {"query_count": 12, "candidate_count": 100, "families": ["TRAIN"]}})
    write_json(prepared / "preparation.json", {"complete": True, "statistics": {
        "families_by_split": {"train": 1, "valid": 1, "test": 2}, "queries_by_split": {"train": 6, "valid": 4, "test": 3}}})
    exact = [{"query_id": f"E{i}", "query_variant": "exact_quote", "split": "test",
        "family_id": "F1" if i < 3 else "F2", "target_asof": DATES[0] if i == 1 else DATES[-1],
        "stratum": {1: "changed_cross_date", 2: "same_date", 3: "unavailable_target"}[i],
        "target_status": "available" if i < 3 else "absent", "target_passage_id": f"CUR{i}" if i < 3 else None,
        "target_revision_id": "r2", "target_key": [f"L{i}", DATES[0] if i == 1 else DATES[-1], "MainProvision", "1"],
        "source_quote_absent_from_target": True if i == 1 else False if i == 2 else None} for i in (1, 2, 3)]
    paraphrase = [{**q, "query_id": q["query_id"] + "|paraphrase", "base_query_id": q["query_id"],
        "query_variant": "generated_paraphrase", "source_quote_absent_from_target": False} for q in exact]
    write_jsonl(prepared / "queries-exact.jsonl", exact)
    write_jsonl(prepared / "queries-paraphrase.jsonl", paraphrase)
    questions = [{"ファイル名": f"Q{i}", "問題文": "問題", "選択肢": "a b c d", "output": "a"} for i in (1, 2)]
    write_jsonl(prepared / "lawqa-external.jsonl", questions)
    external = [{"query_id": question["ファイル名"] + "|" + date, "original_question_id": question["ファイル名"],
        "target_asof": date, "query_variant": "natural_question", "split": "external", "source_role": "external"}
        for question in questions for date in DATES]
    write_jsonl(prepared / "lawqa-retrieval-queries.jsonl", external)
    write_json(controls / "controls.json", {"complete": True, "eligible_queries": 2,
        "counts_by_date": {DATES[0]: {"eligible_queries": 1}, DATES[-1]: {"eligible_queries": 1}}})
    for q in exact[:2]:
        write_json(controls / (q["target_asof"] + ".json"), {"eligible_base_query_ids": [q["query_id"]],
            "removed_chunk_ids": ["c" + q["query_id"]], "target_overrides": []})
    retrieval = []
    for scope in ("natural", "controlled"):
        for q in exact + paraphrase:
            if scope == "controlled" and q["target_status"] != "available":
                continue
            conditions = {}
            for method in METHODS:
                correct = scope == "natural" and q["target_status"] == "available" and not (method == "history_learned" and q["stratum"] == "changed_cross_date")
                accepted = scope == "natural" and q["target_status"] == "available" or scope == "controlled" and method == "asof_hybrid"
                conditions[method] = {"accepted": accepted, "support_confidence": .9 if accepted else .2,
                    "support_threshold": .7, "requested_pair_count": 1, "requested_input_tokens": 3,
                    "evidence_parent_ids": [q["target_passage_id"] if correct else "WRONG"],
                    "address_diagnostics": [{"address": [q["target_key"][0], "MainProvision", "1"],
                        "status": "corpus_support_unavailable" if scope == "controlled" else q["target_status"]}],
                    "structural": {"top_address_correct": True, "top_revision_correct": correct,
                        "top_parent_correct": correct, "target_mrr": float(correct), "target_recall_at_four": correct}}
            retrieval.append({"query_id": q["query_id"], "query_variant": q["query_variant"], "evaluation_scope": scope,
                "target_asof": q["target_asof"], "conditions": conditions})
    for q in external:
        retrieval.append({**q, "evaluation_scope": "lawqa", "conditions": {method: {
            "accepted": True, "support_confidence": .9, "support_threshold": .7,
            "requested_pair_count": 1, "requested_input_tokens": 3} for method in METHODS}})
    write_jsonl(final / "retrieval.jsonl", retrieval)
    answers = []
    for q in external:
        for method in METHODS:
            correct = (q["target_asof"] != DATES[0]) if method == "history_learned" else (q["target_asof"] == DATES[0])
            answers.append({**q, "evaluation_scope": "lawqa", "method": method, "prediction": "a" if correct else "b",
                "released_answer": "a", "correct": correct, "accepted": True, "support_confidence": .9,
                "support_threshold": .7, "prompt_tokens": 10})
    for question in questions:
        for method in ("closed_book", "released_oracle"):
            answers.append({"original_question_id": question["ファイル名"], "evaluation_scope": "lawqa_reference",
                "method": method, "prediction": "a", "released_answer": "a", "correct": True, "accepted": True,
                "prompt_tokens": 10})
    write_jsonl(final / "answers.jsonl", answers)
    write_jsonl(chunks, [{"chunk_id": "cE" + str(i), "parent_passage_id": "CUR" + str(i)} for i in (1, 2)])
    for name, variant in (("test-natural-exact", "exact_quote"), ("test-natural-paraphrase", "generated_paraphrase")):
        rows = [{"query_id": q["query_id"], "asof": {m: [{"chunk_id": "c" + q["query_id"].split("|")[0], "score": 1.}]
                if q["target_status"] == "available" else [] for m in ("bm25", "dense", "hybrid")},
                "history": {"hybrid": []}} for q in exact + paraphrase if q["query_variant"] == variant]
        write_jsonl(discovery / name / "candidates.jsonl", rows)
        write_json(discovery / name / "discovery.json", {"complete": True})
    return {"final_dir": final, "prepared_dir": prepared, "controls_dir": controls,
        "discovery_root": discovery, "chunks_path": chunks, "output_dir": output, "report_path": report}


def test_saved_analysis_pairs_each_date_by_original_question_and_preserves_all_methods(saved_outputs, monkeypatch):
    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Saved final analysis must not fit a model")
    monkeypatch.setattr("hidate.ranking.fit_address_ranker", forbidden_fit)
    original = (saved_outputs["final_dir"] / "ranker-used.json").read_bytes()
    analysis = functions()["analyze_final"](**saved_outputs)
    assert analysis["complete"] and analysis["training_runs"] == analysis["model_forward_calls"] == analysis["dev_updates"] == 0
    assert analysis["sample_counts"] == {"natural": 6, "controlled": 4, "lawqa": 6, "answers": 46}
    quote = next(p for p in analysis["natural"] if p["query_variant"] == "generated_paraphrase"
        and p["stratum"] == "all" and p["target_asof"] == "all" and p["source_quote_absent_from_target"] is True)
    assert quote["n_pairs"] == 1 and quote["n_groups"] == 1
    assert quote["estimates"]["history_learned"]["mrr_at4"]["mean"] == 0.
    assert len(analysis["lawqa"]) == 3 and all(p["n_pairs"] == p["n_groups"] == 2 for p in analysis["lawqa"])
    assert analysis["lawqa_references"]["n_pairs"] == analysis["lawqa_references"]["n_groups"] == 2
    for panel in analysis["lawqa"]:
        assert set(panel["estimates"]) == set(METHODS) | {"closed_book", "released_oracle"}
    paired = json.loads((saved_outputs["output_dir"] / "paired-intervals.json").read_text())
    effects = [p for p in paired["contrasts"] if p["evaluation_scope"] == "lawqa" and p["metric"] == "accuracy"
        and p["left"] == "asof_hybrid" and p["right"] == "history_learned"]
    assert [p["delta_right_minus_left"] for p in effects] == [-1., 1., 1.]
    assert all(p["n_pairs"] == p["n_groups"] == 2 for p in effects)
    controlled = next(p for p in analysis["controlled"] if p["query_variant"] == "exact_quote"
        and p["stratum"] == p["target_asof"] == "all")
    assert "mrr_at4" not in controlled["estimates"]["history_learned"]
    assert controlled["estimates"]["history_learned"]["rejection_rate"]["mean"] == 1.
    assert controlled["estimates"]["asof_hybrid"]["false_support_rate"]["mean"] == 1.
    assert analysis["work"]["devices"]["reranker"]["name"] == "fixture GPU"
    assert analysis["exposures"]["ranker"]["fit_runs"] == 1
    assert (saved_outputs["final_dir"] / "ranker-used.json").read_bytes() == original
    report = saved_outputs["report_path"].read_text()
    assert "history_learned" in report and "-1.0000" in report and "source quote" in report


def test_incomplete_final_output_blocks_analysis_before_other_inputs(tmp_path):
    final = tmp_path / "final"
    write_json(final / "evaluation.json", {"complete": False})
    with pytest.raises(ValueError, match="complete"):
        functions()["analyze_final"](final, tmp_path / "missing", tmp_path / "missing-controls",
            tmp_path / "missing-discovery", tmp_path / "missing-chunks", tmp_path / "output", tmp_path / "report.md")
    assert not (tmp_path / "output").exists()


def test_missing_saved_answer_cannot_be_hidden_by_complete_manifest(saved_outputs):
    path = saved_outputs["final_dir"] / "answers.jsonl"
    write_jsonl(path, load_jsonl(path)[1:])
    with pytest.raises(ValueError, match="answer.*coverage|coverage.*answer"):
        functions()["analyze_final"](**saved_outputs)
