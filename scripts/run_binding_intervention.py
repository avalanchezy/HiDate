"""Prepare, read and score the paired Fiscal209 fixed-address version intervention."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path

from hidate.io import write_json
from hidate import fiscal as fiscal
from hidate.binding_intervention import (CONDITIONS, exact_input_key, freeze_addresses,
    pack_paired_versions, response_from_result, single_sequence)
from hidate.statistics import paired_bootstrap


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def save_rows(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def receipt(**values):
    return {"at_utc": datetime.now(timezone.utc).isoformat(), "job_id": os.environ.get("PBS_JOBID"),
            "seed": 42, "training_runs": 0, "french_fit_runs": 0, "dev_updates": 0, **values}


def minimal_pin(pin):
    return {k: pin[k] for k in ("model_id", "revision")}


def validate_mechanism(path):
    if path is None:
        return None
    config = read_json(path)
    b = config["binding_intervention"]
    if (b["freeze_address_source"] != "raw_final_selected" or b["fit_runs"] != 0
            or b["frozen_alpha"] != 1.0
            or tuple(a["condition_tag"] for a in b["arms"]) != CONDITIONS):
        raise ValueError("Binding worker differs from the frozen fixed-address protocol")
    return b


def write_output(record, result, provenance, responses, receipts):
    response = response_from_result(record, result, provenance)
    saved = {"qid": record["qid"], "condition": record["condition"], "reader_pin": record["reader_pin"],
             "decode": record["decode"], "messages": record["messages"], "input_ids": record["input_ids"],
             "attention_mask": record["attention_mask"], "provenance": provenance, "result": result}
    receipts.write(json.dumps(saved, ensure_ascii=False, allow_nan=False) + "\n")
    receipts.flush()
    responses.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
    responses.flush()
    return saved


def prepare_binding(prepared_dir, retrieved_dir, original_reader_dir, stack_path, output_dir,
                    fiscal_protocol_path, *, tokenizer=None, expected_questions=209,
                    mechanism_protocol_path=None):
    b = validate_mechanism(mechanism_protocol_path)
    protocol, stack = read_json(fiscal_protocol_path), read_json(stack_path)
    if b is not None and (b["expected_questions"] != expected_questions
            or b["packing"]["max_prompt_tokens"] != protocol["reader_prompt_tokens"]):
        raise ValueError("Fiscal cohort or reader budget differs from the frozen mechanism protocol")
    prepared, retrieved, original, output = map(Path, (prepared_dir, retrieved_dir, original_reader_dir, output_dir))
    retrieval_meta, old_reader_meta = read_json(retrieved / "retrieval.json"), read_json(original / "reader.json")
    if not stack["complete"] or not retrieval_meta["complete"] or retrieval_meta["alpha"] != 1.0:
        raise ValueError("Binding intervention requires the completed frozen Qwen3 French transfer")
    existing = output / "preparation.json"
    if existing.exists() and (saved_preparation := read_json(existing))["complete"]:
        if (saved_preparation["reader_pin"] != minimal_pin(stack["models"]["reader"])
                or saved_preparation["expected_questions"] != expected_questions
                or saved_preparation["retrieved_dir"] != str(retrieved)
                or saved_preparation["prompt_budget"] != protocol["reader_prompt_tokens"]
                or saved_preparation["decode"] != {"do_sample": False, "max_new_tokens": protocol["reader_max_new_tokens"]}):
            raise ValueError("Existing completed preparation belongs to a different input contract")
        return saved_preparation
    questions = {q["qid"]: q for q in rows(prepared / "questions.jsonl") if q["scope"] == "in" and q.get("regime") == "R3"}
    if len(questions) != expected_questions:
        raise ValueError(f"Expected every primary question: {len(questions)}/{expected_questions}")
    if b is not None and len({q["gold"]["article_cid"] for q in questions.values()}) != b["expected_gold_article_clusters"]:
        raise ValueError("Primary cohort gold-CID cluster count differs from the frozen protocol")
    for role, field in (("embedding", "embedding_pin"), ("reranker", "reranker_pin")):
        if field in retrieval_meta and minimal_pin(retrieval_meta[field]) != minimal_pin(stack["models"][role]):
            raise ValueError("Saved French retrieval and the frozen transfer stack have different pins")
    versions = {r["id"]: r for r in rows(prepared / "articles.jsonl")}
    retrieved_rows = {r["qid"]: r for r in rows(retrieved / "retrieval.jsonl") if r["qid"] in questions}
    if set(retrieved_rows) != set(questions):
        raise ValueError("Original learned final retrieval does not cover the complete primary cohort")
    pin = minimal_pin(stack["models"]["reader"])
    decode = {"do_sample": False, "max_new_tokens": protocol["reader_max_new_tokens"]}
    if b is not None and decode != b["reader"]["decode"]:
        raise ValueError("Reader decode differs from the frozen mechanism protocol")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(stack["models"]["reader"]["local_path"], local_files_only=True)
    old_responses = {r["qid"]: r for r in rows(original / "responses.jsonl")
                     if r["qid"] in questions and r["condition"] == "history_learned"}
    records, frozen, raw_counts, old_counts = [], [], Counter(), Counter()
    differences = truncated = same_versions = identical_prompts = 0
    for qid, question in questions.items():
        retrieval = retrieved_rows[qid]
        slots = freeze_addresses(question, retrieval, versions, max_passages=protocol["reader_max_passages"])
        cids = [s["article_cid"] for s in slots]
        old_evidence = old_responses[qid]["evidence"]
        old_cids = [e["article_cid"] for e in old_evidence]
        raw_counts[len(cids)] += 1; old_counts[len(old_cids)] += 1
        differences += cids != old_cids
        truncated += any(e.get("truncated", False) for e in old_evidence)
        packed = pack_paired_versions(question["prompt"], slots, versions, tokenizer,
                                     prompt_budget=protocol["reader_prompt_tokens"])
        same_version = all(s["bound_version_id"] == s["source_version_id"] for s in slots)
        identical = packed[CONDITIONS[0]]["input_ids"] == packed[CONDITIONS[1]]["input_ids"] and packed[CONDITIONS[0]]["messages"] == packed[CONDITIONS[1]]["messages"]
        same_versions += same_version; identical_prompts += identical
        frozen.append({"qid": qid, "date_anchor": question["date_anchor"], "slots": slots,
            "freeze_address_source": "raw_final_selected", "frozen_article_cids": cids,
            "original_packed_article_cids": old_cids, "raw_equals_original_packed": cids == old_cids,
            "original_packed_any_truncated": any(e.get("truncated", False) for e in old_evidence),
            "same_version_pair": same_version, "identical_rendered_prompt": identical})
        selected = retrieval["conditions"]["history_learned"]
        for condition in CONDITIONS:
            arm = packed[condition]
            if [e["article_cid"] for e in arm["evidence"]] != cids:
                raise ValueError("Joint packing changed the frozen address identity/order")
            records.append({"qid": qid, "condition": condition, "reader_pin": pin, "decode": decode,
                "date_anchor": question["date_anchor"], "scope": "in", "frozen_article_cids": cids,
                "gold_version_id": question["gold"]["version_id"], "accepted": selected["accepted"],
                "support_confidence": selected["support_confidence"], "support_threshold": selected["support_threshold"],
                **arm})

    # Only exact desired messages need cache lookup. Legacy receipts do not save
    # input IDs; reconstruct them with the same pinned tokenizer and check length.
    desired = {json.dumps(r["messages"], ensure_ascii=False, sort_keys=True): r for r in records}
    cache = {}
    compatible = (old_reader_meta["complete"] and minimal_pin(old_reader_meta["reader_pin"]) == pin
                  and old_reader_meta["decode"] == decode)
    if compatible:
        for line, saved in enumerate(rows(original / "reader-receipts.jsonl"), 1):
            message_key = json.dumps(saved["messages"], ensure_ascii=False, sort_keys=True)
            if message_key not in desired or saved["model"] != pin["model_id"]:
                continue
            candidate, result = desired[message_key], saved["result"]
            if result["max_new_tokens"] != decode["max_new_tokens"] or result["prompt_tokens"] != len(candidate["input_ids"]):
                continue
            ids = tokenizer.apply_chat_template(saved["messages"], tokenize=True, add_generation_prompt=True)
            if list(ids) != candidate["input_ids"]:
                continue
            if "input_ids" in result and single_sequence(result["input_ids"]) != candidate["input_ids"]:
                continue
            if "attention_mask" in result and single_sequence(result["attention_mask"]) != candidate["attention_mask"]:
                continue
            cache.setdefault(exact_input_key(candidate), (result, {
                "kind": "old_exact_prompt_reuse", "source_path": str(original / "reader-receipts.jsonl"),
                "source_line": line, "source_qid": saved["qid"], "source_condition": saved["condition"],
                "input_verification": "saved_ids" if "input_ids" in result else "reconstructed_saved_prompt"}))
    output.mkdir(parents=True, exist_ok=True)
    save_rows(output / "inputs.jsonl", records)
    save_rows(output / "frozen-addresses.jsonl", frozen)
    hits, pending = 0, {}
    with (output / "responses.jsonl").open("w", encoding="utf-8") as responses, \
            (output / "reader-receipts.jsonl").open("w", encoding="utf-8") as receipts:
        for record in records:
            key = exact_input_key(record)
            if key in cache:
                write_output(record, *cache[key], responses, receipts)
                hits += 1
            else:
                pending.setdefault(key, {"qid": record["qid"], "condition": record["condition"]})
    save_rows(output / "pending-inputs.jsonl", pending.values())
    report = receipt(complete=True, questions=len(questions), logical_outputs=len(records), reader_pin=pin, decode=decode,
        protocol_path=str(mechanism_protocol_path) if mechanism_protocol_path else None,
        fiscal_protocol_path=str(fiscal_protocol_path), prepared_dir=str(prepared), retrieved_dir=str(retrieved),
        original_reader_dir=str(original), expected_questions=expected_questions,
        gold_article_clusters=len({q["gold"]["article_cid"] for q in questions.values()}),
        freeze_address_source="raw_final_selected", cid_order_invariant=True,
        old_exact_cache_compatible=compatible, old_exact_cache_hits=hits, pending_logical_outputs=len(records)-hits,
        pending_unique_prompts=len(pending), same_version_pairs=same_versions, identical_rendered_prompt_pairs=identical_prompts,
        raw_vs_original_packed={"different_cid_lists": differences, "raw_slot_counts": dict(raw_counts),
            "original_packed_slot_counts": dict(old_counts), "original_any_truncated": truncated},
        prompt_budget=protocol["reader_prompt_tokens"], model_forward_calls=0,
        old_retrieval_metadata=retrieval_meta)
    write_json(output / "preparation.json", report)
    print(json.dumps({k: report[k] for k in ("complete", "questions", "logical_outputs", "old_exact_cache_hits", "pending_unique_prompts")}))
    return report


def run_reader(output_dir, stack_path, *, reader_factory=None):
    output = Path(output_dir)
    preparation, stack = read_json(output / "preparation.json"), read_json(stack_path)
    if not preparation["complete"] or minimal_pin(stack["models"]["reader"]) != preparation["reader_pin"]:
        raise ValueError("Reader pin differs from complete prepared matched inputs")
    inputs = list(rows(output / "inputs.jsonl"))
    expected = {(r["qid"], r["condition"]) for r in inputs}
    receipts_path, responses_path = output / "reader-receipts.jsonl", output / "responses.jsonl"
    saved_receipts = {(r["qid"], r["condition"]): r for r in rows(receipts_path)}
    completed = {(r["qid"], r["condition"]) for r in rows(responses_path)}
    if not completed.issubset(expected):
        raise ValueError("Existing reader output has records outside this fixed intervention")
    exact, reader = {}, None
    for record in inputs:
        identity = record["qid"], record["condition"]
        if identity in saved_receipts:
            saved = saved_receipts[identity]
            if exact_input_key(saved) != exact_input_key(record):
                raise ValueError("Existing receipt input differs; retain this run and use a new output directory")
            exact.setdefault(exact_input_key(record), (saved["result"], {"kind": "same_run_exact_prompt_reuse",
                "source_path": str(receipts_path), "source_qid": identity[0], "source_condition": identity[1]}))
        elif identity in completed:
            raise ValueError("Completed response lacks its input/answer receipt")
    work = Counter()
    with responses_path.open("a", encoding="utf-8") as responses, receipts_path.open("a", encoding="utf-8") as receipts:
        for record in inputs:
            identity = record["qid"], record["condition"]
            if identity in completed:
                continue
            key = exact_input_key(record)
            if key in exact:
                result, provenance = exact[key]
                work["same_run_prompt_reuses"] += 1
            else:
                if reader is None:
                    if reader_factory is None:
                        import torch
                        from hidate.models import QwenReader
                        torch.manual_seed(42)
                        reader_factory = lambda pin, budget: QwenReader(pin["local_path"], device="cuda", prompt_budget=budget)
                    reader = reader_factory(stack["models"]["reader"], preparation["prompt_budget"])
                result = reader.generate_paraphrase(record["messages"],
                    max_new_tokens=record["decode"]["max_new_tokens"], include_inputs=True)
                if (single_sequence(result["input_ids"]) != record["input_ids"]
                        or single_sequence(result["attention_mask"]) != record["attention_mask"]):
                    raise ValueError("Actual reader inputs differ from prepared matched prompt")
                provenance = {"kind": "new_physical_reader_call", "job_id": os.environ.get("PBS_JOBID")}
                work.update(calls=1, prompt_tokens=result["prompt_tokens"], generated_tokens=result["generated_tokens"])
                work["seconds"] += result["seconds"]
                exact[key] = (result, {"kind": "same_run_exact_prompt_reuse", "source_path": str(receipts_path),
                                      "source_qid": record["qid"], "source_condition": record["condition"]})
            write_output(record, result, provenance, responses, receipts)
            completed.add(identity)
            if len(completed) % 20 == 0:
                print(json.dumps({"completed_logical_outputs": len(completed), "expected": len(expected), "new_physical_calls": work["calls"]}), flush=True)
    physical = {exact_input_key(r): r for r in rows(receipts_path)
                if r["provenance"]["kind"] == "new_physical_reader_call"}
    total_work = {"calls": len(physical), **{key: sum(r["result"][field] for r in physical.values())
                  for key, field in (("prompt_tokens", "prompt_tokens"), ("generated_tokens", "generated_tokens"), ("seconds", "seconds"))}}
    report = receipt(complete=completed == expected, completed_logical_outputs=len(completed), expected_outputs=len(expected),
        reader_pin=preparation["reader_pin"], decode=preparation["decode"],
        new_physical_reader_calls_this_job=work["calls"], work_this_job=dict(work),
        new_physical_reader_calls_total=len(physical), physical_work_total=total_work,
        old_exact_cache_hits=preparation["old_exact_cache_hits"], cid_order_invariant=True)
    write_json(output / "reader.json", report)
    return report


def score_binding(output_dir, prepared_dir, official_scorer_path, *, expected_questions=209):
    output = Path(output_dir)
    preparation = read_json(output / "preparation.json")
    questions = {q["qid"]: q for q in rows(Path(prepared_dir) / "questions.jsonl") if q["scope"] == "in" and q.get("regime") == "R3"}
    if len(questions) != expected_questions:
        raise ValueError("Scoring cohort differs from the complete primary cohort")
    spec = importlib.util.spec_from_file_location("hidate_binding_official_scorer", official_scorer_path)
    scorer = importlib.util.module_from_spec(spec); spec.loader.exec_module(scorer)
    nuggets = defaultdict(list)
    for n in rows(Path(prepared_dir) / "nuggets.jsonl"):
        nuggets[n["qid"]].append(n)
    frozen = {r["qid"]: r for r in rows(output / "frozen-addresses.jsonl")}
    responses = list(rows(output / "responses.jsonl"))
    expected = {(qid, c) for qid in questions for c in CONDITIONS}
    identities = [(r["qid"], r["condition"]) for r in responses]
    if len(identities) != len(set(identities)) or not set(identities).issubset(expected):
        raise ValueError("Duplicate or unexpected matched reader responses")
    scores = []
    for r in responses:
        qid = r["qid"]
        if [e["article_cid"] for e in r["evidence"]] != frozen[qid]["frozen_article_cids"]:
            raise ValueError("Reader response changed the frozen CID order")
        scores.append({"qid": qid, "condition": r["condition"], "accepted": r["accepted"],
            "gold_article_cid": questions[qid]["gold"]["article_cid"],
            **fiscal.score_answer(questions[qid], r["response_text"], nuggets[qid], r["evidence"], scorer)})
    save_rows(output / "scores.jsonl", scores)
    metrics = ("strict", "coverage", "gold_article_recall_at_four", "gold_version_recall_at_four", "top_article_correct", "top_version_correct", "article_citation_nugget", "version_id_cited")
    conditions = {}
    for condition in CONDITIONS:
        selected = [s for s in scores if s["condition"] == condition]
        panel = {"n": len(selected), "expected_n": expected_questions}
        for metric in metrics:
            values = [s[metric] for s in selected if s[metric] is not None]
            panel[metric] = sum(values) / len(values) if values else None
            panel[metric + "_denominator"] = len(values)
        conditions[condition] = panel
    complete = set(identities) == expected
    paired, transitions = {}, Counter()
    if complete:
        indexed = {(s["qid"], s["condition"]): s for s in scores}
        for qid in questions:
            bound = indexed[qid, "binding_bound"]["strict"]; source = indexed[qid, "binding_source"]["strict"]
            transitions["both_strict_correct" if bound and source else "bound_only_strict_correct" if bound
                        else "source_only_strict_correct" if source else "both_strict_incorrect"] += 1
        for metric in ("strict", "coverage", "gold_version_recall_at_four", "top_version_correct"):
            source, bound = [], []
            for qid, q in questions.items():
                base = {"id": qid, "group_id": q["gold"]["article_cid"]}
                source.append({**base, metric: indexed[qid, "binding_source"][metric]})
                bound.append({**base, metric: indexed[qid, "binding_bound"][metric]})
            paired[metric] = paired_bootstrap(source, bound, metric, seed=42, n_resamples=2000)
    groups = {"same_version": set(), "different_version": set(), "identical_rendered_prompt": set(), "different_rendered_prompt": set()}
    for qid, f in frozen.items():
        groups["same_version" if f["same_version_pair"] else "different_version"].add(qid)
        groups["identical_rendered_prompt" if f["identical_rendered_prompt"] else "different_rendered_prompt"].add(qid)
    strata = {name: {c: {"n": len(values := [s for s in scores if s["qid"] in qids and s["condition"] == c]),
                        "strict": sum(s["strict"] for s in values)/len(values) if values else None,
                        "coverage": sum(s["coverage"] for s in values)/len(values) if values else None}
                       for c in CONDITIONS} for name, qids in groups.items()}
    report = receipt(complete=complete, paired_questions=expected_questions if complete else len(set(q for q, _ in identities)),
        logical_outputs=len(scores), expected_outputs=len(expected), conditions=conditions,
        primary_scope="all primary209 forced outputs; no acceptance filter", paired_bound_minus_source=paired,
        strict_transitions=dict(transitions), descriptive_strata=strata,
        official_scorer_path=str(official_scorer_path), cid_order_invariant=True,
        gold_article_clusters=len({q["gold"]["article_cid"] for q in questions.values()}),
        preparation_path=str(output / "preparation.json"), old_exact_cache_hits=preparation["old_exact_cache_hits"],
        interpretation="New matched fixed-address/version conditions; no post-swap reranking; not original unmatched history_unbound versus learned")
    write_json(output / "evaluation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "read", "score"))
    parser.add_argument("--prepared", type=Path, default=Path("data/processed/fiscal"))
    parser.add_argument("--retrieved", type=Path, default=Path("runs/hidate/fiscal-transfer/fiscal-retrieval"))
    parser.add_argument("--original-reader", type=Path, default=Path("runs/hidate/fiscal-transfer/fiscal"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fiscal/official"))
    parser.add_argument("--stack", type=Path, default=Path("models/qwen3/model-stack.json"))
    parser.add_argument("--fiscal-protocol", type=Path, default=Path("configs/fiscal_protocol.json"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/core-mechanisms.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/hidate/mechanism-analysis/binding-fiscal"))
    args = parser.parse_args()
    validate_mechanism(args.protocol)
    if args.action == "prepare":
        prepare_binding(args.prepared, args.retrieved, args.original_reader, args.stack, args.output,
                        args.fiscal_protocol, mechanism_protocol_path=args.protocol)
    elif args.action == "read":
        result = run_reader(args.output, args.stack)
        if not result["complete"]:
            raise ValueError("Matched reader outputs are incomplete")
        score_binding(args.output, args.prepared, args.raw / "scripts/score_nuggets.py")
    else:
        score_binding(args.output, args.prepared, args.raw / "scripts/score_nuggets.py")


if __name__ == "__main__":
    main()
