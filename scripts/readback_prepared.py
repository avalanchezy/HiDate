"""Check saved HiDate query origins and target labels against saved dated evidence."""

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path

from hidate.io import load_jsonl, write_json


def normal(text):
    return "".join(unicodedata.normalize("NFC", text).split())


def readback(source_dir, prepared_dir):
    source, prepared = Path(source_dir), Path(prepared_dir)
    report = json.loads((prepared / "preparation.json").read_text(encoding="utf-8"))
    allocation = json.loads((prepared / "family-split.json").read_text(encoding="utf-8"))
    corpus = {row["passage_id"]: row for row in load_jsonl(prepared / "corpus-passages.jsonl")}
    targets = {tuple(row["address_key"]): row for row in load_jsonl(prepared / "target-addresses.jsonl")}
    bindings = {(row["law_id"], row["requested_asof"]): row
                for row in load_jsonl(source / "asof-bindings.jsonl")}
    selected = load_jsonl(prepared / "selected-query-pairs.jsonl")
    queries = load_jsonl(prepared / "queries-exact.jsonl")
    issues, seen_ids = [], set()
    by_split_stratum, groups, texts, quote_presence = Counter(), Counter(), {}, Counter()
    for pair in selected:
        groups[(pair["family_id"], pair["target_asof"], pair["stratum"])] += 1
    if any(count > report["protocol"]["cap_per_family_target_stratum"] for count in groups.values()):
        issues.append("selected address cap exceeded")
    for query in queries:
        query_id = query["query_id"]
        if query_id in seen_ids:
            issues.append(f"duplicate query ID: {query_id}")
        seen_ids.add(query_id)
        origin = corpus[query["source_passage_id"]]
        source_key, target_key = tuple(query["source_key"]), tuple(query["target_key"])
        source_address = (origin["law_id"], query["source_asof"], origin["scope"], origin["article_num"])
        if source_key != source_address or query["source_asof"] not in origin["asof_dates"]:
            issues.append(f"query origin/address disagreement: {query_id}")
        if (query["source_role"] != "full" or origin["source_role"] != "full"
                or query["split"] != allocation[query["family_id"]]
                or origin["split"] != query["split"] or origin["family_id"] != query["family_id"]):
            issues.append(f"query family allocation disagreement: {query_id}")
        span = query["source_paragraph_text"][query["source_span_start"]:query["source_span_end"]]
        matching = [paragraph for paragraph in origin["visible_paragraphs"]
                    if paragraph["tree_path"] == query["source_tree_path"]]
        if (span != query["query_text"] or span != query["source_quote_text"]
                or len(matching) != 1 or matching[0]["text"] != query["source_paragraph_text"]):
            issues.append(f"literal query span disagreement: {query_id}")
        if query["source_asof"] > query["target_asof"]:
            issues.append(f"future-origin primary query: {query_id}")
        target = targets.get(target_key)
        status = target["status"] if target else "absent"
        binding = bindings[target_key[:2]]
        if (query["target_status"] != status or query["labels"]["target_status"] != status
                or tuple(query["labels"]["target_key"]) != target_key
                or query["target_revision_id"] != binding["revision_id"]
                or query["labels"]["revision_id"] != binding["revision_id"]):
            issues.append(f"dated target label disagreement: {query_id}")
        if status == "available":
            if len(target["passage_ids"]) != 1 or query["target_passage_id"] != target["passage_ids"][0]:
                issues.append(f"available target passage disagreement: {query_id}")
            destination = corpus[query["target_passage_id"]]
            present = normal(query["source_quote_text"]) in normal(destination["visible_text"])
            if query.get("quote_present_in_target") is not present or query.get("source_quote_absent_from_target") is not (not present):
                issues.append(f"quote-presence annotation disagreement: {query_id}")
            if query["stratum"] == "changed_cross_date":
                quote_presence[(query["split"], "quote_present" if present else "quote_absent")] += 1
            if (query["stratum"] == "changed_cross_date"
                    and (query["source_asof"] >= query["target_asof"]
                         or normal(origin["visible_text"]) == normal(destination["visible_text"]))):
                issues.append(f"changed-pair wording/date disagreement: {query_id}")
        else:
            if query["labels"]["passage_id"] is not None:
                issues.append(f"unavailable target has a positive passage label: {query_id}")
            if query.get("quote_present_in_target") is not None or query.get("source_quote_absent_from_target") is not None:
                issues.append(f"unavailable target has a quote-presence claim: {query_id}")
        if query["stratum"] == "same_date" and (query["source_asof"] != query["target_asof"] or status != "available"):
            issues.append(f"same-date query disagreement: {query_id}")
        if query["stratum"] == "unavailable_target" and status not in {"absent", "deleted", "hidden", "no_substantive_text"}:
            issues.append(f"unavailable-target query disagreement: {query_id}")
        by_split_stratum[(query["split"], query["stratum"])] += 1
        texts.setdefault(query["query_text"], set()).add(query["family_id"])
    original = json.loads((source / "lawqa/selection.json").read_text(encoding="utf-8"))["samples"]
    external = load_jsonl(prepared / "lawqa-external.jsonl")
    if external != original or len(external) != 140:
        issues.append("external LawQA rows/fields disagree with released originals")
    guidance = load_jsonl(prepared / "guidance-documents.jsonl")
    if len(guidance) != 11 or any(row["split"] != "external" or not row["text"].strip() for row in guidance):
        issues.append("external guidance is incomplete")
    reserved = set(json.loads((source / "source-exclusions.json").read_text(encoding="utf-8"))["law_ids"])
    if any(row["law_id"] in reserved and row["split"] != "external" for row in corpus.values()):
        issues.append("reserved external law occurs in a supervised split")
    if not report["complete"] or report["statistics"]["exact_queries"] != len(queries):
        issues.append("preparation/query count disagreement")
    baseline = prepared / "queries-exact-before-quote-annotations.jsonl"
    baseline_unchanged = None
    if baseline.exists():
        old_queries = load_jsonl(baseline)
        old_fields = [{key: value for key, value in query.items()
                       if key not in {"quote_present_in_target", "source_quote_absent_from_target"}} for query in queries]
        baseline_unchanged = old_queries == old_fields
        if not baseline_unchanged:
            issues.append("original query IDs/text/labels/allocation changed during annotation")
    return {"status": "PASS" if not issues else "FAIL", "issues": issues,
            "families_by_split": dict(Counter(allocation.values())), "exact_queries": len(queries),
            "queries_by_split_stratum": [{"split": key[0], "stratum": key[1], "count": value}
                                         for key, value in sorted(by_split_stratum.items())],
            "unique_query_texts": len(texts), "query_texts_shared_across_families": sum(len(v) > 1 for v in texts.values()),
            "changed_quote_presence_by_split": [{"split": key[0], "condition": key[1], "count": value}
                                                for key, value in sorted(quote_presence.items())],
            "original_query_fields_unchanged": baseline_unchanged,
            "lawqa_questions": len(external), "guidance_documents": len(guidance),
            "training_runs": 0, "model_inference_runs": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/hidate/full"))
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/processed/hidate"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = readback(args.source_dir, args.prepared_dir)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
