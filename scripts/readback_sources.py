"""Read back complete HiDate source bindings and original tree addresses."""

import argparse
import json
from collections import Counter
from pathlib import Path

from hidate.io import load_jsonl, write_json


def readback(source):
    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    bindings = load_jsonl(source / "asof-bindings.jsonl")
    selected = json.loads((source / "selected-laws.json").read_text(encoding="utf-8"))
    exclusions = json.loads((source / "source-exclusions.json").read_text(encoding="utf-8"))
    issues = []
    selected_ids = {row["law_info"]["law_id"] for row in selected}
    if selected_ids.intersection(exclusions["law_ids"]):
        issues.append("reserved LawQA IDs occur in supervised selection")
    snapshots = {}
    future_enforcement = []
    for binding in bindings:
        raw = json.loads((source / binding["raw_path"]).read_text(encoding="utf-8"))
        key = (binding["law_id"], binding["requested_asof"])
        if key in snapshots:
            issues.append(f"repeated dated binding: {key}")
        snapshots[key] = binding["raw_path"]
        if (binding["status"] != "collected"
                or raw["law_info"]["law_id"] != binding["law_id"]
                or raw["revision_info"]["law_revision_id"] != binding["revision_id"]):
            issues.append(f"source binding disagreement: {key}")
        enforcement = raw["revision_info"].get("amendment_enforcement_date")
        if enforcement and enforcement > binding["requested_asof"]:
            future_enforcement.append({"law_id": key[0], "asof": key[1], "enforcement": enforcement})
    roles, tags, scope_types = Counter(), Counter(), Counter()
    provision_count, last_path, tree = 0, None, None
    with (source / "provisions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["raw_path"] != last_path:
                raw = json.loads((source / row["raw_path"]).read_text(encoding="utf-8"))
                tree, last_path = raw["law_full_text"], row["raw_path"]
            node = tree
            for position in row["tree_path"]:
                node = node["children"][position]
            if (row["revision_id"] != raw["revision_info"]["law_revision_id"]
                    or row["law_id"] != raw["law_info"]["law_id"]):
                issues.append(f"provision source metadata disagreement: {row['slot_id']}")
            if row["article_num"] is not None:
                if node["tag"] != "Article" or node.get("attr", {}).get("Num") != row["article_num"]:
                    issues.append(f"article number/tree-path disagreement: {row['slot_id']}")
            for paragraph in row["paragraphs"]:
                paragraph_node = tree
                for position in paragraph["tree_path"]:
                    paragraph_node = paragraph_node["children"][position]
                if (paragraph_node["tag"] != "Paragraph"
                        or paragraph_node.get("attr", {}).get("Num") != paragraph["num"]):
                    issues.append(f"paragraph number/tree-path disagreement: {row['slot_id']}")
            roles[row["source_role"]] += 1
            tags[node["tag"]] += 1
            scope_types[row["scope"].split(":", 1)[0]] += 1
            provision_count += 1
    questions = json.loads((source / "lawqa/selection.json").read_text(encoding="utf-8"))["samples"]
    filenames = [row["ファイル名"] for row in questions]
    if len(questions) != 140 or len(set(filenames)) != 140:
        issues.append("released original LawQA question identities are incomplete")
    if any(row["output"] not in ("a", "b", "c", "d") for row in questions):
        issues.append("LawQA answer label outside a-d")
    guidance = json.loads((source / "guidance.json").read_text(encoding="utf-8"))
    for document in guidance:
        if (document["status"] != "collected"
                or not (source / document["text_path"]).read_text(encoding="utf-8").strip()
                or not (source / document["pdf_path"]).read_bytes().startswith(b"%PDF-")):
            issues.append(f"guidance source unavailable: {document['source_id']}")
    if not manifest["complete"] or len(bindings) != manifest["successful_snapshots"]:
        issues.append("collection manifest/binding count is incomplete")
    if provision_count != manifest["provision_count"]:
        issues.append("actual provision rows disagree with collection count")
    if future_enforcement:
        issues.append("returned revision enforcement is after requested asof")
    return {"status": "PASS" if not issues else "FAIL", "issues": issues,
            "selected_laws": len(selected),
            "selected_by_type": dict(Counter(row["law_info"]["law_type"] for row in selected)),
            "selected_families": len({row["family_id"] for row in selected}),
            "dated_snapshots": len(bindings), "provisions": provision_count,
            "source_roles": dict(roles), "tree_node_types": dict(tags), "scope_types": dict(scope_types),
            "external_questions": len(questions), "original_question_ids": len(set(filenames)),
            "guidance_documents": len(guidance), "future_enforcement": future_enforcement,
            "training_runs": 0, "model_inference_runs": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = readback(args.source_dir)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
