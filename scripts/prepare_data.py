"""Prepare frozen HiDate addresses and exact quotes from completed source collection."""

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from hidate.io import load_jsonl, write_json, write_jsonl
from hidate.data import (address_index, address_pairs, exact_quote_queries, family_split,
                           select_query_pairs, visible_provision)


def target_status(records):
    if len(records) != 1:
        return "ambiguous"
    record = records[0]
    if record["deleted"]:
        return "deleted"
    if record["hidden"]:
        return "hidden"
    if not record["visible_text"].strip():
        return "no_substantive_text"
    return "available"


def prepare_data(source_dir, output_dir, protocol_path):
    source, output = Path(source_dir), Path(output_dir)
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if not source_manifest["complete"]:
        raise ValueError("HiDate source collection must be complete before preparation")
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    selected_laws = json.loads((source / "selected-laws.json").read_text(encoding="utf-8"))
    allocation = family_split(selected_laws, seed=protocol["seed"])
    output.mkdir(parents=True, exist_ok=True)
    preparation = {"complete": False, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source.resolve()), "output_dir": str(output.resolve()),
        "source_manifest_path": str((source / "manifest.json").resolve()),
        "source_collection_complete": True, "protocol": protocol,
        "training_runs": 0, "model_inference_runs": 0}
    write_json(output / "preparation.json", preparation)
    write_json(output / "family-split.json", allocation)
    bindings = load_jsonl(source / "asof-bindings.jsonl")
    corpus = {}
    source_provisions = 0

    def augmented_provisions():
        nonlocal source_provisions
        last_path, last_snapshot = None, None
        with (source / "provisions.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["raw_path"] != last_path:
                    last_path = record["raw_path"]
                    last_snapshot = json.loads((source / last_path).read_text(encoding="utf-8"))
                augmented = visible_provision(record, last_snapshot)
                augmented["split"] = ("external" if augmented["source_role"] == "external"
                    else allocation[augmented["family_id"]])
                source_provisions += 1
                passage_id = augmented["passage_id"]
                if passage_id not in corpus:
                    passage = dict(augmented)
                    passage.pop("requested_asof")
                    passage.pop("raw_path")
                    passage.update(asof_dates=[], raw_paths=[])
                    corpus[passage_id] = passage
                passage = corpus[passage_id]
                if augmented["requested_asof"] not in passage["asof_dates"]:
                    passage["asof_dates"].append(augmented["requested_asof"])
                if augmented["raw_path"] not in passage["raw_paths"]:
                    passage["raw_paths"].append(augmented["raw_path"])
                yield augmented

    index = address_index(augmented_provisions(), bindings)
    for passage in corpus.values():
        passage["asof_dates"].sort()
        passage["raw_paths"].sort()
    write_jsonl(output / "corpus-passages.jsonl", corpus.values())
    target_counts, target_role_counts = Counter(), Counter()

    def target_rows():
        for key, records in index["addresses"].items():
            law_id, date, scope, address_num = key
            binding = index["bindings"][(law_id, date)]
            status = target_status(records)
            target_counts[status] += 1
            target_role_counts[records[0]["source_role"]] += 1
            yield {"address_key": list(key), "law_id": law_id, "requested_asof": date,
                "scope": scope, "address_num": address_num, "article_num": records[0]["article_num"],
                "slot_id": records[0]["slot_id"], "status": status, "revision_id": binding["revision_id"],
                "passage_ids": [row["passage_id"] for row in records], "raw_path": binding["raw_path"],
                "family_id": records[0]["family_id"], "source_role": records[0]["source_role"],
                "split": records[0]["split"]}

    write_jsonl(output / "target-addresses.jsonl", target_rows())
    pairs = address_pairs(index, protocol["anchor_dates"], allocation)
    pairs = [dict(pair, scope=pair["source_key"][2], article_num=pair["source_key"][3]) for pair in pairs]
    write_jsonl(output / "address-pair-census.jsonl", pairs)
    selected_pairs, selection_counts = select_query_pairs(pairs,
        cap_per_family_target_stratum=protocol["cap_per_family_target_stratum"], seed=protocol["seed"])
    write_jsonl(output / "selected-query-pairs.jsonl", selected_pairs)
    write_json(output / "selection-counts.json", selection_counts)
    queries = exact_quote_queries(selected_pairs, index,
        quote_chars=protocol["quote_chars"], min_chars=protocol["min_chars"])
    query_text_counts = Counter(query["query_text"] for query in queries)
    for query in queries:
        query["query_text_occurrences"] = query_text_counts[query["query_text"]]
    changed_queries = [query for query in queries if query["stratum"] == "changed_cross_date"]

    def changed_quote_counts(rows):
        return {"quote_present": sum(row["quote_present_in_target"] is True for row in rows),
                "quote_absent": sum(row["source_quote_absent_from_target"] is True for row in rows)}
    write_jsonl(output / "queries-exact.jsonl", queries)

    lawqa = json.loads((source / "lawqa/selection.json").read_text(encoding="utf-8"))["samples"]
    write_jsonl(output / "lawqa-external.jsonl", lawqa)
    guidance = []
    for document in json.loads((source / "guidance.json").read_text(encoding="utf-8")):
        if document["status"] != "collected":
            raise ValueError("HiDate guidance collection is not complete")
        text = (source / document["text_path"]).read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError("HiDate guidance text is empty")
        guidance.append({**document, "text": text, "source_role": "external", "split": "external"})
    write_jsonl(output / "guidance-documents.jsonl", guidance)
    statistics = {
        "selected_laws": len(selected_laws), "families": len(allocation),
        "families_by_split": dict(Counter(allocation.values())), "source_provisions": source_provisions,
        "corpus_passages": len(corpus), "corpus_by_role": dict(Counter(row["source_role"] for row in corpus.values())),
        "corpus_by_split": dict(Counter(row["split"] for row in corpus.values())),
        "target_addresses": sum(target_counts.values()), "target_statuses": dict(target_counts),
        "target_by_role": dict(target_role_counts), "address_pairs": len(pairs),
        "pairs_by_stratum": dict(Counter(row["stratum"] for row in pairs)),
        "pairs_by_split": dict(Counter(row["split"] for row in pairs)),
        "pairs_by_target_status": dict(Counter(row["target_status"] for row in pairs)),
        "selected_pairs": len(selected_pairs), "selected_by_stratum": dict(Counter(row["stratum"] for row in selected_pairs)),
        "exact_queries": len(queries), "queries_by_stratum": dict(Counter(row["stratum"] for row in queries)),
        "queries_by_split": dict(Counter(row["split"] for row in queries)),
        "queries_by_target_status": dict(Counter(row["target_status"] for row in queries)),
        "changed_quote_counts": changed_quote_counts(changed_queries),
        "changed_quote_counts_by_split": {split: changed_quote_counts(
            [query for query in changed_queries if query["split"] == split])
            for split in ("train", "valid", "test")},
        "repeated_query_texts": sum(count > 1 for count in query_text_counts.values()),
        "queries_with_repeated_text": sum(count for count in query_text_counts.values() if count > 1),
        "lawqa_questions": len(lawqa), "guidance_documents": len(guidance),
    }
    preparation.update(complete=True, finished_at_utc=datetime.now(timezone.utc).isoformat(),
        statistics=statistics, lawqa_revision=source_manifest["lawqa_revision"],
        outputs={name: name for name in ("family-split.json", "corpus-passages.jsonl", "target-addresses.jsonl",
            "address-pair-census.jsonl", "selected-query-pairs.jsonl", "queries-exact.jsonl",
            "lawqa-external.jsonl", "guidance-documents.jsonl", "selection-counts.json")})
    write_json(output / "preparation.json", preparation)
    print(json.dumps(statistics, ensure_ascii=False), flush=True)
    return preparation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/hidate/full"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/hidate"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/data-protocol.json"))
    args = parser.parse_args()
    prepare_data(args.source_dir, args.output_dir, args.protocol)


if __name__ == "__main__":
    main()
