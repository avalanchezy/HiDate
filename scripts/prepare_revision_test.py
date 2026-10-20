"""PBS-only source-derived confirmatory queries from the frozen 100-family manifest."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from hidate.data import address_index, address_pairs, exact_quote_queries, select_query_pairs


PREFIX = "HOLDOUT"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path, values):
    with Path(path).open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def expand_passage(passage, bindings):
    """Restore dated rows from the scale worker's unchanged visible passage record."""
    base = {k: v for k, v in passage.items() if k not in {"asof_dates", "raw_paths"}}
    for day in passage["asof_dates"]:
        binding = bindings.get((passage["law_id"], day))
        if binding is None:
            continue
        if passage["revision_id"] != binding["revision_id"]:
            raise ValueError("Prepared passage revision differs from the frozen as-of binding")
        yield {**base, "requested_asof": day, "raw_path": binding["raw_path"],
               "source_role": "full", "split": "test", "evaluation_role": "confirmatory_test"}


def address_status(records):
    if len(records) != 1:
        return "ambiguous"
    record = records[0]
    if record["deleted"]:
        return "deleted"
    if record["hidden"]:
        return "hidden"
    return "available" if record["visible_text"].strip() else "no_substantive_text"


def prepare(args):
    manifest, protocol, scale = read(args.family_manifest), read(args.protocol), read(args.scale_index / "index.json")
    families = manifest["roles"]["confirmatory_test"]
    law_map = {r["law_id"]: r for r in families}
    if not scale["complete"] or len(families) != 100 or len(law_map) != 100:
        raise ValueError("Use the completed scale index and the frozen 100-family role")
    if protocol["seed"] != 42 or protocol["cap_per_family_target_stratum"] != 25:
        raise ValueError("Reuse the original seed42/cap25 source-derived protocol")
    allocation = {r["family_id"]: "test" for r in families}
    expert_ids = {r["family_id"] for r in manifest["roles"]["expert_scenarios"]}
    if set(allocation) & expert_ids:
        raise ValueError("Confirmatory and human-scenario families must remain separate")
    frozen_bindings = {(r["law_id"], b["requested_asof"]): {**b, "law_id": r["law_id"],
                        "family_id": r["family_id"], "source_role": "full"}
                       for r in families for b in r["anchor_bindings"]}
    output, sealed = args.output, args.output / "sealed"
    sealed.mkdir(parents=True, exist_ok=True)
    receipt = {"complete": False, "job_id": os.environ.get("PBS_JOBID"), "seed": 42,
               "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "family_manifest": str(args.family_manifest), "source_prepared_index": str(args.scale_index),
               "role": "confirmatory_test", "training_runs": 0, "model_inference_runs": 0,
               "evaluation_runs": 0, "family_split_called": False, "protocol": protocol}
    write(output / "preparation.json", receipt)
    dated_count, unique_passages, seen_laws = 0, 0, set()

    def dated_rows():
        nonlocal dated_count, unique_passages
        for passage in rows(args.scale_index / "extra-corpus-passages.jsonl"):
            if passage["law_id"] not in law_map or passage["scope"] != "MainProvision":
                continue
            if passage["family_id"] != law_map[passage["law_id"]]["family_id"]:
                raise ValueError("Prepared passage family differs from the frozen family")
            unique_passages += 1
            seen_laws.add(passage["law_id"])
            for record in expand_passage(passage, frozen_bindings):
                dated_count += 1
                yield record

    index = address_index(dated_rows(), list(frozen_bindings.values()))
    target_rows = {}
    for key, values in index["addresses"].items():
        law_id, day, scope, number = key
        binding, first = frozen_bindings[law_id, day], values[0]
        target_rows[key] = {"address_key": list(key), "law_id": law_id, "requested_asof": day,
            "scope": scope, "address_num": number, "article_num": first["article_num"],
            "slot_id": first["slot_id"], "status": address_status(values), "revision_id": binding["revision_id"],
            "passage_ids": [r["passage_id"] for r in values], "raw_path": binding["raw_path"],
            "family_id": first["family_id"], "source_role": "full", "split": "test",
            "evaluation_role": "confirmatory_test"}
    matched_targets = 0
    for previous in rows(args.scale_index / "target-addresses.jsonl"):
        if previous["law_id"] not in law_map or previous["scope"] != "MainProvision":
            continue
        key = tuple(previous["address_key"])
        current = target_rows[key]
        for field in ("status", "revision_id", "passage_ids", "raw_path", "family_id", "slot_id"):
            if current[field] != previous[field]:
                raise ValueError("Prepared scale target differs at " + field)
        matched_targets += 1
    if matched_targets != len(target_rows):
        raise ValueError("Not every confirmatory target has the original scale target record")
    pairs = address_pairs(index, protocol["anchor_dates"], allocation)
    pairs = [dict(pair, scope=pair["source_key"][2], article_num=pair["source_key"][3],
                  evaluation_role="confirmatory_test") for pair in pairs]
    selected, selection_counts = select_query_pairs(pairs,
        cap_per_family_target_stratum=25, seed=42)
    queries = exact_quote_queries(selected, index, quote_chars=protocol["quote_chars"], min_chars=protocol["min_chars"])
    for number, query in enumerate(queries, 1):
        query["source_derived_query_id"] = query["query_id"]
        query["query_id"] = f"{PREFIX}-Q{number:06d}-exact"
        query["evaluation_role"] = "confirmatory_test"
    text_counts = Counter(q["query_text"] for q in queries)
    for query in queries:
        query["query_text_occurrences"] = text_counts[query["query_text"]]
    source_ids = {q["source_passage_id"] for q in queries}
    target_ids = {q["target_passage_id"] for q in queries if q["target_status"] == "available"}
    wanted = source_ids | target_ids
    found, chunks_for_laws, chunk_rows = set(), 0, 0
    for chunk in rows(args.scale_index / "chunks.jsonl"):
        chunk_rows += 1
        # The original bank also contains guidance chunks without a law_id.
        if chunk.get("law_id") in law_map:
            chunks_for_laws += 1
            if chunk["parent_passage_id"] in wanted:
                found.add(chunk["parent_passage_id"])
    if found != wanted or chunk_rows != scale["merged_chunk_rows"]:
        raise ValueError("Confirmatory passage references must resolve in the unchanged complete scale chunks")
    write_rows(sealed / "target-addresses.jsonl", target_rows.values())
    write_rows(sealed / "address-pair-census.jsonl", pairs)
    write_rows(sealed / "selected-query-pairs.jsonl", selected)
    write_rows(sealed / "queries-exact.jsonl", queries)
    write_rows(output / "queries-inputs.jsonl", ({k: q[k] for k in
        ("query_id", "query_text", "query_variant", "target_asof", "split", "evaluation_role")} for q in queries))
    write(output / "family-allocation.json", allocation)
    write(output / "selection-counts.json", selection_counts)
    family_query_counts = Counter(q["family_id"] for q in queries)
    stats = {"frozen_families": len(families), "families_with_prepared_main_text": len(seen_laws),
             "families_with_queries": len(family_query_counts),
             "frozen_families_without_queries": sorted(set(allocation) - set(family_query_counts)),
             "prepared_unique_main_passages": unique_passages, "dated_main_provisions": dated_count,
             "frozen_anchor_bindings": len(frozen_bindings), "target_addresses": len(target_rows),
             "scale_targets_matched_exactly": matched_targets,
             "address_pairs": len(pairs), "pairs_by_stratum": dict(Counter(r["stratum"] for r in pairs)),
             "selected_pairs": len(selected), "selected_by_stratum": dict(Counter(r["stratum"] for r in selected)),
             "exact_queries": len(queries), "query_variants_generated": ["exact_quote"],
             "queries_by_stratum": dict(Counter(q["stratum"] for q in queries)),
             "queries_by_target_status": dict(Counter(q["target_status"] for q in queries)),
             "queries_by_target_date": dict(Counter(q["target_asof"] for q in queries)),
             "quote_absent_queries": sum(q["source_quote_absent_from_target"] is True for q in queries),
             "quote_absent_families": len({q["family_id"] for q in queries if q["source_quote_absent_from_target"] is True}),
             "repeated_query_texts": sum(v > 1 for v in text_counts.values()),
             "source_passages_linked": len(source_ids), "available_target_passages_linked": len(target_ids),
             "missing_query_passage_links": 0, "unchanged_scale_chunk_rows": chunk_rows,
             "scale_chunks_from_confirmatory_laws": chunks_for_laws,
             "cap_per_family_target_stratum": 25, "seed": 42}
    write(output / "statistics.json", stats)
    linkage = {"scale_index": str(args.scale_index), "scale_index_job": scale["job_id"],
               "scale_index_complete": True, "scale_chunks": str(args.scale_index / "chunks.jsonl"),
               "scale_embeddings": "Existing scale embedding bank, unchanged; this preparation encodes no text.",
               "existing_scale_chunk_rows": chunk_rows, "new_chunk_rows": 0,
               "new_vectors": 0, "chunk_boundaries_modified": False,
               "passage_ids_reused_without_rewriting": True,
               "visible_text_source": "Completed scale extra-corpus-passages.jsonl, already produced by extract_provisions + visible_provision.",
               "target_address_status_and_identity_match": matched_targets,
               "source_and_available_gold_passages_in_existing_chunks": len(wanted)}
    write(output / "corpus-linkage.json", linkage)
    write(output / "pipeline-paths.json", {"family_manifest": str(args.family_manifest),
        "raw_source_root": manifest["sources_root"], "prepared_visible_passages": str(args.scale_index / "extra-corpus-passages.jsonl"),
        "corpus_chunks": str(args.scale_index / "chunks.jsonl"), "query_inputs": str(output / "queries-inputs.jsonl"),
        "sealed_query_records": str(sealed / "queries-exact.jsonl"),
        "sealed_target_addresses": str(sealed / "target-addresses.jsonl"), "statistics": str(output / "statistics.json")})
    write(sealed / "README.json", {"role": "confirmatory_test_labels_and_provenance",
          "access_plan": "Evaluation code joins these records after the TRAIN/VALID recipe is frozen; manuscript preparation uses aggregate statistics.",
          "automated_source_derived": True, "human_authored": False,
          "original_train_valid_test_modified": False})
    receipt.update(complete=True, finished_at_utc=datetime.now(timezone.utc).isoformat(),
                   statistics=stats, corpus_linkage=linkage, labels_location=str(sealed),
                   paraphrases_generated=0, human_questions_generated=0)
    write(output / "preparation.json", receipt)
    print(json.dumps(stats, ensure_ascii=True, indent=2), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-manifest", type=Path, default=Path("data/processed/hidate-holdout/family-manifest.json"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/data-protocol.json"))
    parser.add_argument("--scale-index", type=Path, default=Path("data/index/hidate/extended-benchmark/scale"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/hidate-holdout/confirmatory"))
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
