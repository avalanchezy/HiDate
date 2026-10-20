"""Freeze common date-wise TEST support-removal banks without model outcomes."""

import json
import os
from pathlib import Path

from hidate.io import load_jsonl, write_json
from hidate.controls import controlled_date_bank


def main():
    prepared, output = Path("data/processed/hidate"), Path("runs/hidate/controlled-removal")
    chunks = load_jsonl(Path("data/index/hidate/chunks.jsonl"))
    parents = load_jsonl(prepared / "corpus-passages.jsonl")
    queries = load_jsonl(prepared / "queries-exact.jsonl")
    targets = {tuple(row["address_key"]): row for row in load_jsonl(prepared / "target-addresses.jsonl")}
    selected = [row for row in queries if row["split"] == "test" and row["target_status"] == "available"]
    counts = {}
    for date in sorted({row["target_asof"] for row in selected}):
        bank = controlled_date_bank(chunks, parents, targets, queries, date)
        audit = bank["audit"]
        audit["target_overrides"] = [bank["target_index"][tuple(key)] for key in audit["blocked_addresses"]]
        audit["eligible_base_query_ids"] = [row["query_id"] for row in selected if row["target_asof"] == date]
        assert all(chunk["parent_passage_id"] not in audit["removed_parent_ids"] for chunk in bank["chunks"])
        write_json(output / (date + ".json"), audit)
        counts[date] = {"eligible_queries": len(audit["eligible_base_query_ids"]),
            "blocked_addresses": len(audit["blocked_addresses"]), "removed_parents": len(audit["removed_parent_ids"]),
            "removed_chunks": len(audit["removed_chunk_ids"]), "remaining_chunks": len(bank["chunks"]),
            "exact_text_aliases": len(audit["extra_exact_text_aliases"])}
    report = {"complete": True, "job_id": os.environ.get("PBS_JOBID"), "query_variant": "exact_quote",
        "eligible_queries": len(selected), "shared_with_generated_variant_by": "base_query_id",
        "counts_by_date": counts, "original_data_changed": False, "training_runs": 0,
        "model_forward_calls": 0, "interpretation": "corpus support unavailable, not legal unanswerability"}
    write_json(output / "controls.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
