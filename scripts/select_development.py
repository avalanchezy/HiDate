"""Freeze HiDate policy from complete VALID scores before held-out inference."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from hidate.io import load_jsonl, write_json
from hidate.selection import choose_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-dir", type=Path, default=Path("runs/hidate/dev"))
    parser.add_argument("--output", type=Path, default=Path("runs/hidate/policy/frozen.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("The saved DEV policy is already frozen; reuse it for held-out evaluation")
    scoring = json.loads((args.dev_dir / "scoring.json").read_text())
    if not scoring["complete"]:
        raise ValueError("DEV model scoring must be complete before selection")
    protocol = json.loads(Path("configs/retrieval-protocol.json").read_text())
    queries = {row["query_id"]: row for name in ["queries-exact.jsonl", "queries-paraphrase.jsonl"]
               for row in load_jsonl(Path("data/processed/hidate") / name) if row["split"] == "valid"}
    rows = load_jsonl(args.dev_dir / "reranked.jsonl")
    ids = [row["query_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(queries):
        raise ValueError("All VALID rows from both query variants must be scored exactly once")
    policy = choose_policy(rows, queries, protocol["dev_selection"]["learned_reranker_mix_grid"],
                           target_precision=protocol["dev_selection"]["support_target_precision"])
    policy.update(complete=True, job_id=os.environ.get("PBS_JOBID"),
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), scoring_path=str(args.dev_dir / "scoring.json"),
        ranker_path=scoring["ranker_path"], protocol_path="configs/retrieval-protocol.json")
    write_json(args.output, policy)
    print(json.dumps(policy), flush=True)


if __name__ == "__main__":
    main()
