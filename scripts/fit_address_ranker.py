"""Fit the predeclared HiDate address ranker once from complete TRAIN discovery."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from hidate.ranking import fit_address_ranker, training_examples


def records(paths):
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", nargs="+", type=Path, required=True)
    parser.add_argument("--discovery", nargs="+", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/retrieval-protocol.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/hidate/ranker/address-model.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("The one TRAIN model already exists; reuse its saved parameters")
    started = time.perf_counter()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    queries = {row["query_id"]: row for row in records(args.queries) if row["split"] == "train"}
    discoveries = [row for row in records(args.discovery) if row["query_id"] in queries]
    seen = [row["query_id"] for row in discoveries]
    if len(seen) != len(set(seen)) or set(seen) != set(queries):
        raise ValueError("Each declared TRAIN query must have exactly one complete discovery record")
    X, y, statistics = training_examples(discoveries, queries, protocol["rank_features"],
        required_variants=protocol["one_train_fit"]["training_variants"])
    model = fit_address_ranker(X, y, protocol["rank_features"])
    model.update(complete=True, job_id=os.environ.get("PBS_JOBID"),
        fitted_at_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - started,
        training_statistics=statistics, protocol_path=str(args.protocol),
        discovery_paths=[str(path) for path in args.discovery])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(statistics), flush=True)


if __name__ == "__main__":
    main()
