"""Acquire, prepare, encode, retrieve, read, and score the frozen FiscalQA R3 external test."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

from hidate import fiscal as fiscal
from hidate.io import write_json
from hidate.ranking import address_probabilities


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        yield from (json.loads(line) for line in stream if line.strip())


def save_rows(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def receipt(**values):
    return {"job_id": os.environ.get("PBS_JOBID"), "at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": 42, "training_runs": 0, "french_fit_runs": 0, "french_dev_updates": 0, **values}


def acquire(args, protocol):
    import requests
    commit = protocol["source_commit"]
    source = protocol["source_repository"]
    raw = Path(args.raw)
    sizes = {}
    for relative in protocol["release_files"]:
        output = raw / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        url = source.replace("https://github.com/", "https://raw.githubusercontent.com/") + f"/{commit}/{relative}"
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        output.write_bytes(response.content)
        sizes[relative] = output.stat().st_size
    write_json(raw.parent / "acquisition.json", receipt(complete=True, source_repository=source,
               source_commit=commit, files_bytes=sizes, data_license="CC-BY-4.0", code_license="MIT"))
    print(json.dumps(sizes), flush=True)


def prepare(args, protocol):
    import pyarrow.parquet as pq
    raw, output = Path(args.raw), Path(args.prepared)
    output.mkdir(parents=True, exist_ok=True)
    source = pq.read_table(raw / "data/articles_cgi_versioned.parquet")
    articles = [{key: fiscal.date_text(row[key]) if key in {"date_debut", "date_fin"} else row[key]
                 for key in ("id", "cid", "num", "etat", "texte_clean", "date_debut", "date_fin", "code_id")}
                for row in source.to_pylist() if row["code_id"] == protocol["retrieval_code_id"]]
    questions = [q for q in load(raw / "data/benchmark/questions.json") if q.get("regime") == "R3"]
    excluded = {line.strip() for line in (raw / "data/benchmark/excluded_qids_scope.txt").read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")}
    if {q["qid"] for q in questions if q["scope"] != "in"} != excluded:
        raise ValueError("Released scope labels differ from the official exclusion list")
    counts = Counter(q["scope"] for q in questions)
    if counts != {"in": 209, "excluded_review": 8, "excluded_arrete": 4}:
        raise ValueError(f"This adapter requires the frozen 221-question R3 release: {counts}")
    version_index = {row["id"]: row for row in articles}
    missing = [q["qid"] for q in questions if q["gold"]["version_id"] not in version_index]
    if missing:
        raise ValueError(f"R3 gold versions absent from the released CGI corpus: {missing}")
    binding_mismatch = []
    for question in questions:
        bound = fiscal.select_versions(articles, question["date_anchor"]).get(question["gold"]["article_cid"])
        if bound is None or bound["id"] != question["gold"]["version_id"]:
            binding_mismatch.append({"qid": question["qid"], "gold_version_id": question["gold"]["version_id"],
                                     "bound_version_id": bound["id"] if bound else None, "scope": question["scope"]})
    chunks = fiscal.article_chunks(articles, size=protocol["chunk_characters"], overlap=protocol["chunk_overlap_characters"])
    save_rows(output / "articles.jsonl", articles)
    save_rows(output / "chunks.jsonl", chunks)
    save_rows(output / "questions.jsonl", questions)
    question_ids = {q["qid"] for q in questions}
    save_rows(output / "nuggets.jsonl", [n for n in load(raw / "data/benchmark/nuggets.json") if n["qid"] in question_ids])
    write_json(output / "preparation.json", receipt(complete=True, protocol=protocol,
        source_article_versions=source.num_rows, retrieval_article_versions=len(articles), chunks=len(chunks),
        r3_questions=len(questions), scope_counts=dict(counts), scored_articles=len({q["gold"]["article_cid"] for q in questions if q["scope"] == "in"}),
        released_articles=len({q["gold"]["article_cid"] for q in questions}),
        gold_binding_disagreements=binding_mismatch, training_role="evaluation_only"))
    print(json.dumps({"versions": len(articles), "chunks": len(chunks), "scope": dict(counts),
                      "gold_binding_disagreements": len(binding_mismatch)}), flush=True)


def effective_pin(stack, role, protocol):
    pin = deepcopy(stack["models"][role])
    pin.setdefault("conventions", {})["instruction"] = protocol["retrieval_instruction"]
    return pin


def encode(args, protocol):
    import torch
    from hidate.models import create_encoder
    torch.manual_seed(42)
    prepared, output = Path(args.prepared), Path(args.encoded)
    output.mkdir(parents=True, exist_ok=True)
    stack = load(args.stack)
    if not stack["complete"]:
        raise ValueError("The multilingual stack acquisition must be complete")
    pin = effective_pin(stack, "embedding", protocol)
    encoder = create_encoder(pin, device="cuda", max_length=protocol["model_max_length"])
    documents = list(rows(prepared / "chunks.jsonl"))
    questions = list(rows(prepared / "questions.jsonl"))
    work = Counter()
    for name, records, kind, key in (("documents", documents, "document", "text"),
                                      ("queries", questions, "query", "prompt")):
        vectors = None
        with (output / f"{name}-receipts.jsonl").open("w", encoding="utf-8") as stream:
            for offset in range(0, len(records), args.batch_size):
                batch = records[offset:offset + args.batch_size]
                result = encoder.encode([record[key] for record in batch], kind=kind)
                batch_vectors = np.asarray(result.pop("embeddings"), dtype=np.float32)
                if vectors is None:
                    vectors = np.lib.format.open_memmap(output / f"{name}.npy", mode="w+", dtype=np.float32,
                                                        shape=(len(records), batch_vectors.shape[1]))
                vectors[offset:offset + len(batch)] = batch_vectors
                stream.write(json.dumps({"offset": offset, "rows": len(batch), **result}) + "\n")
                work.update(batches=1, input_tokens=result["total_tokens"])
                work["model_seconds"] += result["seconds"]
                if offset % (args.batch_size * 100) == 0:
                    print(json.dumps({"phase": name, "encoded": offset + len(batch), "expected": len(records)}), flush=True)
            vectors.flush()
    write_json(output / "encoding.json", receipt(complete=True, documents=len(documents), queries=len(questions),
        embedding_pin=pin, protocol_path=str(args.protocol), work=dict(work)))


def rerank_candidates(candidates, query, version_index, reranker, protocol, alpha, cache, work, stream):
    """Rerank at most ten distinct versions; share exact pairs across conditions."""
    pending = {}
    for condition in candidates.values():
        for candidate in condition:
            # Preserve the HiDate pair contract: rerank the discovered/bound best
            # chunk, then hydrate its parent version for the fixed reader.
            document = candidate["text"]
            candidate["reranker_text"] = document
            key = (query, document)
            if key not in cache:
                pending[key] = None
            work["logical_pairs"] += 1
    keys = list(pending)
    for offset in range(0, len(keys), 8):
        pairs = keys[offset:offset + 8]
        result = reranker.score(pairs)
        for pair, score, logit, tokens in zip(pairs, result["scores"], result["logits"], result["token_counts"]):
            cache[pair] = {"reranker_score": score, "reranker_logit": logit, "pair_tokens": tokens}
        stream.write(json.dumps({"pairs": pairs, "result": result}, ensure_ascii=False) + "\n")
        work.update(physical_pairs=len(pairs), model_batches=1, input_tokens=result["total_tokens"])
        work["model_seconds"] += result["seconds"]
    output = {}
    for name, condition in candidates.items():
        ranked = []
        for i, candidate in enumerate(condition):
            score = cache[(query, candidate["reranker_text"])]
            mixed = alpha * candidate["learned_probability"] + (1 - alpha) * score["reranker_score"] if name == "history_learned" else score["reranker_score"]
            ranked.append({"article_cid": candidate["article_cid"], "version_id": candidate["version_id"],
                           "chunk_id": candidate["chunk_id"], "selection_rank": i + 1,
                           "learned_probability": candidate.get("learned_probability"), **score, "final_score": mixed})
        output[name] = {"candidates": sorted(ranked, key=lambda c: (-c["final_score"], c["selection_rank"], c["chunk_id"])),
                        "requested_pairs": len(ranked)}
    return output


def retrieve(args, protocol):
    import torch
    from hidate.models import create_reranker
    torch.manual_seed(42)
    prepared, encoded, output = Path(args.prepared), Path(args.encoded), Path(args.retrieved)
    output.mkdir(parents=True, exist_ok=True)
    ranker, policy, stack = load(args.ranker), load(args.policy), load(args.stack)
    if not all(record["complete"] for record in (ranker, policy, stack, load(encoded / "encoding.json"))):
        raise ValueError("Reuse the completed Japanese TRAIN ranker, VALID policy, stack, and encoding")
    articles, chunks = list(rows(prepared / "articles.jsonl")), list(rows(prepared / "chunks.jsonl"))
    questions = list(rows(prepared / "questions.jsonl"))
    version_index = {row["id"]: row for row in articles}
    engine = fiscal.FiscalDiscovery(articles, chunks, np.load(encoded / "documents.npy", mmap_mode="r"),
        top_k=protocol["discovery_top_k"], rrf_k=protocol["rrf_k"], static_date=protocol["static_snapshot_date"])
    reranker_pin = effective_pin(stack, "reranker", protocol)
    reranker = create_reranker(reranker_pin, device="cuda", max_length=protocol["model_max_length"])
    queries = np.load(encoded / "queries.npy", mmap_mode="r")
    work, cache = Counter(), {}
    expected = len(questions) * len(fiscal.CONDITIONS)
    with (output / "retrieval.jsonl").open("w", encoding="utf-8") as stream, \
            (output / "reranker-receipts.jsonl").open("w", encoding="utf-8") as receipts:
        for index, question in enumerate(questions):
            discovery = engine.discover(question["prompt"], question["date_anchor"], queries[index])
            candidates = fiscal.select_conditions(discovery, engine.chunks, articles, ranker=ranker, pair_budget=protocol["reranker_top_k"])
            ranked = rerank_candidates(candidates, question["prompt"], version_index, reranker, protocol,
                                       policy["alpha"], cache, work, receipts)
            probabilities = address_probabilities(ranker, [a["features"] for a in discovery["addresses"]])
            unavailable = max((p for p, a in zip(probabilities, discovery["addresses"]) if a["bound_version_id"] is None), default=-1.)
            for name, record in ranked.items():
                evidence = record["candidates"][:protocol["reader_max_passages"]]
                confidence = (evidence[0]["learned_probability"] if name == "history_learned" else evidence[0]["reranker_score"]) if evidence else 0.
                if name == "history_learned" and unavailable >= confidence:
                    confidence = 0.
                if name in {"history_bound", "history_current"}:
                    ordered = discovery["addresses"] if name == "history_bound" else sorted(discovery["addresses"],
                        key=lambda a: (-a["features"]["current_best_cosine"], -a["features"]["max_lexical_overlap_current"],
                                       -a["features"]["max_history_rrf"], a["article_cid"]))
                    if ordered and ordered[0]["bound_version_id"] is None:
                        confidence = 0.
                threshold = policy["method_thresholds"].get(name, {}).get("threshold")
                record.update(evidence=evidence, support_confidence=float(confidence), support_threshold=threshold,
                              accepted=bool(confidence >= threshold) if threshold is not None else True)
            ranked["closed_book"] = {"evidence": [], "accepted": True}
            # This is the sole retrieval condition that reads gold identifiers.
            oracle = fiscal.select_versions(articles, question["date_anchor"]).get(question["gold"]["article_cid"])
            ranked["oracle_asof"] = {"evidence": [{"article_cid": oracle["cid"], "version_id": oracle["id"]}] if oracle else [],
                                      "accepted": True, "oracle_uses_gold_article_cid": True}
            record = {"qid": question["qid"], "date_anchor": question["date_anchor"], "scope": question["scope"],
                      "bank_version_counts": discovery["bank_version_counts"], "addresses": discovery["addresses"], "conditions": ranked}
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            work["questions"] += 1
            if index % 10 == 0:
                stream.flush()
                print(json.dumps({"retrieved": index + 1, "expected": len(questions), "physical_pairs": work["physical_pairs"]}), flush=True)
    write_json(output / "retrieval.json", receipt(complete=work["questions"] == len(questions), questions=len(questions),
        expected_reader_calls=expected, conditions=list(fiscal.CONDITIONS), work=dict(work),
        ranker_path=str(args.ranker), policy_path=str(args.policy), alpha=policy["alpha"],
        embedding_pin=load(encoded / "encoding.json")["embedding_pin"], reranker_pin=reranker_pin,
        static_snapshot_date=protocol["static_snapshot_date"],
        feature_transfer="Same ten names; effective-start gap replaces captured-anchor gap; French no fit or tuning"))


def read(args, protocol):
    import torch
    from hidate.models import QwenReader
    torch.manual_seed(42)
    prepared, retrieved, output = Path(args.prepared), Path(args.retrieved), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    retrieval = load(retrieved / "retrieval.json")
    if not retrieval["complete"]:
        raise ValueError("Complete all external retrieval before loading the fixed reader")
    stack = load(args.stack)
    pin = stack["models"]["reader"]
    reader = QwenReader(pin["local_path"], device="cuda", prompt_budget=protocol["reader_prompt_tokens"])
    articles = {r["id"]: r for r in rows(prepared / "articles.jsonl")}
    questions = {q["qid"]: q for q in rows(prepared / "questions.jsonl")}
    # Resume finished greedy responses; interruption does not start a new run/seed.
    response_path = output / "responses.jsonl"
    completed = {(r["qid"], r["condition"]) for r in rows(response_path)} if response_path.exists() else set()
    work = Counter()
    with response_path.open("a", encoding="utf-8") as responses, (output / "reader-receipts.jsonl").open("a", encoding="utf-8") as receipts:
        for record in rows(retrieved / "retrieval.jsonl"):
            question = questions[record["qid"]]
            for condition in fiscal.CONDITIONS:
                if (question["qid"], condition) in completed:
                    continue
                selected = record["conditions"][condition]
                packed = fiscal.pack_reader(question["prompt"], selected["evidence"], articles, reader.tokenizer,
                    prompt_budget=protocol["reader_prompt_tokens"], max_passages=protocol["reader_max_passages"])
                result = reader.generate_paraphrase(packed["messages"], max_new_tokens=protocol["reader_max_new_tokens"])
                identity = {"qid": question["qid"], "condition": condition, "model": pin["model_id"]}
                actual = packed["evidence"]
                response = {**identity, "response_text": result["text"], "gold_version_id": question["gold"]["version_id"],
                    "retrieved_version_id": actual[0]["version_id"] if actual else None,
                    "evidence": actual, "accepted": selected["accepted"],
                    "support_confidence": selected.get("support_confidence"), "support_threshold": selected.get("support_threshold"),
                    "prompt_tokens": result["prompt_tokens"], "generated_tokens": result["generated_tokens"],
                    "end_status": result["end_status"]}
                responses.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
                receipts.write(json.dumps({**identity, "messages": packed["messages"], "result": result}, ensure_ascii=False) + "\n")
                responses.flush()
                receipts.flush()
                completed.add((question["qid"], condition))
                work.update(calls=1, input_tokens=result["prompt_tokens"], generated_tokens=result["generated_tokens"])
                work["seconds"] += result["seconds"]
                if len(completed) % 25 == 0:
                    print(json.dumps({"reader_calls_complete": len(completed), "expected": retrieval["expected_reader_calls"]}), flush=True)
    complete = len(completed) == retrieval["expected_reader_calls"]
    write_json(output / "reader.json", receipt(complete=complete, completed_calls=len(completed),
        expected_calls=retrieval["expected_reader_calls"], reader_pin=pin, work_this_job=dict(work),
        prompt_budget=protocol["reader_prompt_tokens"], decode={"do_sample": False, "max_new_tokens": protocol["reader_max_new_tokens"]}))
    score(args, protocol)


def score(args, protocol):
    prepared, output = Path(args.prepared), Path(args.output)
    official_path = Path(args.raw) / "scripts/score_nuggets.py"
    spec = importlib.util.spec_from_file_location("fiscalqa_official_score_nuggets", official_path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    questions = {q["qid"]: q for q in rows(prepared / "questions.jsonl")}
    nuggets = defaultdict(list)
    for nugget in rows(prepared / "nuggets.jsonl"):
        nuggets[nugget["qid"]].append(nugget)
    responses = list(rows(output / "responses.jsonl"))
    scores = [{"qid": r["qid"], "condition": r["condition"], "accepted": r["accepted"],
               **fiscal.score_answer(questions[r["qid"]], r["response_text"], nuggets[r["qid"]], r["evidence"], scorer)} for r in responses]
    save_rows(output / "scores.jsonl", scores)
    summary = fiscal.summarize(scores, list(questions.values()))
    expected = {(qid, condition) for qid in questions for condition in fiscal.CONDITIONS}
    actual = {(r["qid"], r["condition"]) for r in responses}
    complete = actual == expected and len(responses) == len(expected)
    write_json(output / "responses.json", responses)
    write_json(output / "evaluation.json", receipt(complete=complete, responses=len(responses), expected_responses=len(expected),
        conditions=list(fiscal.CONDITIONS), strata=summary, official_scorer_path=str(official_path),
        retrieval_metadata=load(Path(args.retrieved) / "retrieval.json"),
        interpretation="FiscalQA R3 external zero-shot; open-source retrieval adaptation, not proprietary C-prod reproduction"))
    print(json.dumps({"complete": complete, "responses": len(responses), "expected_responses": len(expected)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("acquire", "prepare", "encode", "retrieve", "read", "score"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/fiscal_protocol.json"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fiscal/official"))
    parser.add_argument("--prepared", type=Path, default=Path("data/processed/fiscal"))
    parser.add_argument("--encoded", type=Path, default=Path("data/index/fiscal"))
    parser.add_argument("--retrieved", type=Path, default=Path("runs/hidate/fiscal-transfer/fiscal-retrieval"))
    parser.add_argument("--output", type=Path, default=Path("runs/hidate/fiscal-transfer/fiscal"))
    parser.add_argument("--stack", type=Path, default=Path("models/qwen3/model-stack.json"))
    parser.add_argument("--ranker", type=Path, default=Path("runs/hidate/extended-benchmark/qwen3/ranker/address-model.json"))
    parser.add_argument("--policy", type=Path, default=Path("runs/hidate/extended-benchmark/qwen3/policy/frozen.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    protocol = load(args.protocol)
    globals()[args.action](args, protocol)


if __name__ == "__main__":
    main()
