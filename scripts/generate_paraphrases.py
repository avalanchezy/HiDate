"""Generate one frozen greedy search paraphrase per exact HiDate source quote."""

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from hidate.io import load_jsonl, write_json, write_jsonl
from hidate.data import attach_paraphrases


SYSTEM_PROMPT = ("与えられた引用だけを用いて、内容を日常的な日本語の検索依頼一つに言い換えてください。"
    "条件、数値、固有名詞、否定を保ってください。引用が途中で終わっていても続きを補わず、"
    "事実を追加しないでください。検索依頼だけを出力してください。")
USER_PREFIX = "引用:\n"
MAX_NEW_TOKENS = 128


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def generate_paraphrases(queries_path, stack_path, output_path, run_dir, *, reader=None,
                        resume=False, preflight_path=None):
    started = time.perf_counter()
    queries = load_jsonl(queries_path)
    if any(query["query_variant"] != "exact_quote" for query in queries):
        raise ValueError("Paraphrases must originate from the frozen exact_quote queries")
    quotes = list(dict.fromkeys(query["source_quote_text"] for query in queries))
    stack = json.loads(Path(stack_path).read_text(encoding="utf-8"))
    pin = stack["models"]["reader"]
    if not stack["complete"] or not pin["download_complete"]:
        raise ValueError("HiDate fixed reader acquisition must be complete")
    model_pin = {key: pin[key] for key in ("model_id", "revision")}
    if preflight_path is not None:
        preflight = json.loads(Path(preflight_path).read_text(encoding="utf-8"))
        verified_pin = {key: preflight["model_pins"]["reader"][key] for key in ("model_id", "revision")}
        if preflight["status"] != "PASS" or verified_pin != model_pin:
            raise ValueError("HiDate reader pin must match its passed GPU preflight")
    protocol = {"system_prompt": SYSTEM_PROMPT, "user_prefix": USER_PREFIX,
                "max_new_tokens": MAX_NEW_TOKENS, "do_sample": False,
                "input_field": "source_quote_text only; no other query metadata"}
    run, output = Path(run_dir), Path(output_path)
    log_path, manifest_path = run / "generations.jsonl", run / "generation.json"
    previous = load_jsonl(log_path) if log_path.exists() else []
    if previous:
        if not resume:
            raise ValueError("Existing HiDate generations require explicit --resume continuation")
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior_manifest["model_pin"] != model_pin or prior_manifest["protocol"] != protocol:
            raise ValueError("Resume model pin or fixed paraphrase protocol differs")
    cache = {record["source_quote_text"]: record for record in previous if record["completed"]}
    run.mkdir(parents=True, exist_ok=True)
    manifest = {"complete": False, "started_at_utc": utc_now(), "queries_path": str(Path(queries_path).resolve()),
        "output_path": str(output.resolve()), "stack_path": str(Path(stack_path).resolve()),
        "preflight_path": str(Path(preflight_path).resolve()) if preflight_path else None,
        "model_pin": model_pin, "model_local_path": pin["local_path"], "protocol": protocol,
        "exact_queries": len(queries), "unique_source_quotes": len(quotes),
        "cache_policy": "exact source_quote_text equality; only this explicit execution and verified resume",
        "reused_query_quote_rows": len(queries) - len(quotes), "resume": resume,
        "completed_quotes_reused_on_resume": sum(quote in cache for quote in quotes),
        "generation_calls_this_run": 0, "training_runs": 0}

    def save():
        relevant = [cache[quote] for quote in quotes if quote in cache]
        manifest.update(completed_unique_quotes=len(relevant),
            prompt_tokens=sum(record["result"]["prompt_tokens"] for record in relevant),
            generated_tokens=sum(record["result"]["generated_tokens"] for record in relevant),
            model_token_work=sum(record["result"]["total_tokens"] for record in relevant),
            generation_wall_seconds_with_cuda_sync=sum(record["result"]["seconds"] for record in relevant),
            empty_outputs=sum(record["status"] == "empty_output" for record in relevant),
            end_status_counts=dict(Counter(record["result"]["end_status"] for record in relevant)))
        write_json(manifest_path, manifest)

    save()
    with log_path.open("a", encoding="utf-8") as stream:
        for number, quote in enumerate(quotes):
            if quote in cache:
                continue
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PREFIX + quote}]
            record = {"generation_id": f"quote-{number:06d}", "source_quote_text": quote,
                      "messages": messages, "model_pin": model_pin, "started_at_utc": utc_now()}
            if reader is None:
                from hidate.models import QwenReader
                reader = QwenReader(pin["local_path"], device="cuda", prompt_budget=4096)
            manifest["generation_calls_this_run"] += 1
            try:
                result = reader.generate_paraphrase(messages, max_new_tokens=MAX_NEW_TOKENS, include_inputs=True)
            except Exception as error:
                record.update(completed=False, status="model_error", error_type=type(error).__name__, error=str(error))
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                manifest.update(error_type=type(error).__name__, error=str(error), finished_at_utc=utc_now())
                save()
                raise
            record.update(completed=True, result=result, finished_at_utc=utc_now(),
                          status="generated" if result["text"].strip() else "empty_output")
            if record["status"] == "empty_output":
                record["error"] = "Reader returned no nonempty search query; output retained without substitution"
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            cache[quote] = record
            save()
            if len(cache) % 100 == 0 or len(cache) == len(quotes):
                print(json.dumps({key: manifest[key] for key in
                    ("completed_unique_quotes", "unique_source_quotes", "model_token_work", "empty_outputs")}), flush=True)
    links = [{"base_query_id": query["query_id"], "paraphrase": cache[query["source_quote_text"]]["result"]["text"]}
             for query in queries]
    mapped = attach_paraphrases(queries, links)
    generated_text_counts = Counter(row["query_text"] for row in mapped)
    for row in mapped:
        generation = cache[row["source_quote_text"]]
        row.update(query_variant="generated_paraphrase", generation_id=generation["generation_id"],
                   generation_status=generation["status"],
                   generated_query_text_occurrences=generated_text_counts[row["query_text"]])
        if "error" in generation:
            row["generation_error"] = generation["error"]
    write_jsonl(output, mapped)
    manifest.update(generation_attempts_complete=len(cache) == len(quotes), mapped_queries=len(mapped),
        distinct_generated_query_texts=len(generated_text_counts),
        complete=len(mapped) == len(queries) and manifest["empty_outputs"] == 0,
        finished_at_utc=utc_now(), wall_seconds=time.perf_counter() - started,
        model_inference_runs=manifest["generation_calls_this_run"])
    save()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=Path("data/processed/hidate/queries-exact.jsonl"))
    parser.add_argument("--stack", type=Path, default=Path("models/ruri/model-stack.json"))
    parser.add_argument("--preflight", type=Path, default=Path("runs/hidate/pbs/preflight/preflight.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/hidate/queries-paraphrase.jsonl"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/hidate/paraphrases"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = generate_paraphrases(args.queries, args.stack, args.output, args.run_dir,
                                 resume=args.resume, preflight_path=args.preflight)
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
