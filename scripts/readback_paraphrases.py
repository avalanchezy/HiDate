"""Audit saved HiDate paraphrase construction on CPU without model forwards."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import re
import time

from hidate.io import load_jsonl, write_json


SYSTEM_PROMPT = ("与えられた引用だけを用いて、内容を日常的な日本語の検索依頼一つに言い換えてください。"
    "条件、数値、固有名詞、否定を保ってください。引用が途中で終わっていても続きを補わず、"
    "事実を追加しないでください。検索依頼だけを出力してください。")
USER_PREFIX = "引用:\n"
PROTOCOL = {"system_prompt": SYSTEM_PROMPT, "user_prefix": USER_PREFIX, "max_new_tokens": 128,
            "do_sample": False, "input_field": "source_quote_text only; no other query metadata"}
DIGIT_STRINGS = re.compile(r"[0-9０-９]+")


def lengths(values):
    return {"count": len(values), "min": min(values) if values else None,
            "mean": sum(values) / len(values) if values else None, "max": max(values) if values else None,
            "histogram": dict(sorted(Counter(values).items()))}


def number_diagnostics(source, output):
    original, generated = DIGIT_STRINGS.findall(source), DIGIT_STRINGS.findall(output)
    a, b = Counter(original), Counter(generated)
    return {"source_digit_strings": original, "output_digit_strings": generated,
        "preserved_digit_strings": list((a & b).elements()),
        "missing_digit_strings": list((a - b).elements()), "added_digit_strings": list((b - a).elements()),
        "all_source_digit_strings_preserved": not bool(a - b)}


def readback(exact_path, generated_path, generation_dir, stack_path, *, tokenizer=None, eos_token_ids=None,
             preflight_path=None, expected_counts=None):
    started = time.perf_counter()
    exact = load_jsonl(exact_path)
    mapped = load_jsonl(generated_path)
    run = Path(generation_dir)
    logs = load_jsonl(run / "generations.jsonl")
    manifest = json.loads((run / "generation.json").read_text(encoding="utf-8"))
    stack = json.loads(Path(stack_path).read_text(encoding="utf-8"))
    pin = stack["models"]["reader"]
    model_pin = {key: pin[key] for key in ("model_id", "revision")}
    issues = []

    def check(condition, description):
        if not condition:
            issues.append(description)

    check(stack["complete"] and pin["download_complete"], "reader acquisition is not complete")
    check(manifest["model_pin"] == model_pin, "generation manifest reader pin differs from the downloaded stack")
    check(manifest["protocol"] == PROTOCOL, "frozen greedy128/source-quote-only protocol differs")
    check(manifest["training_runs"] == 0, "generation manifest declares training")
    if preflight_path is not None:
        preflight = json.loads(Path(preflight_path).read_text(encoding="utf-8"))
        verified_pin = {key: preflight["model_pins"]["reader"][key] for key in ("model_id", "revision")}
        check(preflight["status"] == "PASS" and verified_pin == model_pin, "reader pin differs from passed GPU preflight")
    package_versions = {}
    if tokenizer is None:
        from importlib.metadata import version
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(pin["local_path"], local_files_only=True)
        package_versions = {name: version(name) for name in ("transformers", "tokenizers")}
        tokenizer_backend = "downloaded pinned Qwen AutoTokenizer; CPU only"
    else:
        tokenizer_backend = "injected CPU fixture tokenizer"
    if eos_token_ids is None:
        config_path = Path(pin["local_path"]) / "generation_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        eos = config.get("eos_token_id")
        eos_token_source = "downloaded generation_config.json"
        if eos is None:
            eos, eos_token_source = tokenizer.eos_token_id, "tokenizer.eos_token_id"
        eos_token_ids = eos if isinstance(eos, list) else [eos]
    else:
        eos_token_source = "explicit fixture EOS token IDs"

    quote_queries = defaultdict(list)
    original_text_counts = Counter(row["query_text"] for row in exact)
    for row in exact:
        identity, quote = row["query_id"], row["source_quote_text"]
        check(row["query_variant"] == "exact_quote" and row["query_text"] == quote,
              f"{identity}: original exact query text/variant disagrees with its source quote")
        check(row["source_span_start"] == 0 and row["source_span_end"] == len(quote)
              and quote == row["source_paragraph_text"][:96], f"{identity}: original source span is not its frozen96 prefix")
        check(row["query_text_occurrences"] == original_text_counts[row["query_text"]],
              f"{identity}: original query_text_occurrences differs from all original rows")
        quote_queries[quote].append(row)
    quote_order = list(quote_queries)
    quote_numbers = {quote: number for number, quote in enumerate(quote_order)}
    check(len({row["query_id"] for row in exact}) == len(exact), "original query IDs are duplicated")
    if expected_counts is not None:
        check(len(exact) == expected_counts["exact_queries"], "full original query count differs from9846")
        check(len(quote_order) == expected_counts["unique_source_quotes"], "full distinct source-quote count differs from4165")

    completed, failed_attempts, token_totals, end_counts = {}, [], Counter(), Counter()
    generation_ids = []
    for record in logs:
        identity, quote = record["generation_id"], record["source_quote_text"]
        generation_ids.append(identity)
        check(quote in quote_queries, f"{identity}: generation source quote is outside the original source quotes")
        check(identity == f"quote-{quote_numbers.get(quote, -1):06d}", f"{identity}: generation ID differs from first-occurrence quote order")
        expected_messages = [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": USER_PREFIX + quote}]
        check(record["messages"] == expected_messages, f"{identity}: messages differ from the exact frozen prompt plus source quote only")
        check(record["model_pin"] == model_pin, f"{identity}: actual generation reader pin differs")
        if not record["completed"]:
            failed_attempts.append({key: record[key] for key in
                ("generation_id", "source_quote_text", "status", "error_type", "error")})
            continue
        check(quote not in completed, f"{identity}: completed source quote is duplicated")
        result = record["result"]
        encoded = tokenizer.apply_chat_template(expected_messages, tokenize=True, add_generation_prompt=True, return_dict=True)
        ids, mask = encoded["input_ids"], encoded["attention_mask"]
        prompt_tokens, generated_ids = sum(mask), result["generated_token_ids"]
        check(result["input_ids"] == [ids], f"{identity}: actual Qwen input token IDs differ from CPU retokenization")
        check(result["attention_mask"] == [mask], f"{identity}: actual Qwen attention mask differs from CPU retokenization")
        check(result["input_shape"] == [1, len(ids)] and result["token_counts"] == [prompt_tokens]
              and result["prompt_tokens"] == prompt_tokens and prompt_tokens <= 4096,
              f"{identity}: prompt shape/nonpadding-token receipt differs")
        check(result["generated_tokens"] == len(generated_ids) and len(generated_ids) <= 128
              and result["max_new_tokens"] == 128, f"{identity}: output length/greedy128 budget receipt differs")
        check(result["total_tokens"] == prompt_tokens + len(generated_ids)
              and result["output_shape"] == [1, len(ids) + len(generated_ids)], f"{identity}: total token work/output shape differs")
        check(tokenizer.decode(generated_ids, skip_special_tokens=True) == result["text"],
              f"{identity}: generated text differs from decoding the actual saved output token IDs")
        ended = bool(generated_ids and generated_ids[-1] in eos_token_ids)
        at_limit = len(generated_ids) >= 128
        end_status = "eos" if ended else "max_tokens" if at_limit else "other_stop"
        check(result["ended_with_eos"] == ended and result["max_new_tokens_reached"] == at_limit
              and result["end_status"] == end_status, f"{identity}: EOS/length-limit/end-status receipt differs")
        check(not any(token in eos_token_ids for token in generated_ids[:-1]),
              f"{identity}: saved generation contains output after a configured EOS token")
        check(math.isfinite(result["seconds"]) and result["seconds"] >= 0., f"{identity}: generation wall-time receipt is invalid")
        expected_status = "generated" if result["text"].strip() else "empty_output"
        check(record["status"] == expected_status, f"{identity}: output status differs from its actual text")
        if expected_status == "empty_output":
            check(bool(record.get("error")), f"{identity}: empty output lacks its explicit retained error")
        completed[quote] = record
        token_totals.update(prompt_tokens=prompt_tokens, generated_tokens=len(generated_ids),
                            model_token_work=prompt_tokens + len(generated_ids))
        token_totals["generation_wall_seconds_with_cuda_sync"] += result["seconds"]
        end_counts[end_status] += 1
    check(set(completed) == set(quote_order), "complete generation quote set is missing or contains extra original-source quotes")
    check(manifest["generation_attempts_complete"], "generation attempts are not complete")
    for key, value in (("exact_queries", len(exact)), ("unique_source_quotes", len(quote_order)),
                       ("completed_unique_quotes", len(completed)), ("mapped_queries", len(mapped))):
        check(manifest[key] == value, f"generation manifest {key} count disagrees")
    for key, value in token_totals.items():
        check(math.isclose(manifest[key], value, rel_tol=1e-12, abs_tol=1e-9), f"generation manifest {key} work receipt disagrees")
    check(manifest["end_status_counts"] == dict(end_counts), "generation manifest end-status counts disagree")
    session_start = datetime.fromisoformat(manifest["started_at_utc"])
    calls_this_run = sum(datetime.fromisoformat(record["started_at_utc"]) >= session_start for record in logs)
    check(manifest["generation_calls_this_run"] == calls_this_run, "generation session model-call receipt disagrees")

    mapped_by_base = {row["base_query_id"]: row for row in mapped}
    generated_text_counts = Counter(row["query_text"] for row in mapped)
    check(len(mapped_by_base) == len(mapped) == len(exact), "mapped base-query IDs are missing, extra or duplicated")
    check([row["base_query_id"] for row in mapped] == [row["query_id"] for row in exact], "all mapped base IDs/order differ from the original query rows")
    check(len({row["query_id"] for row in mapped}) == len(mapped), "generated query IDs are duplicated")
    check(manifest["distinct_generated_query_texts"] == len(generated_text_counts), "manifest distinct generated query-text count disagrees")
    for original in exact:
        identity = original["query_id"]
        if identity not in mapped_by_base:
            continue
        row = mapped_by_base[identity]
        check(row["query_id"] == identity + "|paraphrase" and row["query_variant"] == "generated_paraphrase",
              f"{identity}: mapped ID/variant differs from the existing attach-paraphrases contract")
        for key, value in original.items():
            if key not in {"query_id", "query_variant", "query_text"}:
                check(key in row and row[key] == value, f"{identity}: preserved original field differs: {key}")
        check(row["generated_query_text_occurrences"] == generated_text_counts[row["query_text"]],
              f"{identity}: generated_query_text_occurrences differs from all mapped rows")
        record = completed.get(original["source_quote_text"])
        if record is not None:
            check(row["query_text"] == record["result"]["text"] and row["generation_id"] == record["generation_id"]
                  and row["generation_status"] == record["status"], f"{identity}: mapped text/generation provenance differs from its actual record")
            check(row.get("generation_error") == record.get("error"), f"{identity}: mapped generation error differs")

    quote_records = []
    for quote in quote_order:
        record = completed.get(quote)
        aliases = quote_queries[quote]
        item = {"source_quote_text": quote, "source_characters": len(quote),
            "mapped_base_query_ids": [row["query_id"] for row in aliases],
            "mapped_query_ids": [row["query_id"] + "|paraphrase" for row in aliases],
            "mapped_rows": len(aliases), "splits": dict(Counter(row["split"] for row in aliases)),
            "source_roles": dict(Counter(row["source_role"] for row in aliases))}
        if record is not None:
            text = record["result"]["text"]
            item.update(generation_id=record["generation_id"], generated_query_text=text,
                output_characters=len(text), output_tokens=len(record["result"]["generated_token_ids"]),
                status=record["status"], end_status=record["result"]["end_status"], exact_copy=text == quote,
                empty_output=not text.strip(), literal_digit_diagnostics=number_diagnostics(quote, text))
        else:
            item.update(generation_id=None, generated_query_text=None, status="missing_completed_generation")
        quote_records.append(item)
    available_records = [row for row in quote_records if row["generation_id"] is not None]
    diagnostics = {"distinct_generated_texts_unique_quotes": len({row["generated_query_text"] for row in available_records}),
        "distinct_generated_texts_mapped_queries": len(generated_text_counts),
        "exact_copy_unique_quotes": sum(row["exact_copy"] for row in available_records),
        "exact_copy_mapped_queries": sum(row["mapped_rows"] for row in available_records if row["exact_copy"]),
        "empty_unique_outputs": sum(row["empty_output"] for row in available_records),
        "empty_mapped_outputs": sum(row["mapped_rows"] for row in available_records if row["empty_output"]),
        "quotes_with_source_digit_strings": sum(bool(row["literal_digit_diagnostics"]["source_digit_strings"]) for row in available_records),
        "quotes_with_missing_digit_strings": sum(bool(row["literal_digit_diagnostics"]["missing_digit_strings"]) for row in available_records),
        "quotes_with_added_digit_strings": sum(bool(row["literal_digit_diagnostics"]["added_digit_strings"]) for row in available_records),
        "source_character_lengths": lengths([len(quote) for quote in quote_order]),
        "output_character_lengths": lengths([row["output_characters"] for row in available_records]),
        "output_token_lengths": lengths([row["output_tokens"] for row in available_records]),
        "digit_rule": "maximal [0-9０-９]+ strings, exact width and occurrence counts; no Unicode normalization or kanji-number conversion",
        "interpretation": "literal source/output diagnostics only; not semantic truth, expert ratings, or dataset filtering"}
    check(manifest["empty_outputs"] == diagnostics["empty_unique_outputs"], "manifest empty-output count disagrees")
    sample_quotes = random.Random(42).sample(sorted(quote_order), min(40, len(quote_order)))
    record_by_quote = {row["source_quote_text"]: row for row in quote_records}
    return {"status": "PASS" if not issues else "FAIL", "complete": not bool(issues), "issues": issues,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.perf_counter() - started,
        "job_id": os.environ.get("PBS_JOBID"), "model_pin": model_pin, "model_local_path": pin["local_path"],
        "tokenizer_backend": tokenizer_backend, "package_versions": package_versions,
        "eos_token_ids": eos_token_ids, "eos_token_source": eos_token_source,
        "protocol": PROTOCOL, "generation_manifest_complete": manifest["complete"],
        "paths": {"exact_queries": str(Path(exact_path).resolve()), "generated_queries": str(Path(generated_path).resolve()),
                  "generations": str((run / "generations.jsonl").resolve()), "manifest": str((run / "generation.json").resolve()),
                  "stack": str(Path(stack_path).resolve())},
        "exact_queries_checked": len(exact), "mapped_queries_checked": len(mapped),
        "unique_source_quotes_checked": len(quote_order), "completed_generation_records_checked": len(completed),
        "all_generation_record_ids": generation_ids, "failed_generation_attempts": failed_attempts,
        "generation_calls_total": len(logs), "generation_calls_in_manifest_session": calls_this_run,
        "actual_work_receipts": dict(token_totals), "end_status_counts": dict(end_counts),
        "diagnostics": diagnostics, "generation_records": quote_records,
        "sample_rule": "uniform40 distinct source quotes from sorted ALL unique quotes, random.Random(42); no split or quality filtering",
        "sample_seed": 42, "sample_size": len(sample_quotes), "sample": [record_by_quote[quote] for quote in sample_quotes],
        "verification_scope": "all input messages, pinned-tokenizer input IDs/masks, saved output decoding, EOS/length/work receipts and original query fields; declared greedy configuration, no neural reexecution",
        "training_runs": 0, "model_inference_runs": 0, "rewritten_queries": 0, "filtered_queries": 0, "regenerated_queries": 0}


def write_audit_markdown(report, path):
    lines = ["# HiDate query construction data audit", "",
        "This is a source and construction audit, not a manuscript or a semantic-quality evaluation.", "",
        f"Verification: **{report['status']}**. Checked {report['exact_queries_checked']} original queries, "
        f"{report['mapped_queries_checked']} mapped queries and {report['unique_source_quotes_checked']} distinct source quotes.", "",
        "Input comparison uses the exact frozen Japanese prompt and the source quote only. Metadata dates, IDs, titles, gold labels and unquoted context are not added to model input. Content already present inside the literal quote is retained.", "",
        "All counts and IDs remain in verification.json. No query was filtered, rewritten or regenerated; no training or neural forward was performed.", "",
        "| Literal diagnostic | Count |", "| --- | ---: |"]
    for key in ("distinct_generated_texts_unique_quotes", "distinct_generated_texts_mapped_queries", "exact_copy_unique_quotes",
                "exact_copy_mapped_queries", "empty_unique_outputs", "empty_mapped_outputs", "quotes_with_source_digit_strings",
                "quotes_with_missing_digit_strings", "quotes_with_added_digit_strings"):
        lines.append(f"| {key} | {report['diagnostics'][key]} |")
    lines.extend(["", "Digit strings are maximal runs of ASCII/fullwidth Arabic digits, compared by exact string and occurrence count. A width change is a literal difference; these diagnostics do not establish semantic truth or error.", "",
        "The sample below is uniform40 distinct source quotes from ALL unique quotes, using seed42 and sorted literal quotes. It is a readable audit sample, not a TEST-selection rule or an expert rating.", ""])
    if report["issues"]:
        lines.extend(["## Integrity issues", "", *["- " + issue for issue in report["issues"]], ""])
    for number, row in enumerate(report["sample"], 1):
        source, output = row["source_quote_text"], row["generated_query_text"]
        text = source + (output or "")
        fence = "`" * max(3, max((len(match.group()) + 1 for match in re.finditer(r"`+", text)), default=0))
        lines.extend([f"## Audit sample {number:02d}: {row['generation_id']}", "",
            f"Original rows: {row['mapped_rows']}; splits: {json.dumps(row['splits'], ensure_ascii=False)}.", "",
            "Source quote:", "", fence, source, fence, "", "Generated text:", "", fence,
            output if output is not None else "[Missing completed generation; integrity failure]", fence, ""])
        if row["generation_id"] is not None:
            lines.extend([f"Characters source/output: {row['source_characters']}/{row['output_characters']}; "
                f"output tokens: {row['output_tokens']}; end status: {row['end_status']}; exact copy: {row['exact_copy']}.", "",
                "Literal digit diagnostics: " + json.dumps(row["literal_digit_diagnostics"], ensure_ascii=False), ""])
        lines.extend(["Base query IDs:", "", *["- " + identity for identity in row["mapped_base_query_ids"]], ""])
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact", type=Path, default=Path("data/processed/hidate/queries-exact.jsonl"))
    parser.add_argument("--generated", type=Path, default=Path("data/processed/hidate/queries-paraphrase.jsonl"))
    parser.add_argument("--generation-dir", type=Path, default=Path("runs/hidate/paraphrases"))
    parser.add_argument("--stack", type=Path, default=Path("models/ruri/model-stack.json"))
    parser.add_argument("--preflight", type=Path, default=Path("runs/hidate/pbs/preflight/preflight.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/hidate/paraphrases/verification.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/hidate-query-construction.md"))
    args = parser.parse_args()
    report = readback(args.exact, args.generated, args.generation_dir, args.stack, preflight_path=args.preflight,
                      expected_counts={"exact_queries": 9846, "unique_source_quotes": 4165})
    write_json(args.output, report)
    write_audit_markdown(report, args.report)
    print(json.dumps({key: report[key] for key in ("status", "exact_queries_checked", "mapped_queries_checked",
        "unique_source_quotes_checked", "sample_size", "model_inference_runs")}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
