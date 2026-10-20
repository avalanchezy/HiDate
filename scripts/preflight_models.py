"""Run the fixed HiDate interfaces once on a scheduled GPU, using real probe text."""

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import socket

import torch

from hidate.models import QwenReader, create_encoder, create_reranker
from hidate.sources import extract_provisions


PROBE_IDS = ["322AC0000000049", "129AC0000000089", "417AC0000000086"]
PROBE_ASOF = "2026-09-30"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def probe_paragraphs(source_dir):
    records = []
    for law_id in PROBE_IDS:
        path = Path(source_dir) / "snapshots" / law_id / f"{PROBE_ASOF}.json"
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        for provision in extract_provisions(snapshot):
            if provision["scope"] != "MainProvision" or provision["deleted"] or provision["hidden"]:
                continue
            paragraph = next((p for p in provision["paragraphs"]
                              if p["text"].strip() and not p["deleted"] and not p["hidden"]), None)
            if paragraph is not None:
                records.append({"law_id": law_id, "revision_id": snapshot["revision_info"]["law_revision_id"],
                    "requested_asof": PROBE_ASOF, "raw_path": str(path),
                    "law_title": provision["law_title"], "article_num": provision["article_num"],
                    "paragraph_num": paragraph["num"], "tree_path": paragraph["tree_path"], "text": paragraph["text"]})
                break
        else:
            raise ValueError(f"Probe snapshot contains no substantive main paragraph: {law_id}")
    return records


def masks_pass(result):
    masks = result["attention_mask"]
    return (len(masks) == result["input_shape"][0]
        and all(len(row) == result["input_shape"][1] for row in masks)
        and all(value in (0, 1) for row in masks for value in row)
        and [sum(row) for row in masks] == result["token_counts"]
        and sum(result["token_counts"]) > 0)


def model_details(wrapper):
    return {"class": type(wrapper.model).__name__, "dtype": str(next(wrapper.model.parameters()).dtype),
            "device": str(next(wrapper.model.parameters()).device),
            "attention_implementation": wrapper.model.config._attn_implementation}


def release(wrapper):
    del wrapper.model
    del wrapper.tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def run_preflight(stack_path, source_dir, output_path):
    import transformers
    if not torch.cuda.is_available():
        raise RuntimeError("HiDate preflight requires the allocated CUDA GPU")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stack = json.loads(Path(stack_path).read_text(encoding="utf-8"))
    if not stack["complete"]:
        raise ValueError("Model resource acquisition is not complete")
    paragraphs = probe_paragraphs(source_dir)
    documents = [row["text"] for row in paragraphs]
    queries = [row["law_title"] + "の基本的な規定は何ですか。" for row in paragraphs]
    report = {"status": "RUNNING", "started_at_utc": utc_now(), "job_id": os.environ.get("PBS_JOBID"),
        "hostname": socket.gethostname(), "torch_version": torch.__version__,
        "transformers_version": transformers.__version__, "cuda_version": torch.version.cuda,
        "device": {"name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
                   "bf16_supported": torch.cuda.is_bf16_supported()},
        "model_pins": {role: {key: row[key] for key in ("model_id", "revision", "local_path")}
                       for role, row in stack["models"].items()},
        "source_paragraphs": paragraphs, "training_runs": 0, "stages": {}}

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        encoder = create_encoder(stack["models"]["embedding"], device="cuda", max_length=1024)
        try:
            query_result = encoder.encode(queries, kind="query", include_inputs=True)
            document_result = encoder.encode(documents, kind="document", include_inputs=True)
            dimension = stack["models"]["embedding"]["conventions"]["embedding_dimension"]
            norms = [math.sqrt(sum(value * value for value in row)) for row in document_result["embeddings"]]
            finite = all(math.isfinite(value) for result in (query_result, document_result)
                         for row in result["embeddings"] for value in row)
            unit_norms = all(abs(math.sqrt(sum(value * value for value in row)) - 1.0) < 0.01
                             for result in (query_result, document_result) for row in result["embeddings"])
            stage = {"model": model_details(encoder), "query_texts": queries, "document_texts": documents,
                "queries": query_result, "documents": document_result, "document_norms": norms,
                "checks": {"finite": finite, "unit_norms": unit_norms,
                    "shape": query_result["output_shape"] == document_result["output_shape"] == [3, dimension],
                    "masks": masks_pass(query_result) and masks_pass(document_result),
                    "input_limit": max(query_result["input_shape"][1], document_result["input_shape"][1]) <= 1024}}
            stage["pass"] = all(stage["checks"].values())
            report["stages"]["embedding"] = stage
            save()
        finally:
            release(encoder)
        report["stages"]["embedding"]["cuda_allocated_after_release_bytes"] = torch.cuda.memory_allocated()
        save()

        reranker = create_reranker(stack["models"]["reranker"], device="cuda", max_length=1024)
        try:
            pairs = list(zip(queries, documents))
            result = reranker.score(pairs, include_inputs=True)
            stage = {"model": model_details(reranker), "pairs": pairs, "result": result,
                "checks": {"finite": all(math.isfinite(value) for value in result["logits"] + result["scores"]),
                    "shape": result["output_shape"] == [3, 1],
                    "sigmoid_range": all(0.0 <= value <= 1.0 for value in result["scores"]),
                    "masks": masks_pass(result), "input_limit": result["input_shape"][1] <= 1024}}
            stage["pass"] = all(stage["checks"].values())
            report["stages"]["reranker"] = stage
            save()
        finally:
            release(reranker)
        report["stages"]["reranker"]["cuda_allocated_after_release_bytes"] = torch.cuda.memory_allocated()
        save()

        reader = QwenReader(stack["models"]["reader"]["local_path"], device="cuda", prompt_budget=4096)
        try:
            messages = [{"role": "system", "content": "四つの選択肢から一つを選び、a、b、c、dの一文字で答えてください。"},
                        {"role": "user", "content": "次のうち曜日を表す語はどれですか。\na. 月曜日\nb. 赤色\nc. 三角形\nd. 東京"}]
            choices = reader.score_choices(messages, include_inputs=True)
            paraphrase_messages = [{"role": "user", "content": "次の条文を意味を保って短い日本語に言い換えてください。\n" + documents[0]}]
            paraphrase = reader.generate_paraphrase(paraphrase_messages, max_new_tokens=192, include_inputs=True)
            last_attended = max(index for index, value in enumerate(choices["attention_mask"][0]) if value)
            stage = {"model": model_details(reader), "choice_messages": messages, "choices": choices,
                "paraphrase_messages": paraphrase_messages, "paraphrase": paraphrase,
                "checks": {"finite": all(math.isfinite(value) for value in choices["choice_logprobs"].values()),
                    "four_choice_tokens": set(choices["choice_logprobs"]) == set("abcd")
                        and len(set(choices["choice_token_ids"].values())) == 4,
                    "choice_position": choices["choice_position"] == last_attended,
                    "masks": masks_pass(choices) and masks_pass(paraphrase),
                    "prompt_budget": max(choices["prompt_tokens"], paraphrase["prompt_tokens"]) <= 4096,
                    "generation_shape": paraphrase["output_shape"] == [1, paraphrase["input_shape"][1] + paraphrase["generated_tokens"]]},
                "prediction_is_reported_only": True}
            stage["pass"] = all(stage["checks"].values())
            report["stages"]["reader"] = stage
            save()
        finally:
            release(reader)
        report["stages"]["reader"]["cuda_allocated_after_release_bytes"] = torch.cuda.memory_allocated()
        save()
    except Exception as error:
        report.update(status="ERROR", error_type=type(error).__name__, error=str(error), finished_at_utc=utc_now())
        save()
        raise
    report.update(status="PASS" if all(stage["pass"] for stage in report["stages"].values()) else "FAIL",
                  finished_at_utc=utc_now(), interface_calls=5, loaded_models=3,
                  models_loaded_sequentially=True)
    save()
    print(json.dumps({"status": report["status"], "model_count": 3, "output": str(output)}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", type=Path, default=Path("models/ruri/model-stack.json"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/hidate/probe"))
    parser.add_argument("--output", type=Path, default=Path("runs/hidate/pbs/preflight/preflight.json"))
    args = parser.parse_args()
    return 0 if run_preflight(args.stack, args.source_dir, args.output)["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
