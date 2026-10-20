"""Acquire the HiDate Qwen3 retrieval stack; reuse the original pinned Qwen2.5 reader."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path


TASK = "Given a web search query, retrieve relevant passages that answer the query"
MODELS = {
    "embedding": {
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "runtime": "qwen3",
        "conventions": {"interface": "AutoModel", "embedding_dimension": 1024,
            "pooling": "last_attended_token", "normalization": "l2", "padding_side": "left",
            "query_format": "Instruct: {instruction}\nQuery:{query}", "document_prefix": "",
            "instruction": TASK, "max_sequence_tokens": 32768, "experiment_max_length": 1024,
            "transformers_minimum_version": "4.51.0"}},
    "reranker": {
        "model_id": "Qwen/Qwen3-Reranker-0.6B",
        "runtime": "qwen3",
        "conventions": {"interface": "AutoModelForCausalLM", "padding_side": "left",
            "input_format": "official Instruct/Query/Document template with no-thinking suffix",
            "score": "P(yes | yes,no) at the final prompt token",
            "receipt_logit": "yes_logit_minus_no_logit", "instruction": TASK,
            "max_sequence_tokens": 32768, "experiment_max_length": 1024,
            "transformers_minimum_version": "4.51.0"}},
}
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.md"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def acquire_models(output_dir, reader_stack, *, model_info=None, snapshot_download=None):
    if model_info is None or snapshot_download is None:
        from huggingface_hub import model_info as hub_info, snapshot_download as hub_download
        model_info = model_info or hub_info
        snapshot_download = snapshot_download or hub_download
    original = json.loads(Path(reader_stack).read_text(encoding="utf-8"))
    if not original.get("complete") or not original["models"]["reader"].get("download_complete"):
        raise ValueError("Download the pinned reader first with scripts/download_frozen_package_models.py")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "model-stack.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else
        {"project": "HiDate", "stack_id": "qwen3-0.6b", "started_at_utc": utc_now(),
         "models": {"reader": deepcopy(original["models"]["reader"])},
         "reader_reused_from": str(Path(reader_stack).resolve()),
         "training_runs": 0, "model_inference_runs": 0})
    manifest["complete"] = False
    manifest["download_allow_patterns"] = ALLOW_PATTERNS

    def save():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save()
    for role, definition in MODELS.items():
        if role in manifest["models"]:
            continue
        model_id = definition["model_id"]
        info = model_info(repo_id=model_id, token=False)
        manifest["models"][role] = {**deepcopy(definition), "revision": info.sha,
            "local_path": str(output / role), "resolved_at_utc": utc_now(),
            "revision_source": "huggingface_hub.model_info.sha", "license": "apache-2.0",
            "hub_metadata": {"id": info.id, "private": info.private, "gated": info.gated},
            "model_card_url": f"https://huggingface.co/{model_id}/blob/{info.sha}/README.md",
            "conventions_checked_on": "2026-10-18", "download_complete": False}
        save()
    for role in MODELS:
        record = manifest["models"][role]
        if record.get("download_complete"):
            continue
        print(json.dumps({"role": role, "model_id": record["model_id"], "revision": record["revision"]}), flush=True)
        path = snapshot_download(repo_id=record["model_id"], revision=record["revision"],
            local_dir=record["local_path"], token=False, max_workers=1, allow_patterns=ALLOW_PATTERNS)
        record.update(download_complete=True, snapshot_path=str(Path(path).resolve()),
                      download_finished_at_utc=utc_now())
        save()
    manifest.update(complete=True, finished_at_utc=utc_now())
    save()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("models/qwen3"))
    parser.add_argument("--reader-stack", type=Path, default=Path("models/ruri/model-stack.json"))
    args = parser.parse_args()
    result = acquire_models(args.output_dir, args.reader_stack)
    print(json.dumps({"complete": result["complete"], "stack": str(args.output_dir / "model-stack.json")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
