"""Acquire the fixed public HiDate stack at exact Hub revisions inside PBS.

Only metadata and snapshot files are acquired here. Interface preflight and
model inference are separate jobs; a download receipt is not an inference result.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path


MODEL_STACK = {
    "embedding": {
        "model_id": "cl-nagoya/ruri-v3-310m",
        "license": "apache-2.0",
        "conventions": {
            "interface": "SentenceTransformer.encode",
            "query_prefix": "検索クエリ: ",
            "document_prefix": "検索文書: ",
            "max_sequence_tokens": 8192,
            "embedding_dimension": 768,
            "similarity": "cosine",
            "pooling": {
                "word_embedding_dimension": 768,
                "pooling_mode_cls_token": False,
                "pooling_mode_mean_tokens": True,
                "pooling_mode_max_tokens": False,
                "pooling_mode_mean_sqrt_len_tokens": False,
                "pooling_mode_weightedmean_tokens": False,
                "pooling_mode_lasttoken": False,
                "include_prompt": True,
            },
            "pooling_config_file": "1_Pooling/config.json",
            "sequence_config_file": "sentence_bert_config.json",
            "tokenization": "SentencePiece; no external word segmentation",
            "transformers_minimum_version": "4.48.0",
        },
        "source_files": ["README.md", "1_Pooling/config.json",
                         "sentence_bert_config.json", "config_sentence_transformers.json"],
    },
    "reranker": {
        "model_id": "cl-nagoya/ruri-v3-reranker-310m",
        "license": "apache-2.0",
        "conventions": {
            "interface": "CrossEncoder.predict",
            "input_format": "raw query-document pairs",
            "query_prefix": "",
            "document_prefix": "",
            "max_sequence_tokens": 8192,
            "transformers_minimum_version": "4.48.0",
        },
        "source_files": ["README.md"],
    },
    "reader": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "license": "apache-2.0",
        "conventions": {
            "interface": "AutoModelForCausalLM",
            "chat_template": {
                "interface": "tokenizer.apply_chat_template",
                "source_file": "tokenizer_config.json",
                "tokenize": False,
                "add_generation_prompt": True,
            },
            "default_config_context_tokens": 32768,
            "advertised_extended_context_tokens": 131072,
            "extended_context_requires_yarn_configuration": True,
            "transformers_minimum_version": "4.37.0",
        },
        "source_files": ["README.md", "tokenizer_config.json", "config.json"],
    },
}
CONVENTIONS_CHECKED_ON = "2026-10-14"
DOWNLOAD_ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.md"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def acquire_models(output_dir, *, model_info=None, snapshot_download=None):
    """Pin metadata before downloading; retain every existing pin on resume.

    The two injected functions are the official Hub boundaries, replaced only
    by tiny software fixtures locally. Snapshot download supplies its own cache
    and interrupted-download continuation, always at the saved exact revision.
    """
    if model_info is None or snapshot_download is None:
        import huggingface_hub
        model_info = model_info or huggingface_hub.model_info
        snapshot_download = snapshot_download or huggingface_hub.snapshot_download
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "model-stack.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"project": "HiDate", "started_at_utc": utc_now(), "models": {},
                    "training_runs": 0, "model_inference_runs": 0}
    manifest["complete"] = False
    manifest["download_allow_patterns"] = list(DOWNLOAD_ALLOW_PATTERNS)

    def save():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")

    save()
    for role, definition in MODEL_STACK.items():
        model_id = definition["model_id"]
        if role in manifest["models"]:
            if manifest["models"][role]["model_id"] != model_id:
                raise ValueError(f"Existing {role} pin belongs to a different model")
            continue
        info = model_info(repo_id=model_id, token=False)
        revision = info.sha
        if not revision:
            raise ValueError(f"model_info returned no exact revision for {model_id}")
        card_url = f"https://huggingface.co/{model_id}/blob/{revision}/README.md"
        manifest["models"][role] = {
            "model_id": model_id, "revision": revision,
            "local_path": str(output / role), "resolved_at_utc": utc_now(),
            "revision_source": "huggingface_hub.model_info.sha",
            "hub_metadata": {"id": info.id, "private": info.private, "gated": info.gated},
            "license": definition["license"], "license_source_url": card_url,
            "model_card_url": card_url, "conventions_checked_on": CONVENTIONS_CHECKED_ON,
            "conventions": deepcopy(definition["conventions"]),
            "convention_source_urls": [f"https://huggingface.co/{model_id}/blob/{revision}/{name}"
                                       for name in definition["source_files"]],
            "download_complete": False,
        }
        save()

    for role in MODEL_STACK:
        record = manifest["models"][role]
        print(json.dumps({"role": role, "model_id": record["model_id"],
                          "revision": record["revision"], "local_path": record["local_path"]}), flush=True)
        path = snapshot_download(repo_id=record["model_id"], revision=record["revision"],
                                 local_dir=record["local_path"], token=False, max_workers=1,
                                 allow_patterns=manifest["download_allow_patterns"])
        record.update(download_complete=True, snapshot_path=str(Path(path).resolve()),
                      download_finished_at_utc=utc_now())
        save()
    manifest.update(complete=True, finished_at_utc=utc_now())
    save()
    print(json.dumps({"complete": True, "model_count": len(MODEL_STACK),
                      "manifest_path": str(manifest_path)}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("models/ruri"))
    args = parser.parse_args()
    acquire_models(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
