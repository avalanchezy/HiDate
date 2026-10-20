"""Tiny Hub fixtures exercise revision capture without real model downloads."""

import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/acquire_models.py"
MODEL_IDS = {
    "embedding": "cl-nagoya/ruri-v3-310m",
    "reranker": "cl-nagoya/ruri-v3-reranker-310m",
    "reader": "Qwen/Qwen2.5-7B-Instruct",
}
REVISIONS = {
    "cl-nagoya/ruri-v3-310m": "a" * 40,
    "cl-nagoya/ruri-v3-reranker-310m": "b" * 40,
    "Qwen/Qwen2.5-7B-Instruct": "c" * 40,
}


def acquire():
    assert SCRIPT.exists(), "HiDate model resource acquisition is not implemented"
    return runpy.run_path(str(SCRIPT))["acquire_models"]


class FakeHub:
    """Only the remote metadata and bytes boundary is replaced."""

    def __init__(self, failure=None):
        self.info_calls = []
        self.downloads = []
        self.failure = failure

    def model_info(self, repo_id, *, token):
        self.info_calls.append((repo_id, token))
        return SimpleNamespace(id=repo_id, sha=REVISIONS[repo_id],
                               private=False, gated=False,
                               card_data={"license": "apache-2.0"})

    def snapshot_download(self, repo_id, *, revision, local_dir, token, max_workers,
                          allow_patterns=None):
        self.downloads.append({"repo_id": repo_id, "revision": revision,
                               "local_dir": str(local_dir), "token": token,
                               "max_workers": max_workers,
                               "allow_patterns": allow_patterns})
        if repo_id == self.failure:
            raise OSError("interrupted public snapshot download")
        path = Path(local_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "README.md").write_text(repo_id + "\n" + revision, encoding="utf-8")
        return str(path)


def test_exact_public_revisions_and_distinct_resources_survive_in_manifest(tmp_path):
    output = tmp_path / "models" / "hidate"
    hub = FakeHub()
    manifest = acquire()(output, model_info=hub.model_info,
                         snapshot_download=hub.snapshot_download)
    assert manifest["complete"] is True
    assert manifest["training_runs"] == manifest["model_inference_runs"] == 0
    assert hub.info_calls == [(model_id, False) for model_id in MODEL_IDS.values()]
    assert len(hub.downloads) == 3
    assert {row["revision"] for row in hub.downloads} == {"a" * 40, "b" * 40, "c" * 40}
    assert all(row["token"] is False and row["max_workers"] == 1 for row in hub.downloads)
    assert all(row["allow_patterns"] == ["*.json", "*.safetensors", "*.model", "*.txt", "*.md"]
               for row in hub.downloads)
    stored = json.loads((output / "model-stack.json").read_text(encoding="utf-8"))
    assert stored["download_allow_patterns"] == ["*.json", "*.safetensors", "*.model", "*.txt", "*.md"]
    assert set(stored["models"]) == {"embedding", "reranker", "reader"}
    for role, model_id in MODEL_IDS.items():
        record = stored["models"][role]
        assert record["model_id"] == model_id
        assert record["revision"] == REVISIONS[model_id]
        assert record["local_path"] == str((output / role).resolve())
        assert record["download_complete"] is True
        assert record["license"] == "apache-2.0"
        assert record["revision_source"] == "huggingface_hub.model_info.sha"
        assert record["model_card_url"] == f"https://huggingface.co/{model_id}/blob/{REVISIONS[model_id]}/README.md"
        assert (output / role / "README.md").read_text(encoding="utf-8") == model_id + "\n" + REVISIONS[model_id]
    embedding = stored["models"]["embedding"]["conventions"]
    assert embedding["query_prefix"] == "検索クエリ: "
    assert embedding["document_prefix"] == "検索文書: "
    assert embedding["max_sequence_tokens"] == 8192
    assert embedding["pooling"]["pooling_mode_mean_tokens"] is True
    assert embedding["pooling"]["include_prompt"] is True
    assert stored["models"]["reranker"]["conventions"]["input_format"] == "raw query-document pairs"
    assert stored["models"]["reader"]["conventions"]["chat_template"]["add_generation_prompt"] is True


def test_interrupted_download_keeps_all_pins_and_resume_never_refreshes_them(tmp_path):
    output = tmp_path / "hidate"
    hub = FakeHub(failure="cl-nagoya/ruri-v3-reranker-310m")
    collect = acquire()
    with pytest.raises(OSError, match="interrupted"):
        collect(output, model_info=hub.model_info, snapshot_download=hub.snapshot_download)
    pinned = json.loads((output / "model-stack.json").read_text(encoding="utf-8"))
    assert pinned["complete"] is False
    assert set(pinned["models"]) == {"embedding", "reranker", "reader"}
    assert pinned["models"]["embedding"]["download_complete"] is True
    assert pinned["models"]["reranker"]["download_complete"] is False

    def no_new_pin(repo_id, **kwargs):
        pytest.fail(f"resume refreshed a frozen public revision: {repo_id}")

    resumed = FakeHub()
    manifest = collect(output, model_info=no_new_pin, snapshot_download=resumed.snapshot_download)
    assert manifest["complete"] is True
    for role in MODEL_IDS:
        assert manifest["models"][role]["revision"] == pinned["models"][role]["revision"]
        assert manifest["models"][role]["resolved_at_utc"] == pinned["models"][role]["resolved_at_utc"]
    assert {row["repo_id"]: row["revision"] for row in resumed.downloads} == REVISIONS


def test_completed_resume_reuses_existing_pins_without_model_info(tmp_path):
    output = tmp_path / "hidate"
    hub = FakeHub()
    collect = acquire()
    first = collect(output, model_info=hub.model_info, snapshot_download=hub.snapshot_download)

    def no_new_pin(repo_id, **kwargs):
        pytest.fail(f"completed acquisition refreshed a frozen revision: {repo_id}")

    resumed = FakeHub()
    second = collect(output, model_info=no_new_pin, snapshot_download=resumed.snapshot_download)
    assert second["complete"] is True
    assert {role: row["revision"] for role, row in second["models"].items()} == {
        role: row["revision"] for role, row in first["models"].items()}
    assert len({row["local_dir"] for row in resumed.downloads}) == 3


def test_partial_metadata_resolution_persists_pin_before_later_metadata_failure(tmp_path):
    output = tmp_path / "hidate"
    hub = FakeHub()

    def fail_second_info(repo_id, **kwargs):
        if repo_id == "cl-nagoya/ruri-v3-reranker-310m":
            raise OSError("metadata request interrupted")
        return hub.model_info(repo_id, **kwargs)

    collect = acquire()
    with pytest.raises(OSError, match="metadata request interrupted"):
        collect(output, model_info=fail_second_info, snapshot_download=hub.snapshot_download)
    pinned = json.loads((output / "model-stack.json").read_text(encoding="utf-8"))
    assert pinned["complete"] is False
    assert pinned["models"]["embedding"]["revision"] == "a" * 40
    assert hub.downloads == []

    def only_missing_info(repo_id, **kwargs):
        assert repo_id != "cl-nagoya/ruri-v3-310m", "already resolved pin must be retained"
        return hub.model_info(repo_id, **kwargs)

    resumed = collect(output, model_info=only_missing_info, snapshot_download=hub.snapshot_download)
    assert resumed["complete"] is True
    assert resumed["models"]["embedding"]["resolved_at_utc"] == pinned["models"]["embedding"]["resolved_at_utc"]
