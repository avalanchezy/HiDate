"""A tiny Hub boundary checks that Qwen3 downloads reuse the frozen reader."""

import json
from pathlib import Path
import runpy
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def test_qwen3_acquisition_downloads_two_pinned_models_and_reuses_reader(tmp_path):
    script = ROOT / "scripts/acquire_qwen3_models.py"
    assert script.exists(), "Qwen3 acquisition is not implemented"
    acquire = runpy.run_path(str(script))["acquire_models"]
    reader = {"model_id": "Qwen/Qwen2.5-7B-Instruct", "revision": "reader-pin",
              "local_path": "/existing/reader", "download_complete": True}
    original = tmp_path / "original-stack.json"
    original.write_text(json.dumps({"complete": True, "models": {"reader": reader}}), encoding="utf-8")
    downloads = []

    def info(repo_id, token):
        assert token is False
        return SimpleNamespace(id=repo_id, sha=repo_id.rsplit("/", 1)[1] + "-pin", private=False, gated=False)

    def download(repo_id, revision, local_dir, **kwargs):
        downloads.append((repo_id, revision))
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "README.md").write_text(repo_id, encoding="utf-8")
        return local_dir

    output = tmp_path / "qwen3"
    result = acquire(output, original, model_info=info, snapshot_download=download)
    assert result["complete"] and result["stack_id"] == "qwen3-0.6b"
    assert result["models"]["reader"] == reader
    assert [row[0] for row in downloads] == ["Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Reranker-0.6B"]
    assert result["models"]["embedding"]["conventions"]["embedding_dimension"] == 1024
    assert result["reader_reused_from"] == str(original.resolve())
    stored = json.loads((output / "model-stack.json").read_text(encoding="utf-8"))
    assert stored == result

    def no_new_pin(**kwargs):
        raise AssertionError("a saved model revision must be reused")

    repeated = acquire(output, original, model_info=no_new_pin, snapshot_download=download)
    assert repeated["models"] == result["models"]
    assert len(downloads) == 2
