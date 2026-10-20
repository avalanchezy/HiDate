"""Download only the model revisions already pinned in the research package."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    root = args.root.resolve()
    paths = [root / "models/ruri/model-stack.json", root / "models/qwen3/model-stack.json"]
    downloaded = {}
    for path in paths:
        stack = json.loads(path.read_text(encoding="utf-8"))
        for role, pin in stack["models"].items():
            key = pin["model_id"], pin["revision"]
            folder = root / ("models/ruri" if role == "reader" or path.parent.name == "ruri" else "models/qwen3") / role
            if key not in downloaded:
                snapshot_download(repo_id=key[0], revision=key[1], local_dir=str(folder), token=False,
                                  max_workers=1, allow_patterns=stack["download_allow_patterns"])
                downloaded[key] = folder
            pin.update(local_path=str(downloaded[key]), snapshot_path=str(downloaded[key]), download_complete=True)
        stack.update(complete=True, package_relocated_at_utc=datetime.now(timezone.utc).isoformat())
        if "reader_reused_from" in stack:
            stack["reader_reused_from"] = str(paths[0])
        path.write_text(json.dumps(stack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "unique_pinned_models": len(downloaded), "backbone_training_runs": 0}))


if __name__ == "__main__":
    main()
