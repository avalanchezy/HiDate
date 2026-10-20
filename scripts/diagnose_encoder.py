"""Locate nonfinite states in the first real HiDate corpus batch on an allocated GPU."""

import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from hidate.models import RuriEncoder


def main():
    stack = json.loads(Path("models/ruri/model-stack.json").read_text())
    texts, seen, ids = [], set(), []
    with Path("data/index/hidate/chunks.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["text"] in seen:
                continue
            seen.add(row["text"])
            texts.append(row["text"])
            ids.append(row["chunk_id"])
            if len(texts) == 32:
                break
    encoder = RuriEncoder(stack["models"]["embedding"]["local_path"], device="cuda")
    report = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0),
              "dtype": str(next(encoder.model.parameters()).dtype),
              "attention": encoder.model.config._attn_implementation,
              "reference_compile": getattr(encoder.model.config, "reference_compile", None),
              "chunk_ids": ids, "comparisons": {}}

    def check(name, sample):
        inputs = encoder.tokenizer(["検索文書: " + text for text in sample], padding=True,
            truncation=True, max_length=1024, return_tensors="pt", return_token_type_ids=False)
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.inference_mode():
            result = encoder.model(**inputs, output_hidden_states=True)
        mask = inputs["attention_mask"].bool()
        layers = []
        for index, state in enumerate(result.hidden_states):
            bad = ~torch.isfinite(state).all(dim=-1)
            layers.append({"index": index, "valid_bad_tokens": int((bad & mask).sum()),
                           "padding_bad_tokens": int((bad & ~mask).sum())})
        hidden = result.last_hidden_state.float()
        mean = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).unsqueeze(-1)
        replaced_mean = hidden.masked_fill(~mask.unsqueeze(-1), 0).sum(dim=1) / mask.sum(dim=1).unsqueeze(-1)
        vectors = F.normalize(mean, dim=1)
        good_rows = torch.isfinite(vectors).all(dim=1)
        row = {"input_shape": list(inputs["input_ids"].shape), "hidden_shape": list(hidden.shape),
               "token_counts": mask.sum(dim=1).tolist(), "finite_vector_rows": int(good_rows.sum()),
               "bad_vector_rows": (~good_rows).nonzero().flatten().tolist(),
               "padding_replacement_finite_rows": int(torch.isfinite(replaced_mean).all(dim=1).sum()),
               "layers": layers, "backend_flags": {"flash": torch.backends.cuda.flash_sdp_enabled(),
                   "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                   "math": torch.backends.cuda.math_sdp_enabled()}}
        if good_rows.any():
            row["first_finite_vector"] = vectors[good_rows][0].cpu().tolist()
        report["comparisons"][name] = row
        Path("runs/hidate/pbs/encoder-diagnostic/diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
        del result, hidden, mean, replaced_mean, vectors
        gc.collect()
        torch.cuda.empty_cache()

    check("sdpa_default_batch32", texts)
    check("sdpa_default_single", texts[:1])
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        check("sdpa_math_batch32", texts)
    print(json.dumps({key: {field: value[field] for field in ("input_shape", "finite_vector_rows", "padding_replacement_finite_rows")}
                      for key, value in report["comparisons"].items()}), flush=True)


if __name__ == "__main__":
    main()
