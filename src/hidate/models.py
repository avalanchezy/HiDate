"""Frozen HiDate Transformers interfaces with actual mask-based token receipts."""

from pathlib import Path
from contextlib import nullcontext
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def _load(model_class, model_path, device, tokenizer, model):
    if tokenizer is None or model is None:
        import transformers
        if tokenizer is None:
            tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        if model is None:
            options = {"local_files_only": True, "attn_implementation": "sdpa"}
            if torch.device(device).type == "cuda":
                options["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model = getattr(transformers, model_class).from_pretrained(str(model_path), **options)
    return tokenizer, model.to(device).eval()


def _start(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _receipt(inputs, start, device, include_inputs):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    counts = inputs["attention_mask"].sum(dim=1).detach().cpu().tolist()
    receipt = {"token_counts": [int(value) for value in counts],
               "total_tokens": int(sum(counts)), "seconds": time.perf_counter() - start,
               "input_shape": list(inputs["input_ids"].shape)}
    if include_inputs:
        receipt.update(input_ids=inputs["input_ids"].detach().cpu().tolist(),
                       attention_mask=inputs["attention_mask"].detach().cpu().tolist())
    return receipt


def _to_device(inputs, device):
    return {key: value.to(device) for key, value in inputs.items()}


def _ruri_sdpa_context(device):
    # The fixed variable-length GPU diagnostic isolates NaNs to default SDPA.
    return sdpa_kernel(backends=[SDPBackend.MATH]) if torch.device(device).type == "cuda" else nullcontext()


def create_encoder(pin, device="cuda", max_length=1024, **kwargs):
    """Select the declared retrieval model while keeping the original Ruri API."""
    if pin.get("runtime") == "qwen3" or pin["model_id"].startswith("Qwen/Qwen3-Embedding-"):
        from .qwen3 import DEFAULT_INSTRUCTION, Qwen3Encoder
        instruction = pin.get("conventions", {}).get("instruction", DEFAULT_INSTRUCTION)
        return Qwen3Encoder(pin["local_path"], device=device, max_length=max_length,
                            instruction=instruction, **kwargs)
    return RuriEncoder(pin["local_path"], device=device, max_length=max_length, **kwargs)


def create_reranker(pin, device="cuda", max_length=1024, **kwargs):
    if pin.get("runtime") == "qwen3" or pin["model_id"].startswith("Qwen/Qwen3-Reranker-"):
        from .qwen3 import DEFAULT_INSTRUCTION, Qwen3Reranker
        instruction = pin.get("conventions", {}).get("instruction", DEFAULT_INSTRUCTION)
        return Qwen3Reranker(pin["local_path"], device=device, max_length=max_length,
                             instruction=instruction, **kwargs)
    return RuriReranker(pin["local_path"], device=device, max_length=max_length, **kwargs)


class RuriEncoder:
    def __init__(self, model_path, device="cuda", max_length=1024, *, tokenizer=None, model=None):
        self.model_path, self.device, self.max_length = str(Path(model_path)), device, max_length
        self.tokenizer, self.model = _load("AutoModel", model_path, device, tokenizer, model)

    def encode(self, texts, kind="document", include_inputs=False):
        prefix = {"query": "検索クエリ: ", "document": "検索文書: "}[kind]
        start = _start(self.device)
        inputs = _to_device(self.tokenizer([prefix + text for text in texts], padding=True,
            truncation=True, max_length=self.max_length, return_tensors="pt", return_token_type_ids=False), self.device)
        with torch.inference_mode():
            with _ruri_sdpa_context(self.device):
                hidden = self.model(**inputs).last_hidden_state.float()
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            vectors = F.normalize(mean, p=2, dim=1)
        result = {"embeddings": vectors.cpu().tolist(), "output_shape": list(vectors.shape),
                  "sdpa_backend": "math" if torch.device(self.device).type == "cuda" else "default"}
        result.update(_receipt(inputs, start, self.device, include_inputs))
        return result


class RuriReranker:
    def __init__(self, model_path, device="cuda", max_length=1024, *, tokenizer=None, model=None):
        self.model_path, self.device, self.max_length = str(Path(model_path)), device, max_length
        self.tokenizer, self.model = _load("AutoModelForSequenceClassification", model_path, device, tokenizer, model)

    def score(self, pairs, include_inputs=False):
        queries, documents = zip(*pairs)
        start = _start(self.device)
        inputs = _to_device(self.tokenizer(list(queries), text_pair=list(documents), padding=True,
            truncation=True, max_length=self.max_length, return_tensors="pt", return_token_type_ids=False), self.device)
        with torch.inference_mode():
            with _ruri_sdpa_context(self.device):
                logits = self.model(**inputs).logits.float()
            if logits.ndim != 2 or logits.shape[1] != 1:
                raise ValueError("Ruri reranker must return one logit per query-document pair")
            scores = torch.sigmoid(logits[:, 0])
        result = {"logits": logits[:, 0].cpu().tolist(), "scores": scores.cpu().tolist(),
                  "output_shape": list(logits.shape),
                  "sdpa_backend": "math" if torch.device(self.device).type == "cuda" else "default"}
        result.update(_receipt(inputs, start, self.device, include_inputs))
        return result


class QwenReader:
    def __init__(self, model_path, device="cuda", prompt_budget=4096, *, tokenizer=None, model=None):
        self.model_path, self.device, self.prompt_budget = str(Path(model_path)), device, prompt_budget
        self.tokenizer, self.model = _load("AutoModelForCausalLM", model_path, device, tokenizer, model)

    def _prompt(self, messages):
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        tokens = int(inputs["attention_mask"].sum().item())
        if tokens > self.prompt_budget:
            raise ValueError(f"Full reader prompt has {tokens} tokens, exceeding {self.prompt_budget}; budget evidence before rendering")
        return _to_device(inputs, self.device)

    def score_choices(self, messages, include_inputs=False):
        start = _start(self.device)
        inputs = self._prompt(messages)
        choice_ids = {}
        for label in "abcd":
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Choice label {label} must encode as one Qwen token")
            choice_ids[label] = ids[0]
        positions = torch.arange(inputs["input_ids"].shape[1], device=self.device)
        last = int(positions.masked_fill(inputs["attention_mask"][0] == 0, -1).max().item())
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            logprobs = F.log_softmax(logits[0, last].float(), dim=-1)
        values = {label: float(logprobs[token_id].item()) for label, token_id in choice_ids.items()}
        result = {"choice_logprobs": values, "choice_token_ids": choice_ids, "choice_position": last,
                  "prediction": max(values, key=values.get), "output_shape": list(logits.shape)}
        result.update(_receipt(inputs, start, self.device, include_inputs))
        result["prompt_tokens"] = result["total_tokens"]
        return result

    def generate_paraphrase(self, messages, max_new_tokens=192, include_inputs=False):
        start = _start(self.device)
        inputs = self._prompt(messages)
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        new_ids = output[0, inputs["input_ids"].shape[1]:].cpu().tolist()
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        result = _receipt(inputs, start, self.device, include_inputs)
        prompt_tokens = result["total_tokens"]
        eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        if eos is None:
            eos = getattr(self.tokenizer, "eos_token_id", None)
        eos_ids = eos if isinstance(eos, list) else [eos]
        ended_with_eos = bool(new_ids and new_ids[-1] in eos_ids)
        result.update(text=text,
                      generated_token_ids=new_ids, generated_tokens=len(new_ids),
                      prompt_tokens=prompt_tokens, total_tokens=prompt_tokens + len(new_ids),
                      token_work_unit="prompt_nonpadding_tokens_plus_generated_tokens",
                      output_shape=list(output.shape), ended_with_eos=ended_with_eos,
                      max_new_tokens_reached=len(new_ids) >= max_new_tokens, max_new_tokens=max_new_tokens,
                      end_status="eos" if ended_with_eos else "max_tokens" if len(new_ids) >= max_new_tokens else "other_stop")
        return result
