"""Official Qwen3 retrieval conventions behind the HiDate encode/score interface.

Sources: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B and
https://huggingface.co/Qwen/Qwen3-Reranker-0.6B (checked 2026-10-18).
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from .models import _load, _receipt, _start, _to_device


DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
RERANK_PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
RERANK_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def last_token_pool(hidden, attention_mask):
    if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
        return hidden[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]


class Qwen3Encoder:
    def __init__(self, model_path, device="cuda", max_length=1024, *, instruction=DEFAULT_INSTRUCTION,
                 tokenizer=None, model=None):
        self.model_path, self.device, self.max_length = str(Path(model_path)), device, max_length
        self.instruction = instruction
        self.tokenizer, self.model = _load("AutoModel", model_path, device, tokenizer, model)
        self.tokenizer.padding_side = "left"

    def encode(self, texts, kind="document", include_inputs=False):
        prefix = {"query": f"Instruct: {self.instruction}\nQuery:", "document": ""}[kind]
        texts = [prefix + text for text in texts]
        start = _start(self.device)
        inputs = _to_device(self.tokenizer(texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt", return_token_type_ids=False), self.device)
        with torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state
            vectors = F.normalize(last_token_pool(hidden, inputs["attention_mask"]).float(), p=2, dim=1)
        result = {"embeddings": vectors.cpu().tolist(), "output_shape": list(vectors.shape),
                  "pooling": "last_attended_token", "sdpa_backend": "default"}
        result.update(_receipt(inputs, start, self.device, include_inputs))
        return result


class Qwen3Reranker:
    def __init__(self, model_path, device="cuda", max_length=1024, *, instruction=DEFAULT_INSTRUCTION,
                 tokenizer=None, model=None):
        self.model_path, self.device, self.max_length = str(Path(model_path)), device, max_length
        self.instruction = instruction
        self.tokenizer, self.model = _load("AutoModelForCausalLM", model_path, device, tokenizer, model)
        self.tokenizer.padding_side = "left"
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.prefix_ids = self.tokenizer.encode(RERANK_PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(RERANK_SUFFIX, add_special_tokens=False)

    def score(self, pairs, include_inputs=False):
        start = _start(self.device)
        texts = [f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}" for query, doc in pairs]
        inputs = self.tokenizer(texts, padding=False, truncation="longest_first",
            return_attention_mask=False, return_token_type_ids=False,
            max_length=self.max_length-len(self.prefix_ids)-len(self.suffix_ids))
        inputs["input_ids"] = [self.prefix_ids + row + self.suffix_ids for row in inputs["input_ids"]]
        inputs = _to_device(self.tokenizer.pad(inputs, padding=True, return_tensors="pt"), self.device)
        with torch.inference_mode():
            # Qwen3 supports logits_to_keep; only the final prompt position is needed.
            logits = self.model(**inputs, logits_to_keep=1).logits.float()
            no_yes = logits[:, -1, [self.no_id, self.yes_id]]
            margins = no_yes[:, 1] - no_yes[:, 0]
            scores = F.softmax(no_yes, dim=1)[:, 1]
        result = {"logits": margins.cpu().tolist(), "scores": scores.cpu().tolist(),
            "output_shape": [len(pairs), 1], "raw_output_shape": list(logits.shape),
            "score_format": "conditional_yes_no", "logit_format": "yes_minus_no",
            "yes_token_id": self.yes_id, "no_token_id": self.no_id,
            "no_yes_logits": no_yes.cpu().tolist(), "sdpa_backend": "default"}
        result.update(_receipt(inputs, start, self.device, include_inputs))
        return result
