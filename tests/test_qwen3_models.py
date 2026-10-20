"""CPU tensors distinguish Qwen3 pooling and conditional yes/no scoring."""

import importlib
import math
from types import SimpleNamespace

import pytest
import torch


TASK = "Given a web search query, retrieve relevant passages that answer the query"


def runtime():
    try:
        return importlib.import_module("hidate.qwen3")
    except ModuleNotFoundError:
        pytest.fail("HiDate Qwen3 runtime is not implemented")


class EncoderTokenizer:
    def __init__(self, mask):
        self.mask = torch.tensor(mask)
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return {"input_ids": torch.arange(self.mask.numel()).reshape(self.mask.shape),
                "attention_mask": self.mask.clone()}


class Encoder(torch.nn.Module):
    def forward(self, **inputs):
        hidden = torch.tensor([[[99., 99.], [1., 0.], [3., 4.]],
                               [[0., 5.], [12., 5.], [99., 99.]]])
        return SimpleNamespace(last_hidden_state=hidden)


@pytest.mark.parametrize("mask,want", [
    ([[0, 1, 1], [1, 1, 1]], [[.6, .8], [1/math.sqrt(2), 1/math.sqrt(2)]]),
    ([[1, 1, 1], [1, 1, 0]], [[.6, .8], [12/13, 5/13]]),
])
def test_encoder_pools_last_attended_token_instead_of_mean_or_padding(mask, want):
    tokenizer = EncoderTokenizer(mask)
    wrapper = runtime().Qwen3Encoder("fixture", device="cpu", tokenizer=tokenizer, model=Encoder())
    result = wrapper.encode(["質問一", "質問二"], kind="query", include_inputs=True)
    for actual, expected in zip(result["embeddings"], want):
        assert actual == pytest.approx(expected)
    assert result["output_shape"] == [2, 2]
    assert result["total_tokens"] == 5 and result["token_counts"] == [sum(row) for row in mask]
    assert result["attention_mask"] == mask and result["sdpa_backend"] == "default"
    assert tokenizer.calls[0][0] == [f"Instruct: {TASK}\nQuery:質問一", f"Instruct: {TASK}\nQuery:質問二"]
    assert tokenizer.padding_side == "left"
    wrapper.encode(["文書一", "文書二"], kind="document")
    assert tokenizer.calls[1][0] == ["文書一", "文書二"]


class RerankerTokenizer:
    def __init__(self):
        self.calls = []
        self.encoded = []

    def encode(self, text, **kwargs):
        self.encoded.append(text)
        return [ord(char) for char in text]

    def convert_tokens_to_ids(self, token):
        return {"no": 1, "yes": 2}[token]

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return {"input_ids": [[ord(char) for char in text][:kwargs["max_length"]] for text in texts]}

    def pad(self, inputs, **kwargs):
        width = max(len(row) for row in inputs["input_ids"])
        ids = [[0] * (width-len(row)) + row for row in inputs["input_ids"]]
        mask = [[0] * (width-len(row)) + [1] * len(row) for row in inputs["input_ids"]]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


class Reranker(torch.nn.Module):
    def forward(self, **inputs):
        self.input_ids = inputs["input_ids"]
        self.logits_to_keep = inputs["logits_to_keep"]
        # The irrelevant vocabulary logit must not affect conditional P(yes).
        return SimpleNamespace(logits=torch.tensor([[[0., 3., 1., 1000.]], [[0., 1., 3., 1000.]]]))


def test_reranker_preserves_template_suffix_and_scores_yes_against_no_only():
    tokenizer, model = RerankerTokenizer(), Reranker()
    wrapper = runtime().Qwen3Reranker("fixture", device="cpu", max_length=512,
                                     tokenizer=tokenizer, model=model)
    result = wrapper.score([["質問一", "文書一"], ["質問二", "長文" * 1000]], include_inputs=True)
    assert result["logits"] == [-2., 2.]
    assert result["scores"] == pytest.approx([.119202922, .880797078])
    assert result["output_shape"] == [2, 1] and result["raw_output_shape"] == [2, 1, 4]
    assert result["score_format"] == "conditional_yes_no"
    assert result["yes_token_id"] == 2 and result["no_token_id"] == 1
    assert model.logits_to_keep == 1
    assert result["input_shape"] == [2, 512]
    assert result["total_tokens"] == sum(result["token_counts"])
    assert result["token_counts"][1] == 512
    assert result["attention_mask"][0][0] == 0
    assert tokenizer.calls[0][0][0] == f"<Instruct>: {TASK}\n<Query>: 質問一\n<Document>: 文書一"
    for ids, mask in zip(result["input_ids"], result["attention_mask"]):
        decoded = "".join(chr(token) for token, attended in zip(ids, mask) if attended)
        assert decoded.startswith('<|im_start|>system\nJudge whether the Document')
        assert decoded.endswith('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')


def test_factories_dispatch_each_stack_without_changing_ruri_results():
    from hidate import models
    assert hasattr(models, "create_encoder"), "HiDate model factories are not implemented"
    tokenizer = EncoderTokenizer([[0, 1, 1], [1, 1, 1]])
    qwen_pin = {"model_id": "Qwen/Qwen3-Embedding-0.6B", "runtime": "qwen3", "local_path": "fixture",
                "conventions": {"instruction": "Retrieve relevant Japanese legal provisions"}}
    wrapper = models.create_encoder(qwen_pin, device="cpu", tokenizer=tokenizer, model=Encoder())
    result = wrapper.encode(["質問一", "質問二"], kind="query")
    assert result["embeddings"][0] == pytest.approx([.6, .8])
    assert tokenizer.calls[0][0][0] == "Instruct: Retrieve relevant Japanese legal provisions\nQuery:質問一"
    ruri_pin = {"model_id": "cl-nagoya/ruri-v3-310m", "local_path": "fixture"}
    old = models.create_encoder(ruri_pin, device="cpu", tokenizer=tokenizer, model=Encoder())
    assert isinstance(old, models.RuriEncoder)
    reranker = models.create_reranker({"model_id": "Qwen/Qwen3-Reranker-0.6B", "local_path": "fixture"},
        device="cpu", tokenizer=RerankerTokenizer(), model=Reranker())
    assert reranker.score([["q", "d"], ["q2", "d2"]])["scores"] == pytest.approx([.119202922, .880797078])
