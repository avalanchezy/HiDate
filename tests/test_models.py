"""Small CPU tensors verify HiDate model interfaces without loading model weights."""

import importlib
import math
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch


def runtime():
    try:
        return importlib.import_module("hidate.models")
    except ModuleNotFoundError:
        pytest.fail("HiDate fixed-model runtime is not implemented")


class TokenizerFixture:
    def __init__(self, mask=None):
        self.mask = torch.tensor(mask if mask is not None else [[1, 1, 0], [1, 1, 1]])
        self.calls = []
        self.template_calls = []

    def __call__(self, texts, text_pair=None, **kwargs):
        self.calls.append((list(texts), None if text_pair is None else list(text_pair), kwargs))
        return {"input_ids": torch.arange(self.mask.numel()).reshape(self.mask.shape),
                "attention_mask": self.mask.clone()}

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        return {"input_ids": torch.arange(self.mask.numel()).reshape(self.mask.shape),
                "attention_mask": self.mask.clone()}

    def encode(self, text, **kwargs):
        assert kwargs["add_special_tokens"] is False
        return [{"a": 2, "b": 3, "c": 4, "d": 5}[text]]

    def decode(self, ids, **kwargs):
        assert list(ids) == [6, 7]
        assert kwargs["skip_special_tokens"] is True
        return "短い言い換え。"


class EncoderFixture(torch.nn.Module):
    def forward(self, **inputs):
        self.mask = inputs["attention_mask"].clone()
        return SimpleNamespace(last_hidden_state=torch.tensor([
            [[1.0, 0.0], [0.0, 2.0], [99.0, 99.0]],
            [[0.0, 1.0], [0.0, 1.0], [2.0, 0.0]],
        ]))


class RerankerFixture(torch.nn.Module):
    def __init__(self, logits=None):
        super().__init__()
        self.values = torch.tensor(logits if logits is not None else [[-2.0], [2.0]])

    def forward(self, **inputs):
        self.mask = inputs["attention_mask"].clone()
        return SimpleNamespace(logits=self.values)


class ReaderFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, **inputs):
        self.calls += 1
        self.mask = inputs["attention_mask"].clone()
        logits = torch.zeros(1, 4, 8)
        logits[0, 2, 2:5] = torch.tensor([3.0, 2.0, 1.0])
        logits[0, 3, 5] = 100.0  # Padding position would incorrectly choose d.
        return SimpleNamespace(logits=logits)

    def generate(self, **kwargs):
        self.generation = kwargs
        return torch.cat([kwargs["input_ids"], torch.tensor([[6, 7]])], dim=1)


def test_encoder_masks_padding_means_all_attended_tokens_and_normalizes():
    tokenizer, model = TokenizerFixture(), EncoderFixture()
    encoder = runtime().RuriEncoder("fixture", device="cpu", tokenizer=tokenizer, model=model)
    result = encoder.encode(["質問一", "質問二"], kind="query", include_inputs=True)
    assert result["embeddings"][0] == pytest.approx([1 / math.sqrt(5), 2 / math.sqrt(5)])
    assert result["embeddings"][1] == pytest.approx([1 / math.sqrt(2), 1 / math.sqrt(2)])
    assert result["token_counts"] == [2, 3] and result["total_tokens"] == 5
    assert result["attention_mask"] == [[1, 1, 0], [1, 1, 1]]
    assert tokenizer.calls[0][0] == ["検索クエリ: 質問一", "検索クエリ: 質問二"]
    assert tokenizer.calls[0][2]["max_length"] == 1024
    encoder.encode(["文書一", "文書二"], kind="document")
    assert tokenizer.calls[1][0] == ["検索文書: 文書一", "検索文書: 文書二"]
    assert result["seconds"] >= 0


def test_reranker_uses_raw_pairs_single_logit_sigmoid_and_actual_masks():
    tokenizer, model = TokenizerFixture(), RerankerFixture()
    reranker = runtime().RuriReranker("fixture", device="cpu", tokenizer=tokenizer, model=model)
    result = reranker.score([["質問一", "文書一"], ["質問二", "文書二"]], include_inputs=True)
    assert tokenizer.calls[0][:2] == (["質問一", "質問二"], ["文書一", "文書二"])
    assert result["logits"] == [-2.0, 2.0]
    assert result["scores"] == pytest.approx([0.119202922, 0.880797078])
    assert result["token_counts"] == [2, 3] and result["total_tokens"] == 5
    assert result["attention_mask"] == model.mask.tolist()


def test_reranker_rejects_two_logit_output_instead_of_scoring_wrong_class():
    reranker = runtime().RuriReranker("fixture", device="cpu", tokenizer=TokenizerFixture(),
                                    model=RerankerFixture([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError, match="one logit"):
        reranker.score([["質問一", "文書一"], ["質問二", "文書二"]])


def test_reader_scores_choice_tokens_at_last_attended_prompt_position():
    tokenizer, model = TokenizerFixture([[0, 1, 1, 0]]), ReaderFixture()
    reader = runtime().QwenReader("fixture", device="cpu", tokenizer=tokenizer, model=model)
    messages = [{"role": "user", "content": "a. 月曜日 b. 赤色 c. 三角形 d. 東京"}]
    result = reader.score_choices(messages, include_inputs=True)
    assert result["prediction"] == "a"
    assert result["choice_token_ids"] == {"a": 2, "b": 3, "c": 4, "d": 5}
    assert result["choice_position"] == 2
    assert result["prompt_tokens"] == 2 and result["total_tokens"] == 2
    assert result["choice_logprobs"]["a"] - result["choice_logprobs"]["b"] == pytest.approx(1.0)
    assert sum(math.exp(value) for value in result["choice_logprobs"].values()) < 1.0
    assert tokenizer.template_calls[0][1]["add_generation_prompt"] is True
    assert tokenizer.template_calls[0][1]["return_dict"] is True
    assert result["attention_mask"] == model.mask.tolist()


def test_reader_over_budget_preserves_prompt_and_never_runs_model():
    tokenizer, model = TokenizerFixture([[1] * 4097]), ReaderFixture()
    reader = runtime().QwenReader("fixture", device="cpu", tokenizer=tokenizer, model=model)
    messages = [{"role": "user", "content": "問題と四つの選択肢を含む全文"}]
    with pytest.raises(ValueError, match="4096"):
        reader.score_choices(messages)
    assert tokenizer.template_calls[0][0] == messages
    assert model.calls == 0


def test_reader_greedy_generation_counts_new_tokens_and_forwards_attention_mask():
    tokenizer, model = TokenizerFixture([[0, 1, 1, 0]]), ReaderFixture()
    tokenizer.eos_token_id = 8
    model.generation_config = SimpleNamespace(eos_token_id=[7, 9])
    reader = runtime().QwenReader("fixture", device="cpu", tokenizer=tokenizer, model=model)
    result = reader.generate_paraphrase([{"role": "user", "content": "条文を短く言い換えてください。"}],
                                       max_new_tokens=2, include_inputs=True)
    assert model.generation["do_sample"] is False
    assert model.generation["max_new_tokens"] == 2
    assert model.generation["attention_mask"].tolist() == [[0, 1, 1, 0]]
    assert result["generated_token_ids"] == [6, 7]
    assert result["text"] == "短い言い換え。"
    assert result["prompt_tokens"] == 2 and result["generated_tokens"] == 2
    assert result["total_tokens"] == 4
    assert result["ended_with_eos"] is True
    assert result["end_status"] == "eos"


@pytest.mark.parametrize("role", ["embedding", "reranker"])
def test_ruri_cuda_forward_runs_inside_uniform_math_sdpa_context(monkeypatch, role):
    module = runtime()
    active, entered = [], []

    @contextmanager
    def selected_backend(backends):
        assert backends == [torch.nn.attention.SDPBackend.MATH]
        active.extend(backends)
        entered.append(list(backends))
        try:
            yield
        finally:
            active.clear()

    monkeypatch.setattr(module, "sdpa_kernel", selected_backend, raising=False)
    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(module, "_to_device", lambda inputs, device: inputs)
    fixture = EncoderFixture() if role == "embedding" else RerankerFixture()
    original_forward = fixture.forward

    def checked_forward(**inputs):
        assert active == [torch.nn.attention.SDPBackend.MATH], "CUDA ModernBERT forward must force MATH SDPA"
        return original_forward(**inputs)

    fixture.forward = checked_forward
    if role == "embedding":
        wrapper = module.RuriEncoder("fixture", device="cpu", tokenizer=TokenizerFixture(), model=fixture)
        wrapper.device = "cuda"
        result = wrapper.encode(["文書一", "文書二"])
    else:
        wrapper = module.RuriReranker("fixture", device="cpu", tokenizer=TokenizerFixture(), model=fixture)
        wrapper.device = "cuda"
        result = wrapper.score([["質問一", "文書一"], ["質問二", "文書二"]])
    assert entered == [[torch.nn.attention.SDPBackend.MATH]]
    assert active == []
    assert result["sdpa_backend"] == "math"
    assert result["token_counts"] == [2, 3] and result["total_tokens"] == 5


def test_cpu_ruri_keeps_default_backend_and_records_it(monkeypatch):
    module = runtime()

    def forbidden_math_context(*args, **kwargs):
        pytest.fail("CPU Ruri should keep its default attention backend")

    monkeypatch.setattr(module, "sdpa_kernel", forbidden_math_context, raising=False)
    wrapper = module.RuriEncoder("fixture", device="cpu", tokenizer=TokenizerFixture(), model=EncoderFixture())
    result = wrapper.encode(["文書一", "文書二"])
    assert result.get("sdpa_backend") == "default"


def test_qwen_cuda_keeps_default_sdpa_policy(monkeypatch):
    module = runtime()

    def forbidden_math_context(*args, **kwargs):
        pytest.fail("Qwen CUDA forward must not use the ModernBERT-specific MATH context")

    monkeypatch.setattr(module, "sdpa_kernel", forbidden_math_context, raising=False)
    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(module, "_to_device", lambda inputs, device: inputs)
    original_arange = module.torch.arange

    def cpu_arange(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_arange(*args, **kwargs)

    monkeypatch.setattr(module.torch, "arange", cpu_arange)
    wrapper = module.QwenReader("fixture", device="cpu", tokenizer=TokenizerFixture([[0, 1, 1, 0]]), model=ReaderFixture())
    wrapper.device = "cuda"
    result = wrapper.score_choices([{"role": "user", "content": "a. 月曜日 b. 赤色 c. 三角形 d. 東京"}])
    assert result["prediction"] == "a"
