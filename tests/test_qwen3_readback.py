"""Receipt readback reconstructs the full Qwen3 prompt rather than raw pairs."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def __call__(self, texts, **kwargs):
        return {"input_ids": [self.encode(text)[:kwargs["max_length"]] for text in texts]}

    def pad(self, inputs, **kwargs):
        width = max(map(len, inputs["input_ids"]))
        return {"input_ids": [[0] * (width-len(row)) + row for row in inputs["input_ids"]],
                "attention_mask": [[0] * (width-len(row)) + [1] * len(row) for row in inputs["input_ids"]]}


def test_qwen3_readback_includes_preserved_system_and_assistant_suffix():
    namespace = runpy.run_path(str(ROOT / "scripts/readback_final.py"))
    assert "pair_encoding" in namespace, "Stack-aware pair receipt reconstruction is not implemented"
    pairs = [{"query_text": "質問", "document_text": "文書"},
             {"query_text": "質問二", "document_text": "長" * 2000}]
    pin = {"model_id": "Qwen/Qwen3-Reranker-0.6B", "conventions": {"instruction": "Retrieve relevant legal passages"}}
    result = namespace["pair_encoding"](CharacterTokenizer(), pairs, pin)
    assert len(result["input_ids"][0]) == len(result["input_ids"][1]) == 1024
    assert sum(result["attention_mask"][1]) == 1024
    first = "".join(chr(token) for token, attended in zip(result["input_ids"][0], result["attention_mask"][0]) if attended)
    assert first.startswith('<|im_start|>system\nJudge whether the Document')
    assert '<Instruct>: Retrieve relevant legal passages\n<Query>: 質問\n<Document>: 文書' in first
    for row, mask in zip(result["input_ids"], result["attention_mask"]):
        text = "".join(chr(token) for token, attended in zip(row, mask) if attended)
        assert text.endswith('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
