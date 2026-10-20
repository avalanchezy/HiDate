"""Small CPU-only boundaries for complete query generation and encoding drivers."""

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from hidate.io import load_jsonl, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[1]


def functions(name):
    path = ROOT / "scripts" / name
    assert path.exists(), f"HiDate query driver is not implemented: {name}"
    return runpy.run_path(str(path))


@pytest.fixture
def source_files(tmp_path):
    exact = tmp_path / "queries-exact.jsonl"
    queries = [{"query_id": f"SECRET_QUERY_{i}", "query_variant": "exact_quote", "query_text": text,
        "source_quote_text": text, "source_body_text": text + "入力してはならない後続本文。",
        "law_title": "入力してはならない法名", "law_id": "SECRET_LAW", "target_asof": "2099-01-01",
        "source_role": "full", "split": "train" if i == 0 else "test",
        "labels": {"target_key": ["SECRET_LAW", "2099-01-01", "MainProvision", str(i)],
                   "revision_id": "SECRET_REV", "passage_id": "SECRET_PASSAGE"}}
        for i, text in enumerate(["申請には本人確認が必要である。", "手数料を徴収してはならない。", "申請には本人確認が必要である。"])]
    for query, count in zip(queries, [2, 1, 2]):
        query["query_text_occurrences"] = count
    write_jsonl(exact, queries)
    stack = tmp_path / "stack.json"
    write_json(stack, {"complete": True, "models": {
        "reader": {"model_id": "Qwen/Qwen2.5-7B-Instruct", "revision": "READER_PIN", "local_path": "fixture-reader", "download_complete": True},
        "embedding": {"model_id": "cl-nagoya/ruri-v3-310m", "revision": "RURI_PIN", "local_path": "fixture-ruri", "download_complete": True}}})
    return exact, stack, queries


class Reader:
    def __init__(self, empty=False):
        self.calls = []
        self.empty = empty

    def generate_paraphrase(self, messages, max_new_tokens, include_inputs):
        assert max_new_tokens == 128 and include_inputs is True
        self.calls.append(messages)
        return {"text": " " if self.empty else "この引用で求められる手続について調べたい。",
            "input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]], "input_shape": [1, 3],
            "generated_token_ids": [4, 9], "generated_tokens": 2, "prompt_tokens": 3,
            "total_tokens": 5, "seconds": .125, "ended_with_eos": True, "end_status": "eos",
            "max_new_tokens": 128, "max_new_tokens_reached": False, "output_shape": [1, 5]}


def test_unique_source_quote_generation_never_uses_gold_or_unquoted_body(source_files, tmp_path):
    exact, stack, queries = source_files
    reader = Reader()
    output, run = tmp_path / "paraphrases.jsonl", tmp_path / "run"
    manifest = functions("generate_paraphrases.py")["generate_paraphrases"](
        exact, stack, output, run, reader=reader)
    assert manifest["complete"] is True
    assert len(reader.calls) == 2
    assert [call[1]["content"] for call in reader.calls] == [
        "引用:\n申請には本人確認が必要である。", "引用:\n手数料を徴収してはならない。"]
    assert all("SECRET" not in json.dumps(call) and "2099-01-01" not in json.dumps(call) for call in reader.calls)
    mapped = load_jsonl(output)
    assert len(mapped) == 3
    assert [row["base_query_id"] for row in mapped] == [row["query_id"] for row in queries]
    assert [row["query_id"] for row in mapped] == ["SECRET_QUERY_0|paraphrase", "SECRET_QUERY_1|paraphrase", "SECRET_QUERY_2|paraphrase"]
    assert all(row["query_variant"] == "generated_paraphrase" for row in mapped)
    assert [row["labels"] for row in mapped] == [row["labels"] for row in queries]
    assert [row["split"] for row in mapped] == ["train", "test", "test"]
    assert [row["query_text_occurrences"] for row in mapped] == [2, 1, 2]
    assert [row.get("generated_query_text_occurrences") for row in mapped] == [3, 3, 3]
    generations = load_jsonl(run / "generations.jsonl")
    assert len(generations) == 2
    assert generations[0]["result"]["generated_token_ids"] == [4, 9]
    assert generations[0]["result"]["ended_with_eos"] is True
    assert manifest["unique_source_quotes"] == 2 and manifest["mapped_queries"] == 3
    assert manifest["prompt_tokens"] == 6 and manifest["generated_tokens"] == 4
    assert manifest["model_token_work"] == 10
    assert manifest["distinct_generated_query_texts"] == 1


def test_generation_requires_explicit_same_pin_resume_and_reuses_completed_quotes(source_files, tmp_path):
    exact, stack, _ = source_files
    generate = functions("generate_paraphrases.py")["generate_paraphrases"]
    output, run = tmp_path / "paraphrases.jsonl", tmp_path / "run"
    generate(exact, stack, output, run, reader=Reader())
    with pytest.raises(ValueError, match="resume"):
        generate(exact, stack, output, run, reader=Reader())
    resumed_reader = Reader()
    result = generate(exact, stack, output, run, reader=resumed_reader, resume=True)
    assert result["complete"] is True and resumed_reader.calls == []
    assert len(load_jsonl(run / "generations.jsonl")) == 2
    changed_stack = json.loads(stack.read_text())
    changed_stack["models"]["reader"]["revision"] = "CHANGED_PIN"
    write_json(stack, changed_stack)
    with pytest.raises(ValueError, match="pin|protocol"):
        generate(exact, stack, output, run, reader=Reader(), resume=True)


def test_empty_generation_is_saved_as_error_without_quote_fallback_or_regeneration(source_files, tmp_path):
    exact, stack, _ = source_files
    generate = functions("generate_paraphrases.py")["generate_paraphrases"]
    output, run = tmp_path / "paraphrases.jsonl", tmp_path / "run"
    result = generate(exact, stack, output, run, reader=Reader(empty=True))
    assert result["complete"] is False and result["generation_attempts_complete"] is True
    mapped = load_jsonl(output)
    assert len(mapped) == 3 and all(row["generation_status"] == "empty_output" for row in mapped)
    assert all(row["query_text"] == " " for row in mapped)
    assert result["empty_outputs"] == 2
    resumed_reader = Reader()
    generate(exact, stack, output, run, reader=resumed_reader, resume=True)
    assert resumed_reader.calls == []


def test_explicit_resume_continues_after_saved_model_error(source_files, tmp_path):
    exact, stack, _ = source_files
    generate = functions("generate_paraphrases.py")["generate_paraphrases"]
    output, run = tmp_path / "paraphrases.jsonl", tmp_path / "run"

    class InterruptedReader(Reader):
        def generate_paraphrase(self, messages, max_new_tokens, include_inputs):
            if len(self.calls) == 1:
                raise RuntimeError("fixture interrupted generation")
            return super().generate_paraphrase(messages, max_new_tokens, include_inputs)

    with pytest.raises(RuntimeError, match="interrupted"):
        generate(exact, stack, output, run, reader=InterruptedReader())
    assert load_jsonl(run / "generations.jsonl")[-1]["status"] == "model_error"
    resumed = Reader()
    result = generate(exact, stack, output, run, reader=resumed, resume=True)
    assert result["complete"] is True and len(resumed.calls) == 1
    assert len(load_jsonl(run / "generations.jsonl")) == 3
    assert len(load_jsonl(output)) == 3


def test_passed_preflight_accepts_original_pin_receipts_with_local_path(source_files, tmp_path):
    exact, stack, _ = source_files
    stack_record = json.loads(stack.read_text())
    preflight = tmp_path / "preflight.json"
    write_json(preflight, {"status": "PASS", "model_pins": {
        role: {key: record[key] for key in ("model_id", "revision", "local_path")}
        for role, record in stack_record["models"].items()}})
    result = functions("generate_paraphrases.py")["generate_paraphrases"](
        exact, stack, tmp_path / "paraphrases.jsonl", tmp_path / "run", reader=Reader(), preflight_path=preflight)
    assert result["complete"] is True
    encoded = functions("encode_queries.py")["encode_queries"](
        [exact], stack, tmp_path / "vectors", encoder=Encoder(), preflight_path=preflight)
    assert encoded["complete"] is True


class Encoder:
    def __init__(self, invalid=False):
        self.calls = []
        self.invalid = invalid

    def encode(self, texts, kind, include_inputs=False):
        assert kind == "query" and include_inputs is False
        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), 768), dtype=np.float32)
        for row in range(len(texts)):
            vectors[row, row] = np.nan if self.invalid else 1.
        return {"embeddings": vectors.tolist(), "output_shape": list(vectors.shape),
            "token_counts": [4 + i for i in range(len(texts))],
            "total_tokens": sum(4 + i for i in range(len(texts))), "input_shape": [len(texts), 6],
            "seconds": .25, "sdpa_backend": "math"}


def test_query_encoding_reuses_exact_text_and_preserves_all_aliases(source_files, tmp_path):
    exact, stack, queries = source_files
    generated = tmp_path / "generated.jsonl"
    write_jsonl(generated, [{**queries[0], "query_id": "PARA_0", "base_query_id": queries[0]["query_id"],
                            "query_variant": "generated_paraphrase", "source_role": "external"}])
    encoder = Encoder()
    output = tmp_path / "vectors"
    result = functions("encode_queries.py")["encode_queries"]([exact, generated], stack, output, encoder=encoder)
    assert encoder.calls == [["申請には本人確認が必要である。", "手数料を徴収してはならない。"]]
    assert json.loads((output / "query-ids.json").read_text()) == ["SECRET_QUERY_0", "SECRET_QUERY_1", "SECRET_QUERY_2", "PARA_0"]
    assert json.loads((output / "query-texts.json").read_text()) == ["申請には本人確認が必要である。", "手数料を徴収してはならない。"]
    assert np.load(output / "vector-row-indices.npy").tolist() == [0, 1, 0, 0]
    vectors = np.load(output / "vectors.npy")
    assert vectors.shape == (2, 768) and vectors[0, 0] == vectors[1, 1] == 1
    assert result["complete"] is True and result["query_rows"] == 4
    assert result["unique_model_texts"] == 2 and result["encoded_input_tokens"] == 9
    assert result["padded_token_slots"] == 12 and result["sdpa_backend"] == "math"
    assert result["query_metadata"][-1]["query_variant"] == "generated_paraphrase"
    assert result["query_metadata"][-1]["source_role"] == "external"


def test_exact_variant_encoder_does_not_require_unfinished_paraphrases(source_files, tmp_path):
    exact, stack, _ = source_files
    output = tmp_path / "vectors"
    result = functions("encode_queries.py")["encode_queries"]([exact], stack, output, encoder=Encoder(), variant="exact_quote")
    assert result["complete"] is True and result["query_rows"] == 3
    assert result["variant_counts"] == {"exact_quote": 3}


def test_query_encoder_rejects_nonfinite_vectors_without_completion(source_files, tmp_path):
    exact, stack, _ = source_files
    output = tmp_path / "vectors"
    with pytest.raises(ValueError, match="nonfinite"):
        functions("encode_queries.py")["encode_queries"]([exact], stack, output, encoder=Encoder(invalid=True))
    assert json.loads((output / "encoding.json").read_text())["complete"] is False


@pytest.mark.parametrize("script", ["generate_paraphrases.py", "encode_queries.py"])
def test_model_driver_runs_locally_with_fixture_models(script, source_files, tmp_path, monkeypatch):
    exact, stack, queries = source_files
    stack_record = json.loads(stack.read_text())
    preflight = tmp_path / "preflight.json"
    write_json(preflight, {"status": "PASS", "model_pins": {
        role: {key: record[key] for key in ("model_id", "revision", "local_path")}
        for role, record in stack_record["models"].items()}})
    driver = functions(script)
    argv = [script, "--queries", str(exact), "--stack", str(stack), "--preflight", str(preflight)]
    if script == "generate_paraphrases.py":
        output, run = tmp_path / "generated.jsonl", tmp_path / "generation"
        reader = Reader()
        generate = driver["generate_paraphrases"]
        monkeypatch.setitem(driver["main"].__globals__, "generate_paraphrases",
            lambda *args, **kwargs: generate(*args, **kwargs, reader=reader))
        argv += ["--output", str(output), "--run-dir", str(run)]
    else:
        output = tmp_path / "encoded"
        encoder = Encoder()
        encode = driver["encode_queries"]
        monkeypatch.setitem(driver["main"].__globals__, "encode_queries",
            lambda *args, **kwargs: encode(*args, **kwargs, encoder=encoder))
        argv += ["--output-dir", str(output), "--variant", "exact_quote"]
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.setattr(sys, "argv", argv)
    assert driver["main"]() == 0
    if script == "generate_paraphrases.py":
        assert len(reader.calls) == 2
        assert [row["base_query_id"] for row in load_jsonl(output)] == [q["query_id"] for q in queries]
        assert json.loads((run / "generation.json").read_text())["complete"] is True
    else:
        assert len(encoder.calls) == 1
        assert np.load(output / "vector-row-indices.npy").tolist() == [0, 1, 0]
        assert json.loads((output / "query-ids.json").read_text()) == [q["query_id"] for q in queries]
        assert json.loads((output / "encoding.json").read_text())["complete"] is True
