"""Small fixed-address Fiscal version-intervention and exact-cache fixtures."""

import copy
import json
from pathlib import Path
import runpy

import pytest

from hidate.io import load_jsonl, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
PIN = {"model_id": "fixture/reader", "revision": "frozen"}
DECODE = {"do_sample": False, "max_new_tokens": 192}


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return self.encode(json.dumps(messages, ensure_ascii=False) + "<assistant>")


def version(vid, cid, start, end, text, num="156"):
    return {"id": vid, "cid": cid, "date_debut": start, "date_fin": end,
            "etat": "MODIFIE" if end else "VIGUEUR", "texte_clean": text,
            "num": num, "code_id": "LEGITEXT000006069577"}


@pytest.fixture
def inputs():
    versions = {v["id"]: v for v in [
        version("a-oldest", "A", "1990-01-01", "1999-12-31", "Very old rule 100 francs."),
        version("a-source", "A", "2000-01-01", "2009-12-31", "Historical rule 60000 euros. " * 100),
        version("a-bound", "A", "2010-01-01", None, "Applicable rule 101300 euros. " * 100),
        version("b-bound", "B", "2000-01-01", None, "Other provision. " * 100, num="158")]}
    question = {"qid": "Q1", "regime": "R3", "scope": "in", "prompt": "What is the amount?",
                "date_anchor": "2020-01-01", "gold": {"article_cid": "A", "version_id": "a-bound"}}
    retrieval = {"qid": "Q1", "date_anchor": question["date_anchor"], "scope": "in",
        "addresses": [
            {"article_cid": "A", "bound_version_id": "a-bound", "features": {"max_history_rrf": .03},
             "history_chunk_ids": ["a-source|1250", "a-oldest|0", "a-bound|0"]},
            {"article_cid": "B", "bound_version_id": "b-bound", "features": {"max_history_rrf": .02},
             "history_chunk_ids": ["b-bound|0"]}],
        "conditions": {"history_learned": {"evidence": [
            {"article_cid": "A", "version_id": "a-bound"},
            {"article_cid": "B", "version_id": "b-bound"}],
            "accepted": True, "support_confidence": .99, "support_threshold": .97}}}
    return question, retrieval, versions


def test_freeze_raw_final_cids_order_and_choose_top_fused_source_not_oldest(inputs):
    from hidate.binding_intervention import freeze_addresses
    q, retrieval, versions = inputs
    slots = freeze_addresses(q, retrieval, versions)
    assert [s["article_cid"] for s in slots] == ["A", "B"]
    assert slots[0]["source_version_id"] == "a-source"
    assert slots[0]["source_history_chunk_id"] == "a-source|1250"
    assert slots[0]["bound_version_id"] == "a-bound"
    changed = {**q, "gold": {"article_cid": "WRONG", "version_id": "WRONG"}}
    assert freeze_addresses(changed, retrieval, versions) == slots


def test_joint_slot_cap_keeps_all_raw_addresses_order_and_literal_prefixes(inputs):
    from hidate.binding_intervention import freeze_addresses, pack_paired_versions, CONDITIONS
    q, retrieval, versions = inputs
    original = copy.deepcopy(versions)
    slots = freeze_addresses(q, retrieval, versions)
    packed = pack_paired_versions(q["prompt"], slots, versions, Tokenizer(), prompt_budget=1500)
    caps = set()
    for condition in CONDITIONS:
        arm = packed[condition]
        assert [e["article_cid"] for e in arm["evidence"]] == ["A", "B"]
        assert arm["prompt_tokens"] <= 1500 and len(arm["input_ids"]) == arm["prompt_tokens"]
        caps.add(arm["slot_body_token_cap"])
        for e in arm["evidence"]:
            row = versions[e["version_id"]]
            assert row["texte_clean"].startswith(e["text"].split("===\n", 1)[1])
    assert len(caps) == 1 and versions == original
    assert packed["binding_source"]["evidence"][0]["version_id"] == "a-source"
    assert packed["binding_bound"]["evidence"][0]["version_id"] == "a-bound"


def test_exact_cache_identity_includes_rendered_inputs_pin_and_decode(inputs):
    from hidate.binding_intervention import exact_input_key
    row = {"messages": [{"role": "user", "content": "same"}], "input_ids": [1, 2], "attention_mask": [1, 1],
           "reader_pin": PIN, "decode": DECODE}
    key = exact_input_key(row)
    assert exact_input_key(copy.deepcopy(row)) == key
    assert exact_input_key({**row, "input_ids": [1, 3]}) != key
    assert exact_input_key({**row, "reader_pin": {**PIN, "revision": "other"}}) != key
    assert exact_input_key({**row, "decode": {**DECODE, "max_new_tokens": 191}}) != key


def test_source_revision_cannot_be_future_or_missing_in_history(inputs):
    from hidate.binding_intervention import freeze_addresses
    q, retrieval, versions = inputs
    versions["a-source"]["date_debut"] = "2030-01-01"
    with pytest.raises(ValueError, match="eligible"):
        freeze_addresses(q, retrieval, versions)


def setup_files(tmp_path, inputs):
    from hidate.binding_intervention import freeze_addresses, pack_paired_versions
    q, retrieval, versions = inputs
    prepared, retrieved, original = [tmp_path / p for p in ("prepared", "retrieved", "original")]
    write_jsonl(prepared / "articles.jsonl", versions.values())
    write_jsonl(prepared / "questions.jsonl", [q])
    write_jsonl(prepared / "nuggets.jsonl", [{"qid": "Q1", "kind": "text", "required": True, "nugget_id": "n1", "text": "101300"}])
    write_json(retrieved / "retrieval.json", {"complete": True, "alpha": 1.0})
    write_jsonl(retrieved / "retrieval.jsonl", [retrieval])
    fiscal = {"reader_prompt_tokens": 1500, "reader_max_passages": 4, "reader_max_new_tokens": 192}
    write_json(tmp_path / "fiscal.json", fiscal)
    write_json(tmp_path / "stack.json", {"complete": True, "models": {"reader": {**PIN, "local_path": "fixture"}}})
    write_json(original / "reader.json", {"complete": True, "reader_pin": PIN, "decode": DECODE})
    packed = pack_paired_versions(q["prompt"], freeze_addresses(q, retrieval, versions), versions, Tokenizer(), prompt_budget=1500)
    bound = packed["binding_bound"]
    old_result = {"text": "101300", "prompt_tokens": bound["prompt_tokens"], "generated_tokens": 1,
                  "max_new_tokens": 192, "end_status": "eos", "seconds": 7., "generated_token_ids": [9]}
    write_jsonl(original / "reader-receipts.jsonl", [{"qid": "Q1", "condition": "history_learned", "model": PIN["model_id"],
                "messages": bound["messages"], "result": old_result}])
    # Original sequential packing had dropped B; raw-final freezes A,B nevertheless.
    write_jsonl(original / "responses.jsonl", [{"qid": "Q1", "condition": "history_learned", "evidence": bound["evidence"][:1]}])
    scorer = tmp_path / "official/scripts/score_nuggets.py"
    scorer.parent.mkdir(parents=True)
    scorer.write_text('def normalize(s): return s\ndef parse_numbers(s): return []\ndef match_nugget(n,s,nums): return n["text"] in s\n', encoding="utf-8")
    return dict(prepared_dir=prepared, retrieved_dir=retrieved, original_reader_dir=original,
                stack_path=tmp_path / "stack.json", fiscal_protocol_path=tmp_path / "fiscal.json",
                output_dir=tmp_path / "output", expected_questions=1, tokenizer=Tokenizer()), scorer


def test_prepare_reuses_only_exact_old_prompts_and_reader_runs_only_missing_inputs(tmp_path, inputs):
    script = runpy.run_path(str(ROOT / "scripts/run_binding_intervention.py"))
    args, scorer = setup_files(tmp_path, inputs)
    preparation = script["prepare_binding"](**args)
    assert preparation["complete"] and preparation["old_exact_cache_hits"] == 1
    assert preparation["pending_unique_prompts"] == 1
    assert preparation["raw_vs_original_packed"]["different_cid_lists"] == 1
    frozen = load_jsonl(args["output_dir"] / "frozen-addresses.jsonl")[0]
    assert [s["article_cid"] for s in frozen["slots"]] == ["A", "B"]

    class Reader:
        calls = 0

        def generate_paraphrase(self, messages, max_new_tokens, include_inputs):
            assert include_inputs and max_new_tokens == 192
            self.calls += 1
            ids = Tokenizer().apply_chat_template(messages)
            return {"text": "60000", "prompt_tokens": len(ids), "generated_tokens": 1, "end_status": "eos",
                    "max_new_tokens": 192, "seconds": .5, "input_ids": [ids], "attention_mask": [[1] * len(ids)],
                    "generated_token_ids": [6]}

    reader = Reader()
    report = script["run_reader"](args["output_dir"], args["stack_path"], reader_factory=lambda pin, budget: reader)
    assert report["complete"] and report["new_physical_reader_calls_this_job"] == 1 and reader.calls == 1
    responses = load_jsonl(args["output_dir"] / "responses.jsonl")
    assert len(responses) == 2 and all([e["article_cid"] for e in r["evidence"]] == ["A", "B"] for r in responses)
    again = script["run_reader"](args["output_dir"], args["stack_path"], reader_factory=lambda *a: pytest.fail("Resume loaded a model"))
    assert again["complete"] and again["new_physical_reader_calls_this_job"] == 0
    assert again["new_physical_reader_calls_total"] == 1
    script["prepare_binding"](**args)
    assert len(load_jsonl(args["output_dir"] / "responses.jsonl")) == 2, "Completed preparation must not erase reader progress"
    scores = script["score_binding"](args["output_dir"], args["prepared_dir"], scorer, expected_questions=1)
    assert scores["complete"] and scores["paired_questions"] == 1
    assert scores["conditions"]["binding_bound"]["strict"] == 1
    assert scores["conditions"]["binding_source"]["strict"] == 0


def test_wrong_old_reader_revision_is_not_an_exact_cache_source(tmp_path, inputs):
    prepare = runpy.run_path(str(ROOT / "scripts/run_binding_intervention.py"))["prepare_binding"]
    args, _ = setup_files(tmp_path, inputs)
    write_json(args["original_reader_dir"] / "reader.json", {"complete": True, "reader_pin": {**PIN, "revision": "other"}, "decode": DECODE})
    result = prepare(**args)
    assert result["old_exact_cache_hits"] == 0 and result["pending_unique_prompts"] == 2


def test_identical_arms_share_one_generation_and_retain_same_version_question(tmp_path, inputs):
    script = runpy.run_path(str(ROOT / "scripts/run_binding_intervention.py"))
    q, retrieval, versions = inputs
    retrieval["addresses"][0]["history_chunk_ids"] = ["a-bound|0"]
    args, _ = setup_files(tmp_path, (q, retrieval, versions))
    # Force old cache incompatibility so same-arm reuse must occur in this run.
    write_json(args["original_reader_dir"] / "reader.json", {"complete": True,
               "reader_pin": {**PIN, "revision": "different"}, "decode": DECODE})
    prepared = script["prepare_binding"](**args)
    assert prepared["same_version_pairs"] == prepared["identical_rendered_prompt_pairs"] == 1
    assert prepared["pending_logical_outputs"] == 2 and prepared["pending_unique_prompts"] == 1

    class Reader:
        calls = 0

        def generate_paraphrase(self, messages, **kwargs):
            self.calls += 1; ids = Tokenizer().apply_chat_template(messages)
            return {"text": "101300", "prompt_tokens": len(ids), "generated_tokens": 1, "end_status": "eos",
                    "seconds": .25, "input_ids": [ids], "attention_mask": [[1] * len(ids)]}

    reader = Reader()
    result = script["run_reader"](args["output_dir"], args["stack_path"], reader_factory=lambda *a: reader)
    assert result["complete"] and result["new_physical_reader_calls_total"] == reader.calls == 1
    assert result["work_this_job"]["same_run_prompt_reuses"] == 1


def test_empty_fixed_list_has_same_closed_book_input_without_dropping_question(inputs):
    from hidate.binding_intervention import pack_paired_versions, CONDITIONS
    q, _, versions = inputs
    arms = pack_paired_versions(q["prompt"], [], versions, Tokenizer(), prompt_budget=1500)
    assert arms[CONDITIONS[0]]["messages"] == arms[CONDITIONS[1]]["messages"]
    assert all(a["evidence"] == [] for a in arms.values())
