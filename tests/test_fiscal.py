"""Fiscal external tests focus on temporal identity and scoring errors."""

import importlib.util
from pathlib import Path

import numpy as np

from hidate import fiscal as fiscal


def version(identity, start, end, text, *, cid="a", state="MODIFIE"):
    return {"id": identity, "cid": cid, "code_id": fiscal.CGI, "num": "156",
            "date_debut": start, "date_fin": end, "etat": state, "texte_clean": text}


def test_asof_uses_inclusive_end_and_longest_same_day_version():
    rows = [version("old", "2000-01-01", "2010-01-01", "ancien"),
            version("zero", "2010-01-01", "2010-01-01", "éphémère"),
            version("new", "2010-01-01", None, "nouveau"),
            version("stillborn", "2010-01-01", None, "invalide", state="MODIFIE_MORT_NE")]
    assert fiscal.select_versions(rows, "2010-01-01")["a"]["id"] == "new"
    assert fiscal.select_versions(rows[:1], "2010-01-01")["a"]["id"] == "old"
    assert fiscal.select_versions(rows[:1], "2010-01-02") == {}


def test_historical_binding_replaces_discovered_old_text_and_excludes_future():
    rows = [version("old", "2000-01-01", "2010-01-01", "agricoles déficits ancien"),
            version("new", "2010-01-01", None, "revenus nouveau"),
            version("future", "2030-01-01", None, "agricoles déficits futur"),
            version("distractor", "2000-01-01", None, "autre", cid="b")]
    chunks = fiscal.article_chunks(rows, size=1500, overlap=250)
    vectors = np.array([[1., 0.], [.1, .995], [1., 0.], [0., 1.]], dtype=np.float32)
    engine = fiscal.FiscalDiscovery(rows, chunks, vectors, top_k=50)
    result = engine.discover("agricoles déficits", "2020-01-01", [1., 0.])
    assert all(hit["version_id"] != "future" for hit in result["history"]["hybrid"])
    address = next(row for row in result["addresses"] if row["article_cid"] == "a")
    assert address["bound_version_id"] == "new"
    assert address["features"]["distinct_revision_count"] == 2
    assert set(address["features"]) == set(fiscal.FEATURE_NAMES)
    selected = fiscal.select_conditions(result, engine.chunks, rows)
    assert selected["history_bound"][0]["version_id"] == "new"
    assert selected["history_unbound"][0]["version_id"] == "old"


def test_official_numeric_scoring_and_missing_provenance_do_not_become_correct():
    official = Path(__file__).parents[1] / "tests/fixtures/fiscalqa/scripts/score_nuggets.py"
    spec = importlib.util.spec_from_file_location("fiscal_official_test", official)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    question = {"gold": {"article_cid": "a", "version_id": "new"}}
    nuggets = [{"kind": "regex", "pattern": r"\b156\b", "required": True},
               {"kind": "numeric_tol", "target": 60540., "tol": 1., "required": True},
               {"kind": "provenance", "required": True}]
    score = fiscal.score_answer(question, "Article 156 : 60 540 euros.", nuggets,
                                [{"article_cid": "a", "version_id": "old"}], scorer)
    assert score["strict"] == 1
    assert score["coverage"] == 1
    assert score["gold_version_recall_at_four"] == 0
    assert score["retrieved_version_id"] == "old"
    empty = fiscal.score_answer(question, "", nuggets, [], scorer)
    assert empty["strict"] == 0
    assert empty["gold_article_recall_at_four"] == 0


def test_nongold_discovery_does_not_read_question_gold():
    rows = [version("one", "2000-01-01", None, "agricoles déficits")]
    chunks = fiscal.article_chunks(rows)
    engine = fiscal.FiscalDiscovery(rows, chunks, np.array([[1., 0.]]))
    result = engine.discover("agricoles", "2005-01-01", [1., 0.])
    assert result["addresses"][0]["bound_version_id"] == "one"


def test_scored_and_excluded_strata_have_distinct_denominators():
    questions = [{"qid": "x", "scope": "in"}, {"qid": "y", "scope": "excluded_review"}]
    scores = [{"qid": "x", "condition": "asof_dense", "strict": 1, "coverage": 1.},
              {"qid": "y", "condition": "asof_dense", "strict": 0, "coverage": 0.}]
    result = fiscal.summarize(scores, questions)
    assert result["scored209"]["asof_dense"]["n"] == 1
    assert result["scored209"]["asof_dense"]["strict"] == 1.
    assert result["published221"]["asof_dense"]["strict"] == .5
    assert result["excluded_review"]["asof_dense"]["n"] == 1


def test_packed_reader_provenance_includes_only_text_that_fits():
    class CharacterTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return list(range(sum(len(m["content"]) for m in messages)))

    versions = {"one": version("one", "2000-01-01", None, "a" * 300),
                "two": version("two", "2000-01-01", None, "b" * 300, cid="b")}
    packed = fiscal.pack_reader("Question", [{"version_id": "one"}, {"version_id": "two"}],
                               versions, CharacterTokenizer(), prompt_budget=600)
    assert packed["prompt_tokens"] <= 600
    assert len(packed["evidence"]) == 1
    assert packed["evidence"][0]["version_id"] == "one"
    assert packed["evidence"][0]["truncated"]
    assert "b" * 50 not in packed["messages"][-1]["content"]
