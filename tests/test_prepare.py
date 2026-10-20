"""Small saved official-tree fixtures exercise the complete HiDate preparation CLI."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hidate.io import load_jsonl, write_json, write_jsonl
from hidate.sources import extract_provisions


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_data.py"
DATES = ["2018-09-30", "2026-09-30"]


def node(tag, children=(), **attr):
    return {"tag": tag, "attr": attr, "children": list(children)}


def article(num, text, **attr):
    return node("Article", [node("ArticleTitle", [f"第{num}条"]),
        node("Paragraph", [node("ParagraphNum", []),
            node("ParagraphSentence", [node("Sentence", [text])])], Num="1")], Num=num, **attr)


def raw_snapshot(law_id, date):
    text = "本条は試験法の公開された条文を保持するための規定であり、対象期間を確認する。"
    changed = ("改正前は申請書を窓口に提出し本人確認と受付の手続を完了するものとする。" if date == DATES[0]
               else "改正後は申請書を電子的に提出し本人確認と受付の手続を完了するものとする。")
    missing = "この条文は後年削除される試験規定であり、提出者の届出に必要な事項を定める。" if date == DATES[0] else "削除"
    revision = f"{law_id}_{date.replace('-', '')}_baseline"
    main = [article("1", text), article("2", changed), article("3", missing, Delete=(date == DATES[1]))]
    return {"law_info": {"law_id": law_id, "law_type": "Act"},
        "revision_info": {"law_revision_id": revision, "law_title": "試験法"},
        "law_full_text": node("Law", [node("LawBody", [node("LawTitle", ["試験法"]),
            node("MainProvision", main), node("SupplProvision", [article("1", "附則の適用日は公示後に確定する。")])])])}


@pytest.fixture
def saved_collection(tmp_path):
    source, output = tmp_path / "raw", tmp_path / "prepared"
    source.mkdir()
    selected, provisions, bindings = [], [], []
    for index in range(10):
        law_id = f"300AC00000000{index:02d}"
        selected.append({"law_info": {"law_id": law_id, "law_type": "Act"}, "family_id": law_id})
        for date in DATES:
            snapshot = raw_snapshot(law_id, date)
            path = f"snapshots/{law_id}/{date}.json"
            write_json(source / path, snapshot)
            bindings.append({"law_id": law_id, "requested_asof": date, "status": "collected",
                "revision_id": snapshot["revision_info"]["law_revision_id"], "raw_path": path,
                "source_role": "full", "family_id": law_id})
            provisions.extend({**record, "requested_asof": date, "raw_path": path,
                "source_role": "full", "family_id": law_id} for record in extract_provisions(snapshot))
    # An external law is retained in the corpus, but has no supervised family allocation.
    external = raw_snapshot("323AC0000000025", DATES[1])
    path = "snapshots/323AC0000000025/2026-09-30.json"
    write_json(source / path, external)
    bindings.append({"law_id": "323AC0000000025", "requested_asof": DATES[1], "status": "collected",
        "revision_id": external["revision_info"]["law_revision_id"], "raw_path": path,
        "source_role": "external", "family_id": "323AC0000000025"})
    provisions.extend({**record, "requested_asof": DATES[1], "raw_path": path,
        "source_role": "external", "family_id": "323AC0000000025"} for record in extract_provisions(external))
    questions = {"samples": [
        {"ファイル名": "a.md", "問題文": "正しいものはどれか", "コンテキスト": "原文A",
         "選択肢": "a: A\nb: B\nc: C\nd: D", "output": "a", "references": ["原資料A"]},
        {"ファイル名": "b.md", "問題文": "正しいものはどれか", "コンテキスト": "原文B",
         "選択肢": "a: A\nb: B\nc: C\nd: D", "output": "b", "references": ["原資料B"]},
    ]}
    write_json(source / "lawqa/selection.json", questions)
    write_json(source / "selected-laws.json", selected)
    write_jsonl(source / "provisions.jsonl", provisions)
    write_jsonl(source / "asof-bindings.jsonl", bindings)
    (source / "guidance").mkdir()
    guidance_text = "公式ガイダンス本文\n\n  原文の空白も保持する。\n"
    (source / "guidance/02-250221_kaiji.txt").write_text(guidance_text, encoding="utf-8")
    write_json(source / "guidance.json", [{"source_id": "guidance-02", "title": "公式ガイドライン",
        "url": "https://www.fsa.go.jp/common/law/kaiji/250221_kaiji.pdf", "status": "collected",
        "text_path": "guidance/02-250221_kaiji.txt", "pdf_path": "guidance/02-250221_kaiji.pdf",
        "acquisition_url": "https://warp.ndl.go.jp/verified-original.pdf",
        "archive_capture": "2025-04-01T13:12:32+09:00"}])
    write_json(source / "manifest.json", {"complete": True, "mode": "full", "lawqa_questions": 2,
        "provision_count": len(provisions), "successful_snapshots": len(bindings),
        "successful_guidance": 1, "lawqa_revision": "frozen-release"})
    protocol = tmp_path / "protocol.json"
    write_json(protocol, {"seed": 42, "anchor_dates": DATES, "primary_scope": "MainProvision",
        "cap_per_family_target_stratum": 25, "quote_chars": 96, "min_chars": 24})
    return source, output, protocol, questions, guidance_text


def run_prepare(source, output, protocol):
    assert SCRIPT.exists(), "HiDate preparation CLI is not implemented"
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    return subprocess.run([sys.executable, "-X", "utf8", str(SCRIPT), "--source-dir", str(source),
        "--output-dir", str(output), "--protocol", str(protocol)], env=environment,
        capture_output=True, text=True, encoding="utf-8")


def test_cli_prepares_real_scoped_queries_and_preserves_external_sources(saved_collection):
    source, output, protocol, questions, guidance_text = saved_collection
    completed = run_prepare(source, output, protocol)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    allocation = json.loads((output / "family-split.json").read_text())
    assert set(allocation.values()) == {"train", "valid", "test"}
    assert "323AC0000000025" not in allocation
    corpus = load_jsonl(output / "corpus-passages.jsonl")
    assert len(corpus) == len({row["passage_id"] for row in corpus})
    assert {row["source_role"] for row in corpus} == {"full", "external"}
    assert any(row["scope"].startswith("SupplProvision") for row in corpus)
    external_passages = [row for row in corpus if row["source_role"] == "external"]
    assert len(external_passages) == 4
    assert all(row["split"] == "external" for row in external_passages)
    census = load_jsonl(output / "address-pair-census.jsonl")
    assert {row["stratum"] for row in census} >= {"same_date", "changed_cross_date", "unavailable_target"}
    assert all(row["scope"] == "MainProvision" for row in census)
    assert all(row["source_role"] == "full" for row in census)
    assert all("visible_text" not in row and "text" not in row for row in census)
    queries = load_jsonl(output / "queries-exact.jsonl")
    assert {row["split"] for row in queries} == {"train", "valid", "test"}
    assert {row["query_variant"] for row in queries} == {"exact_quote"}
    missing = [row for row in queries if row["stratum"] == "unavailable_target"]
    assert missing and all(row["target_status"] == "deleted" for row in missing)
    assert all(row["query_text"] in row["source_body_text"] for row in queries)
    assert all(row.get("query_text_occurrences", 0) > 1 for row in queries)
    assert load_jsonl(output / "lawqa-external.jsonl") == questions["samples"]
    guides = load_jsonl(output / "guidance-documents.jsonl")
    assert guides[0]["text"] == guidance_text
    assert guides[0]["archive_capture"] == "2025-04-01T13:12:32+09:00"
    prepared = json.loads((output / "preparation.json").read_text())
    assert prepared["complete"] is True
    assert prepared["statistics"]["lawqa_questions"] == 2
    assert prepared["training_runs"] == prepared["model_inference_runs"] == 0


def test_cli_refuses_incomplete_collection_before_derived_outputs(saved_collection):
    source, output, protocol, _, _ = saved_collection
    write_json(source / "manifest.json", {"complete": False, "mode": "full"})
    completed = run_prepare(source, output, protocol)
    assert completed.returncode != 0
    assert "collection" in completed.stderr.lower() and "complete" in completed.stderr.lower()
    assert not (output / "family-split.json").exists()
    assert not (output / "queries-exact.jsonl").exists()


def test_same_revision_dates_consolidate_corpus_passages(saved_collection):
    source, output, protocol, _, _ = saved_collection
    law_id = "300AC0000000000"
    first = json.loads((source / f"snapshots/{law_id}/{DATES[0]}.json").read_text())
    write_json(source / f"snapshots/{law_id}/{DATES[1]}.json", first)
    rows = load_jsonl(source / "provisions.jsonl")
    rows = [row for row in rows if not (row["law_id"] == law_id and row["requested_asof"] == DATES[1])]
    rows.extend({**record, "requested_asof": DATES[1], "raw_path": f"snapshots/{law_id}/{DATES[1]}.json",
        "source_role": "full", "family_id": law_id} for record in extract_provisions(first))
    write_jsonl(source / "provisions.jsonl", rows)
    bindings = load_jsonl(source / "asof-bindings.jsonl")
    for binding in bindings:
        if binding["law_id"] == law_id and binding["requested_asof"] == DATES[1]:
            binding["revision_id"] = first["revision_info"]["law_revision_id"]
    write_jsonl(source / "asof-bindings.jsonl", bindings)
    completed = run_prepare(source, output, protocol)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    corpus = [row for row in load_jsonl(output / "corpus-passages.jsonl") if row["law_id"] == law_id]
    assert len(corpus) == 4
    assert all(row["asof_dates"] == DATES for row in corpus)
    targets = [row for row in load_jsonl(output / "target-addresses.jsonl") if row["law_id"] == law_id]
    assert {row["requested_asof"] for row in targets} == set(DATES)
    assert {row["revision_id"] for row in targets} == {first["revision_info"]["law_revision_id"]}


def test_direct_supplementary_paragraphs_keep_distinct_target_addresses(saved_collection):
    source, output, protocol, _, _ = saved_collection
    law_id, date = "300AC0000000000", DATES[0]
    raw_path = f"snapshots/{law_id}/{date}.json"
    snapshot = json.loads((source / raw_path).read_text())
    supplemental = snapshot["law_full_text"]["children"][0]["children"][2]
    supplemental["children"].extend([
        node("Paragraph", [node("ParagraphSentence", [node("Sentence", ["附則の第一項は申請手続を定める。"])])], Num="1"),
        node("Paragraph", [node("ParagraphSentence", [node("Sentence", ["附則の第二項は適用日を定める。"])])], Num="2"),
    ])
    write_json(source / raw_path, snapshot)
    rows = load_jsonl(source / "provisions.jsonl")
    rows = [row for row in rows if not (row["law_id"] == law_id and row["requested_asof"] == date)]
    rows.extend({**record, "requested_asof": date, "raw_path": raw_path,
        "source_role": "full", "family_id": law_id} for record in extract_provisions(snapshot))
    write_jsonl(source / "provisions.jsonl", rows)
    completed = run_prepare(source, output, protocol)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    targets = [row for row in load_jsonl(output / "target-addresses.jsonl")
        if row["law_id"] == law_id and row["requested_asof"] == date and row["article_num"] is None]
    assert {row["address_num"] for row in targets} == {"Paragraph_1", "Paragraph_2"}
    assert all(row["status"] == "available" and len(row["passage_ids"]) == 1 for row in targets)


def test_selected_short_origin_is_reported_without_emitting_a_quote(saved_collection):
    source, output, protocol, _, _ = saved_collection
    law_id, date = "300AC0000000000", DATES[0]
    raw_path = f"snapshots/{law_id}/{date}.json"
    snapshot = json.loads((source / raw_path).read_text())
    paragraph = snapshot["law_full_text"]["children"][0]["children"][1]["children"][1]["children"][1]
    paragraph["children"][1]["children"][0]["children"] = ["短い原文。"]
    write_json(source / raw_path, snapshot)
    rows = load_jsonl(source / "provisions.jsonl")
    rows = [row for row in rows if not (row["law_id"] == law_id and row["requested_asof"] == date)]
    rows.extend({**record, "requested_asof": date, "raw_path": raw_path,
        "source_role": "full", "family_id": law_id} for record in extract_provisions(snapshot))
    write_jsonl(source / "provisions.jsonl", rows)
    completed = run_prepare(source, output, protocol)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    selected = load_jsonl(output / "selected-query-pairs.jsonl")
    queries = load_jsonl(output / "queries-exact.jsonl")
    assert any(row["law_id"] == law_id and row["source_asof"] == date and row["article_num"] == "2" for row in selected)
    assert not any(row["law_id"] == law_id and row["source_asof"] == date and row["article_num"] == "2" for row in queries)
    statistics = json.loads((output / "preparation.json").read_text())["statistics"]
    assert statistics["selected_pairs"] > statistics["exact_queries"]
    assert statistics["selected_pairs"] == len(selected)
    assert statistics["exact_queries"] == len(queries)


def test_later_paragraph_change_keeps_query_identity_and_reports_quote_presence(saved_collection):
    source, output, protocol, _, _ = saved_collection
    completed = run_prepare(source, output, protocol)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    before = {row["query_id"]: row for row in load_jsonl(output / "queries-exact.jsonl")}
    law_id, date = "300AC0000000000", DATES[1]
    raw_path = f"snapshots/{law_id}/{date}.json"
    target = json.loads((source / raw_path).read_text())
    origin = json.loads((source / f"snapshots/{law_id}/{DATES[0]}.json").read_text())
    target_article = target["law_full_text"]["children"][0]["children"][1]["children"][1]
    source_article = origin["law_full_text"]["children"][0]["children"][1]["children"][1]
    target_article["children"] = [target_article["children"][0], source_article["children"][1],
        node("Paragraph", [node("ParagraphSentence", [node("Sentence", ["改正された第二段落は別の手続事項を定める。"])])], Num="2")]
    write_json(source / raw_path, target)
    records = load_jsonl(source / "provisions.jsonl")
    records = [row for row in records if not (row["law_id"] == law_id and row["requested_asof"] == date)]
    records.extend({**record, "requested_asof": date, "raw_path": raw_path,
        "source_role": "full", "family_id": law_id} for record in extract_provisions(target))
    write_jsonl(source / "provisions.jsonl", records)
    completed = run_prepare(source, output, protocol)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    after = load_jsonl(output / "queries-exact.jsonl")
    query = next(row for row in after if row["law_id"] == law_id and row["stratum"] == "changed_cross_date")
    statistics = json.loads((output / "preparation.json").read_text())["statistics"]
    assert statistics.get("changed_quote_counts") == {"quote_present": 1, "quote_absent": 9}
    assert statistics["changed_quote_counts_by_split"][query["split"]]["quote_present"] == 1
    assert query["quote_present_in_target"] is True
    assert query["source_quote_absent_from_target"] is False
    original_query = before[query["query_id"]]
    assert all(query[key] == original_query[key] for key in ("query_id", "query_text", "labels", "split"))
    assert len(after) == len(before)
