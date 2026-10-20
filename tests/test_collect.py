"""Small HTTP fixtures for dated source capture, without live API requests."""

import json
import runpy
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/collect_sources.py"
PROBE_IDS = ["322AC0000000049", "129AC0000000089", "417AC0000000086"]
ANCHORS = ["2018-09-30", "2021-09-30", "2024-09-30", "2026-09-30"]


def collector():
    assert SCRIPT.exists(), "HiDate source collector is not implemented"
    return runpy.run_path(str(SCRIPT))["collect_sources"]


def node(tag, children=(), **attr):
    return {"tag": tag, "attr": attr, "children": list(children)}


def snapshot(law_id, asof):
    revision = f"{law_id}_{asof.replace('-', '')}_000000000000000"
    return {
        "law_info": {"law_id": law_id, "law_type": "Act", "promulgation_date": "1947-04-07"},
        "revision_info": {"law_revision_id": revision, "law_title": "対象法律",
                          "amendment_enforcement_date": asof, "mission": "New",
                          "current_revision_status": "PreviousEnforced"},
        "current_revision_info": {"law_revision_id": f"{law_id}_20270401_future"},
        "law_full_text": node("Law", [node("LawBody", [node("LawTitle", ["対象法律"]),
            node("MainProvision", [node("Article", [node("ArticleTitle", ["第一条"]),
                node("Paragraph", [node("ParagraphNum", []),
                    node("ParagraphSentence", [node("Sentence", ["本文。"])] )], Num="1")], Num="1")]),
            node("SupplProvision", [node("Article", [node("ArticleTitle", ["第一条"]),
                node("Paragraph", [node("ParagraphSentence", [node("Sentence", ["附則本文。"])])], Num="1")], Num="1")])])]),
    }


def law_entry(law_id, law_type, title):
    return {"law_info": {"law_id": law_id, "law_type": law_type,
                         "promulgation_date": "2000-01-01"},
            "revision_info": {"law_revision_id": f"{law_id}_20240401_a", "law_type": law_type,
                              "law_title": title, "amendment_enforcement_date": "2024-04-01",
                              "amendment_type": "3",
                              "mission": "New", "current_revision_status": "CurrentEnforced",
                              "repeal_status": "None"}}


class FakeHTTP:
    """The external bytes boundary is fake; all collection and parsing stays real."""

    def __init__(self, pages=None, failure=None, bad_snapshot=None, pdf=b"%PDF-1.7\nfixture"):
        self.urls = []
        self.pages = pages or {}
        self.failure = failure
        self.bad_snapshot = bad_snapshot
        self.pdf = pdf

    def __call__(self, url):
        self.urls.append(url)
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        if self.failure and self.failure in url:
            raise OSError("temporary source unavailable")
        if parsed.path.endswith(".pdf"):
            return self.pdf
        if parsed.path.endswith("/laws"):
            assert query["asof"] == ["2026-09-30"]
            assert query["limit"] == ["100"]
            assert query["law_type"] == ["Act,CabinetOrder,MinisterialOrdinance"]
            return json.dumps(self.pages[int(query["offset"][0])]).encode()
        law_id = parsed.path.rsplit("/", 1)[1]
        if "/law_revisions/" in parsed.path:
            response = {"law_info": {"law_id": law_id}, "revisions": [
                {"law_revision_id": f"{law_id}_20270401_future", "current_revision_status": "UnEnforced"},
                {"law_revision_id": f"{law_id}_20240401_old", "current_revision_status": "PreviousEnforced"}]}
        else:
            assert "/law_data/" in parsed.path
            assert query["law_full_text_format"] == ["json"]
            assert query["json_format"] == ["full"]
            assert query["response_format"] == ["json"]
            response = snapshot(law_id, query["asof"][0])
            if self.bad_snapshot and self.bad_snapshot in url:
                response["law_full_text"] = "base64/XML is not a full JSON tree"
        return json.dumps(response, ensure_ascii=False).encode("utf-8")


@pytest.fixture
def protocol_path(tmp_path):
    law_list = tmp_path / "law_list.json"
    questions = tmp_path / "questions.json"
    law_list.write_bytes((ROOT / "tests/fixtures/legal/lawqa-data-law_list.json").read_bytes())
    # Repeated question stems remain independent original question units.
    questions.write_text(json.dumps({"samples": [
        {"ファイル名": "a.md", "問題文": "正しいものはどれか", "コンテキスト": "A",
         "選択肢": "a: A\nb: B\nc: C\nd: D", "output": "a", "references": ["source-A"]},
        {"ファイル名": "b.md", "問題文": "正しいものはどれか", "コンテキスト": "B",
         "選択肢": "a: A\nb: B\nc: C\nd: D", "output": "b", "references": ["source-B"]},
    ]}, ensure_ascii=False), encoding="utf-8")
    protocol = json.loads((ROOT / "configs/source-protocol.json").read_text(encoding="utf-8"))
    protocol.update(lawqa_sources=str(law_list), lawqa_questions=str(questions), count_per_type=1)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    return path


def pdf_text(pdf_path, text_path):
    assert pdf_path.read_bytes().startswith(b"%PDF-")
    text_path.write_text("公式ガイダンス本文。\n", encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_probe_preserves_actual_asof_ids_full_trees_and_all_guidance(protocol_path, tmp_path):
    output = tmp_path / "capture"
    http = FakeHTTP()
    manifest = collector()("probe", output, protocol_path, transport=http, pdf_to_text=pdf_text,
                           sleep=lambda _: None)
    assert manifest["complete"] is True
    assert manifest["law_ids"] == PROBE_IDS
    assert manifest["requested_snapshots"] == 3
    assert manifest["successful_snapshots"] == 3
    assert manifest["successful_guidance"] == 11
    assert manifest["lawqa_source_documents"] == 24
    assert manifest["lawqa_questions"] == 2
    assert (output / "lawqa/selection.json").read_bytes() == (protocol_path.parent / "questions.json").read_bytes()
    binding = read_jsonl(output / "asof-bindings.jsonl")[0]
    assert binding["revision_id"] == "322AC0000000049_20260930_000000000000000"
    assert binding["requested_asof"] == "2026-09-30"
    raw = json.loads((output / binding["raw_path"]).read_text(encoding="utf-8"))
    assert raw["current_revision_info"]["law_revision_id"] == "322AC0000000049_20270401_future"
    assert raw["law_full_text"]["tag"] == "Law"
    provisions = read_jsonl(output / "provisions.jsonl")
    assert {row["scope"] for row in provisions} >= {"MainProvision"}
    assert {row["requested_asof"] for row in provisions} == {"2026-09-30"}
    assert {row["revision_id"] for row in provisions} == {
        "322AC0000000049_20260930_000000000000000", "129AC0000000089_20260930_000000000000000",
        "417AC0000000086_20260930_000000000000000"}
    assert read_jsonl(output / "failures.jsonl") == []
    assert len(read_jsonl(output / "requests.jsonl")) == 17  # 3 revisions + 3 snapshots + 11 PDFs


def test_resume_uses_exact_law_date_and_pdf_caches(protocol_path, tmp_path):
    output = tmp_path / "capture"
    collect = collector()
    collect("probe", output, protocol_path, transport=FakeHTTP(), pdf_to_text=pdf_text, sleep=lambda _: None)

    def no_fetch(url):
        pytest.fail(f"successful cache was requested again: {url}")

    def no_conversion(pdf_path, text_path):
        pytest.fail("nonempty cached PDF text was converted again")

    manifest = collect("probe", output, protocol_path, transport=no_fetch,
                       pdf_to_text=no_conversion, sleep=lambda _: None)
    assert manifest["complete"] is True
    assert len(read_jsonl(output / "provisions.jsonl")) == 6
    assert len(read_jsonl(output / "requests.jsonl")) == 17


def test_full_follows_next_offset_and_collects_four_anchors_plus_external_release_date(protocol_path, tmp_path):
    pages = {
        0: {"count": 2, "total_count": 3, "next_offset": 100, "laws": [
            law_entry("300AC0000000001", "Act", "独立甲法"),
            law_entry("300CO0000000001", "CabinetOrder", "独立甲法施行令")]},
        100: {"count": 1, "total_count": 3, "next_offset": None, "laws": [
            law_entry("300M00000000001", "MinisterialOrdinance", "独立乙省令")]},
    }
    output = tmp_path / "capture"
    manifest = collector()("full", output, protocol_path, transport=FakeHTTP(pages),
                           pdf_to_text=pdf_text, sleep=lambda _: None)
    assert manifest["complete"] is True
    assert manifest["catalogue_laws"] == 3
    assert manifest["selected_by_type"] == {"Act": 1, "CabinetOrder": 1, "MinisterialOrdinance": 1}
    assert manifest["requested_snapshots"] == 77  # 3 selected * 4, 13 external * 5
    assert (output / "catalogue/2026-09-30/offset-000100.json").exists()
    bindings = read_jsonl(output / "asof-bindings.jsonl")
    assert {row["requested_asof"] for row in bindings if row["law_id"] == "300AC0000000001"} == set(ANCHORS)
    assert {row["requested_asof"] for row in bindings if row["law_id"] == "323AC0000000025"} == {
        "2018-09-30", "2021-09-30", "2024-09-30", "2026-09-30", "2025-11-04"}
    assert {row["family_id"] for row in bindings if row["law_id"] == "300CO0000000001"} == {"300AC0000000001"}


def test_network_failure_retries_twice_and_is_collection_failure(protocol_path, tmp_path):
    output = tmp_path / "capture"
    http = FakeHTTP(failure="law_data/322AC0000000049")
    waits = []
    manifest = collector()("probe", output, protocol_path, transport=http, pdf_to_text=pdf_text,
                           sleep=waits.append)
    assert manifest["complete"] is False
    assert manifest["successful_snapshots"] == 2
    assert len([url for url in http.urls if "law_data/322AC0000000049" in url]) == 3
    assert len(waits) == 2
    failed = [row for row in read_jsonl(output / "asof-bindings.jsonl") if row["law_id"] == PROBE_IDS[0]][0]
    assert failed["status"] == "collection_failure"
    assert "revision_id" not in failed
    assert "deleted" not in failed
    failures = read_jsonl(output / "failures.jsonl")
    assert len(failures) == 1
    assert failures[0]["stage"] == "snapshot"
    assert failures[0]["requested_asof"] == "2026-09-30"


def test_snapshot_parse_failure_keeps_original_response_without_inventing_gold(protocol_path, tmp_path):
    output = tmp_path / "capture"
    manifest = collector()("probe", output, protocol_path,
        transport=FakeHTTP(bad_snapshot="law_data/322AC0000000049"),
        pdf_to_text=pdf_text, sleep=lambda _: None)
    assert manifest["complete"] is False
    assert json.loads((output / "snapshots/322AC0000000049/2026-09-30.json").read_text())["law_full_text"] == (
        "base64/XML is not a full JSON tree")
    failed = read_jsonl(output / "asof-bindings.jsonl")[0]
    assert failed["status"] == "collection_failure"
    assert "revision_id" not in failed
    assert all(row["law_id"] != PROBE_IDS[0] for row in read_jsonl(output / "provisions.jsonl"))


def test_resume_recovers_failed_request_and_retains_failure_history(protocol_path, tmp_path):
    output = tmp_path / "capture"
    collect = collector()
    collect("probe", output, protocol_path, transport=FakeHTTP(failure="law_data/322AC0000000049"),
            pdf_to_text=pdf_text, sleep=lambda _: None)
    recovered = FakeHTTP()
    manifest = collect("probe", output, protocol_path, transport=recovered,
                       pdf_to_text=pdf_text, sleep=lambda _: None)
    assert manifest["complete"] is True
    assert manifest["failed_tasks_this_run"] == 0
    assert len(recovered.urls) == 1
    assert "/law_data/322AC0000000049?asof=2026-09-30&" in recovered.urls[0]
    assert len(read_jsonl(output / "failures.jsonl")) == 1
    assert read_jsonl(output / "asof-bindings.jsonl")[0]["status"] == "collected"
    assert len(read_jsonl(output / "provisions.jsonl")) == 6


def test_same_returned_revision_retains_independent_date_bindings(protocol_path, tmp_path):
    protocol = json.loads(protocol_path.read_text())
    protocol["probe_dates"] = ["2024-09-30", "2026-09-30"]
    protocol_path.write_text(json.dumps(protocol))
    output = tmp_path / "capture"
    http = FakeHTTP()

    def unchanged_revision(url):
        content = http(url)
        if "/law_data/" in url:
            response = json.loads(content)
            response["revision_info"]["law_revision_id"] = response["law_info"]["law_id"] + "_20240401_baseline"
            return json.dumps(response).encode()
        return content

    manifest = collector()("probe", output, protocol_path, transport=unchanged_revision,
                           pdf_to_text=pdf_text, sleep=lambda _: None)
    assert manifest["complete"] is True
    bindings = [row for row in read_jsonl(output / "asof-bindings.jsonl") if row["law_id"] == PROBE_IDS[0]]
    assert [row["requested_asof"] for row in bindings] == ["2024-09-30", "2026-09-30"]
    assert {row["revision_id"] for row in bindings} == {"322AC0000000049_20240401_baseline"}
    assert [row["raw_path"] for row in bindings] == [
        "snapshots/322AC0000000049/2024-09-30.json", "snapshots/322AC0000000049/2026-09-30.json"]
    assert {row["requested_asof"] for row in read_jsonl(output / "provisions.jsonl")} == {
        "2024-09-30", "2026-09-30"}


def test_failed_catalogue_does_not_claim_full_collection_complete(protocol_path, tmp_path):
    output = tmp_path / "capture"
    manifest = collector()("full", output, protocol_path, transport=FakeHTTP(failure="/laws?"),
                           pdf_to_text=pdf_text, sleep=lambda _: None)
    assert manifest["complete"] is False
    assert manifest["catalogue_complete"] is False
    assert manifest["selected_law_ids"] == []
    assert manifest["successful_snapshots"] == 65
    assert manifest["successful_guidance"] == 11
    assert read_jsonl(output / "failures.jsonl")[0]["stage"] == "catalogue"


def test_explicit_guidance_archive_preserves_original_source_url(protocol_path, tmp_path):
    original = "https://www.fsa.go.jp/common/law/kaiji/250221_kaiji.pdf"
    archived = "https://warp.ndl.go.jp/20250403/20250401041232/https://www.fsa.go.jp/common/law/kaiji/250221_kaiji.pdf"
    protocol = json.loads(protocol_path.read_text())
    protocol["guidance_archive_urls"] = {
        original: {"url": archived, "archive_capture": "2025-04-01T13:12:32+09:00"}}
    protocol_path.write_text(json.dumps(protocol))
    output = tmp_path / "capture"
    http = FakeHTTP()
    manifest = collector()("probe", output, protocol_path, transport=http,
                           pdf_to_text=pdf_text, sleep=lambda _: None)
    assert manifest["complete"] is True
    sources = json.loads((output / "guidance.json").read_text())
    archived_source = next(row for row in sources if row["url"] == original)
    assert archived_source.get("acquisition_url") == archived
    assert archived_source["archive_capture"] == "2025-04-01T13:12:32+09:00"
    assert archived_source["acquisition_provenance"] == "National Diet Library WARP capture of the original source URL"
    assert archived in http.urls
    assert original not in http.urls
    assert (output / "lawqa/law_list.json").read_bytes() == (protocol_path.parent / "law_list.json").read_bytes()
    request = next(row for row in read_jsonl(output / "requests.jsonl") if row["url"] == archived)
    assert request["raw_path"] == "guidance/02-250221_kaiji.pdf"


@pytest.mark.parametrize("bad_pdf,empty_text", [(True, False), (False, True)])
def test_guidance_requires_pdf_format_and_nonempty_layout_text(protocol_path, tmp_path, bad_pdf, empty_text):
    output = tmp_path / "capture"

    def convert(pdf_path, text_path):
        assert not bad_pdf, "HTML bytes must not be passed to pdftotext"
        text_path.write_text(" \n" if empty_text else "本文", encoding="utf-8")

    manifest = collector()("probe", output, protocol_path,
        transport=FakeHTTP(pdf=b"<html>not a PDF</html>" if bad_pdf else b"%PDF-1.7\nfixture"),
        pdf_to_text=convert, sleep=lambda _: None)
    assert manifest["complete"] is False
    assert manifest["successful_guidance"] == 0
    assert len(list((output / "guidance").glob("*.pdf"))) == 11
    assert all(row["status"] == "collection_failure" for row in json.loads((output / "guidance.json").read_text()))
    assert len(read_jsonl(output / "failures.jsonl")) == 11
