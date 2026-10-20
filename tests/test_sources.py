import copy
import json
from pathlib import Path

import pytest

from hidate import sources as sources


ROOT = Path(__file__).resolve().parents[1]


def node(tag, *children, **attr):
    return {"tag": tag, "attr": attr, "children": list(children)}


def paragraph(num, text, **attr):
    return node("Paragraph", node("ParagraphNum"),
                node("ParagraphSentence", node("Sentence", text)), Num=num, **attr)


def article(num, text, **attr):
    return node("Article", node("ArticleTitle", "条" + num),
                paragraph("1", text), Num=num, **attr)


def snapshot(*blocks):
    return {"law_info": {"law_id": "LAW"},
            "revision_info": {"law_revision_id": "LAW_REV", "law_title": "試験法"},
            "law_full_text": node("Law", node("LawNum", "試験法律"),
                                  node("LawBody", node("LawTitle", "試験法"), *blocks))}


def at_path(tree, path):
    for index in path:
        tree = tree["children"][index]
    return tree


def test_exact_branch_and_range_numbers_and_paragraph_paths():
    first = article("12_2", "枝条文。", Delete="false", Hide="false")
    first["children"].insert(0, node("ArticleCaption", "（見出し）"))
    first["children"].append(paragraph("2_1", "第二段落。"))
    data = snapshot(node("MainProvision", node("Part", node("Chapter",
                    node("Section", node("Subsection", node("Division", first,
                    article("13:15", "範囲条文。"))))))))
    records = sources.extract_provisions(data)
    assert [r["article_num"] for r in records] == ["12_2", "13:15"]
    record = records[0]
    assert (record["law_id"], record["revision_id"], record["law_title"]) == (
        "LAW", "LAW_REV", "試験法")
    assert record["scope"] == "MainProvision"
    assert record["slot_id"] == "LAW:MainProvision:12_2"
    assert record["title"] == "条12_2"
    assert record["caption"] == "（見出し）"
    assert record["text"] == "枝条文。\n第二段落。"
    assert not record["deleted"] and not record["hidden"]
    assert [p["num"] for p in record["paragraphs"]] == ["1", "2_1"]
    assert record["paragraphs"][0]["text"] == "枝条文。"
    assert record["tree_path"] == [1, 1, 0, 0, 0, 0, 0, 0]
    assert at_path(data["law_full_text"], record["tree_path"]) is first
    assert at_path(data["law_full_text"], record["paragraphs"][0]["tree_path"])["attr"]["Num"] == "1"


def test_nested_new_provision_article_stays_inside_amendment_text():
    enclosing = article("2", "次のように改める。")
    enclosing["children"][1]["children"].append(
        node("AmendProvision", node("NewProvision", article("99", "置換された引用文。"))))
    data = snapshot(node("MainProvision", enclosing,
                         node("NewProvision", article("100", "独立しない。"))))
    records = sources.extract_provisions(data)
    assert [r["article_num"] for r in records] == ["2"]
    assert "置換された引用文。" in records[0]["text"]
    assert "条99" in records[0]["text"]


@pytest.mark.parametrize("attrs,text,deleted,hidden", [
    ({"Delete": "false", "Hide": "false"}, "本文。", False, False),
    ({"Delete": "true", "Hide": "false"}, "旧本文。", True, False),
    ({"Delete": False, "Hide": True}, "表示されない本文。", False, True),
    ({"Delete": "false"}, "削除", True, False),
    ({}, "削除に関する規定。", False, False),
])
def test_deleted_placeholders_and_hidden_flags_are_distinct(attrs, text, deleted, hidden):
    records = sources.extract_provisions(snapshot(node("MainProvision", article("1", text, **attrs))))
    assert records[0]["deleted"] is deleted
    assert records[0]["hidden"] is hidden


def test_supplementary_occurrences_and_appendices_keep_separate_scopes():
    data = snapshot(node("MainProvision", article("1", "本則。")),
                    node("SupplProvision", paragraph("1", "原始附則。")),
                    node("SupplProvision", article("1", "改正附則一。"), AmendLawNum="令和五年法律一号"),
                    node("SupplProvision", article("1", "改正附則二。"), AmendLawNum="令和五年法律一号"),
                    node("SupplProvision", article("1", "別の改正附則。"), AmendmentLawNum="令和六年法律二号"),
                    node("AppdxTable", node("AppdxTableTitle", "別表第一"),
                         node("TableStruct", node("Table", node("TableRow", node("TableColumn", "表の内容。"))))),
                    node("AppdxTable", node("AppdxTableTitle", "別表第二"), "第二表。"))
    records = sources.extract_provisions(data)
    assert [r["scope"] for r in records] == [
        "MainProvision", "SupplProvision:original", "SupplProvision:令和五年法律一号",
        "SupplProvision:令和五年法律一号#2", "SupplProvision:令和六年法律二号",
        "AppdxTable:1", "AppdxTable:2"]
    assert len({r["slot_id"] for r in records}) == 7
    assert records[1]["paragraphs"][0]["num"] == "1"
    assert records[5]["article_num"] is None
    assert records[5]["title"] == "別表第一"
    assert records[5]["text"] == "表の内容。"
    assert at_path(data["law_full_text"], records[5]["tree_path"])["tag"] == "AppdxTable"


def test_item_and_table_boundaries_preserve_inline_sentence_fragments():
    listed = article("1", "")
    listed["children"][1]["children"][1] = node(
        "ParagraphSentence", node("Sentence", "次の", node("Ruby", "表", node("Rt", "ひょう")), "による。"))
    listed["children"][1]["children"].extend([
        node("Item", node("ItemTitle", "一"), node("ItemSentence", node("Sentence", "第一。")), Num="1"),
        node("Item", node("ItemTitle", "二"), node("ItemSentence", node("Sentence", "第", "二。")), Num="2")])
    table = node("AppdxTable", node("AppdxTableTitle", "補償表"), node("TableStruct", node("Table",
        node("TableHeaderRow", node("TableHeaderColumn", "等級"), node("TableHeaderColumn", "災害補償")),
        node("TableRow", node("TableColumn", "第一級"), node("TableColumn", node("Sentence", "一三四〇", "日分"))),
        node("TableRow", node("TableColumn", "第二級"), node("TableColumn", "一一九〇日分")))))
    records = sources.extract_provisions(snapshot(node("MainProvision", listed), table))
    assert records[0]["text"] == "次の表ひょうによる。\n一第一。\n二第二。"
    assert records[0]["paragraphs"][0]["text"] == records[0]["text"]
    assert records[1]["text"] == "等級\t災害補償\n第一級\t一三四〇日分\n第二級\t一一九〇日分"


def test_paragraph_flags_and_inherited_container_flags_keep_text():
    hidden = article("1", "表示属性が偽の段落。", Hide="false", Delete="false")
    hidden["children"][1]["attr"].update(Hide="false", Delete="false")
    hidden["children"].append(paragraph("2", "削除属性のある段落。", Delete="true", Hide="false"))
    deleted = article("2", "削除された章の原文。", Hide="false", Delete="false")
    records = sources.extract_provisions(snapshot(node("MainProvision",
        node("Part", node("Chapter", hidden, Hide="true", Delete="false"),
             node("Chapter", deleted, Delete="true", Hide="false"), Hide="false", Delete="false"))))
    assert records[0]["hidden"] is True
    assert records[0]["deleted"] is False
    assert records[1]["deleted"] is True
    assert records[1]["hidden"] is False
    assert [(p["deleted"], p["hidden"]) for p in records[0]["paragraphs"]] == [(False, False), (True, False)]
    assert records[0]["paragraphs"][1]["text"] == "削除属性のある段落。"
    assert records[1]["text"] == "削除された章の原文。"


def test_saved_snapshots_have_identical_main_content_and_real_deleted_slot():
    snapshots = [json.loads((ROOT / "tests/fixtures/legal" / f"egov-law-version-{i}.json").read_text(encoding="utf-8")) for i in (0, 1)]
    all_records = [sources.extract_provisions(s) for s in snapshots]
    main = [{r["article_num"]: r for r in rows if r["scope"] == "MainProvision"} for rows in all_records]
    assert len(main[0]) == 124
    assert {key: row["text"] for key, row in main[0].items()} == {key: row["text"] for key, row in main[1].items()}
    assert main[0]["8"]["text"] == "削除" and main[0]["8"]["deleted"]
    assert main[0]["1"]["paragraphs"][0]["num"] == "1"
    assert main[0]["1"]["revision_id"] == snapshots[0]["revision_info"]["law_revision_id"]
    assert sum(r["scope"].startswith("AppdxTable:") for r in all_records[0]) == 3


def law(law_id, title, law_type="Act", **revision):
    metadata = {"law_revision_id": law_id + "_asof", "law_type": law_type,
                "law_title": title, "amendment_enforcement_date": "2024-04-01",
                "amendment_type": "3", "mission": "New", "repeal_status": "None",
                "current_revision_status": "PreviousEnforced"}
    metadata.update(revision)
    return {"law_info": {"law_id": law_id, "law_type": law_type, "promulgation_date": "2000-01-01"},
            "revision_info": metadata,
            "current_revision_info": {"law_title": "未来の名称", "amendment_enforcement_date": "2028-01-01"}}


def test_lawqa_reserves_all_sources_and_original_filename_groups():
    law_list = json.loads((ROOT / "tests/fixtures/legal/lawqa-data-law_list.json").read_text(encoding="utf-8"))
    excluded = sources.source_exclusions(law_list)
    assert len(excluded["law_ids"]) == 13
    assert len(excluded["guidance"]) == 11
    assert len(excluded["family_titles"]) == 13
    assert "323AC0000000025" in excluded["law_ids"]
    assert {d["url"] for d in excluded["guidance"]} == {d["url"] for d in law_list if d["url"].endswith(".pdf")}
    payload = json.loads((ROOT / "tests/fixtures/legal/lawqa-data-selection.json").read_text(encoding="utf-8"))
    rows = sources.lawqa_records(payload)
    assert len(rows) == 140
    assert len({r["group_id"] for r in rows}) == 140
    for raw, row in zip(payload["samples"], rows):
        assert row["group_id"] == raw["ファイル名"]
        assert all(row[k] == v for k, v in raw.items())
    assert len({r["問題文"] for r in rows}) < 140


def test_reserved_families_and_longest_exact_act_prefix():
    laws = [law("SHORT", "試験法"), law("LONG", "試験法特別法"),
            law("LONG_ORDER", "試験法特別法施行令", "CabinetOrder"),
            law("SHORT_ORDER", "試験法施行令", "CabinetOrder"),
            law("UNMATCHED", "別名称省令", "MinisterialOrdinance"),
            law("RESERVED_ORD", "予約省令", "MinisterialOrdinance"),
            law("RESERVED_FAMILY", "予約法"),
            law("RESERVED_CHILD", "予約法施行令", "CabinetOrder")]
    selected = sources.catalogue_selection(laws, {"RESERVED_ORD"}, {"予約法"})
    by_id = {r["law_info"]["law_id"]: r for r in selected}
    assert set(by_id) == {"SHORT", "LONG", "LONG_ORDER", "SHORT_ORDER", "UNMATCHED"}
    assert by_id["LONG_ORDER"]["family_id"] == "LONG"
    assert by_id["SHORT_ORDER"]["family_id"] == "SHORT"
    assert by_id["UNMATCHED"]["family_id"] == "UNMATCHED"
    assert by_id["LONG"]["family_id"] == "LONG"
    # Reserving a subordinate law also reserves its parent and siblings.
    selected = sources.catalogue_selection(laws, {"LONG_ORDER"}, {"試験法特別法施行令"})
    assert not {"LONG", "LONG_ORDER"} & {r["law_info"]["law_id"] for r in selected}


def test_catalogue_uses_asof_revision_and_frozen_dates_not_current_metadata():
    valid = law("VALID", "有効法")
    earliest = law("EARLIEST", "開始日法", amendment_enforcement_date="2018-10-01")
    latest = law("LATEST", "終了日法", amendment_enforcement_date="2026-09-30")
    earliest["law_info"]["promulgation_date"] = "2018-09-30"
    invalid = [law("TOO_EARLY", "改正前法", amendment_enforcement_date="2018-09-30"),
               law("TOO_LATE", "将来法", amendment_enforcement_date="2026-10-01"),
               law("SCHEDULED", "予定法", amendment_enforcement_date=None,
                   amendment_scheduled_enforcement_date="2024-04-01"),
               law("UNENFORCED", "未施行法", current_revision_status="UnEnforced"),
               law("PARTIAL", "改正指示法", mission="Partial"),
               law("REPEALED", "廃止法", amendment_type="8", repeal_status="Repeal"),
               law("NO_REVISION", "無改正法", amendment_type="1"),
               law("NEW", "新法")]
    invalid[-1]["law_info"]["promulgation_date"] = "2018-10-01"
    # Eligible current metadata must not rescue an ineligible as-of revision.
    for row in invalid:
        row["current_revision_info"] = copy.deepcopy(valid["revision_info"])
    selected = sources.catalogue_selection(invalid + [valid, earliest, latest], set(), set())
    assert {r["law_info"]["law_id"] for r in selected} == {"VALID", "EARLIEST", "LATEST"}
    assert next(r for r in selected if r["law_info"]["law_id"] == "VALID")["revision_info"]["law_title"] == "有効法"
    custom = {"promulgation_at_most": "2010-01-01", "enforcement_after": "2020-01-01",
              "enforcement_at_most": "2024-12-31", "law_types": ["Act"],
              "catalogue_statuses": ["PreviousEnforced"]}
    assert [r["law_info"]["law_id"] for r in sources.catalogue_selection([valid, earliest, latest], set(), set(), protocol=custom)] == ["VALID"]


def test_catalogue_shuffle_is_seed42_per_type_and_input_order_independent():
    laws = [law(f"A{i}", f"法{i}") for i in range(5)] + [law(f"C{i}", f"令{i}", "CabinetOrder") for i in range(5)]
    first = sources.catalogue_selection(laws, set(), set(), count_per_type=2)
    second = sources.catalogue_selection(list(reversed(laws)), set(), set(), count_per_type=2)
    assert [r["law_info"]["law_id"] for r in first] == ["A3", "A1", "C3", "C1"]
    assert [r["law_info"]["law_id"] for r in second] == ["A3", "A1", "C3", "C1"]


def test_reserved_short_act_does_not_capture_a_distinct_longer_act_family():
    laws = [law("RESERVED", "試験法"), law("LONG", "試験法特別法"),
            law("LONG_ORDER", "試験法特別法施行令", "CabinetOrder"),
            law("RESERVED_ORDER", "試験法施行令", "CabinetOrder")]
    selected = sources.catalogue_selection(laws, {"RESERVED"}, {"試験法"})
    assert {r["law_info"]["law_id"] for r in selected} == {"LONG", "LONG_ORDER"}


def test_reserved_title_prefix_is_excluded_when_root_is_absent_from_catalogue():
    laws = [law("CHILD", "予約法施行令", "CabinetOrder"), law("OTHER", "その他法")]
    selected = sources.catalogue_selection(laws, {"ABSENT_ROOT"}, {"予約法"})
    assert [r["law_info"]["law_id"] for r in selected] == ["OTHER"]
