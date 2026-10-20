import copy
from collections import Counter

from hidate import data as data
from hidate.sources import extract_provisions


DATES = ["2018-09-30", "2021-09-30", "2024-09-30"]


def node(tag, *children, **attrs):
    return {"tag": tag, "attr": attrs, "children": list(children)}


def binding(date, law_id="LAW", **extra):
    return {"law_id": law_id, "requested_asof": date, "revision_id": law_id + "_" + date,
            "status": "collected", "source_role": "full", "family_id": "FAMILY", **extra}


def provision(date, num="1", text="使用する条文。", law_id="LAW", **extra):
    slot = f"{law_id}:MainProvision:{num}"
    return {"law_id": law_id, "requested_asof": date, "scope": "MainProvision",
            "article_num": num, "revision_id": law_id + "_" + date, "slot_id": slot,
            "tree_path": [1, 0], "text": text, "visible_text": text,
            "visible_paragraphs": [{"num": "1", "text": text, "tree_path": [1, 0, 1]}],
            "deleted": False, "hidden": False, "source_role": "full", "family_id": "FAMILY",
            "passage_id": f"{law_id}_{date}|{slot}|1.0", "raw_path": f"snapshots/{law_id}/{date}.json", **extra}


def test_single_family_allocation_is_14_3_3_and_keeps_siblings_together():
    selected = [{"family_id": f"F{i:02d}", "law_info": {"law_id": f"LAW{i}"}} for i in range(20)]
    selected.append({"family_id": "F02", "law_info": {"law_id": "F02_ORDER"}})
    allocation = data.family_split(selected)
    assert Counter(allocation.values()) == {"train": 14, "valid": 3, "test": 3}
    assert {family for family, split in allocation.items() if split == "valid"} == {"F02", "F16", "F07"}
    assert data.family_split(list(reversed(selected))) == allocation


def test_visibility_prunes_flagged_descendants_and_keeps_raw_source_and_paths():
    article = node("Article", node("ArticleTitle", "第一条"), node("ArticleCaption", "（見出し）"),
        node("Paragraph", node("ParagraphNum"), node("ParagraphSentence", node("Sentence", "可視の導入。")),
             node("Item", node("ItemTitle", "一"), node("ItemSentence", node("Sentence", "可視項目。")), Hide="false", Delete="false"),
             node("Item", node("ItemTitle", "二"), node("ItemSentence", node("Sentence", "非表示項目。")), Hide="true"), Num="1"),
        node("Paragraph", node("ParagraphSentence", node("Sentence", "削除属性段落。")), Num="2", Delete="true"),
        node("Paragraph", node("ParagraphSentence", node("Sentence", "非表示段落。")), Num="3", Hide="true"),
        node("Paragraph", node("ParagraphSentence", node("Sentence", "削除")), Num="4"), Num="12_2")
    snapshot = {"law_info": {"law_id": "LAW"}, "revision_info": {"law_revision_id": "REV", "law_title": "試験法"},
                "law_full_text": node("Law", node("LawBody", node("MainProvision", article)))}
    original = copy.deepcopy(snapshot)
    record = extract_provisions(snapshot)[0]
    visible = data.visible_provision(record, snapshot)
    assert visible["text"] == record["text"] and "非表示項目。" in visible["text"]
    assert visible["visible_text"] == "可視の導入。\n一可視項目。"
    assert visible["visible_paragraphs"] == [{"num": "1", "text": "可視の導入。\n一可視項目。", "tree_path": [0, 0, 0, 2]}]
    assert visible["passage_id"] == "REV|LAW:MainProvision:12_2|0.0.0"
    assert snapshot == original
    inherited = data.visible_provision(dict(record, hidden=True), snapshot)
    assert inherited["visible_text"] == "" and inherited["visible_paragraphs"] == []
    assert inherited["text"] == record["text"]


def test_visibility_keeps_empty_table_cell_position_when_cell_is_hidden():
    table = node("AppdxTable", node("AppdxTableTitle", "別表"), node("TableStruct", node("Table",
        node("TableRow", node("TableColumn", "名称"), node("TableColumn", "非表示の値", Hide="true"),
             node("TableColumn", "数量")))))
    snapshot = {"law_info": {"law_id": "LAW"}, "revision_info": {"law_revision_id": "REV", "law_title": "試験法"},
                "law_full_text": node("Law", node("LawBody", table))}
    record = extract_provisions(snapshot)[0]
    visible = data.visible_provision(record, snapshot)
    assert visible["visible_text"] == "名称\t\t数量"
    assert visible["text"] == "名称\t非表示の値\t数量"


def test_address_index_keeps_duplicate_paths_dated_bindings_and_supplementary_paragraphs():
    records = [provision(DATES[0]), provision(DATES[0], tree_path=[2, 0]),
               provision(DATES[1], revision_id="LAW_2018-09-30"),
               provision(DATES[0], num=None, scope="SupplProvision:original", slot_id="LAW:SupplProvision:original:Paragraph_1"),
               provision(DATES[0], num=None, scope="SupplProvision:original", slot_id="LAW:SupplProvision:original:Paragraph_2")]
    index = data.address_index(records, [binding(DATES[0]), binding(DATES[1], revision_id="LAW_2018-09-30")])
    key = ("LAW", DATES[0], "MainProvision", "1")
    assert [r["tree_path"] for r in index["addresses"][key]] == [[1, 0], [2, 0]]
    assert len(index["addresses"][("LAW", DATES[1], "MainProvision", "1")]) == 1
    assert ("LAW", DATES[0], "SupplProvision:original", "Paragraph_1") in index["addresses"]
    assert ("LAW", DATES[0], "SupplProvision:original", "Paragraph_2") in index["addresses"]
    assert index["bindings"][("LAW", DATES[1])]["revision_id"] == "LAW_2018-09-30"


def test_pairs_use_real_visible_changes_and_keep_failure_ambiguity_census():
    records = [provision(DATES[0], "1", "旧条文。"), provision(DATES[1], "1", "新条文。"),
               provision(DATES[0], "2", "共同　法カ\u3099。"), provision(DATES[1], "2", "共同法ガ。"),
               provision(DATES[0], "3", "旧有効条文。"), provision(DATES[1], "3", "削除", deleted=True, visible_text=""),
               provision(DATES[0], "4", "後にない条文。"),
               provision(DATES[0], "5", "重複対象条文。"), provision(DATES[1], "5", "第一の原文。"),
               provision(DATES[1], "5", "第二の原文。", tree_path=[2, 0]),
               provision(DATES[0], "6", "Ａ幅の文字。"), provision(DATES[1], "6", "A幅の文字。"),
               provision(DATES[0], law_id="EXTERNAL", source_role="external", family_id="EXT")]
    bindings = [binding(DATES[0]), binding(DATES[1]), binding(DATES[2], status="collection_failure"),
                binding(DATES[0], law_id="EXTERNAL", source_role="external", family_id="EXT")]
    index = data.address_index(records, bindings)
    pairs = data.address_pairs(index, DATES, {"FAMILY": "train", "EXT": "test"})
    def pair(num, source=DATES[0], target=DATES[1]):
        return next(r for r in pairs if r["source_key"] == ("LAW", source, "MainProvision", num) and r["target_asof"] == target)
    assert pair("1")["stratum"] == "changed_cross_date"
    assert pair("2")["stratum"] == "unchanged_cross_date"
    assert pair("6")["stratum"] == "changed_cross_date"  # NFKC would erase this source difference.
    assert (pair("3")["stratum"], pair("3")["target_status"]) == ("unavailable_target", "deleted")
    assert pair("4")["target_status"] == "absent"
    assert pair("5")["stratum"] == "ambiguous"
    assert pair("1", target=DATES[2])["stratum"] == "collection_failure"
    assert pair("1", target=DATES[0])["stratum"] == "same_date"
    assert pair("1")["split"] == "train" and pair("1")["source_role"] == "full"
    assert pair("1")["target_revision_id"] == "LAW_2021-09-30"
    assert all(r["source_asof"] <= r["target_asof"] for r in pairs)
    assert all(r["law_id"] != "EXTERNAL" for r in pairs)
    assert all("source_record" not in r and "target_record" not in r and "text" not in r for r in pairs)


def test_ambiguous_source_keeps_known_revision_without_selecting_a_passage():
    records = [provision(DATES[0]), provision(DATES[0], tree_path=[2, 0]), provision(DATES[1])]
    index = data.address_index(records, [binding(DATES[0]), binding(DATES[1])])
    pairs = data.address_pairs(index, DATES[:2], {"FAMILY": "valid"})
    ambiguous = next(r for r in pairs if r["source_asof"] == DATES[0] and r["target_asof"] == DATES[1])
    assert ambiguous["stratum"] == "ambiguous" and ambiguous["source_status"] == "ambiguous"
    assert ambiguous["source_revision_id"] == "LAW_2018-09-30"
    assert ambiguous["source_passage_id"] is None
    selected, _ = data.select_query_pairs([ambiguous])
    assert selected == []


def candidate(num, source, stratum="changed_cross_date", family_id="FAMILY", law_id="LAW"):
    return {"source_key": (law_id, source, "MainProvision", num),
            "target_key": (law_id, DATES[2], "MainProvision", num),
            "source_asof": source, "target_asof": DATES[2], "law_id": law_id,
            "family_id": family_id, "stratum": stratum, "target_status": "available",
            "source_role": "full", "split": "train"}


def test_cap_is_unique_scoped_addresses_per_family_and_uses_latest_origin():
    pairs = [candidate(str(i % 15), date, law_id="LAW_A" if i < 15 else "LAW_B") for i in range(30) for date in DATES[:2]]
    pairs += [candidate("ignored", DATES[0], stratum="unchanged_cross_date"),
              candidate("ambiguous", DATES[0], stratum="ambiguous"),
              candidate("failed", DATES[0], stratum="collection_failure")]
    selected, counts = data.select_query_pairs(pairs)
    assert len(selected) == 25
    assert len({(r["law_id"], r["target_key"][2], r["target_key"][3]) for r in selected}) == 25
    assert all(r["source_asof"] == DATES[1] for r in selected)
    assert counts == [{"family_id": "FAMILY", "target_asof": DATES[2], "stratum": "changed_cross_date",
                       "census_pairs": 60, "eligible_addresses": 30, "selected_addresses": 25, "cap": 25}]
    second, _ = data.select_query_pairs([dict(r, model_score=99 if i % 2 else -99) for i, r in enumerate(reversed(pairs))])
    assert [r["source_key"] for r in selected] == [r["source_key"] for r in second]


def test_same_date_selection_uses_target_anchor():
    selected, _ = data.select_query_pairs([
        candidate("1", DATES[0], stratum="same_date"), candidate("1", DATES[2], stratum="same_date")])
    assert len(selected) == 1 and selected[0]["source_asof"] == DATES[2]


def test_quotes_are_exact_spans_skip_short_paragraphs_and_keep_labels_separate():
    text = "  " + "根拠となる法律の本文を正確に検索するための文章です。" * 5
    record = provision(DATES[0], text=text, visible_paragraphs=[
        {"num": "1", "text": "短い条文。", "tree_path": [1, 0, 1]},
        {"num": "2_1", "text": text, "tree_path": [1, 0, 2]}])
    target = provision(DATES[2], text="変更後の本文。")
    index = data.address_index([record, target], [binding(DATES[0]), binding(DATES[2])])
    pair = dict(candidate("1", DATES[0]), source_revision_id=record["revision_id"],
                target_revision_id=target["revision_id"], source_passage_id=record["passage_id"], target_passage_id=target["passage_id"])
    query = data.exact_quote_queries([pair], index)[0]
    assert query["query_variant"] == "exact_quote"
    assert query["source_paragraph_num"] == "2_1"
    assert query["source_tree_path"] == [1, 0, 2]
    assert query["source_span_start"] == 0 and query["source_span_end"] == 96
    assert query["query_text"] == text[:96] and len(query["query_text"]) == 96
    assert query["source_quote_text"] == query["query_text"]
    assert query["source_raw_path"] == record["raw_path"]
    assert query["labels"] == {"target_key": pair["target_key"], "target_status": "available",
                              "revision_id": target["revision_id"], "passage_id": target["passage_id"]}
    assert query["split"] == "train"
    assert "LAW" not in query["query_text"] and record["passage_id"] not in query["query_text"]
    assert data.exact_quote_queries([pair], index, min_chars=500) == []


def test_missing_target_quote_has_unavailable_label_and_no_gold_passage():
    source = provision(DATES[0], text="当該行為の内容について過去の法律が定めていた有効な条文である。")
    index = data.address_index([source], [binding(DATES[0]), binding(DATES[2])])
    pair = dict(candidate("1", DATES[0], stratum="unavailable_target"), target_status="absent",
                source_revision_id=source["revision_id"], target_revision_id="LAW_TARGET_REV",
                source_passage_id=source["passage_id"], target_passage_id=None)
    query = data.exact_quote_queries([pair], index)[0]
    assert query["labels"]["target_status"] == "absent"
    assert query["labels"]["passage_id"] is None


def test_article_change_annotation_distinguishes_unchanged_quote_and_unavailable_target():
    first = "この第一段落の本文は引用による検索に必要な長さを備えた同一の規定である。"
    source = provision(DATES[0], text=first + "\n変更前の第二段落。", visible_paragraphs=[
        {"num": "1", "text": first, "tree_path": [1, 0, 1]},
        {"num": "2", "text": "変更前の第二段落。", "tree_path": [1, 0, 2]}])
    for target_text, expected in [(first + "\n変更後の第二段落。", True),
                                  ("第一段落が全面的に改められた新しい規定である。", False), ("", None)]:
        target = provision(DATES[1], text=target_text, deleted=expected is None)
        index = data.address_index([source, target], [binding(DATES[0]), binding(DATES[1])])
        pair = next(r for r in data.address_pairs(index, DATES[:2], {"FAMILY": "test"})
                    if r["source_asof"] == DATES[0] and r["target_asof"] == DATES[1])
        query = data.exact_quote_queries([pair], index)[0]
        assert query["stratum"] == ("unavailable_target" if expected is None else "changed_cross_date")
        assert query["query_text"] == first and query["source_span_start"] == 0
        assert query["quote_present_in_target"] is expected
        assert query["source_quote_absent_from_target"] is (None if expected is None else not expected)
        assert query["labels"]["target_status"] == pair["target_status"] and query["split"] == "test"


def test_paraphrases_get_distinct_ids_with_same_labels_and_split():
    exact = {"query_id": "QUOTE_ID", "query_text": "原文。", "query_variant": "exact_quote",
             "split": "test", "labels": {"target_status": "available", "passage_id": "PASSAGE"},
             "source_key": ("LAW", DATES[0], "MainProvision", "1"), "source_quote_text": "原文。"}
    paraphrases = data.attach_paraphrases([exact], [{"base_query_id": "QUOTE_ID", "paraphrase": "この規定は何を定めるか。"}])
    assert len(paraphrases) == 1
    result = paraphrases[0]
    assert result["query_id"] != exact["query_id"] and result["base_query_id"] == "QUOTE_ID"
    assert result["query_variant"] == "paraphrase" and result["query_text"] == "この規定は何を定めるか。"
    assert result["split"] == "test" and result["labels"] == exact["labels"]
    assert result["source_key"] == exact["source_key"] and result["source_quote_text"] == "原文。"
    assert exact["query_text"] == "原文。"
