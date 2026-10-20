# HiDate expert-authored Japanese temporal legal retrieval benchmark

This directory contains the expert-authored benchmark described in the HiDate manuscript: **240 base scenarios from 40 legal families, 320 dated query instances across 29 requested dates, 80 reviewed before/after pairs, and 339 scoped, version-linked evidence annotations**. The benchmark asks systems to retrieve evidence and answer Japanese legal scenarios at a specified date.

The benchmark is available as UTF-8 CSV and JSONL tables, with JSON metadata and dated official-source text snapshots. The separate `Expert_Dataset.zip` distribution contains the same benchmark records.

| Location | Contents and role |
| --- | --- |
| [`expert-benchmark/unlabeled/queries.jsonl`](expert-benchmark/unlabeled/queries.jsonl) | The 320 retrieval/reader inputs: query ID, Japanese query text, and requested date. |
| [`expert-benchmark/unlabeled/target-dates.json`](expert-benchmark/unlabeled/target-dates.json) | The 29 distinct dates requested by the queries. |
| [`expert-benchmark/sealed/gold.jsonl`](expert-benchmark/sealed/gold.jsonl) | The 320 matching reference records, including answers, answerability, evidence sets, and review metadata. |
| [`expert-benchmark/sealed/date-pairs.jsonl`](expert-benchmark/sealed/date-pairs.jsonl) | The 80 before/after pairs: 40 with changed answers and 40 with unchanged answers. |
| [`expert-benchmark/annotated-packet/`](expert-benchmark/annotated-packet/) | The human-readable annotation tables, original authoring schema, source index, and 80 dated official-source text snapshots. |
| [`expert-benchmark/sealed/address-aliases.json`](expert-benchmark/sealed/address-aliases.json) and [`source-alignments.json`](expert-benchmark/sealed/source-alignments.json) | Identifier normalization and source-alignment provenance used with the released references. |

Use `unlabeled/queries.jsonl` as model input and join predictions to `sealed/gold.jsonl` by `query_id` for evaluation. The directory name `sealed` marks reference-only records; those records are publicly readable in this release and should be kept out of model inputs and training. All released query records have `split: "test"`.

The [dataset card](expert-benchmark/DATASET_CARD.md) explains the fields, evidence-set logic, answerability counts, temporal-nonapplicability case, and source paths. In particular, `local_source_path` and `readable_text` resolve relative to `expert-benchmark/annotated-packet/`; the corresponding source files are bundled there. The family source table's `before_source_path` and `after_source_path` also resolve to bundled snapshots. The `raw_source_url` and `raw_url` fields identify dated official API responses.

This `data/` directory contains benchmark inputs, reference annotations, and source evidence. Code and usage instructions are in the [repository README](../README.md). Source attribution and licensing scope are described in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
