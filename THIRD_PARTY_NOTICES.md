# Third-party sources and attribution

This notice describes the source material and resource metadata in the HiDate repository. It covers the expert benchmark's 80 dated Japanese statute text snapshots, the upstream test fixtures, and references to external corpora and pretrained models.

## Japanese official statute texts

**Source attribution:** Japanese government, e-Gov Law Search / e-Gov Law API v2.

- [e-Gov Law Search](https://laws.e-gov.go.jp/)
- [e-Gov Law API v2](https://laws.e-gov.go.jp/api/2)
- [e-Gov Law Search usage terms](https://laws.e-gov.go.jp/terms/)
- [Public Data License 1.0](https://www.digital.go.jp/resources/open_data/public_data_license_v1.0)

The [readable source index](data/expert-benchmark/annotated-packet/readable-source-index.json) provides the exact official URL, law ID, revision ID, and requested date for each of the 80 expert-benchmark snapshots. The [family source table](data/expert-benchmark/annotated-packet/06_official_family_sources.csv) records the 40 laws and their before/after source references. Individual [evidence annotations](data/expert-benchmark/annotated-packet/03_evidence_labels.csv) retain their official source links and supporting text.

**Processing by the HiDate authors:** extraction of readable statute text from dated official responses, addition of provision-address markers and source headers, selection of supporting passages, and alignment of benchmark annotations to source addresses and revisions. The benchmark questions, reference answers, and review notes are author annotations. The readable snapshots are processed source representations, not original API JSON responses or official government statements about the benchmark.

Two unmodified e-Gov API responses are also included as test fixtures for the Labor Standards Act (`322AC0000000049`):

| Fixture | Recorded revision ID |
| --- | --- |
| [`tests/fixtures/legal/egov-law-version-0.json`](tests/fixtures/legal/egov-law-version-0.json) | `322AC0000000049_20260717_508AC0000000060` |
| [`tests/fixtures/legal/egov-law-version-1.json`](tests/fixtures/legal/egov-law-version-1.json) | `322AC0000000049_20260624_508AC0000000046` |

The original acquisition metadata records PDL 1.0 for the e-Gov source. Official source material retains its applicable original terms, including source-specific notices. Consult the linked e-Gov terms and Public Data License for reuse of official-site content. Attribution to the official sources should accompany reuse of the included extracts, with the processing identified.

The source-acquisition scripts obtain the full Japanese retrieval corpus, raw API responses, FSA/MHLW guidance collections, and WARP captures from the corresponding public sources. The benchmark includes the readable official-source snapshots linked above, with their source URLs and dated revision identifiers.

## LawQA-JP test fixtures

**Attribution:** Digital Agency, Japan, **lawqa_jp**; Yukio Uematsu and Naoya Ohsugi, *Ground Truth Curation for Legal QA Tasks Using Multiple LLMs* (NLP2025).

- [Official repository](https://github.com/digital-go-jp/lawqa_jp)
- Frozen revision: `fce599797ce9b582045a32a5710d9197e7200959`
- [Original README at the recorded revision](https://github.com/digital-go-jp/lawqa_jp/blob/fce599797ce9b582045a32a5710d9197e7200959/README.md)
- License: PDL 1.0, preserved in [`tests/fixtures/legal/lawqa-LICENSE.md`](tests/fixtures/legal/lawqa-LICENSE.md).

The test fixtures are unmodified copies of upstream `data/selection.json` and `data/law_list.json`, stored here as [`lawqa-data-selection.json`](tests/fixtures/legal/lawqa-data-selection.json) and [`lawqa-data-law_list.json`](tests/fixtures/legal/lawqa-data-law_list.json). They contain the released 140 question records and 24-law source list. Their original question text, choices, released answers, and source fields are retained. Their inclusion as fixtures does not add them to the HiDate expert benchmark or its training data.

## Fiscal-FR-Bench / FiscalQA Pro scorer fixture

**Attribution:** **FiscalQA Pro / Fiscal-FR-Bench v0 (Talia)**; Rose Cymbler, Daniel Guez, and Laurent Fabre, *Temporal Misgrounding in Legal RAG: A Versioned-Corpus Benchmark for French Tax Law*.

- [Official repository](https://github.com/rosecymbler/fiscal-fr-bench)
- Frozen revision: `6ca3807d46926f6f5b355d2d435b4baa05300bdf`
- Included upstream code: [`tests/fixtures/fiscalqa/scripts/score_nuggets.py`](tests/fixtures/fiscalqa/scripts/score_nuggets.py), copied without modification.
- Code license: MIT, with the original copyright and permission notice preserved in [`tests/fixtures/fiscalqa/LICENSE`](tests/fixtures/fiscalqa/LICENSE).

The FiscalQA benchmark questions, nuggets, and legislation corpus are obtained separately from the upstream project. The included license distinguishes its MIT-licensed source code from its separately licensed datasets. HiDate's evaluation scripts and configurations reference those resources; no FiscalQA question or corpus payload is bundled here.

## Pretrained model metadata

The repository includes two metadata files: [`models/ruri/model-stack.json`](models/ruri/model-stack.json) and [`models/qwen3/model-stack.json`](models/qwen3/model-stack.json). They identify the study's pretrained models and frozen Hub revisions:

- `cl-nagoya/ruri-v3-310m`
- `cl-nagoya/ruri-v3-reranker-310m`
- `Qwen/Qwen3-Embedding-0.6B`
- `Qwen/Qwen3-Reranker-0.6B`
- `Qwen/Qwen2.5-7B-Instruct`

The metadata records `apache-2.0` for these model resources and links their source model cards at the recorded revisions. Download the model weights from their original repositories using the supplied acquisition script, and retain the applicable model licenses and notices.

