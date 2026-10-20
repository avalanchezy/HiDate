# HiDate expert benchmark dataset card

## Task and composition

The HiDate expert benchmark evaluates Japanese legal evidence retrieval and answering at a requested date. Each query combines a scenario's facts and question with `target_asof`, the date the answer must address. Reviewed references connect answers to a law, a scoped provision address, and a dated revision.

The released payload contains the following records:

| Component | Count |
| --- | ---: |
| Base scenarios | 240 |
| Legal families | 40 |
| Dated query instances and matching gold records | 320 |
| Distinct requested dates | 29 |
| Single-date scenarios | 160 |
| Paired-date scenarios | 80 |
| Before/after pairs | 80 |
| Pairs with a changed / unchanged answer | 40 / 40 |
| Evidence annotation rows | 339 |
| Dated official-source text snapshots | 80 |

The requested dates range from `2018-12-29` to `2026-08-12`. The four scenario categories contain 60 base scenarios each: `old_terms`, `real_world`, `changed_conditions`, and `insufficient_evidence`. Their dated-instance counts are 90, 90, 80, and 60, respectively. The 80 paired scenarios each contribute two instances; the other 160 scenarios contribute one each.

## Files

Paths in this table are relative to this dataset card.

| File | Role |
| --- | --- |
| [`unlabeled/queries.jsonl`](unlabeled/queries.jsonl) | Model inputs: 320 dated questions. |
| [`unlabeled/target-dates.json`](unlabeled/target-dates.json) | List of the 29 requested dates. |
| [`sealed/gold.jsonl`](sealed/gold.jsonl) | Matching answers, answerability labels, evidence, and review metadata. |
| [`sealed/date-pairs.jsonl`](sealed/date-pairs.jsonl) | Reviewed before/after pair definitions and answer relations. |
| [`sealed/address-aliases.json`](sealed/address-aliases.json) | 19 source-identifier normalization mappings, with original and canonical addresses. |
| [`sealed/source-alignments.json`](sealed/source-alignments.json) | Source-alignment provenance for corrected evidence and the temporal-nonapplicability case. |
| [`annotated-packet/01_scenarios_author.csv`](annotated-packet/01_scenarios_author.csv) | 240 base questions, facts, category labels, assigned laws, and authorship confirmations. |
| [`annotated-packet/02_date_instances_review.csv`](annotated-packet/02_date_instances_review.csv) | 320 dated instances, reference answers, answerability, and review fields. |
| [`annotated-packet/03_evidence_labels.csv`](annotated-packet/03_evidence_labels.csv) | 339 evidence annotations with addresses, versions, source links, and support spans. |
| [`annotated-packet/04_real_date_pairs.csv`](annotated-packet/04_real_date_pairs.csv) | Human-readable view of the 80 date pairs. |
| [`annotated-packet/06_official_family_sources.csv`](annotated-packet/06_official_family_sources.csv) | 40 families with before/after dates, revision IDs, and official source URLs. |
| [`annotated-packet/readable-source-index.json`](annotated-packet/readable-source-index.json) | Index of all 80 bundled source text snapshots. |
| [`annotated-packet/schema.json`](annotated-packet/schema.json) | Original annotation-form fields, allowed values, and annotation rules. |
| [`annotated-packet/source-evidence/`](annotated-packet/source-evidence/) | The 80 readable dated statute snapshots used to inspect evidence. |

CSV files use UTF-8 with a byte-order mark; JSON and JSONL files use UTF-8. The schema describes the annotation-form fields, allowed values, and review rules, including an optional disagreement table. The released tables contain completed annotations; no `05_disagreements.csv` is bundled. Runtime gold also contains the released `retrieval_target_type` and source-alignment additions described below.

## Query and reference roles

`unlabeled/queries.jsonl` is the model-input file:

| Field | Meaning |
| --- | --- |
| `query_id` | Unique dated-instance identifier, also present in gold. |
| `query_text` | Japanese facts and question with the requested date inserted. |
| `target_asof` | Requested legal date in `YYYY-MM-DD` format. |
| `query_variant` | `expert_scenario` for every released query. |
| `source_role` | Runtime corpus-role value, `full` in this release. |
| `split` | `test` for every released query. |

Join predictions to `sealed/gold.jsonl` by `query_id`; its `question_id` identifies the same dated instance. `scenario_id` links instances back to their base scenario. IDs ending in `_B`, `_A`, and `_S` denote before, after, and single-date instances, respectively.

Gold contains the reviewed `reference_answer_ja`, `answer_key_points_ja`, `answerability`, `answer_form`, `insufficiency_reason_ja`, and review metadata. It also carries `family_id`, `category`, evidence rows, and required evidence sets. `target_date` in gold corresponds to `target_asof` in the query.

The `sealed/` directory is a logical separation for evaluation references, not an access restriction. Use its answers and evidence labels for scoring after producing model outputs. The annotation CSV files also contain reference information and should be kept out of model inputs and training.

## Evidence addresses and required sets

Each evidence row identifies a versioned legal provision:

| Field or group | Meaning |
| --- | --- |
| `evidence_id`, `question_id`, `gold_set_id` | Evidence-row ID, associated dated instance, and sufficient-set membership. |
| `law_id` | Official law identifier. |
| `legal_scope` | Provision scope, such as `MainProvision` or a specifically identified supplementary provision. |
| `article_number`, `paragraph_item_path` | Article and paragraph/item location within that scope. |
| `address_role` | How the cited text supports the answer; this release contains 275 `direct_answer` and 64 `context_only` rows. |
| `revision_id`, `binding_status` | Dated source revision and recorded binding status. |
| `effective_start`, `effective_end` | Recorded legal applicability boundaries where established. A blank end date is unspecified. |
| `source_snapshot_date` | Date requested from the source API; distinct from an acquisition timestamp. |
| `official_source_url`, `effective_date_basis_url` | Official source and date-basis links. |
| `local_source_path`, `support_span_ja` | Bundled readable snapshot path and the supporting Japanese text. |
| `evidence_verified_by`, `evidence_status` | Recorded verification provenance and status. |

Keep the scope and paragraph/item path when interpreting an address: an article number alone can refer to different main or supplementary provisions. `sealed/address-aliases.json` records the supplied canonical mappings for source-identifier spelling differences. The provided retrieval metrics compare scoped article identities (`law_id`, `legal_scope`, `article_number`) and revisions. Paragraph/item paths and support spans provide finer annotation detail; these metrics do not score paragraph-level or answer-text correctness.

Within one `gold_set_id`, the required positive evidence rows are jointly needed. Different set IDs represent alternative sufficient sets. The released runtime references use `G1` for all 259 provision-eligible answerable instances. `context_only` rows are retained as supporting context and are excluded from positive retrieval targets by the evaluator. Consequently, the 339 evidence rows are not 339 independent positive retrieval targets.

## Answerability and evaluation coverage

| Released reference type | Instances | Provision-set retrieval scoring |
| --- | ---: | --- |
| `answerable` with `retrieval_target_type: "provision_set"` | 259 | Eligible |
| `answerable` with `retrieval_target_type: "temporal_nonapplicability"` | 1 | No applicable provision set |
| `insufficient_material_evidence` | 37 | No positive evidence set |
| `insufficient_facts` | 23 | No positive evidence set |
| **Total** | **320** | |

There are 260 answerable instances in total. An evidenced negative answer is still answerable. `SC128_B` asks about `2026-06-23`, before the amendment's recorded promulgation date of `2026-06-24`. It therefore has an answerable negative reference, `retrieval_target_type: "temporal_nonapplicability"`, empty `evidence` and `required_evidence_sets`, and a separate `temporal_evidence` record.

Keep this instance in overall coverage and in its released date pair. The supplied evaluator returns no provision-set structural scores for it. Provision-set summaries therefore cover 259 instances; pair summaries requiring provision targets at both dates cover 79 of the 80 released pairs. `retrieval_target_type` alone does not establish eligibility: the insufficient cases retain the default `provision_set` value but have no required evidence set.

The [expert metrics implementation](../../src/hidate/expert_metrics.py) defines the scoring semantics. These counts describe the benchmark annotations, not measured model performance.

## Official sources and provenance paths

The source texts come from the Japanese government's e-Gov Law API v2. The [source index](annotated-packet/readable-source-index.json) records each law, phase, requested date, revision ID, official API URL, and readable text path. Each text snapshot also starts with its law title, identifiers, requested `asof` date, revision ID, and source URL. The readable files contain extracted text with provision-address markers for inspection; they are processed representations of the official source responses.

Resolve `local_source_path` in evidence rows and `readable_text` in the source index relative to `annotated-packet/`. For example, `source-evidence/141AC0000000029_before.txt` is available at [`annotated-packet/source-evidence/141AC0000000029_before.txt`](annotated-packet/source-evidence/141AC0000000029_before.txt).

The family source table's `before_source_path` and `after_source_path` resolve relative to `annotated-packet/` and point to the bundled before/after text snapshots. The `raw_source_url` fields and `temporal_evidence.*.raw_url` identify the dated official API responses used for source alignment. `source_tree_path` preserves the node position in the corresponding JSON response. The source table's `prior_role: "retrieval_distractor"` records the legal family's role in the retrieval corpus before benchmark annotation. `source_retrieved_at` is blank when no acquisition timestamp was supplied; it should not be inferred from a target date.

The source-alignment records distinguish normalization and alignment of existing annotations from new expert ratings. They record `extra_expert_ratings_created: 0`; their presence does not add a new independent review round.

Source attribution and terms are described in [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md). The official texts retain their original terms; this dataset card assigns no additional license to the authors' questions or annotations.
