# Reproduction guide

Run commands from the repository root after completing the [installation steps](../README.md#installation). The Python implementation is provided by the `hidate` package.

## Environments and resources

- The model dependency pins are in [`requirements-models.txt`](../requirements-models.txt). They describe the recorded Python 3.10.19 / CUDA 12.4 environment.
- Install `python -m pip install -e ".[ranking]"` for the LightGBM 4.6.0 ranking implementation.
- The released model-stack JSON files identify the exact model revisions. Run `scripts/download_frozen_package_models.py` to download those revisions and update their local paths.
- Model preflight and neural encoding/scoring use CUDA. CPU stages cover source parsing, lexical indexing, learned address ranking, and saved-output analysis.
- The full corpus comes from official dated source snapshots. The expert benchmark contains the query/annotation package and supporting excerpts.

Run the commands below directly on a machine with the required CPU/GPU resources. For cluster execution, the optional examples in [`tools/pbs`](../tools/pbs) provide batch jobs for each stage. Set `HIDATE_PROJECT_ROOT` to the repository directory and `HIDATE_VENV` to the Python virtual environment; the defaults use the submission directory and its `.venv` subdirectory. Adjust the PBS queue and resource requests for your cluster. The expert-data path is `data/expert-benchmark`.

Use `python -X utf8 -m pytest -q` for the full test suite, including Japanese fixtures on Windows.

## Build the source corpus and ranking artifacts

The following stages establish the inputs required by the expert experiment. The JSON configurations specify source selection, data splits, and ranking parameters.

| Stage | Entry points | Main artifacts |
| --- | --- | --- |
| Acquire the primary dated sources | [`collect_sources.py`](../scripts/collect_sources.py), [`source-protocol.json`](../configs/source-protocol.json) | Official snapshots, revision lists, selected laws, and as-of bindings |
| Prepare the corpus and query split | [`prepare_data.py`](../scripts/prepare_data.py), [`data-protocol.json`](../configs/data-protocol.json) | Corpus passages, target addresses, exact queries, and family split |
| Acquire and check model resources | [`download_frozen_package_models.py`](../scripts/download_frozen_package_models.py), [`preflight_models.py`](../scripts/preflight_models.py) | Local model snapshots and GPU preflight |
| Build and encode the original index | [`build_index.py`](../scripts/build_index.py), [`encode_corpus.py`](../scripts/encode_corpus.py) | Chunks, lexical tokens, and document vectors |
| Prepare the two original query views | [`generate_paraphrases.py`](../scripts/generate_paraphrases.py), [`encode_queries.py`](../scripts/encode_queries.py) | Exact/generated query views and query vectors |
| Develop the original address model | [`discover_candidates.py`](../scripts/discover_candidates.py), [`fit_address_ranker.py`](../scripts/fit_address_ranker.py), [`rerank_development.py`](../scripts/rerank_development.py), [`select_development.py`](../scripts/select_development.py) | Discovery candidates, fitted ranker, and selected policy |
| Fit the original LambdaMART baseline | [`run_lambdamart.py`](../scripts/run_lambdamart.py) | Original address-feature tree model |
| Develop revision-aware ranking | [`run_revision_development.py`](../scripts/run_revision_development.py), [`run_revision_dualview.py`](../scripts/run_revision_dualview.py), [`run_revision_history_stage.py`](../scripts/run_revision_history_stage.py), [`run_revision_ce_training.py`](../scripts/run_revision_ce_training.py), [`freeze_revision_recipe.py`](../scripts/freeze_revision_recipe.py) | Fitted ranking heads, validation outputs, and frozen recipe |
| Expand to the 1,000-law bank | [`collect_extension.py`](../scripts/collect_extension.py), [`scale_worker.py`](../scripts/scale_worker.py), [`scale_protocol.json`](../configs/scale_protocol.json) | Expanded source catalogue and encoded corpus |

The source collector also reads the LawQA source list and question selection named in `configs/source-protocol.json`. The original files at the recorded upstream revision are included as test fixtures. Place copies at the collector's configured paths:

```bash
mkdir -p data/discovery/legal
cp tests/fixtures/legal/lawqa-data-law_list.json data/discovery/legal/
cp tests/fixtures/legal/lawqa-data-selection.json data/discovery/legal/
```

Their provenance and original license are described in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

For example, once those source inputs and the downloaded model snapshots are available:

```bash
python scripts/collect_sources.py --mode full --output-dir data/raw/hidate/full
python scripts/prepare_data.py --source-dir data/raw/hidate/full --output-dir data/processed/hidate
python scripts/build_index.py --prepared-dir data/processed/hidate --output-dir data/index/hidate
```

The expert pipeline consumes fitted ranking artifacts from the development stages, including the frozen recipe, original logistic-regression policy, original LambdaMART model, revision-aware models, and CE-training models. Their paths are configurable through `--freeze`, `--original-lr`, `--original-policy`, `--original-tree`, `--development`, and `--ce-models`. These are generated artifacts rather than files shipped in this repository.

## Prepare the expert-date corpus

After building the original 1,000-law corpus and its embeddings, collect the official revisions needed for the expert target dates:

```bash
python scripts/prepare_expert_dates.py all \
  --dates data/expert-benchmark/unlabeled/target-dates.json \
  --output data/raw/hidate/expert-dates

python scripts/build_expert_index.py prepare \
  --census data/raw/hidate/expert-dates \
  --index data/index/hidate/expert-benchmark/corpus
```

Encode the additional corpus texts and the public expert questions on a GPU:

```bash
python scripts/build_expert_index.py encode \
  --index data/index/hidate/expert-benchmark/corpus

python scripts/encode_queries.py \
  --queries data/expert-benchmark/unlabeled/queries.jsonl \
  --output-dir data/index/hidate/expert-benchmark/queries
```

These commands use the default model stack and preflight paths. The index builder retains the existing corpus and extends its date coverage. Expert query encoding uses the authored questions directly.

## Retrieve and evaluate

With corpus/query embeddings and the development artifacts available, run the four phases in order. The following Bash array makes the public benchmark paths explicit:

```bash
expert_args=(
  --queries data/expert-benchmark/unlabeled/queries.jsonl
  --gold data/expert-benchmark/sealed/gold.jsonl
  --index data/index/hidate/expert-benchmark/corpus
  --query-index data/index/hidate/expert-benchmark/queries
  --output runs/hidate/expert-public
)

python scripts/run_expert_retrieval.py discover "${expert_args[@]}"
python scripts/run_expert_retrieval.py prepare "${expert_args[@]}"
python scripts/run_expert_retrieval.py score "${expert_args[@]}"
python scripts/run_expert_retrieval.py evaluate "${expert_args[@]}"
```

`discover` retrieves historical and requested-date candidates. `prepare` applies the fitted address heads and prepares the text pairs. `score` runs the neural cross-encoder on a GPU. `evaluate` joins the expert labels and writes `final/predictions.jsonl` and `final/evaluation.json`. The discovery and preparation stages use unlabeled queries.

The retrieval protocol uses a discovery budget of 50, reranking pools of 10 distinct parents, and a final evidence budget of four. The frozen development recipe controls the selected method during evaluation.

Analyze the completed predictions with:

```bash
python scripts/analyze_expert_retrieval.py \
  --evaluation runs/hidate/expert-public/final/evaluation.json \
  --predictions runs/hidate/expert-public/final/predictions.jsonl \
  --date-pairs data/expert-benchmark/sealed/date-pairs.jsonl \
  --output runs/hidate/expert-public/analysis
```

This produces `analysis.json`, `retrieval-metrics.csv`, and `date-pair-retrieval.csv`. The analysis includes family-level uncertainty estimates and date-pair retrieval metrics. Date-pair completion measures retrieval of the correct evidence at both dates; answer correctness is a separate evaluation task.

## Use the evidence metrics independently

The metric function can be called directly on CPU after installing the core package:

```python
import json
from pathlib import Path
from hidate.expert_metrics import retrieval_metrics

gold_by_id = {
    row["query_id"]: row
    for row in map(json.loads, Path(
        "data/expert-benchmark/sealed/gold.jsonl"
    ).read_text(encoding="utf-8").splitlines())
}

# Replace with your ranked corpus chunks for this query.
ranked_chunks = []
metrics = retrieval_metrics(gold_by_id["SC001_B"], ranked_chunks, k=4)
print(metrics)
```

Each retrieved statute chunk supplies `source_kind="law"`, `law_id`, `scope`, `article_num`, `slot_id`, and `revision_id`, matching the corpus schema. An empty ranking scores zero on provision-eligible queries. Insufficient and temporal-nonapplicability records return `None` for the four provision-set metrics.
