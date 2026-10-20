# HiDate: Preserving Historical Relevance for Date-Grounded Legal Evidence Retrieval

**Huiqin Tang\*** and **Yi Zhu\*†**

\* Equal contribution. † Corresponding author: Yi Zhu.

HiDate retrieves Japanese legal evidence for a specified date. It uses historical text to discover relevant legal addresses, ranks those addresses with revision-aware features and cross-encoder scores, and returns evidence bound to the requested date. This repository provides the research implementation, experiment configurations, and expert-authored benchmark.

## Code and data

| Resource | Contents |
| --- | --- |
| [Method implementation](src/hidate) | Historical retrieval, legal-address ranking, revision binding, and evaluation |
| [Experiment scripts](scripts) | Source preparation, indexing, model inference, fitting, and analysis |
| [Expert benchmark](data/expert-benchmark) | Public query inputs, reference annotations, date pairs, and supporting source excerpts |
| [Configurations](configs) | Source selection, data preparation, retrieval budgets, and experiment protocols |
| [Model metadata](models) | Model identifiers, pinned revisions, and inference conventions |
| [Reproduction guide](docs/reproduction.md) | Dependency setup and staged experiment commands |
| [PBS examples](tools/pbs) | CPU and GPU batch jobs for the research pipeline |
| [Software tests](tests) | Small fixtures for retrieval, date binding, and evaluation |

## Installation

Use Python 3.10 for the recorded model environment. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

On Windows PowerShell, activate the environment and enable UTF-8 for Python subprocesses:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
```

Run the CPU tests for expert evidence metrics and saved-prediction analysis:

```bash
python -X utf8 -m pytest -q tests/test_expert_metrics.py tests/test_expert_analysis.py
```

The `-X utf8` option handles the Japanese fixtures consistently on Windows.

For model inference and the learned ranking stages:

```bash
python -m pip install -r requirements-models.txt
python -m pip install -e ".[ranking]"
```

With the model and ranking dependencies installed, run the full software suite with `python -X utf8 -m pytest -q`.

The recorded model environment used Python 3.10.19, PyTorch 2.5.0, and CUDA 12.4. Package versions are recorded in [environment/observed-environment.json](environment/observed-environment.json). LightGBM 4.6.0 is specified separately by the ranking implementation. Plotting dependencies are in [requirements-plots.txt](requirements-plots.txt) and use a separate environment.

## Expert benchmark

The benchmark is available directly in [data/expert-benchmark](data/expert-benchmark). It contains 240 scenarios, 320 dated queries, and 80 before/after date pairs.

When using the separate submission archives, extract `Code.zip` as the project root and extract `Expert_Dataset.zip` into its `data/` directory.

- [`unlabeled/queries.jsonl`](data/expert-benchmark/unlabeled/queries.jsonl) supplies the question text and target date used for retrieval.
- [`unlabeled/target-dates.json`](data/expert-benchmark/unlabeled/target-dates.json) lists the dates needed for source preparation.
- [`sealed/gold.jsonl`](data/expert-benchmark/sealed/gold.jsonl) contains reference answers and required evidence sets for evaluation.
- [`sealed/date-pairs.jsonl`](data/expert-benchmark/sealed/date-pairs.jsonl) identifies paired date instances.
- [`annotated-packet`](data/expert-benchmark/annotated-packet) contains the annotation tables and supporting official-source excerpts.

Retrieval uses the unlabeled inputs; the public reference labels in `sealed/` are joined during evaluation.

At an evidence budget of four, the evaluation measures annotated-address recall, revision-correct set coverage, complete-set retrieval, and the reciprocal rank of the first complete set. Required items within a set are jointly necessary; different sets are alternative sufficient evidence. Insufficient-evidence and temporal-nonapplicability cases retain their coverage labels and do not enter provision-set metric means. See [the metric implementation](src/hidate/expert_metrics.py) and [its tests](tests/test_expert_metrics.py).

## Reproducing the experiments

The pipeline proceeds through source acquisition, corpus preparation, encoding, ranking-model development, and held-out retrieval. Full experiments generate the corpus indexes, fitted ranking models, predictions, and result tables. Model weights and generated experiment artifacts are acquired or produced by these stages.

Inspect the expert retrieval interface with:

```bash
python scripts/run_expert_retrieval.py --help
python scripts/analyze_expert_retrieval.py --help
```

The [reproduction guide](docs/reproduction.md) maps each stage to its scripts and gives the expert benchmark commands. Run the scripts directly on a machine with suitable CPU/GPU resources, or use the optional PBS examples for cluster execution.

## Source attribution

Official legal materials and external model resources retain their respective terms. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) records the upstream sources, model identifiers, and attribution information. The annotation packet records the official sources used for each evidence label.
