# Daml FIM Benchmark

A benchmark for **fill-in-the-middle (FIM) code autocompletion on Daml**. It evaluates how well a model completes a span of Daml code given the surrounding prefix and suffix, scores the completions (exact-match plus Daml syntax-aware metrics), and provides a web UI for browsing and comparing runs.

Part of the [Daml Code Assistant](https://github.com/canton-foundation/canton-dev-fund/blob/main/proposals/2026-06-IEU-Daml%20Code%20Assistant.md) project.

## What's here

```
src/daml_fim_benchmark/
├── config.py            # EvaluationConfig (run configuration) + path resolution
├── evaluate.py          # evaluation engine (run with `python -m daml_fim_benchmark.evaluate`)
├── serving.py           # local vLLM server lifecycle + LoRA-adapter merging
├── prompting.py         # FIM prompt construction
├── syntax_metrics.py    # Daml syntax-aware scoring (runs the Daml toolchain)
├── cross_validation.py  # cross-validation fold aggregation
└── server/app.py        # FastAPI server backing the web UI
frontend/                # React web UI (talks to the server over HTTP)
```

## What it evaluates

The benchmark evaluates **local models served with [vLLM](https://github.com/vllm-project/vllm)** (`provider="huggingface"`): a Hugging Face repo id, a merged checkpoint directory, or a LoRA adapter (merged into its base, or served as a vLLM LoRA module). The engine launches its own vLLM server, talks to it over the OpenAI-compatible API, and tears it down when the run finishes. Other providers are rejected at run time.

> Running an evaluation requires a **CUDA GPU** (vLLM); it does not run on CPU-only or Apple Silicon machines. The web UI for browsing existing runs has no such requirement.

## Install

The package uses optional extras so you only install what you need. These use
[uv](https://docs.astral.sh/uv/) (faster); plain `pip` works the same — just drop the `uv` prefix:

```bash
uv pip install -e .                      # core
uv pip install -e '.[serving]'           # + torch / transformers / vLLM (model evaluation; needs a GPU)
uv pip install -e '.[server]'            # + FastAPI server for the web UI
uv pip install -e '.[serving,server]'    # everything
```

## Dataset

Evaluation reads the FIM dataset from `fim/data/<version>/` (default version `vpublic`), expecting `syntax_test.jsonl` and `random_test.jsonl`. Each example carries a repo-relative `file_path` of the form `<repo>/…` (e.g. `daml-finance/src/main/daml/…/Factory.daml`); the leading `<repo>` segment is the **repo identifier** used throughout the benchmark — for per-repo scores and for declaring in-domain repos (see below).

Paths resolve against the benchmark root, which defaults to the current working directory and can be overridden with `DAML_FIM_BENCHMARK_ROOT`.

This benchmark publishes the **evaluation (test) split** (`syntax_test.jsonl` + `random_test.jsonl`). It ships in this repo, so the benchmark runs out of the box with no download.

### Private subset

A further part of the benchmark comes from private Daml codebases that cannot be redistributed. Model developers can have a model evaluated on it, with only aggregate metrics returned and the weights not retained: see [PRIVATE_EVALUATION.md](PRIVATE_EVALUATION.md).

### Daml 2 vs. Daml 3 subsets

Every example is tagged with its Daml language version under `metadata.daml_version` (`"2.x"`, `"3.x"`, or `"unknown"` when no SDK version could be detected), so Daml-2 and Daml-3 results can be reported by filtering on that field — no separate dataset versions needed. The `unknown` bucket is non-trivial (no SDK version was detectable in the source file), so a `2.x` subset plus a `3.x` subset does not cover every example.

### Data sources & attribution

The dataset is derived from publicly available Daml source code; each example embeds verbatim source fragments with their original copyright/SPDX headers preserved. All six source repositories are licensed under **Apache License 2.0**:

| Repo | Copyright | Source |
|---|---|---|
| `daml` | Digital Asset (Switzerland) GmbH | https://github.com/digital-asset/daml |
| `daml-finance` | Digital Asset (Switzerland) GmbH | https://github.com/digital-asset/daml-finance |
| `splice` | Digital Asset (Switzerland) GmbH | https://github.com/hyperledger-labs/splice |
| `canton` | Digital Asset (Switzerland) GmbH | https://github.com/digital-asset/canton |
| `ex-models` | Digital Asset (Switzerland) GmbH | https://github.com/digital-asset-archive/ex-models |
| `account-hierarchy` | ASX Operations Pty Ltd (Synfini) | https://github.com/SynfiniDLT/account-hierarchy |

See [`NOTICE`](NOTICE) for the full attribution.

## Running an evaluation

Run one evaluation straight from the command line — there's a flag for every `EvaluationConfig` field:

```bash
python -m daml_fim_benchmark.evaluate \
    --model-name Qwen/Qwen2.5-Coder-7B \
    --samples-per-repo 50
```

`python -m daml_fim_benchmark.evaluate --help` lists every flag. Results are written to `fim/logs/<run_id>/` (`summary.json` + per-example `results.jsonl`), which the web UI reads back.

For richer setups (cross-validation, batches, or just editor autocomplete over the full config), keep your experiments in a Python file. Copy the example and edit it — it's a reference log of named `EvaluationConfig`s where you point `EXPERIMENT` at the one to run next (or `EXPERIMENTS` at a list):

```bash
cp experiment.example.py experiment.py   # gitignored; edit it
python -m daml_fim_benchmark.evaluate     # auto-loads ./experiment.py
```

```python
# experiment.py
from daml_fim_benchmark.config import EvaluationConfig
QWEN_7B = EvaluationConfig(model_name="Qwen/Qwen2.5-Coder-7B", samples_per_repo=50)
MY_FINETUNE = EvaluationConfig(model_name="/models/run-x", in_domain_repos=["repoA"])
EXPERIMENT = QWEN_7B   # the run; point it at any named config above, or a fresh one
```

Run with no arguments to load `./experiment.py`, or `--config PATH` for a file elsewhere. (`--config` and the per-field flags are mutually exclusive.)

### In-domain / out-of-domain (ID/OOD)

If your model was trained on some of the benchmark repos, declare them so scores are split into in-domain and out-of-domain — every repo in `in_domain_repos` counts as in-domain, every other repo present in the eval set is out-of-domain (omit it and there's no ID/OOD split):

```bash
python -m daml_fim_benchmark.evaluate \
    --model-name /path/to/your/model \
    --in-domain-repos daml-finance splice
```

### Cross-validation

For a cross-validated evaluation, list one model per fold with the repos in-domain for that fold (CV needs the `--config` form). The folds run as separate evaluations and are aggregated into a single CV-parent summary:

```python
# experiment.py
from daml_fim_benchmark.config import EvaluationConfig, CvFold
EXPERIMENT = EvaluationConfig(
    cv_name="my-cv",
    cv_folds=[
        CvFold(model_name="/models/fold1", in_domain_repos=["repoA", "repoB"]),
        CvFold(model_name="/models/fold2", in_domain_repos=["repoC", "repoD"]),
    ],
)
```

## Metrics

Each run reports:
- **Exact-match rate** — the headline accuracy (per-sample / micro average).
- **Per-repo average** — exact-match averaged unweighted across repos (macro average), plus **in-domain / out-of-domain** averages when `in_domain_repos` is set.
- **Daml syntax pass rate** — fraction of completions that are syntactically valid Daml, checked via the Daml toolchain (`syntax_metrics.py`). *Optional — see below.*
- Over-generation (prediction starts with the target plus extra) and per-category / per-source breakdowns.

### Optional: the Daml syntax pass-rate metric

Exact-match (the headline metric and the proposal's scoring basis) needs **nothing beyond the Python deps**. The **syntax pass-rate** metric is optional and needs two extra things; without them it is **skipped cleanly** (the per-result `syntax_check_skip_reason` records why) and exact-match still runs:

1. **The Daml SDK** — the metric shells out to `daml damlc lint` / `daml build`:
   ```bash
   scripts/install_daml.sh        # installs the `daml` assistant (or: curl -sSL https://get.daml.com/ | sh)
   ```
   The assistant auto-fetches the SDK version pinned in each source repo's `daml.yaml` (the dataset spans several versions), so you don't choose one. Requires a JDK 11+. The assistant is deprecated in favour of `dpm` (the Digital Asset Package Manager), but the source repos are pinned to SDK versions `dpm`'s registry doesn't provide, so the assistant — which fetches those exact versions — is what the benchmark uses.

2. **The source repositories** — the check lints each completion against its source project (its `daml.yaml`). Fetch the public, Apache-2.0 source repos, pinned to the exact commits the benchmark was evaluated against:
   ```bash
   scripts/checkout_source_repos.sh           # all repos -> sources/<repo>/
   scripts/checkout_source_repos.sh splice    # or one at a time
   ```

Pinned source commits (also encoded in the script):

| Repo | Commit | Source |
|---|---|---|
| `daml` | `c69d4fd` | https://github.com/digital-asset/daml |
| `daml-finance` | `155f931` | https://github.com/digital-asset/daml-finance |
| `splice` | `e04a9d5` | https://github.com/hyperledger-labs/splice |
| `canton` | `bb8170f` | https://github.com/digital-asset/canton |
| `ex-models` | `ca6818c` | https://github.com/digital-asset-archive/ex-models |
| `account-hierarchy` | `407a020` | https://github.com/SynfiniDLT/account-hierarchy |

The syntax check yields a clean pass/fail only when an example's source project **builds** in your environment. Projects pinned to very old SDKs or with unbuilt cross-repo dependencies (e.g. `account-hierarchy` on Daml `1.17.1`) come back **inconclusive** — recorded in each result's `syntax_check_skip_reason` and excluded from the pass-rate, not guessed. Exact-match is unaffected.

## Baselines

Two reference models are evaluated on the **`vpublic` test split** (full, no per-repo cap): **GitHub Copilot** (a general-purpose assistant) and our **Qwen2.5-Coder-7B-IEU-finetune**. They ship under `fim/logs/` and load directly in the web UI: Copilot as flat **Daml-2** and **Daml-3** runs (`baseline-copilot-daml{2,3}`), the finetune in two variants — the **cross-validated** one below (`baseline-ieu-7b-daml2` + `baseline-ieu-7b-daml3`, a CV parent with two fold subruns — fold 1 trains with `splice` in-domain and holds out `canton`, fold 2 the reverse) and the **served** one (`baseline-ieu-7b-served-daml{2,3}`, see [The served model](#the-served-model) below). Exact-match on the masked span:

| Subset | n | GitHub Copilot | Qwen2.5-Coder-7B-IEU-finetune | Δ (95% CI) | McNemar p |
|---|--:|--:|--:|--:|--:|
| **Daml 3** (`canton`, `splice`) | 2,579 | 28.4% | **32.7%** | **+4.3pp** [+3.0, +5.6] | 4.3e-10 |
| **Daml 2** (`daml-finance`, `account-hierarchy`, `ex-models`) | 4,914 | 17.8% | **20.9%** | **+3.1pp** [+2.2, +4.0] | 1.5e-10 |
| **Combined** (Daml-2 + Daml-3) | 7,493 | 21.4% | **25.0%** | **+3.5pp** [+2.7, +4.3] | 5.4e-19 |

*Note: The finetune is evaluated at FP8 quantization (`fp8_per_block`), which matches our serving configuration. It yields functionally identical accuracy to FP16 while being significantly faster to serve.*

*The finetune's table numbers merge each repo's **held-out fold** results (weighted over examples), so no example is scored in-domain. The web UI's CV parent row shows fold-mean aggregates instead — its unweighted OOD average (33.6% for Daml-3) is the mean of the two fold-level OOD averages, and its weighted total also includes the folds' in-domain halves.*

The finetune significantly outperforms Copilot on both Daml versions. Significance is the paired **McNemar test**, and the CI is on the paired accuracy difference. Per-repo scores are in each run's `summary.json`.

> **What the McNemar test does.** Both models run on the *same* examples, so most examples either both get right or both get wrong — those don't separate the models, so the test sets them aside. It looks only at the examples where the two *disagree* and asks whether that split is a fair coin: if the models were equally good, each disagreement would fall to either model about 50/50. On Daml-3 the disagreements break **214 to 103** in the finetune's favour (out of 317) — a fair coin landing that lopsided is very unlikely, which is what the p-value reports. The accuracy-difference CI is the same fact in percentage-point units, with its uncertainty coming only from those disagreement counts.

Methodology:
- **Full test set**, not a sample — `samples_per_repo` unset, so every example is scored.
- The finetune is a **cross-validated** model; each repo is scored on the **held-out fold** where it is out-of-domain (the Daml-2 repos were never in its training set; `canton`/`splice` come from their respective holdout folds), so no example is scored in-domain.
- Baselines cover the Daml-3 repos (`canton`, `splice`) and the three Daml-2 repos; the `daml` repo (largely untagged SDK versions) is not part of the baseline comparison.

### The served model

The cross-validated numbers above measure held-out generalization: every repo is scored by a fold model that never saw it. The model we actually serve — and publish as open weights at [huggingface.co/intellecteu/daml-fim-7b](https://huggingface.co/intellecteu/daml-fim-7b) — is a variant of the same recipe trained once on **all** of its training repos, with no fold holdout, because that makes it stronger. It ships as `baseline-ieu-7b-served-daml{2,3}`, evaluated with the identical setup (full test split, FP8 `fp8_per_block`, paired against the same Copilot runs):

| Subset | n | GitHub Copilot | IEU-finetune (served) | Δ (95% CI) | McNemar p |
|---|--:|--:|--:|--:|--:|
| **Daml 3** (`canton`, `splice`) — *in-domain* | 2,579 | 28.4% | **33.5%** | **+5.1pp** [+3.7, +6.4] | 2.0e-13 |
| **Daml 2** (3 repos) — *out-of-domain* | 4,914 | 17.8% | **21.3%** | **+3.5pp** [+2.5, +4.4] | 5.8e-13 |
| **Combined** | 7,493 | 21.4% | **25.5%** | **+4.0pp** [+3.3, +4.8] | 1.8e-24 |

For Daml 3 the served model's scores are **in-domain**: `canton` and `splice` are in its training set. The dataset's train/val/test split is file-level, so none of the evaluated files were seen in training (verified: 0 of the test files appear in the model's training data) — but unseen files from a training repo still share conventions and identifiers, which is worth about a point of exact match here (33.5% vs. the 32.7% held-out estimate above). The Daml-2 repos were never in this model's training set, so that row remains a true out-of-domain measurement. Read the two tables together: the cross-validated table is the fair estimate of how the finetune generalizes to Daml code it was never trained on; this table is what the model being served actually scores on the benchmark.

## Web UI

A React + FastAPI app for browsing, filtering, comparing, and annotating runs. See [`frontend/README.md`](frontend/README.md) for how to start it.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
