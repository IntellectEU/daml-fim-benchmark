# Zero-Retention Evaluation: private benchmark subset

The public part of the Daml FIM benchmark ships in this repository and anyone
can run it. A further portion of the benchmark is derived from private Daml
codebases that we cannot redistribute. To let model developers measure their
models on that subset too, we operate a zero-retention evaluation service: you
send us your model, we run the same harness against a private subset on an
isolated machine, and you get the aggregate metrics back. Your weights are not
retained, and the private code never leaves our infrastructure.

## How to request an evaluation

Open a [**"Private evaluation request"** issue](https://github.com/intellecteu/daml-fim-benchmark/issues/new?template=private-eval-request.yml) on this repository — the issue form collects everything we need. 
For questions about the process, you can contact us as specified here: https://docs.catalyx.solutions/support-center/contact-support. 
Issues on this repository are public: for a request that should stay confidential, such as one for an
unreleased model, contact us the same way instead of opening an issue.
The request covers:

- The Hugging Face link to your model at the commit to evaluate (if the repo is private, we tell you which account to give read access). Once you have your report, please revoke that account's access to your repo.
- Any tokenizer or prompt-format notes (FIM token conventions, chat template).

Evaluations are run on a best-effort basis.

## Requirements

- **Files:** safetensors weights, plus plain-text metadata only (configs,
  tokenizer files, chat templates, licence, readme). Repos containing anything
  else, such as pickle checkpoints (`*.bin`, `*.pt`) or executable files, are
  rejected.
- **Architecture:** a standard architecture supported by vLLM. Models requiring
  custom modeling code (`trust_remote_code` / `auto_map`) are rejected — the
  evaluation machine runs no code from the submission.
- **Size:** models up to about 7B parameters, with fp16 or fp8 weights, are
  evaluated free of charge. For larger models, ask first. Models available only
  behind an API can't be evaluated, since the private code can't be sent to a
  third party.

## What runs, and what you get back

Your model is served with vLLM (greedy decoding, the same settings as the
public harness in this repository) against the full Daml 3 private test set.
You receive a JSON report with aggregate metrics only, like this one (dummy values):

```json
{
  "benchmark": "daml-fim private subset (zero-retention evaluation)",
  "evaluated_with": {
    "model": {
      "hf_repo": "your-org/your-daml-model",
      "revision": "0123abcd…"
    },
    "chat_template": "off",
    "dataset_version": "vprivate_daml3",
    "harness_commit": "4567cdef…",
    "vllm": "0.24.0"
  },
  "exact_match_rate": 0.31,
  "prefix_match_rate": 0.02,
  "by_category": {
    "random_span_word": 0.24,
    "random_span_line": 0.41,
    "signatory": 0.52,
    "ensure": 0.18,
    "choice_body": 0.27,
    "template_body": 0.33,
    "script_body": 0.21,
    "observer": 0.47,
    "data_type": 0.36,
    "instance_body": 0.29,
    "interface_body": 0.25
  },
  "by_source": {
    "random": 0.28,
    "syntax": 0.34
  },
  "total_examples": 800,
  "empty_response_count": 3,
  "macro_repo_average": 0.31,
  "repo_scores": [
    {
      "repo": "private_repo_01",
      "exact_match_rate": 0.3,
      "prefix_match_rate": 0.02,
      "total_examples": 400
    },
    {
      "repo": "private_repo_02",
      "exact_match_rate": 0.32,
      "prefix_match_rate": 0.02,
      "total_examples": 400
    }
  ]
}
```

Per-sample outputs are not returned: they would reveal the private benchmark
content. Syntax pass-rate metrics are currently reported on the public subset
only.

## The zero-retention guarantee

- Your weights are downloaded onto a single-purpose evaluation VM created for
  your run, and stored only on its local SSD: ephemeral storage that cannot be
  snapshotted or copied, and that ceases to exist with the VM. Before the
  evaluation starts, the machine's network egress is disabled entirely.
- After the run, the aggregate metrics report is extracted and the VM is
  deleted, together with its local SSD. No copy of your weights or of your
  model's predictions is kept anywhere.
- We retain only the metrics report we send you, the aggregate summary it was
  made from, and a record of your model id and revision, who requested the
  evaluation, and when. These records, including any earlier versions of them,
  are deleted 90 days after the evaluation. Deleted data may remain
  recoverable for up to 7 days under our storage's soft-delete setting.
