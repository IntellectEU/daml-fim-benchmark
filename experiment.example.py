"""Experiment registry for the Daml FIM benchmark.

Copy this to ``experiment.py`` (gitignored), then run:

    python -m daml_fim_benchmark.evaluate            # loads ./experiment.py
    python -m daml_fim_benchmark.evaluate --config path/to/this.py

Keep your experiments here as named ``EvaluationConfig``s — a reference log of
what you've run — and point ``EXPERIMENT`` at the one you want to run next (or
define a fresh config inline). See ``EvaluationConfig`` in
``daml_fim_benchmark.config`` for every field.
"""

from daml_fim_benchmark.config import CvFold, EvaluationConfig

# --- Named experiments (your reference log) ----------------------------------

QWEN25_CODER_7B = EvaluationConfig(
    model_name="Qwen/Qwen2.5-Coder-7B",  # HF repo id, a model dir, or a LoRA adapter dir
    samples_per_repo=50,
)

GRANITE_MICRO = EvaluationConfig(model_name="ibm-granite/granite-4.0-h-micro")

MY_FINETUNE = EvaluationConfig(
    model_name="/models/my-finetune",
    in_domain_repos=["daml-finance", "splice"],  # repos it trained on -> ID; the rest -> OOD
)

MY_CV = EvaluationConfig(
    cv_name="my-cv",
    cv_folds=[
        CvFold(model_name="/models/fold1", in_domain_repos=["repoA", "repoB"]),
        CvFold(model_name="/models/fold2", in_domain_repos=["repoC", "repoD"]),
    ],
)

# --- The experiment to run next ----------------------------------------------
# Point EXPERIMENT at one of the above, or define a fresh EvaluationConfig here.
EXPERIMENT = QWEN25_CODER_7B
# EXPERIMENT = EvaluationConfig(...)

# To run several in one go, define EXPERIMENTS instead (it takes precedence):
# EXPERIMENTS = [QWEN25_CODER_7B, GRANITE_MICRO]
