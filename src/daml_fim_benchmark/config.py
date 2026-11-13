"""Run configuration and path resolution.

Paths anchor on a configurable benchmark root (``DAML_FIM_BENCHMARK_ROOT``,
default: current working directory).
"""

import os
import re
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from types import SimpleNamespace

# --- Experiment configuration ------------------------------------------------


@dataclass
class CvFold:
    """One cross-validation fold: a model plus the repos that are in-domain for it
    (the repos it was trained on). Every other repo present in the eval set is
    treated as out-of-domain for this fold."""

    model_name: str
    in_domain_repos: list[str] = field(default_factory=list)


@dataclass
class EvaluationConfig:
    """Typed run/experiment configuration.

    Write experiment overrides as terse ``SimpleNamespace``s and apply them with
    :meth:`merge`, which copies these defaults and rejects unknown keys (so typos
    in experiment configs fail loudly instead of being silently ignored).
    """

    # fmt: off
    model_name: str                              = ""                  # required: a HF repo id, a model dir, or a LoRA adapter dir
    cv_name: str | None                          = None                # CV grouping/display label
    cv_folds: list[CvFold]                       = field(default_factory=list)   # CV interface: a model + in_domain_repos per fold
    eval_name: str | None                        = None                # Optional explicit eval run-name prefix
    provider: str                                = "huggingface"
    in_domain_repos: list[str] | None            = None                # Repos the model trained on -> ID; every other eval repo -> OOD
    reasoning_effort: str                        = "low"
    max_samples: int | None                      = None                # Legacy fallback when samples_per_repo is unset
    samples_per_repo: int                        = 50
    max_output_tokens: int                       = 128                 # Default; dynamically increased for longer targets
    data_dir: str                                = "fim/data"
    dataset_version: str                         = "vpublic"           # Pinned dataset; also accepts "latest" or other versions
    logs_dir: str                                = "fim/logs"
    prompt_type: str | None                      = None                # L2R, PSM, SPM, IPF, 1S, CUSTOM_INSTR
    parallel: bool | None                        = None                # Model-dependent default when None
    syntax_ratio: float                          = 0.5
    eval_mode: str                               = "all"               # Deprecated; evaluation always uses unified "all"
    eval_on_ood: bool | None                     = None                # Legacy compatibility; ignored
    use_chat_template_for_fim: bool | None       = None                # Auto-resolved by model family; can be overridden
    random_seed: int                             = 42
    print_completions: bool                      = False
    vllm_base_url: str                           = "http://localhost:8001/v1"
    vllm_model_name: str                         = "models/current_model"
    vllm_served_model_name: str | None           = None                # Explicit served name when model_name targets a LoRA module
    vllm_tokenizer_name: str | None              = None                # Explicit tokenizer path/repo for vLLM launch
    vllm_api_key: str                            = "EMPTY"
    vllm_timeout_seconds: float                  = 120.0
    vllm_timeout_per_output_token_seconds: float = 0.05                # Extra timeout budget per requested output token
    vllm_timeout_max_seconds: float              = 480.0               # Upper bound for per-request timeout after scaling
    vllm_timeout_retries: int                    = 2                   # Retry transient per-request vLLM/API timeouts
    vllm_timeout_retry_backoff_seconds: float    = 0.5                 # Linear backoff factor between timeout retries
    vllm_start_timeout_seconds: float            = 300.0
    vllm_max_workers: int                        = 64
    vllm_temperature: float                      = 0.0
    vllm_top_p: float                            = 1.0
    vllm_auto_launch: bool                       = True
    vllm_launch_model: str | None                = None                # If set, must be a merged/full model dir
    vllm_launch_port: int | None                 = None                # Defaults to the port from vllm_base_url
    vllm_dtype: str                              = "float16"
    vllm_quantization: str | None                = None                # vLLM --quantization method, e.g. "fp8" (online quantization; FP8 needs Ada/Hopper GPUs)
    vllm_gpu_memory_utilization: float           = 0.90
    vllm_max_model_len: int                      = 8192
    vllm_max_num_seqs: int | None                = None                # If None, capped to a safe value from vllm_max_workers
    vllm_prefer_binary: bool                     = True                # Prefer external `vllm serve` over `python -m vllm`
    vllm_binary_path: str                        = "~/.local/bin/vllm"
    vllm_merge_parent_dir: str                   = "/tmp"
    vllm_trust_remote_code: bool                 = True                # Allow custom modeling code from the model repo (vLLM and merge). Disable for untrusted submissions.
    vllm_use_lora_for_large_adapters: bool       = True                # For >=7B adapters, prefer vLLM LoRA over merge
    vllm_lora_min_model_size_b: float            = 7.0                 # LoRA-serve when base model size token >= this
    vllm_lora_module_name: str | None            = None                # Internal resolved LoRA module name
    vllm_lora_module_path: str | None            = None                # Internal resolved LoRA module path
    vllm_cleanup_merged_model: bool              = True
    syntax_check_workers: int | None             = None                # Auto by default; set an int to override
    notes: str | None                            = None
    continue_on_sample_error: bool               = False               # Log per-sample failures and continue the run
    post_hoc_truncation: bool                    = False               # Apply LCP sliding window to truncate suffix bleed
    post_hoc_truncation_threshold: int           = 30                  # Min characters required for an LCP truncation match
    # Internal CV metadata, populated by fold expansion (not set by users).
    cv_parent_run_id: str                        = ""
    cv_subrun_index: int | None                  = None
    cv_subrun_count: int | None                  = None
    # fmt: on

    def merge(self, *overrides: SimpleNamespace, allow_unknown_keys: bool = False) -> "EvaluationConfig":
        """Return a copy of this config with the given overrides applied.

        Override keys not present on the dataclass raise ``ValueError`` unless
        ``allow_unknown_keys`` is set (in which case they are ignored).
        """
        valid_keys = {f.name for f in fields(self)}
        updates: dict[str, object] = {}
        for namespace in overrides:
            for key, value in vars(namespace).items():
                if key not in valid_keys:
                    if not allow_unknown_keys:
                        raise ValueError(f"Unknown config key: {key}")
                    continue
                updates[key] = value
        return replace(self, **updates)


# --- Paths -------------------------------------------------------------------


def project_root() -> Path:
    """Root directory that relative paths are resolved against.

    Defaults to the current working directory; override with the
    ``DAML_FIM_BENCHMARK_ROOT`` environment variable.
    """
    env = os.getenv("DAML_FIM_BENCHMARK_ROOT")
    return Path(env).expanduser().resolve() if env else Path.cwd()


def abs_path(rel_path: str) -> Path:
    """Convert a relative path to absolute, relative to the benchmark root."""
    return project_root() / rel_path


def get_latest_version(base_dir: Path) -> str:
    """
    Scans base_dir for versioned folders and returns the highest version.
    Accepted patterns:
      - vN (e.g. v1, v2)
      - vN_suffix (e.g. v2_iso)

    Args:
        base_dir: Directory containing versioned dataset folders

    Returns:
        Version string like "v3" or "v3_iso"

    Raises:
        FileNotFoundError: If base_dir doesn't exist
        ValueError: If no versioned datasets found
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {base_dir}")

    candidates: list[tuple[int, str]] = []
    for path in base_dir.iterdir():
        if not path.is_dir():
            continue
        match = re.match(r"^v(\d+)(?:[_-].+)?$", path.name)
        if match:
            candidates.append((int(match.group(1)), path.name))

    if not candidates:
        raise ValueError(f"No versioned datasets found in {base_dir} (expected names like v1, v2, v2_iso)")

    # Highest numeric version wins. For ties, use lexical order to keep behavior deterministic.
    return max(candidates, key=lambda item: (item[0], item[1]))[1]


def resolve_dataset_version(version: str, data_dir: str | Path) -> str:
    """
    Resolve dataset version string to actual version.

    Args:
        version: Either "latest" (auto-detect) or a pinned version like "v2"
        data_dir: Path to data directory containing versioned folders

    Returns:
        Resolved version string like "v3"
    """
    data_path = abs_path(data_dir) if isinstance(data_dir, str) else data_dir
    if version == "latest":
        return get_latest_version(data_path)
    return version


# --- Model / tokenizer helpers -----------------------------------------------


def get_base_model_for_tokenizer(model_name: str) -> str:
    """
    Get the model name to use for loading the tokenizer.
    If model_name is a local LoRA/PEFT adapter, reads its base_model from adapter_config.json.
    Otherwise returns model_name unchanged.
    """
    import json

    adapter_config_path = Path(model_name) / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            config = json.load(f)
        base_model = config.get("base_model_name_or_path")
        if base_model:
            return base_model

    # For full models, config.json might contain the base model info
    # especially if it's been converted or sharded.
    config_path = Path(model_name) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        # Check if the folder has tokenizer files. If so, we should return model_name
        # regardless of what _name_or_path says, because the tokenizer is local.
        if (Path(model_name) / "tokenizer_config.json").exists():
            return model_name

        base_model = config.get("_name_or_path")
        if base_model:
            return base_model

    # Check for fim_training_config.json, written by the FIM fine-tuning pipeline.
    fim_config_path = Path(model_name) / "fim_training_config.json"
    if fim_config_path.exists():
        try:
            with open(fim_config_path) as f:
                fim_config = json.load(f)
            base_model = fim_config.get("base_model_name")
            if base_model:
                return base_model
        except Exception:
            pass

    return model_name


def infer_default_use_chat_template_for_fim(model_name: str | None) -> bool:
    """
    Infer sensible default prompt wrapping for FIM based on model family/variant.

    Current policy:
    - Qwen base variants (non-instruct) use raw FIM prompts.
    - Everything else defaults to chat-template wrapping.
    """
    name = str(model_name or "").lower()

    # Qwen base models are typically prompted with raw FIM tokens.
    # Instruct/chat variants should still use chat templates.
    if "qwen" in name and "instruct" not in name:
        return False

    return True
