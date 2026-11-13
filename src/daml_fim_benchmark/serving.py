"""vLLM server lifecycle and LoRA-adapter staging for local FIM evaluation.

Launches and tears down a local OpenAI-compatible vLLM server and prepares the
model it serves: merging a LoRA adapter into its base model, or serving the
adapter as a vLLM LoRA module for large bases. Used by
:mod:`daml_fim_benchmark.evaluate`. The heavy ML dependencies (torch, peft,
transformers, safetensors) are imported lazily inside the functions that need
them, so importing this module stays cheap.
"""

import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from daml_fim_benchmark.config import EvaluationConfig, abs_path

HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def _looks_like_model_dir(path: Path) -> bool:
    """True if ``path`` is a full model directory — a config.json plus weights
    (safetensors or pytorch_model.bin) that vLLM can serve directly.

    Example (True)::

        my-model/
          config.json
          model.safetensors          # or model.safetensors.index.json, or pytorch_model.bin
          tokenizer.json

    Returns False for an adapter directory (config.json but no model weights, only
    adapter_model.safetensors) or any path missing config.json.
    """
    if not path.exists() or not path.is_dir():
        return False
    if not (path / "config.json").exists():
        return False
    return (
        (path / "model.safetensors").exists()
        or (path / "model.safetensors.index.json").exists()
        or (path / "pytorch_model.bin").exists()
    )


def _looks_like_adapter_dir(path: Path) -> bool:
    """True if ``path`` is a LoRA/PEFT adapter directory (has adapter_config.json).

    Example (True)::

        my-adapter/
          adapter_config.json        # records base_model_name_or_path
          adapter_model.safetensors

    Returns False for a full model directory (which has config.json but no
    adapter_config.json) — see :func:`_looks_like_model_dir`.
    """
    return path.exists() and path.is_dir() and (path / "adapter_config.json").exists()


def _looks_like_hf_repo_id(model_name: str) -> bool:
    """Best-effort detection for a remote Hugging Face repo id like 'org/model'."""
    value = str(model_name or "").strip()
    if not value:
        return False
    # Avoid treating common local-path forms as remote model IDs.
    if value.startswith((".", "/", "~")):
        return False
    return HF_REPO_ID_RE.match(value) is not None


def resolve_vllm_launch_port(config: EvaluationConfig) -> int:
    """Port the local vLLM server is launched on: explicit ``vllm_launch_port`` if
    set, else the port parsed from ``vllm_base_url``, else 8001."""
    if config.vllm_launch_port:
        return int(config.vllm_launch_port)
    parsed = urlparse(config.vllm_base_url)
    if parsed.port:
        return int(parsed.port)
    return 8001


def _read_base_model_from_adapter(adapter_path: Path) -> str:
    """Return the adapter's base model id (``base_model_name_or_path`` from its
    adapter_config.json); raise if it's missing."""
    with open(adapter_path / "adapter_config.json", "r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    base_model = adapter_cfg.get("base_model_name_or_path")
    if not base_model:
        raise RuntimeError(f"adapter_config.json missing base_model_name_or_path: {adapter_path}")
    return str(base_model)


def _parse_model_size_b(model_name: str) -> float | None:
    """Parse a parameter-count-in-billions token from a model name (e.g. "...-7B"
    -> 7.0), or None when there's no size token. Used to decide LoRA-vs-merge."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", str(model_name))
    if not match:
        return None
    return float(match.group(1))


def _sanitize_lora_module_name(raw: str) -> str:
    """Make ``raw`` safe as a vLLM ``--lora-modules`` name (keep [A-Za-z0-9._-],
    collapse the rest to '-'); fall back to "lora_adapter"."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(raw).strip())
    return value or "lora_adapter"


def _is_qwen35_model_name(model_name: str) -> bool:
    """True for Qwen3.5 base models, which need the conditional-model merge path."""
    value = str(model_name or "").lower()
    return ("qwen3.5" in value) or ("qwen3_5" in value) or ("qwen-3.5" in value)


def _remap_qwen35_adapter_keys_for_conditional(
    adapter_dir: Path,
    remap_tmp_root: Path,
) -> Path:
    """
    Qwen3.5 text-only LoRA adapters are keyed under:
      base_model.model.model.layers.*
    but conditional-model loading expects:
      base_model.model.model.language_model.layers.*
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    staged_adapter_dir = remap_tmp_root / "adapter_for_conditional_merge"
    shutil.copytree(adapter_dir, staged_adapter_dir)

    key_prefix = "base_model.model.model.layers."
    replacement_prefix = "base_model.model.model.language_model.layers."

    safetensors_path = staged_adapter_dir / "adapter_model.safetensors"
    pytorch_bin_path = staged_adapter_dir / "adapter_model.bin"

    def _remap_state_dict_keys(state_dict: dict[str, object]) -> tuple[dict[str, object], int]:
        remapped: dict[str, object] = {}
        remap_count = 0
        for key, value in state_dict.items():
            if key.startswith(key_prefix):
                key = f"{replacement_prefix}{key[len(key_prefix) :]}"
                remap_count += 1
            remapped[key] = value
        return remapped, remap_count

    if safetensors_path.exists():
        metadata = None
        with safe_open(str(safetensors_path), framework="pt") as f:
            metadata = f.metadata()
        state_dict = load_file(str(safetensors_path))
        remapped_state_dict, remap_count = _remap_state_dict_keys(state_dict)
        if remap_count > 0:
            save_file(remapped_state_dict, str(safetensors_path), metadata=metadata)
            print(f"Qwen3.5 merge prep: remapped {remap_count} adapter tensor key(s) for conditional model loading.")
        return staged_adapter_dir

    if pytorch_bin_path.exists():
        state_dict = torch.load(str(pytorch_bin_path), map_location="cpu")
        remapped_state_dict, remap_count = _remap_state_dict_keys(state_dict)
        if remap_count > 0:
            torch.save(remapped_state_dict, str(pytorch_bin_path))
            print(f"Qwen3.5 merge prep: remapped {remap_count} adapter tensor key(s) for conditional model loading.")
        return staged_adapter_dir

    raise RuntimeError(
        "Adapter directory missing LoRA weights (expected adapter_model.safetensors or adapter_model.bin): "
        f"{staged_adapter_dir}"
    )


def merge_adapter_for_vllm(adapter_dir: str, output_dir: Path, dtype: str, trust_remote_code: bool = True) -> str:
    """Merge a LoRA adapter into its base model and write a full model dir for vLLM.

    Loads the adapter's base model at ``dtype``, applies the adapter
    (``merge_and_unload``), and saves the merged model + tokenizer to
    ``output_dir`` (returned). Qwen3.5 adapters take a conditional-model path with
    key remapping and a ``model_type`` fixup for vLLM compatibility. Requires the
    ``serving`` extra (torch/peft/transformers).
    """
    adapter_path = Path(adapter_dir)
    if not _looks_like_adapter_dir(adapter_path):
        raise RuntimeError(f"Adapter directory is invalid: {adapter_dir}")

    base_model = _read_base_model_from_adapter(adapter_path)

    dtype_map = {
        "float16": "float16",
        "bfloat16": "bfloat16",
        "float32": "float32",
    }
    if dtype not in dtype_map:
        raise RuntimeError(f"Unsupported vLLM dtype for merge: {dtype}")

    print(f"Adapter detected. Merging for vLLM launch: {adapter_dir}")
    print(f"Base model for merge: {base_model}")
    print(f"Merged model output: {output_dir}")

    # Keep imports local so non-merge paths do not require these packages.
    import tempfile
    import torch
    from peft import PeftModel
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
    )

    dtype = getattr(torch, dtype_map[dtype])
    base_config = AutoConfig.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    qwen35_conditional_merge = _is_qwen35_model_name(base_model)
    if qwen35_conditional_merge:
        print("Qwen3.5 adapter detected. Using conditional-model merge path for vLLM compatibility.")
        with tempfile.TemporaryDirectory(prefix="qwen35_adapter_remap_") as remap_tmp:
            remapped_adapter = _remap_qwen35_adapter_keys_for_conditional(
                adapter_path,
                Path(remap_tmp),
            )
            model = AutoModelForImageTextToText.from_pretrained(
                base_model,
                dtype=dtype,
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code,
            )
            model = PeftModel.from_pretrained(model, str(remapped_adapter))
            merged = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        )
        model = PeftModel.from_pretrained(model, str(adapter_path))
        merged = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)

    merged.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    tokenizer.save_pretrained(str(output_dir))
    if qwen35_conditional_merge:
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=trust_remote_code)
        processor.save_pretrained(str(output_dir))
        processor_cfg = output_dir / "processor_config.json"
        preprocessor_cfg = output_dir / "preprocessor_config.json"
        if processor_cfg.exists() and not preprocessor_cfg.exists():
            shutil.copy2(processor_cfg, preprocessor_cfg)

    # Normalize merged config metadata minimally for vLLM compatibility.
    # For Qwen3.5, merged save can emit model_type=qwen3_5_text, which some vLLM
    # runtime stacks fail to resolve. We only rewrite model_type and keep the rest
    # of the merged text-model config intact (do not inject multimodal fields).
    cfg_path = output_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            merged_cfg = json.load(f)
        base_cfg_dict = base_config.to_dict()
        if "model_type" in base_cfg_dict:
            merged_cfg["model_type"] = base_cfg_dict["model_type"]
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(merged_cfg, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")

    return str(output_dir)


def resolve_vllm_launch_model(config: EvaluationConfig, output_dir: Path):
    """Decide what vLLM should actually serve for ``config.model_name``.

    Returns ``(launch_model, merge_seconds, merged_for_this_run)``. Resolution:
    an explicit ``vllm_launch_model`` or a full model dir is served as-is; a large
    adapter (base size >= ``vllm_lora_min_model_size_b``) is served as a vLLM LoRA
    module against its base (no merge); a smaller adapter is merged into a temp
    full-model dir under ``output_dir``; a bare HF repo id is served directly.

    Side effect: mutates ``config`` (``vllm_lora_module_name``/``_path``,
    ``vllm_served_model_name``, ``vllm_model_name``, ``vllm_tokenizer_name``) to
    match the chosen strategy.
    """
    config.vllm_lora_module_name = None
    config.vllm_lora_module_path = None

    merge_seconds = 0.0
    merged_for_this_run = False

    if config.vllm_launch_model:
        launch_model = config.vllm_launch_model
        if not _looks_like_model_dir(Path(launch_model)):
            raise RuntimeError(f"vllm_launch_model is not a full model directory: {launch_model}")
        return launch_model, merge_seconds, merged_for_this_run

    local_path = Path(config.model_name)
    if _looks_like_model_dir(local_path):
        return config.model_name, merge_seconds, merged_for_this_run

    if _looks_like_adapter_dir(local_path):
        base_model = _read_base_model_from_adapter(local_path)
        if not config.vllm_tokenizer_name:
            # For merged adapter launches, prefer the base tokenizer explicitly.
            # Some merged local tokenizer artifacts can carry incompatible class
            # metadata (e.g. TokenizersBackend) for the vLLM runtime stack.
            config.vllm_tokenizer_name = base_model
        parsed_size_b = _parse_model_size_b(base_model)
        if config.vllm_use_lora_for_large_adapters and (
            parsed_size_b is not None and parsed_size_b >= float(config.vllm_lora_min_model_size_b)
        ):
            lora_name = _sanitize_lora_module_name(local_path.name)
            if config.vllm_served_model_name is None:
                config.vllm_served_model_name = config.vllm_model_name
            config.vllm_model_name = lora_name
            config.vllm_lora_module_name = lora_name
            config.vllm_lora_module_path = str(local_path)
            print(f"Adapter detected. Serving with vLLM LoRA module (no merge): {local_path}")
            print(f"Base model for vLLM launch: {base_model}")
            print(f"LoRA module name: {lora_name}")
            return base_model, merge_seconds, merged_for_this_run

        merge_parent = Path(config.vllm_merge_parent_dir or "")
        if not merge_parent.is_absolute():
            merge_parent = abs_path(str(merge_parent)) if str(merge_parent) else output_dir
        merge_parent.mkdir(parents=True, exist_ok=True)
        merge_out_dir = merge_parent / f"temp_merged_model_for_vllm_{datetime.now().strftime('%y%m%d-%H%M%S')}"
        merge_start = time.perf_counter()
        launch_model = merge_adapter_for_vllm(
            config.model_name,
            merge_out_dir,
            dtype=config.vllm_dtype,
            trust_remote_code=config.vllm_trust_remote_code,
        )
        merge_seconds = time.perf_counter() - merge_start
        merged_for_this_run = True
        return launch_model, merge_seconds, merged_for_this_run

    if _looks_like_hf_repo_id(config.model_name):
        # Remote HF model id; vLLM can fetch and serve it directly.
        return config.model_name, merge_seconds, merged_for_this_run

    raise RuntimeError("Resolved model is neither a full model directory nor an adapter directory.")


def resolve_effective_vllm_max_num_seqs(
    config: EvaluationConfig,
) -> int:
    """vLLM ``--max-num-seqs`` to launch with: the explicit ``vllm_max_num_seqs``
    if set, else a safe value capped at min(``vllm_max_workers``, 64) — vLLM's own
    default can be too aggressive at long context lengths and over-allocate host
    memory."""
    configured_max_num_seqs = config.vllm_max_num_seqs
    if configured_max_num_seqs is None:
        # vLLM defaults can be too aggressive at long context lengths and cause
        # large host allocations. Keep launch concurrency aligned with eval workers.
        worker_count = int(config.vllm_max_workers or 1)
        return max(1, min(worker_count, 64))
    return max(1, int(configured_max_num_seqs))


def launch_vllm_server(
    config: EvaluationConfig,
    launch_model: str,
    log_path: Path,
):
    """Start a local vLLM OpenAI-compatible server for ``launch_model`` as a
    subprocess (its own process group), streaming output to ``log_path``.

    Builds the serve command from ``config`` (dtype, GPU memory fraction, max model
    len, max-num-seqs, tokenizer, and ``--enable-lora`` flags when serving a LoRA
    module), preferring the ``vllm`` binary over ``python -m vllm`` per
    ``vllm_prefer_binary``. Returns ``(proc, cmd, log_file, effective_max_num_seqs)``;
    the caller is responsible for shutdown. Does not wait for readiness — see
    :func:`preflight_vllm`.
    """
    port = resolve_vllm_launch_port(config)
    effective_max_num_seqs = resolve_effective_vllm_max_num_seqs(config)
    served_model_name = config.vllm_served_model_name or config.vllm_model_name

    common_flags = [
        "--served-model-name",
        served_model_name,
        "--port",
        str(port),
        "--dtype",
        config.vllm_dtype,
        "--gpu-memory-utilization",
        str(config.vllm_gpu_memory_utilization),
        "--max-model-len",
        str(config.vllm_max_model_len),
        "--max-num-seqs",
        str(effective_max_num_seqs),
    ]
    if config.vllm_trust_remote_code:
        common_flags.append("--trust-remote-code")
    if config.vllm_quantization:
        common_flags.extend(["--quantization", config.vllm_quantization])
    tokenizer_name = (config.vllm_tokenizer_name or "").strip()
    if tokenizer_name:
        common_flags.extend(["--tokenizer", tokenizer_name])
    if config.vllm_lora_module_path:
        common_flags.extend(
            [
                "--enable-lora",
                "--lora-modules",
                f"{config.vllm_lora_module_name}={config.vllm_lora_module_path}",
            ]
        )

    prefer_binary = bool(config.vllm_prefer_binary)
    explicit_vllm_bin = (config.vllm_binary_path or "").strip()
    vllm_bin = None
    if explicit_vllm_bin:
        candidate = os.path.expanduser(explicit_vllm_bin)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            vllm_bin = candidate
    if vllm_bin is None:
        vllm_bin = shutil.which("vllm")
    module_available = False
    try:
        module_available = importlib.util.find_spec("vllm.entrypoints.openai.api_server") is not None
    except ModuleNotFoundError:
        module_available = False

    if prefer_binary and vllm_bin:
        cmd = [vllm_bin, "serve", launch_model, *common_flags]
    elif module_available:
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", launch_model, *common_flags]
    elif vllm_bin:
        cmd = [vllm_bin, "serve", launch_model, *common_flags]
    else:
        raise RuntimeError("vLLM is not available in current python and no 'vllm' binary found on PATH.")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return proc, cmd, log_file, effective_max_num_seqs


def shutdown_vllm_server(proc: subprocess.Popen | None, timeout_s: float = 20.0):
    """Stop a vLLM server started by :func:`launch_vllm_server`.

    Sends SIGTERM to the process group, waits up to ``timeout_s`` for a clean
    exit, then escalates to SIGKILL. No-op if ``proc`` is None or already exited.
    """
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def preflight_vllm(
    config: EvaluationConfig, vllm_proc: subprocess.Popen | None = None, vllm_log_path: Path | None = None
):
    """
    Verify vLLM endpoint/model availability before launching full evaluation.
    Also detects early server death if the script launched vLLM itself.
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=config.vllm_base_url,
        api_key=config.vllm_api_key,
        timeout=config.vllm_timeout_seconds,
        max_retries=0,
    )
    deadline = time.time() + float(config.vllm_start_timeout_seconds)
    last_error = None

    while time.time() < deadline:
        if vllm_proc is not None and vllm_proc.poll() is not None:
            hint = f" See log: {vllm_log_path}" if vllm_log_path else ""
            raise RuntimeError(f"vLLM process exited early with code {vllm_proc.returncode}.{hint}")
        try:
            client.completions.create(
                model=config.vllm_model_name,
                prompt="ping",
                max_tokens=1,
                temperature=0.0,
                top_p=1.0,
                n=1,
            )
            return
        except Exception as e:
            last_error = e
            time.sleep(1.0)

    raise RuntimeError(
        "vLLM preflight failed. Ensure the local vLLM server is running and serving "
        f"model '{config.vllm_model_name}' at '{config.vllm_base_url}'."
    ) from last_error
