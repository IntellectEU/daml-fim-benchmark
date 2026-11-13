"""The evaluation engine: runs the FIM benchmark against a local model and scores it.

This is the heart of the benchmark. Given a model (a fine-tuned LoRA adapter, a
merged checkpoint, or a Hugging Face repo id), it:

1. Takes the model to evaluate from ``model_name`` — a Hugging Face repo id, a
   merged-checkpoint directory, or a LoRA adapter directory — used as given.
2. Serves that model locally with vLLM. The engine launches its own
   OpenAI-compatible vLLM server (merging a LoRA adapter into its base model
   first, or serving the adapter as a vLLM LoRA module for large bases), waits
   for it to be ready, then tears it down when the run finishes.
3. Loads benchmark examples and, for each one, builds the FIM prompt
   (see :mod:`daml_fim_benchmark.prompting`), sends it to vLLM, and records
   whether the completion exactly matches the held-out target.
4. Runs Daml syntax checks on every completion
   (see :mod:`daml_fim_benchmark.syntax_metrics`) and aggregates exact-match,
   prefix-match, per-repo, in/out-of-domain and syntax-pass metrics into a
   ``summary.json`` plus a per-example ``results.jsonl`` under ``logs_dir``. These
   are exactly the files the results server and web UI read back.

The benchmark evaluates local ``provider="huggingface"`` models served via vLLM;
any other provider value is rejected at run time.

Entry points:
- :func:`run_evaluation` — evaluate one resolved config end-to-end and return its
  summary dict. Call this directly to drive the benchmark from your own code.
- :func:`main` — the CLI (``python -m daml_fim_benchmark.evaluate``). Build the
  experiment(s) from per-field flags (one per :class:`EvaluationConfig` field) or
  load them from a ``--config`` Python or JSON file, expand any cross-validation folds,
  run each, and write an aggregated CV parent summary.
"""

import argparse
import importlib.util
import json
import re
import os
import types
import typing
from dataclasses import asdict, fields, is_dataclass, replace
import shutil
import time
import sys
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random
import hashlib
from collections import defaultdict, Counter
from uuid import uuid4

from tqdm import tqdm

from daml_fim_benchmark.cross_validation import aggregate_cv_eval_summaries
from daml_fim_benchmark.syntax_metrics import augment_results_with_syntax_metrics
from daml_fim_benchmark.prompting import construct_prompt, resolved_prompt_type
from daml_fim_benchmark.config import (
    CvFold,
    EvaluationConfig,
    abs_path,
    resolve_dataset_version,
    get_base_model_for_tokenizer,
    infer_default_use_chat_template_for_fim,
)

from daml_fim_benchmark.serving import (
    launch_vllm_server,
    preflight_vllm,
    resolve_vllm_launch_model,
    resolve_vllm_launch_port,
    shutdown_vllm_server,
)


def infer_repo_name(file_path: str) -> str:
    """
    Infer repository name from a row's file_path.

    Paths are repo-relative with the repo as the first segment, e.g.
    ``daml-finance/src/main/daml/.../Factory.daml`` -> ``daml-finance``.
    A bare filename (no path beneath a repo) yields ``"unknown"``.
    """
    if not file_path:
        return "unknown"
    normalized = file_path.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]
    if len(parts) >= 2:
        return parts[0]
    return "unknown"


# Keep retry chatter from the HTTP client from breaking tqdm progress bars.
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ==========================================
# CONFIGURATION
# ==========================================

# Token count tiers for dynamic max_output_tokens
# Each tier: (threshold, max_tokens) - if target_tokens <= threshold, use max_tokens
TOKEN_TIERS = [
    (64, 128),  # Short targets: 128 tokens (2x headroom)
    (128, 256),  # Medium targets: 256 tokens
    (256, 512),  # Longer targets: 512 tokens
    (512, 1024),  # Long targets: 1024 tokens
    (float("inf"), 3000),  # Very long targets (script_body): 3000 tokens
]

FIM_STOP_STRINGS = ["<|endoftext|>", "<|end_of_text|>", "<|fim_prefix|>", "<|fim_suffix|>"]
CV_EVAL_NAME_FOLD_RE = re.compile(
    r"^\s*(.*?)\s*\[\s*(?:cv|fold)\s+\d+\s*[/_]\s*\d+\s*\]\s*$",
    flags=re.IGNORECASE,
)


# ==========================================
# MODEL RESOLUTION
# ==========================================


def resolve_eval_mode(config: EvaluationConfig) -> str:
    """
    Resolve evaluation mode with backward compatibility.

    Historical values `id` / `ood` are deprecated. Evaluation now always runs
    in unified `all` mode and derives ID/OOD breakdown from fold train repos.
    """
    legacy_eval_on_ood = config.eval_on_ood
    if isinstance(legacy_eval_on_ood, bool):
        requested = "ood" if legacy_eval_on_ood else "id"
        print(
            f"Deprecated eval mode requested via eval_on_ood={legacy_eval_on_ood} "
            f"({requested}); forcing eval_mode='all'."
        )
        return "all"

    mode = (config.eval_mode or "all").strip().lower()
    if mode not in {"all", "id", "ood"}:
        raise ValueError(f"Invalid eval_mode: {mode}. Expected one of: all, id, ood")
    if mode != "all":
        print(f"Deprecated eval_mode='{mode}' requested; forcing eval_mode='all'.")
    return "all"


def add_repo_fields(rows: list[dict]) -> None:
    for row in rows:
        row.setdefault("repo", infer_repo_name(str(row.get("file_path", ""))))


# ==========================================
# EVALUATION HELPERS
# ==========================================


def get_dynamic_max_tokens(target: str, tokenizer_encode=None) -> int:
    """
    Calculate appropriate max_output_tokens based on target length.
    Uses simple char-based estimation if no tokenizer provided.
    Returns a value that gives the model enough room to generate the full target.
    """
    if tokenizer_encode:
        target_tokens = len(tokenizer_encode(target))
    else:
        # Rough estimate: ~4 chars per token for code
        target_tokens = len(target) // 4

    for threshold, max_tokens in TOKEN_TIERS:
        if target_tokens <= threshold:
            return max_tokens

    return TOKEN_TIERS[-1][1]  # Fallback to largest tier


def is_vllm_local_engine(config: EvaluationConfig) -> bool:
    return config.provider == "huggingface"


def default_parallel(config: EvaluationConfig) -> bool:
    """Returns model/backend-specific default parallelism."""
    if is_vllm_local_engine(config):
        # vLLM is designed for concurrent request scheduling.
        return True
    return config.provider != "huggingface"


def resolve_syntax_check_workers(config: EvaluationConfig, total_examples=None) -> int:
    """
    Resolve syntax-check worker count.
    - Explicit config value wins.
    - Otherwise pick a conservative CPU-based default.
    """
    configured = config.syntax_check_workers
    if configured is None:
        workers = min(8, os.cpu_count() or 1)
    else:
        workers = int(configured)

    workers = max(1, workers)
    if total_examples is not None:
        workers = min(workers, max(1, int(total_examples)))
    return workers


def format_local_fim_prompt(prompt: str, tokenizer, config: EvaluationConfig) -> str:
    """
    Format a prompt for local vLLM inference.
    - chat-template mode: wraps as single user turn
    - raw mode: sends the FIM prompt text verbatim
    """
    if bool(config.use_chat_template_for_fim):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def truncate_suffix_overlap(predicted: str, suffix: str, min_overlap: int = 30) -> str:
    """
    Truncates the predicted string if it starts bleeding into the suffix.
    Uses a Longest Common Prefix (LCP) sliding window approach.
    """
    if not predicted or not suffix or len(suffix) < min_overlap:
        return predicted

    # We only care about the beginning of the suffix for the overlap check
    suffix_start = suffix[: min_overlap * 2]

    # Slide a window across the predicted text to find where the suffix begins
    for i in range(len(predicted)):
        # Calculate how many characters remain in the prediction from index i
        remaining_pred_len = len(predicted) - i

        # The overlap we are checking is bounded by the minimum of:
        # 1. The remaining characters in the prediction
        # 2. The length of our suffix sample
        check_len = min(remaining_pred_len, len(suffix_start))

        if check_len < min_overlap:
            # Not enough characters left in predicted to satisfy min_overlap
            break

        # If the substring matches the beginning of the suffix exactly
        if predicted[i : i + check_len] == suffix_start[:check_len]:
            return predicted[:i]

    return predicted


def normalize(code):
    """Simple normalization for exact match check."""
    # Strip FIM end tokens that may appear in output
    code = code.replace("<|endoftext|>", "").replace("<|end_of_text|>", "")
    return code.strip()


def _is_timeout_exception(exc: Exception) -> bool:
    """Classify request/network timeout exceptions from the HTTP client."""
    if exc is None:
        return False
    type_name = type(exc).__name__.lower()
    if "timeout" in type_name:
        return True
    return "timed out" in str(exc).lower()


def _vllm_request_timeout_seconds(config: EvaluationConfig, max_tokens: int) -> float:
    """Compute per-request timeout budget with optional token-scaled slack."""
    base = float(config.vllm_timeout_seconds or 120.0)
    per_token = max(0.0, float(config.vllm_timeout_per_output_token_seconds or 0.0))
    timeout_max = max(base, float(config.vllm_timeout_max_seconds or base))
    scaled = base + (per_token * max(0, int(max_tokens)))
    return min(timeout_max, scaled)


def _error_result_for_row(row: dict, config: EvaluationConfig, exc: Exception) -> dict:
    prompt = ""
    try:
        prompt = construct_prompt(row, config)
    except Exception:
        prompt = ""
    return {
        **row,
        "predicted": "",
        "raw_response": "",
        "exact_match": False,
        "prefix_match": False,
        "prompt_type": resolved_prompt_type(config),
        "prompt": prompt,
        "input_tokens": 0,
        "output_tokens": 0,
        "provider_model": None,
        "provider_model_name": None,
        "provider_model_engine": None,
        "provider_experiments": None,
        "error": str(exc),
        "error_type": "TimeoutError" if _is_timeout_exception(exc) else type(exc).__name__,
    }


def evaluate_single_row_vllm_prepared(
    row,
    prompt: str,
    formatted_prompt: str,
    max_tokens: int,
    config: EvaluationConfig,
    vllm_client,
    tokenizer,
):
    """Evaluate one row via vLLM OpenAI-compatible completions using a preformatted prompt."""
    timeout_retries = max(0, int(config.vllm_timeout_retries or 0))
    retry_backoff = max(0.0, float(config.vllm_timeout_retry_backoff_seconds or 0.0))
    request_timeout = _vllm_request_timeout_seconds(config, max_tokens)

    response = None
    for attempt in range(timeout_retries + 1):
        try:
            req_client = vllm_client.with_options(timeout=request_timeout)
            response = req_client.completions.create(
                model=config.vllm_model_name,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                temperature=config.vllm_temperature,
                top_p=config.vllm_top_p,
                stop=FIM_STOP_STRINGS,
                n=1,
                presence_penalty=0.0,
                frequency_penalty=0.0,
            )
            break
        except Exception as exc:
            is_timeout = _is_timeout_exception(exc)
            if (not is_timeout) or (attempt >= timeout_retries):
                raise
            # Linear backoff for transient queue spikes/long-tail generations.
            delay = retry_backoff * float(attempt + 1)
            if delay > 0:
                time.sleep(delay)

    if response is None:
        raise RuntimeError("vLLM completion returned no response and no exception.")

    raw_text = response.choices[0].text or ""

    # Extract prediction
    match = re.search(r"<FILL>(.*?)</FILL>", raw_text, re.DOTALL)
    if match:
        predicted = match.group(1)
    else:
        predicted = raw_text.replace("<FILL>", "").replace("</FILL>", "")

    if config.post_hoc_truncation:
        predicted = truncate_suffix_overlap(predicted, row["suffix"], min_overlap=config.post_hoc_truncation_threshold)

    norm_predicted = normalize(predicted)
    norm_target = normalize(row["target"])

    is_match = norm_predicted == norm_target
    is_prefix_match = norm_predicted.startswith(norm_target) and not is_match

    prompt_type = resolved_prompt_type(config)

    usage = getattr(response, "usage", None)
    if usage is None:
        input_tokens = len(tokenizer.encode(formatted_prompt, add_special_tokens=False))
        output_tokens = len(tokenizer.encode(raw_text, add_special_tokens=False))
    else:
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

    return {
        **row,
        "predicted": predicted,
        "raw_response": raw_text,
        "exact_match": is_match,
        "prefix_match": is_prefix_match,
        "prompt_type": prompt_type,
        "prompt": prompt,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "provider_model": None,
        "provider_model_name": None,
        "provider_model_engine": None,
        "provider_experiments": None,
    }


def run_evaluation(config: EvaluationConfig):
    """Run FIM evaluation with the given config."""
    total_start_time = time.perf_counter()
    eval_mode = resolve_eval_mode(config)

    # Work on a copy so resolution/launch tweaks don't mutate the caller's config.
    config = replace(config)
    if not config.model_name:
        raise ValueError(
            "model_name is required: set it to a Hugging Face repo id, a model "
            "directory, or a LoRA adapter directory."
        )

    if config.use_chat_template_for_fim is None:
        model_hint = get_base_model_for_tokenizer(config.model_name)
        config.use_chat_template_for_fim = infer_default_use_chat_template_for_fim(model_hint)

    print(f"Using model: {config.model_name}")
    print(f"FIM chat-template mode: {bool(config.use_chat_template_for_fim)}")

    # In-domain repos (the repos the model trained on) drive the ID/OOD split.
    # Declared explicitly via config; everything else in the eval set is OOD.
    in_domain_repos = (
        {r.strip() for r in config.in_domain_repos if r.strip()}
        if config.in_domain_repos
        else None
    )
    if in_domain_repos is not None:
        print(f"In-domain repos declared: {len(in_domain_repos)} repos")

    # Determine parallelism
    parallel = is_parallel(config)

    # Load rows
    rows = get_rows(config, mode=eval_mode)

    eval_target_map = {"all": "combined repo set"}
    eval_target = eval_target_map.get(eval_mode, eval_mode)
    print(f"\nRunning evaluation for {eval_target} (parallel={parallel})...")
    print(f"Sampling policy: samples_per_repo={config.samples_per_repo}, syntax_ratio={config.syntax_ratio}")
    n_syntax = len([r for r in rows if r.get("dataset_source") == "syntax"])
    n_random = len([r for r in rows if r.get("dataset_source") == "random"])
    n_repos = len({r.get("repo", "unknown") for r in rows})
    print(f"Total samples: {len(rows)} across {n_repos} repos (Syntax: {n_syntax}, Random: {n_random})")

    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")

    cv_parent_run_id = config.cv_parent_run_id or config.cv_name or ""
    cv_subrun_index = config.cv_subrun_index
    cv_subrun_count = config.cv_subrun_count

    eval_name = config.eval_name or ""
    run_name = eval_name
    run_id = str(uuid4())
    out_dir = abs_path(config.logs_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save notes if provided
    if config.notes:
        with open(out_dir / "notes.txt", "w") as f:
            f.write(config.notes.strip())

    # Write initial summary with config (so the results UI can show it during run)
    log_config = get_log_ready_config(config)
    initial_summary = {
        "run_id": run_id,
        "run_name": run_name,
        "model": Path(config.model_name).name,
        "model_path": config.model_name,
        "timestamp": timestamp,
        "dataset": eval_mode,
        "cv_parent_run_id": cv_parent_run_id,
        "cv_subrun_index": cv_subrun_index,
        "cv_subrun_count": cv_subrun_count,
        "config": log_config,
        "status": "in_progress",
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(initial_summary, f, indent=2)

    results_file = out_dir / "results.jsonl"
    results = []
    results_lock = threading.Lock()

    vllm_proc = None
    vllm_log_handle = None
    vllm_cmd = []
    vllm_log_path = out_dir / "vllm_server.log"
    vllm_launch_model = None
    vllm_merged_model_path = None
    vllm_merged_for_this_run = False
    vllm_merge_seconds = 0.0
    vllm_startup_seconds = 0.0
    vllm_effective_max_num_seqs = None

    generation_start_time = time.perf_counter()

    processed_count = 0
    match_count = 0

    def update_progress_score(pbar, new_results):
        """Update progress bar with running exact-match score."""
        nonlocal processed_count, match_count
        processed_count += len(new_results)
        match_count += sum(1 for r in new_results if r.get("exact_match"))
        running_score = (match_count / processed_count) if processed_count else 0
        pbar.set_postfix(score=f"{running_score:.2%}")

    try:
        if not is_vllm_local_engine(config):
            raise RuntimeError(
                f"Unsupported provider '{config.provider}'. This benchmark evaluates local "
                "vLLM-served models only (provider='huggingface')."
            )

        print(
            f"Using vLLM-backed local inference (base_url={config.vllm_base_url}, "
            f"served_model={config.vllm_model_name})..."
        )
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing dependency 'openai' required for the vLLM OpenAI-compatible client. "
                "Install the serving extra ('pip install daml-fim-benchmark[serving]')."
            ) from exc
        from transformers import AutoTokenizer

        if config.vllm_auto_launch:
            launch_port = resolve_vllm_launch_port(config)
            config.vllm_base_url = f"http://127.0.0.1:{launch_port}/v1"
            vllm_launch_model, vllm_merge_seconds, vllm_merged_for_this_run = resolve_vllm_launch_model(
                config, out_dir
            )
            if vllm_merged_for_this_run:
                vllm_merged_model_path = vllm_launch_model
            print(f"Launching local vLLM from model: {vllm_launch_model}")
            if vllm_merged_for_this_run:
                print(f"Adapter merge completed in {vllm_merge_seconds:.1f}s")
            print(f"vLLM logs: {vllm_log_path}")
            vllm_start = time.perf_counter()
            vllm_proc, vllm_cmd, vllm_log_handle, vllm_effective_max_num_seqs = launch_vllm_server(
                config,
                vllm_launch_model,
                vllm_log_path,
            )
            preflight_vllm(config, vllm_proc=vllm_proc, vllm_log_path=vllm_log_path)
            vllm_startup_seconds = time.perf_counter() - vllm_start
            print(f"vLLM preflight: ok (max_num_seqs={vllm_effective_max_num_seqs})")
        else:
            preflight_vllm(config)
            print("vLLM preflight: ok (using existing server)")

        base_model_name = get_base_model_for_tokenizer(config.model_name)
        model_path = Path(config.model_name)
        tokenizer_load_path = (
            config.model_name if (model_path / "tokenizer_config.json").exists() else base_model_name
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path)

        prepared_rows = []
        for row in tqdm(rows, desc="Preparing prompts for vLLM"):
            prompt = construct_prompt(row, config)
            formatted_prompt = format_local_fim_prompt(prompt, tokenizer, config)
            max_tokens = get_dynamic_max_tokens(row["target"], tokenizer_encode=tokenizer.encode)
            prepared_rows.append((row, prompt, formatted_prompt, max_tokens))

        thread_local = threading.local()

        def get_thread_client():
            if not hasattr(thread_local, "client"):
                thread_local.client = OpenAI(
                    base_url=config.vllm_base_url,
                    api_key=config.vllm_api_key,
                    timeout=config.vllm_timeout_seconds,
                    max_retries=0,
                )
            return thread_local.client

        def process_prepared_and_log(prepared):
            row, prompt, formatted_prompt, max_tokens = prepared
            client = get_thread_client()
            try:
                result = evaluate_single_row_vllm_prepared(
                    row=row,
                    prompt=prompt,
                    formatted_prompt=formatted_prompt,
                    max_tokens=max_tokens,
                    config=config,
                    vllm_client=client,
                    tokenizer=tokenizer,
                )
            except Exception as e:
                is_timeout = _is_timeout_exception(e)
                if not (is_timeout or bool(config.continue_on_sample_error)):
                    raise
                sample_id = str(row.get("id", "") or "<no-id>")
                sample_path = str(row.get("file_path", "") or "<no-file-path>")
                kind = "timeout" if is_timeout else "error"
                print(
                    f"Warning: vLLM sample {kind}; marking row as failed and continuing "
                    f"(id={sample_id}, file={sample_path}): {type(e).__name__}: {e}"
                )
                result = _error_result_for_row(row, config, e)

            with results_lock:
                results.append(result)
                with open(results_file, "a") as f:
                    f.write(json.dumps(result) + "\n")

            return result

        if parallel:
            max_workers = max(1, min(config.vllm_max_workers, len(prepared_rows)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_prepared_and_log, prepared): i
                    for i, prepared in enumerate(prepared_rows)
                }
                with tqdm(total=len(prepared_rows), desc=f"Evaluating via vLLM (x{max_workers})") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        update_progress_score(pbar, [result])
                        pbar.update(1)
        else:
            with tqdm(total=len(prepared_rows), desc="Evaluating via vLLM") as pbar:
                for prepared in prepared_rows:
                    result = process_prepared_and_log(prepared)
                    update_progress_score(pbar, [result])
                    pbar.update(1)
    finally:
        shutdown_vllm_server(vllm_proc)
        if vllm_log_handle is not None:
            vllm_log_handle.close()
        if vllm_merged_for_this_run and bool(config.vllm_cleanup_merged_model) and vllm_merged_model_path:
            try:
                shutil.rmtree(vllm_merged_model_path, ignore_errors=True)
                print(f"Cleaned merged model directory: {vllm_merged_model_path}")
            except Exception as e:
                print(f"Warning: failed to clean merged model dir {vllm_merged_model_path}: {e}")

    generation_duration_seconds = time.perf_counter() - generation_start_time

    syntax_total = len(results)
    syntax_workers = resolve_syntax_check_workers(config, total_examples=syntax_total)
    print(f"\nRunning DAML syntax checks on completed files (workers={syntax_workers})...")

    syntax_completed = 0
    syntax_checkable_completed = 0
    syntax_pass_completed = 0
    syntax_progress_lock = threading.Lock()
    syntax_heartbeat_stop = threading.Event()
    syntax_start_time = time.perf_counter()

    def _syntax_heartbeat() -> None:
        while not syntax_heartbeat_stop.wait(10.0):
            with syntax_progress_lock:
                completed = syntax_completed
            if completed >= syntax_total:
                break
            elapsed = time.perf_counter() - syntax_start_time
            print(f"   Syntax checks in progress: {completed}/{syntax_total} completed ({elapsed:.0f}s elapsed)...")

    with tqdm(
        total=syntax_total,
        desc=f"Syntax checks (x{syntax_workers})",
        file=sys.stdout,
        disable=False,
    ) as syntax_pbar:
        syntax_pbar.set_postfix(score=f"{0.0:.2%}", completed=f"0/{syntax_total}")
        syntax_pbar.refresh()

        def _syntax_progress(n=1, checkable_delta=0, pass_delta=0):
            nonlocal syntax_completed, syntax_checkable_completed, syntax_pass_completed
            with syntax_progress_lock:
                syntax_completed += int(n)
                syntax_checkable_completed += int(checkable_delta)
                syntax_pass_completed += int(pass_delta)
                completed = syntax_completed
                running_score = (
                    syntax_pass_completed / syntax_checkable_completed if syntax_checkable_completed else 0.0
                )
            syntax_pbar.update(int(n))
            syntax_pbar.set_postfix(
                score=f"{running_score:.2%}",
                completed=f"{completed}/{syntax_total}",
            )

        heartbeat_thread = None
        if syntax_total > 0:
            heartbeat_thread = threading.Thread(target=_syntax_heartbeat, daemon=True)
            heartbeat_thread.start()

        try:
            syntax_summary = augment_results_with_syntax_metrics(
                results,
                progress_cb=_syntax_progress,
                workers=syntax_workers,
            )
        finally:
            syntax_heartbeat_stop.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=1.0)
    with open(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    error_results = [r for r in results if r.get("error")]
    timeout_error_results = [r for r in error_results if is_timeout_result(r)]
    non_error_results = [
        r
        for r in results
        if isinstance(r, dict)
        and not is_timeout_result(r)
        and not str(r.get("error", "") or "").strip()
        and not str(r.get("error_type", "") or "").strip()
    ]
    empty_response_count = 0
    for row in non_error_results:
        raw_response = row.get("raw_response")
        predicted = row.get("predicted")
        response_text = (
            raw_response if isinstance(raw_response, str) else predicted if isinstance(predicted, str) else ""
        )
        if response_text.strip() == "":
            empty_response_count += 1
    empty_response_rate = empty_response_count / len(non_error_results) if non_error_results else None
    scored_results = [r for r in results if not is_timeout_result(r)]
    accuracy, prefix_match_rate, cat_stats, source_stats = get_stats(scored_results)
    repo_summary = get_repo_summary(scored_results, in_domain_repos=in_domain_repos)
    syntax_domain_summary = get_syntax_domain_summary(
        scored_results,
        in_domain_repos=in_domain_repos,
        eval_mode=eval_mode,
    )
    error_type_counts = dict(Counter(str(r.get("error_type", "Unknown")) for r in error_results))
    provider_model_name_counts = dict(
        Counter(
            str(r.get("provider_model_name", "")).strip()
            for r in non_error_results
            if str(r.get("provider_model_name", "")).strip()
        )
    )
    provider_model_engine_counts = dict(
        Counter(
            str(r.get("provider_model_engine", "")).strip()
            for r in non_error_results
            if str(r.get("provider_model_engine", "")).strip()
        )
    )

    # Accumulated token usage across all results.
    total_input_tokens = sum(r.get("input_tokens", 0) for r in results)
    total_output_tokens = sum(r.get("output_tokens", 0) for r in results)

    # Get config with resolved values for logging
    log_config = get_log_ready_config(config)

    duration_seconds = time.perf_counter() - total_start_time
    setup_duration_seconds = max(0.0, duration_seconds - generation_duration_seconds)

    # Record the fold's ID/OOD split for CV-parent aggregation/display. OOD is the
    # complement of the declared in-domain repos within this run's eval set.
    eval_repos = {str(r.get("repo") or infer_repo_name(str(r.get("file_path", "")))) for r in scored_results}
    eval_repos.discard("unknown")
    cv_train_repos = sorted(in_domain_repos & eval_repos) if in_domain_repos else []
    cv_holdout_repos = sorted(eval_repos - in_domain_repos) if in_domain_repos else []

    summary = {
        "run_id": run_id,
        "run_name": run_name,
        "model": Path(config.model_name).name,
        "model_path": config.model_name,
        "timestamp": timestamp,
        "config": log_config,
        "dataset": eval_mode,
        "cv_parent_run_id": cv_parent_run_id,
        "cv_subrun_index": cv_subrun_index,
        "cv_subrun_count": cv_subrun_count,
        "cv_train_repos": cv_train_repos,
        "cv_holdout_repos": cv_holdout_repos,
        "dataset_version": log_config["dataset_version"],
        "syntax_ratio": config.syntax_ratio,
        "samples_per_repo": config.samples_per_repo,
        "total_samples": len(results),
        "total_examples": len(scored_results),
        "timeout_excluded_count": len(timeout_error_results),
        "timeout_excluded_rate": (len(timeout_error_results) / len(results)) if results else 0.0,
        "error_count": len(error_results),
        "error_rate": (len(error_results) / len(results)) if results else 0.0,
        "error_type_counts": error_type_counts,
        "provider_model_name_counts": provider_model_name_counts,
        "provider_model_engine_counts": provider_model_engine_counts,
        "empty_response_count": empty_response_count,
        "empty_response_rate": empty_response_rate,
        "exact_match_rate": accuracy,
        "prefix_match_rate": prefix_match_rate,
        "duration_seconds": duration_seconds,  # End-to-end wall time incl. merge + vLLM startup
        "setup_duration_seconds": setup_duration_seconds,
        "generation_duration_seconds": generation_duration_seconds,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "by_category": cat_stats,
        "by_source": source_stats,
        **repo_summary,
        **syntax_domain_summary,
        "vllm_runtime": {
            "auto_launch": bool(config.vllm_auto_launch),
            "base_url": config.vllm_base_url,
            "served_model_name": config.vllm_model_name,
            "launch_model": vllm_launch_model,
            "effective_max_num_seqs": vllm_effective_max_num_seqs,
            "gpu_memory_utilization": config.vllm_gpu_memory_utilization,
            "merged_for_this_run": vllm_merged_for_this_run,
            "merge_seconds": vllm_merge_seconds,
            "startup_seconds": vllm_startup_seconds,
            "log_path": str(vllm_log_path) if is_vllm_local_engine(config) and bool(config.vllm_auto_launch) else None,
            "launch_cmd": vllm_cmd if is_vllm_local_engine(config) and bool(config.vllm_auto_launch) else [],
        },
        **syntax_summary,
    }

    # Write summary (after all results are in)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Evaluation complete! Exact match rate: {accuracy:.2%}")
    print(f"   Prefix match rate: {prefix_match_rate:.2%} (over-generated)")
    if timeout_error_results:
        print(
            "   Timeout exclusions from scoring: "
            f"{len(timeout_error_results)}/{len(results)} "
            f"({(len(timeout_error_results) / len(results)):.2%})"
        )
    if error_results:
        print(
            f"   Sample errors: {len(error_results)}/{len(results)} "
            f"({(len(error_results) / len(results)):.2%}) {error_type_counts}"
        )
    in_domain_avg = summary.get("in_domain_repo_average")
    out_domain_avg = summary.get("out_of_domain_repo_average")
    if in_domain_avg is not None or out_domain_avg is not None:
        parts = []
        if in_domain_avg is not None:
            parts.append(f"in-domain repo avg={in_domain_avg:.2%}")
        if out_domain_avg is not None:
            parts.append(f"out-of-domain repo avg={out_domain_avg:.2%}")
        print(f"   Repo-level: {', '.join(parts)}")
    print(
        "   Predicted syntax pass rate: "
        f"{syntax_summary['predicted_syntax_pass_rate']:.2%} "
        f"({syntax_summary['syntax_checkable_examples']}/{len(results)} checkable)"
    )
    print(
        "   Original-file syntax sanity: "
        f"{syntax_summary['original_syntax_pass_rate']:.2%} per sample, "
        f"{syntax_summary['syntax_original_file_pass_rate']:.2%} per file"
    )
    if not syntax_summary["syntax_sanity_passed"]:
        print(
            "   WARNING: original file syntax sanity failed for "
            f"{syntax_summary['syntax_original_files_failed']} files."
        )
    if syntax_summary.get("syntax_skipped_examples", 0) > 0:
        print(
            "   Skipped syntax checks: "
            f"{syntax_summary['syntax_skipped_examples']} "
            "(excluded from predicted syntax score)"
        )
        for reason, count in sorted(
            syntax_summary.get("syntax_skip_reason_counts", {}).items(),
            key=lambda x: (-x[1], x[0]),
        ):
            explanation = syntax_summary.get("syntax_skip_reason_explanations", {}).get(reason, "")
            print(f"      - {reason}: {count} ({explanation})")
    if is_vllm_local_engine(config):
        print(
            "   Runtime timing: "
            f"total={duration_seconds:.2f}s "
            f"(setup={setup_duration_seconds:.2f}s, generation={generation_duration_seconds:.2f}s, "
            f"merge={vllm_merge_seconds:.2f}s, vllm_startup={vllm_startup_seconds:.2f}s)"
        )
    print(f"Results saved to: {out_dir}")

    return summary


def is_parallel(config: EvaluationConfig):
    parallel = config.parallel
    if parallel is None:
        parallel = default_parallel(config)
    return parallel


def get_stats(results):
    # Calculate Stats
    matches = [r for r in results if r["exact_match"]]
    accuracy = len(matches) / len(results) if results else 0

    # Calculate prefix matches
    # Handle cases where 'prefix_match' might not exist in older result objects (safety)
    prefix_matches = [r for r in results if r.get("prefix_match", False)]
    prefix_match_rate = len(prefix_matches) / len(results) if results else 0

    # Group by category
    cats = {}
    sources = {}

    for r in results:
        # Category stats
        c = r.get("category", "unknown")
        if c not in cats:
            cats[c] = []
        cats[c].append(r["exact_match"])

        # Source stats
        s = r.get("dataset_source", "unknown")
        if s not in sources:
            sources[s] = []
        sources[s].append(r["exact_match"])

    cat_stats = {k: sum(v) / len(v) for k, v in cats.items()}
    source_stats = {k: sum(v) / len(v) for k, v in sources.items()}
    return accuracy, prefix_match_rate, cat_stats, source_stats


def is_timeout_result(result: dict) -> bool:
    """True when a sample failed due to timeout and should be excluded from scoring."""
    if not isinstance(result, dict):
        return False
    if str(result.get("error_type", "")).strip() == "TimeoutError":
        return True
    error_text = str(result.get("error", "") or "").lower()
    return "timed out" in error_text


def get_repo_summary(results: list[dict], in_domain_repos: set[str] | None) -> dict:
    """
    Compute per-repo exact/prefix rates and repo-level in/out-domain averages.
    A repo is in-domain when it is in ``in_domain_repos`` (else out-of-domain);
    when ``in_domain_repos`` is None there is no ID/OOD split.
    """
    if not results:
        return {
            "repo_scores": [],
            "total_repo_average": None,
            "in_domain_repo_average": None,
            "out_of_domain_repo_average": None,
            "total_repo_prefix_average": None,
            "in_domain_repo_prefix_average": None,
            "out_of_domain_repo_prefix_average": None,
            "in_domain_repo_count": 0,
            "out_of_domain_repo_count": 0,
            "train_repo_count": len(in_domain_repos) if in_domain_repos is not None else None,
        }

    repo_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"matches": 0, "prefix_matches": 0, "total": 0})
    for r in results:
        repo = str(r.get("repo") or infer_repo_name(str(r.get("file_path", ""))) or "unknown")
        is_match = bool(r.get("exact_match", False))
        is_prefix = bool(r.get("prefix_match", False))
        repo_counts[repo]["total"] += 1
        if is_match:
            repo_counts[repo]["matches"] += 1
        if is_prefix:
            repo_counts[repo]["prefix_matches"] += 1

    repo_scores = []
    all_rates: list[float] = []
    in_rates: list[float] = []
    out_rates: list[float] = []
    all_prefix_rates: list[float] = []
    in_prefix_rates: list[float] = []
    out_prefix_rates: list[float] = []
    for repo in sorted(repo_counts.keys()):
        total = repo_counts[repo]["total"]
        rate = (repo_counts[repo]["matches"] / total) if total else 0.0
        prefix_rate = (repo_counts[repo]["prefix_matches"] / total) if total else 0.0
        in_domain = None
        domain_label = "unknown"
        if in_domain_repos is not None and repo != "unknown":
            in_domain = repo in in_domain_repos
            domain_label = "in_domain" if in_domain else "out_of_domain"
            if in_domain:
                in_rates.append(rate)
                in_prefix_rates.append(prefix_rate)
            else:
                out_rates.append(rate)
                out_prefix_rates.append(prefix_rate)
        all_rates.append(rate)
        all_prefix_rates.append(prefix_rate)
        repo_scores.append(
            {
                "repo": repo,
                "exact_match_rate": rate,
                "prefix_match_rate": prefix_rate,
                "total_examples": total,
                "in_domain": in_domain,
                "domain": domain_label,
            }
        )

    return {
        "repo_scores": repo_scores,
        "total_repo_average": (sum(all_rates) / len(all_rates)) if all_rates else None,
        "in_domain_repo_average": (sum(in_rates) / len(in_rates)) if in_rates else None,
        "out_of_domain_repo_average": (sum(out_rates) / len(out_rates)) if out_rates else None,
        "total_repo_prefix_average": (sum(all_prefix_rates) / len(all_prefix_rates)) if all_prefix_rates else None,
        "in_domain_repo_prefix_average": (sum(in_prefix_rates) / len(in_prefix_rates)) if in_prefix_rates else None,
        "out_of_domain_repo_prefix_average": (sum(out_prefix_rates) / len(out_prefix_rates))
        if out_prefix_rates
        else None,
        "in_domain_repo_count": len(in_rates),
        "out_of_domain_repo_count": len(out_rates),
        "train_repo_count": len(in_domain_repos) if in_domain_repos is not None else None,
    }


def get_syntax_domain_summary(
    results: list[dict],
    in_domain_repos: set[str] | None,
    eval_mode: str,
) -> dict:
    """
    Compute predicted syntax pass rates split by domain.
    - For ID/OOD eval runs: map overall syntax score into the corresponding side.
    - For Multi eval runs: split by in-domain repo membership when available.
    """
    checkable = [r for r in results if isinstance(r.get("predicted_syntax_passed"), bool)]

    def _rate(rows: list[dict]) -> float | None:
        if not rows:
            return None
        return sum(1 for r in rows if r.get("predicted_syntax_passed") is True) / len(rows)

    if eval_mode == "id":
        return {
            "syntax_in_domain_pass_rate": _rate(checkable),
            "syntax_out_of_domain_pass_rate": None,
            "syntax_in_domain_checkable_examples": len(checkable),
            "syntax_out_of_domain_checkable_examples": 0,
        }
    if eval_mode == "ood":
        return {
            "syntax_in_domain_pass_rate": None,
            "syntax_out_of_domain_pass_rate": _rate(checkable),
            "syntax_in_domain_checkable_examples": 0,
            "syntax_out_of_domain_checkable_examples": len(checkable),
        }

    if in_domain_repos is None:
        return {
            "syntax_in_domain_pass_rate": None,
            "syntax_out_of_domain_pass_rate": None,
            "syntax_in_domain_checkable_examples": 0,
            "syntax_out_of_domain_checkable_examples": 0,
        }

    in_rows: list[dict] = []
    out_rows: list[dict] = []
    for row in checkable:
        repo = str(row.get("repo") or infer_repo_name(str(row.get("file_path", ""))) or "unknown")
        if repo == "unknown":
            continue
        if repo in in_domain_repos:
            in_rows.append(row)
        else:
            out_rows.append(row)

    return {
        "syntax_in_domain_pass_rate": _rate(in_rows),
        "syntax_out_of_domain_pass_rate": _rate(out_rows),
        "syntax_in_domain_checkable_examples": len(in_rows),
        "syntax_out_of_domain_checkable_examples": len(out_rows),
    }


def _read_jsonl_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _resolve_id_eval_paths(config: EvaluationConfig) -> tuple[Path, Path]:
    data_dir = abs_path(config.data_dir)
    version = resolve_dataset_version(config.dataset_version, data_dir)
    print(f"Using dataset version: {version}")
    version_dir = data_dir / version
    return version_dir / "syntax_test.jsonl", version_dir / "random_test.jsonl"


def _repo_seed(base_seed: int, repo: str, salt: str) -> int:
    token = f"{base_seed}:{repo}:{salt}".encode("utf-8")
    digest = hashlib.md5(token).hexdigest()
    return int(digest[:8], 16)


def _sample_rows_per_repo(
    syntax_rows: list[dict],
    random_rows: list[dict],
    samples_per_repo: int,
    syntax_ratio: float,
    random_seed: int,
) -> list[dict]:
    if samples_per_repo <= 0:
        return []

    syntax_by_repo: dict[str, list[dict]] = defaultdict(list)
    random_by_repo: dict[str, list[dict]] = defaultdict(list)
    for row in syntax_rows:
        syntax_by_repo[str(row.get("repo", "unknown"))].append(row)
    for row in random_rows:
        random_by_repo[str(row.get("repo", "unknown"))].append(row)

    selected: list[dict] = []
    all_repos = sorted(set(syntax_by_repo.keys()) | set(random_by_repo.keys()))
    for repo in all_repos:
        syntax_repo_rows = list(syntax_by_repo.get(repo, []))
        random_repo_rows = list(random_by_repo.get(repo, []))

        rng_syntax = random.Random(_repo_seed(random_seed, repo, "syntax"))
        rng_random = random.Random(_repo_seed(random_seed, repo, "random"))
        rng_fill = random.Random(_repo_seed(random_seed, repo, "fill"))
        rng_syntax.shuffle(syntax_repo_rows)
        rng_random.shuffle(random_repo_rows)

        target_syntax = int(samples_per_repo * syntax_ratio)
        target_random = samples_per_repo - target_syntax

        taken_syntax = syntax_repo_rows[: min(len(syntax_repo_rows), target_syntax)]
        taken_random = random_repo_rows[: min(len(random_repo_rows), target_random)]

        remaining = samples_per_repo - len(taken_syntax) - len(taken_random)
        if remaining > 0:
            leftover = syntax_repo_rows[len(taken_syntax) :] + random_repo_rows[len(taken_random) :]
            rng_fill.shuffle(leftover)
            filler = leftover[:remaining]
        else:
            filler = []

        selected.extend(taken_syntax)
        selected.extend(taken_random)
        selected.extend(filler)

    rng_final = random.Random(random_seed + 2003)
    rng_final.shuffle(selected)
    return selected


def get_rows(config: EvaluationConfig, mode: str | None = None, dataset_path: str | None = None) -> list[dict]:
    """Load FIM examples for evaluation."""
    if mode is None:
        mode = resolve_eval_mode(config)
    mode = str(mode).strip().lower()
    if mode in {"id", "ood"}:
        print(f"Deprecated get_rows mode='{mode}' requested; forcing mode='all'.")
        mode = "all"
    if mode != "all":
        raise ValueError(f"Invalid mode: {mode}")
    if dataset_path:
        path = abs_path(dataset_path) if not Path(dataset_path).is_absolute() else Path(dataset_path)
        rows = _read_jsonl_rows(path)
        add_repo_fields(rows)
        samples_per_repo = config.samples_per_repo
        if samples_per_repo is not None:
            by_repo: dict[str, list[dict]] = defaultdict(list)
            for row in rows:
                by_repo[str(row.get("repo", "unknown"))].append(row)
            sampled: list[dict] = []
            for repo, repo_rows in by_repo.items():
                seed = _repo_seed(config.random_seed, repo, "custom")
                rng = random.Random(seed)
                repo_rows = list(repo_rows)
                rng.shuffle(repo_rows)
                sampled.extend(repo_rows[:samples_per_repo])
            rng_final = random.Random(config.random_seed + 2003)
            rng_final.shuffle(sampled)
            return sampled
        if config.max_samples:
            return rows[: config.max_samples]
        return rows

    syntax_rows: list[dict] = []
    random_rows: list[dict] = []

    id_syntax_path, id_random_path = _resolve_id_eval_paths(config)
    if id_syntax_path.exists():
        syntax_rows.extend(_read_jsonl_rows(id_syntax_path))
    if id_random_path.exists():
        random_rows.extend(_read_jsonl_rows(id_random_path))
    print(f"Unified eval data (ID source): syntax={id_syntax_path}, random={id_random_path}")

    if not syntax_rows and not random_rows:
        raise ValueError("No evaluation rows found.")

    for r in syntax_rows:
        r["dataset_source"] = "syntax"
    for r in random_rows:
        r["dataset_source"] = "random"

    add_repo_fields(syntax_rows)
    add_repo_fields(random_rows)

    samples_per_repo = config.samples_per_repo
    if samples_per_repo is not None:
        return _sample_rows_per_repo(
            syntax_rows=syntax_rows,
            random_rows=random_rows,
            samples_per_repo=int(samples_per_repo),
            syntax_ratio=float(config.syntax_ratio),
            random_seed=int(config.random_seed),
        )

    # Legacy fallback.
    rng_syntax = random.Random(config.random_seed)
    rng_random = random.Random(config.random_seed + 1)
    rng_syntax.shuffle(syntax_rows)
    rng_random.shuffle(random_rows)

    max_n = config.max_samples
    if max_n:
        n_syntax = int(max_n * config.syntax_ratio)
        n_random = max_n - n_syntax
        syntax_rows = syntax_rows[:n_syntax]
        random_rows = random_rows[:n_random]

    rows = syntax_rows + random_rows
    rng_final = random.Random(config.random_seed + 2)
    rng_final.shuffle(rows)
    return rows


def get_log_ready_config(config: EvaluationConfig) -> dict:
    """
    Returns a config dict suitable for logging/display.
    Resolves 'latest' dataset_version to the actual version.
    """
    config_dict = config.__dict__.copy()

    if config.dataset_version == "latest":
        config_dict["dataset_version"] = resolve_dataset_version("latest", config.data_dir)

    # cv_folds holds CvFold dataclasses; render them as plain dicts so the config
    # stays JSON-serializable in summary.json.
    if config_dict.get("cv_folds"):
        config_dict["cv_folds"] = [asdict(f) if is_dataclass(f) else f for f in config_dict["cv_folds"]]

    return config_dict


def _extract_cv_eval_family(eval_name: str) -> str:
    """Return the eval family base, stripping any trailing '[cv i/n]'/'[fold i/n]'."""
    name = (eval_name or "").strip()
    if not name:
        return ""
    match = CV_EVAL_NAME_FOLD_RE.match(name)
    if not match:
        return name
    return (match.group(1) or "").strip() or name


def _format_cv_fold_run_name(parent_run_name: str, fold_index: int, fold_count: int) -> str:
    if not parent_run_name:
        return ""
    return f"{parent_run_name} [fold {fold_index}/{fold_count}]"


def _warn_if_cv_folds_not_partition(cv_name: str, folds: list[CvFold]) -> None:
    """Warn (non-fatal) if the folds' in-domain repos don't form a leave-one-group-out
    partition.

    In standard k-fold CV each repo is trained on in k-1 folds and held out
    (out-of-domain) in exactly one. So a repo that is in-domain in *every* fold is
    never scored OOD, and a repo held out in *more than one* fold means the holdout
    groups overlap. Either is usually a mistake in how the folds were declared.
    """
    if len(folds) < 2:
        return
    in_domain_sets = [{r.strip() for r in fold.in_domain_repos if r.strip()} for fold in folds]
    universe = set().union(*in_domain_sets)
    never_heldout = sorted(r for r in universe if all(r in s for s in in_domain_sets))
    over_heldout = sorted(r for r in universe if sum(r not in s for s in in_domain_sets) > 1)
    label = f" (cv_name={cv_name})" if cv_name else ""
    if never_heldout:
        print(f"Warning: CV folds{label} never hold out {never_heldout} (always in-domain — never scored OOD).")
    if over_heldout:
        print(
            f"Warning: CV folds{label} hold out {over_heldout} in more than one fold "
            "(overlapping holdouts — folds don't form a clean partition)."
        )


def _expand_cv_folds(experiments: list[EvaluationConfig]) -> list[EvaluationConfig]:
    """Turn each cross-validation experiment into one runnable config per fold.

    An experiment that sets ``cv_folds`` bundles N fold models into a single
    object and is not directly runnable. Replace each such experiment with N
    separate ``EvaluationConfig``s — one per fold — where every fold config is a
    copy of the experiment with ``model_name`` and ``in_domain_repos`` taken from
    that fold, all sharing one freshly minted ``cv_parent_run_id`` (and tagged
    ``cv_subrun_index`` / ``cv_subrun_count``). Those configs are run
    independently; ``_write_eval_cv_parent_summaries`` later groups their
    summaries by ``cv_parent_run_id`` into a single CV-parent summary.

    Example: an experiment with ``cv_folds=[A, B, C]`` becomes three configs run
    as "… [fold 1/3]", "… [fold 2/3]", "… [fold 3/3]". Experiments without
    ``cv_folds`` are returned unchanged.
    """
    expanded: list[EvaluationConfig] = []
    for exp in experiments:
        if not exp.cv_folds:
            expanded.append(exp)
            continue

        _warn_if_cv_folds_not_partition(exp.cv_name or "", exp.cv_folds)
        eval_family = _extract_cv_eval_family(exp.eval_name or "") or exp.cv_name or "cv"
        parent_run_id = str(uuid4())
        fold_count = len(exp.cv_folds)
        for fold_idx, fold in enumerate(exp.cv_folds, start=1):
            notes = f"cv_fold={fold_idx}/{fold_count}"
            if exp.cv_name:
                notes = f"cv_name={exp.cv_name}; {notes}"
            if exp.notes:
                notes = f"{exp.notes}; {notes}"
            expanded.append(
                replace(
                    exp,
                    cv_folds=[],
                    model_name=fold.model_name,
                    in_domain_repos=list(fold.in_domain_repos) or None,
                    notes=notes,
                    cv_parent_run_id=parent_run_id,
                    cv_subrun_index=fold_idx,
                    cv_subrun_count=fold_count,
                    eval_name=_format_cv_fold_run_name(eval_family, fold_idx, fold_count),
                )
            )
    return expanded


def _coerce_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _write_eval_cv_parent_summary(
    *,
    parent_run_id: str,
    fold_summaries: list[dict],
    logs_dir: str,
) -> None:
    """Write one aggregated CV parent summary for eval-only CV fold runs."""
    if not fold_summaries:
        return

    parent_dir = abs_path(logs_dir) / parent_run_id
    parent_dir.mkdir(parents=True, exist_ok=True)

    ordered = sorted(
        fold_summaries,
        key=lambda s: (
            _coerce_int(s.get("cv_subrun_index")) or 10_000,
            str(s.get("run_id", "")),
        ),
    )

    subruns: list[dict] = []
    duration_seconds = 0.0
    timestamps: list[str] = []
    for summary in ordered:
        fold_index = _coerce_int(summary.get("cv_subrun_index")) or 0
        fold_duration = summary.get("duration_seconds")
        if isinstance(fold_duration, (int, float)):
            duration_seconds += float(fold_duration)
        ts = str(summary.get("timestamp", "") or "").strip()
        if ts:
            timestamps.append(ts)
        holdout_repos = summary.get("cv_holdout_repos", [])
        if not isinstance(holdout_repos, list):
            holdout_repos = []
        train_repos = summary.get("cv_train_repos", [])
        if not isinstance(train_repos, list):
            train_repos = []
        subruns.append(
            {
                "fold_index": fold_index,
                "holdout_repos": holdout_repos,
                "train_repos": train_repos,
                "eval_summary": summary,
            }
        )

    aggregated = aggregate_cv_eval_summaries(subruns)
    first_summary = ordered[0]
    first_config = first_summary.get("config", {})
    config_payload = dict(first_config) if isinstance(first_config, dict) else {}
    config_payload["cv_parent_run_id"] = parent_run_id
    config_payload["cv_subrun_index"] = None
    config_payload["cv_subrun_count"] = len(ordered)

    eval_family = _extract_cv_eval_family(str(config_payload.get("eval_name", "") or ""))
    if eval_family:
        config_payload["eval_name"] = eval_family

    parent_timestamp = max(timestamps) if timestamps else datetime.now().strftime("%y%m%d-%H%M%S")

    summary = {
        "run_id": parent_run_id,
        "run_name": str(config_payload.get("eval_name", "") or "").strip(),
        "model": first_summary.get("model", "unknown"),
        "model_path": first_summary.get("model_path", first_summary.get("model", "unknown")),
        "timestamp": parent_timestamp,
        "config": config_payload,
        "dataset": "cv",
        "dataset_version": first_summary.get("dataset_version"),
        "syntax_ratio": first_summary.get("syntax_ratio"),
        "samples_per_repo": first_summary.get("samples_per_repo"),
        "total_examples": aggregated.get("total_examples", 0),
        "exact_match_rate": aggregated.get("exact_match_rate"),
        "prefix_match_rate": aggregated.get("prefix_match_rate"),
        "empty_response_rate": aggregated.get("empty_response_rate"),
        "duration_seconds": duration_seconds,
        "by_category": aggregated.get("by_category", {}),
        "by_source": aggregated.get("by_source", {}),
        "repo_scores": aggregated.get("repo_scores", []),
        "in_domain_repo_average": aggregated.get("in_domain_repo_average"),
        "out_of_domain_repo_average": aggregated.get("out_of_domain_repo_average"),
        "total_repo_prefix_average": aggregated.get("total_repo_prefix_average"),
        "in_domain_repo_prefix_average": aggregated.get("in_domain_repo_prefix_average"),
        "out_of_domain_repo_prefix_average": aggregated.get("out_of_domain_repo_prefix_average"),
        "in_domain_repo_count": aggregated.get("in_domain_repo_count", 0),
        "out_of_domain_repo_count": aggregated.get("out_of_domain_repo_count", 0),
        "train_repo_count": aggregated.get("train_repo_count"),
        "predicted_syntax_pass_rate": aggregated.get("predicted_syntax_pass_rate"),
        "original_syntax_pass_rate": aggregated.get("original_syntax_pass_rate"),
        "syntax_original_file_pass_rate": aggregated.get("syntax_original_file_pass_rate"),
        "syntax_checkable_examples": aggregated.get("syntax_checkable_examples", 0),
        "syntax_in_domain_pass_rate": aggregated.get("syntax_in_domain_pass_rate"),
        "syntax_out_of_domain_pass_rate": aggregated.get("syntax_out_of_domain_pass_rate"),
        "status": "completed",
        "cv_subrun_count": len(ordered),
        "cv_subrun_run_ids": aggregated.get("cv_subrun_run_ids", []),
        "cv_subruns": aggregated.get("cv_subruns", []),
    }
    with open(parent_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)


def _write_eval_cv_parent_summaries(eval_summaries: list[tuple[dict, str]]) -> None:
    """
    Aggregate and write CV parent summaries for eval-list runs.
    Expects one tuple per finished eval run: (summary_dict, logs_dir).
    """
    grouped: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    expected_counts: dict[tuple[str, str], int] = {}

    for summary, logs_dir in eval_summaries:
        if not isinstance(summary, dict):
            continue
        parent_run_id = str(summary.get("cv_parent_run_id", "") or "").strip()
        fold_idx = _coerce_int(summary.get("cv_subrun_index"))
        fold_count = _coerce_int(summary.get("cv_subrun_count"))
        if not parent_run_id or fold_idx is None or fold_count is None or fold_count < 2:
            continue
        key = (str(logs_dir), parent_run_id)
        expected_counts[key] = max(expected_counts.get(key, 0), fold_count)
        grouped[key][fold_idx] = summary

    for (logs_dir, parent_run_id), by_fold_idx in grouped.items():
        expected = expected_counts.get((logs_dir, parent_run_id), 0)
        if expected <= 1:
            continue
        missing = [i for i in range(1, expected + 1) if i not in by_fold_idx]
        if missing:
            print(f"Skipping CV parent summary for {parent_run_id}: missing fold(s) {missing}")
            continue
        ordered = [by_fold_idx[i] for i in range(1, expected + 1)]
        _write_eval_cv_parent_summary(
            parent_run_id=parent_run_id,
            fold_summaries=ordered,
            logs_dir=logs_dir,
        )
        print(f"CV parent summary written: {logs_dir}/{parent_run_id}/summary.json")


def _cli_arg_specs() -> list[tuple[str, str, dict]]:
    """Build ``(flag, dest, add_argument_kwargs)`` for each EvaluationConfig field
    that maps to a scalar or ``list[str]`` command-line flag.

    ``cv_folds`` (``list[CvFold]``) can't be expressed on the command line and is
    available only via ``--config``.
    """
    specs: list[tuple[str, str, dict]] = []
    hints = typing.get_type_hints(EvaluationConfig)
    for f in fields(EvaluationConfig):
        ann = hints[f.name]
        if typing.get_origin(ann) in (typing.Union, types.UnionType):
            non_none = [a for a in typing.get_args(ann) if a is not type(None)]
            ann = non_none[0] if len(non_none) == 1 else None
        flag = "--" + f.name.replace("_", "-")
        if ann is bool:
            specs.append((flag, f.name, {"action": argparse.BooleanOptionalAction}))
        elif ann in (int, float, str):
            specs.append((flag, f.name, {"type": ann}))
        elif typing.get_origin(ann) is list and typing.get_args(ann) == (str,):
            specs.append((flag, f.name, {"nargs": "*", "metavar": "REPO"}))
    return specs


def _load_experiments_from_file(path: str) -> list[EvaluationConfig]:
    """Load ``EXPERIMENT`` (an EvaluationConfig) or ``EXPERIMENTS`` (a list of them)
    from a user-supplied Python config file, or one experiment from a ``.json``
    file holding an object of EvaluationConfig fields."""
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            return [EvaluationConfig(**json.load(f))]
    spec = importlib.util.spec_from_file_location("daml_fim_benchmark._user_experiment", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Could not load config file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "EXPERIMENTS"):
        experiments = list(module.EXPERIMENTS)
    elif hasattr(module, "EXPERIMENT"):
        experiments = [module.EXPERIMENT]
    else:
        raise ValueError(f"{path} must define EXPERIMENT (an EvaluationConfig) or EXPERIMENTS (a list of them).")
    for i, exp in enumerate(experiments, start=1):
        if not isinstance(exp, EvaluationConfig):
            raise TypeError(f"{path}: experiment {i} must be an EvaluationConfig, got {type(exp).__name__}.")
    return experiments


DEFAULT_CONFIG_FILE = "experiment.py"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m daml_fim_benchmark.evaluate",
        description=(
            "Run the Daml FIM benchmark. With no arguments it loads "
            f"{DEFAULT_CONFIG_FILE} (a Python file defining EXPERIMENT / EXPERIMENTS); "
            "otherwise configure the run with the per-field flags below, or point "
            "--config at such a file."
        ),
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help=f"Python file defining EXPERIMENT (an EvaluationConfig) or EXPERIMENTS (a list), "
        "or a .json file holding one object of EvaluationConfig fields; "
        f"defaults to {DEFAULT_CONFIG_FILE} when present. Mutually exclusive with the per-field flags.",
    )
    for flag, dest, kwargs in _cli_arg_specs():
        parser.add_argument(flag, dest=dest, default=argparse.SUPPRESS, **kwargs)
    args = parser.parse_args(argv)

    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    if args.config:
        if overrides:
            parser.error("--config cannot be combined with per-field flags.")
        experiments = _load_experiments_from_file(args.config)
    elif overrides:
        experiments = [EvaluationConfig(**overrides)]
    elif abs_path(DEFAULT_CONFIG_FILE).is_file():
        print(f"No --config or flags given; using {DEFAULT_CONFIG_FILE}.")
        experiments = _load_experiments_from_file(str(abs_path(DEFAULT_CONFIG_FILE)))
    else:
        parser.error(
            f"No experiment specified. Pass per-field flags (e.g. --model-name ...), "
            f"--config PATH, or create {DEFAULT_CONFIG_FILE} (copy experiment.example.py)."
        )

    experiments = _expand_cv_folds(experiments)
    total = len(experiments)
    completed_summaries: list[tuple[dict, str]] = []
    for idx, config in enumerate(experiments, start=1):
        print("=" * 80)
        print(
            f"Starting eval experiment {idx}/{total}: model={config.model_name}, "
            f"dataset_version={config.dataset_version}, samples_per_repo={config.samples_per_repo}"
        )
        print("=" * 80)
        summary = run_evaluation(config)
        if isinstance(summary, dict):
            completed_summaries.append((summary, config.logs_dir))

    _write_eval_cv_parent_summaries(completed_summaries)


if __name__ == "__main__":
    main()
