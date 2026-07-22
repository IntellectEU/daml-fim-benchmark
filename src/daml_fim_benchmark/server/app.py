"""FastAPI results/run-management server behind the benchmark web UI.

Every evaluation run writes a ``summary.json`` (and per-example ``results.jsonl``)
into a folder under ``fim/logs/`` (see :mod:`daml_fim_benchmark.evaluate`). This
server reads those folders back and exposes them as a small REST API that the
React frontend calls to browse, inspect, compare and manage runs:

- ``GET  /api/fim/runs`` — list runs with their headline metrics, for the table.
- ``GET  /api/fim/runs/{run_id}/results`` and ``.../results/{result_id}`` —
  the per-example completions for one run.
- ``PUT  /api/fim/runs/{run_id}/notes`` — edit a run's free-text notes.
- ``POST /api/fim/runs/archive`` and ``.../delete`` — bulk archive (move to
  ``z_archive/``) or delete runs.
- ``POST /api/fim/runs/compare`` — diff metrics across several runs side by side.
- ``POST /api/system/backend/restart`` — restart this server from the UI.

Paths are not hardcoded: the benchmark root comes from
``DAML_FIM_BENCHMARK_ROOT`` (default: current working directory) and the frontend
directory from ``FIM_FRONTEND_DIR``, so the server runs against whatever checkout
or deployment you point it at.
"""

from __future__ import annotations

from functools import lru_cache
import json
import hashlib
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

def _benchmark_root() -> Path:
    """Root directory that benchmark artifacts resolve against.

    Defaults to the current working directory; override with the
    ``DAML_FIM_BENCHMARK_ROOT`` environment variable.
    """
    env = os.getenv("DAML_FIM_BENCHMARK_ROOT")
    return Path(env).expanduser().resolve() if env else Path.cwd()


ROOT_DIR = _benchmark_root()
# Frontend dir defaults to a sibling "frontend/" of the benchmark root; override
# with FIM_FRONTEND_DIR.
REACT_FRONTEND_DIR = Path(os.getenv("FIM_FRONTEND_DIR") or (ROOT_DIR / "frontend")).expanduser()
RESTART_BACKEND_SCRIPT = REACT_FRONTEND_DIR / "scripts" / "restart_backend.sh"
RESTART_BACKEND_LOG = REACT_FRONTEND_DIR / ".logs" / "backend_control.log"
FIM_LOGS_DIR = ROOT_DIR / "fim" / "logs"
FIM_ARCHIVE_DIR = FIM_LOGS_DIR / "z_archive"
# CORS allowlist for the frontend dev server; override with FIM_CORS_ORIGINS
# (comma-separated) since Vite doesn't always land on 5173.
DEFAULT_CORS_ORIGINS: tuple[str, ...] = (
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
)

VALID_EVAL_FILTERS = {"All", "Multi", "ID", "OOD", "Undisclosed", "CV"}


def _ensure_dirs() -> None:
    FIM_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    FIM_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def _safe_json_load(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _safe_text_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _safe_text_read_raw(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _to_float_or_none(value: Any, field: str = "value") -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Invalid float for {field}: bool {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float for {field}: {value!r}") from exc


def _to_int_or_none(value: Any, field: str = "value") -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Invalid int for {field}: bool {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid int for {field}: {value!r}") from exc


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _normalize_repo_scores(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        repo = str(raw.get("repo") or "").strip()
        if not repo:
            continue
        score = _to_float_or_none(
            raw.get("exact_match_rate"),
            field=f"repo_scores[{repo}].exact_match_rate",
        )
        prefix_rate = _to_float_or_none(
            raw.get("prefix_match_rate"),
            field=f"repo_scores[{repo}].prefix_match_rate",
        )
        total_examples = _to_int_or_none(
            raw.get("total_examples"),
            field=f"repo_scores[{repo}].total_examples",
        )
        domain = str(raw.get("domain") or "").strip().lower()
        in_domain_raw = raw.get("in_domain")
        in_domain = in_domain_raw if isinstance(in_domain_raw, bool) else None
        out.append(
            {
                "repo": repo,
                "exact_match_rate": score,
                "prefix_match_rate": prefix_rate,
                "total_examples": total_examples,
                "domain": domain if domain else None,
                "in_domain": in_domain,
            }
        )
    return out


def _mean_or_none(values: Iterable[float]) -> float | None:
    rates = list(values)
    return (sum(rates) / len(rates)) if rates else None


def _build_repo_maps(
    repo_scores: list[dict[str, Any]],
    train_repos: list[str],
) -> tuple[dict[str, float], dict[str, str], dict[str, int]]:
    repo_rate_map: dict[str, float] = {}
    repo_domain_map: dict[str, str] = {}
    repo_example_count_map: dict[str, int] = {}
    train_repo_set = set(train_repos)

    for entry in repo_scores:
        repo = str(entry.get("repo") or "").strip()
        if not repo:
            continue
        rate = _to_float_or_none(
            entry.get("exact_match_rate"),
            field=f"repo_scores[{repo}].exact_match_rate",
        )
        if rate is not None:
            repo_rate_map[repo] = rate

        total_examples = _to_int_or_none(
            entry.get("total_examples"),
            field=f"repo_scores[{repo}].total_examples",
        )
        if total_examples is not None:
            repo_example_count_map[repo] = total_examples

        domain = str(entry.get("domain") or "").strip().lower()
        if domain == "in_domain":
            repo_domain_map[repo] = "id"
        elif domain == "out_of_domain":
            repo_domain_map[repo] = "ood"
        elif repo in train_repo_set:
            repo_domain_map[repo] = "id"
        else:
            in_domain = entry.get("in_domain")
            if isinstance(in_domain, bool):
                repo_domain_map[repo] = "id" if in_domain else "ood"

    return repo_rate_map, repo_domain_map, repo_example_count_map


def _build_repo_prefix_maps(
    repo_scores: list[dict[str, Any]],
    train_repos: list[str],
) -> tuple[dict[str, float], float | None, float | None, float | None]:
    repo_prefix_rate_map: dict[str, float] = {}
    train_repo_set = set(train_repos)
    in_rates: list[float] = []
    out_rates: list[float] = []

    for entry in repo_scores:
        repo = str(entry.get("repo") or "").strip()
        if not repo:
            continue
        rate = _to_float_or_none(
            entry.get("prefix_match_rate"),
            field=f"repo_scores[{repo}].prefix_match_rate",
        )
        if rate is None:
            continue
        repo_prefix_rate_map[repo] = rate

        domain = str(entry.get("domain") or "").strip().lower()
        if domain == "in_domain":
            in_rates.append(rate)
        elif domain == "out_of_domain":
            out_rates.append(rate)
        elif repo in train_repo_set:
            in_rates.append(rate)
        else:
            in_domain = entry.get("in_domain")
            if isinstance(in_domain, bool):
                if in_domain:
                    in_rates.append(rate)
                else:
                    out_rates.append(rate)

    all_rates = list(repo_prefix_rate_map.values())
    total_avg = (sum(all_rates) / len(all_rates)) if all_rates else None
    id_avg = (sum(in_rates) / len(in_rates)) if in_rates else None
    ood_avg = (sum(out_rates) / len(out_rates)) if out_rates else None
    return repo_prefix_rate_map, id_avg, ood_avg, total_avg


def _is_timeout_result(row: dict[str, Any]) -> bool:
    if str(row.get("error_type") or "").strip() == "TimeoutError":
        return True
    error = row.get("error")
    if isinstance(error, dict):
        message = str(error.get("message") or "")
    else:
        message = str(error or "")
    return "timed out" in message.lower()


def _repo_name_for_result_row(row: dict[str, Any]) -> str:
    repo = str(row.get("repo") or "").strip()
    if repo:
        return repo
    return "unknown"


@lru_cache(maxsize=2048)
def _compute_repo_prefix_stats_from_results_cache(
    results_path_str: str,
    mtime_ns: int,
    train_repos_key: tuple[str, ...],
) -> tuple[dict[str, float], float | None, float | None, float | None]:
    del mtime_ns  # cache key includes mtime for invalidation
    train_repo_set = set(train_repos_key)

    repo_totals: dict[str, dict[str, int]] = {}
    for row in _iter_results_file(Path(results_path_str)):
        if _is_timeout_result(row):
            continue
        repo = _repo_name_for_result_row(row)
        bucket = repo_totals.setdefault(repo, {"total": 0, "prefix": 0})
        bucket["total"] += 1
        if bool(row.get("prefix_match", False)):
            bucket["prefix"] += 1

    repo_prefix_rate_map: dict[str, float] = {}
    in_rates: list[float] = []
    out_rates: list[float] = []

    for repo in sorted(repo_totals.keys()):
        total = repo_totals[repo]["total"]
        if total <= 0:
            continue
        rate = repo_totals[repo]["prefix"] / total
        repo_prefix_rate_map[repo] = rate
        if train_repo_set and repo != "unknown":
            if repo in train_repo_set:
                in_rates.append(rate)
            else:
                out_rates.append(rate)

    all_rates = list(repo_prefix_rate_map.values())
    total_avg = (sum(all_rates) / len(all_rates)) if all_rates else None
    id_avg = (sum(in_rates) / len(in_rates)) if in_rates else None
    ood_avg = (sum(out_rates) / len(out_rates)) if out_rates else None
    return repo_prefix_rate_map, id_avg, ood_avg, total_avg


def _compute_repo_prefix_stats_from_results(
    results_path: Path,
    train_repos: list[str],
) -> tuple[dict[str, float], float | None, float | None, float | None]:
    if not results_path.exists() or not results_path.is_file():
        return {}, None, None, None
    try:
        stat = results_path.stat()
    except OSError:
        return {}, None, None, None
    train_repos_key = tuple(sorted({str(repo).strip() for repo in train_repos if str(repo).strip()}))
    return _compute_repo_prefix_stats_from_results_cache(
        str(results_path),
        int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
        train_repos_key,
    )


def _parse_compact_timestamp(timestamp: str | None) -> datetime | None:
    if not timestamp:
        return None
    text = str(timestamp).strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _extract_timestamp_from_run_id(run_id: str) -> datetime | None:
    match = re.search(r"(\d{6}-\d{6})", run_id)
    if not match:
        return None
    return _parse_compact_timestamp(match.group(1))


def _infer_eval_kind(dataset: str | None, run_id: str) -> str:
    ds = (dataset or "").strip().lower()
    rid = (run_id or "").lower()

    if ds in {"cv", "cross_validation", "cross-validation"}:
        return "CV"
    if ds in {"all", "combined", "multi"}:
        return "Multi"
    if ds == "ood":
        return "OOD"
    if ds in {"mixed", "id", "in"}:
        return "ID"
    if ds in {"undisclosed", "unknown"}:
        return "Undisclosed"

    if "_cv_" in rid or "-cv-" in rid:
        return "CV"
    if "_all_" in rid or "-all-" in rid:
        return "Multi"
    if "_ood_" in rid or "-ood-" in rid:
        return "OOD"
    if "_id_" in rid or "-id-" in rid or "_in_" in rid or "_eval_in_" in rid:
        return "ID"

    # Without a declared training-data relationship we cannot know whether the
    # eval repos are in- or out-of-domain for the model.
    return "Undisclosed"


def _infer_dataset_label(summary: dict[str, Any], config: dict[str, Any]) -> str:
    dataset = str(summary.get("dataset", "")).strip().lower()
    if dataset:
        return dataset

    eval_mode = str(config.get("eval_mode", "")).strip().lower()
    if eval_mode in {"all", "id", "ood"}:
        return eval_mode
    if bool(config.get("eval_on_ood", False)):
        return "ood"
    # Nothing declared — leave the label empty so eval-kind inference falls
    # through to its own fallback (Undisclosed) instead of fabricating "id".
    return ""


def _infer_dataset_version(summary: dict[str, Any], config: dict[str, Any]) -> str | None:
    # Prefer explicit evaluation dataset version when present.
    for raw in (
        config.get("eval_dataset_version"),
        summary.get("eval_dataset_version"),
        summary.get("dataset_version"),
        config.get("dataset_version"),
    ):
        value = str(raw or "").strip()
        if value:
            return value
    return None


def _get_base_model_name(model_name: str | None) -> str:
    if not model_name:
        return ""
    value = str(model_name)
    if "/" in value:
        return value.split("/")[-1]
    return value


def _looks_like_finetune_run_name(model_name: str | None) -> bool:
    text = str(model_name or "").strip().lower()
    return text.startswith("run_")


def _remap_to_local_project_path(path_value: Path) -> Path:
    path = path_value.expanduser()
    if path.exists():
        return path
    parts = path.parts
    try:
        marker_idx = parts.index(ROOT_DIR.name)
    except ValueError:
        return path
    if marker_idx + 1 >= len(parts):
        return ROOT_DIR
    return ROOT_DIR / Path(*parts[marker_idx + 1 :])


@lru_cache(maxsize=4096)
def _base_model_from_adapter_dir(path_text: str) -> str:
    text = str(path_text or "").strip()
    if not text:
        return ""
    model_dir = Path(text).expanduser()
    if not model_dir.is_absolute():
        model_dir = ROOT_DIR / model_dir
    model_dir = _remap_to_local_project_path(model_dir)
    if not model_dir.exists() or not model_dir.is_dir():
        return ""

    candidate_files = (
        (model_dir / "fim_training_config.json", "base_model_name"),
        (model_dir / "adapter_config.json", "base_model_name_or_path"),
    )
    for file_path, key in candidate_files:
        payload = _safe_json_load(file_path)
        candidate = _get_base_model_name(payload.get(key))
        if candidate and not _looks_like_finetune_run_name(candidate):
            return candidate
    return ""


def _infer_model_kind(summary: dict[str, Any], config: dict[str, Any]) -> str:
    model = summary.get("model")
    model_kind = _get_base_model_name(model)
    if model_kind and not _looks_like_finetune_run_name(model_kind):
        return model_kind

    for key in ("base_model_name", "base_model_name_or_path", "model_name"):
        candidate = _get_base_model_name(config.get(key))
        if candidate and not _looks_like_finetune_run_name(candidate):
            return candidate

    for path_like in (summary.get("model_path"), config.get("model")):
        path_text = str(path_like or "").strip()
        if not path_text:
            continue

        candidate = _base_model_from_adapter_dir(path_text)
        if candidate:
            return candidate

        path_tail = _get_base_model_name(path_text)
        if path_tail and not _looks_like_finetune_run_name(path_tail):
            return path_tail

    return model_kind


def _get_color_from_string(text: str) -> str:
    if not text:
        return "#d3d3d3"
    hex_dig = hashlib.md5(text.encode("utf-8")).hexdigest()
    r = int(hex_dig[0:2], 16)
    g = int(hex_dig[2:4], 16)
    b = int(hex_dig[4:6], 16)
    r = (r + 255) // 2
    g = (g + 255) // 2
    b = (b + 255) // 2
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_to_rgb(color: str) -> tuple[int, int, int] | None:
    text = str(color or "").strip()
    if len(text) != 7 or not text.startswith("#"):
        return None
    try:
        return int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16)
    except ValueError:
        return None


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return f"#{max(0, min(255, int(r))):02x}{max(0, min(255, int(g))):02x}{max(0, min(255, int(b))):02x}"


def _dataset_version_badge_style(dataset_version: str) -> tuple[str, str, str]:
    bg = _get_color_from_string(dataset_version or "")
    rgb = _hex_to_rgb(bg)
    if rgb is None:
        return "#ecfeff", "#155e75", "#67e8f9"

    r, g, b = rgb
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    fg = "#111827" if luminance >= 0.62 else "#f9fafb"
    border = _rgb_to_hex(r * 0.72, g * 0.72, b * 0.72)
    return bg, fg, border


def _iter_run_dirs(include_archived: bool) -> list[tuple[str, Path, bool]]:
    _ensure_dirs()
    out: list[tuple[str, Path, bool]] = []

    for child in FIM_LOGS_DIR.iterdir():
        if not child.is_dir() or child.name == "z_archive":
            continue
        if not ((child / "summary.json").exists() or (child / "results.jsonl").exists()):
            continue
        out.append((child.name, child, False))

    if include_archived:
        for child in FIM_ARCHIVE_DIR.iterdir():
            if not child.is_dir():
                continue
            if not ((child / "summary.json").exists() or (child / "results.jsonl").exists()):
                continue
            out.append((child.name, child, True))

    def timestamp_key(item: tuple[str, Path, bool]) -> tuple[int, float]:
        _, run_dir, _archived = item
        summary = _safe_json_load(run_dir / "summary.json")
        ts = _parse_compact_timestamp(summary.get("timestamp"))
        if ts is None:
            ts = _extract_timestamp_from_run_id(run_dir.name)
        if ts is not None:
            return (1, ts.timestamp())
        return (0, 0.0)

    # Sort by run_id first (stable, ascending), then by timestamp descending.
    # `list.sort` is stable, so runs with equal timestamp keys — in particular
    # all null-timestamp runs, which share key (0, 0.0) — keep the run_id
    # ascending order from the first pass instead of arbitrary filesystem order.
    out.sort(key=lambda item: item[0])
    out.sort(key=timestamp_key, reverse=True)
    return out


def _normalize_run_id(run_id: str) -> str:
    value = str(run_id or "").strip()
    if not value:
        raise HTTPException(status_code=400, detail="run_id is required")
    if "/" in value or "\\" in value or ".." in value:
        raise HTTPException(status_code=400, detail=f"Invalid run_id: {run_id}")
    return value


def _find_run_dir(run_id: str) -> tuple[Path, bool]:
    run_id = _normalize_run_id(run_id)
    _ensure_dirs()
    active = FIM_LOGS_DIR / run_id
    if active.exists() and active.is_dir():
        return active, False
    archived = FIM_ARCHIVE_DIR / run_id
    if archived.exists() and archived.is_dir():
        return archived, True
    raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")


def _is_run_in_progress(run_id: str, run_dir: Path, archived: bool) -> bool:
    if archived:
        return False

    summary_path = run_dir / "summary.json"
    results_path = run_dir / "results.jsonl"

    explicit_in_progress = False
    if summary_path.exists():
        summary = _safe_json_load(summary_path)
        explicit_in_progress = summary.get("status") == "in_progress"

    if explicit_in_progress and not results_path.exists():
        return True

    if not results_path.exists():
        return False

    try:
        results_mtime = results_path.stat().st_mtime
        if explicit_in_progress:
            return True

        stale_threshold_seconds = 120
        now_ts = datetime.now().timestamp()

        if not summary_path.exists():
            return (now_ts - results_mtime) <= stale_threshold_seconds

        if (now_ts - results_mtime) > stale_threshold_seconds:
            return False

        # A fresh `git clone` writes both files within the same second, which
        # would otherwise look identical to a run that appended to
        # results.jsonl for minutes before writing summary.json at the end.
        # Require a real margin, not just "newer", to tell them apart.
        summary_mtime = summary_path.stat().st_mtime
        min_margin_seconds = 5
        return (results_mtime - summary_mtime) > min_margin_seconds
    except OSError:
        return False


class RunListItem(BaseModel):
    run_id: str
    run_name: str | None = None
    archived: bool
    in_progress: bool
    eval_kind: str

    provider: str | None = None
    model: str | None = None
    base_model: str | None = None
    dataset: str | None = None
    dataset_version: str | None = None
    dataset_version_bg: str | None = None
    dataset_version_fg: str | None = None
    dataset_version_border: str | None = None

    total_examples: int | None = None
    exact_match_rate: float | None = None
    prefix_match_rate: float | None = None
    syntax_pass_rate: float | None = None
    lint_pass_rate: float | None = None
    total_repo_average: float | None = None
    in_domain_repo_average: float | None = None
    out_of_domain_repo_average: float | None = None
    total_repo_prefix_average: float | None = None
    in_domain_repo_prefix_average: float | None = None
    out_of_domain_repo_prefix_average: float | None = None
    lint_in_domain_pass_rate: float | None = None
    lint_out_of_domain_pass_rate: float | None = None
    repo_scores: list[dict[str, Any]] = Field(default_factory=list)
    repo_rate_map: dict[str, float] = Field(default_factory=dict)
    repo_prefix_rate_map: dict[str, float] = Field(default_factory=dict)
    repo_domain_map: dict[str, str] = Field(default_factory=dict)
    repo_example_count_map: dict[str, int] = Field(default_factory=dict)
    train_repos: list[str] = Field(default_factory=list)

    duration_seconds: float | None = None
    train_duration_seconds: float | None = None
    timestamp: str | None = None
    notes: str = ""
    cv_parent_run_id: str | None = None
    cv_subrun_index: int | None = None
    cv_subrun_count: int | None = None
    cv_subrun_run_ids: list[str] = Field(default_factory=list)
    cv_holdout_repos: list[str] = Field(default_factory=list)
    is_cv_parent: bool = False
    is_cv_subrun: bool = False


class RunsResponse(BaseModel):
    items: list[RunListItem]
    total: int


class SaveNotesRequest(BaseModel):
    notes: str = ""


class BulkArchiveRequest(BaseModel):
    run_ids: list[str] = Field(default_factory=list)
    reason: str = ""


class BulkDeleteRequest(BaseModel):
    run_ids: list[str] = Field(default_factory=list)


class CompareRequest(BaseModel):
    run_ids: list[str] = Field(default_factory=list)


class CompareResponse(BaseModel):
    left: RunListItem
    right: RunListItem
    deltas: dict[str, float | None]


class ResultListItem(BaseModel):
    id: str
    file_path: str | None = None
    repo: str | None = None
    category: str | None = None
    dataset_source: str | None = None
    exact_match: bool = False
    prefix_match: bool = False
    predicted_syntax_passed: bool | None = None
    original_syntax_passed: bool | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_preview: str = ""
    target_preview: str = ""
    outcome: str = ""
    error_type: str | None = None
    error_message: str | None = None


class ResultListResponse(BaseModel):
    items: list[ResultListItem]
    total: int
    page: int
    page_size: int
    sample_types: list[str]
    sources: list[str]


class ResultDetailResponse(BaseModel):
    run_id: str
    row: dict[str, Any]
    outcome_match: str
    outcome_syntax: str


VALID_DETAIL_MODES = {"match", "syntax"}
VALID_OUTCOME_FILTER_MATCH = {"all", "match", "prefix", "mismatch"}
VALID_OUTCOME_FILTER_SYNTAX = {"all", "pass", "fail", "inconclusive"}
VALID_RESPONSE_FILTER = {"all", "non_empty", "empty"}


def _response_text_for_result_row(row: dict[str, Any]) -> str:
    value = row.get("raw_response")
    if isinstance(value, str):
        return value
    value = row.get("predicted")
    if isinstance(value, str):
        return value
    return ""


def _resolve_cors_origins() -> list[str]:
    configured = str(os.getenv("FIM_CORS_ORIGINS", "")).strip()
    candidates = configured.split(",") if configured else list(DEFAULT_CORS_ORIGINS)
    out: list[str] = []
    seen: set[str] = set()
    for origin in candidates:
        text = str(origin or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _detail_outcome_match(row: dict[str, Any]) -> str:
    if bool(row.get("exact_match", False)):
        return "match"
    if bool(row.get("prefix_match", False)):
        return "prefix"
    return "mismatch"


def _detail_outcome_syntax(row: dict[str, Any]) -> str:
    pred = row.get("predicted_syntax_passed")
    if isinstance(pred, bool):
        return "pass" if pred else "fail"
    return "inconclusive"


def _detail_outcome(row: dict[str, Any], mode: str) -> str:
    if mode == "syntax":
        return _detail_outcome_syntax(row)
    return _detail_outcome_match(row)


def _clip_text(value: str, limit: int = 240) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


def _iter_results_file(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _cv_parent_heldout_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Merge a CV parent's fold results into the held-out view: each repo's rows
    come from the first fold that scored it out-of-domain, so no row is
    in-domain — the same merge the published comparison numbers use."""
    rows: list[dict[str, Any]] = []
    claimed: set[str] = set()
    for subrun_id in _normalize_string_list(summary.get("cv_subrun_run_ids") or []):
        try:
            subrun_dir, _ = _find_run_dir(subrun_id)
        except HTTPException:
            continue
        results_path = subrun_dir / "results.jsonl"
        if not results_path.exists():
            continue
        sub_summary = _safe_json_load(subrun_dir / "summary.json")
        ood_repos = {
            str(entry.get("repo") or "").strip()
            for entry in _normalize_repo_scores(sub_summary.get("repo_scores") or [])
            if str(entry.get("domain") or "").strip().lower() == "out_of_domain"
        }
        take = ood_repos - claimed
        if not take:
            continue
        claimed |= take
        for row in _iter_results_file(results_path):
            if str(row.get("repo") or "").strip() in take:
                rows.append(row)
    return rows


def _iter_run_results(run_dir: Path):
    """Rows for a run's detail view: its own results.jsonl, or — for a CV parent
    without one — the held-out merge of its folds."""
    path = run_dir / "results.jsonl"
    if path.exists():
        return _iter_results_file(path)
    summary = _safe_json_load(run_dir / "summary.json")
    if _normalize_string_list(summary.get("cv_subrun_run_ids") or []):
        rows = _cv_parent_heldout_rows(summary)
        if rows:
            return rows
        raise HTTPException(
            status_code=404,
            detail="CV parent has no fold results to serve (fold runs missing or empty)",
        )
    raise HTTPException(status_code=404, detail="results.jsonl not found for run")


def _result_matches_filters(
    row: dict[str, Any],
    *,
    mode: str,
    outcome_filter: str,
    sample_type: str,
    source: str,
    response_filter: str,
    query: str,
) -> bool:
    outcome = _detail_outcome(row, mode)
    if outcome_filter != "all" and outcome != outcome_filter:
        return False

    if sample_type != "all":
        if str(row.get("category") or "").strip() != sample_type:
            return False

    if source != "all":
        if str(row.get("dataset_source") or "").strip() != source:
            return False

    response_text = _response_text_for_result_row(row).strip()
    if response_filter == "non_empty" and not response_text:
        return False
    if response_filter == "empty" and response_text:
        return False

    if query:
        hay = " ".join(
            [
                str(row.get("id") or ""),
                str(row.get("file_path") or ""),
                str(row.get("repo") or ""),
                str(row.get("category") or ""),
                str(row.get("dataset_source") or ""),
                response_text,
                str(row.get("target") or ""),
                str(row.get("error") or ""),
            ]
        ).lower()
        if query not in hay:
            return False

    return True


def _row_to_result_list_item(row: dict[str, Any], mode: str) -> ResultListItem:
    response_text = _response_text_for_result_row(row)
    target = row.get("target")
    target_text = target if isinstance(target, str) else ""

    error = row.get("error")
    error_type: str | None = None
    error_message: str | None = None
    if isinstance(error, dict):
        error_type = str(error.get("type") or "") or None
        error_message = str(error.get("message") or "") or None
    elif isinstance(error, str) and error.strip():
        error_message = error.strip()

    return ResultListItem(
        id=str(row.get("id") or ""),
        file_path=str(row.get("file_path") or "") or None,
        repo=str(row.get("repo") or "") or None,
        category=str(row.get("category") or "") or None,
        dataset_source=str(row.get("dataset_source") or "") or None,
        exact_match=bool(row.get("exact_match", False)),
        prefix_match=bool(row.get("prefix_match", False)),
        predicted_syntax_passed=(
            row.get("predicted_syntax_passed") if isinstance(row.get("predicted_syntax_passed"), bool) else None
        ),
        original_syntax_passed=(
            row.get("original_syntax_passed") if isinstance(row.get("original_syntax_passed"), bool) else None
        ),
        input_tokens=_to_int_or_none(row.get("input_tokens"), field="results.input_tokens"),
        output_tokens=_to_int_or_none(row.get("output_tokens"), field="results.output_tokens"),
        response_preview=_clip_text(response_text),
        target_preview=_clip_text(target_text),
        outcome=_detail_outcome(row, mode),
        error_type=error_type,
        error_message=error_message,
    )


def _run_to_item(run_id: str, run_dir: Path, archived: bool) -> RunListItem:
    summary = _safe_json_load(run_dir / "summary.json")
    notes = _safe_text_read(run_dir / "notes.txt")
    config = summary.get("config")
    if not isinstance(config, dict):
        config = {}

    dataset = _infer_dataset_label(summary, config)
    cv_parent_run_id = str(summary.get("cv_parent_run_id") or "").strip() or None
    cv_subrun_run_ids = _normalize_string_list(summary.get("cv_subrun_run_ids") or [])
    cv_subrun_count = _to_int_or_none(summary.get("cv_subrun_count"), field="summary.cv_subrun_count")
    is_cv_subrun = bool(cv_parent_run_id)
    is_cv_parent = bool(cv_subrun_run_ids) or (bool(cv_subrun_count and cv_subrun_count > 1) and not is_cv_subrun)
    if is_cv_subrun or is_cv_parent:
        eval_kind = "CV"
    else:
        eval_kind = _infer_eval_kind(dataset, run_id)
    provider = str(config.get("provider", "")).strip() or None

    model = summary.get("model")
    base_model = _infer_model_kind(summary, config)
    train_repos = _normalize_string_list(summary.get("cv_train_repos") or config.get("train_repos") or [])
    repo_scores = _normalize_repo_scores(summary.get("repo_scores") or [])
    repo_rate_map, repo_domain_map, repo_example_count_map = _build_repo_maps(
        repo_scores,
        train_repos=train_repos,
    )
    (
        repo_prefix_rate_map,
        in_domain_repo_prefix_average,
        out_of_domain_repo_prefix_average,
        total_repo_prefix_average,
    ) = _build_repo_prefix_maps(
        repo_scores,
        train_repos=train_repos,
    )

    syntax_pass_rate = _to_float_or_none(
        summary.get("predicted_syntax_pass_rate"),
        field="summary.predicted_syntax_pass_rate",
    )
    lint_pass_rate = _to_float_or_none(summary.get("lint_pass_rate"), field="summary.lint_pass_rate")
    if lint_pass_rate is None and syntax_pass_rate is not None:
        # Keep list behavior usable when lint metrics are not available.
        lint_pass_rate = syntax_pass_rate
    lint_in_domain_pass_rate = _to_float_or_none(
        summary.get("lint_in_domain_pass_rate"),
        field="summary.lint_in_domain_pass_rate",
    )
    lint_out_of_domain_pass_rate = _to_float_or_none(
        summary.get("lint_out_of_domain_pass_rate"),
        field="summary.lint_out_of_domain_pass_rate",
    )
    if eval_kind == "ID" and lint_in_domain_pass_rate is None:
        lint_in_domain_pass_rate = lint_pass_rate
    if eval_kind == "OOD" and lint_out_of_domain_pass_rate is None:
        lint_out_of_domain_pass_rate = lint_pass_rate

    dataset_version = _infer_dataset_version(summary, config)
    dataset_version_bg: str | None = None
    dataset_version_fg: str | None = None
    dataset_version_border: str | None = None
    if dataset_version:
        dataset_version_bg, dataset_version_fg, dataset_version_border = _dataset_version_badge_style(dataset_version)

    cv_holdout_repos = _normalize_string_list(summary.get("cv_holdout_repos") or [])
    cv_subrun_index = _to_int_or_none(summary.get("cv_subrun_index"), field="summary.cv_subrun_index")
    run_name = str(summary.get("run_name") or config.get("eval_name") or "").strip() or None
    in_domain_repo_average = _to_float_or_none(
        summary.get("in_domain_repo_average"),
        field="summary.in_domain_repo_average",
    )
    out_of_domain_repo_average = _to_float_or_none(
        summary.get("out_of_domain_repo_average"),
        field="summary.out_of_domain_repo_average",
    )
    total_repo_average = _to_float_or_none(summary.get("total_repo_average"), field="summary.total_repo_average")
    summary_total_repo_prefix_average = _to_float_or_none(
        summary.get("total_repo_prefix_average"),
        field="summary.total_repo_prefix_average",
    )
    summary_in_domain_repo_prefix_average = _to_float_or_none(
        summary.get("in_domain_repo_prefix_average"),
        field="summary.in_domain_repo_prefix_average",
    )
    summary_out_of_domain_repo_prefix_average = _to_float_or_none(
        summary.get("out_of_domain_repo_prefix_average"),
        field="summary.out_of_domain_repo_prefix_average",
    )
    if summary_total_repo_prefix_average is not None:
        total_repo_prefix_average = summary_total_repo_prefix_average
    if summary_in_domain_repo_prefix_average is not None:
        in_domain_repo_prefix_average = summary_in_domain_repo_prefix_average
    if summary_out_of_domain_repo_prefix_average is not None:
        out_of_domain_repo_prefix_average = summary_out_of_domain_repo_prefix_average

    if (
        (not repo_prefix_rate_map)
        or in_domain_repo_prefix_average is None
        or out_of_domain_repo_prefix_average is None
        or total_repo_prefix_average is None
    ):
        results_path = run_dir / "results.jsonl"
        (
            computed_repo_prefix_map,
            computed_id_prefix_avg,
            computed_ood_prefix_avg,
            computed_total_prefix_avg,
        ) = _compute_repo_prefix_stats_from_results(results_path, train_repos)
        if not repo_prefix_rate_map and computed_repo_prefix_map:
            repo_prefix_rate_map = computed_repo_prefix_map
        if in_domain_repo_prefix_average is None:
            in_domain_repo_prefix_average = computed_id_prefix_avg
        if out_of_domain_repo_prefix_average is None:
            out_of_domain_repo_prefix_average = computed_ood_prefix_avg
        if total_repo_prefix_average is None:
            total_repo_prefix_average = computed_total_prefix_avg

    train_duration_seconds = _to_float_or_none(
        summary.get("train_duration_seconds"),
        field="summary.train_duration_seconds",
    )
    # In a single-kind run every repo is that kind, so the unweighted average is
    # the mean over all per-repo rates. Only fall back to the pooled (weighted)
    # rate when there are no per-repo scores — then the run is effectively
    # single-repo and the two coincide.
    unweighted_exact_mean = _mean_or_none(repo_rate_map.values())
    unweighted_prefix_mean = _mean_or_none(repo_prefix_rate_map.values())
    if eval_kind == "ID" and in_domain_repo_average is None:
        in_domain_repo_average = (
            unweighted_exact_mean
            if unweighted_exact_mean is not None
            else _to_float_or_none(summary.get("exact_match_rate"), field="summary.exact_match_rate")
        )
    if eval_kind == "OOD" and out_of_domain_repo_average is None:
        out_of_domain_repo_average = (
            unweighted_exact_mean
            if unweighted_exact_mean is not None
            else _to_float_or_none(summary.get("exact_match_rate"), field="summary.exact_match_rate")
        )
    if total_repo_average is None:
        total_repo_average = unweighted_exact_mean
    weighted_prefix_match_rate = _to_float_or_none(
        summary.get("prefix_match_rate"),
        field="summary.prefix_match_rate",
    )
    if eval_kind == "ID" and in_domain_repo_prefix_average is None:
        in_domain_repo_prefix_average = (
            unweighted_prefix_mean if unweighted_prefix_mean is not None else weighted_prefix_match_rate
        )
    if eval_kind == "OOD" and out_of_domain_repo_prefix_average is None:
        out_of_domain_repo_prefix_average = (
            unweighted_prefix_mean if unweighted_prefix_mean is not None else weighted_prefix_match_rate
        )
    if total_repo_prefix_average is None:
        if eval_kind == "ID":
            total_repo_prefix_average = in_domain_repo_prefix_average
        elif eval_kind == "OOD":
            total_repo_prefix_average = out_of_domain_repo_prefix_average

    return RunListItem(
        run_id=run_id,
        run_name=run_name,
        archived=archived,
        in_progress=_is_run_in_progress(run_id, run_dir, archived),
        eval_kind=eval_kind,
        provider=provider,
        model=model,
        base_model=base_model,
        dataset=dataset,
        dataset_version=dataset_version,
        dataset_version_bg=dataset_version_bg,
        dataset_version_fg=dataset_version_fg,
        dataset_version_border=dataset_version_border,
        total_examples=_to_int_or_none(summary.get("total_examples"), field="summary.total_examples"),
        exact_match_rate=_to_float_or_none(summary.get("exact_match_rate"), field="summary.exact_match_rate"),
        prefix_match_rate=_to_float_or_none(summary.get("prefix_match_rate"), field="summary.prefix_match_rate"),
        syntax_pass_rate=syntax_pass_rate,
        lint_pass_rate=lint_pass_rate,
        total_repo_average=total_repo_average,
        in_domain_repo_average=in_domain_repo_average,
        out_of_domain_repo_average=out_of_domain_repo_average,
        total_repo_prefix_average=total_repo_prefix_average,
        in_domain_repo_prefix_average=in_domain_repo_prefix_average,
        out_of_domain_repo_prefix_average=out_of_domain_repo_prefix_average,
        lint_in_domain_pass_rate=lint_in_domain_pass_rate,
        lint_out_of_domain_pass_rate=lint_out_of_domain_pass_rate,
        repo_scores=repo_scores,
        repo_rate_map=repo_rate_map,
        repo_prefix_rate_map=repo_prefix_rate_map,
        repo_domain_map=repo_domain_map,
        repo_example_count_map=repo_example_count_map,
        train_repos=train_repos,
        duration_seconds=_to_float_or_none(summary.get("duration_seconds"), field="summary.duration_seconds"),
        train_duration_seconds=train_duration_seconds,
        timestamp=summary.get("timestamp"),
        notes=notes,
        cv_parent_run_id=cv_parent_run_id,
        cv_subrun_index=cv_subrun_index,
        cv_subrun_count=cv_subrun_count,
        cv_subrun_run_ids=cv_subrun_run_ids,
        cv_holdout_repos=cv_holdout_repos,
        is_cv_parent=is_cv_parent,
        is_cv_subrun=is_cv_subrun,
    )


def _valid_number(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:  # NaN check
        return None
    return parsed


def _backfill_cv_parent_phase_totals(items: list[RunListItem]) -> list[RunListItem]:
    item_by_id = {item.run_id: item for item in items}
    children_by_parent: dict[str, list[RunListItem]] = {}

    for item in items:
        parent_id = str(item.cv_parent_run_id or "").strip()
        if not parent_id:
            continue
        if parent_id not in item_by_id:
            continue
        children_by_parent.setdefault(parent_id, []).append(item)

    for parent_id, children in children_by_parent.items():
        parent = item_by_id.get(parent_id)
        if parent is None:
            continue
        if not (parent.is_cv_parent or parent.eval_kind == "CV"):
            continue

        eval_durations = [value for child in children if (value := _valid_number(child.duration_seconds)) is not None]
        train_durations = [
            value for child in children if (value := _valid_number(child.train_duration_seconds)) is not None
        ]

        if eval_durations:
            parent.duration_seconds = float(sum(eval_durations))
        if train_durations:
            parent.train_duration_seconds = float(sum(train_durations))

    return items


def _write_archive_meta(run_dir: Path, reason: str) -> None:
    payload = {
        "reason": reason.strip(),
        "archived_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with (run_dir / "archive_meta.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        # Non-fatal metadata write.
        pass


APP_STARTED_AT_UTC = datetime.now(timezone.utc).isoformat()

app = FastAPI(title="DCA FIM Frontend API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_resolve_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "pid": os.getpid(),
        "started_at_utc": APP_STARTED_AT_UTC,
    }


@app.post("/api/system/backend/restart")
def restart_backend(request: Request) -> dict[str, Any]:
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost"}:
        raise HTTPException(status_code=403, detail="Backend restart is only allowed from localhost")

    if not RESTART_BACKEND_SCRIPT.exists() or not RESTART_BACKEND_SCRIPT.is_file():
        raise HTTPException(status_code=500, detail=f"Restart script not found: {RESTART_BACKEND_SCRIPT}")

    RESTART_BACKEND_LOG.parent.mkdir(parents=True, exist_ok=True)
    restart_cmd = f"sleep 0.6; bash {shlex.quote(str(RESTART_BACKEND_SCRIPT))}"
    with RESTART_BACKEND_LOG.open("a", encoding="utf-8") as log_fp:
        subprocess.Popen(
            ["bash", "-lc", restart_cmd],
            cwd=str(ROOT_DIR),
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )

    return {
        "status": "restarting",
        "api_port": str(os.getenv("API_PORT", "8010")),
        "script": str(RESTART_BACKEND_SCRIPT),
    }


@app.get("/api/fim/runs", response_model=RunsResponse)
def list_runs(
    include_archived: bool = False,
    eval_filter: str = "All",
    q: str = "",
) -> RunsResponse:
    if eval_filter not in VALID_EVAL_FILTERS:
        raise HTTPException(status_code=400, detail=f"Invalid eval_filter: {eval_filter}")

    all_items: list[RunListItem] = []
    q_norm = q.strip().lower()

    for run_id, run_dir, archived in _iter_run_dirs(include_archived=include_archived):
        item = _run_to_item(run_id=run_id, run_dir=run_dir, archived=archived)
        all_items.append(item)

    _backfill_cv_parent_phase_totals(all_items)

    items: list[RunListItem] = []
    for item in all_items:
        if eval_filter != "All" and item.eval_kind != eval_filter:
            continue

        if q_norm:
            hay = " ".join(
                [
                    item.run_id,
                    item.run_name or "",
                    item.model or "",
                    item.base_model or "",
                    item.dataset or "",
                    item.dataset_version or "",
                    item.notes or "",
                ]
            ).lower()
            if q_norm not in hay:
                continue

        items.append(item)

    return RunsResponse(items=items, total=len(items))


@app.get("/api/fim/runs/{run_id}/results", response_model=ResultListResponse)
def list_run_results(
    run_id: str,
    mode: str = "match",
    outcome_filter: str = "all",
    sample_type: str = "all",
    source: str = "all",
    response_filter: str = "all",
    q: str = "",
    page: int = 1,
    page_size: int = 25,
) -> ResultListResponse:
    run_id = _normalize_run_id(run_id)
    if mode not in VALID_DETAIL_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")
    if mode == "match":
        if outcome_filter not in VALID_OUTCOME_FILTER_MATCH:
            raise HTTPException(status_code=400, detail=f"Invalid outcome_filter for match: {outcome_filter}")
    else:
        if outcome_filter not in VALID_OUTCOME_FILTER_SYNTAX:
            raise HTTPException(status_code=400, detail=f"Invalid outcome_filter for syntax: {outcome_filter}")
    if response_filter not in VALID_RESPONSE_FILTER:
        raise HTTPException(status_code=400, detail=f"Invalid response_filter: {response_filter}")
    if page < 1:
        raise HTTPException(status_code=400, detail="page must be >= 1")
    if page_size < 1 or page_size > 200:
        raise HTTPException(status_code=400, detail="page_size must be between 1 and 200")

    run_dir, _ = _find_run_dir(run_id)

    query = q.strip().lower()
    sample_types: set[str] = set()
    sources: set[str] = set()
    filtered_rows: list[dict[str, Any]] = []

    for row in _iter_run_results(run_dir):
        category_val = str(row.get("category") or "").strip()
        if category_val:
            sample_types.add(category_val)
        source_val = str(row.get("dataset_source") or "").strip()
        if source_val:
            sources.add(source_val)
        if _result_matches_filters(
            row,
            mode=mode,
            outcome_filter=outcome_filter,
            sample_type=sample_type,
            source=source,
            response_filter=response_filter,
            query=query,
        ):
            filtered_rows.append(row)

    total = len(filtered_rows)
    start = (page - 1) * page_size
    end = start + page_size
    page_rows = filtered_rows[start:end]
    items = [_row_to_result_list_item(row, mode=mode) for row in page_rows]

    return ResultListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        sample_types=sorted(sample_types),
        sources=sorted(sources),
    )


@app.get("/api/fim/runs/{run_id}/results/{result_id}", response_model=ResultDetailResponse)
def get_run_result_detail(run_id: str, result_id: str) -> ResultDetailResponse:
    run_id = _normalize_run_id(run_id)
    result_id = str(result_id or "").strip()
    if not result_id:
        raise HTTPException(status_code=400, detail="result_id is required")

    run_dir, _ = _find_run_dir(run_id)

    for row in _iter_run_results(run_dir):
        if str(row.get("id") or "") == result_id:
            return ResultDetailResponse(
                run_id=run_id,
                row=row,
                outcome_match=_detail_outcome_match(row),
                outcome_syntax=_detail_outcome_syntax(row),
            )

    raise HTTPException(status_code=404, detail=f"Result not found: {result_id}")


@app.put("/api/fim/runs/{run_id}/notes", response_model=RunListItem)
def save_notes(run_id: str, body: SaveNotesRequest) -> RunListItem:
    run_id = _normalize_run_id(run_id)
    run_dir, archived = _find_run_dir(run_id)
    notes_path = run_dir / "notes.txt"
    notes_path.write_text(body.notes or "", encoding="utf-8")
    return _run_to_item(run_id=run_id, run_dir=run_dir, archived=archived)


@app.post("/api/fim/runs/archive")
def archive_runs(body: BulkArchiveRequest) -> dict[str, Any]:
    run_ids = [_normalize_run_id(rid) for rid in body.run_ids if str(rid).strip()]
    if not run_ids:
        raise HTTPException(status_code=400, detail="No run_ids provided")

    _ensure_dirs()
    archived: list[str] = []
    skipped: list[str] = []

    for run_id in run_ids:
        active_dir = FIM_LOGS_DIR / run_id
        if not active_dir.exists() or not active_dir.is_dir():
            if (FIM_ARCHIVE_DIR / run_id).exists():
                skipped.append(run_id)
                continue
            raise HTTPException(status_code=404, detail=f"Run not found in active logs: {run_id}")

        target = FIM_ARCHIVE_DIR / run_id
        if target.exists():
            idx = 1
            while True:
                candidate = FIM_ARCHIVE_DIR / f"{run_id}__archived_{idx}"
                if not candidate.exists():
                    target = candidate
                    break
                idx += 1

        shutil.move(str(active_dir), str(target))
        _write_archive_meta(target, reason=body.reason)
        archived.append(target.name)

    return {"archived": archived, "skipped": skipped, "count": len(archived)}


@app.post("/api/fim/runs/delete")
def delete_runs(body: BulkDeleteRequest) -> dict[str, Any]:
    run_ids = [_normalize_run_id(rid) for rid in body.run_ids if str(rid).strip()]
    if not run_ids:
        raise HTTPException(status_code=400, detail="No run_ids provided")

    deleted: list[str] = []
    missing: list[str] = []

    for run_id in run_ids:
        found = False
        for base in (FIM_LOGS_DIR, FIM_ARCHIVE_DIR):
            run_dir = base / run_id
            if run_dir.exists() and run_dir.is_dir():
                shutil.rmtree(run_dir)
                found = True
        if found:
            deleted.append(run_id)
        else:
            missing.append(run_id)

    return {"deleted": deleted, "missing": missing, "count": len(deleted)}


@app.post("/api/fim/runs/compare", response_model=CompareResponse)
def compare_runs(body: CompareRequest) -> CompareResponse:
    if len(body.run_ids) != 2:
        raise HTTPException(status_code=400, detail="compare requires exactly 2 run_ids")

    left_id, right_id = (_normalize_run_id(body.run_ids[0]), _normalize_run_id(body.run_ids[1]))
    left_dir, left_archived = _find_run_dir(left_id)
    right_dir, right_archived = _find_run_dir(right_id)

    left = _run_to_item(left_id, left_dir, left_archived)
    right = _run_to_item(right_id, right_dir, right_archived)

    def delta(a: float | None, b: float | None) -> float | None:
        if a is None or b is None:
            return None
        return b - a

    deltas = {
        "exact_match_rate": delta(left.exact_match_rate, right.exact_match_rate),
        "prefix_match_rate": delta(left.prefix_match_rate, right.prefix_match_rate),
        "syntax_pass_rate": delta(left.syntax_pass_rate, right.syntax_pass_rate),
        "lint_pass_rate": delta(left.lint_pass_rate, right.lint_pass_rate),
        "duration_seconds": delta(left.duration_seconds, right.duration_seconds),
    }

    return CompareResponse(left=left, right=right, deltas=deltas)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("daml_fim_benchmark.server.app:app", host="127.0.0.1", port=8010, reload=False)
