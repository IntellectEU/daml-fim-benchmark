"""
Helpers to compute DAML syntax-check metrics for FIM evaluation runs.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections.abc import Callable
from typing import Any

SKIP_REASON_EXPLANATIONS: dict[str, str] = {
    "missing_file_path": "Result row has no file_path.",
    "missing_file": "Referenced file is not present in current workspace.",
    "file_read_error": "File could not be read (IO/encoding issue).",
    "span_resolution_failed": "Could not safely map target span to current file text.",
    "syntax_checker_inconclusive": (
        "Lint returned non-parser errors (e.g. typecheck/dependency/environment), "
        "so this row is excluded from syntax scoring."
    ),
    "original_sanity_failed": "Original file failed strict parser syntax sanity check.",
}

# Process-wide locks so concurrent backfill jobs cannot mutate/lint the same file
# or run overlapping lint/build work for the same DAML project root.
_GLOBAL_FILE_LOCKS: dict[str, threading.Lock] = {}
_GLOBAL_PROJECT_LOCKS: dict[str, threading.RLock] = {}
_GLOBAL_FILE_LOCKS_GUARD = threading.Lock()
_GLOBAL_PROJECT_LOCKS_GUARD = threading.Lock()
_ACTIVE_FILE_MUTATIONS: dict[str, tuple[Path, str]] = {}
_ACTIVE_FILE_MUTATIONS_GUARD = threading.Lock()
_RESTORE_HOOKS_GUARD = threading.Lock()
_RESTORE_HOOKS_INSTALLED = False
_PREVIOUS_RESTORE_SIGNAL_HANDLERS: dict[int, Any] = {}


def _project_root() -> Path:
    """Return repository root regardless of the current working directory."""
    return Path(__file__).resolve().parent.parent


def resolve_file_path_for_syntax_check(file_path_raw: str | Path) -> Path:
    """
    Resolve a dataset row's file_path to an on-disk source file for the syntax check.

    Rows store repo-relative paths (``<repo>/...``); the source repos are checked
    out under ``sources/<repo>/...`` (see scripts/checkout_source_repos.sh). We try
    the path as-is and under ``sources/``, relative to both the current working
    directory and the package root, so the check works from any invocation dir.
    """
    file_path = Path(str(file_path_raw))
    if file_path.is_absolute():
        return file_path

    for base in (Path.cwd(), _project_root()):
        for candidate in ((base / file_path).resolve(), (base / "sources" / file_path).resolve()):
            if candidate.exists():
                return candidate

    # Deterministic fallback for error reporting/caching: where it would be checked out.
    return (Path.cwd() / "sources" / file_path).resolve()


def find_daml_yaml_root(start_path: str | Path) -> Path | None:
    """Find the nearest parent directory that contains daml.yaml."""
    current = Path(start_path).resolve()
    if current.is_file():
        current = current.parent

    while True:
        if (current / "daml.yaml").exists():
            return current
        if current.parent == current:
            return None
        current = current.parent


def find_daml_build_root(start_path: str | Path) -> Path | None:
    """
    Find a DAML build root:
    - nearest parent containing multi-package.yaml, else
    - nearest parent containing daml.yaml.
    """
    current = Path(start_path).resolve()
    if current.is_file():
        current = current.parent

    search_dir = current
    while True:
        if (search_dir / "multi-package.yaml").exists():
            return search_dir
        if search_dir.parent == search_dir:
            break
        search_dir = search_dir.parent

    return find_daml_yaml_root(start_path)


def _find_source_repo_root(path: Path) -> Path | None:
    """
    Find the ``sources/<repo>`` root for a file path if present (the per-repo
    checkout dir), used to locate a repo-level .envrc.
    """
    current = path.resolve()
    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        if candidate.parent.name == "sources":
            return candidate
    return None


def _track_active_file_mutation(file_path: Path, original_content: str) -> None:
    key = str(file_path.resolve())
    with _ACTIVE_FILE_MUTATIONS_GUARD:
        _ACTIVE_FILE_MUTATIONS[key] = (file_path.resolve(), original_content)


def _restore_tracked_file_mutation(file_path: Path, original_content: str) -> None:
    key = str(file_path.resolve())
    resolved = file_path.resolve()
    resolved.write_text(original_content, encoding="utf-8")
    with _ACTIVE_FILE_MUTATIONS_GUARD:
        current = _ACTIVE_FILE_MUTATIONS.get(key)
        if current is not None and current[1] == original_content:
            _ACTIVE_FILE_MUTATIONS.pop(key, None)


def _restore_all_active_file_mutations() -> None:
    while True:
        with _ACTIVE_FILE_MUTATIONS_GUARD:
            pending = list(_ACTIVE_FILE_MUTATIONS.items())
        if not pending:
            return

        restored_any = False
        for key, (path, original_content) in pending:
            try:
                path.write_text(original_content, encoding="utf-8")
            except OSError:
                continue
            with _ACTIVE_FILE_MUTATIONS_GUARD:
                current = _ACTIVE_FILE_MUTATIONS.get(key)
                if current is not None and current[1] == original_content:
                    _ACTIVE_FILE_MUTATIONS.pop(key, None)
            restored_any = True

        if not restored_any:
            return


def _restore_signal_handler(signum: int, frame: Any) -> None:
    _restore_all_active_file_mutations()

    previous = _PREVIOUS_RESTORE_SIGNAL_HANDLERS.get(signum, signal.SIG_DFL)
    if callable(previous):
        previous(signum, frame)
        return
    if previous == signal.SIG_IGN:
        return
    if signum == signal.SIGINT:
        signal.default_int_handler(signum, frame)
    raise SystemExit(128 + signum)


def _install_restore_hooks_once() -> None:
    global _RESTORE_HOOKS_INSTALLED
    with _RESTORE_HOOKS_GUARD:
        if _RESTORE_HOOKS_INSTALLED:
            return
        _RESTORE_HOOKS_INSTALLED = True

    atexit.register(_restore_all_active_file_mutations)
    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            _PREVIOUS_RESTORE_SIGNAL_HANDLERS[sig] = signal.getsignal(sig)
            signal.signal(sig, _restore_signal_handler)
        except (ValueError, OSError, RuntimeError):
            # Signal handler registration can fail outside the main thread.
            continue


_install_restore_hooks_once()


class DamlSyntaxChecker:
    """
    Runs `daml damlc lint` with project-aware setup and caching.
    """

    def __init__(self):
        self._command_prefix_cache: dict[str, list[str]] = {}
        self._direnv_allowed: set[str] = set()
        self._target_flag_cache: dict[str, str | None] = {}
        self._project_build_cache: dict[str, tuple[bool, str | None]] = {}
        self._cache_guard = threading.Lock()

    def _get_file_lock(self, file_path: Path) -> threading.Lock:
        key = str(file_path.resolve())
        with _GLOBAL_FILE_LOCKS_GUARD:
            lock = _GLOBAL_FILE_LOCKS.get(key)
            if lock is None:
                lock = threading.Lock()
                _GLOBAL_FILE_LOCKS[key] = lock
            return lock

    def _get_project_lock(self, project_root: Path) -> threading.RLock:
        key = str(project_root.resolve())
        with _GLOBAL_PROJECT_LOCKS_GUARD:
            lock = _GLOBAL_PROJECT_LOCKS.get(key)
            if lock is None:
                # Reentrant: lint wrappers hold project lock while internals may
                # call _ensure_project_prepared(), which also takes this lock.
                lock = threading.RLock()
                _GLOBAL_PROJECT_LOCKS[key] = lock
            return lock

    def _project_lock_for_file(self, file_path: Path) -> threading.RLock | None:
        project_root = find_daml_build_root(file_path) or find_daml_yaml_root(file_path)
        if project_root is None:
            return None
        return self._get_project_lock(project_root)

    def _command_prefix_for(self, file_path: Path) -> list[str]:
        source_root = _find_source_repo_root(file_path)
        if source_root is None:
            return []

        key = str(source_root)
        cached = self._command_prefix_cache.get(key)
        if cached is not None:
            return cached

        prefix: list[str] = []
        envrc = source_root / ".envrc"
        if envrc.exists() and shutil.which("direnv"):
            if key not in self._direnv_allowed:
                # Best-effort; if this fails we continue without raising.
                subprocess.run(
                    ["direnv", "allow"],
                    cwd=str(source_root),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self._direnv_allowed.add(key)
            prefix = ["direnv", "exec", "."]

        self._command_prefix_cache[key] = prefix
        return prefix

    def _target_flag_for(self, daml_root: Path) -> str | None:
        key = str(daml_root.resolve())
        with self._cache_guard:
            if key in self._target_flag_cache:
                return self._target_flag_cache[key]

        target_flag: str | None = None
        daml_yaml = daml_root / "daml.yaml"
        if daml_yaml.exists():
            try:
                text = daml_yaml.read_text(encoding="utf-8")
            except OSError:
                text = ""

            # Example supported values:
            # - --target=1.15
            # - --target 1.15
            match = re.search(r"--target(?:=|\s+)([0-9A-Za-z_.-]+)", text)
            if match:
                target_flag = f"--target={match.group(1)}"

        with self._cache_guard:
            self._target_flag_cache[key] = target_flag
        return target_flag

    @staticmethod
    def _extract_parser_errors(stderr: str) -> list[str]:
        """
        Extract only parser diagnostics (syntax errors) from damlc lint stderr.
        """
        blocks = re.findall(r"File:\s*.*?(?=\nFile:\s*|\Z)", stderr, flags=re.DOTALL)
        parser_blocks: list[str] = []
        for block in blocks:
            source_match = re.search(r"^\s*Source:\s*(.+)$", block, flags=re.MULTILINE)
            severity_match = re.search(r"^\s*Severity:\s*(\S+)$", block, flags=re.MULTILINE)
            source = source_match.group(1).strip() if source_match else ""
            severity = severity_match.group(1).strip() if severity_match else ""
            if source == "parser" and severity == "DsError":
                parser_blocks.append(block.strip())
        return parser_blocks

    @staticmethod
    def _extract_diagnostics(stderr: str) -> list[dict[str, str]]:
        blocks = re.findall(r"File:\s*.*?(?=\nFile:\s*|\Z)", stderr, flags=re.DOTALL)
        diagnostics: list[dict[str, str]] = []
        for block in blocks:
            source_match = re.search(r"^\s*Source:\s*(.+)$", block, flags=re.MULTILINE)
            severity_match = re.search(r"^\s*Severity:\s*(\S+)$", block, flags=re.MULTILINE)
            source = source_match.group(1).strip() if source_match else ""
            severity = severity_match.group(1).strip() if severity_match else ""
            diagnostics.append(
                {
                    "source": source,
                    "severity": severity,
                    "raw": block.strip(),
                }
            )
        return diagnostics

    @staticmethod
    def _is_env_setup_error(stderr: str) -> bool:
        lowered = stderr.lower()
        markers = (
            "can't find a package database",
            "could not find module",
            "openbinaryfile: does not exist",
            "failed to parse daml.yaml",
            "environment variable",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _extract_missing_dar_paths(stderr: str) -> list[str]:
        # Example:
        # damlc: ../model/.daml/dist/finlib-2.0.0.dar: openBinaryFile: does not exist
        return re.findall(r"damlc:\s+([^\n:]+\.dar):\s+openBinaryFile:\s+does not exist", stderr)

    def _run_lint_process(self, file_path: Path, daml_root: Path) -> subprocess.CompletedProcess[str]:
        rel_path = os.path.relpath(str(file_path), str(daml_root))
        command = self._command_prefix_for(file_path) + ["daml", "damlc", "lint"]
        target_flag = self._target_flag_for(daml_root)
        if target_flag:
            command.append(target_flag)
        command.append(rel_path)
        return subprocess.run(
            command,
            cwd=str(daml_root),
            capture_output=True,
            text=True,
            check=False,
        )

    def _run_build_process(self, file_path: Path, project_root: Path) -> subprocess.CompletedProcess[str]:
        command = self._command_prefix_for(file_path) + ["daml", "build"]
        if (project_root / "multi-package.yaml").exists():
            command.append("--all")
        return subprocess.run(
            command,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
        )

    def _build_missing_dars_once(self, file_path: Path, daml_root: Path, build_stderr: str) -> None:
        for raw_path in self._extract_missing_dar_paths(build_stderr):
            missing = (daml_root / raw_path).resolve()
            marker = f"{os.sep}.daml{os.sep}dist{os.sep}"
            missing_str = str(missing)
            if marker not in missing_str:
                continue
            project_path_str = missing_str.split(marker, 1)[0]
            dep_project = Path(project_path_str)
            if not (dep_project / "daml.yaml").exists():
                continue
            # Best-effort dependency build.
            self._run_build_process(file_path, dep_project)

    def _ensure_project_prepared(self, file_path: Path, daml_root: Path, prior_stderr: str) -> None:
        """
        Best effort: build project context once per root if lint errors indicate setup gaps.
        """
        key = str(daml_root.resolve())
        with self._cache_guard:
            cached = self._project_build_cache.get(key)
        if cached is not None:
            return

        project_lock = self._get_project_lock(daml_root)
        with project_lock:
            with self._cache_guard:
                cached = self._project_build_cache.get(key)
            if cached is not None:
                return

            project_root = find_daml_build_root(daml_root) or daml_root
            build_proc = self._run_build_process(file_path, project_root)
            if build_proc.returncode == 0:
                with self._cache_guard:
                    self._project_build_cache[key] = (True, None)
                return

            # Handle package layouts that refer to sibling DARs not built yet.
            stderr = build_proc.stderr or ""
            self._build_missing_dars_once(file_path, daml_root, stderr)
            retry_proc = self._run_build_process(file_path, project_root)
            if retry_proc.returncode == 0:
                with self._cache_guard:
                    self._project_build_cache[key] = (True, None)
                return

            combined_err = (retry_proc.stderr or build_proc.stderr or prior_stderr or "").strip()
            with self._cache_guard:
                self._project_build_cache[key] = (False, combined_err)

    def _lint_file_unlocked(self, file_path: Path) -> tuple[bool | None, str | None]:
        # The syntax pass-rate metric shells out to the Daml SDK. When it isn't
        # installed, skip cleanly (exact-match still runs) rather than erroring
        # per example. See scripts/install_daml.sh / the README.
        if shutil.which("daml") is None:
            return None, "daml toolchain not installed; skipped syntax check (see README)"

        daml_root = find_daml_yaml_root(file_path)
        if daml_root is None:
            return None, (
                f"No daml.yaml found for: {file_path}. Run scripts/checkout_source_repos.sh "
                "to fetch the source repos the syntax check needs."
            )

        try:
            process = self._run_lint_process(file_path, daml_root)
        except FileNotFoundError as e:
            return None, f"Failed to execute DAML CLI for {file_path}. Error: {e}"

        def _classify(proc: subprocess.CompletedProcess[str]) -> tuple[str, str | None]:
            stderr_local = proc.stderr or ""
            parser_errors_local = self._extract_parser_errors(stderr_local)
            if parser_errors_local:
                return "syntax_fail", "\n\n".join(parser_errors_local)

            diagnostics = self._extract_diagnostics(stderr_local)
            has_error = any(d["severity"] == "DsError" for d in diagnostics)
            if not has_error:
                return "ok", None

            # Non-parser DsError: setup/typecheck/not-found. Not a syntax parse failure.
            return "inconclusive", (stderr_local or proc.stdout or "").strip()

        status, message = _classify(process)
        if status == "ok":
            return True, None
        if status == "syntax_fail":
            return False, message

        stderr = process.stderr or ""
        if self._is_env_setup_error(stderr):
            self._ensure_project_prepared(file_path, daml_root, stderr)
            retry = self._run_lint_process(file_path, daml_root)
            status_retry, message_retry = _classify(retry)
            if status_retry == "ok":
                return True, None
            if status_retry == "syntax_fail":
                return False, message_retry
            return None, message_retry

        return None, message

    def lint_file(self, file_path: Path) -> tuple[bool | None, str | None]:
        project_lock = self._project_lock_for_file(file_path)
        file_lock = self._get_file_lock(file_path)
        if project_lock is None:
            with file_lock:
                return self._lint_file_unlocked(file_path)
        with project_lock:
            with file_lock:
                return self._lint_file_unlocked(file_path)

    def lint_file_with_content(self, file_path: Path, content: str) -> tuple[bool | None, str | None]:
        project_lock = self._project_lock_for_file(file_path)
        file_lock = self._get_file_lock(file_path)
        if project_lock is None:
            with file_lock:
                original_content = file_path.read_text(encoding="utf-8")
                if original_content == content:
                    return self._lint_file_unlocked(file_path)

                _track_active_file_mutation(file_path, original_content)
                try:
                    file_path.write_text(content, encoding="utf-8")
                    return self._lint_file_unlocked(file_path)
                finally:
                    _restore_tracked_file_mutation(file_path, original_content)

        with project_lock:
            with file_lock:
                original_content = file_path.read_text(encoding="utf-8")
                if original_content == content:
                    return self._lint_file_unlocked(file_path)

                _track_active_file_mutation(file_path, original_content)
                try:
                    file_path.write_text(content, encoding="utf-8")
                    return self._lint_file_unlocked(file_path)
                finally:
                    _restore_tracked_file_mutation(file_path, original_content)


def resolve_span_in_original_file(row: dict[str, Any], original_text: str) -> tuple[int, int] | None:
    """
    Resolve target span in the *full original file*.
    Returns (start, end) character offsets or None if it cannot be determined safely.
    """
    metadata = row.get("metadata") or {}
    target = row.get("target", "")
    candidates: list[tuple[int, int]] = []

    start_pos = metadata.get("start_pos")
    end_pos = metadata.get("end_pos")
    if isinstance(start_pos, int) and isinstance(end_pos, int):
        candidates.append((start_pos, end_pos))

    original_prefix_len = metadata.get("original_prefix_len")
    if isinstance(original_prefix_len, int):
        target_len = metadata.get("original_target_len")
        if not isinstance(target_len, int):
            target_len = metadata.get("span_length")
        if not isinstance(target_len, int):
            target_len = len(target)
        candidates.append((original_prefix_len, original_prefix_len + target_len))

    prefix_chars_removed = metadata.get("prefix_chars_removed")
    if isinstance(prefix_chars_removed, int):
        start = prefix_chars_removed + len(row.get("prefix", ""))
        end = start + len(target)
        candidates.append((start, end))

    seen: set[tuple[int, int]] = set()
    for start, end in candidates:
        if (start, end) in seen:
            continue
        seen.add((start, end))

        if start < 0 or end < start or end > len(original_text):
            continue
        if original_text[start:end] == target:
            return start, end

    return None


def reconstruct_row_snapshot_text(row: dict[str, Any]) -> tuple[str, tuple[int, int]] | None:
    """
    Reconstruct full original file text from row payload when available.
    This avoids file-drift issues during retroactive backfills.
    """
    metadata = row.get("metadata") or {}
    prefix = row.get("prefix")
    target = row.get("target")
    suffix = row.get("suffix")
    if not isinstance(prefix, str) or not isinstance(target, str) or not isinstance(suffix, str):
        return None

    if metadata.get("trimmed") is True:
        return None
    if metadata.get("prefix_chars_removed", 0) != 0:
        return None
    if metadata.get("suffix_chars_removed", 0) != 0:
        return None

    original_prefix_len = metadata.get("original_prefix_len")
    if isinstance(original_prefix_len, int) and original_prefix_len != len(prefix):
        return None
    original_target_len = metadata.get("original_target_len")
    if isinstance(original_target_len, int) and original_target_len != len(target):
        return None
    original_suffix_len = metadata.get("original_suffix_len")
    if isinstance(original_suffix_len, int) and original_suffix_len != len(suffix):
        return None

    text = prefix + target + suffix
    span = (len(prefix), len(prefix) + len(target))
    return text, span


def augment_results_with_syntax_metrics(
    results: list[dict[str, Any]],
    progress_cb: Callable[..., None] | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """
    Adds per-example syntax metrics in-place and returns aggregate syntax summary fields.
    """
    checker = DamlSyntaxChecker()
    original_file_cache: dict[str, tuple[bool | None, str | None]] = {}
    original_cache_lock = threading.Lock()

    def process_row(row: dict[str, Any]) -> None:
        row["original_syntax_passed"] = None
        row["predicted_syntax_passed"] = None
        row.pop("syntax_check_skip_reason", None)
        row.pop("syntax_check_error", None)

        file_path_raw = row.get("file_path")
        if not file_path_raw:
            row["syntax_check_skip_reason"] = "missing_file_path"
            return

        file_path = resolve_file_path_for_syntax_check(file_path_raw)
        if not file_path.exists():
            row["syntax_check_skip_reason"] = "missing_file"
            return

        snapshot = reconstruct_row_snapshot_text(row)
        if snapshot is not None:
            original_file_text, (start, end) = snapshot
            text_source = "row_snapshot"
        else:
            try:
                original_file_text = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                row["syntax_check_skip_reason"] = "file_read_error"
                return

            span = resolve_span_in_original_file(row, original_file_text)
            if span is None:
                row["syntax_check_skip_reason"] = "span_resolution_failed"
                return
            start, end = span
            text_source = "disk_file"

        text_hash = hashlib.sha1(original_file_text.encode("utf-8")).hexdigest()
        cache_key = f"{file_path.resolve()}::{text_source}::{text_hash}"
        with original_cache_lock:
            cached = original_file_cache.get(cache_key)

        if cached is None:
            if text_source == "row_snapshot":
                computed = checker.lint_file_with_content(file_path, original_file_text)
            else:
                computed = checker.lint_file(file_path)
            with original_cache_lock:
                cached = original_file_cache.setdefault(cache_key, computed)

        original_ok, original_err = cached
        if not isinstance(original_ok, bool):
            row["syntax_check_skip_reason"] = "syntax_checker_inconclusive"
            if original_err:
                row["syntax_check_error"] = original_err
            return
        row["original_syntax_passed"] = original_ok
        if not original_ok:
            row["syntax_check_skip_reason"] = "original_sanity_failed"
            if original_err:
                row["syntax_check_error"] = original_err
            return

        predicted = row.get("predicted", "")
        if predicted == row.get("target", ""):
            row["predicted_syntax_passed"] = original_ok
            return

        completed_text = original_file_text[:start] + predicted + original_file_text[end:]
        predicted_ok, predicted_err = checker.lint_file_with_content(file_path, completed_text)
        if not isinstance(predicted_ok, bool):
            row["predicted_syntax_passed"] = None
            row["syntax_check_skip_reason"] = "syntax_checker_inconclusive"
            if predicted_err:
                row["syntax_check_error"] = predicted_err
            return
        row["predicted_syntax_passed"] = predicted_ok
        if not predicted_ok and predicted_err:
            row["syntax_check_error"] = predicted_err

    def _progress_deltas(row: dict[str, Any]) -> tuple[int, int]:
        pred = row.get("predicted_syntax_passed")
        if not isinstance(pred, bool):
            return 0, 0
        return 1, 1 if pred else 0

    def _notify_progress(checkable_delta: int, pass_delta: int) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(1, checkable_delta, pass_delta)
            return
        except TypeError:
            pass
        try:
            progress_cb(1)
            return
        except TypeError:
            progress_cb()

    workers = max(1, int(workers))
    if workers == 1:
        for row in results:
            process_row(row)
            checkable_delta, pass_delta = _progress_deltas(row)
            _notify_progress(checkable_delta, pass_delta)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_row = {executor.submit(process_row, row): row for row in results}
            for future in as_completed(future_to_row):
                future.result()
                row = future_to_row[future]
                checkable_delta, pass_delta = _progress_deltas(row)
                _notify_progress(checkable_delta, pass_delta)

    return compute_syntax_summary(results)


# Backwards-compatible aliases for existing imports.
_resolve_span_in_original_file = resolve_span_in_original_file
_reconstruct_row_snapshot_text = reconstruct_row_snapshot_text


def compute_syntax_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Computes run-level syntax summary fields from per-example result rows.
    """
    total = len(results)
    predicted_checkable = 0
    predicted_pass = 0
    original_pass = 0
    original_checked = 0

    files_checked: dict[str, bool] = {}
    skip_reason_counts: dict[str, int] = {}

    for row in results:
        pred = row.get("predicted_syntax_passed")
        orig = row.get("original_syntax_passed")
        if isinstance(orig, bool):
            original_checked += 1
            if orig:
                original_pass += 1

            file_path = row.get("file_path")
            if isinstance(file_path, str):
                # If any sample for the same file fails original sanity, keep it failed.
                files_checked[file_path] = files_checked.get(file_path, True) and orig

        if not isinstance(pred, bool):
            reason = row.get("syntax_check_skip_reason")
            if isinstance(reason, str) and reason:
                skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + 1
            continue

        predicted_checkable += 1
        if pred:
            predicted_pass += 1

    skipped = total - predicted_checkable
    file_total = len(files_checked)
    file_pass = sum(1 for ok in files_checked.values() if ok)

    predicted_rate = (predicted_pass / predicted_checkable) if predicted_checkable else 0.0
    original_rate = (original_pass / original_checked) if original_checked else 0.0
    original_file_rate = (file_pass / file_total) if file_total else 0.0

    return {
        "predicted_syntax_pass_rate": predicted_rate,
        "original_syntax_pass_rate": original_rate,
        "syntax_checkable_examples": predicted_checkable,
        "syntax_skipped_examples": skipped,
        "syntax_skip_reason_counts": skip_reason_counts,
        "syntax_skip_reason_explanations": {
            reason: SKIP_REASON_EXPLANATIONS.get(reason, "Unspecified skip reason.")
            for reason in skip_reason_counts
        },
        "syntax_inconclusive_examples": skip_reason_counts.get("syntax_checker_inconclusive", 0),
        "syntax_original_checked_examples": original_checked,
        "syntax_original_files_checked": file_total,
        "syntax_original_file_pass_rate": original_file_rate,
        "syntax_original_files_failed": file_total - file_pass,
        "syntax_sanity_passed": file_total == file_pass,
    }


def load_results_jsonl(path: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not path.exists():
        return results
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def write_results_jsonl(path: Path, results: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
