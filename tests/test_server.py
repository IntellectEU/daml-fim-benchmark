"""FastAPI results server (no model needed; TestClient over the shipped baselines).

DAML_FIM_BENCHMARK_ROOT must point at the repo before the app module is imported,
because the server resolves FIM_LOGS_DIR at import time.
"""
import os
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["DAML_FIM_BENCHMARK_ROOT"] = str(REPO_ROOT)

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from daml_fim_benchmark.server.app import (  # noqa: E402
    DEFAULT_CORS_ORIGINS,
    _extract_timestamp_from_run_id,
    _infer_dataset_label,
    _infer_eval_kind,
    _is_run_in_progress,
    _iter_run_dirs,
    _normalize_run_id,
    _resolve_cors_origins,
    _run_to_item,
    app,
)

client = TestClient(app)

BASELINE_RUNS = {
    "baseline-ieu-7b-daml3",
    "baseline-ieu-7b-daml3-fold1",
    "baseline-ieu-7b-daml3-fold2",
    "baseline-ieu-7b-daml2",
    "baseline-copilot-daml3",
    "baseline-copilot-daml2",
}


def test_health():
    assert client.get("/health").status_code == 200


def test_runs_lists_baselines():
    resp = client.get("/api/fim/runs")
    assert resp.status_code == 200
    run_ids = {item["run_id"] for item in resp.json()["items"]}
    assert BASELINE_RUNS.issubset(run_ids)


def test_every_eval_kind_is_a_valid_filter():
    """Every eval kind the UI can display must be accepted (and round-trip) as a filter."""
    all_kinds = {item["eval_kind"] for item in client.get("/api/fim/runs").json()["items"]}
    assert {"Undisclosed", "OOD", "CV"}.issubset(all_kinds)
    for kind in all_kinds:
        resp = client.get("/api/fim/runs", params={"eval_filter": kind})
        assert resp.status_code == 200, f"eval_filter={kind} rejected: {resp.text}"
        listed = {item["eval_kind"] for item in resp.json()["items"]}
        assert listed == {kind}, f"eval_filter={kind} returned kinds {listed}"


def test_per_result_detail_returns_prefix_suffix():
    """The per-result detail endpoint returns the FIM context (prefix/suffix)."""
    listing = client.get("/api/fim/runs/baseline-copilot-daml3/results").json()
    assert listing["items"], "no results listed for baseline-copilot-daml3"
    rid = listing["items"][0]["id"]

    detail = client.get(f"/api/fim/runs/baseline-copilot-daml3/results/{rid}").json()
    row = detail["row"]
    for key in ("prefix", "suffix", "target", "predicted"):
        assert key in row
    assert row["prefix"], "prefix is empty in the detail response"
    assert row["suffix"], "suffix is empty in the detail response"


def test_unknown_result_id_is_404():
    assert client.get("/api/fim/runs/baseline-copilot-daml3/results/does-not-exist").status_code == 404


def test_run_id_path_traversal_rejected():
    with pytest.raises(Exception):
        _normalize_run_id("../etc/passwd")
    with pytest.raises(Exception):
        _normalize_run_id("a/b")


def test_normalize_run_id_passes_clean_value():
    assert _normalize_run_id("baseline-copilot-daml3") == "baseline-copilot-daml3"


def test_infer_eval_kind():
    assert _infer_eval_kind("cv", "x") == "CV"
    assert _infer_eval_kind("all", "x") == "Multi"
    assert _infer_eval_kind("ood", "x") == "OOD"
    assert _infer_eval_kind(None, "qwen_7b_cv_2") == "CV"
    assert _infer_eval_kind("undisclosed", "x") == "Undisclosed"
    # No declared training-data relationship: ID/OOD cannot be known.
    assert _infer_eval_kind(None, "plain-run") == "Undisclosed"


def test_extract_timestamp_from_run_id():
    assert _extract_timestamp_from_run_id("run_260624-134653_x") is not None
    assert _extract_timestamp_from_run_id("baseline-copilot-daml3") is None


def test_run_listing_order_is_deterministic():
    """The shipped baselines all have timestamp: null, so filesystem iteration
    order alone would be arbitrary. Listing twice must give the same order,
    and among null-timestamp runs that order must be run_id ascending."""
    first = [run_id for run_id, _run_dir, _archived in _iter_run_dirs(include_archived=False)]
    second = [run_id for run_id, _run_dir, _archived in _iter_run_dirs(include_archived=False)]
    assert first == second

    null_ts_baselines = [rid for rid in first if rid in BASELINE_RUNS]
    assert len(null_ts_baselines) == len(BASELINE_RUNS)
    assert null_ts_baselines == sorted(null_ts_baselines)


def test_run_listing_order_via_api_is_stable_across_calls():
    first = [item["run_id"] for item in client.get("/api/fim/runs").json()["items"]]
    second = [item["run_id"] for item in client.get("/api/fim/runs").json()["items"]]
    assert first == second


def test_is_run_in_progress_false_right_after_fresh_checkout(tmp_path):
    """A `git clone` writes results.jsonl and summary.json within the same
    second. That must not be mistaken for a run still appending results."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    results_path.write_text('{"id": "1"}\n', encoding="utf-8")
    summary_path.write_text("{}", encoding="utf-8")

    now = time.time()
    # Both files "checked out" a few seconds ago, results.jsonl a hair newer
    # than summary.json -- well within a checkout's same-second write window.
    os.utime(summary_path, (now - 60, now - 60))
    os.utime(results_path, (now - 60, now - 60 + 1))

    assert _is_run_in_progress("run", run_dir, archived=False) is False


def test_is_run_in_progress_true_when_results_newer_by_a_real_margin(tmp_path):
    """A genuinely running eval appends results.jsonl for minutes before
    writing summary.json at the end, so results.jsonl ends up recent and
    clearly newer than summary.json."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    results_path.write_text('{"id": "1"}\n', encoding="utf-8")
    summary_path.write_text("{}", encoding="utf-8")

    now = time.time()
    os.utime(summary_path, (now - 60, now - 60))
    os.utime(results_path, (now - 5, now - 5))

    assert _is_run_in_progress("run", run_dir, archived=False) is True


def test_is_run_in_progress_false_when_results_stale(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    results_path.write_text('{"id": "1"}\n', encoding="utf-8")
    summary_path.write_text("{}", encoding="utf-8")

    now = time.time()
    os.utime(summary_path, (now - 1000, now - 1000))
    os.utime(results_path, (now - 500, now - 500))

    assert _is_run_in_progress("run", run_dir, archived=False) is False


def test_is_run_in_progress_true_when_no_summary_yet(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    results_path = run_dir / "results.jsonl"
    results_path.write_text('{"id": "1"}\n', encoding="utf-8")

    now = time.time()
    os.utime(results_path, (now, now))

    assert _is_run_in_progress("run", run_dir, archived=False) is True


def test_undeclared_run_is_undisclosed_via_run_to_item(tmp_path):
    """A run whose summary/config declare no training-data relationship must be
    classified Undisclosed end to end, not fall back to ID."""
    assert _infer_dataset_label({}, {}) == ""
    run_dir = tmp_path / "some-run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text('{"total_examples": 1, "exact_match_rate": 1.0}')
    item = _run_to_item(run_id="some-run", run_dir=run_dir, archived=False)
    assert item.eval_kind == "Undisclosed"


def test_cv_parent_details_serve_heldout_merge():
    """The CV parent has no results.jsonl; its details must be the held-out merge
    of its folds: canton rows from fold 1, splice rows from fold 2."""
    listing = client.get("/api/fim/runs/baseline-ieu-7b-daml3/results", params={"page_size": 1}).json()
    assert listing["total"] == 2579

    # Exact matches over the merge are the published held-out numerator (32.7%).
    matches = client.get(
        "/api/fim/runs/baseline-ieu-7b-daml3/results",
        params={"page_size": 1, "outcome_filter": "match"},
    ).json()
    assert matches["total"] == 843

    # canton rows must come from fold 1 (the fold that held canton out), not from
    # fold 2 where canton was in-domain. Pin it via an example whose two folds
    # predicted different completions.
    import json as _json

    def rows_by_id(run_id: str, repo: str) -> dict:
        path = REPO_ROOT / "fim" / "logs" / run_id / "results.jsonl"
        out = {}
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = _json.loads(line)
                if row.get("repo") == repo:
                    out[row["id"]] = row
        return out

    fold1 = rows_by_id("baseline-ieu-7b-daml3-fold1", "canton")
    fold2 = rows_by_id("baseline-ieu-7b-daml3-fold2", "canton")
    diverging_id = next(
        rid for rid, row in fold1.items() if row.get("predicted") != fold2[rid].get("predicted")
    )
    detail = client.get(f"/api/fim/runs/baseline-ieu-7b-daml3/results/{diverging_id}").json()
    assert detail["row"]["predicted"] == fold1[diverging_id].get("predicted")
    assert detail["row"]["predicted"] != fold2[diverging_id].get("predicted")


def test_cv_parent_with_missing_folds_is_404_not_empty(tmp_path):
    """A CV parent whose fold runs are gone must error clearly, not pretend to
    be a run with zero results."""
    from fastapi import HTTPException

    from daml_fim_benchmark.server.app import _iter_run_results

    run_dir = tmp_path / "orphan-parent"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text('{"cv_subrun_run_ids": ["gone-fold-1", "gone-fold-2"]}')
    with pytest.raises(HTTPException) as exc:
        _iter_run_results(run_dir)
    assert exc.value.status_code == 404
    assert "fold" in exc.value.detail


def test_details_category_and_source_filters_are_independent():
    base = "/api/fim/runs/baseline-ieu-7b-daml3/results"
    listing = client.get(base, params={"page_size": 1}).json()
    # Option lists come from distinct fields: sources are the test-split names,
    # categories are span kinds and never include the source names.
    assert listing["sources"] == ["random", "syntax"]
    assert "syntax" not in listing["sample_types"]
    assert "random" not in listing["sample_types"]

    by_syntax = client.get(base, params={"page_size": 1, "source": "syntax"}).json()
    by_random = client.get(base, params={"page_size": 1, "source": "random"}).json()
    assert by_syntax["total"] + by_random["total"] == listing["total"]
    assert by_syntax["total"] == 419

    # The two filters compose.
    combined = client.get(
        base, params={"page_size": 1, "source": "syntax", "sample_type": "ensure"}
    ).json()
    assert 0 < combined["total"] < by_syntax["total"]


def test_unweighted_backfill_is_per_repo_mean_not_pooled_rate():
    """total_repo_average is absent from the shipped Copilot summaries; the server
    must backfill it with the unweighted per-repo mean, not the pooled rate."""
    items = {item["run_id"]: item for item in client.get("/api/fim/runs").json()["items"]}
    run = items["baseline-copilot-daml2"]
    assert run["eval_kind"] == "Undisclosed"
    assert run["in_domain_repo_average"] is None
    assert run["out_of_domain_repo_average"] is None

    repo_rates = [entry["exact_match_rate"] for entry in run["repo_scores"]]
    expected = sum(repo_rates) / len(repo_rates)
    assert run["total_repo_average"] == pytest.approx(expected)
    # daml-finance dominates the pool, so the two must differ for this run.
    assert run["total_repo_average"] != pytest.approx(run["exact_match_rate"])


def test_cors_origins_default_covers_common_vite_ports():
    origins = set(_resolve_cors_origins())
    assert origins == set(DEFAULT_CORS_ORIGINS)
    assert "http://localhost:5173" in origins
    assert "http://localhost:5174" in origins
    assert "http://127.0.0.1:5173" in origins
    assert "http://127.0.0.1:5174" in origins


def test_cors_origins_configurable_via_env(monkeypatch):
    monkeypatch.setenv("FIM_CORS_ORIGINS", "http://example.com:1234, http://example.com:1234, http://foo.test")
    assert _resolve_cors_origins() == ["http://example.com:1234", "http://foo.test"]
