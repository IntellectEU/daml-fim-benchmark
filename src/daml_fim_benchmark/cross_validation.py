"""Eval-side aggregation of cross-validation fold summaries into a CV parent.

Given the per-fold evaluation summaries, combine them into a single parent
summary: weighted overall metrics, mean fold-level ID/OOD averages, and merged
per-repo scores.
"""

from __future__ import annotations


def _weighted_mean(entries: list[tuple[float, float]]) -> float | None:
    if not entries:
        return None
    total_weight = sum(w for _, w in entries)
    if total_weight <= 0:
        return None
    return sum(v * w for v, w in entries) / total_weight


def aggregate_cv_eval_summaries(subruns: list[dict]) -> dict:
    """
    Aggregate subrun eval summaries into one CV-level summary payload.
    """
    if not subruns:
        raise ValueError("Cannot aggregate empty CV subrun list.")

    weights = [max(1, int(r.get("eval_summary", {}).get("total_examples", 0) or 0)) for r in subruns]
    total_examples_values = [int(r.get("eval_summary", {}).get("total_examples", 0) or 0) for r in subruns]
    total_examples = int(round(sum(total_examples_values) / len(total_examples_values))) if total_examples_values else 0

    metric_keys = [
        "exact_match_rate",
        "prefix_match_rate",
        "empty_response_rate",
        "predicted_syntax_pass_rate",
        "original_syntax_pass_rate",
        "syntax_original_file_pass_rate",
        "syntax_in_domain_pass_rate",
        "syntax_out_of_domain_pass_rate",
    ]
    metrics: dict[str, float | None] = {}
    for key in metric_keys:
        values: list[tuple[float, float]] = []
        for subrun, weight in zip(subruns, weights):
            value = subrun.get("eval_summary", {}).get(key)
            if value is None:
                continue
            try:
                values.append((float(value), float(weight)))
            except (TypeError, ValueError):
                continue
        metrics[key] = _weighted_mean(values)

    def _avg_rate_map(field: str) -> dict[str, float]:
        buckets: dict[str, list[tuple[float, float]]] = {}
        for subrun, weight in zip(subruns, weights):
            mapping = subrun.get("eval_summary", {}).get(field)
            if not isinstance(mapping, dict):
                continue
            for name, value in mapping.items():
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    continue
                buckets.setdefault(str(name), []).append((v, float(weight)))
        out: dict[str, float] = {}
        for name, vals in buckets.items():
            mean = _weighted_mean(vals)
            if mean is not None:
                out[name] = mean
        return out

    by_category = _avg_rate_map("by_category")
    by_source = _avg_rate_map("by_source")

    # CV parent ID/OOD averages should be the plain mean of fold-level ID/OOD values.
    def _mean_fold_metric(field: str) -> tuple[float | None, int]:
        values: list[float] = []
        for subrun in subruns:
            value = subrun.get("eval_summary", {}).get(field)
            if value is None:
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        if not values:
            return None, 0
        return (sum(values) / len(values)), len(values)

    total_repo_average, _ = _mean_fold_metric("total_repo_average")
    in_domain_repo_average, in_domain_repo_count = _mean_fold_metric("in_domain_repo_average")
    out_of_domain_repo_average, out_of_domain_repo_count = _mean_fold_metric("out_of_domain_repo_average")
    in_domain_repo_prefix_average, _ = _mean_fold_metric("in_domain_repo_prefix_average")
    out_of_domain_repo_prefix_average, _ = _mean_fold_metric("out_of_domain_repo_prefix_average")
    total_repo_prefix_average, _ = _mean_fold_metric("total_repo_prefix_average")

    # Aggregate repo-level scores.
    repo_buckets: dict[str, list[tuple[float, float, float, int]]] = {}
    for subrun in subruns:
        repo_scores = subrun.get("eval_summary", {}).get("repo_scores")
        if not isinstance(repo_scores, list):
            continue
        for entry in repo_scores:
            if not isinstance(entry, dict):
                continue
            repo = str(entry.get("repo", "")).strip()
            if not repo:
                continue
            try:
                rate = float(entry.get("exact_match_rate", 0.0))
            except (TypeError, ValueError):
                continue
            try:
                prefix_rate = float(entry.get("prefix_match_rate", 0.0))
            except (TypeError, ValueError):
                prefix_rate = 0.0
            try:
                examples = int(entry.get("total_examples", 0) or 0)
            except (TypeError, ValueError):
                examples = 0
            weight = float(max(1, examples))
            repo_buckets.setdefault(repo, []).append((rate, prefix_rate, weight, examples))

    repo_scores_agg: list[dict] = []
    for repo in sorted(repo_buckets.keys()):
        vals = repo_buckets[repo]
        rate = _weighted_mean([(r, w) for r, _, w, _ in vals]) or 0.0
        prefix_rate = _weighted_mean([(p, w) for _, p, w, _ in vals]) or 0.0
        avg_examples = int(round(sum(ex for _, _, _, ex in vals) / len(vals))) if vals else 0
        repo_scores_agg.append(
            {
                "repo": repo,
                "exact_match_rate": rate,
                "prefix_match_rate": prefix_rate,
                "total_examples": avg_examples,
                "in_domain": None,
                "domain": "mixed",
            }
        )

    syntax_checkable_examples = int(
        round(
            sum(int(s.get("eval_summary", {}).get("syntax_checkable_examples", 0) or 0) for s in subruns)
            / len(subruns)
        )
    )

    subrun_rows = []
    for subrun in subruns:
        eval_summary = subrun.get("eval_summary", {})
        subrun_rows.append(
            {
                "index": int(subrun.get("fold_index", 0) or 0),
                "eval_run_id": str(eval_summary.get("run_id", "")).strip(),
                "holdout_repos": list(subrun.get("holdout_repos", [])),
                "train_repos": list(subrun.get("train_repos", [])),
                "exact_match_rate": eval_summary.get("exact_match_rate"),
                "prefix_match_rate": eval_summary.get("prefix_match_rate"),
            }
        )

    return {
        "total_examples": total_examples,
        "syntax_checkable_examples": syntax_checkable_examples,
        "by_category": by_category,
        "by_source": by_source,
        "repo_scores": repo_scores_agg,
        "total_repo_average": total_repo_average,
        "in_domain_repo_average": in_domain_repo_average,
        "out_of_domain_repo_average": out_of_domain_repo_average,
        "total_repo_prefix_average": total_repo_prefix_average,
        "in_domain_repo_prefix_average": in_domain_repo_prefix_average,
        "out_of_domain_repo_prefix_average": out_of_domain_repo_prefix_average,
        "in_domain_repo_count": in_domain_repo_count,
        "out_of_domain_repo_count": out_of_domain_repo_count,
        "train_repo_count": None,
        "cv_subruns": subrun_rows,
        "cv_subrun_run_ids": [row["eval_run_id"] for row in subrun_rows if row["eval_run_id"]],
        **metrics,
    }
