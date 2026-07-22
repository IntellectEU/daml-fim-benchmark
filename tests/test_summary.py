"""Summary aggregation: micro/macro rates and in/out-of-domain split."""
from daml_fim_benchmark.evaluate import get_repo_summary, get_stats


def _r(repo, exact, prefix=False, category="signatory", source="syntax"):
    return {
        "repo": repo,
        "file_path": f"{repo}/M.daml",
        "exact_match": exact,
        "prefix_match": prefix,
        "category": category,
        "dataset_source": source,
    }


def test_get_stats_micro_average():
    results = [_r("a", True), _r("a", False), _r("b", True), _r("b", True)]
    accuracy, prefix_rate, cat_stats, source_stats = get_stats(results)
    assert accuracy == 0.75  # 3/4 exact (per-sample / micro)
    assert prefix_rate == 0.0


def test_get_stats_empty():
    accuracy, prefix_rate, cat_stats, source_stats = get_stats([])
    assert accuracy == 0 and prefix_rate == 0


def test_get_stats_category_and_source_breakdown():
    results = [
        _r("a", True, category="ensure", source="syntax"),
        _r("a", False, category="ensure", source="syntax"),
        _r("b", True, category="key", source="random"),
    ]
    _, _, cat_stats, source_stats = get_stats(results)
    assert cat_stats["ensure"] == 0.5
    assert cat_stats["key"] == 1.0
    assert source_stats["syntax"] == 0.5
    assert source_stats["random"] == 1.0


def test_repo_summary_per_repo_rates_no_id_ood():
    # Per-repo exact-match rates, and no ID/OOD averages when in_domain_repos is None.
    # repo a: 1/2 = 0.5 ; repo b: 2/2 = 1.0
    results = [_r("a", True), _r("a", False), _r("b", True), _r("b", True)]
    s = get_repo_summary(results, in_domain_repos=None)
    scores = {x["repo"]: x["exact_match_rate"] for x in s["repo_scores"]}
    assert scores == {"a": 0.5, "b": 1.0}
    assert s["in_domain_repo_average"] is None
    assert s["out_of_domain_repo_average"] is None


def test_repo_summary_id_ood_split():
    # Two repos per domain, with *different sizes*, so the averages exercise the
    # unweighted mean across repos (macro) — not a single repo and not a pooled
    # (micro) rate. Per-repo: a=1/2=0.5, c=4/4=1.0 (in); b=1/2=0.5, d=0/1=0.0 (out).
    results = (
        [_r("a", True), _r("a", False)]
        + [_r("c", True)] * 4
        + [_r("b", True), _r("b", False)]
        + [_r("d", False)]
    )
    s = get_repo_summary(results, in_domain_repos={"a", "c"})
    # macro (unweighted mean of per-repo rates):
    assert s["in_domain_repo_average"] == 0.75   # mean(0.5, 1.0); micro would be 5/6
    assert s["out_of_domain_repo_average"] == 0.25  # mean(0.5, 0.0); micro would be 1/3
    assert s["in_domain_repo_count"] == 2
    assert s["out_of_domain_repo_count"] == 2


def test_repo_summary_empty():
    s = get_repo_summary([], in_domain_repos={"a"})
    assert s["repo_scores"] == []
    assert s["train_repo_count"] == 1


def test_repo_summary_infers_repo_from_file_path_when_missing():
    rows = [
        {"file_path": "splice/M.daml", "exact_match": True, "prefix_match": False},
        {"file_path": "splice/N.daml", "exact_match": False, "prefix_match": False},
    ]
    s = get_repo_summary(rows, in_domain_repos=None)
    scores = {x["repo"]: x["exact_match_rate"] for x in s["repo_scores"]}
    assert scores == {"splice": 0.5}
