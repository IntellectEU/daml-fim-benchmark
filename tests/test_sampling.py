"""Deterministic per-repo sampling (reproducibility is a core promise)."""
from daml_fim_benchmark.evaluate import _repo_seed, _sample_rows_per_repo


def _rows(repo, kind, n):
    return [{"id": f"{repo}-{kind}-{i}", "repo": repo} for i in range(n)]


def test_repo_seed_is_deterministic_and_varies():
    assert _repo_seed(42, "daml", "syntax") == _repo_seed(42, "daml", "syntax")
    assert _repo_seed(42, "daml", "syntax") != _repo_seed(42, "daml", "random")
    assert _repo_seed(42, "daml", "syntax") != _repo_seed(7, "daml", "syntax")


def test_same_seed_same_selection():
    syn = _rows("daml", "syn", 100)
    rnd = _rows("daml", "rnd", 100)
    a = _sample_rows_per_repo(syn, rnd, samples_per_repo=20, syntax_ratio=0.5, random_seed=42)
    b = _sample_rows_per_repo(syn, rnd, samples_per_repo=20, syntax_ratio=0.5, random_seed=42)
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_different_seed_changes_selection():
    syn = _rows("daml", "syn", 100)
    rnd = _rows("daml", "rnd", 100)
    a = _sample_rows_per_repo(syn, rnd, samples_per_repo=20, syntax_ratio=0.5, random_seed=1)
    b = _sample_rows_per_repo(syn, rnd, samples_per_repo=20, syntax_ratio=0.5, random_seed=2)
    assert [r["id"] for r in a] != [r["id"] for r in b]


def test_cap_per_repo_respected():
    syn = _rows("a", "syn", 100) + _rows("b", "syn", 100)
    rnd = _rows("a", "rnd", 100) + _rows("b", "rnd", 100)
    out = _sample_rows_per_repo(syn, rnd, samples_per_repo=10, syntax_ratio=0.5, random_seed=42)
    per_repo = {}
    for r in out:
        per_repo[r["repo"]] = per_repo.get(r["repo"], 0) + 1
    assert per_repo == {"a": 10, "b": 10}


def test_syntax_ratio_split():
    syn = _rows("a", "syn", 100)
    rnd = _rows("a", "rnd", 100)
    out = _sample_rows_per_repo(syn, rnd, samples_per_repo=10, syntax_ratio=0.5, random_seed=42)
    n_syn = sum(1 for r in out if r["id"].startswith("a-syn"))
    n_rnd = sum(1 for r in out if r["id"].startswith("a-rnd"))
    assert n_syn == 5 and n_rnd == 5


def test_zero_samples_returns_empty():
    assert _sample_rows_per_repo(_rows("a", "syn", 5), _rows("a", "rnd", 5),
                                 samples_per_repo=0, syntax_ratio=0.5, random_seed=42) == []


def test_filler_when_one_pool_short():
    # Only 2 syntax rows but want 5 syntax (ratio 0.5 of 10) -> filler from random pool.
    syn = _rows("a", "syn", 2)
    rnd = _rows("a", "rnd", 100)
    out = _sample_rows_per_repo(syn, rnd, samples_per_repo=10, syntax_ratio=0.5, random_seed=42)
    assert len(out) == 10  # cap still met via filler
