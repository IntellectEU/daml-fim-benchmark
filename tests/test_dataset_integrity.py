"""The shipped public test split is well-formed."""
import json

import pytest

PUBLIC_REPOS = {"daml", "daml-finance", "splice", "canton", "ex-models", "account-hierarchy"}
DATASET_FILES = ["syntax_test.jsonl", "random_test.jsonl"]


def _rows(repo_root, fname):
    path = repo_root / "fim" / "data" / "vpublic" / fname
    assert path.exists(), f"missing dataset file: {path}"
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.mark.parametrize("fname", DATASET_FILES)
def test_required_fields_present(repo_root, fname):
    rows = _rows(repo_root, fname)
    assert rows, f"{fname} is empty"
    for r in rows:
        for key in ("id", "prefix", "suffix", "target", "file_path", "metadata"):
            assert key in r, f"{fname}: row {r.get('id')} missing {key}"
        assert "daml_version" in r["metadata"]


def test_ids_unique_across_split(repo_root):
    ids = []
    for fname in DATASET_FILES:
        ids += [r["id"] for r in _rows(repo_root, fname)]
    assert len(ids) == len(set(ids)), "duplicate example ids in the test split"


@pytest.mark.parametrize("fname", DATASET_FILES)
def test_file_paths_resolve_to_public_repos(repo_root, fname):
    for r in _rows(repo_root, fname):
        parts = r["file_path"].split("/")
        assert parts[0] in PUBLIC_REPOS, f"{fname}: unexpected repo {parts[0]!r}"
        assert len(parts) >= 2, f"{fname}: file_path has no path beneath the repo: {r['file_path']!r}"


@pytest.mark.parametrize("fname", DATASET_FILES)
def test_daml_version_values_are_known(repo_root, fname):
    allowed = {"2.x", "3.x", "unknown"}
    for r in _rows(repo_root, fname):
        assert r["metadata"]["daml_version"] in allowed
