"""Config defaults, path resolution, version resolution, and merge()."""
from types import SimpleNamespace

import pytest

from daml_fim_benchmark.config import (
    EvaluationConfig,
    abs_path,
    get_latest_version,
    project_root,
    resolve_dataset_version,
)


def test_default_version_is_public():
    assert EvaluationConfig().dataset_version == "vpublic"


def test_project_root_uses_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DAML_FIM_BENCHMARK_ROOT", str(tmp_path))
    assert project_root() == tmp_path.resolve()
    assert abs_path("fim/data") == tmp_path.resolve() / "fim" / "data"


def test_project_root_defaults_to_cwd(monkeypatch, tmp_path):
    monkeypatch.delenv("DAML_FIM_BENCHMARK_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert project_root() == tmp_path


def test_resolve_dataset_version_passthrough():
    assert resolve_dataset_version("vpublic", "fim/data") == "vpublic"


def test_resolve_dataset_version_latest(tmp_path):
    for name in ("v1", "v2", "v2_iso", "not_a_version"):
        (tmp_path / name).mkdir()
    assert resolve_dataset_version("latest", tmp_path) == "v2_iso"  # highest numeric, lexical tie-break


def test_get_latest_version_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_latest_version(tmp_path / "nope")


def test_get_latest_version_none_found(tmp_path):
    (tmp_path / "random_dir").mkdir()
    with pytest.raises(ValueError):
        get_latest_version(tmp_path)


def test_merge_applies_known_keys():
    cfg = EvaluationConfig()
    merged = cfg.merge(SimpleNamespace(samples_per_repo=7, dataset_version="vX"))
    assert merged.samples_per_repo == 7
    assert merged.dataset_version == "vX"
    assert cfg.samples_per_repo != 7  # original unchanged (returns a copy)


def test_merge_rejects_unknown_key():
    with pytest.raises(ValueError):
        EvaluationConfig().merge(SimpleNamespace(not_a_real_field=1))


def test_merge_allows_unknown_when_flagged():
    merged = EvaluationConfig().merge(SimpleNamespace(not_a_real_field=1), allow_unknown_keys=True)
    assert not hasattr(merged, "not_a_real_field")
