"""Repo identifier inference from a row's file_path."""
import pytest

from daml_fim_benchmark.evaluate import add_repo_fields, infer_repo_name


@pytest.mark.parametrize(
    "file_path,expected",
    [
        ("daml-finance/src/Foo.daml", "daml-finance"),  # repo = first segment
        ("splice/a/b/c.daml", "splice"),
        ("canton/X.daml", "canton"),
        ("", "unknown"),
        ("Foo.daml", "unknown"),  # bare filename, no repo beneath
        ("daml-finance", "unknown"),  # single segment
    ],
)
def test_infer_repo_name(file_path, expected):
    assert infer_repo_name(file_path) == expected


def test_infer_repo_name_handles_backslashes():
    assert infer_repo_name("daml\\M.daml") == "daml"


def test_add_repo_fields_sets_repo_without_clobbering():
    rows = [
        {"file_path": "canton/M.daml"},
        {"file_path": "daml/N.daml", "repo": "preset"},
    ]
    add_repo_fields(rows)
    assert rows[0]["repo"] == "canton"
    assert rows[1]["repo"] == "preset"  # existing value preserved
