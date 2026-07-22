"""The Daml syntax-pass-rate check degrades gracefully when the toolchain is
absent — it skips with a clear reason instead of erroring, so exact-match still
runs for users without the Daml SDK installed."""
import daml_fim_benchmark.syntax_metrics as sm


def test_syntax_check_skips_when_daml_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(sm.shutil, "which", lambda name: None)
    checker = sm.DamlSyntaxChecker()

    passed, reason = checker.lint_file(tmp_path / "M.daml")

    assert passed is None  # skipped, not pass/fail
    assert reason is not None
    assert "daml" in reason.lower() and "not installed" in reason.lower()


def test_resolver_maps_repo_path_under_sources(monkeypatch, tmp_path):
    # file_path is repo-relative (<repo>/...); the checkout lives under sources/.
    src = tmp_path / "sources" / "daml-finance" / "src" / "M.daml"
    src.parent.mkdir(parents=True)
    src.write_text("module M where\n")
    monkeypatch.chdir(tmp_path)

    resolved = sm.resolve_file_path_for_syntax_check("daml-finance/src/M.daml")

    assert resolved == src.resolve()


def test_skip_reason_points_at_missing_sources(monkeypatch, tmp_path):
    # daml present, but no source project (no daml.yaml up-tree) -> points at the
    # checkout script rather than a bare failure.
    monkeypatch.setattr(sm.shutil, "which", lambda name: "/usr/bin/daml")
    checker = sm.DamlSyntaxChecker()

    passed, reason = checker.lint_file(tmp_path / "no_project" / "M.daml")

    assert passed is None
    assert "daml.yaml" in reason and "checkout_source_repos" in reason
