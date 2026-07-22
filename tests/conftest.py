"""Shared fixtures and lightweight fakes for the benchmark test suite.

No GPU, vLLM, or Daml toolchain is required: the package's heavy imports
(torch/transformers/openai/vLLM) are all lazy, and the model call is faked here.
"""
from pathlib import Path

import pytest

from daml_fim_benchmark.config import EvaluationConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Fake vLLM OpenAI-compatible client + tokenizer
# --------------------------------------------------------------------------- #
class _Choice:
    def __init__(self, text):
        self.text = text


class _Response:
    def __init__(self, text, usage=None):
        self.choices = [_Choice(text)]
        self.usage = usage


class FakeCompletions:
    def __init__(self, text, usage=None):
        self._text = text
        self._usage = usage

    def create(self, **kwargs):
        return _Response(self._text, self._usage)


class FakeVllmClient:
    """Stands in for the OpenAI client: ``.with_options(...).completions.create(...)``
    returns a single canned completion. ``text`` may be a str (same for every call)
    or a callable ``(kwargs) -> str`` to vary by request."""

    def __init__(self, text, usage=None):
        self._text = text
        self._usage = usage

    def with_options(self, **_kwargs):
        return self

    @property
    def completions(self):
        return FakeCompletions(self._text, self._usage)


class FakeTokenizer:
    """Minimal tokenizer: encodes by whitespace splitting (only used when the
    fake response carries no ``usage``)."""

    def encode(self, text, add_special_tokens=False):
        return str(text).split()


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


@pytest.fixture
def make_config():
    def _make(**overrides):
        cfg = EvaluationConfig()
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    return _make


@pytest.fixture
def make_row():
    def _make(prefix="PREFIX\n", suffix="\nSUFFIX", target="x", repo="daml-finance", **meta):
        row = {
            "id": meta.pop("id", "id-" + target),
            "file_path": meta.pop("file_path", f"{repo}/M.daml"),
            "prefix": prefix,
            "suffix": suffix,
            "target": target,
            "category": meta.pop("category", "signatory"),
            "repo": repo,
        }
        if meta:
            row["metadata"] = meta
        return row

    return _make


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT
