"""FIM prompt construction per prompt type."""
from types import SimpleNamespace

import pytest

from daml_fim_benchmark.prompting import (
    construct_prompt,
    default_prompt_type,
    resolved_prompt_type,
)


def _cfg(prompt_type=None, provider="huggingface"):
    return SimpleNamespace(prompt_type=prompt_type, provider=provider)


def test_default_prompt_type():
    assert default_prompt_type("huggingface") == "PSM"
    assert default_prompt_type("openai") == "CUSTOM_INSTR"


def test_default_prompt_type_unknown_raises():
    with pytest.raises(ValueError):
        default_prompt_type("nope")


def test_resolved_prompt_type_falls_back_to_provider_default():
    assert resolved_prompt_type(_cfg(prompt_type=None, provider="huggingface")) == "PSM"
    assert resolved_prompt_type(_cfg(prompt_type="SPM")) == "SPM"


def test_psm_layout():
    row = {"prefix": "PRE", "suffix": "SUF"}
    out = construct_prompt(row, _cfg(prompt_type="PSM"))
    assert out == "<|fim_prefix|>PRE<|fim_suffix|>SUF<|fim_middle|>"


def test_l2r_is_prefix_only():
    row = {"prefix": "PRE", "suffix": "SUF"}
    assert construct_prompt(row, _cfg(prompt_type="L2R")) == "PRE"


def test_spm_layout():
    row = {"prefix": "PRE", "suffix": "SUF"}
    assert construct_prompt(row, _cfg(prompt_type="SPM")) == "SUF\n[END]\nPRE"


def test_custom_instr_contains_prefix_and_suffix():
    row = {"prefix": "the-prefix", "suffix": "the-suffix"}
    out = construct_prompt(row, _cfg(prompt_type="CUSTOM_INSTR", provider="openai"))
    assert "the-prefix" in out and "the-suffix" in out
    assert "<FILL>" in out  # instructs the wrapped-output format


def test_unknown_prompt_type_raises():
    row = {"prefix": "p", "suffix": "s"}
    with pytest.raises(ValueError):
        construct_prompt(row, _cfg(prompt_type="NOPE"))
