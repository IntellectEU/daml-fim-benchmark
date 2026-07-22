"""Exact-match / over-generation scoring and normalization.

These guard the heart of the benchmark: a silent bug here corrupts every number.
"""
from daml_fim_benchmark.evaluate import (
    evaluate_single_row_vllm_prepared,
    normalize,
    truncate_suffix_overlap,
)

from conftest import FakeVllmClient


def _score(row, completion_text, config, tokenizer, usage=None, max_tokens=64):
    client = FakeVllmClient(completion_text, usage=usage)
    return evaluate_single_row_vllm_prepared(
        row=row,
        prompt="<prompt>",
        formatted_prompt="<formatted>",
        max_tokens=max_tokens,
        config=config,
        vllm_client=client,
        tokenizer=tokenizer,
    )


# --------------------------- normalize() ----------------------------------- #
def test_normalize_strips_whitespace():
    assert normalize("  abc \n") == "abc"


def test_normalize_strips_fim_end_tokens():
    assert normalize("abc<|endoftext|>") == "abc"
    assert normalize("abc<|end_of_text|>") == "abc"


# --------------------------- exact match ----------------------------------- #
def test_exact_match_true(make_config, make_row, fake_tokenizer):
    cfg = make_config()
    row = make_row(target="signatory owner")
    r = _score(row, "signatory owner", cfg, fake_tokenizer)
    assert r["exact_match"] is True
    assert r["prefix_match"] is False  # disjoint: an exact match is NOT over-generation
    assert r["predicted"] == "signatory owner"


def test_exact_match_ignores_surrounding_whitespace(make_config, make_row, fake_tokenizer):
    cfg = make_config()
    row = make_row(target="signatory owner")
    r = _score(row, "   signatory owner  \n", cfg, fake_tokenizer)
    assert r["exact_match"] is True
    assert r["prefix_match"] is False


# --------------------------- over-generation ------------------------------- #
def test_over_generation_is_prefix_not_exact(make_config, make_row, fake_tokenizer):
    cfg = make_config()
    row = make_row(target="signatory owner")
    r = _score(row, "signatory owner, observer bob", cfg, fake_tokenizer)
    assert r["exact_match"] is False
    assert r["prefix_match"] is True  # predicted starts with target + extra


def test_disjointness_invariant_holds_for_many_cases(make_config, make_row, fake_tokenizer):
    cfg = make_config()
    for completion in ["signatory owner", "signatory owner extra", "totally different", ""]:
        row = make_row(target="signatory owner")
        r = _score(row, completion, cfg, fake_tokenizer)
        # The core invariant: exact and over-generation are mutually exclusive.
        assert not (r["exact_match"] and r["prefix_match"])


# --------------------------- mismatch -------------------------------------- #
def test_mismatch(make_config, make_row, fake_tokenizer):
    cfg = make_config()
    row = make_row(target="signatory owner")
    r = _score(row, "observer bob", cfg, fake_tokenizer)
    assert r["exact_match"] is False
    assert r["prefix_match"] is False


def test_empty_prediction_is_mismatch(make_config, make_row, fake_tokenizer):
    cfg = make_config()
    row = make_row(target="signatory owner")
    r = _score(row, "", cfg, fake_tokenizer)
    assert r["exact_match"] is False
    assert r["prefix_match"] is False


# --------------------------- <FILL> extraction ----------------------------- #
def test_fill_tags_are_extracted(make_config, make_row, fake_tokenizer):
    cfg = make_config()
    row = make_row(target="signatory owner")
    r = _score(row, "noise <FILL>signatory owner</FILL> trailing", cfg, fake_tokenizer)
    assert r["predicted"] == "signatory owner"
    assert r["exact_match"] is True


# --------------------------- token accounting ------------------------------ #
def test_usage_tokens_used_when_present(make_config, make_row, fake_tokenizer):
    cfg = make_config()
    row = make_row(target="x")

    class Usage:
        prompt_tokens = 11
        completion_tokens = 3

    r = _score(row, "x", cfg, fake_tokenizer, usage=Usage())
    assert r["input_tokens"] == 11
    assert r["output_tokens"] == 3


def test_tokenizer_fallback_when_no_usage(make_config, make_row, fake_tokenizer):
    cfg = make_config()
    row = make_row(target="a b c")
    r = _score(row, "a b c", cfg, fake_tokenizer, usage=None)
    # FakeTokenizer encodes by whitespace split: raw_text "a b c" -> 3 tokens.
    assert r["output_tokens"] == 3


# --------------------------- truncate_suffix_overlap ----------------------- #
def test_truncate_suffix_overlap_trims_bleed():
    suffix = "X" * 40  # >= default min_overlap (30)
    predicted = "answer" + suffix
    assert truncate_suffix_overlap(predicted, suffix) == "answer"


def test_truncate_suffix_overlap_noop_when_short_suffix():
    suffix = "short"  # < min_overlap
    predicted = "answer" + suffix
    assert truncate_suffix_overlap(predicted, suffix) == predicted
