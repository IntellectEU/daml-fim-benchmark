"""End-to-end scoring pipeline with a mocked provider (no GPU).

Drives several rows through the real scoring function with a fake vLLM client,
then aggregates with the real summary functions — exercising the full
score -> aggregate path that produces a run's headline numbers.
"""
from daml_fim_benchmark.evaluate import (
    evaluate_single_row_vllm_prepared,
    get_repo_summary,
    get_stats,
)

from conftest import FakeVllmClient


def _run_rows(specs, config, tokenizer):
    """specs: list of (repo, target, completion). Returns scored result dicts."""
    results = []
    for i, (repo, target, completion) in enumerate(specs):
        row = {
            "id": f"row-{i}",
            "file_path": f"{repo}/M.daml",
            "prefix": "PRE",
            "suffix": "SUF",
            "target": target,
            "repo": repo,
            "category": "signatory",
            "dataset_source": "syntax",
        }
        client = FakeVllmClient(completion)
        results.append(
            evaluate_single_row_vllm_prepared(
                row=row, prompt="p", formatted_prompt="fp", max_tokens=64,
                config=config, vllm_client=client, tokenizer=tokenizer,
            )
        )
    return results


def test_pipeline_scores_and_aggregates(make_config, fake_tokenizer):
    cfg = make_config()
    specs = [
        ("daml-finance", "aaa", "aaa"),          # exact
        ("daml-finance", "bbb", "bbb extra"),    # over-generation
        ("splice", "ccc", "ccc"),                # exact
        ("splice", "ddd", "wrong"),              # mismatch
    ]
    results = _run_rows(specs, cfg, fake_tokenizer)

    accuracy, prefix_rate, _, _ = get_stats(results)
    assert accuracy == 0.5         # 2/4 exact
    assert prefix_rate == 0.25     # 1/4 over-generation

    summary = get_repo_summary(results, in_domain_repos={"daml-finance"})
    repo = {x["repo"]: x for x in summary["repo_scores"]}
    assert repo["daml-finance"]["exact_match_rate"] == 0.5
    assert repo["splice"]["exact_match_rate"] == 0.5
    # daml-finance in-domain (0.5), splice out-of-domain (0.5)
    assert summary["in_domain_repo_average"] == 0.5
    assert summary["out_of_domain_repo_average"] == 0.5


def test_pipeline_result_rows_have_expected_shape(make_config, fake_tokenizer):
    cfg = make_config()
    results = _run_rows([("canton", "x", "x")], cfg, fake_tokenizer)
    r = results[0]
    for key in ("id", "file_path", "prefix", "suffix", "target", "predicted",
                "exact_match", "prefix_match", "prompt_type"):
        assert key in r
    assert r["exact_match"] is True
