"""Pure helpers from serving.py (the GPU/vLLM paths themselves are out of CI scope)."""
import json

import pytest

from daml_fim_benchmark.serving import (
    _looks_like_adapter_dir,
    _looks_like_hf_repo_id,
    _looks_like_model_dir,
    _parse_model_size_b,
    _read_base_model_from_adapter,
    _sanitize_lora_module_name,
    resolve_vllm_launch_port,
)
from daml_fim_benchmark.config import EvaluationConfig


def test_looks_like_model_dir(tmp_path):
    assert not _looks_like_model_dir(tmp_path)  # no config.json
    (tmp_path / "config.json").write_text("{}")
    assert not _looks_like_model_dir(tmp_path)  # config but no weights
    (tmp_path / "model.safetensors").write_text("")
    assert _looks_like_model_dir(tmp_path)


def test_looks_like_adapter_dir(tmp_path):
    assert not _looks_like_adapter_dir(tmp_path)
    (tmp_path / "adapter_config.json").write_text("{}")
    assert _looks_like_adapter_dir(tmp_path)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Qwen/Qwen2.5-Coder-7B", True),
        ("org/model", True),
        ("./local/path", False),
        ("/abs/path", False),
        ("~/home/path", False),
        ("", False),
    ],
)
def test_looks_like_hf_repo_id(name, expected):
    assert _looks_like_hf_repo_id(name) is expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Qwen2.5-Coder-7B", 7.0),
        ("model-13b", 13.0),
        ("granite-3.5b-base", 3.5),
        ("no-size-here", None),
    ],
)
def test_parse_model_size_b(name, expected):
    assert _parse_model_size_b(name) == expected


def test_sanitize_lora_module_name():
    assert _sanitize_lora_module_name("my model/v2") == "my-model-v2"
    assert _sanitize_lora_module_name("") == "lora_adapter"
    assert _sanitize_lora_module_name("ok.name-1") == "ok.name-1"


def test_read_base_model_from_adapter(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-Coder-7B"}))
    assert _read_base_model_from_adapter(tmp_path) == "Qwen/Qwen2.5-Coder-7B"


def test_read_base_model_from_adapter_missing_field(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({}))
    with pytest.raises(RuntimeError):
        _read_base_model_from_adapter(tmp_path)


def test_resolve_vllm_launch_port():
    assert resolve_vllm_launch_port(EvaluationConfig(vllm_launch_port=9999)) == 9999
    assert resolve_vllm_launch_port(EvaluationConfig(vllm_base_url="http://localhost:8001/v1")) == 8001
