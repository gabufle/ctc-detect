"""Tests for adapter-config gating in the detection module.

These directly exercise the gate that fails loudly when the model directory
is not a proper PEFT/LoRA adapter — the fine-tuned CTC model is a LoRA
adapter, so silently loading anything else (e.g. the base Geneformer, which
scores near random) would be a subtle, dangerous failure.
"""

import json

import pytest

from ctcdetect.detect import _validate_adapter_config


def _write_adapter_config(model_dir, payload):
    """Write an adapter_config.json into model_dir and return the dir."""
    (model_dir / "adapter_config.json").write_text(json.dumps(payload))
    return model_dir


def test_validate_adapter_config_missing_file(tmp_path):
    """No adapter_config.json at all → SystemExit with a clear message."""
    with pytest.raises(SystemExit):
        _validate_adapter_config(tmp_path)


def test_validate_adapter_config_malformed_json(tmp_path):
    """adapter_config.json that is not valid JSON → SystemExit."""
    (tmp_path / "adapter_config.json").write_text("{ this is not json")
    with pytest.raises(SystemExit):
        _validate_adapter_config(tmp_path)


def test_validate_adapter_config_not_lora(tmp_path):
    """A non-LoRA PEFT config (wrong peft_type) → SystemExit."""
    _write_adapter_config(
        tmp_path,
        {"peft_type": "PREFIX_TUNING", "base_model_name_or_path": "ctheodoris/Geneformer-V1-10M"},
    )
    with pytest.raises(SystemExit):
        _validate_adapter_config(tmp_path)


def test_validate_adapter_config_missing_base_model(tmp_path):
    """A LoRA config with no base_model_name_or_path → SystemExit."""
    _write_adapter_config(tmp_path, {"peft_type": "LORA"})
    with pytest.raises(SystemExit):
        _validate_adapter_config(tmp_path)


def test_validate_adapter_config_valid(tmp_path):
    """A well-formed LoRA adapter config is accepted and returned."""
    _write_adapter_config(
        tmp_path,
        {"peft_type": "LORA", "base_model_name_or_path": "ctheodoris/Geneformer-V1-10M"},
    )
    cfg = _validate_adapter_config(tmp_path)
    assert cfg["base_model_name_or_path"] == "ctheodoris/Geneformer-V1-10M"


def test_validate_adapter_config_lora_case_insensitive(tmp_path):
    """peft_type matching is case-insensitive ('lora' is still LoRA)."""
    _write_adapter_config(
        tmp_path,
        {"peft_type": "lora", "base_model_name_or_path": "ctheodoris/Geneformer-V1-10M"},
    )
    assert _validate_adapter_config(tmp_path)["peft_type"] == "lora"


def test_validate_adapter_config_message_is_clear(tmp_path, capfd):
    """The failure message names the problem so the user can act on it.

    Uses capfd (fd-level) rather than capsys because the module-level rich
    Console captured sys.stdout at import time.
    """
    with pytest.raises(SystemExit):
        _validate_adapter_config(tmp_path)
    out = capfd.readouterr().out
    assert "adapter" in out.lower()
