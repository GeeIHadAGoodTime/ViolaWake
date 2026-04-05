"""Unit tests for violawake_sdk._constants."""

from __future__ import annotations

from violawake_sdk import _constants


def test_get_feature_config_returns_expected_keys() -> None:
    config = _constants.get_feature_config()

    assert set(config) == {
        "feature_type",
        "n_mels",
        "n_fft",
        "hop_length",
        "win_length",
        "f_min",
        "f_max",
        "sample_rate",
        "clip_samples",
        "use_pcen",
        "pcen_gain",
        "pcen_bias",
        "pcen_power",
        "pcen_time_constant",
        "pcen_eps",
    }


def test_get_feature_config_matches_module_constants() -> None:
    config = _constants.get_feature_config()

    assert config["feature_type"] == _constants.FEATURE_TYPE
    assert config["n_mels"] == _constants.N_MELS_MEL
    assert config["n_fft"] == _constants.N_FFT_MEL
    assert config["hop_length"] == _constants.HOP_LENGTH_MEL
    assert config["win_length"] == _constants.WIN_LENGTH_MEL
    assert config["f_min"] == _constants.F_MIN
    assert config["f_max"] == _constants.F_MAX
    assert config["sample_rate"] == _constants.SAMPLE_RATE
    assert config["clip_samples"] == _constants.CLIP_SAMPLES
    assert config["use_pcen"] == _constants.USE_PCEN
