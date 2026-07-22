"""Pipeline configuration validation schema."""

from __future__ import annotations

from typing import Any

REQUIRED_PIPELINE_KEYS: set[str] = {"data", "target"}
OPTIONAL_PIPELINE_KEYS: dict[str, type] = {
    "test_size": float,
    "random_state": int,
    "cv_folds": int,
    "optimize": bool,
    "models": list,
    "impute_strategy": str,
    "scaling_method": str,
    "encoding_method": str,
}


def validate_pipeline_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_PIPELINE_KEYS - set(config.keys())
    if missing:
        errors.append(f"Missing required config keys: {', '.join(sorted(missing))}")
    for key, expected_type in OPTIONAL_PIPELINE_KEYS.items():
        if key in config and not isinstance(config[key], expected_type):
            errors.append(f"Config key '{key}' should be {expected_type.__name__}, got {type(config[key]).__name__}")
    if "test_size" in config and not 0 < config["test_size"] < 1:
        errors.append(f"test_size must be between 0 and 1, got {config.get('test_size')}")
    if "cv_folds" in config and config["cv_folds"] < 2:
        errors.append(f"cv_folds must be >= 2, got {config.get('cv_folds')}")
    return errors
