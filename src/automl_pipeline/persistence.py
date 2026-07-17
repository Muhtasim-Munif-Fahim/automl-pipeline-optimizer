"""Model persistence: save/load trained models and pipeline artifacts."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def save_model(model: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)


def load_model(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def save_pipeline_artifacts(artifacts: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, artifact in artifacts.items():
        if isinstance(artifact, dict):
            (output_dir / f"{name}.json").write_text(
                json.dumps(artifact, indent=2, default=str), encoding="utf-8"
            )
        else:
            save_model(artifact, output_dir / f"{name}.pkl")


def load_pipeline_artifacts(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    artifacts: dict[str, Any] = {}
    for path in output_dir.glob("*.json"):
        artifacts[path.stem] = json.loads(path.read_text())
    for path in output_dir.glob("*.pkl"):
        artifacts[path.stem] = load_model(path)
    return artifacts
