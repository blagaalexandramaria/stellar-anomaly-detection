"""Load machine-readable frozen publication registries shipped with the package."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any


def repository_root() -> Path:
    """Return the source-checkout root used for publication registry fallback."""
    return Path(__file__).resolve().parents[2]


def publication_registry(name: str) -> dict[str, Any]:
    """Load a packaged registry, falling back to its publication snapshot."""
    packaged = files("stellar_anomaly_detection").joinpath("data", "registries", name)
    if packaged.is_file():
        return json.loads(packaged.read_text())

    path = repository_root() / "publication" / "supplementary" / "registries" / name
    if path.is_file():
        return json.loads(path.read_text())
    raise FileNotFoundError(f"Required frozen publication registry is missing: {name}")
