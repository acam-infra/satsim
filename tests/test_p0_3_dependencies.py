"""P0-3: numerical scripts must be runnable from the project environment."""
from __future__ import annotations

import importlib
import tomllib
from pathlib import Path


def _project() -> dict:
    root = Path(__file__).resolve().parents[1]
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))


def test_numpy_scipy_are_declared_and_importable() -> None:
    deps = {d.split(">=")[0].split("==")[0].split("[")[0].lower()
            for d in _project()["project"]["dependencies"]}
    assert {"numpy", "scipy"} <= deps
    assert importlib.import_module("numpy") is not None
    assert importlib.import_module("scipy.optimize") is not None
