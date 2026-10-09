"""Configuration loading. One YAML in, one nested dict out, with resolved paths."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path.home() / "GammaData"


def _expand(value: str) -> Path:
    """Expand ${VARS} and ~ in a config path; GAMMA_DATA defaults to ~/GammaData."""
    env = {**os.environ}
    env.setdefault("GAMMA_DATA", str(DEFAULT_DATA_DIR))
    for k, v in env.items():
        value = value.replace(f"${{{k}}}", v)
    return Path(os.path.expanduser(value))


def load_config(path: str | Path = "configs/default.yaml", overrides: dict | None = None) -> dict:
    """Load a YAML config, resolve dataset paths against the repo root, apply overrides.

    `overrides` is a flat dict of dotted keys, e.g. {"parameterize.specparam.fit_range_hz": [3, 95]}.
    """
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key in ("bids_root", "derivatives", "tables", "qc"):
        p = _expand(str(cfg["dataset"][key]))
        cfg["dataset"][key] = str(p if p.is_absolute() else REPO_ROOT / p)

    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Return a deep copy of cfg with dotted-key overrides applied."""
    cfg = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return cfg


def ensure_dirs(cfg: dict) -> None:
    for key in ("derivatives", "tables", "qc"):
        Path(cfg["dataset"][key]).mkdir(parents=True, exist_ok=True)
