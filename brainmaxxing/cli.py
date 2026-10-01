"""Shared command-line plumbing for the numbered scripts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import ensure_dirs, load_config
from .data import list_subjects, processed_subjects


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--subjects", nargs="*", default=None,
                   help="e.g. sub-001 sub-002; default = every subject present on disk")
    p.add_argument("--n-first", type=int, default=None, help="use only the first N subjects")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def setup(args, source: str = "raw") -> tuple[dict, list[str]]:
    """source='raw' lists subjects with raw data on disk; 'derivatives' lists processed subjects."""
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    subjects = args.subjects or (processed_subjects(cfg) if source == "derivatives" else list_subjects(cfg))
    if args.n_first:
        subjects = subjects[: args.n_first]
    return cfg, subjects


def deriv_path(cfg: dict, subject: str, task: str, suffix: str, ext: str = ".npz") -> Path:
    d = Path(cfg["dataset"]["derivatives"]) / subject
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{subject}_task-{task}_{suffix}{ext}"
