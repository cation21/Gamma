"""Dataset access: download from OpenNeuro, read participants, load one block as an MNE Raw."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Non-EEG channels present in the Biosemi BDF files (8 external + GSR/resp/temp/status).
_MISC_PREFIXES = ("EXG", "GSR", "Erg", "Resp", "Plet", "Temp", "Status")


def download(cfg: dict, subjects: list[str] | None = None) -> Path:
    """Download the four task BDF files (+ sidecars) for the requested subjects.

    Skips `sourcedata/` entirely. `subjects` like ["sub-001", "sub-002"]; None = all 98.
    Roughly 600 MB per subject.
    """
    import openneuro

    root = Path(cfg["dataset"]["bids_root"])
    root.mkdir(parents=True, exist_ok=True)
    include = ["participants.tsv", "participants.json", "dataset_description.json", "README", "CHANGES"]
    if subjects is None:
        include.append("sub-*")
    else:
        include.extend(subjects)
    openneuro.download(
        dataset=cfg["dataset"]["id"],
        tag=cfg["dataset"]["snapshot"],
        target_dir=root,
        include=include,
        exclude=["sourcedata"],
    )
    return root


def truncated_blocks(cfg: dict, subject: str, tolerance_s: float = 5.0) -> list[str]:
    """Blocks whose BDF on disk is shorter than the sidecar's RecordingDuration (partial download).

    Reads only the BDF header. A partial file left by an interrupted download is otherwise
    indistinguishable from a complete one to the rest of the pipeline.
    """
    import json

    out = []
    for task in available_blocks(cfg, subject):
        bdf = block_path(cfg, subject, task)
        side = bdf.with_suffix(".json")
        try:
            expected = json.loads(side.read_text()).get("RecordingDuration") if side.exists() else None
            actual = mne.io.read_raw_bdf(bdf, preload=False, verbose="ERROR").times[-1]
        except Exception:
            out.append(task)
            continue
        if expected is not None and actual < expected - tolerance_s:
            out.append(task)
    return out


def download_with_retry(cfg: dict, subjects: list[str], attempts: int = 8, first_wait_s: float = 30.0) -> bool:
    """openneuro-py raises on any failed file or metadata timeout; the network at night is flaky.
    Retry with exponential backoff (30 s ... ~1 h total). Returns False if every attempt failed."""
    import time

    wait = first_wait_s
    for i in range(attempts):
        try:
            download(cfg, subjects=subjects)
            return True
        except Exception as exc:
            log.warning("download of %s failed (attempt %d/%d): %s", subjects, i + 1, attempts, str(exc)[:120])
            if i == attempts - 1:
                return False
            time.sleep(wait)
            wait = min(wait * 2, 900)
    return False


def ensure_complete(cfg: dict, subject: str, retries: int = 2) -> list[str]:
    """Download the subject if absent; re-download any truncated block. Returns blocks that are
    still missing or truncated (the caller decides whether to skip the subject)."""
    if not available_blocks(cfg, subject):
        if not download_with_retry(cfg, [subject]):
            return list(cfg["dataset"]["blocks"])
    bad = truncated_blocks(cfg, subject)
    for _ in range(retries):
        if not bad:
            break
        log.warning("%s: truncated download for %s, re-downloading", subject, bad)
        for task in bad:
            block_path(cfg, subject, task).unlink(missing_ok=True)
        if not download_with_retry(cfg, [subject]):
            break
        bad = truncated_blocks(cfg, subject)
    return bad


def load_participants(cfg: dict) -> pd.DataFrame:
    """participants.tsv with derived columns: meditator (bool), tradition, practice years."""
    root = Path(cfg["dataset"]["bids_root"])
    df = pd.read_csv(root / "participants.tsv", sep="\t", na_values=["n/a"])
    df["group"] = df["group"].str.lower()
    df["meditator"] = df["group"] != "ctr"
    df["tradition"] = df["group"].map(
        {"ctr": "control", "htr": "himalayan", "sny": "isha", "vip": "vipassana", "tm": "tm"}
    )
    # 'tm' (n=4, sub-056..059) is not one of the three traditions in the published design and is
    # not documented in participants.json; excluded from the primary analysis via stats.exclude_groups.
    df["in_primary"] = ~df["group"].isin(["tm"])
    df["years_of_practice"] = pd.to_numeric(df["years_of_practice"], errors="coerce")
    # participants.tsv spells one entry "thought"; normalise to the two documented levels
    df["first_session"] = df["first_session"].replace({"thought": "thinking"})
    # participants.tsv spells one entry "thought"; normalise to the two documented levels
    df["first_session"] = df["first_session"].str.lower().str.strip().replace({"thought": "thinking", "think": "thinking",
                                                                                "med": "meditation"})
    df["subject"] = df["participant_id"]
    return df


def list_subjects(cfg: dict) -> list[str]:
    root = Path(cfg["dataset"]["bids_root"])
    return sorted(p.name for p in root.glob("sub-*") if p.is_dir())


def block_path(cfg: dict, subject: str, task: str) -> Path:
    root = Path(cfg["dataset"]["bids_root"])
    return root / subject / "eeg" / f"{subject}_task-{task}_eeg.bdf"


def available_blocks(cfg: dict, subject: str) -> list[str]:
    """Blocks whose raw BDF is on disk (input side)."""
    return [t for t in cfg["dataset"]["blocks"] if block_path(cfg, subject, t).exists()]


def processed_subjects(cfg: dict) -> list[str]:
    """Subjects with at least one QC record in the derivatives (output side)."""
    root = Path(cfg["dataset"]["derivatives"])
    return sorted(p.name for p in root.glob("sub-*") if any(p.glob("*_qc.json")))


def processed_blocks(cfg: dict, subject: str) -> list[str]:
    """Blocks with a QC record in the derivatives, whether or not the raw file still exists."""
    root = Path(cfg["dataset"]["derivatives"]) / subject
    return [t for t in cfg["dataset"]["blocks"] if (root / f"{subject}_task-{t}_qc.json").exists()]


def load_raw(cfg: dict, subject: str, task: str, preload: bool = True) -> mne.io.Raw:
    """Read one block's BDF, keep the 64 scalp channels, attach the Biosemi64 montage.

    Biosemi records against CMS/DRL; the data are not referenced until preprocess.rereference().
    """
    path = block_path(cfg, subject, task)
    raw = mne.io.read_raw_bdf(path, preload=preload, verbose="ERROR")
    misc = [ch for ch in raw.ch_names if ch.startswith(_MISC_PREFIXES)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw.set_channel_types({ch: "misc" for ch in misc})
    raw.pick("eeg")
    montage = mne.channels.make_standard_montage(cfg["preprocess"]["montage"])
    raw.set_montage(montage, on_missing="warn")
    raw.info["line_freq"] = cfg["preprocess"]["line_freq_hz"]
    return raw


def block_metadata(cfg: dict, task: str) -> dict:
    return dict(cfg["dataset"]["blocks"][task])


def roi_picks(ch_names: list[str], roi_channels: list[str]) -> np.ndarray:
    """Indices of ROI channels that actually exist in ch_names (montage naming may drift)."""
    return np.array([ch_names.index(c) for c in roi_channels if c in ch_names], dtype=int)
