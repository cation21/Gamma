"""Config loading: data paths come from GAMMA_DATA, not a hard-coded machine path."""

from pathlib import Path

from medgamma.config import DEFAULT_DATA_DIR, load_config


def test_data_paths_follow_gamma_data(monkeypatch, tmp_path):
    monkeypatch.setenv("GAMMA_DATA", str(tmp_path))
    cfg = load_config()
    assert Path(cfg["dataset"]["bids_root"]) == tmp_path / "ds003969"
    assert Path(cfg["dataset"]["derivatives"]) == tmp_path / "derivatives"


def test_data_paths_default_to_home(monkeypatch):
    monkeypatch.delenv("GAMMA_DATA", raising=False)
    cfg = load_config()
    assert Path(cfg["dataset"]["bids_root"]) == DEFAULT_DATA_DIR / "ds003969"
