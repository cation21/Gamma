"""One-pass processing of a block: preprocess -> spectra for every variant -> small outputs only.

Stages 02 and 03 keep the pre-ICA epochs on disk (77 MB per block) so variants can be rebuilt.
This module does the same work in memory and writes only what later stages read:
  <sub>_task-<task>_desc-<variant>_psd.npz   (4 variants, ~240 KB each)
  <sub>_task-<task>_desc-clean-average_irasa.npz
  <sub>_task-<task>_ica.fif, _qc.json
About 4.5 MB per subject instead of 300 MB. Used by scripts/07_stream.py and the Colab notebook.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from .cli import deriv_path
from .data import available_blocks, block_path, load_raw
from .preprocess import preprocess_block
from .spectra import build_variant, compute_psd, interpolate_line_noise, irasa_components

log = logging.getLogger(__name__)

ALL_VARIANTS = ["clean-average", "clean-csd", "raw-average", "raw-csd"]
DEFAULT_VARIANT = "clean-average"


def block_done(cfg: dict, sub: str, task: str) -> bool:
    return deriv_path(cfg, sub, task, "qc", ".json").exists()


def process_block(cfg: dict, sub: str, task: str, variants=ALL_VARIANTS, irasa_all: bool = False,
                  save_epochs: bool = False) -> str:
    """Preprocess one block and write PSD/IRASA/ICA/QC. Returns a one-line summary."""
    t0 = time.time()
    qc_path = deriv_path(cfg, sub, task, "qc", ".json")
    try:
        raw = load_raw(cfg, sub, task)
        side = block_path(cfg, sub, task).with_suffix(".json")
        expected = json.loads(side.read_text()).get("RecordingDuration") if side.exists() else None
        if expected and raw.times[-1] < expected - 5:
            raise RuntimeError(f"recording on disk is {raw.times[-1]:.0f}s but sidecar says {expected}s (truncated download)")
        epochs, ica, qc = preprocess_block(raw, cfg)
        del raw
    except Exception as exc:
        log.exception("%s %s FAILED: %s", sub, task, exc)
        qc_path.write_text(json.dumps({"subject": sub, "task": task, "failed": str(exc)}, indent=2))
        return f"{sub} {task} FAILED: {exc}"

    roi = cfg["spectra"]["rois"][cfg["stats"]["primary_roi"]]
    min_epochs = cfg["preprocess"]["epochs"].get("min_epochs", 30)
    qc["variants"] = {}
    for variant in variants:
        ica_clean, ref = variant.split("-")
        try:
            ep, vqc = build_variant(epochs, ica, ica_clean == "clean", ref, cfg)
        except Exception as exc:
            log.exception("%s %s %s FAILED: %s", sub, task, variant, exc)
            qc["variants"][variant] = {"failed": str(exc)}
            continue
        qc["variants"][variant] = vqc
        if vqc["n_epochs_kept"] < min_epochs:
            log.warning("%s %s %s: only %d epochs kept (< %d), variant skipped", sub, task, variant,
                        vqc["n_epochs_kept"], min_epochs)
            continue
        freqs, psd = compute_psd(ep, cfg)
        psd = interpolate_line_noise(freqs, psd, cfg["preprocess"]["line_freq_hz"],
                                     cfg["spectra"]["line_interp_halfwidth_hz"])
        np.savez_compressed(deriv_path(cfg, sub, task, f"desc-{variant}_psd"),
                            freqs=freqs, psd=psd, ch_names=np.array(ep.ch_names))
        if irasa_all or variant == DEFAULT_VARIANT:
            picks = [c for c in roi if c in ep.ch_names]
            f_ir, ap, osc, fit = irasa_components(ep, cfg, picks)
            np.savez_compressed(deriv_path(cfg, sub, task, f"desc-{variant}_irasa"),
                                freqs=f_ir, aperiodic=ap, oscillatory=osc, ch_names=np.array(picks),
                                fit_slope=fit["Slope"].values, fit_intercept=fit["Intercept"].values,
                                fit_r2=fit["R^2"].values)
    if save_epochs:
        epochs.save(deriv_path(cfg, sub, task, "desc-preica_epo", ".fif"), overwrite=True, verbose="ERROR")
    ica.save(deriv_path(cfg, sub, task, "ica", ".fif"), overwrite=True, verbose="ERROR")
    main = qc["variants"].get(DEFAULT_VARIANT, {})
    qc["n_epochs_kept"] = main.get("n_epochs_kept", 0)
    qc["epoch_reject_fraction"] = main.get("epoch_reject_fraction", 1.0)
    qc.update(subject=sub, task=task, seconds=round(time.time() - t0, 1))
    qc_path.write_text(json.dumps(qc, indent=2))
    kept = {v: q.get("n_epochs_kept", "x") for v, q in qc["variants"].items()}
    return (f"{sub} {task}: bads={qc['bad_channels']} ics_out={qc['ica_n_excluded']} "
            f"muscle={qc['ica_muscle_variance_ratio']:.3f} kept={kept} of {qc['n_epochs_total']} "
            f"({qc['seconds']:.0f}s)")


def delete_raw(cfg: dict, sub: str) -> int:
    """Remove the subject's BDF files (keep sidecars). Returns bytes freed."""
    freed = 0
    for task in available_blocks(cfg, sub):
        p = block_path(cfg, sub, task)
        freed += p.stat().st_size
        p.unlink()
    return freed


def subject_outputs_present(cfg: dict, sub: str, n_blocks: int = 4) -> bool:
    root = Path(cfg["dataset"]["derivatives"]) / sub
    return root.exists() and len(list(root.glob("*_qc.json"))) >= n_blocks
