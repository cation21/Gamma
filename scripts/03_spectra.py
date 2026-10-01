"""Stage 3: power spectra for every block under each preprocessing variant.

Variants = ica_clean {true,false} x reference {average,csd}. Per block and variant, writes
  <sub>_task-<task>_desc-<variant>_psd.npz     freqs, psd (n_ch x n_freq, line noise interpolated), ch_names
  <sub>_task-<task>_desc-<variant>_irasa.npz   IRASA aperiodic/oscillatory PSDs for the primary ROI channels

IRASA runs only for the default variant (clean-average) unless --irasa-all is given.

Usage: python scripts/03_spectra.py [--subjects ...] [--variants clean-average ...] [--irasa-all] [--n-jobs 4]
"""

import logging
import sys
import time
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainmaxxing.cli import base_parser, deriv_path, setup  # noqa: E402
from brainmaxxing.data import available_blocks  # noqa: E402
from brainmaxxing.spectra import (build_variant, compute_psd, interpolate_line_noise,  # noqa: E402
                                  irasa_components)

log = logging.getLogger("03_spectra")

ALL_VARIANTS = ["clean-average", "clean-csd", "raw-average", "raw-csd"]
DEFAULT_VARIANT = "clean-average"


def parse_variant(name: str) -> tuple[bool, str]:
    clean, ref = name.split("-")
    return clean == "clean", ref


def run_block(cfg: dict, sub: str, task: str, variants: list[str], irasa_all: bool,
              overwrite: bool) -> str:
    roi = cfg["spectra"]["rois"][cfg["stats"]["primary_roi"]]
    epo_path = deriv_path(cfg, sub, task, "desc-preica_epo", ".fif")
    if not epo_path.exists():
        return f"{sub} {task}: no preprocessed epochs, run 02 first"
    epochs = ica = None
    done = []
    for variant in variants:
        out = deriv_path(cfg, sub, task, f"desc-{variant}_psd")
        if out.exists() and not overwrite:
            continue
        if epochs is None:
            epochs = mne.read_epochs(epo_path, preload=True, verbose="ERROR")
            ica = mne.preprocessing.read_ica(deriv_path(cfg, sub, task, "ica", ".fif"), verbose="ERROR")
        t0 = time.time()
        ica_clean, ref = parse_variant(variant)
        ep, vqc = build_variant(epochs, ica, ica_clean, ref, cfg)
        if vqc["n_epochs_kept"] < cfg["preprocess"]["epochs"].get("min_epochs", 30):
            done.append(f"{variant} (skipped, {vqc['n_epochs_kept']} epochs)")
            continue
        freqs, psd = compute_psd(ep, cfg)
        psd = interpolate_line_noise(freqs, psd, cfg["preprocess"]["line_freq_hz"],
                                     cfg["spectra"]["line_interp_halfwidth_hz"])
        np.savez_compressed(out, freqs=freqs, psd=psd, ch_names=np.array(ep.ch_names))
        if irasa_all or variant == DEFAULT_VARIANT:
            picks = [c for c in roi if c in ep.ch_names]
            f_ir, ap, osc, fit = irasa_components(ep, cfg, picks)
            np.savez_compressed(deriv_path(cfg, sub, task, f"desc-{variant}_irasa"),
                                freqs=f_ir, aperiodic=ap, oscillatory=osc, ch_names=np.array(picks),
                                fit_slope=fit["Slope"].values, fit_intercept=fit["Intercept"].values,
                                fit_r2=fit["R^2"].values)
        done.append(f"{variant} ({time.time() - t0:.0f}s)")
    return f"{sub} {task}: " + (", ".join(done) if done else "up to date")


def main():
    p = base_parser(__doc__)
    p.add_argument("--variants", nargs="*", default=ALL_VARIANTS)
    p.add_argument("--irasa-all", action="store_true")
    p.add_argument("--n-jobs", type=int, default=1)
    args = p.parse_args()
    cfg, subjects = setup(args)
    todo = [(sub, task) for sub in subjects for task in available_blocks(cfg, sub)]
    log.info("%d blocks with %d worker(s)", len(todo), args.n_jobs)
    if args.n_jobs == 1:
        for sub, task in todo:
            log.info(run_block(cfg, sub, task, args.variants, args.irasa_all, args.overwrite))
    else:
        from joblib import Parallel, delayed
        for msg in Parallel(n_jobs=args.n_jobs, return_as="generator")(
                delayed(run_block)(cfg, sub, task, args.variants, args.irasa_all, args.overwrite)
                for sub, task in todo):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
