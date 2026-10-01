"""Stage 6: figures for the poster and paper, from the tables written by stages 3-5.

Writes to figures/:
  group_spectra.png        mean log-PSD, meditator x condition, primary ROI
  interaction_<dv>.png     2x2 plots for exponent, periodic and naive gamma power
  spec_curve_<term>_<dv>.png  specification curves (if stats_multiverse.csv exists)
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainmaxxing.cli import base_parser, deriv_path, setup  # noqa: E402
from brainmaxxing.data import processed_blocks, block_metadata, load_participants  # noqa: E402
from brainmaxxing.plots import group_spectra, interaction_plot, specification_curve  # noqa: E402
from brainmaxxing.spectra import clean_spectrum, roi_spectrum  # noqa: E402
from brainmaxxing.stats import subject_means  # noqa: E402

log = logging.getLogger("06_figures")


def main():
    args = base_parser(__doc__).parse_args()
    cfg, subjects = setup(args, source="derivatives")
    tables = Path(cfg["dataset"]["tables"])
    figdir = Path(__file__).resolve().parents[1] / "figures"
    figdir.mkdir(exist_ok=True)
    roi = cfg["stats"]["primary_roi"]
    chans = cfg["spectra"]["rois"][roi]
    participants = load_participants(cfg).set_index("subject")
    excluded = set(cfg["stats"].get("exclude_groups", []))
    subjects = [s for s in subjects if s in participants.index and participants.loc[s, "group"] not in excluded]

    # --- group spectra ---
    rows, freqs = [], None
    for sub in subjects:
        for task in processed_blocks(cfg, sub):
            path = deriv_path(cfg, sub, task, "desc-clean-average_psd")
            if not path.exists():
                continue
            z = np.load(path)
            freqs = z["freqs"]
            rows.append({"meditator": bool(participants.loc[sub, "meditator"]),
                         "condition": block_metadata(cfg, task)["condition"],
                         "psd": roi_spectrum(clean_spectrum(freqs, z["psd"], cfg), list(z["ch_names"]), chans)})
    if rows:
        fig = group_spectra(freqs, pd.DataFrame(rows), cfg["parameterize"]["specparam"]["fit_range_hz"])
        fig.savefig(figdir / "group_spectra.png", dpi=200)

    # --- 2x2 interaction plots ---
    fits = pd.read_csv(tables / "fits_primary.csv")
    df = fits[(fits.roi == roi) & (fits.fit_ok == True) & (~fits.group.isin(excluded))]  # noqa: E712
    for dv in ["exponent", "offset", "periodic_gamma_power", "naive_gamma_power"]:
        interaction_plot(subject_means(df, dv), dv).savefig(figdir / f"interaction_{dv}.png", dpi=200)

    # --- specification curves ---
    mv_path = tables / "stats_multiverse.csv"
    if mv_path.exists():
        mv = pd.read_csv(mv_path)
        for term, tag in [("meditator", "trait"), ("condition[T.meditation]", "state"),
                          ("meditator:condition[T.meditation]", "interaction")]:
            for dv in ["exponent", "periodic_gamma_power", "naive_gamma_power"]:
                if ((mv.term == term) & (mv.dv == dv)).any():
                    specification_curve(mv, term, dv).savefig(figdir / f"spec_curve_{tag}_{dv}.png", dpi=200)
    log.info("figures written to %s", figdir)


if __name__ == "__main__":
    main()
