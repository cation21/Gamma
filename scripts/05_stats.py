"""Stage 5: the state x trait models, equivalence tests, dose-response and EMG-confound checks.

Reads results/tables/fits_primary.csv (and fits_multiverse.csv if present). Writes:
  results/tables/stats_mixed_2x2.csv      mixed-model terms per DV, primary ROI
  results/tables/stats_tost.csv           equivalence tests for trait and state contrasts
  results/tables/stats_dose_response.csv  Spearman rho with years of practice
  results/tables/stats_emg_confound.csv   trait effect with/without EMG covariates
  results/tables/stats_multiverse.csv     mixed-model terms for every specification

Usage: python scripts/05_stats.py [--roi parieto_occipital]
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medgamma.cli import base_parser, setup  # noqa: E402
from medgamma.stats import (PRIMARY_DVS, dose_response, emg_confound, mixed_2x2,  # noqa: E402
                                state_tost, trait_tost)

log = logging.getLogger("05_stats")


def main():
    p = base_parser(__doc__)
    p.add_argument("--roi", default=None)
    args = p.parse_args()
    cfg, _ = setup(args)
    roi = args.roi or cfg["stats"]["primary_roi"]
    tables = Path(cfg["dataset"]["tables"])

    fits = pd.read_csv(tables / "fits_primary.csv")
    df = fits[(fits.roi == roi) & (fits.fit_ok == True)].copy()  # noqa: E712
    df = df[~df.group.isin(cfg["stats"].get("exclude_groups", []))]
    log.info("%d usable rows from %d subjects in ROI %s", len(df), df.subject.nunique(), roi)
    n_med, n_ctr = df[df.meditator].subject.nunique(), df[~df.meditator].subject.nunique()
    if min(n_med, n_ctr) < 4:
        sys.exit(f"need at least 4 meditators and 4 controls with usable fits; have {n_med}/{n_ctr}. "
                 "Run more subjects through stages 1-4 first.")
    dvs = PRIMARY_DVS + [c for c in ["irasa_exponent", "irasa_periodic_gamma_power"] if c in df]

    parts = []
    for dv in dvs:
        try:
            parts.append(mixed_2x2(df, dv, cfg))
        except Exception as exc:  # zero-variance DV, singular Hessian, non-convergence
            log.warning("mixed model for %s skipped: %s", dv, str(exc)[:120])
    mixed = pd.concat(parts, ignore_index=True)
    mixed.to_csv(tables / "stats_mixed_2x2.csv", index=False)
    key = mixed[mixed.term.isin(["meditator", "condition[T.meditation]",
                                 "meditator:condition[T.meditation]"])]
    print("\n=== Mixed model, ROI", roi, "===")
    print(key[["dv", "term", "estimate", "ci_low", "ci_high", "p"]].round(4).to_string(index=False))

    def safe(fn, *a):
        try:
            return fn(*a)
        except Exception as exc:
            log.warning("%s skipped: %s", fn.__name__, str(exc)[:100])
            return None
    tost = pd.DataFrame([r for r in [safe(trait_tost, df, dv, cfg) for dv in dvs] + [safe(state_tost, df, dv, cfg) for dv in dvs] if r])
    tost.to_csv(tables / "stats_tost.csv", index=False)

    dose = pd.DataFrame([r for r in [safe(dose_response, df, dv) for dv in dvs] if r])
    dose.to_csv(tables / "stats_dose_response.csv", index=False)
    print("\n=== Dose-response (Spearman with years of practice) ===")
    print(dose.round(4).to_string(index=False))

    if {"emg_proxy_logpower", "ica_muscle_variance_ratio"} <= set(df.columns):
        parts = [r for r in [safe(emg_confound, df, dv, cfg) for dv in
                             ["exponent", "naive_gamma_power", "periodic_gamma_power"]] if r is not None]
        if parts:
            emg = pd.concat(parts, ignore_index=True)
            emg.to_csv(tables / "stats_emg_confound.csv", index=False)
            print("\n=== Effects before/after EMG adjustment ===")
            print(emg[emg.term.isin(["meditator", "condition[T.meditation]",
                                     "meditator:condition[T.meditation]"])]
                  [["dv", "model", "term", "estimate", "ci_low", "ci_high", "p"]].round(4).to_string(index=False))

    mv_path = tables / "fits_multiverse.csv"
    if mv_path.exists():
        mv = pd.read_csv(mv_path)
        rows = []
        for spec_id, d in mv[mv.fit_ok == True].groupby("spec_id"):  # noqa: E712
            if d.subject.nunique() < 5:
                continue
            for dv in ["exponent", "periodic_gamma_power", "naive_gamma_power"]:
                try:
                    r = mixed_2x2(d, dv, cfg)
                except Exception as exc:
                    log.warning("%s %s failed: %s", spec_id, dv, exc)
                    continue
                r["spec_id"] = spec_id
                rows.append(r)
        if rows:
            pd.concat(rows, ignore_index=True).to_csv(tables / "stats_multiverse.csv", index=False)
            log.info("wrote stats_multiverse.csv")


if __name__ == "__main__":
    main()
