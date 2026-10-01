"""specparam (FOOOF) wrappers producing tidy per-spectrum rows.

Every fit yields the same columns whether or not a gamma peak was found, so results from
different subjects/blocks/specifications concatenate cleanly.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from specparam import SpectralModel

log = logging.getLogger(__name__)

FIT_COLUMNS = [
    "offset", "knee", "exponent", "r2", "mae", "n_peaks", "fit_ok",
    "gamma_peak_cf", "gamma_peak_pw", "gamma_peak_bw", "gamma_peak_present", "gamma_peak_pinned",
    "periodic_gamma_power", "aperiodic_gamma_power", "naive_gamma_power",
    "fit_fmin", "fit_fmax", "aperiodic_mode",
]


def make_model(sp: dict, aperiodic_mode: str | None = None) -> SpectralModel:
    return SpectralModel(
        aperiodic_mode=aperiodic_mode or sp["aperiodic_mode"],
        peak_width_limits=list(sp["peak_width_limits"]),
        max_n_peaks=sp["max_n_peaks"],
        min_peak_height=sp["min_peak_height"],
        peak_threshold=sp["peak_threshold"],
        verbose=False,
    )


def _nan_row(fit_range, aperiodic_mode) -> dict:
    row = {c: np.nan for c in FIT_COLUMNS}
    row.update(fit_ok=False, gamma_peak_present=False, gamma_peak_pinned=False, n_peaks=0,
               fit_fmin=fit_range[0], fit_fmax=fit_range[1], aperiodic_mode=aperiodic_mode)
    return row


def fit_spectrum(freqs: np.ndarray, psd: np.ndarray, cfg: dict,
                 fit_range: tuple[float, float] | None = None,
                 aperiodic_mode: str | None = None) -> dict:
    """Fit one spectrum. Returns a flat dict (see FIT_COLUMNS).

    periodic_gamma_power: mean of the peak (flattened) component, log10 units, over the gamma band.
    aperiodic_gamma_power: mean of the aperiodic fit, log10 units, over the gamma band.
    naive_gamma_power: mean of the raw log10 spectrum over the gamma band (the conventional measure).
    """
    sp = cfg["parameterize"]["specparam"]
    band = cfg["parameterize"]["gamma_band_hz"]
    fit_range = tuple(fit_range or sp["fit_range_hz"])
    aperiodic_mode = aperiodic_mode or sp["aperiodic_mode"]
    row = _nan_row(fit_range, aperiodic_mode)

    m = make_model(sp, aperiodic_mode)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(np.asarray(freqs, float), np.asarray(psd, float), list(fit_range))
    except Exception as exc:  # specparam raises on degenerate spectra
        log.warning("specparam fit failed on %s-%s Hz: %s", *fit_range, exc)
        return row
    if not m.results.has_model:
        return row

    ap = m.get_params("aperiodic")
    if aperiodic_mode == "knee":
        row["offset"], row["knee"], row["exponent"] = map(float, ap)
    else:
        row["offset"], row["exponent"] = map(float, ap)
    row["r2"] = float(m.get_metrics("gof_rsquared"))
    row["mae"] = float(m.get_metrics("error_mae"))
    lo_e, hi_e = sp.get("exponent_range", [0.0, 4.0])
    row["fit_ok"] = bool(row["r2"] >= sp["min_r2"] and lo_e <= row["exponent"] <= hi_e)

    peaks = np.atleast_2d(m.get_params("peak"))
    peaks = peaks[~np.isnan(peaks).any(axis=1)] if peaks.size else peaks
    row["n_peaks"] = int(len(peaks))
    in_band = peaks[(peaks[:, 0] >= band[0]) & (peaks[:, 0] <= band[1])] if len(peaks) else peaks
    if len(in_band):
        best = in_band[np.argmax(in_band[:, 1])]
        row["gamma_peak_cf"], row["gamma_peak_pw"], row["gamma_peak_bw"] = map(float, best)
        lo, hi = sp["peak_width_limits"]
        # A bandwidth sitting on a fitting limit is a spectral line (min) or the fitter absorbing
        # curvature at the range edge (max), not an oscillation. Recorded, but not counted.
        pinned = bool(best[2] <= lo * 1.01 or best[2] >= hi * 0.99)
        row["gamma_peak_pinned"] = pinned
        row["gamma_peak_present"] = not pinned

    f = m.data.freqs
    bm = (f >= band[0]) & (f <= band[1])
    if bm.any():
        row["periodic_gamma_power"] = float(m.results.model.get_component("peak", "log")[bm].mean())
        row["aperiodic_gamma_power"] = float(m.results.model.get_component("aperiodic", "log")[bm].mean())
    fm = (np.asarray(freqs) >= band[0]) & (np.asarray(freqs) <= band[1])
    row["naive_gamma_power"] = float(np.log10(np.asarray(psd)[fm]).mean()) if fm.any() else np.nan
    return row


def fit_many(freqs: np.ndarray, psds: np.ndarray, cfg: dict, labels: list[str],
             fit_range=None, aperiodic_mode=None, label_name: str = "channel") -> pd.DataFrame:
    """Fit each row of `psds` (n x n_freq); returns a DataFrame with one row per label."""
    rows = []
    for lab, psd in zip(labels, psds):
        r = fit_spectrum(freqs, psd, cfg, fit_range, aperiodic_mode)
        r[label_name] = lab
        rows.append(r)
    return pd.DataFrame(rows)


def irasa_row(freqs: np.ndarray, ap: np.ndarray, osc: np.ndarray, fit: pd.DataFrame,
              cfg: dict) -> dict:
    """Summarise IRASA output for one (already ROI-pooled or single-channel) spectrum set.

    Slope from YASA is the power-law exponent with sign (negative); we report exponent = -slope
    to match specparam's convention.
    """
    band = cfg["parameterize"]["gamma_band_hz"]
    bm = (freqs >= band[0]) & (freqs <= band[1])
    ap_m = 10 ** np.log10(np.atleast_2d(ap)).mean(axis=0)
    osc_m = np.atleast_2d(osc).mean(axis=0)
    total = ap_m + osc_m
    return {
        "irasa_exponent": float(-fit["Slope"].mean()),
        "irasa_intercept": float(fit["Intercept"].mean()),
        "irasa_r2": float(fit["R^2"].mean()),
        # oscillatory residual relative to aperiodic, log units, within the gamma band
        "irasa_periodic_gamma_power": float((np.log10(np.clip(total[bm], 1e-30, None))
                                             - np.log10(ap_m[bm])).mean()) if bm.any() else np.nan,
        "irasa_aperiodic_gamma_power": float(np.log10(ap_m[bm]).mean()) if bm.any() else np.nan,
    }
