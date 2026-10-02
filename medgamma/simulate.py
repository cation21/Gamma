"""Ground-truth simulation: can the pipeline tell a real gamma peak from a broadband shift?

Paired design, mirroring the within-subject state contrast in ds003969. Each simulated subject
has base parameters (with subject-level jitter). Its 'thinking' spectrum uses those parameters;
its 'meditation' spectrum applies the scenario's modification. Both get independent noise.
The identical `fit_spectrum` used on real data is run on every spectrum, and the per-subject
meditation - thinking difference is summarised per measure.

Naive band power should report an increase in every non-null scenario; only the decomposition
should say which kind. The between-subject (trait) analogue is the same computation with more
noise, so the paired version is the cleaner validity check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from specparam.sim import sim_power_spectrum

from .parameterize import fit_spectrum

MEASURES = ["naive_gamma_power", "periodic_gamma_power", "aperiodic_gamma_power",
            "exponent", "offset"]


def _base_params(rng, s: dict) -> dict:
    """Subject-level parameters: population values plus between-subject spread."""
    bsd = s["between_subject_sd"]
    return {
        "offset": s["base_aperiodic"]["offset"] + rng.normal(0, bsd["offset"]),
        "exponent": s["base_aperiodic"]["exponent"] + rng.normal(0, bsd["exponent"]),
        "peaks": [[cf + rng.normal(0, 0.5), max(pw + rng.normal(0, 0.05), 0.02), sd]
                  for cf, pw, sd in s["base_peaks"]],
    }


def _apply_scenario(base: dict, scenario: dict, s: dict, rng) -> tuple[dict, dict]:
    """Block-level parameters: subject values, block-to-block drift, plus scenario deltas."""
    wsd = s["within_subject_sd"]
    offset = base["offset"] + rng.normal(0, wsd["offset"]) + scenario.get("offset_delta", 0.0)
    exponent = base["exponent"] + rng.normal(0, wsd["exponent"]) + scenario.get("exponent_delta", 0.0)
    peaks = [[cf, max(pw + rng.normal(0, 0.03), 0.02), sd] for cf, pw, sd in base["peaks"]]
    if "peak" in scenario:
        cf, pw, sd = scenario["peak"]
        peaks.append([cf + rng.normal(0, 2.0), pw, sd])
    return {"fixed": [offset, exponent]}, {"gaussian": peaks}


def simulate_subject(cfg: dict, scenario: dict, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns freqs, thinking spectrum, meditation spectrum for one simulated subject."""
    s = cfg["simulation"]
    base = _base_params(rng, s)
    ap0, pk0 = _apply_scenario(base, {}, s, rng)
    ap1, pk1 = _apply_scenario(base, scenario, s, rng)
    freqs, think = sim_power_spectrum(s["freq_range_hz"], ap0, pk0, nlv=s["noise_level"],
                                      freq_res=s["freq_res_hz"])
    _, med = sim_power_spectrum(s["freq_range_hz"], ap1, pk1, nlv=s["noise_level"],
                                freq_res=s["freq_res_hz"])
    return freqs, think, med


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Independent-groups d, b minus a."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((b.mean() - a.mean()) / sp) if sp > 0 else np.nan


def paired_d(diff: np.ndarray) -> float:
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    sd = diff.std(ddof=1)
    if sd == 0:  # e.g. periodic power identically zero in both conditions: no effect, not undefined
        return 0.0 if diff.mean() == 0 else np.nan
    return float(diff.mean() / sd)


def run_simulation(cfg: dict, fit_range=None, aperiodic_mode=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-spectrum fits, per-scenario summary of paired differences)."""
    s = cfg["simulation"]
    rng = np.random.default_rng(s["seed"])
    fits = []
    for name, scenario in s["scenarios"].items():
        for i in range(s["n_per_group"]):
            freqs, think, med = simulate_subject(cfg, scenario, rng)
            for cond, psd in (("thinking", think), ("meditation", med)):
                row = fit_spectrum(freqs, psd, cfg, fit_range, aperiodic_mode)
                row.update(scenario=name, condition=cond, sim_subject=i)
                fits.append(row)
    fits = pd.DataFrame(fits)

    summary = []
    for name, df in fits.groupby("scenario", sort=False):
        wide = df.pivot(index="sim_subject", columns="condition")
        for meas in MEASURES:
            a = wide[(meas, "thinking")].astype(float).values
            b = wide[(meas, "meditation")].astype(float).values
            diff = b - a
            t, p = stats.ttest_rel(b, a, nan_policy="omit")
            summary.append({"scenario": name, "measure": meas, "thinking_mean": np.nanmean(a),
                            "meditation_mean": np.nanmean(b), "mean_diff": np.nanmean(diff),
                            "cohens_d": paired_d(diff), "t": float(t), "p": float(p)})
        det_a = wide[("gamma_peak_present", "thinking")].astype(float).mean()
        det_b = wide[("gamma_peak_present", "meditation")].astype(float).mean()
        summary.append({"scenario": name, "measure": "gamma_peak_detection_rate",
                        "thinking_mean": det_a, "meditation_mean": det_b, "mean_diff": det_b - det_a,
                        "cohens_d": np.nan, "t": np.nan, "p": np.nan})
    return fits, pd.DataFrame(summary)
