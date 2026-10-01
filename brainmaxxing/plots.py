"""Figures. Each function returns a matplotlib Figure; scripts decide where to save."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"control": "#6b7280", "meditator": "#0f766e",
          "thinking": "#b45309", "meditation": "#1d4ed8"}


def simulation_recovery(summary: pd.DataFrame) -> plt.Figure:
    """Cohen's d per measure per scenario: shows naive band power conflating the two mechanisms."""
    measures = ["naive_gamma_power", "periodic_gamma_power", "exponent", "offset"]
    scen = list(summary.scenario.unique())
    fig, ax = plt.subplots(figsize=(8, 3.6))
    w = 0.8 / len(measures)
    for i, m in enumerate(measures):
        d = [summary[(summary.scenario == s) & (summary.measure == m)]["cohens_d"].values[0] for s in scen]
        ax.bar(np.arange(len(scen)) + i * w, d, w, label=m.replace("_", " "))
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(np.arange(len(scen)) + w * (len(measures) - 1) / 2)
    ax.set_xticklabels([s.replace("_", "\n") for s in scen], fontsize=8)
    ax.set_ylabel("Cohen's d (meditation - thinking)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Simulation: what each measure reports under known ground truth", fontsize=10)
    fig.tight_layout()
    return fig


def group_spectra(freqs: np.ndarray, spectra: pd.DataFrame, fit_range=None) -> plt.Figure:
    """Mean log-PSD by meditator x condition. `spectra` columns: meditator, condition, psd (array)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for (med, cond), df in spectra.groupby(["meditator", "condition"]):
        arr = np.log10(np.stack(df["psd"].values))
        lab = f"{'meditator' if med else 'control'} / {cond}"
        color = COLORS["meditator" if med else "control"]
        ls = "-" if cond == "meditation" else "--"
        ax.plot(freqs, arr.mean(0), color=color, ls=ls, label=lab, lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("log10 power")
    if fit_range:
        ax.axvspan(*fit_range, color="#e5e7eb", alpha=0.5, lw=0)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def interaction_plot(sm: pd.DataFrame, dv: str) -> plt.Figure:
    """2x2 means with subject lines; `sm` from stats.subject_means."""
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    for med, df in sm.groupby("meditator"):
        color = COLORS["meditator" if med else "control"]
        for _, r in df.iterrows():
            ax.plot([0, 1], [r["thinking"], r["meditation"]], color=color, alpha=0.12, lw=0.8)
        ax.errorbar([0, 1], [df["thinking"].mean(), df["meditation"].mean()],
                    yerr=[df["thinking"].sem(), df["meditation"].sem()], color=color, lw=2.2,
                    marker="o", label=f"{'meditators' if med else 'controls'} (n={len(df)})")
    ax.set_xticks([0, 1], ["thinking", "meditation"])
    ax.set_ylabel(dv.replace("_", " "))
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def specification_curve(spec: pd.DataFrame, term: str, dv: str) -> plt.Figure:
    """Effect estimates across specifications, sorted. `spec` has one row per spec x term x dv."""
    d = spec[(spec.term == term) & (spec.dv == dv)].sort_values("estimate").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    sig = d["p"] < 0.05
    ax.errorbar(d.index, d["estimate"], yerr=[d["estimate"] - d["ci_low"], d["ci_high"] - d["estimate"]],
                fmt="none", ecolor="#9ca3af", lw=0.7)
    ax.scatter(d.index, d["estimate"], c=np.where(sig, "#0f766e", "#9ca3af"), s=14, zorder=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("specification (sorted)")
    ax.set_ylabel(f"{term} effect on {dv}")
    ax.set_title(f"{int(sig.sum())}/{len(d)} specifications significant at p<.05", fontsize=9)
    fig.tight_layout()
    return fig
