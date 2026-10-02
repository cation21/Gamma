"""Statistics for the state x trait design.

Long table convention (one row per subject x block x ROI x specification):
  subject, meditator (bool), tradition, condition (meditation|thinking), run (1|2),
  first_session, years_of_practice, emg_proxy_logpower, ica_muscle_variance_ratio,
  + the measures from parameterize.FIT_COLUMNS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.formula.api as smf

PRIMARY_DVS = ["exponent", "offset", "periodic_gamma_power", "aperiodic_gamma_power",
               "naive_gamma_power"]


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["meditator"] = d["meditator"].astype(int)
    d["condition"] = pd.Categorical(d["condition"], categories=["thinking", "meditation"])
    d["first_session"] = pd.Categorical(d["first_session"], categories=["thinking", "meditation"])
    return d


def mixed_2x2(df: pd.DataFrame, dv: str, cfg: dict, covariates: list[str] | None = None
              ) -> pd.DataFrame:
    """Linear mixed model: dv ~ meditator * condition + first_session (+ covariates), random
    intercept per subject. `meditator` is the trait effect, `condition` the state effect,
    `meditator:condition` the interaction that tests whether Jerbi's and Ray's findings conflict.
    """
    d = _prepare(df).dropna(subset=[dv] + list(covariates or []))
    if d[dv].std() < 1e-6:
        raise ValueError(f"{dv} has no variance (SD={d[dv].std():.2e}); nothing to model")
    formula = cfg["stats"]["formula"].format(dv=dv)
    if covariates:
        formula += " + " + " + ".join(covariates)
    model = smf.mixedlm(formula, d, groups=d[cfg["stats"]["groups"]])
    res = model.fit(reml=False, method=["lbfgs"])
    ci = res.conf_int()
    out = pd.DataFrame({"term": res.params.index, "estimate": res.params.values,
                        "se": res.bse.values, "ci_low": ci[0].values, "ci_high": ci[1].values,
                        "z": res.tvalues.values, "p": res.pvalues.values})
    out.insert(0, "dv", dv)
    out["n_obs"] = int(res.nobs)
    out["n_subjects"] = d[cfg["stats"]["groups"]].nunique()
    return out


def subject_means(df: pd.DataFrame, dv: str) -> pd.DataFrame:
    """Average the two runs of each condition; also return the within-subject state difference."""
    g = df.groupby(["subject", "meditator", "tradition", "years_of_practice", "condition"],
                   dropna=False)[dv].mean().unstack("condition").reset_index()
    g["state_diff"] = g["meditation"] - g["thinking"]
    return g


def trait_tost(df: pd.DataFrame, dv: str, cfg: dict) -> dict:
    """Equivalence test on the trait contrast (meditators vs controls, condition-averaged).

    Bounds are +/- `equivalence_bounds_sd` pooled SDs, i.e. the smallest effect of interest.
    """
    sm = subject_means(df, dv)
    sm["avg"] = sm[["meditation", "thinking"]].mean(axis=1)
    a = sm.loc[sm.meditator, "avg"].dropna().values
    b = sm.loc[~sm.meditator, "avg"].dropna().values
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    bound = cfg["stats"]["equivalence_bounds_sd"] * sp
    t = pg.tost(a, b, bound=bound, paired=False)
    return {"dv": dv, "contrast": "trait", "bound": float(bound), "cohens_d": float(pg.compute_effsize(a, b)),
            "p_tost": float(t["pval"].iloc[0]), "n_meditators": len(a), "n_controls": len(b)}


def state_tost(df: pd.DataFrame, dv: str, cfg: dict) -> dict:
    sm = subject_means(df, dv).dropna(subset=["state_diff"])
    diff = sm["state_diff"].values
    bound = cfg["stats"]["equivalence_bounds_sd"] * diff.std(ddof=1)
    t = pg.tost(sm["meditation"].values, sm["thinking"].values, bound=bound, paired=True)
    return {"dv": dv, "contrast": "state", "bound": float(bound),
            "cohens_d": float(diff.mean() / diff.std(ddof=1)),
            "p_tost": float(t["pval"].iloc[0]), "n_subjects": len(diff)}


def dose_response(df: pd.DataFrame, dv: str) -> dict:
    """Spearman correlation of the condition-averaged measure with years of practice (meditators)."""
    sm = subject_means(df, dv)
    sm = sm[sm.meditator].dropna(subset=["years_of_practice"])
    sm["avg"] = sm[["meditation", "thinking"]].mean(axis=1)
    sm = sm.dropna(subset=["avg"])
    r = pg.corr(sm["years_of_practice"], sm["avg"], method="spearman")
    return {"dv": dv, "n": int(r["n"].iloc[0]), "rho": float(r["r"].iloc[0]),
            "p": float(r["p_val"].iloc[0])}


def emg_confound(df: pd.DataFrame, dv: str, cfg: dict) -> pd.DataFrame:
    """Does the trait effect on `dv` survive adjustment for scalp-muscle proxies?"""
    base = mixed_2x2(df, dv, cfg)
    base["model"] = "unadjusted"
    adj = mixed_2x2(df, dv, cfg, covariates=["emg_proxy_logpower", "ica_muscle_variance_ratio"])
    adj["model"] = "emg_adjusted"
    return pd.concat([base, adj], ignore_index=True)
