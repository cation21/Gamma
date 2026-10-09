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
from statsmodels.stats.multitest import multipletests

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


KEY_TERMS = ["meditator", "condition[T.meditation]", "meditator:condition[T.meditation]"]


def adjust_pvalues(mixed: pd.DataFrame, family: list[str] = PRIMARY_DVS) -> pd.DataFrame:
    """Add Benjamini-Hochberg FDR and Holm-adjusted p-values for the key model terms.

    The family is the primary measures that were fitted, corrected separately for each term
    (trait, state, interaction). IRASA measures are cross-checks of the specparam exponent and
    periodic power, so they are left out of the family and get no adjusted p.
    """
    out = mixed.copy()
    out["p_fdr_bh"] = np.nan
    out["p_holm"] = np.nan
    sel = out.dv.isin(family) & out.term.isin(KEY_TERMS)
    for _, idx in out[sel].groupby("term").groups.items():
        p = out.loc[idx, "p"].to_numpy()
        out.loc[idx, "p_fdr_bh"] = multipletests(p, method="fdr_bh")[1]
        out.loc[idx, "p_holm"] = multipletests(p, method="holm")[1]
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


def _years_table(df: pd.DataFrame, dv: str) -> pd.DataFrame:
    """Meditators with known years of practice: per-subject level, state effect and tradition dummies."""
    sm = subject_means(df, dv)
    sm = sm[sm.meditator].dropna(subset=["years_of_practice"]).copy()
    sm["avg"] = sm[["meditation", "thinking"]].mean(axis=1)
    for t in sorted(sm.tradition.unique())[1:]:
        sm[f"trad_{t}"] = (sm.tradition == t).astype(float)
    return sm


def practice_effects(df: pd.DataFrame, dv: str, cfg: dict) -> list[dict]:
    """Does the effect depend on years of practice? Meditators only; all tests exploratory.

    - years_x_state: mixed model dv ~ log2(years) * condition + first_session + tradition + (1|subject);
      the interaction asks whether the meditation - thinking difference changes with experience
      (per doubling of years of practice).
    - state_rho / state_rho_partial: Spearman of years with each subject's state effect
      (meditation - thinking), raw and partialled for tradition (years differ strongly by tradition).
    - level_rho_partial: as the existing dose-response, partialled for tradition.
    - long_vs_short: state effect in meditators above vs at/below the median years (Mann-Whitney U),
      and each subgroup's trait contrast against controls (Welch t, condition-averaged).
    """
    rows = []
    d = _prepare(df[df.meditator].dropna(subset=["years_of_practice"]))
    d["log2_years"] = np.log2(d["years_of_practice"])
    trad = " + C(tradition)" if d.tradition.nunique() > 1 else ""
    try:
        res = smf.mixedlm(f"{dv} ~ log2_years * condition + first_session{trad}", d,
                          groups=d[cfg["stats"]["groups"]]).fit(reml=False, method=["lbfgs"])
        ci = res.conf_int()
        for term, name in [("log2_years", "years_level"), ("log2_years:condition[T.meditation]", "years_x_state")]:
            rows.append({"dv": dv, "analysis": name, "estimate": float(res.params[term]),
                         "ci_low": float(ci.loc[term, 0]), "ci_high": float(ci.loc[term, 1]),
                         "p": float(res.pvalues[term]), "n": int(d.subject.nunique())})
    except np.linalg.LinAlgError:  # near-constant DV (periodic power is ~0 in most blocks)
        pass

    sm = _years_table(df, dv)
    covar = [c for c in sm if c.startswith("trad_")]
    st = sm.dropna(subset=["state_diff"])
    r = pg.corr(st["years_of_practice"], st["state_diff"], method="spearman")
    rows.append({"dv": dv, "analysis": "state_rho", "estimate": float(r["r"].iloc[0]),
                 "p": float(r["p_val"].iloc[0]), "n": int(r["n"].iloc[0])})
    for name, y, data in [("state_rho_partial", "state_diff", st), ("level_rho_partial", "avg", sm.dropna(subset=["avg"]))]:
        r = (pg.partial_corr(data, x="years_of_practice", y=y, covar=covar, method="spearman") if covar
             else pg.corr(data["years_of_practice"], data[y], method="spearman"))
        ci95 = r["CI95" if "CI95" in r else "CI95%"].iloc[0]
        rows.append({"dv": dv, "analysis": name, "estimate": float(r["r"].iloc[0]),
                     "ci_low": float(ci95[0]), "ci_high": float(ci95[1]),
                     "p": float(r["p_val"].iloc[0]), "n": int(r["n"].iloc[0])})

    med = st["years_of_practice"].median()
    long_, short = st[st.years_of_practice > med], st[st.years_of_practice <= med]
    u = pg.mwu(long_["state_diff"], short["state_diff"])
    rows.append({"dv": dv, "analysis": "long_vs_short_state", "estimate": float(long_.state_diff.mean() - short.state_diff.mean()),
                 "p": float(u["p_val"].iloc[0]), "n": len(st), "split_years": float(med),
                 "n_long": len(long_), "n_short": len(short)})
    ctrl = subject_means(df, dv)
    ctrl = ctrl[~ctrl.meditator][["meditation", "thinking"]].mean(axis=1).dropna()
    for name, grp in [("long_vs_controls_trait", long_), ("short_vs_controls_trait", short)]:
        a = grp[["meditation", "thinking"]].mean(axis=1).dropna()
        t = pg.ttest(a, ctrl, paired=False)
        rows.append({"dv": dv, "analysis": name, "estimate": float(a.mean() - ctrl.mean()),
                     "cohens_d": float(pg.compute_effsize(a, ctrl)), "p": float(t["p_val"].iloc[0]),
                     "n": len(a), "split_years": float(med)})
    return rows


def adjust_family(table: pd.DataFrame, by: str, family: list[str] = PRIMARY_DVS) -> pd.DataFrame:
    """BH-FDR and Holm across the primary measures, separately within each level of `by`."""
    out = table.copy()
    out["p_fdr_bh"] = np.nan
    out["p_holm"] = np.nan
    for _, idx in out[out.dv.isin(family)].groupby(by).groups.items():
        p = out.loc[idx, "p"].to_numpy()
        out.loc[idx, "p_fdr_bh"] = multipletests(p, method="fdr_bh")[1]
        out.loc[idx, "p_holm"] = multipletests(p, method="holm")[1]
    return out


def emg_confound(df: pd.DataFrame, dv: str, cfg: dict) -> pd.DataFrame:
    """Does the trait effect on `dv` survive adjustment for scalp-muscle proxies?"""
    base = mixed_2x2(df, dv, cfg)
    base["model"] = "unadjusted"
    adj = mixed_2x2(df, dv, cfg, covariates=["emg_proxy_logpower", "ica_muscle_variance_ratio"])
    adj["model"] = "emg_adjusted"
    return pd.concat([base, adj], ignore_index=True)
