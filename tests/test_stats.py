"""Stats stage on a synthetic long table: checks the models run and recover planted effects."""

import numpy as np
import pandas as pd
import pytest

from medgamma.config import load_config
from medgamma.stats import (dose_response, emg_confound, mixed_2x2, state_tost, subject_means,
                                trait_tost)


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/default.yaml")


@pytest.fixture(scope="module")
def table():
    """40 subjects x 4 blocks. Planted: trait +0.20 on exponent, state -0.10, no interaction."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(40):
        med = i < 24
        u = rng.normal(0, 0.15)
        years = rng.uniform(3, 40) if med else 0.0
        first = "meditation" if i % 2 == 0 else "thinking"
        emg = rng.normal(-13.5, 0.3)
        for task, cond, run in [("med1breath", "meditation", 1), ("think1", "thinking", 1),
                                ("med2", "meditation", 2), ("think2", "thinking", 2)]:
            exp = 1.6 + 0.2 * med - 0.1 * (cond == "meditation") + 0.01 * years + u + rng.normal(0, 0.05)
            rows.append(dict(subject=f"sub-{i:03d}", task=task, condition=cond, run=run,
                             meditator=med, tradition="x" if med else "control",
                             first_session=first, years_of_practice=years,
                             exponent=exp, offset=-11 + u + rng.normal(0, 0.1),
                             periodic_gamma_power=abs(rng.normal(0, 0.01)),
                             aperiodic_gamma_power=-14 + rng.normal(0, 0.1),
                             naive_gamma_power=-14 + 0.3 * med + rng.normal(0, 0.1),
                             emg_proxy_logpower=emg + rng.normal(0, 0.05),
                             ica_muscle_variance_ratio=abs(rng.normal(0.05, 0.02)),
                             fit_ok=True, roi="parieto_occipital"))
    return pd.DataFrame(rows)


def test_mixed_2x2_recovers_planted_effects(cfg, table):
    res = mixed_2x2(table, "exponent", cfg).set_index("term")
    assert abs(res.loc["condition[T.meditation]", "estimate"] + 0.10) < 0.03
    assert res.loc["condition[T.meditation]", "p"] < 0.001
    assert res.loc["meditator", "estimate"] > 0.15  # +0.2 trait plus the years term
    assert res.loc["meditator:condition[T.meditation]", "p"] > 0.05
    assert res["n_subjects"].iloc[0] == 40


def test_subject_means_and_tost(cfg, table):
    sm = subject_means(table, "exponent")
    assert len(sm) == 40 and {"meditation", "thinking", "state_diff"} <= set(sm.columns)
    st = state_tost(table, "exponent", cfg)
    assert st["cohens_d"] < -1  # strong planted state effect: not equivalent to zero
    assert st["p_tost"] > 0.05
    tr_null = trait_tost(table, "periodic_gamma_power", cfg)   # no planted trait effect
    tr_real = trait_tost(table, "naive_gamma_power", cfg)      # planted +0.3 trait effect
    assert abs(tr_null["cohens_d"]) < 0.5 and tr_real["cohens_d"] > 1
    assert tr_null["p_tost"] < tr_real["p_tost"]


def test_dose_response(table):
    r = dose_response(table, "exponent")
    assert r["n"] == 24 and r["rho"] > 0.3


def test_emg_confound_runs(cfg, table):
    out = emg_confound(table, "exponent", cfg)
    assert set(out.model) == {"unadjusted", "emg_adjusted"}
    assert (out.term == "emg_proxy_logpower").sum() == 1
