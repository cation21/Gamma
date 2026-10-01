"""Unit tests that run without the dataset: config, spectra helpers, specparam wrapper, simulation."""

import numpy as np
import pytest
from specparam.sim import sim_power_spectrum

from brainmaxxing.config import apply_overrides, load_config
from brainmaxxing.parameterize import FIT_COLUMNS, fit_spectrum
from brainmaxxing.simulate import cohens_d, run_simulation
from brainmaxxing.spectra import interpolate_line_noise, naive_band_power, roi_spectrum


@pytest.fixture(scope="module")
def cfg():
    c = load_config("configs/default.yaml")
    return apply_overrides(c, {"simulation.n_per_group": 24})


def test_overrides_are_deep_copies(cfg):
    c2 = apply_overrides(cfg, {"parameterize.specparam.fit_range_hz": [3, 95]})
    assert cfg["parameterize"]["specparam"]["fit_range_hz"] == [30, 120]
    assert c2["parameterize"]["specparam"]["fit_range_hz"] == [3, 95]


def test_line_interpolation_removes_spike():
    freqs = np.arange(1, 200.5, 0.5)
    psd = 10.0 / freqs ** 1.5
    spiked = psd.copy()
    spiked[np.isclose(freqs, 50)] *= 50
    spiked[np.isclose(freqs, 100)] *= 20
    fixed = interpolate_line_noise(freqs, spiked, 50, 2.0)
    assert np.allclose(np.log10(fixed), np.log10(psd), atol=0.02)
    # 2-D input works too
    fixed2 = interpolate_line_noise(freqs, np.stack([spiked, spiked]), 50, 2.0)
    assert fixed2.shape == (2, len(freqs))


def test_roi_spectrum_geometric_mean():
    psd = np.array([[1.0, 10.0], [100.0, 10.0]])
    out = roi_spectrum(psd, ["A", "B"], ["A", "B", "missing"])
    assert np.allclose(out, [10.0, 10.0])
    with pytest.raises(ValueError):
        roi_spectrum(psd, ["A", "B"], ["nope"])


def test_fit_spectrum_recovers_known_parameters(cfg):
    freqs, psd = sim_power_spectrum([3, 200], {"fixed": [1.0, 1.5]},
                                    {"gaussian": [[10, 0.6, 2], [80, 0.3, 8]]}, nlv=0.005)
    row = fit_spectrum(freqs, psd, cfg)
    assert set(FIT_COLUMNS) <= set(row)
    assert row["fit_ok"]
    assert abs(row["exponent"] - 1.5) < 0.15
    assert row["gamma_peak_present"]
    assert abs(row["gamma_peak_cf"] - 80) < 5
    assert row["periodic_gamma_power"] > 0.05
    assert np.isclose(row["naive_gamma_power"], naive_band_power(freqs, psd, (60, 110)))


def test_truncated_peak_biases_exponent(cfg):
    """Documents why the primary fit range is 30-120 and not 30-95: a broad gamma peak whose
    upper tail is cut by the range edge drags the exponent down."""
    freqs, psd = sim_power_spectrum([3, 200], {"fixed": [1.0, 1.5]},
                                    {"gaussian": [[10, 0.6, 2], [80, 0.3, 8]]}, nlv=0.005)
    narrow = fit_spectrum(freqs, psd, cfg, fit_range=(30, 95))["exponent"]
    wide = fit_spectrum(freqs, psd, cfg, fit_range=(30, 120))["exponent"]
    assert narrow < 1.5 - 0.15
    assert abs(wide - 1.5) < 0.1


def test_fit_spectrum_knee_mode(cfg):
    freqs, psd = sim_power_spectrum([3, 200], {"knee": [1.0, 100.0, 2.0]}, {"gaussian": [[10, 0.6, 2]]})
    row = fit_spectrum(freqs, psd, cfg, fit_range=(3, 190), aperiodic_mode="knee")
    assert row["fit_ok"]
    assert not np.isnan(row["knee"])
    assert not row["gamma_peak_present"]


def test_fit_spectrum_degenerate_input_returns_nan_row(cfg):
    freqs = np.arange(1, 200.5, 0.5)
    row = fit_spectrum(freqs, np.zeros_like(freqs), cfg)
    assert row["fit_ok"] is False or row["fit_ok"] == False  # noqa: E712
    assert np.isnan(row["exponent"])


def test_simulation_attributes_effects_correctly(cfg):
    fits, summary = run_simulation(cfg)
    d = summary.set_index(["scenario", "measure"])["cohens_d"]
    det = summary.set_index(["scenario", "measure"])["mean_diff"]
    # naive band power rises in every non-null scenario: it cannot tell them apart
    for s in ["true_periodic_gamma", "aperiodic_offset_shift", "aperiodic_exponent_shift", "both"]:
        assert d[(s, "naive_gamma_power")] > 0.4, s
    # a real peak shows up as periodic power and as a detected peak; a broadband shift does not
    assert d[("true_periodic_gamma", "periodic_gamma_power")] > 1.5
    assert det[("true_periodic_gamma", "gamma_peak_detection_rate")] > 0.8
    assert abs(d[("aperiodic_offset_shift", "periodic_gamma_power")]) < 0.5
    assert abs(det[("aperiodic_offset_shift", "gamma_peak_detection_rate")]) < 0.2
    # an exponent shift is seen as an exponent shift, and an offset shift as an offset shift
    assert d[("aperiodic_exponent_shift", "exponent")] < -1.5
    assert d[("aperiodic_offset_shift", "offset")] > 1.5
    assert abs(d[("aperiodic_offset_shift", "exponent")]) < 0.5
    # nothing is reported when nothing changed
    for m in ["naive_gamma_power", "periodic_gamma_power", "exponent", "offset"]:
        assert abs(d[("no_change", m)]) < 0.5, m


def test_cohens_d_sign():
    assert cohens_d(np.zeros(10), np.ones(10) + np.random.default_rng(0).normal(0, 0.1, 10)) > 5
