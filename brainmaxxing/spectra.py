"""Power spectra from epochs: Welch PSD, line-noise interpolation, ROI pooling, IRASA."""

from __future__ import annotations

import logging

import mne
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def build_variant(epochs: mne.Epochs, ica, ica_clean: bool, reference: str, cfg: dict
                  ) -> tuple[mne.Epochs, dict]:
    """Apply the multiverse preprocessing axes to a copy of the pre-ICA, average-referenced epochs.

    Order: ICA component removal -> peak-to-peak epoch rejection (in microvolts, so before CSD)
    -> re-referencing. Returns the epochs and a QC dict with epoch counts.
    """
    from .preprocess import reject_epochs

    ep = epochs.copy()
    if ica_clean and ica is not None and ica.exclude:
        ica.apply(ep, verbose="ERROR")
    ep, qc = reject_epochs(ep, cfg)
    if len(ep) == 0:
        return ep, qc
    if reference == "csd":
        ep = mne.preprocessing.compute_current_source_density(ep, verbose="ERROR")
    elif reference != "average":
        raise ValueError(reference)
    return ep, qc


def compute_psd(epochs: mne.Epochs, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD per channel, aggregated across epochs (median by default). Returns (freqs, psd)."""
    s = cfg["spectra"]
    spec = epochs.compute_psd(method=s["method"], fmin=s["fmin"], fmax=s["fmax"], n_fft=s["n_fft"],
                              n_per_seg=s["n_fft"], window="hann", verbose="ERROR")
    data = spec.get_data()  # (n_epochs, n_ch, n_freq)
    agg = np.median if s["average"] == "median" else np.mean
    return spec.freqs, agg(data, axis=0)


def interpolate_lines(freqs: np.ndarray, psd: np.ndarray, lines: list[float], halfwidth: float
                      ) -> np.ndarray:
    """Replace bins within +/- halfwidth of each listed frequency by log-linear interpolation."""
    out = np.array(psd, dtype=float, copy=True)
    logp = np.log10(out)
    mask = np.zeros_like(freqs, dtype=bool)
    for h in lines:
        mask |= np.abs(freqs - h) <= halfwidth
    if not mask.any():
        return out
    good = ~mask
    if logp.ndim == 1:
        logp[mask] = np.interp(freqs[mask], freqs[good], logp[good])
    else:
        for i in range(logp.shape[0]):
            logp[i, mask] = np.interp(freqs[mask], freqs[good], logp[i, good])
    return 10 ** logp


def clean_spectrum(freqs: np.ndarray, psd: np.ndarray, cfg: dict) -> np.ndarray:
    """Mains harmonics plus the equipment lines listed in the config. Idempotent."""
    s = cfg["spectra"]
    line = cfg["preprocess"]["line_freq_hz"]
    mains = list(np.arange(line, freqs[-1] + line, line))
    out = interpolate_lines(freqs, psd, mains, s["line_interp_halfwidth_hz"])
    if s.get("extra_line_hz"):
        out = interpolate_lines(freqs, out, list(s["extra_line_hz"]), s.get("extra_line_halfwidth_hz", 1.0))
    return out


def interpolate_line_noise(freqs: np.ndarray, psd: np.ndarray, line_freq: float,
                           halfwidth: float) -> np.ndarray:
    """Replace bins near each mains harmonic by log-linear interpolation from the flanks.

    Preferred over notch filtering because a notch leaves a trough that biases aperiodic fits.
    `psd` is (..., n_freq); works for 1-D and 2-D.
    """
    out = np.array(psd, dtype=float, copy=True)
    logp = np.log10(out)
    harmonics = np.arange(line_freq, freqs[-1] + line_freq, line_freq)
    mask = np.zeros_like(freqs, dtype=bool)
    for h in harmonics:
        mask |= np.abs(freqs - h) <= halfwidth
    if not mask.any():
        return out
    good = ~mask
    if logp.ndim == 1:
        logp[mask] = np.interp(freqs[mask], freqs[good], logp[good])
    else:
        for i in range(logp.shape[0]):
            logp[i, mask] = np.interp(freqs[mask], freqs[good], logp[i, good])
    return 10 ** logp


def roi_spectrum(psd: np.ndarray, ch_names: list[str], roi_channels: list[str]) -> np.ndarray:
    """Geometric mean across ROI channels (mean in log space), so one hot channel cannot dominate."""
    idx = [ch_names.index(c) for c in roi_channels if c in ch_names]
    if not idx:
        raise ValueError("no ROI channels present")
    return 10 ** np.log10(psd[idx]).mean(axis=0)


def naive_band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:
    """Mean log10 power in a band: the conventional measure the decomposition is compared against."""
    m = (freqs >= band[0]) & (freqs <= band[1])
    return float(np.log10(psd[..., m]).mean())


def irasa_components(epochs: mne.Epochs, cfg: dict, picks: list[str] | None = None
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """IRASA (Wen & Liu 2016) via YASA on the kept epochs concatenated in time.

    Returns freqs, aperiodic PSD, oscillatory PSD (each n_ch x n_freq) and the per-channel
    power-law fit table (Intercept, Slope, R^2). Epoch boundaries introduce small
    discontinuities; the median Welch average in YASA is robust to them.
    """
    import yasa

    ir = cfg["parameterize"]["irasa"]
    ep = epochs.copy()
    if picks:
        ep.pick(picks)
    data = ep.get_data()  # (n_epochs, n_ch, n_times)
    cont = np.concatenate(list(data), axis=-1)
    freqs, ap, osc, fit = yasa.irasa(cont, sf=ep.info["sfreq"], ch_names=ep.ch_names,
                                     band=tuple(ir["band_hz"]), hset=ir["hset"],
                                     win_sec=ir["win_sec"], return_fit=True, verbose=False)
    return freqs, ap, osc, fit
