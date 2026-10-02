"""Preprocessing of one block: filter, bad channels, reference, ICA + ICLabel, epoching, EMG proxy.

Design: the expensive step (ICA) runs once per block. The ICA solution and labels are saved so
downstream stages can build spectra with or without component rejection (multiverse axis
`ica_clean`) without refitting.
"""

from __future__ import annotations

import logging

import mne
import numpy as np
from mne.preprocessing import ICA

log = logging.getLogger(__name__)


def detect_bad_channels(raw: mne.io.Raw, zscore: float, ptp_uv: float = 300.0,
                        seg_s: float = 2.0) -> list[str]:
    """Flag channels that are (a) log-variance outliers relative to the cap (robust z), (b) flat,
    or (c) have a median peak-to-peak over `seg_s` windows above `ptp_uv` -- an absolute criterion
    that still fires when a whole cluster of channels is bad and inflates the MAD."""
    data = raw.get_data(picks="eeg")
    names = np.array(raw.copy().pick("eeg").ch_names)
    logvar = np.log(np.var(data, axis=1) + 1e-30)
    med = np.median(logvar)
    mad = np.median(np.abs(logvar - med)) * 1.4826 + 1e-12
    z = (logvar - med) / mad
    flat = np.var(data, axis=1) < 1e-14
    n_seg = int(seg_s * raw.info["sfreq"])
    n = data.shape[1] // n_seg * n_seg
    ptp = np.ptp(data[:, :n].reshape(data.shape[0], -1, n_seg), axis=2)
    huge = np.median(ptp, axis=1) * 1e6 > ptp_uv
    bad = set(names[np.abs(z) > zscore]) | set(names[flat]) | set(names[huge])
    return sorted(str(b) for b in bad)


def basic_clean(raw: mne.io.Raw, cfg: dict) -> tuple[mne.io.Raw, dict]:
    """Crop edges, resample, high-pass, (optional notch), detect + interpolate bad channels."""
    p = cfg["preprocess"]
    qc: dict = {}
    tmax = raw.times[-1] - p["crop_edges_s"]
    raw.crop(tmin=p["crop_edges_s"], tmax=tmax)
    raw.resample(p["resample_hz"], npad="auto")
    raw.filter(l_freq=p["highpass_hz"], h_freq=p["lowpass_hz"], fir_design="firwin", verbose="ERROR")
    if p["notch"]:
        harmonics = np.arange(p["line_freq_hz"], raw.info["sfreq"] / 2, p["line_freq_hz"])
        raw.notch_filter(harmonics, verbose="ERROR")
    bads = detect_bad_channels(raw, p["bad_channel_zscore"], p.get("bad_channel_ptp_uv", 300.0),
                               cfg["preprocess"]["epochs"]["length_s"])
    qc["bad_channels"] = bads
    if len(bads) > p.get("max_bad_channels", 16):
        raise RuntimeError(f"{len(bads)} bad channels ({bads}); block rejected")
    raw.info["bads"] = bads
    if bads:
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    return raw, qc


def emg_proxy(raw: mne.io.Raw, cfg: dict) -> dict:
    """Broadband high-frequency log-power at temporal sites, on data NOT yet cleaned by ICA.

    This is the covariate for the 'steeper slope = less scalp muscle' test. Computed after
    referencing so it is on the same footing as the spectra it is compared against.
    """
    e = cfg["preprocess"]["emg_proxy"]
    picks = [ch for ch in e["channels"] if ch in raw.ch_names]
    psd = raw.compute_psd(method="welch", picks=picks, fmin=e["band_hz"][0], fmax=e["band_hz"][1],
                          n_fft=int(raw.info["sfreq"] * 2), verbose="ERROR")
    power = psd.get_data()  # (n_ch, n_freq)
    logp = np.log10(power.mean(axis=1))
    return {"emg_proxy_channels": picks, "emg_proxy_logpower": float(logp.mean()),
            "emg_proxy_per_channel": dict(zip(picks, map(float, logp)))}


def rereference(raw: mne.io.Raw, scheme: str) -> mne.io.Raw:
    if scheme == "average":
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    elif scheme == "csd":
        raw = mne.preprocessing.compute_current_source_density(raw, verbose="ERROR")
    else:
        raise ValueError(f"unknown reference scheme {scheme!r}")
    return raw


def fit_ica(raw: mne.io.Raw, cfg: dict) -> tuple[ICA, dict]:
    """Extended-infomax ICA on a 1-100 Hz, average-referenced copy, labelled with ICLabel.

    Returns the ICA object with `.exclude` populated, and a QC dict including the fraction of
    variance explained by muscle components (second EMG proxy).
    """
    from mne_icalabel import label_components

    p = cfg["preprocess"]["ica"]
    # ICLabel wants 1-100 Hz, average reference. Fitting on a 256 Hz copy halves the cost without
    # touching the data the ICA is later applied to.
    fit_raw = raw.copy().filter(l_freq=1.0, h_freq=100.0, fir_design="firwin", verbose="ERROR")
    fit_raw.resample(p.get("fit_sfreq_hz", 256), npad="auto")
    ica = ICA(n_components=p["n_components"], method=p["method"],
              fit_params=dict(extended=True), random_state=p["random_state"], max_iter="auto")
    ica.fit(fit_raw, verbose="ERROR")
    labels = label_components(fit_raw, ica, method="iclabel")
    names, proba = labels["labels"], np.asarray(labels["y_pred_proba"])
    exclude = [i for i, (lab, pr) in enumerate(zip(names, proba))
               if lab in p["reject_labels"] and pr >= p["label_threshold"]]
    ica.exclude = exclude

    muscle_idx = [i for i, lab in enumerate(names) if lab == "muscle artifact"]
    muscle_var = 0.0
    if muscle_idx:
        muscle_var = float(ica.get_explained_variance_ratio(fit_raw, components=muscle_idx,
                                                            ch_type="eeg")["eeg"])
    qc = {
        "ica_n_components": int(ica.n_components_),
        "ica_labels": list(names),
        "ica_proba": [float(x) for x in proba],
        "ica_excluded": exclude,
        "ica_n_excluded": len(exclude),
        "ica_muscle_components": muscle_idx,
        "ica_muscle_variance_ratio": muscle_var,
    }
    return ica, qc


def make_epochs(raw: mne.io.Raw, cfg: dict) -> mne.Epochs:
    """Fixed-length epochs, NOT yet rejected: rejection happens after ICA cleaning (see
    reject_epochs), otherwise a heavy blinker loses every epoch before the blinks are removed."""
    e = cfg["preprocess"]["epochs"]
    return mne.make_fixed_length_epochs(raw, duration=e["length_s"], overlap=e["overlap_s"],
                                        preload=True, verbose="ERROR")


def reject_epochs(epochs: mne.Epochs, cfg: dict) -> tuple[mne.Epochs, dict]:
    """Drop epochs where any EEG channel exceeds the peak-to-peak threshold. Returns a copy."""
    e = cfg["preprocess"]["epochs"]
    ep = epochs.copy()
    n_before = len(ep)
    ep.drop_bad(reject=dict(eeg=e["reject_ptp_uv"] * 1e-6),
                flat=dict(eeg=e.get("flat_ptp_uv", 0.5) * 1e-6), verbose="ERROR")
    qc = {"n_epochs_total": n_before, "n_epochs_kept": len(ep),
          "epoch_reject_fraction": 1 - len(ep) / max(n_before, 1)}
    return ep, qc


def preprocess_block(raw: mne.io.Raw, cfg: dict) -> tuple[mne.Epochs, ICA, dict]:
    """Full chain for one block. Returns unrejected pre-ICA epochs (average-referenced), the ICA,
    and QC. Downstream: spectra.build_variant applies ICA, rejects epochs, re-references.
    """
    raw, qc = basic_clean(raw, cfg)
    raw = rereference(raw, "average")
    qc.update(emg_proxy(raw, cfg))
    ica, qc_ica = fit_ica(raw, cfg)
    qc.update(qc_ica)
    epochs = make_epochs(raw, cfg)
    qc["n_epochs_total"] = len(epochs)
    qc["sfreq"] = float(raw.info["sfreq"])
    qc["duration_s"] = float(raw.times[-1])
    return epochs, ica, qc
