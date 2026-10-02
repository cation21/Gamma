"""Stage 4: parameterise ROI spectra and build the long analysis table.

Primary table (results/tables/fits_primary.csv): default fit range / mode, clean-average variant,
all ROIs, plus IRASA columns where available.
Multiverse table (results/tables/fits_multiverse.csv, with --multiverse): every combination in
configs/multiverse.yaml, primary ROI only.
Channel table (results/tables/fits_channels.csv, with --channels): per-channel fits for topographies.

Usage: python scripts/04_parameterize.py [--multiverse] [--channels] [--subjects ...]
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medgamma.cli import base_parser, deriv_path, setup  # noqa: E402
from medgamma.config import REPO_ROOT  # noqa: E402
from medgamma.data import processed_blocks, block_metadata, load_participants  # noqa: E402
from medgamma.parameterize import fit_many, fit_spectrum, irasa_row  # noqa: E402
from medgamma.spectra import clean_spectrum, roi_spectrum  # noqa: E402

log = logging.getLogger("04_parameterize")
QC_KEYS = ["emg_proxy_logpower", "ica_muscle_variance_ratio", "ica_n_excluded", "n_epochs_kept",
           "epoch_reject_fraction", "bad_channels"]


def load_psd(cfg, sub, task, variant):
    path = deriv_path(cfg, sub, task, f"desc-{variant}_psd")
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    return z["freqs"], clean_spectrum(z["freqs"], z["psd"], cfg), list(z["ch_names"])


def load_qc(cfg, sub, task) -> dict:
    path = deriv_path(cfg, sub, task, "qc", ".json")
    if not path.exists():
        return {}
    qc = json.loads(path.read_text())
    out = {k: qc.get(k) for k in QC_KEYS}
    out["bad_channels"] = ";".join(out.get("bad_channels") or [])
    out["n_bad_channels"] = len(qc.get("bad_channels") or [])
    return out


def meta_row(participants, cfg, sub, task) -> dict:
    pr = participants.set_index("subject").loc[sub]
    bm = block_metadata(cfg, task)
    return {"subject": sub, "task": task, "condition": bm["condition"], "run": bm["run"],
            "meditator": bool(pr["meditator"]), "tradition": pr["tradition"], "group": pr["group"],
            "first_session": pr["first_session"], "years_of_practice": pr["years_of_practice"],
            "age": pr["age"], "gender": pr["gender"]}


def main():
    p = base_parser(__doc__)
    p.add_argument("--multiverse", action="store_true")
    p.add_argument("--channels", action="store_true")
    args = p.parse_args()
    cfg, subjects = setup(args, source="derivatives")
    participants = load_participants(cfg)
    rois = cfg["spectra"]["rois"]
    primary_roi = cfg["stats"]["primary_roi"]
    mv = yaml.safe_load(open(REPO_ROOT / "configs" / "multiverse.yaml", encoding="utf-8"))

    primary, multiverse, channels = [], [], []
    for sub in subjects:
        for task in processed_blocks(cfg, sub):
            meta = meta_row(participants, cfg, sub, task)
            meta.update(load_qc(cfg, sub, task))

            loaded = load_psd(cfg, sub, task, "clean-average")
            if loaded is None:
                log.warning("%s %s: no clean-average PSD", sub, task)
                continue
            freqs, psd, ch_names = loaded

            # --- primary: every ROI, default spec ---
            for roi, chans in rois.items():
                spec = roi_spectrum(psd, ch_names, chans)
                row = fit_spectrum(freqs, spec, cfg)
                row.update(meta, roi=roi, variant="clean-average", fit_range_name="primary")
                if roi == primary_roi:
                    ir = deriv_path(cfg, sub, task, "desc-clean-average_irasa")
                    if ir.exists():
                        z = np.load(ir)
                        fit = pd.DataFrame({"Slope": z["fit_slope"], "Intercept": z["fit_intercept"],
                                            "R^2": z["fit_r2"]})
                        row.update(irasa_row(z["freqs"], z["aperiodic"], z["oscillatory"], fit, cfg))
                primary.append(row)

            if args.channels:
                df = fit_many(freqs, psd, cfg, ch_names)
                for k, v in meta.items():
                    df[k] = v
                channels.append(df)

            # --- multiverse: primary ROI, every specification ---
            if args.multiverse:
                for variant in ["clean-average", "clean-csd", "raw-average", "raw-csd"]:
                    loaded = load_psd(cfg, sub, task, variant)
                    if loaded is None:
                        continue
                    f, ps, chn = loaded
                    spec = roi_spectrum(ps, chn, rois[primary_roi])
                    for rname, frange in mv["fit_range_hz"].items():
                        for mode in mv["aperiodic_mode"]:
                            row = fit_spectrum(f, spec, cfg, fit_range=frange, aperiodic_mode=mode)
                            row.update(meta, roi=primary_roi, variant=variant, fit_range_name=rname,
                                       spec_id=f"{variant}|{rname}|{mode}")
                            multiverse.append(row)
            log.info("%s %s done", sub, task)

    tables = Path(cfg["dataset"]["tables"])
    pd.DataFrame(primary).to_csv(tables / "fits_primary.csv", index=False)
    log.info("wrote fits_primary.csv (%d rows)", len(primary))
    if multiverse:
        pd.DataFrame(multiverse).to_csv(tables / "fits_multiverse.csv", index=False)
        log.info("wrote fits_multiverse.csv (%d rows)", len(multiverse))
    if channels:
        pd.concat(channels, ignore_index=True).to_csv(tables / "fits_channels.csv", index=False)


if __name__ == "__main__":
    main()
