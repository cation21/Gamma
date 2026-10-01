"""Stage 2: preprocess every block of every subject.

Per block, writes to results/derivatives/<sub>/:
  <sub>_task-<task>_desc-preica_epo.fif   average-referenced, 1 Hz high-passed, bads interpolated
  <sub>_task-<task>_ica.fif                ICA solution with ICLabel-based .exclude
  <sub>_task-<task>_qc.json                bad channels, ICLabel labels, EMG proxies, epoch counts

Usage: python scripts/02_preprocess.py [--subjects sub-001 ...] [--n-first N] [--overwrite] [--n-jobs 4]
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainmaxxing.cli import base_parser, deriv_path, setup  # noqa: E402
from brainmaxxing.data import available_blocks, load_raw  # noqa: E402
from brainmaxxing.preprocess import preprocess_block  # noqa: E402

log = logging.getLogger("02_preprocess")


def run_block(cfg: dict, sub: str, task: str) -> str:
    qc_path = deriv_path(cfg, sub, task, "qc", ".json")
    t0 = time.time()
    try:
        raw = load_raw(cfg, sub, task)
        epochs, ica, qc = preprocess_block(raw, cfg)
    except Exception as exc:
        log.exception("%s %s FAILED: %s", sub, task, exc)
        qc_path.write_text(json.dumps({"subject": sub, "task": task, "failed": str(exc)}, indent=2))
        return f"{sub} {task} FAILED"
    epochs.save(deriv_path(cfg, sub, task, "desc-preica_epo", ".fif"), overwrite=True, verbose="ERROR")
    ica.save(deriv_path(cfg, sub, task, "ica", ".fif"), overwrite=True, verbose="ERROR")
    qc.update(subject=sub, task=task, seconds=round(time.time() - t0, 1))
    qc_path.write_text(json.dumps(qc, indent=2))
    msg = (f"{sub} {task}: bads={qc['bad_channels']} excluded_ics={qc['ica_n_excluded']} "
           f"muscle_var={qc['ica_muscle_variance_ratio']:.3f} "
           f"epochs={qc['n_epochs_total']} ({qc['seconds']:.0f}s)")
    log.info(msg)
    return msg


def main():
    p = base_parser(__doc__)
    p.add_argument("--n-jobs", type=int, default=1, help="blocks processed in parallel")
    args = p.parse_args()
    cfg, subjects = setup(args)
    todo = [(sub, task) for sub in subjects for task in available_blocks(cfg, sub)
            if args.overwrite or not deriv_path(cfg, sub, task, "qc", ".json").exists()]
    log.info("%d blocks to process with %d worker(s)", len(todo), args.n_jobs)
    if args.n_jobs == 1:
        for sub, task in todo:
            run_block(cfg, sub, task)
    else:
        from joblib import Parallel, delayed
        for msg in Parallel(n_jobs=args.n_jobs, verbose=5, return_as="generator")(
                delayed(run_block)(cfg, sub, task) for sub, task in todo):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
