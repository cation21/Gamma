"""Check derivatives for inconsistencies and print the subjects that need reprocessing.

A subject is flagged when any block has: a QC record marked failed for a non-data reason
(file not found, permission error — i.e. a race between two runs), a QC record without its
PSD files (or vice versa), or a processed duration far below the sidecar's RecordingDuration.

Usage: python scripts/verify_outputs.py            # prints one subject id per line
       python scripts/verify_outputs.py --verbose  # also prints the reason
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainmaxxing.cli import base_parser, setup  # noqa: E402
from brainmaxxing.data import processed_blocks  # noqa: E402

RACE_MARKERS = ("not found", "being used by another process", "PermissionError", "Errno 2")


def main():
    args = base_parser(__doc__).parse_args()
    cfg, subjects = setup(args, source="derivatives")
    deriv = Path(cfg["dataset"]["derivatives"])
    raw = Path(cfg["dataset"]["bids_root"])
    flagged = {}
    for sub in subjects:
        for task in processed_blocks(cfg, sub):
            qc = json.loads((deriv / sub / f"{sub}_task-{task}_qc.json").read_text())
            psd = (deriv / sub / f"{sub}_task-{task}_desc-clean-average_psd.npz").exists()
            reason = None
            if "failed" in qc:
                if any(m in qc["failed"] for m in RACE_MARKERS):
                    reason = f"{task}: failed for non-data reason ({qc['failed'][:50]})"
            else:
                side = raw / sub / "eeg" / f"{sub}_task-{task}_eeg.json"
                expected = json.loads(side.read_text()).get("RecordingDuration") if side.exists() else None
                if expected and qc.get("duration_s", 0) < expected - 15:
                    reason = f"{task}: processed {qc.get('duration_s', 0):.0f}s of {expected}s"
                elif not psd and qc.get("n_epochs_kept", 0) >= cfg["preprocess"]["epochs"].get("min_epochs", 30):
                    reason = f"{task}: QC ok but PSD missing"
                elif "variants" not in qc:
                    reason = f"{task}: QC from an older pipeline version"
            if reason:
                flagged.setdefault(sub, []).append(reason)
        if len(processed_blocks(cfg, sub)) < 4:
            flagged.setdefault(sub, []).append(f"only {len(processed_blocks(cfg, sub))} of 4 blocks have QC")
    for sub, reasons in flagged.items():
        print(sub if not args.verbose else f"{sub}  {'; '.join(reasons)}")
    if args.verbose:
        print(f"\n{len(flagged)} subject(s) need reprocessing", file=sys.stderr)


if __name__ == "__main__":
    main()
