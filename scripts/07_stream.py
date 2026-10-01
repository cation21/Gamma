"""Stage 7: streaming replacement for stages 1-3. Per subject: download -> process 4 blocks ->
write ~4.5 MB of outputs -> delete the raw BDFs. Peak disk use ~1 GB regardless of how many
subjects you run. Resumable: subjects with complete outputs are skipped.

Usage:
  python scripts/07_stream.py --n-first 20 --n-jobs 4
  python scripts/07_stream.py --subjects sub-001 sub-002 --keep-raw
  python scripts/07_stream.py                       # all 98 subjects

Then continue with 04_parameterize.py, 05_stats.py, 06_figures.py as usual.
"""

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainmaxxing.cli import base_parser, setup  # noqa: E402
from brainmaxxing.data import (available_blocks, block_path, download, ensure_complete,  # noqa: E402
                               load_participants)
from brainmaxxing.pipeline import (ALL_VARIANTS, delete_raw, process_block,  # noqa: E402
                                   subject_outputs_present)

log = logging.getLogger("07_stream")


def main():
    p = base_parser(__doc__)
    p.add_argument("--n-jobs", type=int, default=4, help="blocks of one subject processed in parallel")
    p.add_argument("--keep-raw", action="store_true", help="do not delete BDFs after processing")
    p.add_argument("--save-epochs", action="store_true", help="also keep pre-ICA epochs (77 MB/block)")
    p.add_argument("--irasa-all", action="store_true")
    args = p.parse_args()
    cfg, _ = setup(args)

    # Subject list comes from participants.tsv (downloaded on first call), not from disk.
    root = Path(cfg["dataset"]["bids_root"])
    if not (root / "participants.tsv").exists():
        from brainmaxxing.data import download_with_retry
        download_with_retry(cfg, [])
    subjects = args.subjects or load_participants(cfg)["subject"].tolist()
    if args.n_first:
        subjects = subjects[: args.n_first]

    pending = [s for s in subjects if args.overwrite or not subject_outputs_present(cfg, s)]
    log.info("%d of %d subjects still to process", len(pending), len(subjects))

    def fetch(sub):
        """Never raises: a subject that cannot be fetched is logged and skipped."""
        try:
            bad = ensure_complete(cfg, sub)
        except Exception as exc:
            log.error("%s: fetch failed: %s", sub, str(exc)[:160])
            return sub
        if bad:
            log.error("%s: blocks %s missing/truncated after retries; skipped", sub, bad)
            for task in bad:
                block_path(cfg, sub, task).unlink(missing_ok=True)
        return sub

    # One download runs ahead of processing so network time overlaps CPU time.
    pool = ThreadPoolExecutor(max_workers=1)
    prefetch = pool.submit(fetch, pending[0]) if pending else None

    t_start = time.time()
    for i, sub in enumerate(pending, 1):
        t0 = time.time()
        prefetch.result()
        t_dl = time.time() - t0
        if i < len(pending):
            prefetch = pool.submit(fetch, pending[i])
        blocks = available_blocks(cfg, sub)
        if not blocks:
            log.warning("[%d/%d] %s: no blocks after download, skipping", i, len(pending), sub)
            continue

        if args.n_jobs == 1:
            msgs = [process_block(cfg, sub, t, ALL_VARIANTS, args.irasa_all, args.save_epochs) for t in blocks]
        else:
            from joblib import Parallel, delayed
            msgs = Parallel(n_jobs=min(args.n_jobs, len(blocks)))(
                delayed(process_block)(cfg, sub, t, ALL_VARIANTS, args.irasa_all, args.save_epochs)
                for t in blocks)
        for m in msgs:
            log.info(m)
        if not args.keep_raw:
            freed = delete_raw(cfg, sub)
            log.info("%s: deleted raw (%.0f MB)", sub, freed / 1e6)
        elapsed = time.time() - t_start
        log.info("[%d/%d] %s done in %.0fs (waited %.0fs for download); elapsed %.1f min, ETA %.1f min",
                 i, len(pending), sub, time.time() - t0, t_dl, elapsed / 60,
                 elapsed / i * (len(pending) - i) / 60)
    pool.shutdown()


if __name__ == "__main__":
    main()
