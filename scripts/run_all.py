"""One command for the whole thing: stream every subject, then parameterise, model, plot, report.

  python scripts/run_all.py                    # resume: skips subjects already processed
  python scripts/run_all.py --overwrite        # clean rerun of all 98 (~7 h) — do this once before submission
  python scripts/run_all.py --n-first 30       # a subset
  python scripts/run_all.py --skip-stream      # only the analysis stages, on whatever is processed

Each stage's output goes to results/logs/run_all_<stage>.log. Safe to re-run at any time.
On Windows, to keep it alive after closing the terminal:
  Start-Process .\\.venv\\Scripts\\python.exe -ArgumentList "scripts\\run_all.py" -WindowStyle Hidden
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LOGS = ROOT / "results" / "logs"


def run(stage: str, *args: str) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"run_all_{stage}.log"
    cmd = [PY, str(ROOT / "scripts" / f"{stage}.py"), *args]
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] {stage} {' '.join(args)}", flush=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n===== {time.ctime()} {' '.join(cmd)}\n")
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT,
                             env={**__import__('os').environ, "PYTHONIOENCODING": "utf-8"})
    print(f"[{time.strftime('%H:%M:%S')}] {stage} finished rc={rc} in {(time.time() - t0) / 60:.1f} min "
          f"(log: {log.relative_to(ROOT)})", flush=True)
    return rc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--overwrite", action="store_true", help="reprocess every subject from scratch")
    p.add_argument("--n-first", type=int, default=None)
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--skip-stream", action="store_true", help="skip download/preprocessing")
    p.add_argument("--no-multiverse", action="store_true")
    args = p.parse_args()

    if not args.skip_stream:
        stream = ["--n-jobs", str(args.n_jobs)]
        if args.overwrite:
            stream.append("--overwrite")
        if args.n_first:
            stream += ["--n-first", str(args.n_first)]
        if args.subjects:
            stream += ["--subjects", *args.subjects]
        if run("07_stream", *stream) != 0:
            print("streaming stage reported errors; continuing with whatever was processed", flush=True)

    par = [] if args.no_multiverse else ["--multiverse", "--channels"]
    run("04_parameterize", *par)
    run("05_stats")
    run("06_figures")
    run("08_report")
    report = ROOT / "results" / "REPORT.md"
    if report.exists():
        print("\n" + report.read_text(encoding="utf-8"))
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
