"""Stage 0: validate the pipeline on simulated spectra with known ground truth.

Usage: python scripts/00_simulate.py [--config configs/default.yaml] [--fit-range 30 95] [--mode fixed]
Writes results/tables/sim_fits.csv, sim_summary.csv and figures/sim_recovery.png.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medgamma.cli import base_parser, setup  # noqa: E402
from medgamma.plots import simulation_recovery  # noqa: E402
from medgamma.simulate import run_simulation  # noqa: E402


def main():
    p = base_parser(__doc__)
    p.add_argument("--fit-range", nargs=2, type=float, default=None)
    p.add_argument("--mode", choices=["fixed", "knee"], default=None)
    args = p.parse_args()
    cfg, _ = setup(args)

    fits, summary = run_simulation(cfg, fit_range=args.fit_range, aperiodic_mode=args.mode)
    tables = Path(cfg["dataset"]["tables"])
    fits.to_csv(tables / "sim_fits.csv", index=False)
    summary.to_csv(tables / "sim_summary.csv", index=False)

    figdir = Path(__file__).resolve().parents[1] / "figures"
    figdir.mkdir(exist_ok=True)
    simulation_recovery(summary).savefig(figdir / "sim_recovery.png", dpi=200)

    pivot = summary.pivot(index="scenario", columns="measure", values="cohens_d").round(2)
    print("\nCohen's d (meditator - control) by scenario and measure:\n")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
