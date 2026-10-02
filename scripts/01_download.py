"""Stage 1: download ds003969 task files from OpenNeuro (skips sourcedata/).

Usage:
  python scripts/01_download.py --subjects sub-001 sub-002     # dev subset, ~1.2 GB
  python scripts/01_download.py --n-first 10
  python scripts/01_download.py                                # all 98, ~58 GB
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medgamma.cli import base_parser  # noqa: E402
from medgamma.config import load_config  # noqa: E402
from medgamma.data import download  # noqa: E402


def main():
    p = base_parser(__doc__)
    args = p.parse_args()
    cfg = load_config(args.config)
    subjects = args.subjects
    if subjects is None and args.n_first:
        subjects = [f"sub-{i:03d}" for i in range(1, args.n_first + 1)]
    root = download(cfg, subjects)
    print(f"downloaded to {root}")


if __name__ == "__main__":
    main()
