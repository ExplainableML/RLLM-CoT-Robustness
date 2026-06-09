#!/usr/bin/env python3
"""Convenience wrapper for the paper's append-'Wait' ablation.

The input should be an existing intervention JSONL file. The script appends a
configurable doubt marker to those exact rows, so the ablation changes only the
model resume point. Continue, judge, and score with the normal scripts.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).with_name("append_marker_to_interventions.py")
    if "--marker" not in sys.argv:
        sys.argv.extend(["--marker", "Wait"])
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
