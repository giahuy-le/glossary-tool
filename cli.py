"""CLI entry point for the glossary tool."""

from __future__ import annotations

import sys

from .config import OUTPUT_PHASE3, require_api_credentials
from .phase1 import run_phase1
from .phase2 import run_phase2
from .phase3 import run_phase3


def main(argv: list[str] | None = None):
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        raise SystemExit("Usage: python glossary.py input.csv")

    require_api_credentials()

    input_file = args[0]
    df_out, texts = run_phase1(input_file)
    df_norm = run_phase2(df_out, texts)
    run_phase3(df_norm, texts)
    print(f"🏁 All done! Output: {OUTPUT_PHASE3}")

