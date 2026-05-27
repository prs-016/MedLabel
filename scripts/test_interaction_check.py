#!/usr/bin/env python3
"""
Test interaction_check (DDInter + FDA).

  export PYTHONPATH=src
  python scripts/test_interaction_check.py
  python scripts/test_interaction_check.py --query ibuprofen --scanned acetaminophen
  python scripts/test_interaction_check.py --pair "acetaminophen|ibuprofen" --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _setup() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    _setup()
    from agent.interaction_check import interaction_check

    p = argparse.ArgumentParser()
    p.add_argument("--query", default="ibuprofen", help="Other drug or full query")
    p.add_argument("--scanned", default="acetaminophen", help="Scanned drug (drug_a)")
    p.add_argument("--pair", help="Override as drug_a|drug_b")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if args.pair:
        result = interaction_check(args.pair)
    else:
        result = interaction_check(args.query, scanned_drug=args.scanned)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.format_for_agent())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
