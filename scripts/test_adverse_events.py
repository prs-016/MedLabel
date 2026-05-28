#!/usr/bin/env python3
"""
Test adverse_events (openFDA FAERS + label).

  export PYTHONPATH=src
  python scripts/test_adverse_events.py
  python scripts/test_adverse_events.py --drug ibuprofen
  python scripts/test_adverse_events.py --scanned acetaminophen --query "side effects"
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
    from agent.adverse_events import lookup_adverse_events

    p = argparse.ArgumentParser()
    p.add_argument("--drug", help="Drug name to look up")
    p.add_argument("--scanned", help="Scanned drug from label")
    p.add_argument("--query", default="What are the side effects?", help="User question")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    drug = args.drug or args.query
    result = lookup_adverse_events(drug, scanned_drug=args.scanned)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.format_for_agent())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
