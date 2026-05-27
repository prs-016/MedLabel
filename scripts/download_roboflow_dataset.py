#!/usr/bin/env python3
"""
Download the MedLabel YOLO dataset from Roboflow.

Minimum config in .env:
  ROBOFLOW_API_KEY=...

Optional (only if you have multiple projects):
  ROBOFLOW_PROJECT=your-project-slug
  ROBOFLOW_PROJECT_URL=https://app.roboflow.com/workspace/project
  ROBOFLOW_VERSION=1

  python scripts/download_roboflow_dataset.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass


def _setup_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    _load_env()
    _setup_path()

    p = argparse.ArgumentParser(description="Download labeled dataset from Roboflow.")
    p.add_argument("--out", type=Path, default=Path("data/labeled"))
    p.add_argument("--format", default="yolov11")
    p.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY", ""))
    p.add_argument("--workspace", default=os.getenv("ROBOFLOW_WORKSPACE") or None)
    p.add_argument("--project", default=os.getenv("ROBOFLOW_PROJECT") or None)
    p.add_argument(
        "--project-url",
        default=os.getenv("ROBOFLOW_PROJECT_URL") or None,
        help="Full Roboflow project URL (alternative to workspace+project)",
    )
    ver_env = os.getenv("ROBOFLOW_VERSION", "").strip()
    p.add_argument("--version", type=int, default=int(ver_env) if ver_env.isdigit() else None)
    p.add_argument("--list", action="store_true", help="List projects for this API key and exit")
    args = p.parse_args()

    if not args.api_key:
        print("error: set ROBOFLOW_API_KEY in .env", file=sys.stderr)
        return 1

    if args.list:
        from api.roboflow_client import list_projects

        for proj in list_projects(args.api_key):
            print(
                f"{proj['workspace']}/{proj['project']}  "
                f"name={proj['name']!r}  versions={proj['versions']}  "
                f"classes={proj['classes']}"
            )
        return 0

    try:
        from api.roboflow_client import latest_version_number, resolve_project

        workspace, project, version = resolve_project(
            args.api_key,
            workspace=args.workspace,
            project=args.project,
            project_url=args.project_url,
            version=None if args.version is None else args.version,
        )
        if args.version is None:
            version = latest_version_number(args.api_key, workspace, project)
        else:
            version = args.version
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        print("hint: python scripts/download_roboflow_dataset.py --list", file=sys.stderr)
        return 1

    try:
        from roboflow import Roboflow
    except ImportError:
        print("error: pip install roboflow", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=args.api_key)
    print(f"Downloading {workspace}/{project} v{version} ({args.format}) …")
    dataset = (
        rf.workspace(workspace)
        .project(project)
        .version(version)
        .download(args.format, location=str(args.out))
    )
    loc = getattr(dataset, "location", None) or getattr(dataset, "path", None) or args.out
    print(f"done: {loc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
