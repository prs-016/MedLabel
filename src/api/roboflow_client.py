"""
Resolve Roboflow workspace/project from API key.

Uses the official Python SDK (default workspace for your key). You only need
ROBOFLOW_API_KEY unless you have multiple projects — then set ROBOFLOW_PROJECT
or ROBOFLOW_PROJECT_URL.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

# Prefer projects whose slug/name matches MedLabel packaging labels
_PROJECT_HINTS = ("flat", "cylindrical", "medlabel", "label", "medicine", "drug")


def parse_project_url(url: str) -> tuple[str, str]:
    """Parse https://app.roboflow.com/<workspace>/<project>/..."""
    path = urlparse(url.strip()).path.strip("/")
    parts = [p for p in path.split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Could not parse workspace/project from URL: {url}")
    return parts[0], parts[1]


def list_projects(api_key: str) -> list[dict[str, Any]]:
    """List projects in the default workspace for this API key."""
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    ws = rf.workspace()
    if isinstance(ws, dict):
        ws_slug = ws.get("url") or ""
        raw_projects = ws.get("projects") or []
    else:
        ws_slug = getattr(ws, "url", "") or ""
        raw_projects = getattr(ws, "projects", None)
        if callable(raw_projects):
            raw_projects = raw_projects()
        elif hasattr(ws, "list_projects") and callable(ws.list_projects):
            raw_projects = ws.list_projects()
        raw_projects = raw_projects or []

    out: list[dict[str, Any]] = []
    for item in raw_projects:
        if isinstance(item, str):
            pid = item
            name = item.split("/")[-1]
        else:
            pid = str(item.get("id", ""))
            name = item.get("name", pid)
        if "/" in pid:
            workspace, project = pid.split("/", 1)
        else:
            workspace, project = ws_slug, pid
        out.append(
            {
                "workspace": workspace,
                "project": project,
                "id": pid,
                "name": name,
                "versions": 1,
                "classes": [],
            }
        )
    return out


def _score_project(project_slug: str) -> int:
    s = project_slug.lower()
    return sum(10 for hint in _PROJECT_HINTS if hint in s)


def latest_version_number(api_key: str, workspace: str, project: str) -> int:
    """Return the newest generated dataset version, or raise with UI instructions."""
    from roboflow import Roboflow

    proj = Roboflow(api_key=api_key).workspace(workspace).project(project)
    versions: list[int] = []
    if hasattr(proj, "list_versions") and callable(proj.list_versions):
        raw = proj.list_versions()
        if raw:
            for v in raw:
                if isinstance(v, int):
                    versions.append(v)
                elif isinstance(v, dict) and "version" in v:
                    versions.append(int(v["version"]))
                elif hasattr(v, "version"):
                    versions.append(int(v.version))
    if not versions and hasattr(proj, "versions") and callable(proj.versions):
        raw = proj.versions()
        versions = [int(x) for x in raw if str(x).isdigit()]
    if not versions:
        raise RuntimeError(
            f"No exported dataset versions for {workspace}/{project}. "
            "In Roboflow: open the project → Generate → create Version 1 (or higher), "
            "then run the download script again."
        )
    return max(versions)


def resolve_project(
    api_key: str,
    *,
    workspace: str | None = None,
    project: str | None = None,
    project_url: str | None = None,
    version: int | None = None,
) -> tuple[str, str, int]:
    """
    Return (workspace_slug, project_slug, version_number).
    """
    if project_url:
        workspace, project = parse_project_url(project_url)

    if workspace and project:
        ver = version if version is not None else 1
        return workspace, project, ver

    projects = list_projects(api_key)
    if not projects:
        raise RuntimeError("No Roboflow projects found for this API key.")

    if project and not workspace:
        matches = [p for p in projects if p["project"].lower() == project.lower()]
        if len(matches) == 1:
            p = matches[0]
            return p["workspace"], p["project"], version or 1
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple workspaces have project {project!r}. Set ROBOFLOW_PROJECT_URL."
            )

    if workspace and not project:
        matches = [p for p in projects if p["workspace"].lower() == workspace.lower()]
        if len(matches) == 1:
            p = matches[0]
            return p["workspace"], p["project"], version or 1

    if len(projects) == 1:
        p = projects[0]
        return p["workspace"], p["project"], version or 1

    ranked = sorted(projects, key=lambda p: _score_project(p["project"]), reverse=True)
    if ranked and _score_project(ranked[0]["project"]) > 0:
        p = ranked[0]
        return p["workspace"], p["project"], version or 1

    lines = [
        "Multiple Roboflow projects — set ROBOFLOW_PROJECT or ROBOFLOW_PROJECT_URL:",
    ]
    for p in projects:
        lines.append(f"  - {p['workspace']}/{p['project']}  ({p['name']!r})")
    raise RuntimeError("\n".join(lines))
