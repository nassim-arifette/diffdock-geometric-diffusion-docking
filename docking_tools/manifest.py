"""The run manifest: the single file the viewer and the exporter read.

All paths inside the manifest are relative to the run directory, so a run directory can be moved,
zipped, or reopened on another machine without breaking anything.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from ._util import read_json, sha256_file, write_json

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"


class ManifestError(RuntimeError):
    pass


def save_manifest(run_dir: Path, manifest: dict) -> Path:
    path = Path(run_dir) / MANIFEST_NAME
    write_json(path, manifest)
    return path


def load_manifest(path_or_dir: os.PathLike | str) -> dict:
    p = Path(path_or_dir)
    if p.is_dir():
        p = p / MANIFEST_NAME
    if not p.is_file():
        raise ManifestError(f"no manifest at {p}")
    m = read_json(p)
    if m.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"unsupported manifest schema {m.get('schema_version')!r}")
    return m


def referenced_files(manifest: dict) -> List[dict]:
    """Every file entry ``{path, sha256}`` the manifest points at."""
    files = []
    for key in ("input_csv", "input_csv_portable", "config"):
        if manifest.get(key):
            files.append(manifest[key])
    for c in manifest.get("complexes", []):
        files.append(c["receptor"])
        if c.get("ligand_file"):
            files.append(c["ligand_file"])
        if c.get("reference"):
            files.append(c["reference"])
        for p in c.get("poses", []):
            files.append({"path": p["path"], "sha256": p["sha256"]})
            if p.get("frames"):
                files.append({"path": p["frames"]["path"], "sha256": p["frames"]["sha256"]})
    return files


def verify_manifest(run_dir: Path, manifest: dict) -> None:
    """Raise ManifestError unless every referenced file exists and still has the recorded hash."""
    problems = []
    for entry in referenced_files(manifest):
        path = Path(run_dir) / entry["path"]
        if not path.is_file():
            problems.append(f"missing: {entry['path']}")
        elif entry.get("sha256") and sha256_file(path) != entry["sha256"]:
            problems.append(f"hash mismatch: {entry['path']}")
    if problems:
        raise ManifestError("manifest verification failed:\n  " + "\n  ".join(problems))
