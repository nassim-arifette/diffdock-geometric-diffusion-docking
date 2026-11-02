"""Package a finished run for download, after checking that the archive really holds the predictions."""
from __future__ import annotations

import tarfile
from pathlib import Path
from typing import List, Optional

from .manifest import MANIFEST_NAME, ManifestError, load_manifest, referenced_files, verify_manifest


def export_run(run_dir: Path | str, dest_dir: Optional[Path | str] = None) -> Path:
    """Create ``<run_id>.tar.gz`` containing the whole run directory and return its path.

    Refuses to export a failed run, a run whose files changed since the manifest was written, or an archive that
    does not contain every file the manifest references.
    """
    run_dir = Path(run_dir).resolve()
    manifest = load_manifest(run_dir)
    if manifest.get("status") != "ok":
        raise ManifestError(f"refusing to export run with status {manifest.get('status')!r}: {manifest.get('error')}")
    verify_manifest(run_dir, manifest)

    run_id = manifest["run_id"]
    dest_dir = Path(dest_dir).resolve() if dest_dir else run_dir.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / f"{run_id}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(run_dir, arcname=run_id)

    names = list_archive(archive)
    expected = {f"{run_id}/{MANIFEST_NAME}"} | {f"{run_id}/{e['path']}" for e in referenced_files(manifest)}
    missing = sorted(expected - set(names))
    if missing:
        archive.unlink()
        raise ManifestError(f"archive is missing {len(missing)} file(s), e.g. {missing[:3]}")
    n_poses = sum(len(c["poses"]) for c in manifest["complexes"])
    if n_poses == 0:
        archive.unlink()
        raise ManifestError("archive would contain no poses")
    return archive


def list_archive(archive: Path | str) -> List[str]:
    with tarfile.open(archive, "r:gz") as tar:
        return [m.name for m in tar.getmembers() if m.isfile()]
