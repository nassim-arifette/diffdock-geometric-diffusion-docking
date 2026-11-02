"""Keep datasets, runs and exports on Google Drive so a recycled Colab VM loses nothing.

Layout under ``<drive_root>`` (default ``/content/drive/MyDrive/DiffDock_colab``)::

    runs/<run_id>/...          full run directory (manifest.json, inputs, predictions, logs)  -- failed runs too
    exports/<run_id>.tar.gz    the checked archive from export_run
    datasets/<name>/...        input data (bundled example, PDBBind subset + embeddings)
    cache/...                  big downloads reused by setup_env (checkpoints, ESM2 weights)
    index.json                 one line per persisted run

Every copy is verified (file sizes and SHA-256 for single files, the manifest hashes for runs) so "saved to Drive"
means "read back and matches", not "the copy call returned".
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Optional

from ._util import read_json, sha256_file, utc_now, write_json
from .manifest import MANIFEST_NAME, ManifestError, load_manifest, verify_manifest

DEFAULT_MOUNT = "/content/drive"
DEFAULT_ROOT = "/content/drive/MyDrive/DiffDock_colab"


class DriveError(RuntimeError):
    pass


def _mount_point(mount: Optional[str]) -> str:
    return mount or os.environ.get("DOCKING_DRIVE_MOUNT") or DEFAULT_MOUNT


def drive_mounted(mount: Optional[str] = None) -> bool:
    return Path(_mount_point(mount), "MyDrive").is_dir()


def ensure_drive(root: str = DEFAULT_ROOT, mount: Optional[str] = None) -> Optional[Path]:
    """Return the Drive root directory (created if needed), mounting Drive from a Colab notebook if necessary.

    Returns None -- never raises -- when Drive is unavailable, so a notebook still runs without it.
    """
    mount = _mount_point(mount)
    if not drive_mounted(mount):
        try:  # inside a Colab notebook this shows the usual authorisation prompt
            from google.colab import drive  # type: ignore

            drive.mount(mount)
        except Exception as e:  # noqa: BLE001 - not on Colab, or the user declined
            print(f"Google Drive not available ({e.__class__.__name__}); results stay on the VM only.")
            return None
    if not drive_mounted(mount):
        print("Google Drive did not mount; results stay on the VM only.")
        return None
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _copy_file_verified(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    if dst.stat().st_size != src.stat().st_size or sha256_file(dst) != sha256_file(src):
        raise DriveError(f"copy of {src} to {dst} does not match the source")


def _copy_tree_verified(src: Path, dst: Path) -> int:
    n = 0
    for f in sorted(src.rglob("*")):
        if f.is_file():
            _copy_file_verified(f, dst / f.relative_to(src))
            n += 1
    return n


def persist_run(run_dir: Path | str, drive_root: Path | str, archive: Optional[Path | str] = None) -> Path:
    """Copy a finished (or failed) run to Drive, verify it, and record it in ``index.json``."""
    run_dir, drive_root = Path(run_dir), Path(drive_root)
    dest = drive_root / "runs" / run_dir.name
    n = _copy_tree_verified(run_dir, dest)
    status = None
    if (dest / MANIFEST_NAME).exists():
        manifest = load_manifest(dest)
        status = manifest.get("status")
        verify_manifest(dest, manifest)  # hashes recorded at run time still match the Drive copy
    entry = {"run_id": run_dir.name, "status": status, "files": n, "saved_utc": utc_now(), "path": str(dest)}
    if archive:
        arch = Path(archive)
        _copy_file_verified(arch, drive_root / "exports" / arch.name)
        entry["archive"] = str(drive_root / "exports" / arch.name)
    index = drive_root / "index.json"
    entries = read_json(index) if index.exists() else []
    entries = [e for e in entries if e.get("run_id") != run_dir.name] + [entry]
    write_json(index, entries)
    return dest


def persist_dataset(src: Path | str, drive_root: Path | str, name: str) -> Path:
    """Copy an input dataset (file or directory) to ``datasets/<name>`` and verify it."""
    src, drive_root = Path(src), Path(drive_root)
    dest = drive_root / "datasets" / name
    if src.is_dir():
        if not _copy_tree_verified(src, dest):
            raise DriveError(f"nothing to copy in {src}")
    else:
        _copy_file_verified(src, dest / src.name)
    return dest


def cache_restore(cache_dir: Optional[Path | str], name: str, dest: Path, sha256: Optional[str] = None) -> bool:
    """Copy ``cache_dir/name`` to ``dest`` when it exists (and matches ``sha256``). True if restored."""
    if cache_dir is None:
        return False
    src = Path(cache_dir) / name
    if not src.is_file() or dest.exists():
        return False
    if sha256 and sha256_file(src) != sha256:
        return False  # stale/corrupt cache entry: ignore it and download again
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    return True


def cache_store(cache_dir: Optional[Path | str], name: str, src: Path) -> bool:
    if cache_dir is None or not src.is_file():
        return False
    dst = Path(cache_dir) / name
    if (dst.is_file() and dst.stat().st_size == src.stat().st_size
            and sha256_file(dst) == sha256_file(src)):
        return False
    _copy_file_verified(src, dst)
    return True
