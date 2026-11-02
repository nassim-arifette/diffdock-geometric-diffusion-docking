"""Prepare inputs, run the pinned DiffDock, verify the output, and write ``manifest.json``.

A run is only called successful (``manifest["status"] == "ok"``) when every complex in the input manifest has the
expected number of SDF poses that RDKit parses back as 3D molecules. A crash, a skipped complex or a silent
upstream failure all end in ``status == "failed"`` with the reason recorded -- never in an empty "success".
"""
from __future__ import annotations

import datetime as _dt
import os
import platform
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ._util import GpuMemoryPoller, gpu_info, read_json, relpath, run_streaming, sha256_file, utc_now
from .inputs import (
    InputError,
    InputRow,
    LigandInfo,
    ReceptorInfo,
    is_pandas_unsafe_name,
    portable_rows,
    resolve_ligand,
    resolve_receptor,
    safe_complex_name,
    validate_input_csv,
    write_input_csv,
)
from .manifest import SCHEMA_VERSION, save_manifest, verify_manifest
from .poses import PoseError, collect_poses
from .setup_env import Layout, torch_env

DEFAULT_CONFIG_NAME = "default_inference_args.yaml"


class DockingRunError(RuntimeError):
    def __init__(self, message: str, manifest_path: Optional[Path] = None):
        super().__init__(message)
        self.manifest_path = manifest_path


@dataclass
class PreparedComplex:
    row: InputRow
    receptor: ReceptorInfo
    ligand: LigandInfo
    ligand_file: Optional[Path]
    reference: Optional[Path]
    reference_note: Optional[str]


@dataclass
class PreparedRun:
    run_dir: Path
    csv_path: Path  # absolute paths: what DiffDock reads
    complexes: List[PreparedComplex]
    portable_csv_path: Optional[Path] = None  # same rows, paths relative to this CSV's folder: survives a move to Drive


def new_run_dir(runs_root: Path, label: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-") or "run"
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(runs_root) / f"{stamp}_{safe}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def prepare_inputs(
    run_dir: Path,
    *,
    complex_name: str,
    receptor: str,
    ligand: str,
    chains: Optional[Sequence[str]] = None,
    reference: Optional[str] = None,
    reference_note: Optional[str] = None,
) -> PreparedRun:
    """Resolve receptor + ligand into ``run_dir/inputs`` and write the CSV the upstream script reads."""
    inputs = Path(run_dir) / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    if is_pandas_unsafe_name(complex_name):
        raise InputError(
            f"complex_name {complex_name!r} would be read as a number or NaN by pandas inside DiffDock; "
            f"use e.g. {safe_complex_name(complex_name)!r}"
        )
    rec = resolve_receptor(receptor, inputs, chains=chains)
    lig = resolve_ligand(ligand)

    ligand_file: Optional[Path] = None
    if lig.kind == "file":
        src = Path(lig.description)
        ligand_file = inputs / f"{complex_name}_ligand{src.suffix.lower()}"
        shutil.copyfile(src, ligand_file)
        lig_for_csv = str(ligand_file.resolve())
    else:
        lig_for_csv = lig.description

    ref_path: Optional[Path] = None
    if reference:
        src = Path(reference).expanduser()
        if not src.is_file():
            raise FileNotFoundError(f"reference ligand not found: {reference}")
        ref_path = inputs / f"{complex_name}_reference{src.suffix.lower()}"
        shutil.copyfile(src, ref_path)

    row = InputRow(
        complex_name=complex_name,
        protein_path=str(rec.path.resolve()),
        ligand_description=lig_for_csv,
    )
    csv_path = write_input_csv(inputs / "input_protein_ligand.csv", [row])
    portable = write_input_csv(inputs / "input_protein_ligand.portable.csv", portable_rows([row], inputs))
    return PreparedRun(
        run_dir=Path(run_dir),
        csv_path=csv_path,
        complexes=[PreparedComplex(row, rec, lig, ligand_file, ref_path, reference_note)],
        portable_csv_path=portable,
    )


def build_run_config(repo: Path, overrides: Dict[str, object], out_path: Path) -> dict:
    """Write ``default_inference_args.yaml`` + overrides as this run's config and return the effective dict.

    The upstream script applies every key of ``--config`` *after* the command line, so a flag such as
    ``--samples_per_complex 40`` passed next to ``--config`` is silently overridden by the YAML. Putting the
    overrides into the config file is the only way they take effect, and it records them for free.
    """
    import yaml

    cfg = yaml.safe_load((Path(repo) / DEFAULT_CONFIG_NAME).read_text(encoding="utf-8"))
    unknown = [k for k in overrides if k not in cfg and k not in {"batch_size", "save_visualisation"}]
    if unknown:
        raise ValueError(f"unknown inference parameter(s) {unknown}; known keys: {sorted(cfg)}")
    cfg.update(overrides)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8", newline="\n")
    return cfg


def _file_entry(run_dir: Path, path: Path) -> dict:
    return {"path": relpath(path, run_dir), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _tail(path: Path, n: int = 25) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def run_diffdock(
    workdir: Path | str,
    prepared: PreparedRun,
    *,
    overrides: Optional[Dict[str, object]] = None,
    seed: Optional[int] = 0,
    python_exe: Optional[Path | str] = None,
    echo: bool = True,
) -> dict:
    """Run inference for ``prepared`` and return the written manifest. Raises DockingRunError on failure."""
    layout = Layout(Path(workdir).resolve())
    run_dir = prepared.run_dir
    # DOCKING_TOOLS_PYTHON lets a user (or a test) point at their own interpreter instead of the micromamba env
    py = str(python_exe or os.environ.get("DOCKING_TOOLS_PYTHON") or layout.python)
    if not Path(py).exists():
        raise DockingRunError(f"DiffDock environment not found at {py}; run docking_tools.setup_env first")

    validate_input_csv(prepared.csv_path)
    config_path = run_dir / "config" / "inference_args.yaml"
    effective = build_run_config(layout.repo, dict(overrides or {}), config_path)
    expected = int(effective["samples_per_complex"])
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for pc in prepared.complexes:  # upstream exits 0 when a complex fails: stale poses from an earlier call must not pass as new
        shutil.rmtree(pred_dir / pc.row.complex_name, ignore_errors=True)
    log_path = run_dir / "logs" / "inference.log"
    log_path.parent.mkdir(exist_ok=True)

    cmd = [
        py, "-m", "inference",
        "--config", str(config_path.resolve()),
        "--protein_ligand_csv", str(prepared.csv_path.resolve()),
        "--out_dir", str(pred_dir.resolve()),
    ]
    if seed is not None:
        cmd += ["--seed", str(int(seed))]

    t0 = time.time()
    with GpuMemoryPoller() as poller:
        rc = run_streaming(cmd, cwd=layout.repo, env=torch_env(layout), log_path=log_path, echo=echo)
    wall = round(time.time() - t0, 1)

    setup_info = read_json(layout.setup_info) if layout.setup_info.exists() else None
    manifest: dict = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_dir.name,
        "created_utc": utc_now(),
        "status": "ok",
        "error": None,
        "command": cmd,
        "returncode": rc,
        "seed": seed,
        "input_csv": _file_entry(run_dir, prepared.csv_path),
        "input_csv_portable": _file_entry(run_dir, prepared.portable_csv_path) if prepared.portable_csv_path else None,
        "config": _file_entry(run_dir, config_path),
        "inference_parameters": effective,
        "upstream": (setup_info or {}).get("upstream"),
        "patch": (setup_info or {}).get("patch"),
        "checkpoints": (setup_info or {}).get("checkpoints"),
        "esm_weights": (setup_info or {}).get("esm_weights"),
        "packages": (setup_info or {}).get("packages"),
        "environment": {
            "notebook_python": sys.version.split()[0],
            "platform": platform.platform(),
            "gpu": gpu_info(),
            "gpu_smoke": (setup_info or {}).get("gpu_smoke"),
            "setup_seconds_total": (setup_info or {}).get("setup_seconds_total"),
        },
        "timings": {
            "inference_wall_seconds": wall,
            "note": "inference wall time includes protein-language-model embedding and graph building, not setup",
        },
        "gpu_peak_memory_mib": poller.peak_mib,
        "log": relpath(log_path, run_dir),
        "complexes": [],
    }

    failures: List[str] = []
    if rc != 0:
        failures.append(f"inference exited with code {rc}")
    for pc in prepared.complexes:
        name = pc.row.complex_name
        entry = {
            "complex_name": name,
            "status": "ok",
            "error": None,
            "receptor": {
                **_file_entry(run_dir, pc.receptor.path),
                "source": pc.receptor.source,
                "chains": pc.receptor.chains,
                "n_residues": pc.receptor.n_residues,
                "removed_hetero_atoms": pc.receptor.removed_hetero,
                "removed_waters": pc.receptor.removed_waters,
            },
            "ligand": {
                "description": pc.ligand.description if pc.ligand.kind == "smiles" else None,
                "kind": pc.ligand.kind,
                "source": pc.ligand.source,
                "canonical_smiles": pc.ligand.canonical_smiles,
                "n_heavy_atoms": pc.ligand.n_heavy_atoms,
                "unspecified_stereocenters": pc.ligand.unspecified_stereocenters,
            },
            "ligand_file": _file_entry(run_dir, pc.ligand_file) if pc.ligand_file else None,
            "reference": (
                {**_file_entry(run_dir, pc.reference), "provenance": pc.reference_note} if pc.reference else None
            ),
            "poses": [],
        }
        try:
            if rc != 0:
                raise PoseError(f"inference exited with code {rc}")
            poses = collect_poses(pred_dir / name, name, expected=expected, expect_frames=bool(effective.get("save_visualisation")))
            entry["poses"] = [
                {
                    "rank": p.rank,
                    "pose_id": p.pose_id,
                    "path": relpath(p.path, run_dir),
                    "sha256": p.sha256,
                    "confidence": p.confidence,
                    "confidence_rounded": p.confidence_rounded,
                    "n_atoms": p.n_atoms,
                    "n_heavy_atoms": p.n_heavy_atoms,
                    "frames": (
                        {"path": relpath(p.frames_path, run_dir), "sha256": p.frames_sha256, "n_models": p.n_frame_models}
                        if p.frames_path else None
                    ),
                }
                for p in poses
            ]
            entry["scores_file"] = (
                relpath(pred_dir / name / "scores.json", run_dir) if (pred_dir / name / "scores.json").exists() else None
            )
        except PoseError as e:
            entry["status"] = "failed"
            entry["error"] = str(e)
            failures.append(f"{name}: {e}")
        manifest["complexes"].append(entry)

    if failures:
        manifest["status"] = "failed"
        manifest["error"] = "; ".join(failures)
        manifest["log_tail"] = _tail(log_path)
    else:
        verify_manifest(run_dir, manifest)
    path = save_manifest(run_dir, manifest)
    if failures:
        raise DockingRunError(f"docking run failed: {manifest['error']}\n--- log tail ---\n{manifest['log_tail']}", path)
    return manifest


def default_complex_name(receptor: str, ligand: str) -> str:
    """A short, filesystem-safe name derived from the receptor spec (PDB id or file stem).

    Names that pandas would misread (``1e20``, ``123``, ``na``) get a ``c_`` prefix, see :func:`safe_complex_name`.
    """
    stem = Path(receptor.strip()).stem if Path(receptor.strip()).suffix else receptor.strip()
    return safe_complex_name(stem.lower())
