"""Run a frozen docking pilot without regenerating candidates for each selector.

    python -m docking_tools.pilot --manifest benchmark/pilot.json --split dev \
        --workdir /content/diffdock_work --out /content/diffdock_pilot

The manifest fixes complex identities, input hashes, seeds and candidate count.
Each complex/seed is checkpointed separately. Re-running verifies and reuses
finished candidate pools. Failed cases remain in the denominator and are only
retried with --retry-failed. Reference coordinates are never a docking input:
the chemical graph becomes an isomeric SMILES, while the SDF is evaluation-only.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

from ._util import read_json, sha256_file, utc_now, write_json
from .drive import persist_run
from .inputs import safe_complex_name
from .manifest import load_manifest, verify_manifest
from .run import prepare_inputs, run_diffdock
from .setup_env import setup


class PilotError(ValueError):
    pass


def read_protocol(path: Path, split: Optional[str] = None) -> tuple[dict, list[dict]]:
    """Validate the declared experiment before spending GPU time."""
    protocol = read_json(path)
    if protocol.get("schema_version") != 1 or not protocol.get("benchmark_id"):
        raise PilotError("protocol requires schema_version=1 and a benchmark_id")
    seeds = protocol.get("seeds")
    if (not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise PilotError("seeds must be distinct non-negative integers")
    count = protocol.get("samples_per_complex")
    if type(count) is not int or count < 1:
        raise PilotError("samples_per_complex must be a positive integer")
    rows = protocol.get("complexes")
    if not isinstance(rows, list) or not rows:
        raise PilotError("protocol contains no complexes")
    seen = set()
    pdb_splits = {}
    chosen = []
    for item in rows:
        name = item.get("complex_id")
        if not isinstance(name, str) or not name or name in seen:
            raise PilotError(f"invalid or repeated complex_id: {name!r}")
        seen.add(name)
        if name in (".", "..") or safe_complex_name(name) != name:
            raise PilotError(f"complex_id {name!r} is not filesystem/pandas safe")
        if item.get("split") not in ("dev", "eval"):
            raise PilotError(f"{name}: split must be dev or eval")
        # The public dataset uses PDB_CCD identifiers; two ligands from one PDB
        # are not independent development/evaluation targets.
        if re.fullmatch(r"[0-9A-Za-z]{4}_[0-9A-Za-z]+", name):
            pdb = name.split("_")[0].upper()
            if pdb in pdb_splits and pdb_splits[pdb] != item["split"]:
                raise PilotError(f"PDB {pdb} occurs in both development and evaluation")
            pdb_splits[pdb] = item["split"]
        resolved = dict(item)
        for kind in ("receptor", "reference"):
            spec = item.get(kind)
            expected = item.get(kind + "_sha256")
            if not spec or not isinstance(expected, str) or len(expected) != 64:
                raise PilotError(f"{name}: {kind} path and sha256 are required")
            target = (path.parent / spec).resolve()
            if not target.is_file() or sha256_file(target) != expected:
                raise PilotError(f"{name}: {kind} missing or hash changed: {target}")
            resolved[kind] = str(target)
        if split is None or item["split"] == split:
            chosen.append(resolved)
    if not chosen:
        raise PilotError(f"no complexes selected for split {split!r}")
    return protocol, chosen


def ligand_smiles(reference: Path) -> str:
    """Chemical identity only: serialize the graph, never the crystallographic coordinates."""
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(reference), sanitize=True, removeHs=True)
    molecules = list(supplier)
    if not molecules or any(m is None for m in molecules):
        raise PilotError(f"reference must contain readable ligand copies: {reference}")
    identities = {Chem.MolToSmiles(m, isomericSmiles=True, canonical=True) for m in molecules}
    if len(identities) != 1:
        raise PilotError("reference ligand copies have different chemical identities or stereochemistry")
    mol = molecules[0]
    if not mol.GetNumHeavyAtoms():
        raise PilotError("reference ligand has no heavy atoms")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise PilotError("disconnected ligand is outside this single-ligand docking protocol")
    return identities.pop()


def _verified_finished(out: Path, entry: dict) -> bool:
    run_dir = out / entry["run_dir"]
    manifest = load_manifest(run_dir)
    verify_manifest(run_dir, manifest)
    if manifest.get("status") != "ok":
        return False
    evaluation = run_dir / "evaluation.json"
    return (evaluation.is_file() and entry.get("evaluation_sha256") == sha256_file(evaluation)
            and entry.get("manifest_sha256") == sha256_file(run_dir / "manifest.json"))


def run_pilot(
    manifest_path: Path, *, workdir: Path, out: Path, split: Optional[str] = None,
    gnina: Optional[str] = None, drive_root: Optional[Path] = None,
    skip_setup: bool = False, retry_failed: bool = False, python_exe: Optional[str] = None,
) -> dict:
    manifest_path, out, workdir = manifest_path.resolve(), out.resolve(), workdir.resolve()
    protocol, complexes = read_protocol(manifest_path, split)
    config = {"protocol_sha256": sha256_file(manifest_path), "split": split,
              "gnina": str(gnina) if gnina else None}
    if gnina:
        executable = shutil.which(str(gnina)) or str(gnina)
        if not Path(executable).is_file():
            raise PilotError(f"GNINA executable not found: {gnina}")
        config["gnina_sha256"] = sha256_file(executable)
        gnina = executable
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "pilot_state.json"
    if state_path.exists():
        state = read_json(state_path)
        if state.get("config") != config:
            raise PilotError("output belongs to another protocol/split/scorer; use a separate --out")
    else:
        state = {"schema_version": 1, "benchmark_id": protocol["benchmark_id"],
                 "config": config, "started_utc": utc_now(), "status": "running", "cases": {}}
        shutil.copyfile(manifest_path, out / "protocol.json")

    def checkpoint():
        state["updated_utc"] = utc_now()
        write_json(state_path, state)
        if drive_root:
            write_json(drive_root / "pilots" / out.name / "pilot_state.json", state)

    if drive_root:
        drive_root = drive_root.resolve()
        if not drive_root.parent.is_dir():
            raise PilotError("Drive parent is absent; mount it before requesting persistence")
        drive_root.mkdir(exist_ok=True)
    checkpoint()
    if not skip_setup:
        try:
            setup(workdir, cache_dir=drive_root / "cache" if drive_root else None)
        except Exception as exc:
            state.update(status="setup_failed", error={"type": type(exc).__name__, "message": str(exc)})
            checkpoint()
            raise

    from .evaluation import evaluate_run

    for item in complexes:
        for seed in protocol["seeds"]:
            case_id = f"{item['complex_id']}_seed{seed}"
            previous = state["cases"].get(case_id)
            if previous and previous["status"] == "ok":
                if not _verified_finished(out, previous):
                    raise PilotError(f"{case_id}: recorded result changed; use a new output directory")
                print(f"[REUSE] {case_id}: verified frozen candidates and evaluation", flush=True)
                continue
            if previous and previous["status"] == "failed" and not retry_failed:
                print(f"[FAILED, retained] {case_id}: {previous.get('error')}", flush=True)
                continue
            # A scorer failure or interrupted evaluation must never trigger new sampling.
            reuse_pool = False
            if previous:
                old_run = out / previous["run_dir"]
                if (old_run / "manifest.json").is_file():
                    old_manifest = load_manifest(old_run)
                    if old_manifest.get("status") == "ok":
                        verify_manifest(old_run, old_manifest)
                        reuse_pool = True
            if reuse_pool:
                attempt, relative = previous["attempt"], previous["run_dir"]
                run_dir = out / relative
            else:
                attempt = 1 if not previous else previous.get("attempt", 1) + 1
                relative = f"runs/{case_id}_attempt{attempt}"
                run_dir = out / relative
                run_dir.mkdir(parents=True, exist_ok=False)
            entry = {"complex_id": item["complex_id"], "split": item["split"], "seed": seed,
                     "attempt": attempt, "run_dir": relative, "status": "running"}
            if previous:
                entry["previous_attempts"] = previous.get("previous_attempts", []) + [
                    {k: v for k, v in previous.items() if k != "previous_attempts"}]
            state["cases"][case_id] = entry
            checkpoint()
            print(f"[RUN] {case_id}", flush=True)
            try:
                smiles = ligand_smiles(Path(item["reference"]))
                entry["input_isomeric_smiles"] = smiles
                if not reuse_pool:
                    prepared = prepare_inputs(run_dir, complex_name=item["complex_id"],
                                              receptor=item["receptor"], ligand=smiles,
                                              reference=item["reference"],
                                              reference_note="frozen pilot native ligand; coordinates used only for evaluation")
                    run_diffdock(workdir, prepared, seed=seed, python_exe=python_exe,
                                 overrides={"samples_per_complex": protocol["samples_per_complex"],
                                            "save_visualisation": True})
                else:
                    print(f"[REUSE POOL] {case_id}: evaluating existing candidates", flush=True)
                evaluation_attempt = 1
                evaluation_path = run_dir / f"evaluation.attempt{evaluation_attempt}.json"
                while evaluation_path.exists():
                    evaluation_attempt += 1
                    evaluation_path = run_dir / f"evaluation.attempt{evaluation_attempt}.json"
                entry["evaluation_attempt"] = evaluation_attempt
                result = evaluate_run(run_dir, gnina=gnina, output=evaluation_path)
                if result.get("status") != "ok":
                    raise PilotError("evaluation did not finish successfully")
                # The inspector reads the last completed evaluation; failed attempts remain separate.
                write_json(run_dir / "evaluation.json", result)
                entry.update(status="ok", manifest_sha256=sha256_file(run_dir / "manifest.json"),
                             evaluation_sha256=sha256_file(run_dir / "evaluation.json"))
            except Exception as exc:
                entry.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
                print(f"[FAIL] {case_id}: {exc}", file=sys.stderr, flush=True)
            finally:
                entry["finished_utc"] = utc_now()
                checkpoint()
                if drive_root:
                    persist_run(run_dir, drive_root)

    state["n_expected"] = len(complexes) * len(protocol["seeds"])
    state["n_ok"] = sum(e["status"] == "ok" for e in state["cases"].values())
    state["n_failed"] = sum(e["status"] == "failed" for e in state["cases"].values())
    state["status"] = "complete" if state["n_ok"] == state["n_expected"] else "complete_with_failures"
    checkpoint()
    return state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--workdir", default=Path("/content/diffdock_work"), type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", choices=("dev", "eval"))
    parser.add_argument("--gnina")
    parser.add_argument("--drive-root", type=Path)
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--python-exe")
    args = parser.parse_args(argv)
    state = run_pilot(args.manifest, workdir=args.workdir, out=args.out, split=args.split,
                      gnina=args.gnina, drive_root=args.drive_root, skip_setup=args.skip_setup,
                      retry_failed=args.retry_failed, python_exe=args.python_exe)
    print(json.dumps({k: state[k] for k in ("status", "n_expected", "n_ok", "n_failed")}, indent=2))
    return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
