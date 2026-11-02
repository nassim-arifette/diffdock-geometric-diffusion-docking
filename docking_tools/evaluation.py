"""Evaluate saved candidates without letting the crystal pose select them.

The input molecule defines identity; PoseBusters ``dock`` defines physical
validity. Reference coordinates are used only for the reported, unaligned
symmetry-aware RMSD. Scores and refined coordinates are stored separately.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

from ._util import read_json, relpath, sha256_file, utc_now, write_json
from .manifest import load_manifest, verify_manifest

RMSD_THRESHOLD = 2.0


class InvalidPoseError(ValueError):
    """The stored candidate is chemically unreadable, rather than the validator being unavailable."""


def read_molecules(path):
    from rdkit import Chem

    molecules = list(Chem.SDMolSupplier(str(path), removeHs=True, sanitize=True))
    if not molecules or any(m is None or not m.GetNumConformers() for m in molecules):
        raise ValueError(f"not a complete, readable molecular SDF: {path}")
    if any(not m.GetConformer().Is3D() for m in molecules):
        raise ValueError(f"SDF is not marked as 3D: {path}")
    return molecules


def _identity(mol, expected, *, from_coordinates=True):
    """Match charge, bonds and every specified stereocentre, allowing unspecified input stereo."""
    from rdkit import Chem

    candidate = Chem.RemoveHs(Chem.Mol(mol))
    target = Chem.RemoveHs(Chem.Mol(expected))
    if from_coordinates:
        Chem.AssignStereochemistryFrom3D(candidate, replaceExistingTags=True)
    connectivity = (Chem.MolToSmiles(candidate, isomericSmiles=False)
                    == Chem.MolToSmiles(target, isomericSmiles=False))
    stereo = connectivity and candidate.HasSubstructMatch(target, useChirality=True)
    return {"input_connectivity_and_charge": connectivity, "input_stereochemistry": bool(stereo)}


def receptor_frame_rmsd(probe, reference, *, max_matches=10000):
    """Heavy-atom RMSD under graph symmetry, with NO fitting or coordinate mutation.

    A translated perfect conformer must remain wrong. Chirality is checked
    separately against the input molecule; matching here preserves graph bonds.
    Refuse a truncated symmetry search instead of reporting an approximate minimum.
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign

    p = Chem.RemoveHs(Chem.Mol(probe))
    r = Chem.RemoveHs(Chem.Mol(reference))
    if Chem.MolToSmiles(p, isomericSmiles=False) != Chem.MolToSmiles(r, isomericSmiles=False):
        raise ValueError("reference and predicted ligand have different connectivity or charge")
    # RDKit's own terminal-group symmetrisation is skipped when explicit maps
    # are passed. Build its O/N resonance query ourselves so we can also detect
    # enumeration overflow, which CalcRMS's automatic mapping does not expose.
    # See Code/GraphMol/MolAlign/AlignMolecules.cpp::symmetrizeTerminalAtoms.
    query = Chem.RWMol(p)
    pattern = Chem.MolFromSmarts("[O,N;D1;$([O,N;D1]-[*]=[O,N;D1]),$([O,N;D1]=[*]-[O,N;D1])]~[*]")
    for terminal, center in p.GetSubstructMatches(pattern):
        query.GetAtomWithIdx(terminal).SetFormalCharge(0)
        query.ReplaceBond(query.GetBondBetweenAtoms(terminal, center).GetIdx(), Chem.BondFromSmarts("-,="))
    matches = r.GetSubstructMatches(query, uniquify=False, useChirality=False, maxMatches=max_matches + 1)
    if not matches or len(matches) > max_matches:
        raise ValueError("symmetry mapping is absent or exceeds the explicit search limit")
    atom_maps = [list(enumerate(match)) for match in matches]
    result = float(rdMolAlign.CalcRMS(p, r, map=atom_maps, symmetrizeConjugatedTerminalGroups=True))
    if not math.isfinite(result):
        raise ValueError("non-finite RMSD")
    return result


class PhysicalValidator:
    def __init__(self):
        from posebusters import PoseBusters

        self.buster = PoseBusters(config="dock", max_workers=0)
        if any(m["function"] == "rmsd" for m in self.buster.config["modules"]):
            raise ValueError("validity configuration must not contain a reference-RMSD test")
        self.config = self.buster.config
        self.config_sha256 = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()

    def check(self, pose_path, receptor, expected):
        import numpy as np

        try:
            molecules = read_molecules(pose_path)
        except ValueError as exc:
            raise InvalidPoseError(str(exc)) from exc
        if len(molecules) != 1:
            raise InvalidPoseError("each candidate file must contain exactly one molecule")
        checks = _identity(molecules[0], expected)
        # Never pass mol_true: crystal coordinates cannot affect filtering.
        table = self.buster.bust(mol_pred=Path(pose_path), mol_cond=Path(receptor), full_report=False)
        if len(table) != 1 or not len(table.columns):
            raise ValueError("PoseBusters did not return exactly one nonempty row")
        for name, value in table.iloc[0].items():
            # NaN/unknown is failure. A numeric/string column is not a truthy pass.
            checks[str(name)] = bool(value) if isinstance(value, (bool, np.bool_)) else False
        return molecules[0], checks


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def select_variants(poses, *, gnina_available=False, threshold=RMSD_THRESHOLD, reference_available=True):
    """Selection reads confidence, physical validity and CNNscore, never RMSD."""
    if any(not _finite(p.get("confidence")) for p in poses):
        raise ValueError("full-precision finite DiffDock confidences are required")
    refined = [p["refined"] for p in poses if p.get("refined")]
    specs = {
        "A_confidence": (True, poses, "confidence", False),
        "B_valid_confidence": (True, poses, "confidence", True),
        "C_valid_gnina": (gnina_available, poses, "gnina_score", True),
        "D_refined_gnina": (gnina_available, refined, "gnina_score", True),
    }
    result = {}
    for name, (available, pool, score, require_valid) in specs.items():
        eligible = [p for p in pool if _finite(p.get(score)) and (not require_valid or p.get("valid") is True)]
        selected = sorted(eligible, key=lambda p: (-p[score], p["pose_id"])) if available else []
        near = lambda p: _finite(p.get("rmsd_angstrom")) and p["rmsd_angstrom"] < threshold
        joint = lambda p: near(p) and p.get("valid") is True
        top = selected[0] if selected else None
        result[name] = {
            "available": available, "selected_pose_ids": [p["pose_id"] for p in selected],
            "empty_selection": not selected if available else None,
            "top1_rmsd_success": bool(top and near(top)) if available and reference_available else None,
            "top1_valid": bool(top and top.get("valid") is True) if available else None,
            "top1_joint_success": bool(top and joint(top)) if available and reference_available else None,
            "top5_joint_success": any(joint(p) for p in selected[:5]) if available and reference_available else None,
        }
    return result


class Gnina:
    """Explicit score-only versus local minimisation, with logged binary identity."""
    def __init__(self, executable):
        resolved = shutil.which(str(executable)) or str(executable)
        self.executable = Path(resolved).resolve()
        if not self.executable.is_file():
            raise FileNotFoundError(f"GNINA executable not found: {executable}")
        version = subprocess.run([str(self.executable), "--version"], capture_output=True, text=True, timeout=30)
        if version.returncode:
            raise RuntimeError("GNINA --version failed")
        self.info = {"version": version.stdout.strip(), "binary_sha256": sha256_file(self.executable),
                     "ranking_field": "CNNscore", "higher_is_better": True}

    def run(self, receptor, pose, output, *, minimize=False):
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(self.executable), "-r", str(receptor), "-l", str(pose),
               "--minimize" if minimize else "--score_only", "--seed", "0", "--cpu", "1", "-o", str(output)]
        started = time.monotonic()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        output.with_suffix(".log").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
        if proc.returncode:
            raise RuntimeError(f"GNINA exited {proc.returncode}; see {output.with_suffix('.log').name}")
        molecules = read_molecules(output)
        if len(molecules) != 1 or not molecules[0].HasProp("CNNscore"):
            raise ValueError("GNINA output must contain one molecule with CNNscore")
        score = float(molecules[0].GetProp("CNNscore"))
        if not math.isfinite(score):
            raise ValueError("GNINA returned a non-finite CNNscore")
        if not minimize and receptor_frame_rmsd(molecules[0], read_molecules(pose)[0]) > 0.002:
            raise ValueError("score-only GNINA unexpectedly changed ligand coordinates")
        return {"score": score, "seconds": round(time.monotonic() - started, 3), "command": cmd,
                "output_sha256": sha256_file(output)}


def evaluate_run(run_dir, *, gnina=None, output=None):
    from rdkit import Chem

    run_dir = Path(run_dir).resolve()
    manifest = load_manifest(run_dir)
    verify_manifest(run_dir, manifest)
    if manifest["status"] != "ok":
        raise ValueError("evaluate a successful run; failed generation belongs in the benchmark denominator")
    validator = PhysicalValidator()
    scorer = Gnina(gnina) if gnina else None
    output = Path(output) if output else run_dir / "evaluation.json"
    if output.exists():
        raise FileExistsError(f"evaluation already exists: {output}; use a new output path for a new evaluation")
    result = {
        "schema_version": 1, "status": "running", "run_id": manifest["run_id"], "created_utc": utc_now(),
        "manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "protocol": {"rmsd_threshold_angstrom": RMSD_THRESHOLD, "alignment": False,
                     "reference_policy": "minimum over all same-identity crystal copies, evaluation only",
                     "validity": "PoseBusters dock plus input connectivity, charge and specified stereo",
                     "refinement": "all raw candidates independently; then revalidate and rank by CNNscore"},
        "backends": {"rdkit": importlib.metadata.version("rdkit"),
                     "posebusters": importlib.metadata.version("posebusters"),
                     "posebusters_config_sha256": validator.config_sha256,
                     "gnina": scorer.info if scorer else None}, "complexes": [],
    }
    write_json(output, result)
    started = time.monotonic()
    try:
        for complex_entry in manifest["complexes"]:
            receptor = run_dir / complex_entry["receptor"]["path"]
            refs = read_molecules(run_dir / complex_entry["reference"]["path"]) if complex_entry.get("reference") else []
            expected = Chem.MolFromSmiles(complex_entry["ligand"]["canonical_smiles"])
            if expected is None:
                raise ValueError("input ligand identity is missing or invalid")
            if any(not all(_identity(m, expected).values()) for m in refs):
                raise ValueError("reference file contains a different molecular identity or stereo")
            entry = {"complex_name": complex_entry["complex_name"], "poses": [], "variants": {}, "oracle": {}}
            result["complexes"].append(entry)
            for pose in complex_entry["poses"]:
                row = {k: pose[k] for k in ("pose_id", "rank", "path", "sha256", "confidence")}
                row.update(valid=False, checks={}, rmsd_angstrom=None, gnina_score=None, errors=[], refined=None)
                entry["poses"].append(row)
                _assess(row, run_dir, receptor, expected, refs, validator)
                if scorer:
                    target = run_dir / "postprocess" / output.stem / row["pose_id"]
                    try:
                        raw_score = scorer.run(receptor, run_dir / row["path"], target / "scored.sdf")
                        row["gnina_score"] = raw_score["score"]
                        row["scoring"] = raw_score
                    except Exception as exc:
                        row["errors"].append(f"score_only: {type(exc).__name__}: {exc}")
                    try:
                        refined_path = target / "refined.sdf"
                        refinement = scorer.run(receptor, run_dir / row["path"], refined_path, minimize=True)
                        refined = {"pose_id": row["pose_id"] + "-refined", "source_pose_id": row["pose_id"],
                                   "path": relpath(refined_path, run_dir), "sha256": sha256_file(refined_path),
                                   "valid": False, "checks": {}, "rmsd_angstrom": None,
                                   "gnina_score": refinement["score"], "errors": [], "refinement": refinement}
                        row["refined"] = refined
                    except Exception as exc:
                        row["errors"].append(f"minimize: {type(exc).__name__}: {exc}")
                    if row["refined"] is not None:
                        _assess(row["refined"], run_dir, receptor, expected, refs, validator)
                write_json(output, result)
            entry["reference_available"] = bool(refs)
            entry["variants"] = select_variants(entry["poses"], gnina_available=scorer is not None, reference_available=bool(refs))
            near = lambda p: _finite(p.get("rmsd_angstrom")) and p["rmsd_angstrom"] < RMSD_THRESHOLD
            entry["oracle"] = {
                "raw_rmsd_success": any(near(p) for p in entry["poses"]) if refs else None,
                "raw_joint_success": any(near(p) and p["valid"] for p in entry["poses"]) if refs else None,
                "refined_joint_success": any(p.get("refined") and near(p["refined"]) and p["refined"]["valid"] for p in entry["poses"]) if scorer and refs else None,
            }
        result["status"] = "ok"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["evaluation_wall_seconds"] = round(time.monotonic() - started, 3)
        write_json(output, result)
    return result


def _assess(row, run_dir, receptor, expected, references, validator):
    try:
        mol, row["checks"] = validator.check(run_dir / row["path"], receptor, expected)
    except InvalidPoseError as exc:
        row["checks"] = {"readable_molecule": False}
        row["valid"] = False
        row["errors"].append(f"invalid_pose: {exc}")
        return
    # Other failures (a crashed backend, malformed backend output, failed RMSD
    # calculation) must escape: they are not measured zero-percent validity.
    row["valid"] = bool(row["checks"]) and all(row["checks"].values())
    if references and row["checks"].get("input_connectivity_and_charge", True):
        row["rmsd_angstrom"] = min(receptor_frame_rmsd(mol, r) for r in references)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--gnina", help="optional installed GNINA binary; enables C and D")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = evaluate_run(args.run, gnina=args.gnina, output=args.output)
    print(json.dumps({c["complex_name"]: c["variants"] for c in result["complexes"]}, indent=2))


if __name__ == "__main__":
    main()
