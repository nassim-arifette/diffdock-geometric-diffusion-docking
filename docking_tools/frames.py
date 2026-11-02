"""Reverse-diffusion frames: what DiffDock saves with ``--save_visualisation``, parsed for the diffusion slider.

Upstream writes ``rank<k>_reverseprocess.pdb`` with one ``MODEL`` per recorded state of that candidate pose. Its writer
(``utils/visualise.py::PDBFile`` driven by ``inference.py`` and ``utils/sampling.py``) adds, in this order:

    part 0, order 0   the RDKit molecule as built (with hydrogens, the conformer the run started from)
    part 0, order 1   the ligand's initial positions (heavy atoms; the hydrogens keep their old positions)
    part 1, order 1   the randomised starting pose (the "noise" end of the trajectory)
    part 1, order t+2 the pose after denoising step t = 0 .. N-1
    part 1, order 2   the final pose, added once more at the end -- which *overwrites* the slot of step 0

so the file holds ``[conformer, initial, random start, FINAL, step 1, step 2, ..., step N-1]``: step 0 is not recoverable
and the final pose appears twice (slot 3 and last). Rather than trusting that layout blindly, :func:`trajectory`
identifies the final pose by comparing with the predicted pose's coordinates and reports what it found.

Only heavy atoms are returned: the hydrogens in these files are not moved by the sampler.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


class FrameError(RuntimeError):
    pass


@dataclass
class Frame:
    index: int  # position in the returned trajectory
    kind: str  # "random_start" | "denoising_step" | "final"
    step: Optional[int]  # denoising step t (0-based; step 0 is not recorded by upstream), None for the random start
    coords: np.ndarray  # (n_heavy, 3)


@dataclass
class Trajectory:
    frames: List[Frame]
    n_models_in_file: int
    notes: List[str] = field(default_factory=list)

    def coords(self) -> np.ndarray:
        return np.stack([f.coords for f in self.frames])


def read_models(path: Path | str) -> List[List[str]]:
    """Split a multi-model PDB file into per-model atom lines (ATOM/HETATM only)."""
    models: List[List[str]] = []
    current: Optional[List[str]] = None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        rec = line[:6].strip()
        if rec == "MODEL":
            current = []
        elif rec == "ENDMDL":
            if current is not None:
                models.append(current)
            current = None
        elif rec in ("ATOM", "HETATM") and current is not None:
            current.append(line)
    if current:  # file without a closing ENDMDL
        models.append(current)
    if not models:
        raise FrameError(f"no MODEL records in {path}")
    return models


def _element(line: str) -> str:
    el = line[76:78].strip()
    if el:
        return el.capitalize()
    name = line[12:16].strip()
    return "".join(c for c in name if c.isalpha())[:1].upper()


def heavy_coords(model_lines: List[str]) -> np.ndarray:
    pts = [
        (float(l[30:38]), float(l[38:46]), float(l[46:54]))
        for l in model_lines
        if _element(l) not in ("H", "D")
    ]
    return np.asarray(pts, dtype=np.float64).reshape(-1, 3)


def sdf_heavy_coords(sdf_path: Path | str) -> np.ndarray:
    from rdkit import Chem

    mol = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)[0]
    if mol is None:
        raise FrameError(f"cannot read {sdf_path}")
    pos = mol.GetConformer().GetPositions()
    return pos[[a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]]


def trajectory(frames_pdb: Path | str, final_pose_sdf: Path | str, *, tol: float = 0.02) -> Trajectory:
    """The denoising trajectory of one pose: random start, the recorded steps, and the final pose last.

    ``tol`` (angstrom, max coordinate difference) absorbs the 3-decimal rounding of PDB/SDF files.
    """
    final = sdf_heavy_coords(final_pose_sdf)
    models = read_models(frames_pdb)
    n = len(final)
    coords = []
    for m in models:
        c = heavy_coords(m)
        coords.append(c[:n] if len(c) >= n else c)
    notes: List[str] = []
    is_final = [len(c) == n and float(np.abs(c - final).max()) <= tol for c in coords]
    if not any(is_final):
        raise FrameError(
            f"none of the {len(models)} frames in {frames_pdb} matches the predicted pose {final_pose_sdf}; "
            "the file layout is not the one this parser understands"
        )
    last_final = max(i for i, f in enumerate(is_final) if f)
    # models 0 and 1 are the starting conformer / initial positions; model 2 is the random start
    if len(models) < 4 or last_final < 3:
        raise FrameError(f"unexpected frame layout in {frames_pdb}: {len(models)} models, final pose at {last_final}")
    start = coords[2]
    frames = [Frame(0, "random_start", None, start)]
    for i in range(3, last_final):
        if is_final[i] and i == 3:
            notes.append("model 4 repeats the final pose: upstream overwrites the slot of denoising step 0 with it; skipped")
            continue
        # model i holds the pose after denoising step (i - 3); step 0 is the overwritten slot
        frames.append(Frame(len(frames), "denoising_step", i - 3, coords[i]))
    frames.append(Frame(len(frames), "final", last_final - 3, coords[last_final]))
    if len(models) > last_final + 1:
        notes.append(f"{len(models) - last_final - 1} model(s) after the final pose ignored")
    return Trajectory(frames=frames, n_models_in_file=len(models), notes=notes)


def to_multimodel_pdb(mol, traj: Trajectory) -> str:
    """The trajectory as a multi-model PDB string (heavy atoms, one MODEL per frame) for a viewer's frame slider."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    heavy = Chem.RemoveHs(Chem.Mol(mol))
    out = []
    for k, fr in enumerate(traj.frames):
        m = Chem.Mol(heavy)
        conf = m.GetConformer()
        for i, (x, y, z) in enumerate(fr.coords):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        block = Chem.MolToPDBBlock(m).splitlines()
        out.append(f"MODEL     {k + 1:>4}")
        out.extend(l for l in block if l.startswith(("ATOM", "HETATM", "CONECT")))
        out.append("ENDMDL")
    out.append("END")
    return "\n".join(out) + "\n"
