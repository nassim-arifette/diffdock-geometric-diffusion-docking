"""Find, parse and identify the poses written by one DiffDock run."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ._util import read_json, sha256_file

RANK_RE = re.compile(r"^rank(\d+)_confidence(-?\d+(?:\.\d+)?)\.sdf$")


class PoseError(RuntimeError):
    """The prediction directory is missing, incomplete or unreadable."""


@dataclass
class Pose:
    rank: int
    pose_id: str
    path: Path
    sha256: str
    confidence_rounded: float  # as written in the file name (2 decimals)
    confidence: Optional[float]  # full precision from scores.json, None if unavailable
    n_atoms: int
    n_heavy_atoms: int
    frames_path: Optional[Path] = None  # rank<k>_reverseprocess.pdb, present when run with --save_visualisation
    frames_sha256: Optional[str] = None
    n_frame_models: Optional[int] = None


def parse_sdf(path: Path) -> dict:
    """Parse the first molecule with RDKit; require at least one atom and a 3D conformer."""
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    mol = supplier[0] if len(supplier) else None
    if mol is None:
        raise PoseError(f"RDKit cannot parse {path}")
    if mol.GetNumAtoms() == 0 or mol.GetNumConformers() == 0:
        raise PoseError(f"{path} has no atoms or no conformer")
    conf = mol.GetConformer()
    if not conf.Is3D():
        raise PoseError(f"{path} conformer is not 3D")
    return {
        "n_atoms": mol.GetNumAtoms(),
        "n_heavy_atoms": sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1),
    }


def collect_poses(
    complex_dir: Path, complex_name: str, expected: Optional[int] = None, *, expect_frames: bool = False
) -> List[Pose]:
    """Return poses sorted by rank. Raises PoseError unless every file parses and the set is complete.

    With ``expect_frames`` (the run was started with ``save_visualisation``) every pose must also have its
    ``rank<k>_reverseprocess.pdb`` file with at least one MODEL.
    """
    if not complex_dir.is_dir():
        raise PoseError(f"no prediction directory {complex_dir}")
    found = {}
    for p in complex_dir.iterdir():
        m = RANK_RE.match(p.name)
        if m:
            found[int(m.group(1))] = (p, float(m.group(2)))
    if not found:
        raise PoseError(f"no rank*_confidence*.sdf files in {complex_dir}")
    ranks = sorted(found)
    if ranks != list(range(1, len(ranks) + 1)):
        raise PoseError(f"ranks are not contiguous from 1: {ranks}")
    if expected is not None and len(ranks) != expected:
        raise PoseError(f"expected {expected} poses in {complex_dir}, found {len(ranks)}")

    top = complex_dir / "rank1.sdf"
    if not top.exists():
        raise PoseError(f"{top} missing")
    if sha256_file(top) != sha256_file(found[1][0]):
        raise PoseError("rank1.sdf differs from rank1_confidence*.sdf")

    scores: Optional[List[float]] = None
    scores_path = complex_dir / "scores.json"
    if scores_path.exists():
        data = read_json(scores_path)
        scores = data.get("confidence")
        if scores is not None and len(scores) != len(ranks):
            raise PoseError(f"scores.json has {len(scores)} values for {len(ranks)} poses")

    poses: List[Pose] = []
    for rank in ranks:
        path, rounded = found[rank]
        meta = parse_sdf(path)
        digest = sha256_file(path)
        full = None if scores is None else float(scores[rank - 1])
        if full is not None and abs(round(full, 2) - rounded) > 0.011:
            raise PoseError(f"rank {rank}: file name says {rounded} but scores.json says {full}")
        frames_path = complex_dir / f"rank{rank}_reverseprocess.pdb"
        frames_digest = n_models = None
        if frames_path.is_file():
            n_models = sum(1 for line in frames_path.read_text(errors="replace").splitlines() if line.startswith("MODEL"))
            if n_models == 0:
                raise PoseError(f"{frames_path.name} contains no MODEL records")
            frames_digest = sha256_file(frames_path)
        elif expect_frames:
            raise PoseError(f"rank {rank}: {frames_path.name} is missing although frames were requested")
        else:
            frames_path = None
        poses.append(
            Pose(
                rank=rank,
                pose_id=f"{complex_name}-rank{rank:02d}-{digest[:8]}",
                path=path,
                sha256=digest,
                confidence_rounded=rounded,
                confidence=full,
                n_atoms=meta["n_atoms"],
                n_heavy_atoms=meta["n_heavy_atoms"],
                frames_path=frames_path,
                frames_sha256=frames_digest,
                n_frame_models=n_models,
            )
        )
    return poses
