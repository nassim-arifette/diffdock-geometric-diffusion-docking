"""Rigid-body and torsion geometry of a ligand pose, as DiffDock uses it. NumPy only (RDKit helpers are optional).

A DiffDock pose update is three separate pieces:  a translation (R^3), a rotation about the ligand centroid (SO(3))
and one angle per rotatable bond (a torus).  The functions here follow the conventions of upstream
``utils/torsion.py`` and ``utils/diffusion_utils.py::modify_conformer_coordinates`` so the teaching notebook shows the
real thing:

* A torsion edge is stored as ``(u, v)`` plus a boolean ``mask`` of the atoms that move. ``v`` is on the moving side.
  The rotation axis points from ``v`` towards ``u``; a positive angle is a right-handed rotation about that axis.
* Upstream rotates the *smaller* connected component and only keeps bonds whose smaller side has more than one atom.
  The test is purely graph-based (any bond whose removal disconnects the graph), not chemical: it does not look at bond
  order, so a C=C double bond or an amide C-N bond counts as a torsion like a single bond does.
* After applying torsions, the flexible conformer is re-aligned onto the rigid-body-updated conformer with a Kabsch fit,
  so torsion updates do not leak a net rotation or translation into the pose.

Terminology: ``centroid`` is the unweighted mean of the coordinates (what DiffDock uses); a centre of mass weights by atomic
mass.  For a molecule they differ.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

Array = np.ndarray


# ------------------------------------------------------------------------------------------------ basic quantities
def as_coords(x) -> Array:
    """Float64 (N, 3) copy. Integer input is converted *before* any transformation, never assigned back into an int array."""
    a = np.array(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"expected an (N, 3) coordinate array, got shape {a.shape}")
    return a


def centroid(x) -> Array:
    """Unweighted mean position (DiffDock's 'ligand centre')."""
    return as_coords(x).mean(axis=0)


def center_of_mass(x, masses: Sequence[float]) -> Array:
    x = as_coords(x)
    m = np.asarray(masses, dtype=np.float64)
    if m.shape != (len(x),):
        raise ValueError("need one mass per atom")
    return (x * m[:, None]).sum(axis=0) / m.sum()


def translate(x, t) -> Array:
    return as_coords(x) + np.asarray(t, dtype=np.float64)


def rotation_matrix(axis, angle: float) -> Array:
    """Rodrigues: right-handed rotation by ``angle`` (radians) about ``axis`` (need not be normalised)."""
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n == 0:
        raise ValueError("rotation axis has zero length")
    k = axis / n
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def rotation_from_rotvec(v) -> Array:
    """SO(3) element from a rotation vector (axis * angle), as DiffDock's rotation update is expressed."""
    v = np.asarray(v, dtype=np.float64)
    angle = np.linalg.norm(v)
    return np.eye(3) if angle < 1e-12 else rotation_matrix(v, angle)


def rotate_about(x, R: Array, center: Optional[Sequence[float]] = None) -> Array:
    """Rotate every atom by ``R`` about ``center`` (default: the centroid)."""
    x = as_coords(x)
    c = x.mean(axis=0) if center is None else np.asarray(center, dtype=np.float64)
    return (x - c) @ R.T + c


# ------------------------------------------------------------------------------------------------ molecular graph
def adjacency_from_bonds(n_atoms: int, bonds: Iterable[Tuple[int, int]]) -> List[Set[int]]:
    adj: List[Set[int]] = [set() for _ in range(n_atoms)]
    for a, b in bonds:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _reach(adj: Sequence[Set[int]], start: int, banned: Tuple[int, int]) -> Set[int]:
    seen, stack = {start}, [start]
    while stack:
        i = stack.pop()
        for j in adj[i]:
            if {i, j} == set(banned) or j in seen:
                continue
            seen.add(j)
            stack.append(j)
    return seen


def split_at_bond(adj: Sequence[Set[int]], u: int, v: int) -> Optional[Tuple[Set[int], Set[int]]]:
    """Atoms on u's side and on v's side after cutting bond u-v, or None if u-v is in a ring (cutting it disconnects nothing)."""
    if v not in adj[u]:
        raise ValueError(f"atoms {u} and {v} are not bonded")
    side_u = _reach(adj, u, (u, v))
    if v in side_u:
        return None
    return side_u, _reach(adj, v, (u, v))


@dataclass(frozen=True)
class TorsionEdge:
    u: int
    v: int  # on the moving side
    mask: Tuple[bool, ...]  # True for atoms that move

    @property
    def n_moving(self) -> int:
        return sum(self.mask)

    def moving_atoms(self) -> List[int]:
        return [i for i, m in enumerate(self.mask) if m]


def torsion_edges(adj: Sequence[Set[int]]) -> List[TorsionEdge]:
    """The rotatable bonds DiffDock would use (see module docstring): bridges whose smaller side has > 1 atom.

    ``v`` is placed on the smaller side, which is the side that rotates. When both sides are equally large, ``u`` is
    the lower-numbered atom of the bond.
    """
    n = len(adj)
    out: List[TorsionEdge] = []
    for a in range(n):
        for b in sorted(adj[a]):
            if b < a:
                continue
            sides = split_at_bond(adj, a, b)
            if sides is None:
                continue
            side_a, side_b = sides
            small, u, v = (side_a, b, a) if len(side_a) < len(side_b) else (side_b, a, b)
            if len(small) > 1:
                mask = tuple(i in small for i in range(n))
                out.append(TorsionEdge(u=u, v=v, mask=mask))
    return out


# ------------------------------------------------------------------------------------------------ torsion
def apply_torsion(x, edge: TorsionEdge, angle: float) -> Array:
    """Rotate the atoms of ``edge.mask`` by ``angle`` radians about the axis through the bond (pointing v -> u).

    Float coordinates are produced even for integer input. Every bond length and angle within each side, and the
    geometry of the bond itself, is preserved; only the dihedral across the bond changes.
    """
    x = as_coords(x)
    axis = x[edge.u] - x[edge.v]
    if np.linalg.norm(axis) == 0:
        raise ValueError("torsion bond has zero length")
    R = rotation_matrix(axis, angle)
    mask = np.asarray(edge.mask, dtype=bool)
    out = x.copy()
    out[mask] = (x[mask] - x[edge.v]) @ R.T + x[edge.v]
    return out


def dihedral(x, a: int, b: int, c: int, d: int) -> float:
    """IUPAC dihedral angle a-b-c-d in degrees, in (-180, 180]."""
    x = as_coords(x)
    b0, b1, b2 = x[a] - x[b], x[c] - x[b], x[d] - x[c]
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


# ------------------------------------------------------------------------------------------------ alignment / RMSD
def kabsch(P, Q) -> Tuple[Array, Array]:
    """Best rigid transform taking P onto Q: ``Q ~= P @ R.T + t``. R is a proper rotation (det +1, never a reflection)."""
    P, Q = as_coords(P), as_coords(Q)
    if P.shape != Q.shape:
        raise ValueError("P and Q must have the same shape")
    cp, cq = P.mean(axis=0), Q.mean(axis=0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cq - R @ cp
    return R, t


def align(P, Q) -> Array:
    """P moved onto Q by the best rigid transform."""
    R, t = kabsch(P, Q)
    return as_coords(P) @ R.T + t


def rmsd(P, Q) -> float:
    """RMSD *in place*, atom i of P against atom i of Q, no fitting. This is the docking RMSD: it includes placement error."""
    P, Q = as_coords(P), as_coords(Q)
    if P.shape != Q.shape:
        raise ValueError("P and Q must have the same shape")
    return float(np.sqrt(((P - Q) ** 2).sum(axis=1).mean()))


def aligned_rmsd(P, Q) -> float:
    """RMSD after optimally superimposing P on Q: measures conformation difference only and hides translation/rotation error."""
    return rmsd(align(P, Q), Q)


# ------------------------------------------------------------------------------------------------ checks
def bond_lengths(x, bonds: Iterable[Tuple[int, int]]) -> Array:
    x = as_coords(x)
    return np.array([np.linalg.norm(x[a] - x[b]) for a, b in bonds])


def max_bond_length_change(before, after, bonds: Sequence[Tuple[int, int]]) -> float:
    return float(np.max(np.abs(bond_lengths(after, bonds) - bond_lengths(before, bonds)))) if len(bonds) else 0.0


def max_pairwise_distance_change(before, after) -> float:
    a, b = as_coords(before), as_coords(after)
    da = np.linalg.norm(a[:, None] - a[None], axis=-1)
    db = np.linalg.norm(b[:, None] - b[None], axis=-1)
    return float(np.abs(da - db).max())


# ------------------------------------------------------------------------------------------------ DiffDock-style update
def pose_update(
    x,
    edges: Sequence[TorsionEdge],
    translation=(0.0, 0.0, 0.0),
    rotvec=(0.0, 0.0, 0.0),
    torsions: Optional[Sequence[float]] = None,
) -> Array:
    """One DiffDock pose update (mirrors upstream ``modify_conformer_coordinates``).

    1. rigid part: rotate about the centroid by ``rotvec`` (axis * angle, radians), then translate;
    2. torsions: rotate each edge's moving side (sequentially, each axis taken from the current coordinates);
    3. re-align the flexible conformer onto the rigid one with Kabsch, so torsions add no net rotation/translation.
    """
    x = as_coords(x)
    center = x.mean(axis=0)
    rigid = (x - center) @ rotation_from_rotvec(rotvec).T + np.asarray(translation, dtype=np.float64) + center
    if torsions is None or len(edges) == 0 or not np.any(np.asarray(torsions) != 0):
        return rigid
    if len(torsions) != len(edges):
        raise ValueError(f"{len(edges)} torsion edges but {len(torsions)} angles")
    flex = rigid.copy()
    for edge, ang in zip(edges, torsions):
        if ang != 0:
            flex = apply_torsion(flex, edge, ang)
    return align(flex, rigid)


# ------------------------------------------------------------------------------------------------ RDKit helpers (optional)
def heavy_atom_graph(mol) -> Tuple[Array, List[Tuple[int, int]], List[Set[int]], List[str]]:
    """Coordinates, bonds, adjacency and symbols of the heavy atoms of an RDKit molecule with a conformer.

    DiffDock removes hydrogens by default, so its torsion definition is evaluated on the heavy-atom graph.
    """
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    new = {old: i for i, old in enumerate(heavy)}
    pos = mol.GetConformer().GetPositions()[heavy]
    bonds = [(new[b.GetBeginAtomIdx()], new[b.GetEndAtomIdx()]) for b in mol.GetBonds()
             if b.GetBeginAtomIdx() in new and b.GetEndAtomIdx() in new]
    adj = adjacency_from_bonds(len(heavy), bonds)
    symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in heavy]
    return pos, bonds, adj, symbols


def bond_orders(mol) -> Dict[Tuple[int, int], float]:
    """Bond order (1.0, 1.5, 2.0...) keyed by the *heavy-atom* index pair used by :func:`heavy_atom_graph`."""
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    new = {old: i for i, old in enumerate(heavy)}
    out = {}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in new and j in new:
            out[tuple(sorted((new[i], new[j])))] = b.GetBondTypeAsDouble()
    return out
