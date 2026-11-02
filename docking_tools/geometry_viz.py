"""3D views and numeric check panels for the geometry teaching notebook (needs RDKit + py3Dmol)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import geometry as g

BEFORE_COLOR = "#8a8f98"  # grey carbon: where the molecule started
AFTER_COLOR = "#e8830c"  # orange carbon: after the transformation
AXIS_COLOR = "#d62728"
MOVING_COLOR = "#1f77b4"


def mol_with_coords(mol, coords):
    """Heavy-atom copy of ``mol`` (same atom order as :func:`geometry.heavy_atom_graph`) placed at ``coords``."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    heavy = Chem.RemoveHs(Chem.Mol(mol))
    coords = g.as_coords(coords)
    if heavy.GetNumAtoms() != len(coords):
        raise ValueError(f"{heavy.GetNumAtoms()} heavy atoms but {len(coords)} coordinates")
    conf = heavy.GetConformer()
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    return heavy


def _block(mol, coords) -> str:
    from rdkit import Chem

    return Chem.MolToMolBlock(mol_with_coords(mol, coords))


def build_view(
    mol,
    before,
    after=None,
    *,
    edge: Optional[g.TorsionEdge] = None,
    labels: bool = True,
    width: int = 560,
    height: int = 420,
):
    """Overlay of ``before`` (grey) and ``after`` (orange). With ``edge``: the rotation axis (red arrow, drawn from the
    pivot atom ``v`` towards ``u``) and the atoms that move (blue spheres). Atom indices are those used everywhere else."""
    import py3Dmol

    before = g.as_coords(before)
    view = py3Dmol.view(width=width, height=height)
    view.addModel(_block(mol, before), "mol")
    view.setStyle({"model": 0}, {"stick": {"color": BEFORE_COLOR, "radius": 0.12}})
    shown = before
    if after is not None:
        after = g.as_coords(after)
        view.addModel(_block(mol, after), "mol")
        view.setStyle({"model": 1}, {"stick": {"colorscheme": "orangeCarbon", "radius": 0.17}})
        shown = after
    if edge is not None:
        axis = before[edge.u] - before[edge.v]
        axis = axis / np.linalg.norm(axis)
        start, end = before[edge.v], before[edge.v] + 2.2 * axis
        view.addArrow({
            "start": dict(x=float(start[0]), y=float(start[1]), z=float(start[2])),
            "end": dict(x=float(end[0]), y=float(end[1]), z=float(end[2])),
            "radius": 0.09, "color": AXIS_COLOR,
        })
        for i in edge.moving_atoms():
            view.addStyle({"model": 0, "index": i}, {"sphere": {"radius": 0.3, "color": MOVING_COLOR, "opacity": 0.55}})
    if labels:
        for i, p in enumerate(shown):
            view.addLabel(str(i), {
                "position": dict(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                "fontSize": 11, "fontColor": "black", "backgroundOpacity": 0.0, "showBackground": False,
                "inFront": True, "alignment": "center",
            })
    view.zoomTo()
    return view


def sweep_view(mol, coords, edge: g.TorsionEdge, n_frames: int = 36, width: int = 480, height: int = 380):
    """Animated sweep of one torsion over a full turn. An illustration produced by rotating a bond, not model output."""
    import py3Dmol
    from rdkit import Chem

    sdf = "".join(
        Chem.MolToMolBlock(mol_with_coords(mol, g.apply_torsion(coords, edge, 2 * np.pi * k / n_frames))) + "$$$$\n"
        for k in range(n_frames)
    )
    view = py3Dmol.view(width=width, height=height)
    view.addModelsAsFrames(sdf, "sdf")
    view.setStyle({"stick": {"colorscheme": "orangeCarbon", "radius": 0.16}})
    view.animate({"loop": "forward", "interval": 90})
    view.zoomTo()
    return view


def torsion_table(mol, edges: Sequence[g.TorsionEdge], symbols: Sequence[str]) -> List[Dict]:
    """One row per torsion DiffDock would sample, with the chemistry upstream's graph-only rule ignores."""
    orders = g.bond_orders(mol)
    rows = []
    for k, e in enumerate(edges):
        order = orders[tuple(sorted((e.u, e.v)))]
        rows.append({
            "edge": k,
            "bond": f"{symbols[e.u]}{e.u} - {symbols[e.v]}{e.v}",
            "axis": f"{e.v} -> {e.u}",
            "bond order": order,
            "atoms that move": e.n_moving,
            "note": "" if order == 1.0 else "not a single bond, but upstream still treats it as a torsion",
        })
    return rows


def checks_text(
    before,
    after,
    bonds: Sequence[Tuple[int, int]],
    *,
    expected_centroid_shift=None,
    moving_mask: Optional[Sequence[bool]] = None,
) -> str:
    """Numbers that must hold after a transformation, formatted for printing."""
    before, after = g.as_coords(before), g.as_coords(after)
    lines = [
        f"max bond-length change        : {g.max_bond_length_change(before, after, bonds):.2e} A",
        f"max pairwise-distance change  : {g.max_pairwise_distance_change(before, after):.2e} A   (0 = rigid motion)",
        f"centroid shift                : {np.round(g.centroid(after) - g.centroid(before), 4).tolist()}",
        f"in-place RMSD before -> after : {g.rmsd(after, before):.3f} A",
        f"RMSD after best superposition : {g.aligned_rmsd(after, before):.3f} A",
    ]
    if expected_centroid_shift is not None:
        err = np.linalg.norm((g.centroid(after) - g.centroid(before)) - np.asarray(expected_centroid_shift))
        lines.append(f"centroid shift vs requested   : error {err:.2e} A")
    if moving_mask is not None:
        m = np.asarray(moving_mask, dtype=bool)
        lines.append(f"largest move of the fixed side: {np.linalg.norm(after[~m] - before[~m], axis=1).max():.2e} A")
    return "\n".join(lines)


def live_panel(mol, before, bonds, controls: Dict[str, object], compute, *, labels: bool = True):
    """Interactive panel: ``controls`` maps names to ipywidgets; ``compute(**values)`` returns a dict with ``after`` and
    optionally ``edge``, ``expected_centroid_shift``, ``moving_mask``. Every widget change redraws the overlay and the checks."""
    import ipywidgets as w
    from IPython.display import display

    out = w.Output()

    def render(_=None):
        values = {name: ctl.value for name, ctl in controls.items()}
        result = compute(**values)
        after = result["after"]
        with out:
            out.clear_output(wait=True)
            build_view(mol, before, after, edge=result.get("edge"), labels=labels).show()
            print(checks_text(before, after, bonds, expected_centroid_shift=result.get("expected_centroid_shift"),
                              moving_mask=result.get("moving_mask")))
            if result.get("note"):
                print(result["note"])

    for ctl in controls.values():
        ctl.observe(render, names="value")
    display(w.VBox([w.HBox(list(controls.values()), layout=w.Layout(flex_flow="row wrap")), out]))
    render()
    return controls
