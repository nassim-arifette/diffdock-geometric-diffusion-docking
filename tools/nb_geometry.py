"""notebooks/01_geometry.ipynb"""
import nbformat as nbf

from nbtools import bootstrap, code, finish, md


def build():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# The geometry behind DiffDock: translation, rotation, torsion

DiffDock does not generate atom coordinates directly. It moves a ligand through the space of **poses**, which is a
product of three pieces:

| piece | dimensions | what it does |
|---|---|---|
| translation | 3 | moves the ligand's centroid |
| rotation | 3 (SO(3)) | rotates the whole ligand about its centroid |
| torsions | one per rotatable bond (a torus) | rotates the smaller side of a bond about the bond axis |

Diffusion adds noise to these coordinates and the network learns to remove it. This notebook builds each piece by hand on
a **real molecule** (RDKit conformer, atom indices shown), checks the invariants numerically, and then shows how DiffDock
**composes** them. Everything here is plain geometry: nothing is model output. (Real reverse-diffusion frames come from
DiffDock's `--save_visualisation` and are shown in `04_pose_inspector.ipynb`.)

**Reading the pictures.** Grey = where the molecule started, orange = after the transformation, red arrow = rotation
axis, blue spheres = atoms that move in a torsion. The numbers under each picture must hold for the transformation to be
valid.
"""),
        bootstrap(),
        md("## 2. A real molecule"),
        code('''
#@title 2. Molecule
SMILES = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"  #@param {type:"string"}
SEED = 42  #@param {type:"integer"}

import numpy as np
import pandas as pd
from IPython.display import display
from rdkit import Chem
from rdkit.Chem import AllChem

from docking_tools import geometry as g
from docking_tools import geometry_viz as gv

mol = Chem.AddHs(Chem.MolFromSmiles(SMILES))
if AllChem.EmbedMolecule(mol, randomSeed=SEED) != 0:
    raise ValueError(f"RDKit could not build a 3D conformer for {SMILES!r}")
X, BONDS, ADJ, SYMBOLS = g.heavy_atom_graph(mol)  # DiffDock works on the heavy-atom graph by default
EDGES = g.torsion_edges(ADJ)
print(f"{len(X)} heavy atoms, {len(BONDS)} bonds, {len(EDGES)} torsions")
gv.build_view(mol, X).show()
'''),
        md("""
### Which bonds are torsions?
DiffDock's rule (upstream `get_transformation_mask`) is **graph-based**: a bond is a torsion when cutting it splits the molecule
and the smaller side has more than one atom; the smaller side is the one that rotates. It never looks at bond order, so a
double bond or an amide C-N bond is treated like a single bond. The table flags those.
"""),
        code('''
display(pd.DataFrame(gv.torsion_table(mol, EDGES, SYMBOLS)))
'''),
        md("""
### Centroid or centre of mass?
DiffDock rotates about the **centroid** (the unweighted mean of the atom positions). A centre of mass weights atoms by mass.
They are different points, so the two words are not interchangeable.
"""),
        code('''
MASS = {"C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06, "F": 18.998, "Cl": 35.45, "Br": 79.904, "P": 30.974}
masses = [MASS.get(s, 12.0) for s in SYMBOLS]
print("centroid       :", np.round(g.centroid(X), 3))
print("centre of mass :", np.round(g.center_of_mass(X, masses), 3))
print("distance between them: %.3f A" % np.linalg.norm(g.centroid(X) - g.center_of_mass(X, masses)))
'''),
        md("## 3. Translation\nThree numbers. Every distance inside the molecule must stay the same, and the centroid moves by exactly the requested vector."),
        code('''
import ipywidgets as w

def slider(name, lo, hi, step, value=0.0):
    return w.FloatSlider(value=value, min=lo, max=hi, step=step, description=name, continuous_update=False,
                         layout=w.Layout(width="260px"))

gv.live_panel(
    mol, X, BONDS,
    {"tx": slider("shift x (A)", -8, 8, 0.1), "ty": slider("shift y (A)", -8, 8, 0.1), "tz": slider("shift z (A)", -8, 8, 0.1)},
    lambda tx, ty, tz: {"after": g.translate(X, [tx, ty, tz]), "expected_centroid_shift": [tx, ty, tz]},
)
'''),
        md("""
## 4. Rotation
DiffDock expresses the rotation update as a **rotation vector**: its direction is the axis and its length is the angle
(radians). The rotation is about the centroid, so the centroid does not move and all distances are preserved.
"""),
        code('''
def rotation_case(rx, ry, rz):
    rv = np.array([rx, ry, rz])
    R = g.rotation_from_rotvec(rv)
    angle = np.degrees(np.linalg.norm(rv))
    return {"after": g.rotate_about(X, R), "expected_centroid_shift": [0, 0, 0],
            "note": f"rotation angle {angle:.1f} deg; det(R) = {np.linalg.det(R):.6f}; R R^T = I within {np.abs(R @ R.T - np.eye(3)).max():.1e}"}

gv.live_panel(
    mol, X, BONDS,
    {"rx": slider("rot x (rad)", -3.14, 3.14, 0.05), "ry": slider("rot y (rad)", -3.14, 3.14, 0.05), "rz": slider("rot z (rad)", -3.14, 3.14, 0.05)},
    rotation_case,
)
'''),
        md("""
## 5. Torsion
Pick a rotatable bond. The **red arrow** is the axis, drawn from the pivot atom `v` towards `u`; the **blue spheres** are the
atoms that rotate (the smaller side). Bond lengths, bond angles and everything on the fixed side must not change; only the
dihedral across the bond does. A positive angle is a right-handed rotation about the arrow.
"""),
        code('''
edge_choice = w.Dropdown(
    options=[(f"edge {k}: {SYMBOLS[e.u]}{e.u} - {SYMBOLS[e.v]}{e.v}  ({e.n_moving} atoms move)", k) for k, e in enumerate(EDGES)],
    description="bond", layout=w.Layout(width="420px"))

def torsion_case(edge_k, angle_deg):
    edge = EDGES[edge_k]
    after = g.apply_torsion(X, edge, np.radians(angle_deg))
    a = next(i for i in ADJ[edge.u] if i != edge.v)
    d = next(i for i in ADJ[edge.v] if i != edge.u)
    change = (g.dihedral(after, a, edge.u, edge.v, d) - g.dihedral(X, a, edge.u, edge.v, d) + 180) % 360 - 180
    return {"after": after, "edge": edge, "moving_mask": edge.mask,
            "note": f"dihedral {a}-{edge.u}-{edge.v}-{d} changes by {change:+.1f} deg for a requested {angle_deg:+.1f} deg "
                    f"(opposite sign: the axis points v -> u)"}

gv.live_panel(mol, X, BONDS, {"edge_k": edge_choice, "angle_deg": slider("angle (deg)", -180, 180, 1.0)}, torsion_case)
'''),
        md("""
### What the first version of this notebook got wrong
The original `apply_torsion` rotated about a **global z axis** through one atom (not about the bond), assumed that atoms
*later in the array* are the ones downstream of the bond, and ran on an **integer** coordinate array, so every rotated point
was truncated to integers when written back. The cell below runs a verbatim copy of that helper next to the correct one on
its own 5-atom toy chain.
"""),
        code('''
def legacy_apply_torsion(ligand, bond_idx, angle):  # kept only to show the bug
    rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    out = ligand.copy()
    centre = ligand[bond_idx]
    for i in range(bond_idx + 1, len(ligand)):
        out[i] = centre + rot @ (ligand[i] - centre)
    return out

chain = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 1, 0], [4, 1, 1]])          # integer array, as before
chain_bonds = [(0, 1), (1, 2), (2, 3), (3, 4)]
chain_edge = next(e for e in g.torsion_edges(g.adjacency_from_bonds(5, chain_bonds)) if {e.u, e.v} == {2, 3})

old = legacy_apply_torsion(chain, 2, np.pi / 4)
new = g.apply_torsion(chain, chain_edge, np.pi / 4)
print("bond 2-3 length   before: %.4f" % g.bond_lengths(chain, chain_bonds)[2])
print("                  old   : %.4f   <- shortened by integer truncation" % g.bond_lengths(old, chain_bonds)[2])
print("                  fixed : %.4f" % g.bond_lengths(new, chain_bonds)[2])
print("atom 3 (on the bond) moved by the old code: %.3f A; by the fixed code: %.3f A"
      % (np.linalg.norm(old[3] - chain[3]), np.linalg.norm(new[chain_edge.v] - chain[chain_edge.v])))
'''),
        md("""
## 6. Putting them together the way DiffDock does
One DiffDock step applies (1) the rotation about the centroid and the translation, (2) the torsions, one bond after another,
and then (3) **re-aligns the flexible result onto the rigid one with a Kabsch fit**, so that torsion updates cannot smuggle in a
net rotation or translation. Each piece has its own controls.
"""),
        code('''
controls = {
    "tx": slider("shift x", -5, 5, 0.1), "ty": slider("shift y", -5, 5, 0.1), "tz": slider("shift z", -5, 5, 0.1),
    "rx": slider("rot x", -3.14, 3.14, 0.05), "ry": slider("rot y", -3.14, 3.14, 0.05), "rz": slider("rot z", -3.14, 3.14, 0.05),
}
for k, e in enumerate(EDGES):
    controls[f"tor{k}"] = slider(f"torsion {k} (deg)", -180, 180, 1.0)

def full_update(**v):
    t = [v["tx"], v["ty"], v["tz"]]
    tors = [np.radians(v[f"tor{k}"]) for k in range(len(EDGES))]
    return {"after": g.pose_update(X, EDGES, t, [v["rx"], v["ry"], v["rz"]], tors), "expected_centroid_shift": t}

gv.live_panel(mol, X, BONDS, controls, full_update, labels=False)
'''),
        md("""
## 7. Comparing poses: why docking RMSD is not computed after alignment
Superimposing a prediction on the reference before measuring RMSD throws away exactly the error docking cares about.
`rmsd` below is **in place** (atom *i* against atom *i*, no fitting); `aligned_rmsd` first finds the best rigid transform.
"""),
        code('''
rng = np.random.default_rng(0)
cases = {
    "translated by 3 A": g.pose_update(X, EDGES, translation=[3, 0, 0]),
    "rotated by 90 deg": g.pose_update(X, EDGES, rotvec=[0, 0, np.pi / 2]),
    "all torsions +60 deg": g.pose_update(X, EDGES, torsions=[np.radians(60)] * len(EDGES)),
    "translated + one torsion": g.pose_update(X, EDGES, translation=[3, 0, 0], torsions=[np.radians(90)] + [0] * (len(EDGES) - 1)),
}
display(pd.DataFrame(
    {"in-place RMSD (A)": [g.rmsd(p, X) for p in cases.values()],
     "RMSD after superposition (A)": [g.aligned_rmsd(p, X) for p in cases.values()]},
    index=list(cases)).round(3))
print("A translated or rotated copy is far from the reference in place, yet has ~0 RMSD after superposition.")
'''),
        md("## 8. One torsion, swept through a full turn\nAn *illustration*: a single bond rotated through 360 degrees by the function above. It is not output of the model."),
        code('''
gv.sweep_view(mol, X, EDGES[min(1, len(EDGES) - 1)]).show()
'''),
        md("""
**Next.** The same translation/rotation/torsion decomposition is what the pose inspector (`04_pose_inspector.ipynb`) shows for real
DiffDock runs, frame by frame, from the reverse-diffusion frames DiffDock saves with `--save_visualisation`.
"""),
    ]
    return finish(nb, "01_geometry.ipynb")
