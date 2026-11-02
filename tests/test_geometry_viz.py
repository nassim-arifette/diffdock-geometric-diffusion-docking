import numpy as np
import pytest

from docking_tools import geometry as g
from docking_tools import geometry_viz as gv

Chem = pytest.importorskip("rdkit.Chem")
pytest.importorskip("py3Dmol")


@pytest.fixture(scope="module")
def setup():
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O"))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    x, bonds, adj, symbols = g.heavy_atom_graph(mol)
    return mol, x, bonds, g.torsion_edges(adj), symbols


def test_mol_with_coords_places_atoms_and_checks_size(setup):
    mol, x, *_ = setup
    placed = gv.mol_with_coords(mol, x + 5.0)
    np.testing.assert_allclose(placed.GetConformer().GetPositions(), x + 5.0, atol=1e-6)
    assert placed.GetNumAtoms() == len(x)
    with pytest.raises(ValueError):
        gv.mol_with_coords(mol, x[:-1])


def test_build_view_overlays_before_and_after_with_labels(setup):
    mol, x, bonds, edges, _ = setup
    after = g.apply_torsion(x, edges[0], 0.8)
    html = gv.build_view(mol, x, after, edge=edges[0])._make_html()
    assert html.count("addModel") == 2
    assert "addArrow" in html  # rotation axis
    assert html.count("addLabel") == len(x)  # one index label per heavy atom
    assert html.count('"sphere"') >= edges[0].n_moving  # moving side highlighted


def test_build_view_without_edge_or_labels(setup):
    mol, x, *_ = setup
    html = gv.build_view(mol, x, labels=False)._make_html()
    assert html.count("addModel") == 1 and "addArrow" not in html and "addLabel" not in html


def test_sweep_view_contains_every_frame_and_animates(setup):
    mol, x, bonds, edges, _ = setup
    html = gv.sweep_view(mol, x, edges[1], n_frames=12)._make_html()
    assert html.count("$$$$") >= 12 and "animate" in html


def test_torsion_table_flags_non_single_bonds():
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)NC(C)C=CC"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    x, bonds, adj, symbols = g.heavy_atom_graph(mol)
    rows = gv.torsion_table(mol, g.torsion_edges(adj), symbols)
    assert any(r["bond order"] == 2.0 and "not a single bond" in r["note"] for r in rows)
    assert all(r["atoms that move"] > 1 for r in rows)


def test_checks_text_reports_the_numbers(setup):
    mol, x, bonds, edges, _ = setup
    t = np.array([1.0, 0.0, 0.0])
    text = gv.checks_text(x, g.translate(x, t), bonds, expected_centroid_shift=t)
    assert "max bond-length change" in text and "0.00e+00" in text or "e-1" in text
    assert "in-place RMSD before -> after : 1.000 A" in text
    assert "RMSD after best superposition : 0.000 A" in text
    y = g.apply_torsion(x, edges[0], 1.0)
    text2 = gv.checks_text(x, y, bonds, moving_mask=edges[0].mask)
    assert "largest move of the fixed side: 0.00e+00" in text2


def test_live_panel_redraws_when_a_control_changes(setup, capsys):
    ipywidgets = pytest.importorskip("ipywidgets")
    mol, x, bonds, edges, _ = setup
    calls = []

    def compute(t):
        calls.append(t)
        return {"after": g.translate(x, [t, 0, 0]), "expected_centroid_shift": [t, 0, 0]}

    slider = ipywidgets.FloatSlider(value=0.0, min=-5, max=5, description="t")
    controls = gv.live_panel(mol, x, bonds, {"t": slider}, compute)
    assert calls == [0.0]  # drawn once on creation
    slider.value = 2.5  # observers run synchronously
    assert calls == [0.0, 2.5]
