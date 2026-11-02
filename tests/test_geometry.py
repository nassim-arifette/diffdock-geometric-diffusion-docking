"""Geometry of a pose: every claim the teaching notebook makes is checked here, including the bugs of the old toy code."""
import numpy as np
import pytest

from docking_tools import geometry as g

rng = np.random.default_rng(0)
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


@pytest.fixture(scope="module")
def mol3d():
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(IBUPROFEN))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    return mol


@pytest.fixture(scope="module")
def ligand(mol3d):
    pos, bonds, adj, symbols = g.heavy_atom_graph(mol3d)
    return {"x": pos, "bonds": bonds, "adj": adj, "edges": g.torsion_edges(adj), "symbols": symbols, "mol": mol3d}


def random_rotation():
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
        [2*(b*c + a*d), a*a - b*b + c*c - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c), 2*(c*d + a*b), a*a - b*b - c*c + d*d],
    ])


# -------------------------------------------------------------------------------------------- the bugs of the old toy
LEGACY_LIGAND = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 1, 0], [4, 1, 1]])  # int array, as in Diffdock_test.ipynb


def legacy_apply_torsion(ligand, bond_idx, angle):
    """Verbatim logic of the original notebook helper (global z axis, 'atoms after bond_idx', writes into input dtype)."""
    rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    out = ligand.copy()
    centre = ligand[bond_idx]
    for i in range(bond_idx + 1, len(ligand)):
        out[i] = centre + rot @ (ligand[i] - centre)
    return out


def test_legacy_helper_truncates_integer_coordinates():
    bonds = [(0, 1), (1, 2), (2, 3), (3, 4)]
    before = g.bond_lengths(LEGACY_LIGAND, bonds)
    after = g.bond_lengths(legacy_apply_torsion(LEGACY_LIGAND, 2, np.pi / 4), bonds)
    assert before[2] == pytest.approx(np.sqrt(2))  # 1.4142...
    assert after[2] == pytest.approx(1.0)  # silently shortened: the rotated point was cast to int
    assert not np.allclose(before, after)


def test_legacy_helper_rotates_about_global_z_not_the_bond():
    """Atoms 0-2 lie on the x axis; the 'bond' 2-3 points along (1,1,0). The old code never uses that direction."""
    floaty = LEGACY_LIGAND.astype(float)
    new = legacy_apply_torsion(floaty, 2, np.pi / 2)
    # a torsion about bond 2-3 must leave atom 3 (on the axis) in place; the legacy code moves it
    assert not np.allclose(new[3], floaty[3])


def test_new_helper_has_neither_problem():
    adj = g.adjacency_from_bonds(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    edge = next(e for e in g.torsion_edges(adj) if {e.u, e.v} == {2, 3})
    before = LEGACY_LIGAND
    after = g.apply_torsion(before, edge, np.pi / 4)
    bonds = [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert g.max_bond_length_change(before, after, bonds) < 1e-12
    on_axis = edge.u if not edge.mask[edge.u] else edge.v
    np.testing.assert_allclose(after[edge.v], before[edge.v], atol=1e-12)  # the pivot atom does not move
    assert after.dtype == np.float64


# -------------------------------------------------------------------------------------------- rigid motions
def test_rotation_matrix_is_proper_and_right_handed():
    R = g.rotation_matrix([0, 0, 1], np.pi / 2)
    np.testing.assert_allclose(R @ np.eye(3), np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), atol=1e-12)
    for _ in range(20):
        R = g.rotation_matrix(rng.normal(size=3), rng.uniform(-np.pi, np.pi))
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)


def test_rotvec_matches_axis_angle_and_zero_is_identity():
    np.testing.assert_allclose(g.rotation_from_rotvec([0, 0, 0]), np.eye(3))
    np.testing.assert_allclose(g.rotation_from_rotvec(np.array([0, 0, 1.0]) * 0.7), g.rotation_matrix([0, 0, 1], 0.7))


def test_translation_and_rotation_preserve_all_distances(ligand):
    x = ligand["x"]
    t = g.translate(x, [3, -2, 1])
    assert g.max_pairwise_distance_change(x, t) < 1e-10
    np.testing.assert_allclose(g.centroid(t), g.centroid(x) + [3, -2, 1], atol=1e-10)
    r = g.rotate_about(x, g.rotation_matrix([1, 2, 3], 1.1))
    assert g.max_pairwise_distance_change(x, r) < 1e-10
    np.testing.assert_allclose(g.centroid(r), g.centroid(x), atol=1e-10)  # rotates about the centroid


def test_centroid_is_not_centre_of_mass(ligand):
    masses = [{"C": 12.011, "O": 15.999}[s] for s in ligand["symbols"]]
    com, cen = g.center_of_mass(ligand["x"], masses), g.centroid(ligand["x"])
    assert np.linalg.norm(com - cen) > 1e-3  # distinct quantities for a heteroatom-containing molecule
    np.testing.assert_allclose(g.center_of_mass(ligand["x"], np.ones(len(ligand["x"]))), cen)


def test_as_coords_rejects_bad_shapes():
    with pytest.raises(ValueError):
        g.as_coords([[1, 2], [3, 4]])


# -------------------------------------------------------------------------------------------- torsion edges
def test_chain_split_and_ring_detection():
    adj = g.adjacency_from_bonds(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 2), (3, 5)])  # ring 2-3-4, tail 0-1, 5
    assert g.split_at_bond(adj, 1, 2) == ({0, 1}, {2, 3, 4, 5})
    assert g.split_at_bond(adj, 2, 3) is None  # ring bond: cutting it disconnects nothing
    with pytest.raises(ValueError):
        g.split_at_bond(adj, 0, 3)  # not bonded


def test_torsion_edges_on_ibuprofen_match_upstream_rule(ligand):
    edges, adj = ligand["edges"], ligand["adj"]
    assert len(edges) == 4  # iPr-CH2, CH2-aryl, aryl-CH, CH-COOH  (methyl and carbonyl-oxygen bonds are terminal)
    n = len(ligand["x"])
    for e in edges:
        sides = g.split_at_bond(adj, e.u, e.v)
        assert sides is not None
        side_u, side_v = sides
        moving = set(e.moving_atoms())
        assert moving == side_v and e.v in moving and e.u not in moving  # v is on the rotating side
        assert len(side_v) <= len(side_u) and len(side_v) > 1  # the smaller side rotates; terminal bonds are excluded
        assert len(moving) + len(side_u) == n
    ring_atoms = {i for i, a in enumerate(adj) if any(g.split_at_bond(adj, i, j) is None for j in a)}
    assert all(e.u not in ring_atoms or e.v not in ring_atoms for e in edges)


def test_upstream_rule_is_graph_based_not_chemical():
    """An amide-like C-N bond (partial double bond) and a C=C both count, because only connectivity is tested."""
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)NC(C)C=CC"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    pos, bonds, adj, _ = g.heavy_atom_graph(mol)
    orders = g.bond_orders(mol)
    kinds = {orders[tuple(sorted((e.u, e.v)))] for e in g.torsion_edges(adj)}
    assert 2.0 in kinds  # the C=C bond is selected although it is a double bond


# -------------------------------------------------------------------------------------------- torsion application
def test_torsion_preserves_geometry_and_changes_only_the_dihedral(ligand):
    x, bonds, adj = ligand["x"], ligand["bonds"], ligand["adj"]
    for edge in ligand["edges"]:
        for angle in (0.3, -1.2, np.pi):
            y = g.apply_torsion(x, edge, angle)
            assert g.max_bond_length_change(x, y, bonds) < 1e-10
            fixed = ~np.asarray(edge.mask)
            np.testing.assert_allclose(y[fixed], x[fixed], atol=1e-12)  # the other side does not move at all
            np.testing.assert_allclose(y[edge.v], x[edge.v], atol=1e-12)  # pivot stays
            # distances inside each side are unchanged
            for side in (fixed, ~fixed):
                assert g.max_pairwise_distance_change(x[side], y[side]) < 1e-10


def test_torsion_changes_the_dihedral_by_the_requested_angle(ligand):
    x, adj = ligand["x"], ligand["adj"]
    edge = ligand["edges"][1]
    a = next(i for i in adj[edge.u] if i != edge.v)  # neighbour of u on the fixed side
    d = next(i for i in adj[edge.v] if i != edge.u)  # neighbour of v on the moving side
    for angle_deg in (15.0, 90.0, -60.0):
        before = g.dihedral(x, a, edge.u, edge.v, d)
        after = g.dihedral(g.apply_torsion(x, edge, np.radians(angle_deg)), a, edge.u, edge.v, d)
        delta = (after - before + 180) % 360 - 180
        # sign convention: positive angle = right-handed about the axis pointing v -> u, which is *minus* the IUPAC change
        assert abs(delta) == pytest.approx(abs(angle_deg), abs=1e-6)
        assert delta == pytest.approx(-angle_deg, abs=1e-6)


def test_torsion_composition_and_inverse(ligand):
    x, edge = ligand["x"], ligand["edges"][0]
    y = g.apply_torsion(g.apply_torsion(x, edge, 0.4), edge, 0.7)
    np.testing.assert_allclose(y, g.apply_torsion(x, edge, 1.1), atol=1e-10)
    np.testing.assert_allclose(g.apply_torsion(y, edge, -1.1), x, atol=1e-10)


def test_torsion_on_zero_length_bond_is_an_error():
    adj = g.adjacency_from_bonds(4, [(0, 1), (1, 2), (2, 3)])
    edge = g.torsion_edges(adj)[0]
    x = np.zeros((4, 3))
    with pytest.raises(ValueError):
        g.apply_torsion(x, edge, 0.5)


# -------------------------------------------------------------------------------------------- Kabsch / RMSD
def test_kabsch_recovers_a_random_rigid_transform():
    P = rng.normal(size=(12, 3))
    R0, t0 = random_rotation(), rng.normal(size=3)
    Q = P @ R0.T + t0
    R, t = g.kabsch(P, Q)
    np.testing.assert_allclose(R, R0, atol=1e-10)
    np.testing.assert_allclose(t, t0, atol=1e-10)
    assert g.aligned_rmsd(P, Q) < 1e-10


def test_kabsch_never_returns_a_reflection():
    P = rng.normal(size=(10, 3))
    Q = P * np.array([1, 1, -1])  # mirror image: no proper rotation maps P onto Q exactly
    R, _ = g.kabsch(P, Q)
    assert np.linalg.det(R) == pytest.approx(1.0)
    assert g.aligned_rmsd(P, Q) > 0.1  # so the residual is honest, not hidden by a reflection


def test_in_place_rmsd_sees_placement_error_aligned_rmsd_does_not():
    ref = rng.normal(size=(15, 3))
    shifted = ref + np.array([3.0, 0, 0])
    assert g.rmsd(shifted, ref) == pytest.approx(3.0)  # docking metric: a displaced pose is wrong
    assert g.aligned_rmsd(shifted, ref) < 1e-10  # alignment would call the same pose perfect
    assert g.rmsd(ref, ref) == 0.0


def test_rmsd_shape_mismatch():
    with pytest.raises(ValueError):
        g.rmsd(np.zeros((3, 3)), np.zeros((4, 3)))


# -------------------------------------------------------------------------------------------- DiffDock-style update
def test_pose_update_identity_and_rigid_only(ligand):
    x, edges = ligand["x"], ligand["edges"]
    np.testing.assert_allclose(g.pose_update(x, edges), x, atol=1e-12)
    np.testing.assert_allclose(g.pose_update(x, edges, torsions=[0] * len(edges)), x, atol=1e-12)
    t = np.array([1.0, 2.0, -3.0])
    y = g.pose_update(x, edges, translation=t, rotvec=[0.2, -0.4, 0.9])
    assert g.max_pairwise_distance_change(x, y) < 1e-10
    np.testing.assert_allclose(g.centroid(y), g.centroid(x) + t, atol=1e-10)


def test_pose_update_torsions_do_not_leak_rotation_or_translation(ligand):
    x, edges, bonds = ligand["x"], ligand["edges"], ligand["bonds"]
    tors = [0.8, -1.1, 0.5, 2.0]
    y = g.pose_update(x, edges, torsions=tors)
    assert g.max_bond_length_change(x, y, bonds) < 1e-9
    np.testing.assert_allclose(g.centroid(y), g.centroid(x), atol=1e-9)  # no net translation
    # no net rotation: the Kabsch re-alignment is what removes the one a raw torsion sequence would induce
    raw = x.copy()
    for e, a in zip(edges, tors):
        raw = g.apply_torsion(raw, e, a)
    assert g.rmsd(y, x) < g.rmsd(raw, x) + 1e-9
    R, _ = g.kabsch(y, x)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_pose_update_all_three_pieces_compose(ligand):
    x, edges, bonds = ligand["x"], ligand["edges"], ligand["bonds"]
    t, rv, tors = np.array([0.5, 0.5, 0.5]), np.array([0.3, 0.1, -0.2]), [0.5, 0.5, -0.5, 1.0]
    y = g.pose_update(x, edges, t, rv, tors)
    assert g.max_bond_length_change(x, y, bonds) < 1e-9
    np.testing.assert_allclose(g.centroid(y), g.centroid(x) + t, atol=1e-9)
    with pytest.raises(ValueError):
        g.pose_update(x, edges, torsions=[0.1])  # wrong number of angles
