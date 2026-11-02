import copy
import math

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from docking_tools.evaluation import _identity, receptor_frame_rmsd, select_variants, PhysicalValidator


def molecule(smiles="CCO"):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    return Chem.RemoveHs(mol)


def test_docking_rmsd_keeps_receptor_frame_and_does_not_move_pose():
    reference = molecule()
    pose = Chem.Mol(reference)
    before = pose.GetConformer().GetPositions().copy()
    for i, xyz in enumerate(before):
        pose.GetConformer().SetAtomPosition(i, (float(xyz[0] + 8), float(xyz[1]), float(xyz[2])))
    assert receptor_frame_rmsd(pose, reference) == pytest.approx(8.0)
    assert pose.GetConformer().GetAtomPosition(0).x == pytest.approx(before[0, 0] + 8)


def test_docking_rmsd_handles_atom_reordering_and_symmetry():
    reference = molecule("c1ccccc1")
    pose = Chem.RenumberAtoms(reference, list(reversed(range(reference.GetNumAtoms()))))
    assert receptor_frame_rmsd(pose, reference) == pytest.approx(0.0, abs=1e-8)
    with pytest.raises(ValueError, match="search limit"):
        receptor_frame_rmsd(pose, reference, max_matches=1)


def test_wrong_identity_is_not_a_good_rmsd():
    with pytest.raises(ValueError, match="different connectivity"):
        receptor_frame_rmsd(molecule("CCN"), molecule("CCO"))


def test_symmetry_includes_resonance_equivalent_carboxylate_oxygens():
    reference = molecule("CC(=O)[O-]")
    pose = Chem.Mol(reference)
    oxygens = [a.GetIdx() for a in pose.GetAtoms() if a.GetAtomicNum() == 8]
    xyz = pose.GetConformer().GetPositions().copy()
    pose.GetConformer().SetAtomPosition(oxygens[0], xyz[oxygens[1]])
    pose.GetConformer().SetAtomPosition(oxygens[1], xyz[oxygens[0]])
    assert receptor_frame_rmsd(pose, reference) == pytest.approx(0.0, abs=1e-8)


def test_identity_is_checked_from_3d_not_just_existing_sdf_tags():
    pose = molecule("N[C@@H](C)C(=O)O")
    expected = Chem.MolFromSmiles("N[C@@H](C)C(=O)O")
    assert all(_identity(pose, expected).values())
    # Mirror coordinates while keeping the original graph's stereo tags.
    for i, xyz in enumerate(pose.GetConformer().GetPositions()):
        pose.GetConformer().SetAtomPosition(i, (float(-xyz[0]), float(xyz[1]), float(xyz[2])))
    assert _identity(pose, expected)["input_stereochemistry"] is False


def rows():
    return [
        dict(pose_id="a", confidence=5.0, valid=False, rmsd_angstrom=0.5, gnina_score=0.9, refined=None),
        dict(pose_id="b", confidence=4.0, valid=True, rmsd_angstrom=4.0, gnina_score=0.8, refined=None),
        dict(pose_id="c", confidence=3.0, valid=True, rmsd_angstrom=1.0, gnina_score=0.7, refined=None),
    ]


def test_filter_and_scores_cannot_peek_at_reference_rmsd():
    data = rows()
    first = select_variants(data, gnina_available=True)
    assert first["A_confidence"]["selected_pose_ids"][0] == "a"
    assert first["A_confidence"]["top1_rmsd_success"] is True
    assert first["A_confidence"]["top1_joint_success"] is False
    assert first["B_valid_confidence"]["selected_pose_ids"][0] == "b"
    assert first["B_valid_confidence"]["top1_joint_success"] is False
    assert first["B_valid_confidence"]["top5_joint_success"] is True
    changed = copy.deepcopy(data)
    for p in changed:
        p["rmsd_angstrom"] = 100 - p["rmsd_angstrom"]
    second = select_variants(changed, gnina_available=True)
    for variant in first:
        assert first[variant]["selected_pose_ids"] == second[variant]["selected_pose_ids"]


def test_no_valid_candidate_is_failure_not_fallback_or_dropped_case():
    data = rows()
    for p in data:
        p["valid"] = False
    result = select_variants(data, gnina_available=True)
    for name in ("B_valid_confidence", "C_valid_gnina", "D_refined_gnina"):
        assert result[name]["empty_selection"] is True
        assert result[name]["top1_joint_success"] is False


def test_unrun_gnina_is_missing_not_measured_zero_and_nonfinite_score_is_refused():
    result = select_variants(rows())
    assert result["C_valid_gnina"]["available"] is False
    assert result["C_valid_gnina"]["top1_joint_success"] is None
    data = rows()
    data[0]["confidence"] = math.nan
    with pytest.raises(ValueError, match="finite"):
        select_variants(data)


def test_no_crystal_reference_means_unknown_accuracy_not_zero():
    result = select_variants(rows(), reference_available=False)
    assert result["A_confidence"]["top1_rmsd_success"] is None
    assert result["A_confidence"]["top1_joint_success"] is None
    assert result["A_confidence"]["top1_valid"] is False


def test_posebusters_validity_has_no_reference_pose_argument(pdb_file, ligand_sdf):
    validator = PhysicalValidator()
    expected = Chem.SDMolSupplier(str(ligand_sdf), removeHs=True)[0]
    _, checks = validator.check(ligand_sdf, pdb_file, expected)
    assert len(checks) > 10
    assert all(type(v) is bool for v in checks.values())
    assert not any("rmsd" in name.lower() for name in checks)
    assert checks["input_connectivity_and_charge"]
