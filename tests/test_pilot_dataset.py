import zipfile

import pytest

from docking_tools.pilot_dataset import load_final_ids, split_cases, _members, _eligibility


def test_final_ids_reject_old428_and_duplicates(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("1ABC_LIG\n1ABC_LIG\n")
    with pytest.raises(ValueError, match="unique"):
        load_final_ids(path, expected_count=2)
    with pytest.raises(ValueError, match="308"):
        load_final_ids(path)


def test_split_is_deterministic_order_independent_and_pdb_disjoint():
    cases = [{"complex_id": f"{i:04d}_ABC"} for i in range(20)] + [{"complex_id": "0001_DEF"}]
    selected = split_cases(cases, n_dev=3, n_eval=5)
    assert selected == split_cases(list(reversed(cases)), n_dev=3, n_eval=5)
    assert len({c["complex_id"].split("_")[0] for c in selected}) == 8
    assert [c["split"] for c in selected].count("dev") == 3


def test_archive_resolution_does_not_extract_arbitrary_paths(tmp_path):
    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("../../escape.txt", "bad")
        zf.writestr("dataset/1ABC_X/1abc_x_protein.pdb", "protein")
        zf.writestr("dataset/1ABC_X/1abc_x_ligands.sdf", "ligand")
    with zipfile.ZipFile(path) as zf:
        resolved = _members(zf, ["1ABC_X"])
        assert set(resolved["1ABC_X"]) == {"receptor", "reference"}
    assert not (tmp_path.parent / "escape.txt").exists()


def test_pilot_excludes_receptors_with_metals(pdb_file, ligand_sdf):
    assert _eligibility(pdb_file, ligand_sdf) == "receptor_contains_metal:ZN"
