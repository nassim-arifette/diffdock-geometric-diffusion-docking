import pytest

from docking_tools import inputs
from docking_tools.inputs import InputError, InputRow


def test_clean_receptor_keeps_only_protein_first_model(pdb_file, tmp_path):
    out = tmp_path / "clean.pdb"
    info = inputs.clean_receptor(pdb_file, out)
    text = out.read_text()
    assert "9X0" not in text and "HOH" not in text and " ZN " not in text
    assert "99.104" not in text  # second MODEL dropped
    assert "BSER" not in text  # alternate location B dropped
    assert "ASER" not in text and "CA  SER" in text  # alternate location A kept, altloc flag blanked
    assert "MSE" not in text and "MET B   3" in text and " SD " in text  # selenomethionine -> methionine
    assert info.removed_hetero == {"9X0": 1, "ZN": 1}
    assert info.removed_waters == 1
    assert info.chains == ["A", "B"]
    assert info.n_residues == 3


def test_clean_receptor_chain_filter(pdb_file, tmp_path):
    info = inputs.clean_receptor(pdb_file, tmp_path / "a.pdb", chains=["A"])
    assert info.chains == ["A"]
    assert "SER B" not in (tmp_path / "a.pdb").read_text()


def test_clean_receptor_without_protein_fails(tmp_path):
    p = tmp_path / "lig_only.pdb"
    p.write_text("HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n")
    with pytest.raises(InputError):
        inputs.clean_receptor(p, tmp_path / "o.pdb")


def test_resolve_receptor_pdb_id_downloads(monkeypatch, pdb_file, tmp_path):
    monkeypatch.setattr(inputs, "_fetch", lambda url, timeout=60: pdb_file.read_bytes())
    info = inputs.resolve_receptor("6AGT", tmp_path / "w")
    assert info.source == "rcsb:6AGT" and info.path.name == "6AGT_protein.pdb"
    assert (tmp_path / "w" / "raw" / "6AGT.pdb").exists()


def test_resolve_receptor_rejects_garbage(tmp_path):
    with pytest.raises(InputError):
        inputs.resolve_receptor("not a pdb id", tmp_path)


def test_resolve_ligand_smiles_and_errors(ligand_sdf):
    lig = inputs.resolve_ligand("COc(cc1)ccc1C#N")
    assert lig.kind == "smiles" and lig.n_heavy_atoms == 10 and lig.canonical_smiles
    with pytest.raises(InputError):
        inputs.resolve_ligand("this is not smiles ((")
    assert inputs.resolve_ligand(str(ligand_sdf)).kind == "file"
    with pytest.raises(InputError):
        inputs.resolve_ligand("/no/such/file.sdf")


def test_resolve_ligand_pubchem_cid(monkeypatch):
    monkeypatch.setattr(inputs, "_fetch", lambda url, timeout=60: b"COc1ccc(C#N)cc1\n")
    lig = inputs.resolve_ligand("12345")
    assert lig.source == "pubchem:12345" and lig.kind == "smiles"


def test_csv_has_the_upstream_schema(pdb_file, tmp_path):
    csv_path = inputs.write_input_csv(tmp_path / "in.csv", [InputRow("c1", str(pdb_file), "CCO")])
    lines = csv_path.read_text().splitlines()
    # header copied from upstream data/protein_ligand_example.csv at the pinned commit
    assert lines[0] == "complex_name,protein_path,ligand_description,protein_sequence"
    assert lines[1].endswith(",CCO,")


def test_csv_is_read_like_upstream_does(pdb_file, tmp_path):
    import pandas as pd

    csv_path = inputs.write_input_csv(tmp_path / "in.csv", [InputRow("c1", str(pdb_file), "CCO")])
    df = pd.read_csv(csv_path)  # upstream: df['protein_sequence'] -> set_nones
    assert [s if str(s) != "nan" else None for s in df["protein_sequence"].tolist()] == [None]
    assert df["complex_name"].tolist() == ["c1"] and df["ligand_description"].tolist() == ["CCO"]


@pytest.mark.parametrize(
    "rows, msg",
    [
        ([InputRow("a", "{pdb}", "CCO"), InputRow("a", "{pdb}", "CCN")], "duplicate"),
        ([InputRow("bad name", "{pdb}", "CCO")], "complex_name"),
        ([InputRow("a", "/missing.pdb", "CCO")], "receptor file not found"),
        ([InputRow("a", "{pdb}", "")], "empty ligand"),
        ([InputRow("a", "{pdb}", "/missing.sdf")], "ligand file not found"),
        ([InputRow("a", "", "CCO")], "empty protein_path"),
    ],
)
def test_csv_validation_rejects(pdb_file, tmp_path, rows, msg):
    rows = [InputRow(r.complex_name, r.protein_path.replace("{pdb}", str(pdb_file)), r.ligand_description) for r in rows]
    with pytest.raises(InputError, match=msg):
        inputs.write_input_csv(tmp_path / "in.csv", rows)
