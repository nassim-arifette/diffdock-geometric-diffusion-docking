import sys
import zipfile
from pathlib import Path

import pytest

from docking_tools import benchmark_prep as bp
from docking_tools.setup_env import Layout


def make_zip(path: Path, ids, missing_ligand=()):
    with zipfile.ZipFile(path, "w") as z:
        for i in ids:
            z.writestr(f"PDBBind_processed/{i}/{i}_protein_processed.pdb", "ATOM")
            if i not in missing_ligand:
                z.writestr(f"PDBBind_processed/{i}/{i}_ligand.sdf", "sdf")
            z.writestr(f"PDBBind_processed/{i}/{i}_ligand.mol2", "mol2")
        z.writestr("PDBBind_processed/9zzz/9zzz_protein_processed.pdb", "SHOULD NOT BE EXTRACTED")
        z.writestr("PDBBind_processed/9zzz/9zzz_ligand.sdf", "x")
    return path


def test_extract_subset_only_unpacks_requested_ids(tmp_path):
    z = make_zip(tmp_path / "p.zip", ["1a0q", "2xyz"])
    out = bp.extract_subset(z, ["1A0Q"], tmp_path / "data")
    assert set(out) == {"1a0q"}
    assert sorted(p.name for p in (tmp_path / "data").iterdir()) == ["1a0q"]  # 2xyz and 9zzz untouched
    assert (tmp_path / "data" / "1a0q" / "1a0q_ligand.sdf").exists()


def test_extract_subset_unknown_id(tmp_path):
    z = make_zip(tmp_path / "p.zip", ["1a0q"])
    with pytest.raises(bp.BenchmarkPrepError, match="not found"):
        bp.extract_subset(z, ["1a0q", "7abc"], tmp_path / "d")


def test_extract_subset_incomplete_complex(tmp_path):
    z = make_zip(tmp_path / "p.zip", ["1a0q"], missing_ligand=["1a0q"])
    with pytest.raises(bp.BenchmarkPrepError, match="incomplete"):
        bp.extract_subset(z, ["1a0q"], tmp_path / "d")


def test_read_split_ids(tmp_path):
    f = tmp_path / "split"
    f.write_text("6QQW\n6d08\n\n6qqw\n6jap\n")
    assert bp.read_split_ids(f) == ["6qqw", "6d08", "6jap"]
    assert bp.read_split_ids(f, limit=2) == ["6qqw", "6d08"]
    f.write_text("not-an-id\n")
    with pytest.raises(bp.BenchmarkPrepError):
        bp.read_split_ids(f)


def test_prepare_subset_orchestration_with_fake_env(tmp_path, monkeypatch):
    """Everything except torch/ESM: the subset is unpacked, FASTA + embeddings are requested for it alone."""
    wd = tmp_path / "wd"
    (wd / "DiffDock" / "datasets").mkdir(parents=True)
    (wd / "DiffDock" / "datasets" / "esm_embedding_preparation.py").write_text(
        "import argparse, os\n"
        "p = argparse.ArgumentParser(); p.add_argument('--data_dir'); p.add_argument('--out_file'); p.add_argument('--dataset')\n"
        "a = p.parse_args()\n"
        "open(a.out_file, 'w').write(''.join(f'>{n}_chain_0\\nACDEFG\\n' for n in sorted(os.listdir(a.data_dir))))\n"
    )
    fake_esm = tmp_path / "fake_esm.py"
    fake_esm.write_text("import sys; open(sys.argv[2], 'wb').write(b'embeddings for ' + open(sys.argv[1], 'rb').read())\n")
    monkeypatch.setattr(bp, "ESM_SCRIPT", fake_esm)
    monkeypatch.setattr(Layout, "python", property(lambda self: Path(sys.executable)))
    z = make_zip(tmp_path / "p.zip", ["1a0q", "2xyz", "3abc"])
    s = bp.prepare_subset(wd, ["1a0q", "3abc"], pdbbind_zip=z)
    assert s["n_complexes"] == 2 and s["n_sequences"] == 2
    assert bp.fasta_ids(Path(s["fasta"])) == ["1a0q_chain_0", "3abc_chain_0"]  # 2xyz was never processed
    assert Path(s["split_file"]).read_text().split() == ["1a0q", "3abc"]
    assert (Path(s["data_dir"]).parent / "subset_manifest.json").exists()


def test_prepare_subset_fails_when_embedding_step_fails(tmp_path, monkeypatch):
    wd = tmp_path / "wd"
    (wd / "DiffDock" / "datasets").mkdir(parents=True)
    (wd / "DiffDock" / "datasets" / "esm_embedding_preparation.py").write_text(
        "import argparse; p = argparse.ArgumentParser(); p.add_argument('--data_dir'); p.add_argument('--out_file'); p.add_argument('--dataset')\n"
        "a = p.parse_args(); open(a.out_file, 'w').write('>1a0q_chain_0\\nACD\\n')\n"
    )
    bad = tmp_path / "bad_esm.py"
    bad.write_text("import sys; sys.exit(3)\n")
    monkeypatch.setattr(bp, "ESM_SCRIPT", bad)
    monkeypatch.setattr(Layout, "python", property(lambda self: Path(sys.executable)))
    with pytest.raises(bp.BenchmarkPrepError, match="embedding generation failed"):
        bp.prepare_subset(wd, ["1a0q"], pdbbind_zip=make_zip(tmp_path / "p.zip", ["1a0q"]))


def test_real_upstream_fasta_script_on_the_bundled_example(tmp_path):
    """Run upstream's own esm_embedding_preparation.py (needs biopython) on its bundled 1a0q complex."""
    pytest.importorskip("Bio")
    from test_setup_env import _upstream_source  # noqa: PLC0415

    src = _upstream_source()
    if src is None:
        pytest.skip("upstream not reachable")
    from docking_tools import setup_env

    mp = pytest.MonkeyPatch()
    mp.setattr(setup_env, "UPSTREAM_REPO", src)
    try:
        layout = Layout(tmp_path / "wd")
        setup_env.clone_upstream(layout)
    finally:
        mp.undo()
    data = tmp_path / "PDBBind_processed"
    (data / "1a0q").mkdir(parents=True)
    for f in (layout.repo / "data" / "1a0q").iterdir():
        (data / "1a0q" / f.name).write_bytes(f.read_bytes())
    mp2 = pytest.MonkeyPatch()
    mp2.setattr(Layout, "python", property(lambda self: Path(sys.executable)))
    try:
        fasta = bp.make_fasta(layout, data, tmp_path / "out.fasta")  # the real function, the real upstream script
    finally:
        mp2.undo()
    ids = bp.fasta_ids(fasta)
    assert ids and all(i.startswith("1a0q_chain_") for i in ids)
    seq_lines = [l for l in fasta.read_text().splitlines() if l and not l.startswith(">")]
    assert seq_lines and all(set(l) <= set("ACDEFGHIKLMNPQRSTVWY-") for l in seq_lines)
