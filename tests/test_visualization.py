import sys

import pytest

from docking_tools import run, visualization


@pytest.fixture
def finished_run(fake_workdir, pdb_file, monkeypatch):
    monkeypatch.setenv("FAKE_DIFFDOCK_MODE", "ok")
    rd = run.new_run_dir(fake_workdir / "runs", "toy")
    prepared = run.prepare_inputs(rd, complex_name="toy", receptor=str(pdb_file), ligand="COc(cc1)ccc1C#N")
    run.run_diffdock(fake_workdir, prepared, python_exe=sys.executable, echo=False)
    return rd


def test_pose_table_pairs_full_precision_score_with_file(finished_run):
    from docking_tools.manifest import load_manifest

    rows = visualization.pose_table(load_manifest(finished_run))
    assert len(rows) == 10 and rows[0]["rank"] == 1
    assert rows[0]["confidence"] != rows[0]["confidence_in_filename"]  # 0.99957 vs 1.0
    assert rows[0]["file"].endswith("rank1_confidence1.00.sdf")


def test_show_pose_loads_the_predicted_ligand_not_just_the_protein(finished_run):
    view = visualization.show_pose(finished_run, rank=3)
    html = view._make_html()
    assert html.count("addModel") == 2  # receptor + ligand
    ligand_sdf = (finished_run / "predictions" / "toy").glob("rank3_confidence*.sdf")
    first_atom_line = next(ligand_sdf).read_text().splitlines()[4]
    assert first_atom_line.split()[0] in html  # coordinates of that specific pose are embedded


def test_show_pose_unknown_rank(finished_run):
    with pytest.raises(KeyError):
        visualization.show_pose(finished_run, rank=99)
