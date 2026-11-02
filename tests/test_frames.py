import sys

import numpy as np
import pytest

from docking_tools import frames, run
from docking_tools.manifest import load_manifest
from docking_tools.poses import PoseError, collect_poses

Chem = pytest.importorskip("rdkit.Chem")


@pytest.fixture
def run_with_frames(fake_workdir, pdb_file, monkeypatch):
    monkeypatch.setenv("FAKE_DIFFDOCK_MODE", "ok")
    rd = run.new_run_dir(fake_workdir / "runs", "toy")
    prepared = run.prepare_inputs(rd, complex_name="toy", receptor=str(pdb_file), ligand="COc(cc1)ccc1C#N")
    manifest = run.run_diffdock(
        fake_workdir, prepared, python_exe=sys.executable, echo=False,
        overrides={"samples_per_complex": 3, "save_visualisation": True},
    )
    return rd, manifest, fake_workdir, prepared


def test_manifest_records_a_frames_file_per_pose(run_with_frames):
    rd, manifest, *_ = run_with_frames
    for p in manifest["complexes"][0]["poses"]:
        assert p["frames"]["path"].endswith(f"rank{p['rank']}_reverseprocess.pdb")
        assert p["frames"]["n_models"] == 3 + 1 + 4  # conformer, initial, start, final-in-slot-3, steps 1..4
        assert (rd / p["frames"]["path"]).is_file()


def test_frames_are_covered_by_manifest_verification(run_with_frames):
    from docking_tools.manifest import ManifestError, verify_manifest

    rd, manifest, *_ = run_with_frames
    pose = manifest["complexes"][0]["poses"][0]
    (rd / pose["frames"]["path"]).write_text("MODEL\nENDMDL\n")
    with pytest.raises(ManifestError, match="hash mismatch"):
        verify_manifest(rd, manifest)


def test_requested_frames_that_are_missing_fail_the_run(fake_workdir, pdb_file, monkeypatch):
    monkeypatch.setenv("FAKE_DIFFDOCK_MODE", "ok")
    rd = run.new_run_dir(fake_workdir / "runs", "toy")
    prepared = run.prepare_inputs(rd, complex_name="toy", receptor=str(pdb_file), ligand="CCO")
    run.run_diffdock(fake_workdir, prepared, python_exe=sys.executable, echo=False)  # no frames requested: fine
    d = rd / "predictions" / "toy"
    assert collect_poses(d, "toy")[0].frames_path is None
    with pytest.raises(PoseError, match="frames were requested"):
        collect_poses(d, "toy", expect_frames=True)


def test_trajectory_recovers_start_steps_and_final(run_with_frames):
    rd, manifest, *_ = run_with_frames
    pose = manifest["complexes"][0]["poses"][0]
    traj = frames.trajectory(rd / pose["frames"]["path"], rd / pose["path"])
    kinds = [f.kind for f in traj.frames]
    assert kinds[0] == "random_start" and kinds[-1] == "final" and set(kinds[1:-1]) == {"denoising_step"}
    assert traj.n_models_in_file == 8
    assert any("overwrites the slot of denoising step 0" in n for n in traj.notes)
    steps = [f.step for f in traj.frames if f.kind == "denoising_step"]
    assert steps == sorted(steps) and steps[0] == 1  # step 0 is the overwritten slot: not recoverable
    final = frames.sdf_heavy_coords(rd / pose["path"])
    np.testing.assert_allclose(traj.frames[-1].coords, final, atol=0.02)
    # the trajectory moves monotonically from the noisy start to the final pose
    dist = [float(np.sqrt(((f.coords - final) ** 2).sum(1).mean())) for f in traj.frames]
    assert dist == sorted(dist, reverse=True) and dist[0] > 5 and dist[-1] < 0.02
    assert traj.coords().shape == (len(traj.frames), len(final), 3)


def test_hydrogens_are_not_part_of_the_frames(run_with_frames):
    rd, manifest, *_ = run_with_frames
    pose = manifest["complexes"][0]["poses"][0]
    models = frames.read_models(rd / pose["frames"]["path"])
    assert any(" H" in line[76:78] for line in models[0])  # upstream's models carry (stale) hydrogens ...
    traj = frames.trajectory(rd / pose["frames"]["path"], rd / pose["path"])
    assert len(traj.frames[0].coords) == manifest["complexes"][0]["poses"][0]["n_heavy_atoms"]  # ... we drop them


def test_a_file_that_does_not_match_the_pose_is_an_error(run_with_frames):
    rd, manifest, *_ = run_with_frames
    p0, p1 = manifest["complexes"][0]["poses"][:2]
    with pytest.raises(frames.FrameError, match="none of the"):
        frames.trajectory(rd / p0["frames"]["path"], rd / p1["path"])  # frames of pose 1 against the SDF of pose 2


def test_unexpected_layout_is_reported_not_guessed(tmp_path, run_with_frames):
    rd, manifest, *_ = run_with_frames
    pose = manifest["complexes"][0]["poses"][0]
    models = frames.read_models(rd / pose["frames"]["path"])
    short = tmp_path / "short.pdb"
    short.write_text("".join("MODEL\n" + "\n".join(m) + "\nENDMDL\n" for m in models[:2]))
    with pytest.raises(frames.FrameError):
        frames.trajectory(short, rd / pose["path"])
    (tmp_path / "empty.pdb").write_text("")
    with pytest.raises(frames.FrameError, match="no MODEL"):
        frames.read_models(tmp_path / "empty.pdb")


def test_multimodel_pdb_export_has_one_model_per_frame(tmp_path, run_with_frames):
    rd, manifest, *_ = run_with_frames
    pose = manifest["complexes"][0]["poses"][0]
    traj = frames.trajectory(rd / pose["frames"]["path"], rd / pose["path"])
    mol = Chem.SDMolSupplier(str(rd / pose["path"]), sanitize=False)[0]
    text = frames.to_multimodel_pdb(mol, traj)
    assert text.count("MODEL") == len(traj.frames) and text.count("ENDMDL") == len(traj.frames)
    out = tmp_path / "traj.pdb"
    out.write_text(text)
    models = frames.read_models(out)
    assert len(models) == len(traj.frames) and len(frames.heavy_coords(models[0])) == len(traj.frames[0].coords)
