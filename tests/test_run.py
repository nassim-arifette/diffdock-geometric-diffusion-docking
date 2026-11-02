"""run_diffdock end to end against a fake upstream: success path, every failure path, manifest integrity, export."""
import json
import sys
import tarfile

import pytest

from docking_tools import export, run
from docking_tools.manifest import ManifestError, load_manifest, verify_manifest
from docking_tools.poses import PoseError, collect_poses


def prepare(workdir, pdb_file, ligand="COc(cc1)ccc1C#N", name="toy"):
    rd = run.new_run_dir(workdir / "runs", name)
    return run.prepare_inputs(rd, complex_name=name, receptor=str(pdb_file), ligand=ligand)


def go(workdir, prepared, monkeypatch=None, mode="ok", **kw):
    if monkeypatch is not None:
        monkeypatch.setenv("FAKE_DIFFDOCK_MODE", mode)
    return run.run_diffdock(workdir, prepared, python_exe=sys.executable, echo=False, **kw)


def test_success_writes_a_complete_manifest(fake_workdir, pdb_file, monkeypatch):
    prepared = prepare(fake_workdir, pdb_file)
    m = go(fake_workdir, prepared, monkeypatch, seed=3)
    assert m["status"] == "ok" and m["returncode"] == 0 and m["seed"] == 3
    (c,) = m["complexes"]
    assert [p["rank"] for p in c["poses"]] == list(range(1, 11))
    top = c["poses"][0]
    assert top["confidence"] == pytest.approx(1.0 - 0.0004321)  # full precision survives
    assert top["confidence_rounded"] == 1.0
    assert top["pose_id"].startswith("toy-rank01-") and len(top["sha256"]) == 64
    # what the form asked for is what was run
    assert c["ligand"]["canonical_smiles"] and c["receptor"]["removed_hetero_atoms"] == {"9X0": 1, "ZN": 1}
    assert m["checkpoints"]["files_sha256"] and m["upstream"]["commit"] == "abc" and m["patch"]["sha256"] == "def"
    assert m["timings"]["inference_wall_seconds"] >= 0 and m["environment"]["setup_seconds_total"] == 12.5
    assert "--seed" in m["command"] and m["command"][m["command"].index("--seed") + 1] == "3"
    # persisted and reloadable, with all hashes still valid
    again = load_manifest(prepared.run_dir)
    assert again == m
    verify_manifest(prepared.run_dir, again)


def test_overrides_take_effect_and_are_recorded(fake_workdir, pdb_file, monkeypatch):
    # the fake (like upstream) lets the config file win over the command line, so overrides must go through it
    m = go(fake_workdir, prepare(fake_workdir, pdb_file), monkeypatch, overrides={"samples_per_complex": 4, "inference_steps": 8})
    assert m["inference_parameters"]["samples_per_complex"] == 4 and m["inference_parameters"]["inference_steps"] == 8
    assert len(m["complexes"][0]["poses"]) == 4
    assert "--samples_per_complex" not in m["command"]


def test_unknown_override_is_rejected(fake_workdir, pdb_file):
    with pytest.raises(ValueError, match="unknown inference parameter"):
        go(fake_workdir, prepare(fake_workdir, pdb_file), overrides={"samples": 3})


def test_missing_environment_is_a_clear_error(fake_workdir, pdb_file):
    with pytest.raises(run.DockingRunError, match="setup_env"):
        run.run_diffdock(fake_workdir, prepare(fake_workdir, pdb_file), python_exe="/no/python", echo=False)


@pytest.mark.parametrize(
    "mode, needle",
    [
        ("crash", "exited with code 1"),
        ("silent", "no rank*_confidence*.sdf"),  # exit 0 but nothing written: must NOT count as success
        ("short", "expected 10 poses"),
        ("corrupt", "cannot parse"),
    ],
)
def test_failures_are_recorded_not_swallowed(fake_workdir, pdb_file, monkeypatch, mode, needle):
    prepared = prepare(fake_workdir, pdb_file)
    with pytest.raises(run.DockingRunError) as exc:
        go(fake_workdir, prepared, monkeypatch, mode=mode)
    assert needle in str(exc.value)
    m = load_manifest(prepared.run_dir)  # a failed run still leaves an inspectable manifest
    assert m["status"] == "failed" and needle in m["error"] and m["complexes"][0]["status"] == "failed"
    with pytest.raises(ManifestError, match="refusing to export"):
        export.export_run(prepared.run_dir)


def test_export_contains_every_referenced_file(fake_workdir, pdb_file, monkeypatch, tmp_path):
    prepared = prepare(fake_workdir, pdb_file)
    m = go(fake_workdir, prepared, monkeypatch)
    archive = export.export_run(prepared.run_dir, tmp_path / "out")
    names = export.list_archive(archive)
    run_id = m["run_id"]
    assert f"{run_id}/manifest.json" in names
    for p in m["complexes"][0]["poses"]:
        assert f"{run_id}/{p['path']}" in names
    assert f"{run_id}/{m['complexes'][0]['receptor']['path']}" in names
    with tarfile.open(archive) as t:  # the archived manifest is the same one
        assert json.load(t.extractfile(f"{run_id}/manifest.json"))["run_id"] == run_id


def test_export_detects_tampering(fake_workdir, pdb_file, monkeypatch):
    prepared = prepare(fake_workdir, pdb_file)
    m = go(fake_workdir, prepared, monkeypatch)
    pose = prepared.run_dir / m["complexes"][0]["poses"][2]["path"]
    pose.write_text(pose.read_text() + "\n")
    with pytest.raises(ManifestError, match="hash mismatch"):
        export.export_run(prepared.run_dir)


def test_collect_poses_rejects_non_contiguous_ranks(fake_workdir, pdb_file, monkeypatch):
    prepared = prepare(fake_workdir, pdb_file)
    go(fake_workdir, prepared, monkeypatch)
    d = prepared.run_dir / "predictions" / "toy"
    next(d.glob("rank3_confidence*.sdf")).unlink()
    with pytest.raises(PoseError, match="not contiguous"):
        collect_poses(d, "toy")


def test_collect_poses_checks_scores_against_file_names(fake_workdir, pdb_file, monkeypatch):
    prepared = prepare(fake_workdir, pdb_file)
    go(fake_workdir, prepared, monkeypatch)
    d = prepared.run_dir / "predictions" / "toy"
    s = json.loads((d / "scores.json").read_text())
    s["confidence"][4] += 1.0
    (d / "scores.json").write_text(json.dumps(s))
    with pytest.raises(PoseError, match="rank 5"):
        collect_poses(d, "toy")


def test_file_ligand_is_copied_into_the_run(fake_workdir, pdb_file, ligand_sdf, monkeypatch):
    prepared = prepare(fake_workdir, pdb_file, ligand=str(ligand_sdf))
    m = go(fake_workdir, prepared, monkeypatch)
    lf = m["complexes"][0]["ligand_file"]
    assert lf["path"] == "inputs/toy_ligand.sdf" and (prepared.run_dir / lf["path"]).exists()
    assert m["complexes"][0]["ligand"]["description"] is None  # a path is not a SMILES


def test_reference_is_recorded_with_provenance(fake_workdir, pdb_file, ligand_sdf, monkeypatch):
    rd = run.new_run_dir(fake_workdir / "runs", "ref")
    prepared = run.prepare_inputs(
        rd, complex_name="toy", receptor=str(pdb_file), ligand="CCO", reference=str(ligand_sdf), reference_note="native ligand"
    )
    m = go(fake_workdir, prepared, monkeypatch)
    ref = m["complexes"][0]["reference"]
    assert ref["provenance"] == "native ligand" and (rd / ref["path"]).exists()
