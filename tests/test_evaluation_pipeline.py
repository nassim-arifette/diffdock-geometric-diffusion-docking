"""Integration at the saved-run boundary, including infrastructure failures."""
import json
import sys

import pytest

from docking_tools import evaluation, run


def finished_file_input(fake_workdir, pdb_file, ligand_sdf):
    rd = run.new_run_dir(fake_workdir / "runs", "file_input")
    prepared = run.prepare_inputs(rd, complex_name="toy", receptor=str(pdb_file),
                                  ligand=str(ligand_sdf), reference=str(ligand_sdf),
                                  reference_note="test crystal reference")
    run.run_diffdock(fake_workdir, prepared, python_exe=sys.executable, echo=False,
                    overrides={"samples_per_complex": 2})
    return rd


def test_a_file_input_run_can_be_evaluated_without_missing_identity(fake_workdir, pdb_file, ligand_sdf):
    rd = finished_file_input(fake_workdir, pdb_file, ligand_sdf)
    result = evaluation.evaluate_run(rd)
    assert result["status"] == "ok"
    poses = result["complexes"][0]["poses"]
    assert len(poses) == 2
    assert all(p["checks"]["input_connectivity_and_charge"] for p in poses)
    assert all(isinstance(p["rmsd_angstrom"], float) for p in poses)


def test_validator_crash_is_not_reported_as_measured_zero_validity(
    fake_workdir, pdb_file, ligand_sdf, monkeypatch,
):
    rd = finished_file_input(fake_workdir, pdb_file, ligand_sdf)

    class BrokenBackend:
        config_sha256 = "0" * 64

        def check(self, *args):
            raise RuntimeError("physical-validation backend crashed")

    monkeypatch.setattr(evaluation, "PhysicalValidator", BrokenBackend)
    with pytest.raises(RuntimeError, match="backend crashed"):
        evaluation.evaluate_run(rd)
    report = json.loads((rd / "evaluation.json").read_text())
    assert report["status"] == "failed"


def test_refined_validator_crash_is_not_mislabeled_as_a_gnina_failure(
    fake_workdir, pdb_file, ligand_sdf, monkeypatch,
):
    import shutil
    rd = finished_file_input(fake_workdir, pdb_file, ligand_sdf)
    real_validator = evaluation.PhysicalValidator()

    class BreakOnRefined:
        config_sha256 = real_validator.config_sha256
        calls = 0

        def check(self, *args):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("refined validation backend crashed")
            return real_validator.check(*args)

    class Scorer:
        def __init__(self, executable):
            self.info = {"version": "test"}

        def run(self, receptor, pose, output, **kwargs):
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pose, output)
            return {"score": 1.0}

    monkeypatch.setattr(evaluation, "PhysicalValidator", BreakOnRefined)
    monkeypatch.setattr(evaluation, "Gnina", Scorer)
    with pytest.raises(RuntimeError, match="refined validation backend crashed"):
        evaluation.evaluate_run(rd, gnina="test")
