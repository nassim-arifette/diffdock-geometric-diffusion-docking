"""Fixed candidate pools, resumed scoring and denominators across failures."""
import json
import sys
from types import SimpleNamespace

import pytest

from docking_tools import pilot
from docking_tools._util import sha256_file, write_json


def protocol_file(tmp_path, pdb_file, ligand_sdf):
    path = tmp_path / "pilot.json"
    write_json(path, {
        "schema_version": 1, "benchmark_id": "test", "seeds": [7], "samples_per_complex": 3,
        "complexes": [{"complex_id": "toy", "split": "dev", "receptor": str(pdb_file),
                       "reference": str(ligand_sdf), "receptor_sha256": sha256_file(pdb_file),
                       "reference_sha256": sha256_file(ligand_sdf)}],
    })
    return path


def evaluator(monkeypatch, *, fail=False):
    def evaluate_run(run_dir, *, gnina=None, output=None):
        assert not output.exists(), "a previous evaluation attempt must be retained"
        if fail:
            write_json(output, {"status": "failed", "error": "scorer unavailable"})
            raise RuntimeError("scorer unavailable")
        result = {"status": "ok", "complexes": [{"complex_name": "toy", "poses": []}]}
        write_json(output, result)
        return result
    monkeypatch.setitem(sys.modules, "docking_tools.evaluation", SimpleNamespace(evaluate_run=evaluate_run))


def test_pilot_passes_only_chemical_identity_and_reuses_verified_results(
    fake_workdir, pdb_file, ligand_sdf, tmp_path, monkeypatch,
):
    path = protocol_file(tmp_path, pdb_file, ligand_sdf)
    evaluator(monkeypatch)
    real = pilot.run_diffdock
    calls = []

    def dock(workdir, prepared, **kwargs):
        calls.append(prepared)
        assert prepared.complexes[0].ligand.kind == "smiles"
        assert prepared.complexes[0].ligand_file is None
        assert prepared.complexes[0].reference.is_file()
        return real(workdir, prepared, **kwargs)

    monkeypatch.setattr(pilot, "run_diffdock", dock)
    out = tmp_path / "results"
    first = pilot.run_pilot(path, workdir=fake_workdir, out=out, skip_setup=True, python_exe=sys.executable)
    assert first["status"] == "complete" and first["n_expected"] == first["n_ok"] == 1
    second = pilot.run_pilot(path, workdir=fake_workdir, out=out, skip_setup=True, python_exe=sys.executable)
    assert second["n_ok"] == 1 and len(calls) == 1


def test_failure_remains_in_denominator_and_retry_does_not_resample(
    fake_workdir, pdb_file, ligand_sdf, tmp_path, monkeypatch,
):
    path = protocol_file(tmp_path, pdb_file, ligand_sdf)
    evaluator(monkeypatch, fail=True)
    out = tmp_path / "results"
    first = pilot.run_pilot(path, workdir=fake_workdir, out=out, skip_setup=True, python_exe=sys.executable)
    assert first["n_expected"] == first["n_failed"] == 1 and first["n_ok"] == 0
    case = first["cases"]["toy_seed7"]
    before = sha256_file(out / case["run_dir"] / "manifest.json")
    evaluator(monkeypatch)
    monkeypatch.setattr(pilot, "run_diffdock", lambda *a, **kw: pytest.fail("regenerated a frozen candidate pool"))
    second = pilot.run_pilot(path, workdir=fake_workdir, out=out, skip_setup=True, retry_failed=True)
    assert second["n_ok"] == 1
    recovered = second["cases"]["toy_seed7"]
    assert recovered["run_dir"] == case["run_dir"] and recovered["manifest_sha256"] == before
    assert recovered["evaluation_attempt"] == 2
    assert json.loads((out / case["run_dir"] / "evaluation.attempt1.json").read_text())["status"] == "failed"


def test_changed_input_or_protocol_is_rejected_before_sampling(pdb_file, ligand_sdf, tmp_path):
    path = protocol_file(tmp_path, pdb_file, ligand_sdf)
    pdb_file.write_text("modified receptor")
    with pytest.raises(pilot.PilotError, match="hash changed"):
        pilot.read_protocol(path)


def test_same_complex_cannot_appear_on_both_splits(pdb_file, ligand_sdf, tmp_path):
    path = protocol_file(tmp_path, pdb_file, ligand_sdf)
    data = json.loads(path.read_text())
    data["complexes"].append(dict(data["complexes"][0], split="eval"))
    write_json(path, data)
    with pytest.raises(pilot.PilotError, match="repeated complex_id"):
        pilot.read_protocol(path)


def test_changed_evaluation_is_not_silently_accepted(
    fake_workdir, pdb_file, ligand_sdf, tmp_path, monkeypatch,
):
    path = protocol_file(tmp_path, pdb_file, ligand_sdf)
    evaluator(monkeypatch)
    out = tmp_path / "results"
    first = pilot.run_pilot(path, workdir=fake_workdir, out=out, skip_setup=True, python_exe=sys.executable)
    entry = first["cases"]["toy_seed7"]
    (out / entry["run_dir"] / "evaluation.json").write_text("{}")
    with pytest.raises(pilot.PilotError, match="recorded result changed"):
        pilot.run_pilot(path, workdir=fake_workdir, out=out, skip_setup=True)
