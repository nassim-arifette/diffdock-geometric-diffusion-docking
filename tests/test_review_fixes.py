"""Regression tests for the defects an independent review of Phase 1 found (each was checked against the code first)."""
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

from docking_tools import _util, benchmark_prep, drive, inputs, run, setup_env
from docking_tools.inputs import InputError, InputRow
from docking_tools.setup_env import Layout


# ------------------------------------------------------------------------------------------ PubChem stereochemistry
def test_pubchem_requests_the_isomeric_smiles_property():
    assert inputs.PUBCHEM_URL.endswith("/property/SMILES/TXT")  # `CanonicalSMILES` drops stereo, `SMILES` keeps it
    assert inputs.PUBCHEM_NAME_URL.endswith("/property/SMILES/TXT")
    assert "CanonicalSMILES" not in inputs.PUBCHEM_URL + inputs.PUBCHEM_NAME_URL


def test_unspecified_stereocenters_are_counted():
    flat = inputs.check_smiles("C[C@H](N)C(=O)O")  # alanine with one defined centre
    assert flat["unspecified_stereocenters"] == 0
    assert inputs.check_smiles("CC(N)C(=O)O")["unspecified_stereocenters"] == 1
    assert inputs.check_smiles("CC(N)C(O)C(=O)O")["unspecified_stereocenters"] == 2
    assert inputs.resolve_ligand("CC(N)C(=O)O").unspecified_stereocenters == 1


# ------------------------------------------------------------------------------------------ complex names vs pandas
@pytest.mark.parametrize("name", ["1e20", "123", "0", "1.5", "inf", "nan", "NaN", "na", "NA", "null", "None", "n/a", "-nan"])
def test_names_pandas_would_misread_are_flagged(name):
    assert inputs.is_pandas_unsafe_name(name)


@pytest.mark.parametrize("name", ["1a0q", "6agt", "c_1e20", "complex_0", "x1", "1abc", "e20"])
def test_ordinary_names_are_fine(name):
    assert not inputs.is_pandas_unsafe_name(name)


def test_safe_names_survive_a_real_pandas_round_trip(tmp_path):
    pd = pytest.importorskip("pandas")
    for raw in ["1e20", "123", "na", "null", "inf", "1a0q", "weird name!"]:
        safe = inputs.safe_complex_name(raw)
        csv = tmp_path / "t.csv"
        csv.write_text("complex_name,protein_path,ligand_description,protein_sequence\n" + f"{safe},p.pdb,CCO,\n")
        got = pd.read_csv(csv)["complex_name"].tolist()
        assert got == [safe] and isinstance(got[0], str), (raw, safe, got)


def test_unsafe_names_are_rejected_with_a_suggestion(pdb_file, tmp_path):
    with pytest.raises(InputError, match="c_1e20"):
        inputs.write_input_csv(tmp_path / "in.csv", [InputRow("1e20", str(pdb_file), "CCO")])
    with pytest.raises(InputError, match="pandas"):
        run.prepare_inputs(run.new_run_dir(tmp_path, "x"), complex_name="123", receptor=str(pdb_file), ligand="CCO")


def test_default_complex_name_is_pandas_safe():
    assert run.default_complex_name("1E20", "CCO") == "c_1e20"
    assert run.default_complex_name("6AGT", "CCO") == "6agt"
    assert run.default_complex_name("/data/my protein.pdb", "CCO") == "my-protein"


# ------------------------------------------------------------------------------------------ ligand files / odd strings
def test_only_formats_upstream_can_read_are_accepted(tmp_path):
    mol = tmp_path / "lig.mol"
    mol.write_text("\n  RDKit          3D\n\n  0  0  0  0  0  0  0  0  0  0999 V2000\nM  END\n")
    assert ".mol" not in inputs.LIGAND_FILE_SUFFIXES
    with pytest.raises(InputError, match="convert a .mol file to .sdf"):
        inputs.resolve_ligand(str(mol))


def test_a_very_long_smiles_does_not_crash_the_path_check():
    smiles = "C" * 300  # > 255 bytes with no slash: Path(...).exists() raises ENAMETOOLONG on Linux
    assert inputs.is_existing_file(smiles) is False
    assert inputs.resolve_ligand("C" * 300).kind == "smiles"
    assert inputs.is_existing_file("x" * 10000) is False
    assert inputs.is_existing_file("a\nb") is False


def test_rcsb_download_scans_the_whole_file(monkeypatch, tmp_path):
    header = "\n".join(f"REMARK {i:>3} filler" for i in range(6000)) + "\n"
    atom = "ATOM      1  N   ALA A   1      11.104  13.207   2.100  1.00 20.00           N\n"
    monkeypatch.setattr(inputs, "_fetch", lambda url, timeout=60: (header + atom).encode())
    assert inputs.download_pdb("1abc", tmp_path).exists()  # the first ATOM record is on line 6001


# ------------------------------------------------------------------------------------------ portable input CSV
def test_portable_csv_survives_a_move_and_can_be_rebased(fake_workdir, pdb_file, ligand_sdf, monkeypatch, tmp_path):
    rd = run.new_run_dir(fake_workdir / "runs", "toy")
    prepared = run.prepare_inputs(rd, complex_name="toy", receptor=str(pdb_file), ligand=str(ligand_sdf))
    rows = inputs.read_input_csv(prepared.portable_csv_path)
    assert not Path(rows[0].protein_path).is_absolute() and not Path(rows[0].ligand_description).is_absolute()

    moved = tmp_path / "elsewhere" / "inputs"  # the VM is recycled; only the Drive copy of inputs/ is left
    import shutil

    shutil.copytree(rd / "inputs", moved)
    shutil.rmtree(rd)
    rebased = inputs.rebase_csv(moved / "input_protein_ligand.portable.csv", moved, tmp_path / "rebased.csv")
    (row,) = inputs.validate_input_csv(rebased)  # absolute paths that exist at the new location
    assert Path(row.protein_path).is_absolute() and str(moved.resolve()) in row.protein_path


def test_manifest_records_and_verifies_the_portable_csv(fake_workdir, pdb_file, monkeypatch):
    monkeypatch.setenv("FAKE_DIFFDOCK_MODE", "ok")
    rd = run.new_run_dir(fake_workdir / "runs", "toy")
    prepared = run.prepare_inputs(rd, complex_name="toy", receptor=str(pdb_file), ligand="CCO")
    m = run.run_diffdock(fake_workdir, prepared, python_exe=sys.executable, echo=False)
    assert m["input_csv_portable"]["path"] == "inputs/input_protein_ligand.portable.csv"
    from docking_tools.manifest import ManifestError, verify_manifest

    (rd / m["input_csv_portable"]["path"]).write_text("tampered\n")
    with pytest.raises(ManifestError, match="hash mismatch"):
        verify_manifest(rd, m)


# ------------------------------------------------------------------------------------------ stale poses
def test_stale_poses_from_an_earlier_call_never_count_as_success(fake_workdir, pdb_file, monkeypatch):
    monkeypatch.setenv("FAKE_DIFFDOCK_MODE", "ok")
    rd = run.new_run_dir(fake_workdir / "runs", "toy")
    prepared = run.prepare_inputs(rd, complex_name="toy", receptor=str(pdb_file), ligand="CCO")
    run.run_diffdock(fake_workdir, prepared, python_exe=sys.executable, echo=False)
    assert (rd / "predictions" / "toy" / "rank1.sdf").exists()
    monkeypatch.setenv("FAKE_DIFFDOCK_MODE", "silent")  # same run directory, upstream now silently produces nothing
    with pytest.raises(run.DockingRunError, match="no rank"):
        run.run_diffdock(fake_workdir, prepared, python_exe=sys.executable, echo=False)


# ------------------------------------------------------------------------------------------ child processes
def test_interrupting_the_stream_terminates_the_child(monkeypatch):
    class Boom(KeyboardInterrupt):
        pass

    pids = []
    real_popen = subprocess.Popen

    def spy(*a, **k):
        p = real_popen(*a, **k)
        pids.append(p)
        return p

    monkeypatch.setattr(subprocess, "Popen", spy)

    def explode(_):
        raise Boom()

    monkeypatch.setattr(sys.stdout, "write", explode)
    cmd = [sys.executable, "-c", "import time; print('go', flush=True); time.sleep(60)"]
    t0 = time.time()
    with pytest.raises(Boom):
        _util.run_streaming(cmd)
    assert time.time() - t0 < 30
    assert pids[0].poll() is not None  # the sleeping child was terminated, not left running


def test_write_json_is_atomic(tmp_path):
    target = tmp_path / "a" / "x.json"
    _util.write_json(target, {"k": 1})
    assert json.loads(target.read_text()) == {"k": 1} and not list(tmp_path.rglob("*.tmp"))
    with pytest.raises(TypeError):
        _util.write_json(target, {"bad": object()})  # serialisation fails halfway
    assert json.loads(target.read_text()) == {"k": 1}  # the old content is untouched


# ------------------------------------------------------------------------------------------ setup robustness
def make_models_zip(path):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("score_model/model_parameters.yml", "a: 1")
        z.writestr("score_model/best_ema_inference_epoch_model.pt", b"x" * 10)
        z.writestr("confidence_model/model_parameters.yml", "a: 1")
        z.writestr("confidence_model/best_model_epoch75.pt", b"y" * 10)


def test_model_extraction_is_all_or_nothing(tmp_path, monkeypatch):
    lay = Layout(tmp_path / "wd")
    lay.workdir.mkdir()
    archive = lay.workdir / "diffdock_models_v1.1.zip"
    make_models_zip(archive)
    monkeypatch.setattr(setup_env, "MODELS_ZIP_SHA256", setup_env.sha256_file(archive))

    real_extract = zipfile.ZipFile.extractall

    def interrupted(self, path=None, members=None, pwd=None):
        real_extract(self, path, members, pwd)
        raise KeyboardInterrupt  # killed after the files were written but before the extraction was declared complete

    monkeypatch.setattr(zipfile.ZipFile, "extractall", interrupted)
    with pytest.raises(KeyboardInterrupt):
        setup_env.download_models(lay)
    assert not setup_env.model_files_present(lay) and not lay.models_dir.exists()  # nothing half-done is visible

    monkeypatch.setattr(zipfile.ZipFile, "extractall", real_extract)
    hashes = setup_env.download_models(lay)
    assert setup_env.model_files_present(lay) and set(hashes) == set(setup_env.MODEL_FILES)
    assert not lay.models_dir.with_name("v1.1.partial").exists()


def test_a_truncated_checkpoint_is_not_accepted(tmp_path, monkeypatch):
    lay = Layout(tmp_path / "wd")
    lay.workdir.mkdir()
    archive = lay.workdir / "diffdock_models_v1.1.zip"
    make_models_zip(archive)
    monkeypatch.setattr(setup_env, "MODELS_ZIP_SHA256", setup_env.sha256_file(archive))
    setup_env.download_models(lay)
    assert setup_env.model_files_present(lay)
    (lay.models_dir / "score_model" / "best_ema_inference_epoch_model.pt").write_bytes(b"")
    assert not setup_env.model_files_present(lay)  # an empty/short file no longer passes
    (lay.models_dir / ".extracted_from").write_text("0" * 64)
    assert not setup_env.model_files_present(lay)  # nor does a marker from a different archive


def test_model_archive_with_path_traversal_is_refused(tmp_path, monkeypatch):
    lay = Layout(tmp_path / "wd")
    lay.workdir.mkdir()
    archive = lay.workdir / "diffdock_models_v1.1.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../evil.txt", "x")
    monkeypatch.setattr(setup_env, "MODELS_ZIP_SHA256", setup_env.sha256_file(archive))
    with pytest.raises(RuntimeError, match="unsafe path"):
        setup_env.download_models(lay)
    assert not (tmp_path / "evil.txt").exists()


def test_all_six_lookup_tables_are_required_and_partial_sets_rebuilt(tmp_path, monkeypatch):
    lay = Layout(tmp_path / "wd")
    lay.repo.mkdir(parents=True)
    for f in setup_env.TABLE_FILES[:2]:  # the two files the old check looked at, plus nothing else
        (lay.repo / f).write_bytes(b"partial")
    (lay.repo / ".p.npy").write_bytes(b"x")
    assert not setup_env.tables_present(lay)  # four of six missing: an interrupted run
    calls = []

    def fake_run(cmd, cwd=None, env=None):
        calls.append(cmd)
        assert not any((lay.repo / f).exists() for f in setup_env.TABLE_FILES)  # stale partial files were removed first
        for f in setup_env.TABLE_FILES:
            (lay.repo / f).write_bytes(b"table")

    monkeypatch.setattr(setup_env, "_run", fake_run)
    setup_env.precompute_tables(lay)
    assert len(calls) == 1 and setup_env.tables_present(lay)
    setup_env.precompute_tables(lay)  # complete now: no second build
    assert len(calls) == 1


def test_missing_table_after_the_build_is_an_error(tmp_path, monkeypatch):
    lay = Layout(tmp_path / "wd")
    lay.repo.mkdir(parents=True)
    monkeypatch.setattr(setup_env, "_run", lambda *a, **k: (lay.repo / ".p.npy").write_bytes(b"x"))
    with pytest.raises(RuntimeError, match="did not write"):
        setup_env.precompute_tables(lay)


def test_tables_are_restored_from_the_drive_cache_instead_of_rebuilt(tmp_path, monkeypatch):
    lay = Layout(tmp_path / "wd")
    lay.repo.mkdir(parents=True)
    cache = tmp_path / "cache"
    for f in setup_env.TABLE_FILES:
        (cache / "lookup_tables").mkdir(parents=True, exist_ok=True)
        (cache / "lookup_tables" / f).write_bytes(b"cached " + f.encode())
    monkeypatch.setattr(setup_env, "_run", lambda *a, **k: pytest.fail("rebuilt despite a warm cache"))
    setup_env._cached_tables(lay, cache)
    assert (lay.repo / ".p.npy").read_bytes() == b"cached .p.npy" and setup_env.tables_present(lay)


def test_esm_weights_are_hash_checked_and_a_bad_file_is_replaced(tmp_path, monkeypatch):
    lay = Layout(tmp_path / "wd")
    good = {"esm2_t33_650M_UR50D.pt": b"weights", "esm2_t33_650M_UR50D-contact-regression.pt": b"regr"}
    import hashlib

    monkeypatch.setattr(setup_env, "ESM_WEIGHT_SHA256", {n: hashlib.sha256(b).hexdigest() for n, b in good.items()})
    ckpt = lay.torch_home / "hub" / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "esm2_t33_650M_UR50D.pt").write_bytes(b"CORRUPT")  # right name, wrong content
    downloads = []

    def fake_download(url, dest, **k):
        downloads.append(dest.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(good[dest.name])

    monkeypatch.setattr(setup_env, "_download", fake_download)
    assert not setup_env.esm_weights_ok(lay)
    hashes = setup_env.download_esm_weights(lay)
    assert set(downloads) == set(good) and setup_env.esm_weights_ok(lay) and set(hashes) == set(good)
    monkeypatch.setattr(setup_env, "_download", lambda url, dest, **k: dest.write_bytes(b"still wrong"))
    (ckpt / "esm2_t33_650M_UR50D.pt").unlink()
    with pytest.raises(RuntimeError, match="sha256"):
        setup_env.download_esm_weights(lay)


def test_the_pinned_esm_hashes_have_the_right_shape():
    assert set(setup_env.ESM_WEIGHT_SHA256) == {u.rsplit("/", 1)[-1] for u in setup_env.ESM_WEIGHT_URLS}
    assert all(len(h) == 64 and set(h) <= set("0123456789abcdef") for h in setup_env.ESM_WEIGHT_SHA256.values())


def test_rerunning_setup_keeps_the_time_of_the_run_that_did_the_work(tmp_path, monkeypatch):
    lay = Layout(tmp_path / "wd")
    lay.workdir.mkdir()
    lay.setup_info.write_text(json.dumps({"step_seconds": {"lookup_tables": 777.0, "environment": 198.6}}))
    for name in ["ensure_micromamba", "create_env"]:
        monkeypatch.setattr(setup_env, name, lambda *a, **k: None)
    monkeypatch.setattr(setup_env, "clone_upstream", lambda l: setup_env.UPSTREAM_COMMIT)
    monkeypatch.setattr(setup_env, "apply_patch", lambda l: "p" * 64)
    monkeypatch.setattr(setup_env, "_cached_models", lambda l, c: {"score_model/x": "1"})
    monkeypatch.setattr(setup_env, "_cached_esm_weights", lambda l, c: dict(setup_env.ESM_WEIGHT_SHA256))
    monkeypatch.setattr(setup_env, "_cached_tables", lambda l, c: None)
    monkeypatch.setattr(setup_env, "gpu_smoke_test", lambda l: {"cuda_available": False})
    monkeypatch.setattr(setup_env, "env_versions", lambda l: {"python": "3.9.18"})
    monkeypatch.setattr(setup_env, "gpu_info", lambda: None)
    # everything reports itself as already done
    monkeypatch.setattr(setup_env.Layout, "mamba", property(lambda self: tmp_path))  # exists
    monkeypatch.setattr(setup_env, "env_is_ready", lambda l: True)
    monkeypatch.setattr(setup_env, "model_files_present", lambda l: True)
    monkeypatch.setattr(setup_env, "esm_weights_ok", lambda l: True)
    monkeypatch.setattr(setup_env, "tables_present", lambda l: True)
    monkeypatch.setattr(setup_env, "_git_ok", lambda *a, **k: True)
    (lay.repo / ".git").mkdir(parents=True)
    monkeypatch.setattr(Layout, "python", property(lambda self: Path(sys.executable)))
    info = setup_env.setup(lay.workdir)
    assert info["step_seconds"]["lookup_tables"] == 777.0 and info["step_seconds"]["environment"] == 198.6
    assert info["setup_seconds_total"] >= 975.6  # not reset to ~0 by a no-op re-run
    assert "lookup_tables" in info["last_run"]["steps_already_done"]
    assert info["esm_weights"]["files_sha256"] and info["schema_version"] == 2


def test_gpu_smoke_test_failure_gives_actionable_advice(tmp_path, monkeypatch):
    lay = Layout(tmp_path)
    monkeypatch.setattr(Layout, "python", property(lambda self: Path(sys.executable)))  # no torch there: the probe fails
    with pytest.raises(RuntimeError, match="sm_86"):
        setup_env.gpu_smoke_test(lay)


def test_gpu_smoke_test_reports_a_cpu_runtime(tmp_path, monkeypatch):
    lay = Layout(tmp_path)
    monkeypatch.setattr(setup_env, "GPU_SMOKE_CODE", "import json; print(json.dumps({'cuda_available': False}))")
    monkeypatch.setattr(Layout, "python", property(lambda self: Path(sys.executable)))
    assert setup_env.gpu_smoke_test(lay) == {"cuda_available": False}


# ------------------------------------------------------------------------------------------ PDBBind subsets
def test_two_different_subsets_of_the_same_size_get_different_folders():
    a = benchmark_prep.subset_name(["1a0q", "2abc"])
    b = benchmark_prep.subset_name(["1a0q", "3xyz"])
    assert a != b and a.startswith("pdbbind_subset_2_") and b.startswith("pdbbind_subset_2_")
    assert a == benchmark_prep.subset_name(["2ABC", "1a0q"])  # order/case do not matter


def test_prepare_subset_refuses_a_folder_that_holds_other_complexes(tmp_path):
    root = tmp_path / "out"
    (root / "PDBBind_processed" / "9zzz").mkdir(parents=True)
    zpath = tmp_path / "p.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("PDBBind_processed/1a0q/1a0q_protein_processed.pdb", "ATOM")
        z.writestr("PDBBind_processed/1a0q/1a0q_ligand.sdf", "x")
    with pytest.raises(benchmark_prep.BenchmarkPrepError, match="other complexes"):
        benchmark_prep.prepare_subset(tmp_path / "wd", ["1a0q"], pdbbind_zip=zpath, out_dir=root)


# ------------------------------------------------------------------------------------------ Drive
def test_index_json_write_is_atomic_and_idempotent(tmp_path):
    root = tmp_path / "root"
    run_dir = tmp_path / "r1"
    (run_dir / "predictions").mkdir(parents=True)
    (run_dir / "a.txt").write_text("x")
    drive.persist_run(run_dir, root)
    drive.persist_run(run_dir, root)
    assert len(json.loads((root / "index.json").read_text())) == 1 and not list(root.rglob("*.tmp"))
