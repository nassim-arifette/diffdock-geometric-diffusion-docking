import json
import hashlib
import sys
import zipfile
from pathlib import Path

import pytest

from docking_tools import drive, export, run, setup_env
from docking_tools.manifest import ManifestError
from docking_tools.setup_env import Layout


def finished(fake_workdir, pdb_file, monkeypatch, mode="ok"):
    monkeypatch.setenv("FAKE_DIFFDOCK_MODE", mode)
    rd = run.new_run_dir(fake_workdir / "runs", "toy")
    prepared = run.prepare_inputs(rd, complex_name="toy", receptor=str(pdb_file), ligand="CCO")
    try:
        run.run_diffdock(fake_workdir, prepared, python_exe=sys.executable, echo=False)
    except run.DockingRunError:
        pass
    return rd


def test_persist_run_copies_verifies_and_indexes(fake_workdir, pdb_file, monkeypatch, tmp_path):
    rd = finished(fake_workdir, pdb_file, monkeypatch)
    archive = export.export_run(rd, tmp_path / "exp")
    root = tmp_path / "drive" / "DiffDock_colab"
    dest = drive.persist_run(rd, root, archive)
    assert (dest / "manifest.json").exists() and (dest / "predictions" / "toy" / "rank1.sdf").exists()
    assert (root / "exports" / archive.name).exists()
    (entry,) = json.loads((root / "index.json").read_text())
    assert entry["run_id"] == rd.name and entry["status"] == "ok" and entry["archive"].endswith(archive.name)
    drive.persist_run(rd, root)  # idempotent: re-saving replaces the entry, never duplicates it
    assert len(json.loads((root / "index.json").read_text())) == 1


def test_failed_runs_are_saved_too(fake_workdir, pdb_file, monkeypatch, tmp_path):
    rd = finished(fake_workdir, pdb_file, monkeypatch, mode="crash")
    dest = drive.persist_run(rd, tmp_path / "root")
    assert json.loads((dest / "manifest.json").read_text())["status"] == "failed"
    assert (dest / "logs" / "inference.log").exists()
    assert json.loads((tmp_path / "root" / "index.json").read_text())[0]["status"] == "failed"


def test_persist_run_detects_a_corrupted_copy(fake_workdir, pdb_file, monkeypatch, tmp_path):
    rd = finished(fake_workdir, pdb_file, monkeypatch)
    real = drive._copy_file_verified

    def corrupting(src, dst):
        real(src, dst)
        if src.name == "rank2_confidence0.63.sdf":
            dst.write_text("garbage")

    monkeypatch.setattr(drive, "_copy_file_verified", corrupting)
    with pytest.raises(ManifestError, match="hash mismatch"):  # the manifest hashes are re-checked on the Drive copy
        drive.persist_run(rd, tmp_path / "root")


def test_persist_dataset(tmp_path):
    src = tmp_path / "ds"
    (src / "a").mkdir(parents=True)
    (src / "a" / "x.txt").write_text("x")
    (src / "y.txt").write_text("y")
    dest = drive.persist_dataset(src, tmp_path / "root", "bundled")
    assert (dest / "a" / "x.txt").read_text() == "x" and (dest / "y.txt").read_text() == "y"
    single = drive.persist_dataset(src / "y.txt", tmp_path / "root", "single")
    assert (single / "y.txt").read_text() == "y"


def test_ensure_drive_without_colab_returns_none(tmp_path, capsys):
    assert drive.ensure_drive(str(tmp_path / "root"), mount=str(tmp_path / "nomount")) is None
    assert "not available" in capsys.readouterr().out


def test_ensure_drive_with_existing_mount(tmp_path):
    (tmp_path / "mnt" / "MyDrive").mkdir(parents=True)
    root = drive.ensure_drive(str(tmp_path / "mnt" / "MyDrive" / "DiffDock_colab"), mount=str(tmp_path / "mnt"))
    assert root is not None and root.is_dir()


def test_cache_restore_and_store_roundtrip(tmp_path):
    f = tmp_path / "w" / "weights.pt"
    f.parent.mkdir()
    f.write_bytes(b"1234")
    assert drive.cache_store(tmp_path / "cache", "weights.pt", f) is True
    assert drive.cache_store(tmp_path / "cache", "weights.pt", f) is False  # already cached
    f.unlink()
    assert drive.cache_restore(tmp_path / "cache", "weights.pt", f) is True and f.read_bytes() == b"1234"
    f.unlink()
    assert drive.cache_restore(tmp_path / "cache", "weights.pt", f, sha256="0" * 64) is False  # wrong hash ignored
    assert drive.cache_restore(None, "weights.pt", f) is False


def test_setup_uses_the_drive_cache_instead_of_downloading(tmp_path, monkeypatch):
    """With the model archive and ESM weights in the cache, setup's models/esm steps never touch the network."""
    archive = tmp_path / "cache" / "diffdock_models_v1.1.zip"
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("score_model/model_parameters.yml", "a: 1")
        z.writestr("score_model/best_ema_inference_epoch_model.pt", b"x")
        z.writestr("confidence_model/model_parameters.yml", "a: 1")
        z.writestr("confidence_model/best_model_epoch75.pt", b"y")
    for url in setup_env.ESM_WEIGHT_URLS:
        (tmp_path / "cache" / url.rsplit("/", 1)[-1]).write_bytes(b"w")
    monkeypatch.setattr(setup_env, "ESM_WEIGHT_SHA256", {
        u.rsplit("/", 1)[-1]: hashlib.sha256(b"w").hexdigest() for u in setup_env.ESM_WEIGHT_URLS
    })
    monkeypatch.setattr(setup_env, "MODELS_ZIP_SHA256", setup_env.sha256_file(archive))
    monkeypatch.setattr(setup_env, "_download", lambda *a, **k: pytest.fail("network used despite a warm cache"))
    lay = Layout(tmp_path / "wd")
    lay.workdir.mkdir()
    hashes = setup_env._cached_models(lay, tmp_path / "cache")
    setup_env._cached_esm_weights(lay, tmp_path / "cache")
    assert "score_model/best_ema_inference_epoch_model.pt" in hashes
    for url in setup_env.ESM_WEIGHT_URLS:
        assert (lay.torch_home / "hub" / "checkpoints" / url.rsplit("/", 1)[-1]).read_bytes() == b"w"


def test_setup_stores_fresh_downloads_in_the_cache(tmp_path, monkeypatch):
    src_zip = tmp_path / "src.zip"
    with zipfile.ZipFile(src_zip, "w") as z:
        z.writestr("score_model/model_parameters.yml", "a")
        z.writestr("score_model/best_ema_inference_epoch_model.pt", b"x")
        z.writestr("confidence_model/model_parameters.yml", "a")
        z.writestr("confidence_model/best_model_epoch75.pt", b"y")
    monkeypatch.setattr(setup_env, "MODELS_ZIP_SHA256", setup_env.sha256_file(src_zip))
    monkeypatch.setattr(setup_env, "ESM_WEIGHT_SHA256", {
        u.rsplit("/", 1)[-1]: setup_env.sha256_file(src_zip) for u in setup_env.ESM_WEIGHT_URLS
    })
    monkeypatch.setattr(setup_env, "_download", lambda url, dest, **k: (dest.parent.mkdir(parents=True, exist_ok=True), dest.write_bytes(src_zip.read_bytes())))
    lay = Layout(tmp_path / "wd")
    lay.workdir.mkdir()
    setup_env._cached_models(lay, tmp_path / "cache")
    setup_env._cached_esm_weights(lay, tmp_path / "cache")
    assert (tmp_path / "cache" / "diffdock_models_v1.1.zip").exists()
    assert all((tmp_path / "cache" / u.rsplit("/", 1)[-1]).exists() for u in setup_env.ESM_WEIGHT_URLS)


def test_cache_store_repairs_same_size_corruption(tmp_path):
    source = tmp_path / "weights.pt"
    source.write_bytes(b"good")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / source.name).write_bytes(b"bad!")
    assert drive.cache_store(cache, source.name, source)
    source.unlink()
    assert drive.cache_restore(cache, source.name, source, hashlib.sha256(b"good").hexdigest())
    assert source.read_bytes() == b"good"
