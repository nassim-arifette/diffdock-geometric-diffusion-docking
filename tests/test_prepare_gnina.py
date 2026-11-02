"""The downloaded executable must never replace a verified binary on failure."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("prepare_gnina", Path(__file__).parents[1] / "tools" / "prepare_gnina.py")
gnina = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gnina)


def release(data=b"a verified binary"):
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "url": "https://example.invalid/gnina"}


def test_truncated_download_is_rejected_and_never_promoted(tmp_path, monkeypatch):
    monkeypatch.setattr(gnina, "urlopen", lambda *a, **kw: io.BytesIO(b"short"))
    target = tmp_path / "gnina"
    with pytest.raises(ValueError, match="size mismatch"):
        gnina.download_binary(target, release())
    assert not target.exists()
    assert not list(tmp_path.glob("*.part"))


def test_equal_length_tampering_is_rejected(tmp_path, monkeypatch):
    expected = b"approved"
    monkeypatch.setattr(gnina, "urlopen", lambda *a, **kw: io.BytesIO(b"tampered"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        gnina.download_binary(tmp_path / "gnina", release(expected))
    assert not (tmp_path / "gnina").exists()


def test_valid_download_is_reused_without_network(tmp_path, monkeypatch):
    data = b"a verified binary"
    target = tmp_path / "gnina"
    monkeypatch.setattr(gnina, "urlopen", lambda *a, **kw: io.BytesIO(data))
    gnina.download_binary(target, release(data))
    monkeypatch.setattr(gnina, "urlopen", lambda *a, **kw: pytest.fail("Verified binary should be reused"))
    gnina.download_binary(target, release(data))
    assert target.read_bytes() == data


def test_existing_wrong_binary_is_preserved_and_refused(tmp_path, monkeypatch):
    target = tmp_path / "gnina"
    target.write_bytes(b"user file")
    monkeypatch.setattr(gnina, "urlopen", lambda *a, **kw: pytest.fail("Must not overwrite an existing binary"))
    with pytest.raises(ValueError):
        gnina.download_binary(target, release())
    assert target.read_bytes() == b"user file"


def test_missing_libraries_are_reported_without_confusing_present_paths():
    output = """linux-vdso.so.1 (0x123)
    libcudnn.so.9 => not found
    libcublas.so.12 => not found
    libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x456)
    libcudnn.so.9 => not found
"""
    assert gnina.missing_libraries(output) == ["libcublas.so.12", "libcudnn.so.9"]


def test_one_molecule_argument_fails_before_platform_or_download(tmp_path):
    with pytest.raises(SystemExit) as exc:
        gnina.main(["--out", str(tmp_path / "install"), "--ligand", "pose.sdf"])
    assert exc.value.code == 2
    assert not (tmp_path / "install").exists()


def test_bundle_refuses_a_changed_library(tmp_path, monkeypatch):
    data = b"release"
    digest = hashlib.sha256(data).hexdigest()
    monkeypatch.setitem(gnina.RELEASE, "sha256", digest)
    (tmp_path / gnina.RELEASE["asset"]).write_bytes(data)
    (tmp_path / "gnina").write_bytes(b"launcher")
    (tmp_path / "library.so").write_bytes(b"changed")
    manifest = {"binary": {"file": gnina.RELEASE["asset"], "sha256": digest},
                "launcher": {"file": "gnina", "sha256": hashlib.sha256(b"launcher").hexdigest()},
                "libraries": [{"file": "library.so", "sha256": hashlib.sha256(b"original").hexdigest()}]}
    (tmp_path / "runtime_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="path/hash mismatch"):
        gnina.verify_runtime_bundle(tmp_path)


def test_bundle_refuses_paths_outside_its_directory(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    root.mkdir()
    data = b"release"
    digest = hashlib.sha256(data).hexdigest()
    monkeypatch.setitem(gnina.RELEASE, "sha256", digest)
    (root / gnina.RELEASE["asset"]).write_bytes(data)
    (tmp_path / "outside").write_bytes(b"launcher")
    manifest = {"binary": {"file": gnina.RELEASE["asset"], "sha256": digest},
                "launcher": {"file": "../outside", "sha256": hashlib.sha256(b"launcher").hexdigest()},
                "libraries": []}
    (root / "runtime_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="path/hash mismatch"):
        gnina.verify_runtime_bundle(root)
