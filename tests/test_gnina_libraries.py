import hashlib
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("bundle_gnina_libraries", Path(__file__).parents[1] / "tools" / "bundle_gnina_libraries.py")
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


def test_different_installed_bytes_never_enter_the_bundle(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"changed")
    target = tmp_path / "libenv" / "library.so"
    with pytest.raises(ValueError, match="wheel RECORD"):
        bundle.checked_copy(source, target, hashlib.sha256(b"original").hexdigest())
    assert not target.exists()


def test_existing_library_is_checked_and_never_replaced(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"approved")
    target.write_bytes(b"another run")
    with pytest.raises(ValueError, match="overwrite"):
        bundle.checked_copy(source, target, hashlib.sha256(b"approved").hexdigest())
    assert target.read_bytes() == b"another run"


def test_verified_copy_can_be_resumed(tmp_path):
    source, target = tmp_path / "source", tmp_path / "libs" / "library.so"
    source.write_bytes(b"approved")
    expected = hashlib.sha256(b"approved").hexdigest()
    bundle.checked_copy(source, target, expected)
    bundle.checked_copy(source, target, expected)
    assert bundle.sha256_file(target) == expected
