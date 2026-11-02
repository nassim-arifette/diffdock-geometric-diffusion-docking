"""Package the local code for Colab without requiring a premature GitHub push."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ("docking_tools", "notebooks", "tools", "tests", "docs", "report", "site")


def build():
    destination = ROOT / "colab_artifacts"
    destination.mkdir(exist_ok=True)
    files = [p for name in INCLUDE for p in (ROOT / name).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".log")]
    files += [ROOT / name for name in ("README.md", "requirements-tools.txt", "requirements-dev.txt")]
    files = sorted(p for p in files if p.is_file())
    archive = destination / "diffdock_colab.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for path in files:
            tf.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
    manifest = {"archive": archive.name, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "files": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (destination / "package_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(archive)
    print("SHA256", manifest["sha256"])


if __name__ == "__main__":
    build()
