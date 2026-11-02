"""Verify an uploaded bundle and start the acceptance run without blocking Colab's kernel.

Execute with ``colab exec --file .../tools/colab_launch.py -s cv-diffdock``.
The helper and upstream environments are isolated from Colab's notebook kernel.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

CONTENT = Path("/content")
ARCHIVE = CONTENT / "diffdock_colab.tar.gz"
MANIFEST = CONTENT / "diffdock_package_manifest.json"
ROOT = Path(os.environ.get("DIFFDOCK_PROJECT_ROOT", str(CONTENT / "diffdock_project"))).resolve()
STATE = Path(os.environ.get("DIFFDOCK_JOB_STATE", str(CONTENT / "diffdock_job"))).resolve()


def launch():
    if not ROOT.is_relative_to(CONTENT) or not STATE.is_relative_to(CONTENT):
        raise ValueError("project and job directories must be within /content")
    manifest = json.loads(MANIFEST.read_text())
    actual = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    if actual != manifest["sha256"]:
        raise RuntimeError("uploaded archive hash mismatch")
    pid_file = STATE / "pid"
    if pid_file.exists():
        pid = int(pid_file.read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError(f"job {pid} still exists; inspect it before launching again")
    if ROOT.exists():
        raise RuntimeError("project directory already exists; inspect the prior attempt first")
    ROOT.mkdir()
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        archive.extractall(ROOT, filter="data")
    for name, digest in manifest["files"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"extracted source hash mismatch: {name}")
    STATE.mkdir(exist_ok=True)
    (STATE / "package_manifest.json").write_text(json.dumps(manifest, indent=2))
    runner = ROOT / "tools" / "colab_acceptance.py"
    with (STATE / "launcher.log").open("ab", buffering=0) as log:
        child = subprocess.Popen(
            [sys.executable, "-u", str(runner), "--project", str(ROOT), "--state", str(STATE)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    pid_file.write_text(str(child.pid))
    print(json.dumps({"pid": child.pid, "bundle_sha256": actual, "log": str(STATE / "launcher.log")}))


if __name__ == "__main__":
    launch()
