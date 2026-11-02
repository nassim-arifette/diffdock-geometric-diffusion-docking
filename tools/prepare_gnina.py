"""Fetch the pinned official GNINA Linux binary and check its runtime libraries.

No pip, apt, conda, CUDA or Torch installation is changed. An optional real
score-only/minimisation check uses this repository's existing GNINA adapter.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from urllib.request import Request, urlopen


RELEASE = {
    "tag": "v1.3.3",
    "commit": "6fe1ce2bb9c35c8067de9f49bb9169857dfbad70",
    "asset_id": 462167293,
    "asset": "gnina.cuda12.8.static",
    "size": 2056131000,
    "sha256": "3340c1f49cd3c7c84d8699182a1c6af13c7fa2a22448d1204640446106f72172",
    "url": "https://github.com/gnina/gnina/releases/download/v1.3.3/gnina.cuda12.8.static",
    "metadata_url": "https://api.github.com/repos/gnina/gnina/releases/tags/v1.3.3",
    "published_utc": "2026-06-29T14:47:27Z",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_binary(path, release=RELEASE):
    path = Path(path)
    if path.stat().st_size != release["size"]:
        raise ValueError(f"GNINA size mismatch: {path}; expected {release['size']} bytes")
    actual = sha256_file(path)
    if actual != release["sha256"]:
        raise ValueError(f"GNINA SHA256 mismatch: {path}; got {actual}")
    return actual


def download_binary(destination, release=RELEASE):
    destination = Path(destination)
    if destination.exists():
        verify_binary(destination, release)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix="gnina-", suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
            request = Request(release["url"], headers={"User-Agent": "diffdock-reproducible-pilot/1"})
            with urlopen(request, timeout=120) as response:
                downloaded = 0
                for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                    stream.write(chunk)
                    downloaded += len(chunk)
                    if downloaded > release["size"]:
                        raise ValueError("GNINA download exceeded the pinned byte size")
                    if downloaded // (128 * 1024 * 1024) != (downloaded - len(chunk)) // (128 * 1024 * 1024):
                        print(f"GNINA downloaded {downloaded}/{release['size']} bytes", flush=True)
        verify_binary(temporary, release)
        if destination.exists():
            raise FileExistsError(f"GNINA destination appeared during download: {destination}")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def missing_libraries(ldd_output):
    return sorted(set(re.findall(r"^\s*(\S+)\s+=>\s+not found\s*$", ldd_output, re.MULTILINE)))


def command_record(command, timeout=120):
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    return {"command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def verify_runtime_bundle(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "runtime_manifest.json").read_text())
    if manifest["binary"] != {"file": RELEASE["asset"], "sha256": RELEASE["sha256"]}:
        raise ValueError("Runtime bundle references a different GNINA binary")
    for entry in [manifest["binary"], manifest["launcher"]] + manifest["libraries"]:
        path = (root / entry["file"]).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Runtime bundle path/hash mismatch: {entry['file']}")
    return root / manifest["launcher"]["file"], manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/content/gnina"))
    parser.add_argument("--bundled-libraries", action="store_true", help="Use the verified launcher created by bundle_gnina_libraries.py")
    parser.add_argument("--receptor", type=Path, help="Optional real test receptor, in the ligand coordinate frame")
    parser.add_argument("--ligand", type=Path, help="Optional real SDF pose; requires both molecule arguments")
    args = parser.parse_args(argv)
    if bool(args.receptor) != bool(args.ligand):
        parser.error("--receptor and --ligand must be supplied together")
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        parser.error("This pinned executable requires Linux x86_64")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    binary = args.out / RELEASE["asset"]
    started = dt.datetime.now(dt.timezone.utc)
    stamp = started.strftime("%Y%m%dT%H%M%S%fZ")
    report_path = args.out / f"installation-{stamp}.json"
    report = {"release": RELEASE, "started_utc": started.isoformat(), "status": "running",
              "binary": str(binary), "platform": platform.platform(),
              "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
              "real_scoring_validated": False}

    def save():
        temp = report_path.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temp.replace(report_path)

    save()
    try:
        download_binary(binary)
        # No downloaded code is run before both release checks succeed.
        report["binary_sha256"] = verify_binary(binary)
        binary.chmod(binary.stat().st_mode | 0o100)
        executable = binary
        if args.bundled_libraries:
            executable, report["runtime_manifest"] = verify_runtime_bundle(args.out)
            loader_command = ["/lib64/ld-linux-x86-64.so.2", "--inhibit-rpath", "", "--library-path",
                              str(args.out / "libenv" / "lib") + ":/usr/lib64-nvidia", "--list", str(binary)]
        else:
            loader_command = ["ldd", str(binary)]
        report["ldd"] = command_record(loader_command)
        report["missing_libraries"] = missing_libraries(report["ldd"]["stdout"] + "\n" + report["ldd"]["stderr"])
        if report["ldd"]["returncode"] or report["missing_libraries"]:
            raise RuntimeError("GNINA runtime library check failed; see the installation report. No packages were changed.")
        report["version"] = command_record([str(executable), "--version"])
        if report["version"]["returncode"]:
            raise RuntimeError("GNINA --version failed; see the installation report")
        report["help"] = command_record([str(executable), "--help"])
        if report["help"]["returncode"]:
            raise RuntimeError("GNINA --help failed; see the installation report")
        if args.receptor:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from docking_tools.evaluation import Gnina
            receptor, ligand = args.receptor.resolve(), args.ligand.resolve()
            report["inputs"] = {"receptor": {"path": str(receptor), "sha256": sha256_file(receptor)},
                                "ligand": {"path": str(ligand), "sha256": sha256_file(ligand)}}
            output = args.out / f"smoke-{stamp}"
            scorer = Gnina(executable)
            report["score_only"] = scorer.run(receptor, ligand, output / "score_only.sdf")
            report["minimize"] = scorer.run(receptor, ligand, output / "minimized.sdf", minimize=True)
            report["real_scoring_validated"] = True
        report["status"] = "ok"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr)
    finally:
        report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        save()
        print(f"GNINA installation report: {report_path}", flush=True)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
