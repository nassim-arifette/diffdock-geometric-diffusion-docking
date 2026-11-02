"""Copy verified installed NVIDIA CUDA 12 libraries into an isolated GNINA bundle.

This Colab-image-specific recovery does not install or modify any distribution.
Every shared library is checked against its wheel RECORD before copying.
"""
import argparse
import base64
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import subprocess


PINS = {
    "nvidia-cublas-cu12": "12.9.2.10",
    "nvidia-cuda-runtime-cu12": "12.9.79",
    "nvidia-cufft-cu12": "11.4.1.4",
    "nvidia-cusolver-cu12": "11.7.5.82",
    "nvidia-cusparse-cu12": "12.5.10.65",
    "nvidia-nvjitlink-cu12": "12.9.86",
    "nvidia-cudnn-cu12": "9.27.0.42",
    "nvidia-curand-cu12": "10.3.10.19",
    "nvidia-cuda-nvrtc-cu12": "12.9.86",
}
BINARY_SHA256 = "3340c1f49cd3c7c84d8699182a1c6af13c7fa2a22448d1204640446106f72172"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_copy(source, destination, expected):
    actual = sha256_file(source)
    if actual != expected:
        raise ValueError(f"Source does not match its wheel RECORD: {source}")
    if destination.exists():
        if sha256_file(destination) != expected:
            raise ValueError(f"Refusing to overwrite a different library: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        shutil.copyfile(source, temporary)
        if sha256_file(temporary) != expected:
            raise ValueError(f"Copy integrity failed: {destination}")
        temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/content/gnina"))
    args = parser.parse_args()
    root = args.out.resolve()
    binary = root / "gnina.cuda12.8.static"
    if sha256_file(binary) != BINARY_SHA256:
        raise ValueError("Original GNINA release binary SHA256 mismatch")
    libs = root / "libenv" / "lib"
    manifest = {"source": "installed NVIDIA wheels, verified against wheel RECORD",
                "binary": {"file": binary.name, "sha256": BINARY_SHA256},
                "packages": [], "libraries": []}
    for name, version in PINS.items():
        dist = metadata.distribution(name)
        if dist.version != version:
            raise ValueError(f"Expected {name}=={version}, found {dist.version}; review pins for this runtime")
        manifest["packages"].append({"name": name, "version": version})
        matched = 0
        for file in dist.files or []:
            source = Path(dist.locate_file(file)).resolve()
            if file.name.startswith("lib") and ".so" in file.name and "/lib/" in str(file).replace("\\", "/"):
                if not file.hash or file.hash.mode != "sha256":
                    raise ValueError(f"Missing wheel RECORD SHA256: {file}")
                expected = base64.urlsafe_b64decode(file.hash.value + "==").hex()
                destination = libs / file.name
                checked_copy(source, destination, expected)
                manifest["libraries"].append({"package": name, "source": str(source),
                                               "file": str(destination.relative_to(root)),
                                               "size": destination.stat().st_size, "sha256": expected})
                matched += 1
            elif ".dist-info/" in str(file) and ("license" in file.name.lower() or file.name == "METADATA"):
                dest = root / "libenv" / "licenses" / name / file.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copyfile(source, dest)
        if not matched:
            raise ValueError(f"No shared libraries found for {name}")
        print(f"Bundled {name}=={version}: {matched} libraries", flush=True)
    loader = Path("/lib64/ld-linux-x86-64.so.2")
    if not loader.is_file():
        raise RuntimeError("Linux x86_64 ELF loader not found")
    library_path = str(libs) + ":/usr/lib64-nvidia"
    command = [str(loader), "--inhibit-rpath", "", "--library-path", library_path, "--list", str(binary)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    manifest["loader_check"] = {"command": command, "returncode": result.returncode,
                                "stdout": result.stdout, "stderr": result.stderr}
    if result.returncode:
        raise RuntimeError(f"Isolated ELF loader failed: {result.stderr}")
    for line in result.stdout.splitlines():
        if any(lib in line for lib in ("libcudnn", "libcudart", "libcusparse", "libcufft", "libcublas", "libcusolver", "libnvJitLink")):
            if str(libs) not in line:
                raise RuntimeError(f"GPU dependency escaped the isolated bundle: {line}")
    launcher = root / "gnina"
    body = '#!/bin/sh\nDIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\nexec /lib64/ld-linux-x86-64.so.2 --inhibit-rpath "" --library-path "$DIR/libenv/lib:/usr/lib64-nvidia" "$DIR/gnina.cuda12.8.static" "$@"\n'
    if launcher.exists() and launcher.read_text() != body:
        raise ValueError("Refusing to overwrite a different GNINA launcher")
    launcher.write_text(body)
    launcher.chmod(0o755)
    manifest["launcher"] = {"file": launcher.name, "sha256": sha256_file(launcher)}
    version_result = subprocess.run([str(launcher), "--version"], capture_output=True, text=True, timeout=60)
    manifest["version"] = {"returncode": version_result.returncode, "stdout": version_result.stdout, "stderr": version_result.stderr}
    manifest["status"] = "ok" if version_result.returncode == 0 else "failed"
    target = root / "runtime_manifest.json"
    target.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"status": manifest["status"], "version": manifest["version"], "manifest": str(target)}), flush=True)
    return version_result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
