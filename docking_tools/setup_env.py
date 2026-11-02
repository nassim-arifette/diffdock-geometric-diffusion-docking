"""Reproducible setup of the pinned upstream DiffDock + its Python environment.

Usage (Colab):  ``!python -m docking_tools.setup_env --workdir /content/diffdock_work``

What it does, in order (each step is idempotent, so re-running after a disconnect is cheap):

1. micromamba (pinned release binary)
2. clone gcorso/DiffDock at ``UPSTREAM_COMMIT`` and apply ``patches/diffdock_seed_and_scores.patch``
3. create the conda/pip environment from ``env/diffdock_colab_env.yml`` (Python 3.9, torch 1.13.1)
4. download the v1.1 (DiffDock-L) checkpoints and verify the archive hash
5. download the ESM2-650M weights used for the protein language-model embeddings
6. precompute the SO(2)/SO(3) lookup tables that upstream otherwise builds on the first run

Everything is recorded in ``<workdir>/diffdock_setup.json`` so a run manifest can point at it.
Setup time is therefore kept separate from per-complex inference time.
"""
from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ._util import gpu_info, read_json, run_streaming, sha256_file, utc_now, write_json
from .drive import cache_restore, cache_store

UPSTREAM_REPO = "https://github.com/gcorso/DiffDock.git"
UPSTREAM_COMMIT = "85c49b60d3e0b0182a59ee43a34a6d7036981284"  # main on 2025-05-02, the revision tested here

MODELS_URL = "https://github.com/gcorso/DiffDock/releases/download/v1.1/diffdock_models.zip"
MODELS_ZIP_SHA256 = "5a95b6a1555be47ab1d6f0a8ffd25152f7fe32f5956005bb821e13e7a37d4a3d"
MODELS_SUBDIR = Path("workdir") / "v1.1"  # matches model_dir in default_inference_args.yaml

MICROMAMBA_URL = "https://github.com/mamba-org/micromamba-releases/releases/download/1.5.8-0/micromamba-linux-64"

ESM_WEIGHT_URLS = [
    "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt",
    "https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt",
]
# hashes of the files as downloaded on a Colab T4 on 2026-10-21; a different hash means a different (or corrupt) file
ESM_WEIGHT_SHA256 = {
    "esm2_t33_650M_UR50D.pt": "ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c",
    "esm2_t33_650M_UR50D-contact-regression.pt": "8ffe6edbd4173dc8d45c2cd5cb27d43aad77ec26b4c768200c58ae1f96693575",
}

# upstream's so3.py / torus.py write these into the *current directory* on first import (13 min on a Colab VM, 465 MB);
# all six must exist, so3 writes four of them first and torus the other two
TABLE_FILES = [
    ".so3_omegas_array4.npy", ".so3_cdf_vals4.npy", ".so3_score_norms4.npy", ".so3_exp_score_norms4.npy",
    ".p.npy", ".score.npy",
]
MODELS_MARKER = ".extracted_from"  # records which archive hash a complete extraction came from

PACKAGE_DIR = Path(__file__).resolve().parent
PATCH_FILE = PACKAGE_DIR / "patches" / "diffdock_seed_and_scores.patch"
ENV_FILE = PACKAGE_DIR / "env" / "diffdock_colab_env.yml"

IMPORT_CHECK = "import torch, torch_geometric, torch_cluster, torch_scatter, e3nn, esm, prody, rdkit, pandas, yaml"
VERSION_PACKAGES = [
    "torch", "torch-geometric", "torch-cluster", "torch-scatter", "torch-sparse", "e3nn", "fair-esm",
    "prody", "rdkit", "numpy", "scipy", "pandas", "networkx", "scikit-learn", "biopython",
]


@dataclass(frozen=True)
class Layout:
    """Where everything lives under one work directory."""

    workdir: Path

    @property
    def repo(self) -> Path:
        return self.workdir / "DiffDock"

    @property
    def mamba(self) -> Path:
        return self.workdir / "bin" / "micromamba"

    @property
    def mamba_root(self) -> Path:
        return self.workdir / "micromamba"

    @property
    def env_prefix(self) -> Path:
        return self.mamba_root / "envs" / "diffdock"

    @property
    def python(self) -> Path:
        return self.env_prefix / "bin" / "python"

    @property
    def models_dir(self) -> Path:
        return self.repo / MODELS_SUBDIR

    @property
    def torch_home(self) -> Path:
        return self.workdir / "torch_home"

    @property
    def setup_info(self) -> Path:
        return self.workdir / "diffdock_setup.json"


def _run(cmd: List[str], *, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> None:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    rc = run_streaming([str(c) for c in cmd], cwd=cwd, env=env)
    if rc != 0:
        raise RuntimeError(f"command failed with exit code {rc}: {' '.join(str(c) for c in cmd)}")


def _download(url: str, dest: Path, *, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            print(f"download {url} (attempt {attempt}/{retries})", flush=True)
            with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out, 1 << 20)
            tmp.replace(dest)
            return
        except Exception as e:  # noqa: BLE001 - retry any network failure
            last = e
            time.sleep(2 * attempt)
    raise RuntimeError(f"could not download {url}: {last!r}")


def ensure_micromamba(layout: Layout) -> None:
    if layout.mamba.exists():
        return
    _download(MICROMAMBA_URL, layout.mamba)
    layout.mamba.chmod(layout.mamba.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def clone_upstream(layout: Layout) -> str:
    """Clone (once) and check out the pinned revision. Returns the checked-out commit."""
    if not (layout.repo / ".git").exists():
        layout.workdir.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--quiet", UPSTREAM_REPO, str(layout.repo)])
    head = _git_out(layout.repo, "rev-parse", "HEAD")
    if head != UPSTREAM_COMMIT:
        _run(["git", "checkout", "--quiet", "--detach", UPSTREAM_COMMIT], cwd=layout.repo)
        head = _git_out(layout.repo, "rev-parse", "HEAD")
    if head != UPSTREAM_COMMIT:
        raise RuntimeError(f"expected upstream {UPSTREAM_COMMIT}, got {head}")
    return head


def _git_out(repo: Path, *args: str) -> str:
    import subprocess

    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def _git_ok(repo: Path, *args: str) -> bool:
    import subprocess

    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True).returncode == 0


def _patched_files() -> List[str]:
    import subprocess

    out = subprocess.run(["git", "apply", "--numstat", str(PATCH_FILE)], capture_output=True, text=True, check=True).stdout
    return [line.split("\t")[2] for line in out.splitlines() if line.count("\t") >= 2]


def apply_patch(layout: Layout) -> str:
    """Apply our small patch (``--seed`` for python/numpy/torch/RDKit, ``scores.json``, readable failure logs).

    Idempotent, and safe when an *older* version of the patch is already applied in this managed clone: the files the
    patch touches are first restored to the pinned commit. Returns the patch sha256.
    """
    patch = str(PATCH_FILE)
    if _git_ok(layout.repo, "apply", "--reverse", "--check", patch):
        print("patch already applied")
    else:
        files = _patched_files()
        if files and not _git_ok(layout.repo, "diff", "--quiet", "--", *files):
            print("restoring files modified by an earlier patch version:", ", ".join(files))
            _run(["git", "checkout", "--quiet", "--", *files], cwd=layout.repo)
        _run(["git", "apply", "--check", patch], cwd=layout.repo)
        _run(["git", "apply", patch], cwd=layout.repo)
    return sha256_file(PATCH_FILE)


def render_env_spec(cpu: bool) -> str:
    """Return the environment yml text; ``cpu`` swaps the CUDA 11.7 wheels for CPU wheels."""
    text = ENV_FILE.read_text(encoding="utf-8")
    if cpu:
        text = text.replace("whl/cu117", "whl/cpu").replace("+cu117", "+cpu").replace("pt113cu117", "pt113cpu")
    return text


def env_is_ready(layout: Layout) -> bool:
    if not layout.python.exists():
        return False
    import subprocess

    return subprocess.run([str(layout.python), "-c", IMPORT_CHECK], capture_output=True).returncode == 0


def create_env(layout: Layout, cpu: bool) -> None:
    if env_is_ready(layout):
        print("environment already usable")
        return
    spec = layout.workdir / ("diffdock_env_cpu.yml" if cpu else "diffdock_env.yml")
    spec.write_text(render_env_spec(cpu), encoding="utf-8", newline="\n")
    if layout.env_prefix.exists():  # half-built environment from an interrupted run
        shutil.rmtree(layout.env_prefix)
    env = {"MAMBA_ROOT_PREFIX": str(layout.mamba_root)}
    _run([str(layout.mamba), "create", "-y", "-p", str(layout.env_prefix), "-f", str(spec)], env=env)
    _run([str(layout.python), "-c", IMPORT_CHECK + "; print('imports ok')"])


def env_versions(layout: Layout) -> Dict[str, Optional[str]]:
    import subprocess

    code = (
        "import json, importlib.metadata as m\n"
        f"names = {VERSION_PACKAGES!r}\n"
        "out = {}\n"
        "for n in names:\n"
        "    try: out[n] = m.version(n)\n"
        "    except m.PackageNotFoundError: out[n] = None\n"
        "import sys; out['python'] = sys.version.split()[0]\n"
        "print(json.dumps(out))\n"
    )
    res = subprocess.run([str(layout.python), "-c", code], capture_output=True, text=True, check=True)
    import json

    return json.loads(res.stdout.strip().splitlines()[-1])


MODEL_FILES = [
    "score_model/model_parameters.yml",
    "score_model/best_ema_inference_epoch_model.pt",
    "confidence_model/model_parameters.yml",
    "confidence_model/best_model_epoch75.pt",
]


def model_files_present(layout: Layout) -> bool:
    """True only for a *complete* extraction of the pinned archive (the marker is written last, after a rename)."""
    d = layout.models_dir
    marker = d / MODELS_MARKER
    if not marker.is_file() or marker.read_text().strip() != MODELS_ZIP_SHA256:
        return False
    return all((d / f).is_file() and (d / f).stat().st_size > 0 for f in MODEL_FILES)


def _extract_models_atomic(archive: Path, layout: Layout) -> None:
    """Unpack into a sibling temp directory, check it, then rename: an interrupted extraction leaves nothing half-done."""
    final = layout.models_dir
    tmp = final.with_name(final.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():  # refuse path traversal even though the archive hash is pinned
            target = (tmp / info.filename).resolve()
            if tmp.resolve() not in target.parents and target != tmp.resolve():
                raise RuntimeError(f"unsafe path in model archive: {info.filename!r}")
        zf.extractall(tmp)
        sizes = {i.filename: i.file_size for i in zf.infolist() if not i.is_dir()}
    for f in MODEL_FILES:
        p = tmp / f
        if not p.is_file() or p.stat().st_size != sizes.get(f):
            raise RuntimeError(f"model archive did not contain a complete {f}")
    (tmp / MODELS_MARKER).write_text(MODELS_ZIP_SHA256 + "\n")
    if final.exists():
        shutil.rmtree(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, final)


def download_models(layout: Layout) -> Dict[str, str]:
    """Fetch + verify the DiffDock-L v1.1 checkpoints. Returns {relative path: sha256}."""
    if not model_files_present(layout):
        archive = layout.workdir / "diffdock_models_v1.1.zip"
        if not archive.exists():
            _download(MODELS_URL, archive)
        digest = sha256_file(archive)
        if digest != MODELS_ZIP_SHA256:
            archive.unlink()
            raise RuntimeError(f"model archive hash mismatch: {digest} != {MODELS_ZIP_SHA256}")
        _extract_models_atomic(archive, layout)
    return {
        p.relative_to(layout.models_dir).as_posix(): sha256_file(p)
        for p in sorted(layout.models_dir.rglob("*"))
        if p.is_file() and p.name != MODELS_MARKER
    }


def torch_env(layout: Layout) -> Dict[str, str]:
    """Environment variables for every subprocess that runs DiffDock code."""
    return {"TORCH_HOME": str(layout.torch_home), "PYTHONUNBUFFERED": "1"}


def esm_weights_ok(layout: Layout) -> bool:
    ckpt_dir = layout.torch_home / "hub" / "checkpoints"
    return all(
        (ckpt_dir / n).is_file() and sha256_file(ckpt_dir / n) == h for n, h in ESM_WEIGHT_SHA256.items()
    )


def download_esm_weights(layout: Layout) -> Dict[str, str]:
    """Download (if missing) and hash-check the ESM2-650M weights DiffDock uses for protein embeddings."""
    ckpt_dir = layout.torch_home / "hub" / "checkpoints"
    for url in ESM_WEIGHT_URLS:
        name = url.rsplit("/", 1)[-1]
        dest = ckpt_dir / name
        if dest.is_file() and sha256_file(dest) != ESM_WEIGHT_SHA256[name]:
            print(f"{name}: hash mismatch, downloading again")
            dest.unlink()
        if not dest.is_file():
            _download(url, dest)
        got = sha256_file(dest)
        if got != ESM_WEIGHT_SHA256[name]:
            dest.unlink()
            raise RuntimeError(f"{name}: sha256 {got} != expected {ESM_WEIGHT_SHA256[name]}")
    return dict(ESM_WEIGHT_SHA256)


def tables_present(layout: Layout) -> bool:
    return all((layout.repo / f).is_file() and (layout.repo / f).stat().st_size > 0 for f in TABLE_FILES)


def precompute_tables(layout: Layout) -> None:
    """Build upstream's SO(2)/SO(3) lookup tables once. A partial set (interrupted run) is deleted and rebuilt."""
    if tables_present(layout):
        return
    for f in TABLE_FILES:
        (layout.repo / f).unlink(missing_ok=True)
    _run([str(layout.python), "utils/precompute_series.py"], cwd=layout.repo, env=torch_env(layout))
    missing = [f for f in TABLE_FILES if not (layout.repo / f).is_file()]
    if missing:
        raise RuntimeError(f"precompute_series.py finished but did not write {missing}")


def _cached_models(layout: Layout, cache_dir: Optional[Path]) -> Dict[str, str]:
    archive = layout.workdir / "diffdock_models_v1.1.zip"
    if not model_files_present(layout):
        cache_restore(cache_dir, archive.name, archive, MODELS_ZIP_SHA256)
    hashes = download_models(layout)
    cache_store(cache_dir, archive.name, archive)
    return hashes


def _cached_esm_weights(layout: Layout, cache_dir: Optional[Path]) -> Dict[str, str]:
    ckpt_dir = layout.torch_home / "hub" / "checkpoints"
    for n, h in ESM_WEIGHT_SHA256.items():
        cache_restore(cache_dir, n, ckpt_dir / n, h)
    hashes = download_esm_weights(layout)
    for n in ESM_WEIGHT_SHA256:
        cache_store(cache_dir, n, ckpt_dir / n)
    return hashes


def _cached_tables(layout: Layout, cache_dir: Optional[Path]) -> None:
    """The lookup tables take ~13 minutes to build and are identical on every VM: keep them in the cache."""
    if not tables_present(layout):
        for f in TABLE_FILES:
            cache_restore(cache_dir, f"lookup_tables/{f}", layout.repo / f)
    precompute_tables(layout)
    for f in TABLE_FILES:
        cache_store(cache_dir, f"lookup_tables/{f}", layout.repo / f)


GPU_SMOKE_CODE = """
import json, sys, torch
out = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
if out["cuda_available"]:
    dev = torch.device("cuda")
    out["device"] = torch.cuda.get_device_name(0)
    out["capability"] = list(torch.cuda.get_device_capability(0))
    out["compiled_arches"] = torch.cuda.get_arch_list()
    a = torch.randn(256, 256, device=dev)
    torch.cuda.synchronize(); (a @ a).sum().item()
    import torch_scatter, torch_cluster
    src = torch.randn(10, 4, device=dev); idx = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], device=dev)
    torch_scatter.scatter_add(src, idx, dim=0)
    x = torch.randn(32, 3, device=dev)
    torch_cluster.radius_graph(x, r=2.0)
    torch.cuda.synchronize()
    out["kernels_ok"] = True
print(json.dumps(out))
"""


def gpu_smoke_test(layout: Layout) -> dict:
    """Run one CUDA kernel of each kind DiffDock needs (matmul, torch_scatter, torch_cluster) in the pinned environment.

    torch 1.13.1+cu117 ships no kernels newer than sm_86 (an H100 is sm_90), so a Colab GPU that is too new would otherwise
    only fail minutes into docking. Returns the record; raises with advice when CUDA exists but a kernel fails.
    """
    import json
    import subprocess

    res = subprocess.run([str(layout.python), "-c", GPU_SMOKE_CODE], capture_output=True, text=True, env={**os.environ, **torch_env(layout)})
    if res.returncode != 0:
        raise RuntimeError(
            "GPU smoke test failed in the DiffDock environment (torch 1.13.1+cu117 supports GPUs up to sm_86: T4, L4 and A100 "
            "work; an H100 needs a different torch build). Pick another runtime GPU.\n" + res.stderr[-1500:]
        )
    record = json.loads(res.stdout.strip().splitlines()[-1])
    print("GPU smoke test:", json.dumps(record))
    return record


def setup(
    workdir: os.PathLike | str,
    *,
    cpu: bool = False,
    steps: Optional[List[str]] = None,
    cache_dir: Optional[os.PathLike | str] = None,
) -> dict:
    """Run all setup steps. ``cache_dir`` (e.g. on Google Drive) keeps the big downloads between VMs."""
    layout = Layout(Path(workdir).resolve())
    cache = Path(cache_dir) if cache_dir else None
    layout.workdir.mkdir(parents=True, exist_ok=True)
    info: dict = read_json(layout.setup_info) if layout.setup_info.exists() else {}
    timings: Dict[str, float] = info.get("step_seconds", {})
    wanted = set(steps) if steps else None
    worked: List[str] = []
    cached: List[str] = []

    def step(name: str, fn: Callable[[], object], done: Optional[Callable[[], bool]] = None) -> object:
        """Run a step. ``step_seconds[name]`` keeps the time of the run that did real work: re-running a finished step
        (a few seconds of checks) must not overwrite the 13 minutes the table build really took."""
        if wanted is not None and name not in wanted:
            return None
        already = bool(done()) if done else False
        print(f"\n=== {name}{' (already done)' if already else ''} ===", flush=True)
        t0 = time.time()
        result = fn()
        elapsed = round(time.time() - t0, 1)
        if already and name in timings:
            cached.append(name)
        else:
            timings[name] = elapsed
            worked.append(name)
        return result

    step("micromamba", lambda: ensure_micromamba(layout), lambda: layout.mamba.exists())
    commit = step("upstream", lambda: clone_upstream(layout), lambda: (layout.repo / ".git").exists())
    patch_sha = step("patch", lambda: apply_patch(layout), lambda: _git_ok(layout.repo, "apply", "--reverse", "--check", str(PATCH_FILE)))
    step("environment", lambda: create_env(layout, cpu), lambda: env_is_ready(layout))
    model_hashes = step("models", lambda: _cached_models(layout, cache), lambda: model_files_present(layout))
    esm_hashes = step("esm_weights", lambda: _cached_esm_weights(layout, cache), lambda: esm_weights_ok(layout))
    step("lookup_tables", lambda: _cached_tables(layout, cache), lambda: tables_present(layout))
    gpu_record = step("gpu_smoke", lambda: gpu_smoke_test(layout)) if layout.python.exists() else None

    info.update(
        {
            "schema_version": 2,
            "completed_utc": utc_now(),
            "upstream": {"repo": UPSTREAM_REPO, "commit": commit or info.get("upstream", {}).get("commit")},
            "patch": {"file": PATCH_FILE.name, "sha256": patch_sha or info.get("patch", {}).get("sha256")},
            "env_spec": {"file": ENV_FILE.name, "sha256": sha256_file(ENV_FILE), "cpu_build": cpu},
            "checkpoints": {
                "archive_url": MODELS_URL,
                "archive_sha256": MODELS_ZIP_SHA256,
                "files_sha256": model_hashes or info.get("checkpoints", {}).get("files_sha256"),
            },
            "esm_weights": {"urls": ESM_WEIGHT_URLS, "files_sha256": esm_hashes or info.get("esm_weights", {}).get("files_sha256")},
            "packages": env_versions(layout) if layout.python.exists() else None,
            "gpu": gpu_info(),
            "gpu_smoke": gpu_record or info.get("gpu_smoke"),
            "step_seconds": timings,
            "setup_seconds_total": round(sum(timings.values()), 1),
            "last_run": {"steps_with_work": worked, "steps_already_done": cached, "cache_dir": str(cache) if cache else None},
        }
    )
    write_json(layout.setup_info, info)
    print(f"\nsetup finished in {info['setup_seconds_total']} s -> {layout.setup_info}")
    return info


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", default="/content/diffdock_work")
    p.add_argument("--cpu", action="store_true", help="CPU-only wheels (local testing; Colab should use the default)")
    p.add_argument("--steps", nargs="*", help="run only these steps (default: all)")
    p.add_argument("--cache-dir", help="directory (e.g. on Google Drive) that keeps checkpoints / ESM weights between VMs")
    args = p.parse_args(argv)
    setup(args.workdir, cpu=args.cpu, steps=args.steps, cache_dir=args.cache_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
