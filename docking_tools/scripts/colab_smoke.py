"""Run the notebook's docking pipeline headlessly, retaining evidence on failure.

    python docking_tools/scripts/colab_smoke.py --workdir /content/diffdock_work

Mount Drive with the Colab CLI first, then pass --drive-root. This runner never
opens an authentication prompt. The default requires a GPU; --allow-cpu is an
explicit diagnostic fallback. Notebook dependencies must already be installed.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from docking_tools import setup_env
from docking_tools._util import gpu_info, read_json, sha256_file, utc_now, write_json
from docking_tools.drive import persist_dataset, persist_run
from docking_tools.export import export_run
from docking_tools.manifest import load_manifest, verify_manifest
from docking_tools.poses import collect_poses
from docking_tools.run import new_run_dir, prepare_inputs, run_diffdock


def verify_run(run_dir: Path, *, samples: int, seed: int) -> dict:
    """A successful subprocess is insufficient: re-read poses, trajectories and hashes."""
    manifest = load_manifest(run_dir)
    verify_manifest(run_dir, manifest)
    if manifest.get("status") != "ok":
        raise RuntimeError(f"run did not succeed: {manifest.get('error')}")
    if manifest["seed"] != seed:
        raise RuntimeError("recorded seed differs from the requested seed")
    c = manifest["complexes"][0]
    poses = collect_poses(run_dir / "predictions" / c["complex_name"], c["complex_name"],
                          expected=samples, expect_frames=True)
    if any(p.confidence is None for p in poses):
        raise RuntimeError("full-precision scores are missing")
    scores = [p.confidence for p in poses]
    if scores != sorted(scores, reverse=True):
        raise RuntimeError("poses are not sorted by confidence")
    from docking_tools.frames import trajectory
    for pose in poses:
        trajectory(pose.frames_path, pose.path)
    return manifest


def run_smoke(args) -> int:
    workdir, out = Path(args.workdir).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {
        "schema_version": 1, "started_utc": utc_now(), "status": "running",
        "checks": [], "runs": [], "error": None,
        "requested": {"samples": args.samples, "seed": args.seed, "repeat": args.repeat,
                      "require_gpu": not args.allow_cpu, "skip_setup": args.skip_setup},
        "source_sha256": {
            p.relative_to(REPO).as_posix(): sha256_file(p)
            for p in sorted((REPO / "docking_tools").rglob("*"))
            if p.is_file() and p.suffix in (".py", ".yml", ".patch")
        },
    }
    report_path = out / "phase1_verification.json"
    drive_root = None

    def checkpoint():
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        write_json(report_path, report)
        if drive_root is not None:
            write_json(drive_root / "verification" / "phase1_verification.json", report)

    def checked(name, detail=None):
        report["checks"].append({"name": name, "passed": True, "detail": detail})
        print(f"[PASS] {name}", flush=True)
        checkpoint()

    checkpoint()
    try:
        if args.drive_root:
            candidate = Path(args.drive_root).resolve()
            # Do not create a look-alike MyDrive tree on an unmounted VM.
            if not candidate.parent.is_dir():
                raise RuntimeError("Drive parent directory is absent; mount Drive before --drive-root")
            candidate.mkdir(exist_ok=True)
            drive_root = candidate
            checked("Drive output is writable", str(drive_root))
        gpu = gpu_info()
        report["gpu"] = gpu
        if not args.allow_cpu and not gpu:
            raise RuntimeError("No GPU found. Allocate a GPU Colab runtime, or explicitly use --allow-cpu.")
        checked("runtime selected", gpu or "CPU diagnostic requested")
        if not args.skip_setup:
            setup_env.setup(workdir, cpu=args.allow_cpu and not gpu,
                            cache_dir=drive_root / "cache" if drive_root else None)
        layout = setup_env.Layout(workdir)
        info = read_json(layout.setup_info)
        if info.get("upstream", {}).get("commit") != setup_env.UPSTREAM_COMMIT:
            raise RuntimeError("setup does not record the pinned upstream commit")
        if not args.allow_cpu and not (info.get("gpu_smoke") or {}).get("kernels_ok"):
            raise RuntimeError("the pinned environment has not passed its GPU kernel smoke test")
        report["setup"] = info
        checked("pinned environment ready")

        bundled = layout.repo / "data" / "1a0q"
        receptor, ligand = bundled / "1a0q_protein_processed.pdb", bundled / "1a0q_ligand.sdf"
        manifests = []
        for attempt in range(1 + int(args.repeat)):
            run_dir = new_run_dir(workdir / "runs", f"1a0q_smoke_{attempt + 1}")
            report["runs"].append({"run_dir": str(run_dir), "status": "preparing"})
            checkpoint()
            prepared = prepare_inputs(run_dir, complex_name="1a0q", receptor=str(receptor),
                                      ligand=str(ligand), reference=str(ligand),
                                      reference_note="bundled PDBBind 1A0Q native ligand; evaluation only")
            if drive_root:
                persist_dataset(run_dir / "inputs", drive_root, run_dir.name + "_inputs")
            try:
                run_diffdock(workdir, prepared, seed=args.seed,
                             overrides={"samples_per_complex": args.samples, "save_visualisation": True},
                             python_exe=args.python_exe)
                manifest = verify_run(run_dir, samples=args.samples, seed=args.seed)
                archive = export_run(run_dir, out / "exports")
                report["runs"][-1].update(status="ok", archive=str(archive),
                                         manifest_sha256=sha256_file(run_dir / "manifest.json"))
                if drive_root:
                    persist_run(run_dir, drive_root, archive)
                manifests.append(manifest)
                checked(f"run {attempt + 1}: poses, scores, trajectories and export verified")
            finally:
                if drive_root:
                    persist_run(run_dir, drive_root)
        if args.repeat:
            first, second = [m["complexes"][0]["poses"] for m in manifests]
            report["repeatability"] = {
                "same_confidences": [p["confidence"] for p in first] == [p["confidence"] for p in second],
                "byte_identical_poses": [p["sha256"] for p in first] == [p["sha256"] for p in second],
                "interpretation": "diagnostic only; CUDA scatter does not guarantee bitwise determinism",
            }
        report["status"] = "ok"
        return_code = 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if report["runs"] and report["runs"][-1]["status"] != "ok":
            report["runs"][-1].update(status="failed", error=report["error"])
        traceback.print_exc()
        return_code = 1
    finally:
        report["finished_utc"] = utc_now()
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        write_json(report_path, report)
        if drive_root is not None:
            try:
                write_json(drive_root / "verification" / "phase1_verification.json", report)
            except OSError as exc:
                print(f"Drive report save failed; local evidence retained: {exc}", file=sys.stderr)
                return_code = 1
        print(f"Verification report: {report_path}", flush=True)
    return return_code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", default="/content/diffdock_work")
    ap.add_argument("--out", default="/content/phase1_out")
    ap.add_argument("--drive-root", help="mounted Drive location; no authentication is attempted")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", action="store_true", help="repeat the run as a determinism diagnostic")
    ap.add_argument("--skip-setup", action="store_true", help="reuse an already verified pinned environment")
    ap.add_argument("--allow-cpu", action="store_true", help="explicitly permit a CPU diagnostic")
    ap.add_argument("--python-exe", help="alternate inference Python (normally the pinned environment)")
    args = ap.parse_args(argv)
    if args.samples < 1:
        ap.error("--samples must be positive")
    return run_smoke(args)


if __name__ == "__main__":
    sys.exit(main())
