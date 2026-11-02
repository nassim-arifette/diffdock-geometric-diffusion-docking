"""Run the real docking notebook on a Colab VM and check what it produced (Phase 1 acceptance test).

On the VM (unpacked repo at /content/repo)::

    python /content/repo/docking_tools/scripts/colab_verify.py [--drive-root /content/drive/MyDrive/DiffDock_colab]

It executes ``notebooks/02_docking_colab.ipynb`` top to bottom with the form defaults (bundled 1A0Q example),
saves the executed notebook, then independently re-checks the run directory the notebook wrote, runs a second
docking with the same seed to see whether poses repeat, and writes ``phase1_verification.json``
(and copies everything to Drive when it is mounted). Exit code 0 only if every hard check passed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CHECKS: list = []


def check(name: str, ok: bool, detail: str = "", hard: bool = True) -> bool:
    CHECKS.append({"check": name, "ok": bool(ok), "detail": detail, "hard": hard})
    print(f"[{'PASS' if ok else ('FAIL' if hard else 'NOTE')}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)
    return bool(ok)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/content/diffdock_work")
    ap.add_argument("--drive-root", default="/content/drive/MyDrive/DiffDock_colab")
    ap.add_argument("--out", default="/content/phase1_out")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit==2026.9.1", "py3Dmol==2.5.5", "pyyaml",
                    "nbclient", "ipykernel", "ipywidgets", "pandas"], check=True)

    from docking_tools import setup_env
    from docking_tools.drive import ensure_drive
    from docking_tools.manifest import load_manifest, verify_manifest
    from docking_tools.poses import collect_poses

    drive_root = ensure_drive(args.drive_root)
    check("google drive mounted and writable", drive_root is not None, str(drive_root), hard=False)

    # ---- execute the real notebook (this includes the one-time setup) --------------------------------------
    import nbformat
    from nbclient import NotebookClient

    env = dict(os.environ, DIFFDOCK_WORKDIR=args.workdir, PYTHONUNBUFFERED="1")
    if drive_root is not None:
        env["DOCKING_DRIVE_ROOT"] = str(drive_root)
    os.environ.update(env)
    nb_path = REPO / "notebooks" / "02_docking_colab.ipynb"
    nb = nbformat.read(str(nb_path), as_version=4)
    executed = out / "02_docking_colab.executed.ipynb"
    client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(nb_path.parent)}})
    notebook_error = None
    t0 = time.time()
    try:
        client.execute()
    except Exception as e:  # noqa: BLE001 - keep going: the executed notebook shows how far it got
        notebook_error = f"{e.__class__.__name__}: {str(e)[-1500:]}"
    nbformat.write(nb, str(executed))
    check("notebook 02 executed without error", notebook_error is None, notebook_error or f"{time.time() - t0:.0f}s")

    # ---- independent checks on what the notebook left behind -----------------------------------------------
    layout = setup_env.Layout(Path(args.workdir))
    info = json.loads(layout.setup_info.read_text()) if layout.setup_info.exists() else {}
    check("setup recorded the pinned upstream commit", info.get("upstream", {}).get("commit") == setup_env.UPSTREAM_COMMIT,
          str(info.get("upstream", {}).get("commit")))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=layout.repo, capture_output=True, text=True).stdout.strip()
    check("DiffDock checkout is at the pinned commit", head == setup_env.UPSTREAM_COMMIT, head)
    reverse = subprocess.run(["git", "apply", "--reverse", "--check", str(setup_env.PATCH_FILE)], cwd=layout.repo)
    check("seed/scores patch is applied", reverse.returncode == 0)
    check("checkpoint archive hash matches", info.get("checkpoints", {}).get("archive_sha256") == setup_env.MODELS_ZIP_SHA256
          and model_files_ok(layout))
    pk = info.get("packages") or {}
    check("env packages are the pinned versions",
          pk.get("python", "").startswith("3.9") and pk.get("torch", "").startswith("1.13.1") and pk.get("e3nn") == "0.5.1",
          json.dumps({k: pk.get(k) for k in ("python", "torch", "torch-geometric", "e3nn", "fair-esm", "rdkit", "prody")}))
    cuda = subprocess.run([str(layout.python), "-c", "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"],
                          capture_output=True, text=True)
    check("pinned torch sees the GPU", cuda.stdout.startswith("True"), cuda.stdout.strip() or cuda.stderr[-300:])

    runs = sorted((Path(args.workdir) / "runs").glob("*")) if (Path(args.workdir) / "runs").exists() else []
    check("exactly one run directory was created", len(runs) == 1, str([r.name for r in runs]))
    manifest = None
    if runs:
        run_dir = runs[-1]
        try:
            manifest = load_manifest(run_dir)
            verify_manifest(run_dir, manifest)
            check("manifest loads and every referenced file hash verifies", True, run_dir.name)
        except Exception as e:  # noqa: BLE001
            check("manifest loads and every referenced file hash verifies", False, repr(e))
        if manifest:
            c = manifest["complexes"][0]
            check("run status ok", manifest["status"] == "ok", str(manifest.get("error")))
            check("10 poses recorded", len(c["poses"]) == 10, str(len(c["poses"])))
            poses = collect_poses(run_dir / "predictions" / c["complex_name"], c["complex_name"], expected=10)
            check("all poses parse as 3D molecules (re-checked independently)", len(poses) == 10)
            confs = [p["confidence"] for p in c["poses"]]
            check("full-precision confidences present and sorted best-first",
                  all(x is not None for x in confs) and confs == sorted(confs, reverse=True), str([round(x, 4) for x in confs[:4]]))
            check("seed, upstream commit, checkpoint hashes and GPU recorded in manifest",
                  manifest["seed"] == 0 and manifest["upstream"]["commit"] == setup_env.UPSTREAM_COMMIT
                  and bool(manifest["checkpoints"]["files_sha256"]) and bool(manifest["environment"]["gpu"]),
                  json.dumps(manifest["environment"]["gpu"]))
            check("effective parameters are the run config (CLI could not override them)",
                  manifest["inference_parameters"]["samples_per_complex"] == 10 and "--samples_per_complex" not in manifest["command"])
            check("top-1 pose not absurd: ligand centre within 15 A of the receptor centre", centre_ok(run_dir, c, poses), hard=False)
            check("ligand preview/receptor cleaning recorded", c["receptor"]["n_residues"] > 50, str(c["receptor"]["n_residues"]))
            check("run + export + inputs saved to Drive",
                  drive_root is not None and (drive_root / "runs" / run_dir.name / "manifest.json").exists()
                  and (drive_root / "exports" / f"{run_dir.name}.tar.gz").exists()
                  and (drive_root / "datasets" / f"{c['complex_name']}_inputs").exists(),
                  str(drive_root), hard=drive_root is not None)

            # ---- determinism: same seed, second docking ------------------------------------------------
            try:
                from docking_tools.run import new_run_dir, prepare_inputs, run_diffdock

                second_dir = new_run_dir(Path(args.workdir) / "runs_repeat", "repeat")
                prepared = prepare_inputs(second_dir, complex_name=c["complex_name"],
                                          receptor=str(Path(args.workdir) / "DiffDock/data/1a0q/1a0q_protein_processed.pdb"),
                                          ligand=str(Path(args.workdir) / "DiffDock/data/1a0q/1a0q_ligand.sdf"))
                m2 = run_diffdock(args.workdir, prepared, seed=0, echo=False)
                same_conf = [p["confidence"] for p in m2["complexes"][0]["poses"]] == confs
                same_files = [p["sha256"] for p in m2["complexes"][0]["poses"]] == [p["sha256"] for p in c["poses"]]
                check("same seed reproduces identical confidences", same_conf, hard=False)
                check("same seed reproduces byte-identical pose files", same_files, hard=False)
                check("second run completed", m2["status"] == "ok",
                      f"{m2['timings']['inference_wall_seconds']}s vs first {manifest['timings']['inference_wall_seconds']}s")
            except Exception as e:  # noqa: BLE001
                check("second run completed", False, repr(e)[-400:])

    hard_fail = [c for c in CHECKS if c["hard"] and not c["ok"]]
    report = {
        "total_seconds": round(time.time() - t_start),
        "passed": sum(c["ok"] for c in CHECKS), "failed_hard": len(hard_fail), "checks": CHECKS,
        "setup_seconds": info.get("setup_seconds_total"), "step_seconds": info.get("step_seconds"),
        "manifest_summary": None if not manifest else {
            "run_id": manifest["run_id"], "status": manifest["status"], "gpu": manifest["environment"]["gpu"],
            "inference_wall_seconds": manifest["timings"]["inference_wall_seconds"],
            "gpu_peak_memory_mib": manifest["gpu_peak_memory_mib"], "top_confidences": [p["confidence"] for p in manifest["complexes"][0]["poses"][:5]],
        },
    }
    (out / "phase1_verification.json").write_text(json.dumps(report, indent=2))
    if drive_root is not None:
        d = drive_root / "verification"
        d.mkdir(exist_ok=True)
        for f in out.iterdir():
            (d / f.name).write_bytes(f.read_bytes())
        print("verification files saved to", d)
    print(f"\nRESULT: {report['passed']}/{len(CHECKS)} checks passed, {len(hard_fail)} hard failure(s)")
    return 1 if hard_fail else 0


def model_files_ok(layout) -> bool:
    from docking_tools.setup_env import model_files_present

    return model_files_present(layout)


def centre_ok(run_dir, c, poses) -> bool:
    import numpy as np
    from rdkit import Chem

    rec = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in
                    (run_dir / c["receptor"]["path"]).read_text().splitlines() if l.startswith("ATOM")])
    mol = Chem.SDMolSupplier(str(poses[0].path), sanitize=False, removeHs=False)[0]
    lig = mol.GetConformer().GetPositions()
    return float(np.linalg.norm(lig.mean(0) - rec.mean(0))) < 15.0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)
