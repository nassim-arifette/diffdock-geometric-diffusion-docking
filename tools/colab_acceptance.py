"""Logged GPU acceptance: isolated helper install, real docking, then physical evaluation."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True, type=Path)
    ap.add_argument("--state", required=True, type=Path)
    args = ap.parse_args()
    state = {"status": "running", "stages": [], "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    status_file = args.state / "status.json"

    def save():
        target = status_file.with_suffix(".tmp")
        target.write_text(json.dumps(state, indent=2))
        target.replace(status_file)

    def run(stage, command):
        state["current_stage"] = stage
        save()
        print(f"STAGE {stage}: {command}", flush=True)
        subprocess.run([str(x) for x in command], cwd=args.project, check=True)
        state["stages"].append(stage)
        save()

    save()
    try:
        run("gpu_inventory", ["nvidia-smi"])
        # Colab's system interpreter may lack a working ensurepip. Use an
        # independent Python 3.11 matching the locally verified helper stack.
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError("uv is required to provision the isolated Python 3.11 helper")
        env = args.project / ".helper-py311"
        run("helper_environment", [uv, "venv", "--python", "3.11", "--seed", str(env)])
        python = env / "bin" / "python"
        run("helper_packages", [python, "-m", "pip", "install", "-r", "requirements-dev.txt"])
        run("gpu_acceptance", [python, "-u", "docking_tools/scripts/colab_smoke.py", "--workdir", "/content/diffdock_work", "--out", "/content/phase1_out", "--repeat"])
        report = json.loads(Path("/content/phase1_out/phase1_verification.json").read_text())
        for i, item in enumerate(report["runs"]):
            run(f"physical_evaluation_{i + 1}", [python, "-m", "docking_tools.evaluation", item["run_dir"]])
            run(f"inspector_{i + 1}", [python, "-m", "docking_tools.inspector", item["run_dir"]])
        state["status"] = "ok"
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        state["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
    return 0 if state["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
