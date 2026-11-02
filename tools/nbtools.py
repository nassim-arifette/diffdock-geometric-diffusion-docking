"""Helpers shared by the notebook builders. The notebooks in ``notebooks/`` are generated: edit the builders, not the JSON."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO / "notebooks"

# pinned notebook-side dependencies (DiffDock itself runs in its own pinned environment)
PIP_PINS = ["rdkit==2026.9.1", "py3Dmol==2.5.5", "pyyaml"]


def md(src: str):
    return nbf.v4.new_markdown_cell(src.strip("\n"))


def code(src: str, *tags: str):
    cell = nbf.v4.new_code_cell(src.strip("\n"))
    if tags:
        cell.metadata["tags"] = list(tags)
    return cell


def bootstrap(extra_pip=()):
    """Cell 1 of every notebook: find (or fetch and update) the repository that contains ``docking_tools/``."""
    pins = list(PIP_PINS) + list(extra_pip)
    return code(
        '''
#@title 1. Get the helper code
import os, subprocess, sys
from pathlib import Path

REPO_URL = "https://github.com/nassim-arifette/diffdock-geometric-diffusion-docking.git"  #@param {type:"string"}
REPO_REF = "main"  #@param {type:"string"}

IN_COLAB = "google.colab" in sys.modules
here = Path.cwd()
checkout = next((p for p in (here, here.parent) if (p / "docking_tools").is_dir()), None)
if checkout is None:  # fresh Colab VM: fetch the repository that contains docking_tools/
    checkout = Path("/content/diffdock-geometric-diffusion-docking")
    if not checkout.exists():
        subprocess.run(["git", "clone", "--quiet", "--branch", REPO_REF, REPO_URL, str(checkout)], check=True)
    else:  # a checkout from an earlier cell run: pick up newer commits instead of silently using stale code
        for args in (["fetch", "--quiet", "origin", REPO_REF], ["checkout", "--quiet", REPO_REF], ["merge", "--ff-only", "--quiet", "FETCH_HEAD"]):
            if subprocess.run(["git", "-C", str(checkout), *args]).returncode != 0:
                print("warning: could not update", checkout, "- using it as it is")
                break
if str(checkout) not in sys.path:
    sys.path.insert(0, str(checkout))

if IN_COLAB:  # the notebook's own Python only needs these; DiffDock runs in its own pinned environment
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", %s], check=True)

import docking_tools
print("docking_tools", docking_tools.__version__, "from", checkout)
''' % repr(pins)
    )


def finish(nb, name: str, out_dir: Path = NOTEBOOK_DIR) -> Path:
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["colab"] = {"name": name, "provenance": []}
    nbf.validate(nb)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    nbf.write(nb, str(path))
    print("wrote", path)
    return path
