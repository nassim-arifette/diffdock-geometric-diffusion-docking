"""Build notebook 05; scientific settings and selection are frozen before inference."""
import nbformat as nbf
from nbtools import bootstrap, code, finish, md


def build():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""# Fixed-candidate docking pilot

Does filtering, independent scoring or local refinement improve selection from
the same DiffDock-L pool? See `docs/research.md` for the primary-source review.
This notebook contains **no precomputed benchmark result**. Start with development;
apply the unchanged protocol to evaluation. Default: 5 dev + 20 eval complexes,
20 candidates and 3 seeds. Reference coordinates are evaluation-only.
"""),
        bootstrap(["posebusters==0.6.5"]),
        code('''
from pathlib import Path
import json
from docking_tools.drive import ensure_drive, persist_dataset
from docking_tools._util import gpu_info

SPLIT = "dev"  #@param ["dev", "eval"]
GNINA = ""  #@param {type:"string"}
SAVE_TO_DRIVE = True  #@param {type:"boolean"}
WORKDIR = Path("/content/diffdock_work")
DATASET = Path("/content/posebusters_pilot")
OUTPUT = Path("/content/pilot_" + SPLIT)
SEEDS = [0, 1, 2]
SAMPLES = 20
if gpu_info() is None:
    raise RuntimeError("Select a GPU runtime before starting this pilot.")
drive_root = ensure_drive() if SAVE_TO_DRIVE else None
print("Hardware:", gpu_info())
print("Selectors:", "A/B/C/D" if GNINA else "A/B only; C/D unmeasured")
'''),
        md("""## Freeze inputs

Intersect the original archive with the final 308-case list, apply declared
eligibility checks, and select one ligand per PDB before running any model.
All exclusions and hashes are kept. This does not establish unseen protein families.
"""),
        code('''
from docking_tools.pilot_dataset import prepare_dataset
from docking_tools.pilot import read_protocol
manifest = DATASET / "pilot_manifest.json"
if not manifest.exists():
    prepare_dataset(DATASET, n_dev=5, n_eval=20, seeds=SEEDS, samples_per_complex=SAMPLES)
protocol, cases = read_protocol(manifest, SPLIT)
if protocol["seeds"] != SEEDS or protocol["samples_per_complex"] != SAMPLES:
    raise ValueError("Frozen protocol settings differ; use a new dataset/output directory.")
print(json.dumps({"selected": [c["complex_id"] for c in cases], "seeds": SEEDS, "samples": SAMPLES}, indent=2))
if drive_root:
    persist_dataset(DATASET, drive_root, "posebusters_pilot")
'''),
        md("""## Generate once and evaluate

Finished pools are verified and reused. Failures are retained rather than replaced
with easier cases. GNINA is optional: specify its binary to measure C/D, whose
version, hash and commands are saved. The empty value measures only A/B.
"""),
        code('''
from docking_tools.pilot import run_pilot
state = run_pilot(manifest, workdir=WORKDIR, out=OUTPUT, split=SPLIT,
                  gnina=GNINA or None, drive_root=drive_root)
print({k: state[k] for k in ("status", "n_expected", "n_ok", "n_failed")})
'''),
        md("""## Paired results

Main outcome: top-1 heavy-atom RMSD **< 2 Å** AND physical validity against the
prepared receptor. No ligand alignment. Failures remain in denominators;
bootstrap uncertainty groups every seed of one complex. This is a pilot.
"""),
        code('''
from docking_tools.pilot_summary import summarize_pilot
from IPython.display import Markdown, display
summary = summarize_pilot(OUTPUT)
display(Markdown((OUTPUT / "summary.md").read_text()))
if drive_root:
    for filename in ("summary.json", "summary.md"):
        persist_dataset(OUTPUT / filename, drive_root, "pilot_" + SPLIT + "_summary")
'''),
        md("## Inspect saved predictions and reference overlays"),
        code('''
from docking_tools.inspector import write_inspector
from IPython.display import FileLink
completed = [c for c in state["cases"].values() if c["status"] == "ok"]
if completed:
    html = write_inspector(OUTPUT / completed[0]["run_dir"])
    display(FileLink(str(html)))
else:
    print("No complete evaluation: inspect pilot_state.json and failure logs.")
'''),
    ]
    return finish(nb, "05_fixed_candidate_pilot.ipynb")


if __name__ == "__main__":
    build()
