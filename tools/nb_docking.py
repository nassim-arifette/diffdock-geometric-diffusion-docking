"""notebooks/02_docking_colab.ipynb and notebooks/03_pdbbind_benchmark_prep.ipynb"""
import nbformat as nbf

from nbtools import bootstrap, code, finish, md


def build_docking():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# DiffDock docking on Colab: one complex, from input to inspected poses

This notebook takes **one protein + one ligand**, runs DiffDock-L, and opens **the poses it actually produced**.
Every step reads the same input file, which is shown to you right before the run, and every run leaves a
`manifest.json` recording what was run (upstream commit, checkpoint hashes, parameters, seed, GPU, timings, pose
files with hashes and full-precision scores).

| Step | What happens | Typical time (T4) |
|---|---|---|
| 1-2 | helper code, your inputs | seconds |
| 3-4 | **one-time setup** of the pinned environment (cached on Drive for the next VM) | ~17 min the first time, ~2 min with a warm Drive cache |
| 5-6 | resolve inputs, **preview exactly what will be docked** | seconds |
| 7 | docking | minutes (includes protein-language-model embedding) |
| 8 | inspect poses | seconds |
| 9 | export | seconds |

Use a **GPU runtime** (Runtime > Change runtime type): a T4, L4 or A100. It also runs on CPU, but much more slowly.
Leave both inputs empty to dock the bundled **1A0Q** example (its native ligand file is the starting point; the
coordinates of the crystal pose are not used, DiffDock regenerates a conformer).

**About the seed.** `SEED` seeds python, numpy, torch and RDKit's starting-conformer generator. GPU scatter operations can
still differ in the last digits, so identical seeds give near-identical, not guaranteed bit-identical, poses.
"""),
        bootstrap(),
        md("## 2. What to dock\nGive **both** a receptor and a ligand, or leave **both** empty for the bundled example."),
        code('''
#@title 2. Inputs
RECEPTOR = ""  #@param {type:"string"}
#@markdown PDB id (e.g. `1A0Q`) or path to a `.pdb` file. Waters, ions, cofactors and any co-crystallised ligand are removed.
LIGAND = ""  #@param {type:"string"}
#@markdown SMILES, PubChem CID, or path to an `.sdf`/`.mol2`/`.pdbqt`/`.pdb` file. Use isomeric SMILES for chiral molecules.
CHAINS = ""  #@param {type:"string"}
#@markdown Comma-separated chain ids to keep (empty = all chains).
COMPLEX_NAME = ""  #@param {type:"string"}
SAMPLES = 10  #@param {type:"integer"}
INFERENCE_STEPS = 20  #@param {type:"integer"}
SEED = 0  #@param {type:"integer"}
SAVE_FRAMES = True  #@param {type:"boolean"}
#@markdown Also save every reverse-diffusion step of every pose (for the diffusion slider in the inspector).
SAVE_TO_DRIVE = True  #@param {type:"boolean"}
#@markdown Saves the inputs, every run (failed ones too), the export archive and the big downloads to Google Drive.
DRIVE_DIR = "/content/drive/MyDrive/DiffDock_colab"  #@param {type:"string"}

WORKDIR = os.environ.get("DIFFDOCK_WORKDIR", "/content/diffdock_work")
DRIVE_DIR = os.environ.get("DOCKING_DRIVE_ROOT", DRIVE_DIR)
RECEPTOR, LIGAND = RECEPTOR.strip(), LIGAND.strip()
if bool(RECEPTOR) != bool(LIGAND):
    raise ValueError("Give both RECEPTOR and LIGAND, or leave both empty to use the bundled 1A0Q example.")
print("work directory:", WORKDIR)
'''),
        code('''
#@title Mount Google Drive (optional)
from docking_tools.drive import ensure_drive

drive_root = ensure_drive(DRIVE_DIR) if SAVE_TO_DRIVE else None
print("saving to:", drive_root if drive_root else "(Drive not used: results stay on this VM)")
'''),
        md("## 3. Runtime check"),
        code('''
#@title 3. GPU?
from docking_tools._util import gpu_info

gpu = gpu_info()
print(gpu if gpu else "No GPU found: docking will run on CPU and be slow. Runtime > Change runtime type > GPU.")
''', "colab-setup"),
        md("""
## 4. One-time setup (pinned environment)
Installs micromamba, clones `gcorso/DiffDock` at a **pinned commit**, applies a small patch (adds `--seed`, writes
full-precision scores, makes failures readable in the log), builds the Python 3.9 / torch 1.13.1 environment, downloads
the DiffDock-L checkpoints and the ESM2 weights and **checks their SHA-256**, precomputes upstream's lookup tables (the
slow step: ~13 min), and smoke-tests the GPU with the pinned torch. With a Drive cache, the big downloads and the tables
are restored instead of rebuilt. Safe to re-run: finished steps are skipped and their original timings are kept.
"""),
        code('''
#@title 4. Set up DiffDock
from docking_tools.setup_env import setup

setup_info = setup(WORKDIR, cache_dir=drive_root / "cache" if drive_root else None)
print("upstream commit:", setup_info["upstream"]["commit"])
print("setup time (s) :", setup_info["setup_seconds_total"], "| GPU smoke test:", (setup_info.get("gpu_smoke") or {}).get("device"))
''', "colab-setup"),
        md("## 5. Resolve the inputs\nOne input file (`input_protein_ligand.csv`) is written inside the run directory and is used for the preview, the docking and the export."),
        code('''
#@title 5. Prepare inputs
from docking_tools.run import default_complex_name, new_run_dir, prepare_inputs

bundled = Path(WORKDIR) / "DiffDock" / "data" / "1a0q"
use_example = not RECEPTOR
receptor = str(bundled / "1a0q_protein_processed.pdb") if use_example else RECEPTOR
ligand = str(bundled / "1a0q_ligand.sdf") if use_example else LIGAND
name = COMPLEX_NAME.strip() or ("1a0q" if use_example else default_complex_name(RECEPTOR, LIGAND))

run_dir = new_run_dir(Path(WORKDIR) / "runs", name)
prepared = prepare_inputs(
    run_dir,
    complex_name=name,
    receptor=receptor,
    ligand=ligand,
    chains=[c.strip() for c in CHAINS.split(",") if c.strip()] or None,
    reference=ligand if use_example else None,
    reference_note="native ligand of 1A0Q from the PDBBind-processed set (not used by docking)" if use_example else None,
)
print("run directory :", run_dir)
print("input file    :", prepared.csv_path)
rec = prepared.complexes[0].receptor
print(f"receptor      : {rec.source}, chains {rec.chains}, {rec.n_residues} residues")
if rec.removed_hetero or rec.removed_waters:
    print(f"removed       : {rec.removed_waters} water atoms, other hetero atoms {rec.removed_hetero}")
lig = prepared.complexes[0].ligand
if lig.unspecified_stereocenters:
    print(f"WARNING: the ligand SMILES leaves {lig.unspecified_stereocenters} stereocentre(s) undefined; "
          "DiffDock will dock one arbitrary stereoisomer. Use an isomeric SMILES to fix it.")
'''),
        code('''
#@title 6. Preview exactly what will be docked
from docking_tools.visualization import preview_inputs

preview_inputs(prepared.csv_path)
'''),
        md("## 7. Dock\nSuccess means the expected number of SDF poses exist **and parse back as 3D molecules**; anything else raises with the log tail, and the failed run's `manifest.json` is kept for inspection."),
        code('''
#@title 7. Run DiffDock
from docking_tools.drive import persist_dataset, persist_run
from docking_tools.run import run_diffdock

if drive_root:  # the dataset: exactly the inputs this run uses, saved before anything can go wrong
    persist_dataset(run_dir / "inputs", drive_root, f"{name}_inputs")
overrides = {"samples_per_complex": SAMPLES, "inference_steps": INFERENCE_STEPS, "actual_steps": INFERENCE_STEPS - 1}
if SAVE_FRAMES:
    overrides["save_visualisation"] = True
try:
    manifest = run_diffdock(
        WORKDIR,
        prepared,
        # overrides go into this run's config file: upstream applies the config AFTER the command line,
        # so a flag such as --samples_per_complex 40 next to --config would be silently ignored.
        overrides=overrides,
        seed=SEED,
    )
finally:  # results are saved automatically, also when the run failed (the failed manifest + log are the evidence)
    if drive_root:
        print("saved run to:", persist_run(run_dir, drive_root))
c = manifest["complexes"][0]
print(f"\\nstatus: {manifest['status']}   poses: {len(c['poses'])}   docking wall time: {manifest['timings']['inference_wall_seconds']} s")
print("GPU peak memory (MiB, whole GPU):", manifest["gpu_peak_memory_mib"])
print("manifest:", run_dir / "manifest.json")
'''),
        md("## 8. Inspect the poses\nThe viewer loads the receptor **and the selected predicted ligand** from the saved files. Side chains within 5 Å of the ligand are shown as grey sticks. The dropdown shows the full-precision confidence; the SDF file name only carries two decimals. Validity checks, contacts, the reference overlay and the diffusion slider are in `04_pose_inspector.ipynb` and in the web app."),
        code('''
#@title 8. Pose table and viewer
import pandas as pd
from docking_tools.visualization import pose_inspector, pose_table

display(pd.DataFrame(pose_table(manifest)).drop(columns=["complex"]))
pose_inspector(run_dir)
'''),
        md("## 9. Export\nThe archive is built from the manifest and **checked**: every pose, the receptor, the input file and the config must be inside, and file hashes must still match."),
        code('''
#@title 9. Export
from docking_tools.export import export_run, list_archive

archive = export_run(run_dir, Path(WORKDIR) / "exports")
names = list_archive(archive)
print(f"{archive}  ({archive.stat().st_size / 1e6:.2f} MB, {len(names)} files)")
print("\\n".join("  " + n for n in names[:12]) + ("\\n  ..." if len(names) > 12 else ""))

if drive_root:
    print("saved run + archive to:", persist_run(run_dir, drive_root, archive))
if IN_COLAB:
    from google.colab import files
    files.download(str(archive))
'''),
        md("## 10. Reopen a saved run later\nNo GPU, no setup: unpack the archive (or point at any run directory containing `manifest.json`) and view it."),
        code('''
#@title 10. Reopen a saved run
SAVED_RUN = ""  #@param {type:"string"}
from docking_tools.manifest import load_manifest, verify_manifest

reopen = Path(SAVED_RUN) if SAVED_RUN else run_dir
verify_manifest(reopen, load_manifest(reopen))
print("manifest verified:", reopen)
pose_inspector(reopen)
'''),
        md("""
---
**Notes**
- `6AGT` (used in the first notebook) is *P. falciparum* lysyl-tRNA synthetase, whose crystallographic ligand is `9X0`, not 4-methoxybenzonitrile. Docking that SMILES into it is an *exploratory prediction* with no experimental reference; it can't support an accuracy claim. The redocking demo with the native 9X0 lives in the redocking section of the experiment notebook.
- Benchmark preparation (many PDBBind complexes, embeddings) lives in `03_pdbbind_benchmark_prep.ipynb`.
"""),
    ]
    return finish(nb, "02_docking_colab.ipynb")


def build_benchmark_prep():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# PDBBind subset preparation (benchmark input)

Moved out of the single-example notebook. Single-complex docking (`02_docking_colab.ipynb`) needs **none** of this.

This notebook prepares only the PDBBind complexes **you list** (or the first *N* of an upstream split):
the 2 GB archive is downloaded once, but only the requested complexes are unpacked, and the protein FASTA and
ESM2 embeddings are computed for those complexes alone (the old notebook embedded all 24 926 sequences).

Outputs go to `<workdir>/benchmark/pdbbind_subset_<N>_<hash>/` (the hash identifies the exact set of ids):
`PDBBind_processed/<id>/`, `prepared_for_esm.fasta`, `esm2_embeddings.pt`, `split_subset.txt`, `subset_manifest.json`.
"""),
        bootstrap(),
        code('''
#@title 2. Which complexes?
IDS = "1a0q"  #@param {type:"string"}
#@markdown Space- or comma-separated PDB ids. Leave empty to take the first N ids of an upstream split file instead.
SPLIT_FILE = "data/splits/timesplit_test"  #@param {type:"string"}
LIMIT = 20  #@param {type:"integer"}
SAVE_TO_DRIVE = True  #@param {type:"boolean"}
DRIVE_DIR = "/content/drive/MyDrive/DiffDock_colab"  #@param {type:"string"}
WORKDIR = os.environ.get("DIFFDOCK_WORKDIR", "/content/diffdock_work")
DRIVE_DIR = os.environ.get("DOCKING_DRIVE_ROOT", DRIVE_DIR)

from docking_tools.drive import ensure_drive
drive_root = ensure_drive(DRIVE_DIR) if SAVE_TO_DRIVE else None
'''),
        code('''
#@title 3. Environment (same pinned setup as the docking notebook)
from docking_tools.setup_env import setup

setup(WORKDIR, cache_dir=drive_root / "cache" if drive_root else None)
''', "colab-setup"),
        code('''
#@title 4. Prepare the subset
from docking_tools import benchmark_prep as bp
from docking_tools.setup_env import Layout

layout = Layout(Path(WORKDIR).resolve())
ids = [i for i in IDS.replace(",", " ").split()] or bp.read_split_ids(layout.repo / SPLIT_FILE, LIMIT)
print(len(ids), "complexes:", ids[:10], "..." if len(ids) > 10 else "")
summary = bp.prepare_subset(WORKDIR, ids)
print({k: summary[k] for k in ("n_complexes", "n_sequences", "data_dir", "embeddings", "split_file")})

if drive_root:  # the dataset (complexes + FASTA + embeddings) is saved automatically
    from docking_tools.drive import persist_dataset
    print("saved to:", persist_dataset(Path(summary["data_dir"]).parent, drive_root, Path(summary["data_dir"]).parent.name))
'''),
    ]
    return finish(nb, "03_pdbbind_benchmark_prep.ipynb")
