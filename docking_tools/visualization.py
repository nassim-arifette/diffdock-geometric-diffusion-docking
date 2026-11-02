"""Look at the receptor, the ligand you asked for, and the poses DiffDock actually produced."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .inputs import validate_input_csv
from .manifest import load_manifest

RECEPTOR_COLOR = "#b8bcc4"
LIGAND_CARBON = "cyanCarbon"
POCKET_RADIUS = 5.0


def pose_table(manifest: dict) -> List[dict]:
    """One row per pose with the full-precision score next to the file it belongs to."""
    rows = []
    for c in manifest["complexes"]:
        for p in c["poses"]:
            rows.append(
                {
                    "complex": c["complex_name"],
                    "rank": p["rank"],
                    "pose_id": p["pose_id"],
                    "confidence": p["confidence"],
                    "confidence_in_filename": p["confidence_rounded"],
                    "file": p["path"],
                }
            )
    return rows


def build_view(receptor_pdb: str, ligand_sdf: str, *, width: int = 640, height: int = 460, pocket: bool = True):
    """Subdued receptor cartoon, ligand sticks, and (optionally) side chains within 5 A of the ligand."""
    import py3Dmol

    view = py3Dmol.view(width=width, height=height)
    view.addModel(receptor_pdb, "pdb")
    view.addModel(ligand_sdf, "sdf")
    view.setStyle({"model": 0}, {"cartoon": {"color": RECEPTOR_COLOR, "opacity": 0.8}})
    if pocket:
        view.addStyle(
            {"model": 0, "within": {"distance": POCKET_RADIUS, "sel": {"model": 1}}},
            {"stick": {"colorscheme": "grayCarbon", "radius": 0.12}},
        )
    view.setStyle({"model": 1}, {"stick": {"colorscheme": LIGAND_CARBON, "radius": 0.22}})
    view.zoomTo({"model": 1})
    return view


def _read(run_dir: Path, rel: str) -> str:
    return (run_dir / rel).read_text(encoding="utf-8")


def show_pose(run: Path | str, complex_name: Optional[str] = None, rank: int = 1, **kw):
    """Render one stored pose (receptor + predicted ligand) straight from the manifest."""
    run_dir = Path(run)
    manifest = load_manifest(run_dir)
    c = _complex(manifest, complex_name)
    pose = next((p for p in c["poses"] if p["rank"] == rank), None)
    if pose is None:
        raise KeyError(f"rank {rank} not in run (have {[p['rank'] for p in c['poses']]})")
    view = build_view(_read(run_dir, c["receptor"]["path"]), _read(run_dir, pose["path"]), **kw)
    return view


def _complex(manifest: dict, name: Optional[str]) -> dict:
    cs = manifest["complexes"]
    if name is None:
        if len(cs) != 1:
            raise ValueError(f"run has several complexes {[c['complex_name'] for c in cs]}; pass complex_name")
        return cs[0]
    for c in cs:
        if c["complex_name"] == name:
            return c
    raise KeyError(name)


def pose_inspector(run: Path | str, complex_name: Optional[str] = None, **kw):
    """Dropdown over the poses of a stored run; each selection redraws receptor + that pose."""
    import ipywidgets as w
    from IPython.display import display

    run_dir = Path(run)
    manifest = load_manifest(run_dir)
    c = _complex(manifest, complex_name)
    receptor = _read(run_dir, c["receptor"]["path"])

    options = []
    for p in c["poses"]:
        conf = f"{p['confidence']:.6f}" if p["confidence"] is not None else f"~{p['confidence_rounded']:.2f}"
        options.append((f"rank {p['rank']:>2}  confidence {conf}", p["rank"]))
    dropdown = w.Dropdown(options=options, value=c["poses"][0]["rank"], description="Pose", layout=w.Layout(width="420px"))
    info = w.HTML()
    out = w.Output()

    def draw(_=None):
        pose = next(p for p in c["poses"] if p["rank"] == dropdown.value)
        info.value = f"<code>{pose['pose_id']}</code> &nbsp; file: <code>{pose['path']}</code>"
        with out:
            out.clear_output(wait=True)
            build_view(receptor, _read(run_dir, pose["path"]), **kw).show()

    dropdown.observe(draw, names="value")
    display(w.VBox([dropdown, info, out]))
    draw()
    return dropdown


def preview_inputs(csv_path: Path | str, *, width: int = 520, height: int = 380) -> None:
    """Show exactly what is about to be docked: receptor chains and the ligand, read back from the CSV."""
    import py3Dmol
    from IPython.display import display
    from rdkit.Chem import Draw

    for r in validate_input_csv(Path(csv_path)):
        print(f"complex   : {r.complex_name}")
        print(f"receptor  : {r.protein_path}")
        print(f"ligand    : {r.ligand_description}")
        pdb_text = Path(r.protein_path).read_text(encoding="utf-8")
        n_res = len({(l[21:22], l[22:27]) for l in pdb_text.splitlines() if l.startswith("ATOM")})
        chains = sorted({l[21:22] for l in pdb_text.splitlines() if l.startswith("ATOM")})
        print(f"            {n_res} residues, chains {chains}")

        view = py3Dmol.view(width=width, height=height)
        view.addModel(pdb_text, "pdb")
        view.setStyle({"cartoon": {"color": RECEPTOR_COLOR}})
        view.zoomTo()
        view.show()

        mol = _ligand_mol(r.ligand_description)
        if mol is not None:
            display(Draw.MolToImage(mol, size=(360, 260)))
        else:
            print("            (ligand could not be drawn in 2D, but is passed to DiffDock as written)")


def _ligand_mol(description: str):
    """RDKit molecule for a SMILES string or a .sdf/.mol2 path (None when RDKit cannot read it)."""
    from rdkit import Chem

    path = Path(description)
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".sdf":
            suppl = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=True)
            return suppl[0] if len(suppl) else None
        if suffix == ".mol2":
            return Chem.MolFromMol2File(str(path), sanitize=True, removeHs=True)
        return None
    return Chem.MolFromSmiles(description)
