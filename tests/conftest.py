"""Shared fixtures: a tiny synthetic protein + ligand and a fake 'upstream DiffDock' that mimics its output layout."""
import json
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PDB_TEXT = "\n".join(
    [
        "HEADER    TEST",
        "MODEL        1",
        "ATOM      1  N   ALA A   1      11.104  13.207   2.100  1.00 20.00           N",
        "ATOM      2  CA  ALA A   1      12.560  13.207   2.100  1.00 20.00           C",
        "ATOM      3  C   ALA A   1      13.060  14.650   2.100  1.00 20.00           C",
        "ATOM      4  O   ALA A   1      12.300  15.600   2.100  1.00 20.00           O",
        "ATOM      5  CB  ALA A   1      13.060  12.500   3.300  1.00 20.00           C",
        "ATOM      6  N   SER B   2      14.400  14.800   2.100  1.00 20.00           N",
        "ATOM      7  CA ASER B   2      15.000  16.100   2.100  0.60 20.00           C",
        "ATOM      8  CA BSER B   2      15.100  16.200   2.200  0.40 20.00           C",
        "HETATM    9  SE  MSE B   3      16.000  17.000   2.100  1.00 20.00          SE",
        "HETATM   10  C1  9X0 A 601       5.000   5.000   5.000  1.00 20.00           C",
        "HETATM   11  O   HOH A 701       1.000   1.000   1.000  1.00 20.00           O",
        "HETATM   12 ZN    ZN A 801       2.000   2.000   2.000  1.00 20.00          ZN",
        "ENDMDL",
        "MODEL        2",
        "ATOM      1  N   ALA A   1      99.104  13.207   2.100  1.00 20.00           N",
        "ENDMDL",
        "END",
    ]
) + "\n"


@pytest.fixture
def pdb_file(tmp_path) -> Path:
    p = tmp_path / "toy.pdb"
    p.write_text(PDB_TEXT)
    return p


def _embed(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    return mol


@pytest.fixture
def ligand_sdf(tmp_path) -> Path:
    from rdkit import Chem

    p = tmp_path / "lig.sdf"
    w = Chem.SDWriter(str(p))
    w.write(_embed("COc1ccc(C#N)cc1"))
    w.close()
    return p


FAKE_INFERENCE = textwrap.dedent(
    """
    \"\"\"Stand-in for upstream inference.py: same CLI, same output layout (heavy-atom SDFs, scores.json, optional
    reverse-process frames in upstream's real layout), no torch.\"\"\"
    import argparse, csv, json, os, sys
    import numpy as np
    import yaml
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D

    MODE = os.environ.get("FAKE_DIFFDOCK_MODE", "ok")
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=argparse.FileType("r"))
    p.add_argument("--protein_ligand_csv")
    p.add_argument("--out_dir")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--samples_per_complex", type=int, default=10)
    p.add_argument("--save_visualisation", action="store_true", default=False)
    args = p.parse_args()
    cfg = yaml.safe_load(args.config)
    for k, v in cfg.items():          # same precedence as upstream: the YAML wins over the command line
        setattr(args, k, v)
    if MODE == "crash":
        print("Traceback (most recent call last): boom", file=sys.stderr)
        sys.exit(1)
    rows = list(csv.DictReader(open(args.protein_ligand_csv)))
    for row in rows:                   # upstream creates every output directory before doing any work
        os.makedirs(os.path.join(args.out_dir, row["complex_name"]), exist_ok=True)
    if MODE == "silent":               # upstream exits 0 even when every complex was skipped
        sys.exit(0)


    def pdb_model(mol_h, heavy_xyz):
        \"\"\"One MODEL like upstream's PDBFile: all atoms, but only the heavy-atom coordinates are updated.\"\"\"
        m = Chem.Mol(mol_h)
        conf = m.GetConformer()
        for i, (x, y, z) in enumerate(heavy_xyz):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        block = Chem.MolToPDBBlock(m).split("\\n")[:-2]
        return "MODEL\\n" + "\\n".join(block) + "\\nENDMDL\\n"


    for row in rows:
        d = os.path.join(args.out_dir, row["complex_name"])
        n = args.samples_per_complex - (1 if MODE == "short" else 0)
        smi = row["ligand_description"]
        mol = Chem.MolFromSmiles(smi) if not smi.lower().endswith((".sdf", ".mol2", ".pdbqt", ".pdb")) else Chem.SDMolSupplier(smi, removeHs=False)[0]
        mol_h = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_h, randomSeed=args.seed or 0)
        conf_vals = [1.0 - 0.37 * i - 0.0004321 for i in range(n)]
        n_heavy = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() > 1)
        base = mol_h.GetConformer().GetPositions()[:n_heavy]
        for i, c in enumerate(conf_vals):
            final = base + np.array([0.1 * i, 0.0, 0.0])
            m = Chem.RemoveHs(Chem.Mol(mol_h))             # upstream writes heavy atoms only (remove_hs)
            for j in range(m.GetNumAtoms()):
                m.GetConformer().SetAtomPosition(j, Point3D(*[float(v) for v in final[j]]))
            for name in ([f"rank{i+1}.sdf"] if i == 0 else []) + [f"rank{i+1}_confidence{c:.2f}.sdf"]:
                w = Chem.SDWriter(os.path.join(d, name))
                w.write(m)
                w.close()
            if args.save_visualisation:
                steps = 5
                start = final + np.array([6.0, -4.0, 3.0])
                models = [pdb_model(mol_h, base), pdb_model(mol_h, base), pdb_model(mol_h, start)]
                models.append(pdb_model(mol_h, final))     # order 2: the final pose overwrites denoising step 0
                for t in range(1, steps):
                    models.append(pdb_model(mol_h, start + (final - start) * (t + 1) / steps))
                open(os.path.join(d, f"rank{i+1}_reverseprocess.pdb"), "w").write("".join(models))
        if MODE == "corrupt":
            open(os.path.join(d, "rank2_confidence0.63.sdf"), "w").write("not an sdf\\n")
        json.dump({"complex_name": row["complex_name"], "seed": args.seed, "confidence": conf_vals},
                  open(os.path.join(d, "scores.json"), "w"))
    print("fake diffdock done")
    """
)

DEFAULT_CFG = "samples_per_complex: 10\ninference_steps: 20\nactual_steps: 19\nmodel_dir: ./workdir/v1.1/score_model\n"


@pytest.fixture
def fake_workdir(tmp_path) -> Path:
    """workdir/DiffDock with a fake inference.py, default config, and the bundled-example files."""
    wd = tmp_path / "work"
    repo = wd / "DiffDock"
    (repo / "data" / "1a0q").mkdir(parents=True)
    (repo / "inference.py").write_text(FAKE_INFERENCE)
    (repo / "default_inference_args.yaml").write_text(DEFAULT_CFG)
    (repo / "data" / "1a0q" / "1a0q_protein_processed.pdb").write_text(PDB_TEXT)
    from rdkit import Chem

    w = Chem.SDWriter(str(repo / "data" / "1a0q" / "1a0q_ligand.sdf"))
    w.write(_embed("COc1ccc(C#N)cc1"))
    w.close()
    (wd / "diffdock_setup.json").write_text(
        json.dumps(
            {
                "upstream": {"repo": "x", "commit": "abc"},
                "patch": {"sha256": "def"},
                "checkpoints": {"files_sha256": {"score_model/x.pt": "00"}},
                "packages": {"torch": "1.13.1+cu117"},
                "setup_seconds_total": 12.5,
            }
        )
    )
    return wd
