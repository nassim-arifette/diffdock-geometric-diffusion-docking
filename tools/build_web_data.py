"""Export the archived 1A0Q acceptance run for the website.

Reads one GPU acceptance run (DiffDock-L, T4, ten candidates) and writes
web/public/data/case.json with: the prepared receptor, the crystallographic
ligand, every candidate pose with its confidence and PoseBusters checks, the
real reverse-diffusion frames, and per-frame quantities recomputed here with
the project's own functions:

* symmetry-aware heavy-atom RMSD to the crystal pose, without alignment
  (docking_tools.evaluation.receptor_frame_rmsd);
* minimum heavy-atom distance to the receptor.

The export refuses to write unless the final-frame RMSD it recomputes equals the
RMSD recorded in the run's evaluation.json, and the final frame equals the pose.
"""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from docking_tools.evaluation import read_molecules, receptor_frame_rmsd  # noqa: E402
from docking_tools.frames import trajectory  # noqa: E402
from docking_tools.geometry import heavy_atom_graph, torsion_edges  # noqa: E402

RUN = ROOT / "colab_artifacts/acceptance_20261101/runs/20261101-222233_1a0q_smoke_1"
REVIEW = ROOT / "colab_artifacts/acceptance_review/physical_review_v2.json"
OUT = ROOT / "web/public/data"
CLASH = 2.2  # angstrom, heavy atom to heavy atom: far below any real contact


def protein_heavy(pdb_text):
    xyz = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[76:78].strip() != "H":
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(xyz)


def main():
    from rdkit import Chem

    evaluation = json.loads((RUN / "evaluation.json").read_text(encoding="utf-8"))
    if evaluation.get("status") != "ok":
        raise SystemExit("the run's evaluation is not complete")
    case = evaluation["complexes"][0]
    name = case["complex_name"]
    args = {}
    for line in (RUN / "config/inference_args.yaml").read_text(encoding="utf-8").splitlines():
        if ":" in line and not line.startswith(" "):
            k, v = line.split(":", 1)
            args[k.strip()] = v.strip()
    pdb_text = (RUN / f"inputs/{name}_protein_processed_protein.pdb").read_text(encoding="utf-8")
    protein = protein_heavy(pdb_text)
    reference = read_molecules(RUN / f"inputs/{name}_reference.sdf")[0]
    ref_xyz = reference.GetConformer().GetPositions()
    pocket_center = ref_xyz.mean(axis=0)

    def min_dist(xyz):
        d = np.linalg.norm(xyz[:, None, :] - protein[None, :, :], axis=2)
        return d.min(axis=1)

    poses = []
    for p in sorted(case["poses"], key=lambda q: q["rank"]):
        sdf = RUN / p["path"]
        mol = read_molecules(sdf)[0]
        traj = trajectory(RUN / f"predictions/{name}/rank{p['rank']}_reverseprocess.pdb", sdf)
        frames, rmsds, dmins = [], [], []
        for fr in traj.frames:
            m = Chem.Mol(mol)
            conf = m.GetConformer()
            for i, (x, y, z) in enumerate(fr.coords):
                conf.SetAtomPosition(i, (float(x), float(y), float(z)))
            rmsds.append(round(receptor_frame_rmsd(m, reference), 3))
            dmins.append(round(float(min_dist(fr.coords).min()), 3))
            frames.append(np.round(fr.coords, 3).tolist())
        if abs(rmsds[-1] - p["rmsd_angstrom"]) > 0.01:
            raise SystemExit(f"rank {p['rank']}: recomputed RMSD {rmsds[-1]} != recorded {p['rmsd_angstrom']}")
        final = mol.GetConformer().GetPositions()
        if np.abs(np.array(frames[-1]) - final).max() > 0.02:
            raise SystemExit(f"rank {p['rank']}: final frame differs from the pose")
        atom_d = min_dist(final)
        poses.append({
            "rank": p["rank"], "confidence": p["confidence"], "rmsd": p["rmsd_angstrom"], "valid": p["valid"],
            "failed_checks": [k for k, v in p["checks"].items() if v is False],
            "checks": p["checks"],
            "molblock": Chem.MolToMolBlock(mol),
            "frames": frames, "frame_kinds": [f.kind for f in traj.frames], "frame_steps": [f.step for f in traj.frames],
            "frame_rmsd": rmsds, "frame_min_distance": dmins,
            "clash_atoms": [int(i) for i in np.where(atom_d < CLASH)[0]],
            "min_distance": round(float(atom_d.min()), 3),
        })

    _, bonds, adj, symbols = heavy_atom_graph(reference)
    torsions = [{"u": e.u, "v": e.v, "moving": e.moving_atoms()} for e in torsion_edges(adj)]
    review = json.loads(REVIEW.read_text(encoding="utf-8"))["runs"][0]["independent_distance"]
    data = {
        "schema_version": 1,
        "complex": name.upper(),
        "run_id": evaluation["run_id"],
        "created_utc": evaluation["created_utc"],
        "backends": evaluation["backends"],
        "protocol": evaluation["protocol"],
        "inference": {k: args.get(k) for k in ("samples_per_complex", "inference_steps", "actual_steps", "sigma_schedule", "ckpt", "confidence_ckpt")},
        "clash_threshold": CLASH,
        "pocket_center": np.round(pocket_center, 3).tolist(),
        "protein_pdb": pdb_text,
        "reference": {"molblock": Chem.MolToMolBlock(reference), "smiles": Chem.MolToSmiles(reference),
                      "min_distance": round(float(min_dist(ref_xyz).min()), 3),
                      "symbols": symbols, "bonds": bonds, "torsions": torsions},
        "review": review,
        "poses": poses,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "case.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"{name}: {len(poses)} poses, {sum(len(p['frames']) for p in poses)} frames, "
          f"{len(torsions)} torsions, best RMSD {min(p['rmsd'] for p in poses):.2f} A -> {OUT / 'case.json'}")


if __name__ == "__main__":
    main()
