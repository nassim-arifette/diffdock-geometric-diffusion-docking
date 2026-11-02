"""Freeze a small, auditable subset of the final 308-case PoseBusters benchmark.

Download only the published 53.7 MB archive and the authors' corrected case list.
Never use all 428 cases in the original ZIP as if they were the final benchmark.
Sampling happens before docking and is independent of predictions or RMSD.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
import zipfile
from pathlib import Path

from ._util import relpath, sha256_file, utc_now, write_json

ARCHIVE_URL = "https://zenodo.org/records/8278563/files/posebusters_paper_data.zip?download=1"
ARCHIVE_MD5 = "f004ac7c4e68317b5348497d2bb6bee6"
FINAL_LIST_URL = "https://github.com/maabuu/posebusters/files/14516485/posebusters_pdb_ccd_ids.txt"
METALS = {"LI", "NA", "K", "RB", "CS", "MG", "CA", "SR", "BA", "MN", "FE", "CO", "NI", "CU", "ZN", "CD", "HG"}


def _download(url, path):
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".download")
    with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as dest:
        while block := response.read(1 << 20):
            dest.write(block)
    partial.replace(path)
    return path


def _md5(path):
    h = hashlib.md5()
    with Path(path).open("rb") as source:
        while block := source.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def load_final_ids(path, *, expected_count=308):
    ids = [s.strip().upper() for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    if len(ids) != expected_count or len(ids) != len(set(ids)):
        raise ValueError(f"expected {expected_count} unique final-benchmark IDs; got {len(ids)}")
    if any(not re.fullmatch(r"[0-9A-Z]{4}_[0-9A-Z]+", key) for key in ids):
        raise ValueError("unexpected final-benchmark identifier")
    return ids


def _members(archive, ids):
    """Resolve documented names, never extract arbitrary ZIP paths."""
    wanted = {x.upper(): {} for x in ids}
    for info in archive.infolist():
        parts = info.filename.replace("\\", "/").split("/")
        if len(parts) < 2 or info.is_dir():
            continue
        key = parts[-2].upper()
        if key not in wanted:
            continue
        for kind, suffix in (("receptor", "_protein.pdb"), ("reference", "_ligands.sdf")):
            if parts[-1].lower() == (key + suffix).lower():
                if kind in wanted[key]:
                    raise ValueError(f"ambiguous archive file for {key}/{kind}")
                wanted[key][kind] = info
    return wanted


def _eligibility(receptor, reference):
    from rdkit import Chem
    from .evaluation import read_molecules, _identity

    text = Path(receptor).read_text(encoding="utf-8", errors="replace")
    metals = sorted({line[76:78].strip().upper() for line in text.splitlines()
                     if line.startswith(("ATOM  ", "HETATM")) and line[76:78].strip().upper() in METALS})
    if metals:
        return "receptor_contains_metal:" + ",".join(metals)
    # LINK records document covalent or coordination edges; exclude conservatively.
    if any(line.startswith("LINK  ") for line in text.splitlines()):
        return "receptor_has_LINK_record"
    try:
        molecules = read_molecules(reference)
        if any(len(Chem.GetMolFrags(m)) != 1 for m in molecules):
            return "disconnected_reference"
        expected = molecules[0]
        if any(not all(_identity(m, expected).values()) for m in molecules):
            return "mixed_reference_identity_or_stereo"
        if not 6 <= expected.GetNumHeavyAtoms() <= 60:
            return "outside_predeclared_6_to_60_heavy_atoms"
        if any(tag == "?" for _, tag in Chem.FindMolChiralCenters(expected, includeUnassigned=True)):
            return "unspecified_reference_stereochemistry"
    except Exception as exc:
        return f"reference_validation:{type(exc).__name__}:{exc}"
    return None


def split_cases(cases, *, n_dev, n_eval, seed=20261101):
    """At most one ligand per PDB; deterministic ordering independent of files/list order."""
    if n_dev < 1 or n_eval < 1:
        raise ValueError("development and evaluation both need at least one complex")
    ordered = sorted(cases, key=lambda c: hashlib.sha256(f"{seed}:{c['complex_id']}".encode()).hexdigest())
    chosen = []
    seen_pdb = set()
    for case in ordered:
        pdb = case["complex_id"].split("_")[0]
        if pdb in seen_pdb:
            continue
        seen_pdb.add(pdb)
        chosen.append(dict(case, split="dev" if len(chosen) < n_dev else "eval"))
        if len(chosen) == n_dev + n_eval:
            return chosen
    raise ValueError(f"need {n_dev+n_eval} distinct eligible PDB entries; found {len(chosen)}")


def prepare_dataset(out, *, archive=None, final_list=None, n_dev=5, n_eval=20,
                    selection_seed=20261101, seeds=(0, 1, 2), samples_per_complex=20):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "pilot_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"frozen manifest already exists: {manifest_path}; use another output directory")
    if not seeds or len(seeds) != len(set(seeds)) or samples_per_complex < 1:
        raise ValueError("unique sampling seeds and a positive candidate count are required")
    archive = Path(archive) if archive else _download(ARCHIVE_URL, out / "sources" / "posebusters_paper_data.zip")
    final_list = Path(final_list) if final_list else _download(FINAL_LIST_URL, out / "sources" / "posebusters_pdb_ccd_ids.txt")
    if _md5(archive) != ARCHIVE_MD5:
        raise ValueError("PoseBusters archive MD5 does not match its Zenodo record")
    ids = load_final_ids(final_list)
    excluded, candidates = [], []
    with zipfile.ZipFile(archive) as zf:
        files = _members(zf, ids)
        for key in sorted(ids):
            if set(files[key]) != {"receptor", "reference"}:
                excluded.append({"complex_id": key, "reason": "missing_archive_inputs"})
                continue
            folder = out / "inputs" / key
            folder.mkdir(parents=True, exist_ok=True)
            paths = {"receptor": folder / "protein.pdb", "reference": folder / "reference.sdf"}
            for kind, path in paths.items():
                path.write_bytes(zf.read(files[key][kind]))
            reason = _eligibility(paths["receptor"], paths["reference"])
            if reason:
                excluded.append({"complex_id": key, "reason": reason})
                continue
            candidates.append({"complex_id": key, **{k: relpath(v, out) for k, v in paths.items()},
                               **{k + "_sha256": sha256_file(v) for k, v in paths.items()}})
    selected = split_cases(candidates, n_dev=n_dev, n_eval=n_eval, seed=selection_seed)
    selection = {c["complex_id"] for c in selected}
    excluded.extend({"complex_id": c["complex_id"], "reason": "eligible_not_selected"} for c in candidates if c["complex_id"] not in selection)
    manifest = {
        "schema_version": 1, "benchmark_id": "posebusters308_pilot", "created_utc": utc_now(),
        "selection_seed": selection_seed, "seeds": list(seeds), "samples_per_complex": samples_per_complex,
        "sources": {"archive_url": ARCHIVE_URL, "archive_md5": ARCHIVE_MD5,
                    "archive_sha256": sha256_file(archive), "final_list_url": FINAL_LIST_URL,
                    "final_list_sha256": sha256_file(final_list), "n_final_cases": len(ids)},
        "eligibility": "single connected ligand, 6-60 heavy atoms, defined tetrahedral stereo, no PDB metals or LINK records",
        "limitations": ["PDB-disjoint is not protein-family-disjoint",
                        "overlap with DiffDock-L training has not been audited; no unseen-target claim",
                        "metal/LINK screening is conservative, not a complete covalency classifier",
                        "receptor preparation removes all HETATM records; cofactor/water effects excluded"],
        "complexes": selected, "excluded": excluded,
    }
    write_json(manifest_path, manifest)
    return manifest_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--archive", type=Path)
    ap.add_argument("--final-list", type=Path)
    ap.add_argument("--dev", type=int, default=5)
    ap.add_argument("--eval", type=int, default=20)
    ap.add_argument("--selection-seed", type=int, default=20261101)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--samples", type=int, default=20)
    args = ap.parse_args(argv)
    print(prepare_dataset(args.out, archive=args.archive, final_list=args.final_list, n_dev=args.dev,
                          n_eval=args.eval, selection_seed=args.selection_seed, seeds=args.seeds,
                          samples_per_complex=args.samples))


if __name__ == "__main__":
    main()
