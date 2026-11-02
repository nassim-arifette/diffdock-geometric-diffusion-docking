"""Prepare *only the PDBBind complexes an experiment needs* (protein sequences + ESM2 embeddings).

The old notebook embedded all ~19k PDBBind complexes (24 926 sequences, ~1 h) before a single example could run.
Here the 2 GB archive is only unpacked for the requested ids, and FASTA / embeddings are built for those ids alone.
Single-complex docking (``02_docking_colab.ipynb``) needs none of this: inference embeds its own receptor.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ._util import run_streaming, sha256_file
from .setup_env import Layout, _download, torch_env

PDBBIND_ZIP_URL = "https://zenodo.org/records/6408497/files/PDBBind.zip?download=1"
PDBBIND_DIR_IN_ZIP = "PDBBind_processed"
PACKAGE_DIR = Path(__file__).resolve().parent
ESM_SCRIPT = PACKAGE_DIR / "scripts" / "esm_embeddings.py"
PDB_ID_RE = re.compile(r"^[0-9][a-z0-9]{3}$")


class BenchmarkPrepError(RuntimeError):
    pass


def subset_name(ids: Sequence[str]) -> str:
    """``pdbbind_subset_<n>_<hash>``: stable for one id set, different for any other set (even of the same size)."""
    digest = hashlib.sha256(",".join(sorted(i.lower() for i in ids)).encode()).hexdigest()[:8]
    return f"pdbbind_subset_{len(ids)}_{digest}"


def read_split_ids(split_file: Path | str, limit: Optional[int] = None) -> List[str]:
    """Read PDB ids (one per line) from an upstream split file such as ``data/splits/timesplit_test``."""
    ids = [l.strip().lower() for l in Path(split_file).read_text().splitlines() if l.strip()]
    bad = [i for i in ids if not PDB_ID_RE.match(i)]
    if bad:
        raise BenchmarkPrepError(f"not PDB ids: {bad[:5]}")
    seen, out = set(), []
    for i in ids:  # keep order, drop duplicates
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out[:limit] if limit else out


def download_pdbbind_zip(dest: Path | str) -> Path:
    dest = Path(dest)
    if not dest.exists():
        _download(PDBBIND_ZIP_URL, dest)
    if not zipfile.is_zipfile(dest):
        dest.unlink()
        raise BenchmarkPrepError("downloaded PDBBind archive is not a valid zip; removed, please retry")
    return dest


def extract_subset(zip_path: Path | str, ids: Sequence[str], dest_dir: Path | str) -> Dict[str, List[str]]:
    """Extract ``PDBBind_processed/<id>/*`` for the requested ids only. Returns {id: [files]}."""
    dest_dir = Path(dest_dir)
    wanted = {i.lower() for i in ids}
    members: Dict[str, List[zipfile.ZipInfo]] = {i: [] for i in wanted}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            parts = Path(info.filename).parts
            if len(parts) >= 3 and parts[0] == PDBBIND_DIR_IN_ZIP and parts[1].lower() in wanted and not info.is_dir():
                members[parts[1].lower()].append(info)
        missing = sorted(i for i, m in members.items() if not m)
        if missing:
            raise BenchmarkPrepError(f"ids not found in the archive: {missing}")
        extracted: Dict[str, List[str]] = {}
        for pid, infos in members.items():
            out = dest_dir / pid
            out.mkdir(parents=True, exist_ok=True)
            extracted[pid] = []
            for info in infos:
                target = out / Path(info.filename).name  # flat per-complex directory, as upstream expects
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted[pid].append(target.name)
    # upstream reads <id>_protein_processed.pdb (or <id>_protein.pdb) and <id>_ligand.sdf
    problems = [
        pid for pid, files in extracted.items()
        if not ({f"{pid}_protein_processed.pdb", f"{pid}_protein.pdb"} & set(files)) or f"{pid}_ligand.sdf" not in files
    ]
    if problems:
        raise BenchmarkPrepError(f"incomplete complexes (need protein pdb + ligand sdf): {problems}")
    return extracted


def make_fasta(layout: Layout, data_dir: Path, out_fasta: Path) -> Path:
    """Upstream's own ``datasets/esm_embedding_preparation.py``, pointed at the subset directory."""
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(layout.python), "datasets/esm_embedding_preparation.py",
        "--data_dir", str(data_dir), "--out_file", str(out_fasta), "--dataset", "pdbbind",
    ]
    # run as a script, sys.path[0] is datasets/ -- the repo root must be importable for `from datasets.constants import ...`
    env = {**torch_env(layout), "PYTHONPATH": str(layout.repo)}
    rc = run_streaming(cmd, cwd=layout.repo, env=env)
    if rc != 0 or not out_fasta.exists() or out_fasta.stat().st_size == 0:
        raise BenchmarkPrepError("FASTA generation failed")
    return out_fasta


def make_embeddings(layout: Layout, fasta: Path, out_pt: Path, truncation: int = 4096) -> Path:
    cmd = [str(layout.python), str(ESM_SCRIPT), str(fasta), str(out_pt), "--truncation", str(truncation)]
    rc = run_streaming(cmd, cwd=layout.repo, env=torch_env(layout))
    if rc != 0 or not out_pt.exists():
        raise BenchmarkPrepError("ESM embedding generation failed")
    return out_pt


def fasta_ids(fasta: Path) -> List[str]:
    return [l[1:].split()[0] for l in fasta.read_text().splitlines() if l.startswith(">")]


def prepare_subset(
    workdir: Path | str,
    ids: Sequence[str],
    *,
    pdbbind_zip: Optional[Path | str] = None,
    out_dir: Optional[Path | str] = None,
) -> dict:
    """Unpack + embed exactly ``ids``. Returns a summary dict (also written as ``subset_manifest.json``)."""
    layout = Layout(Path(workdir).resolve())
    ids = [i.lower() for i in ids]
    if not ids:
        raise BenchmarkPrepError("no ids given")
    # the directory name identifies the exact id set, so two different subsets of the same size never share (and pollute) a folder
    root = Path(out_dir) if out_dir else layout.workdir / "benchmark" / subset_name(ids)
    data_dir = root / PDBBIND_DIR_IN_ZIP
    if data_dir.is_dir():
        stale = sorted(set(os.listdir(data_dir)) - set(ids))
        if stale:  # upstream's FASTA script embeds every directory it finds, so extra complexes would be silently processed
            raise BenchmarkPrepError(f"{data_dir} already contains other complexes {stale[:5]}; use a fresh out_dir")
    zip_path = Path(pdbbind_zip) if pdbbind_zip else download_pdbbind_zip(layout.workdir / "PDBBind.zip")

    extracted = extract_subset(zip_path, ids, data_dir)
    fasta = make_fasta(layout, data_dir, root / "prepared_for_esm.fasta")
    emb = make_embeddings(layout, fasta, root / "esm2_embeddings.pt")

    expected = fasta_ids(fasta)
    covered = {e.rsplit("_chain_", 1)[0] for e in expected}
    if covered != set(ids):
        raise BenchmarkPrepError(f"FASTA covers {sorted(covered)} but {sorted(ids)} were requested")
    split_file = root / "split_subset.txt"
    split_file.write_text("\n".join(ids) + "\n", encoding="utf-8", newline="\n")

    summary = {
        "ids": ids,
        "n_complexes": len(ids),
        "n_sequences": len(expected),
        "data_dir": str(data_dir),
        "fasta": str(fasta),
        "embeddings": str(emb),
        "embeddings_sha256": sha256_file(emb),
        "split_file": str(split_file),
        "source_archive": PDBBIND_ZIP_URL,
        "files": extracted,
    }
    (root / "subset_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    return summary
