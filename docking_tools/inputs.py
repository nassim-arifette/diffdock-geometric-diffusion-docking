"""Turn what a user typed in the notebook form into one validated DiffDock input manifest.

The CSV schema is the one read by the pinned upstream ``inference.py``::

    complex_name,protein_path,ligand_description,protein_sequence

``ligand_description`` is a SMILES string or a path to a molecule file; ``protein_path`` is a PDB file.
"""
from __future__ import annotations

import csv
import re
import shutil
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ._util import sha256_file

CSV_COLUMNS = ["complex_name", "protein_path", "ligand_description", "protein_sequence"]
PDB_ID_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
# exactly the formats upstream's read_molecule() understands (datasets/process_mols.py); a .mol file is NOT among them
LIGAND_FILE_SUFFIXES = (".sdf", ".mol2", ".pdbqt", ".pdb")
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
# PubChem's `SMILES` property is the isomeric one (keeps stereo); the older `CanonicalSMILES` drops stereochemistry
PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/SMILES/TXT"
PUBCHEM_NAME_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/SMILES/TXT"

# strings pandas turns into NaN when reading a CSV (upstream reads our file with pd.read_csv)
_PANDAS_NA = {
    "", "#n/a", "#n/a n/a", "#na", "-1.#ind", "-1.#qnan", "-nan", "1.#ind", "1.#qnan", "<na>", "n/a", "na",
    "nan", "null", "none",
}
COMPLEX_NAME_RE = re.compile(r"[A-Za-z0-9_.\-]+")


class InputError(ValueError):
    """The user supplied something we cannot dock."""


def is_pandas_unsafe_name(name: str) -> bool:
    """True if ``pd.read_csv`` would not give this complex name back as the same string.

    ``1e20`` becomes the float 1e+20, ``123`` the integer 123, ``inf`` infinity, and ``na`` / ``null`` NaN; upstream then
    crashes on ``complex_names[i] + '_chain_'`` or silently renames the complex ``complex_0``.
    """
    if name.strip().lower() in _PANDAS_NA:
        return True
    try:
        float(name)
    except ValueError:
        return False
    return True


def safe_complex_name(name: str) -> str:
    """A filesystem-safe complex name that survives a pandas round trip (prefixed with ``c_`` when needed)."""
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip()).strip("-")
    if not clean or is_pandas_unsafe_name(clean):
        clean = f"c_{clean}" if clean else "complex"
    return clean


def is_existing_file(spec: str) -> bool:
    """``Path(spec).is_file()`` that is False (not an exception) for strings that cannot be paths, e.g. a long SMILES."""
    if not spec or "\n" in spec or len(spec) > 4096:
        return False
    try:
        return Path(spec).expanduser().is_file()
    except (OSError, ValueError):  # ENAMETOOLONG, embedded NUL, ...
        return False


@dataclass
class ReceptorInfo:
    path: Path
    source: str  # "rcsb:6AGT" or "file:/..."
    chains: List[str]
    n_residues: int
    removed_hetero: Dict[str, int] = field(default_factory=dict)  # residue name -> atom count dropped
    removed_waters: int = 0


@dataclass
class LigandInfo:
    description: str  # SMILES or absolute path as written to the CSV
    kind: str  # "smiles" | "file"
    source: str  # "smiles" | "pubchem:CID" | "file:/..."
    canonical_smiles: Optional[str] = None
    n_heavy_atoms: Optional[int] = None
    unspecified_stereocenters: Optional[int] = None  # chiral centres the SMILES leaves undefined (docked as an arbitrary choice)


def _fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "docking_tools/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_pdb(pdb_id: str, dest_dir: Path) -> Path:
    """Download the raw RCSB entry (not cached under a shared name: the caller owns ``dest_dir``)."""
    pdb_id = pdb_id.strip()
    if not PDB_ID_RE.match(pdb_id):
        raise InputError(f"{pdb_id!r} is not a 4-character PDB id")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{pdb_id.upper()}.pdb"
    try:
        data = _fetch(RCSB_URL.format(pdb_id=pdb_id.upper()))
    except Exception as e:  # noqa: BLE001
        raise InputError(f"could not download PDB {pdb_id} from RCSB: {e!r}") from e
    if not any(line.startswith(("ATOM", "HETATM")) for line in data.decode("utf-8", "replace").splitlines()):
        raise InputError(f"RCSB returned no atom records for {pdb_id}")
    dest.write_bytes(data)
    return dest


def clean_receptor(src: Path, dest: Path, *, chains: Optional[Sequence[str]] = None) -> ReceptorInfo:
    """Write a protein-only PDB: first model, ATOM records, altloc A/blank, optional chain filter.

    Waters, ions, cofactors and the co-crystallised ligand are *dropped* and reported, so the file used for
    docking, scoring and viewing never contains the native ligand. Modified residues stored as HETATM (e.g.
    SEP, TPO) are also dropped; they appear in ``removed_hetero``.
    """
    wanted = set(chains) if chains else None
    kept: List[str] = []
    removed: Dict[str, int] = {}
    waters = 0
    residues = set()
    seen_chains: List[str] = []
    model_no = 1
    for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
        rec = line[:6].strip()
        if rec == "MODEL":
            model_no = int(line[10:14].strip() or 1)
            continue
        if rec not in ("ATOM", "HETATM") or model_no != 1:
            continue
        line = line.ljust(80)
        resname = line[17:20].strip()
        chain = line[21:22].strip() or "A"
        if wanted is not None and chain not in wanted:
            continue
        if rec == "HETATM":
            if resname in ("HOH", "WAT", "DOD"):
                waters += 1
                continue
            if resname != "MSE":
                removed[resname] = removed.get(resname, 0) + 1
                continue
            # selenomethionine is an ordinary methionine for DiffDock
            line = "ATOM  " + line[6:17] + "MET" + line[20:]
            if line[12:16].strip() == "SE":
                line = line[:12] + " SD " + line[16:76] + " S" + line[78:]
        altloc = line[16:17]
        if altloc not in (" ", "A"):
            continue
        kept.append((line[:16] + " " + line[17:]).rstrip())
        residues.add((chain, line[22:27]))
        if chain not in seen_chains:
            seen_chains.append(chain)
    if not any(l.startswith("ATOM") for l in kept):
        raise InputError(f"no protein ATOM records left in {src} (chains={sorted(wanted) if wanted else 'all'})")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(kept) + "\nEND\n", encoding="utf-8", newline="\n")
    return ReceptorInfo(
        path=dest, source=f"file:{src}", chains=seen_chains, n_residues=len(residues),
        removed_hetero=removed, removed_waters=waters,
    )


def resolve_receptor(
    spec: str, work_dir: Path, *, chains: Optional[Sequence[str]] = None, clean: bool = True
) -> ReceptorInfo:
    """``spec`` is a 4-character PDB id or a path to a .pdb file."""
    spec = spec.strip()
    if not spec:
        raise InputError("empty receptor")
    p = Path(spec).expanduser()
    if is_existing_file(spec):
        raw, source = p, f"file:{p.resolve()}"
        label = p.stem
    elif PDB_ID_RE.match(spec):
        raw, source = download_pdb(spec, work_dir / "raw"), f"rcsb:{spec.upper()}"
        label = spec.upper()
    else:
        raise InputError(f"receptor {spec!r} is neither an existing file nor a PDB id")
    dest = work_dir / f"{label}_protein.pdb"
    if clean:
        info = clean_receptor(raw, dest, chains=chains)
    else:
        shutil.copyfile(raw, dest)
        info = ReceptorInfo(path=dest, source=source, chains=[], n_residues=0)
    info.source = source
    return info


def pubchem_smiles(cid_or_name: str) -> str:
    cid_or_name = cid_or_name.strip()
    if cid_or_name.isdigit():
        url = PUBCHEM_URL.format(cid=cid_or_name)
    else:
        url = PUBCHEM_NAME_URL.format(name=urllib.parse.quote(cid_or_name))
    try:
        smi = _fetch(url).decode().strip().splitlines()[0].strip()
    except Exception as e:  # noqa: BLE001
        raise InputError(f"PubChem lookup failed for {cid_or_name!r}: {e!r}") from e
    if not smi:
        raise InputError(f"PubChem returned no SMILES for {cid_or_name!r}")
    return smi


def check_smiles(smiles: str) -> dict:
    """Parse with RDKit; report canonical SMILES, heavy-atom count and undefined stereocentres. Raises InputError on bad SMILES."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InputError(f"RDKit cannot parse SMILES {smiles!r}")
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    return {
        "canonical_smiles": Chem.MolToSmiles(mol),
        "n_heavy_atoms": mol.GetNumHeavyAtoms(),
        "unspecified_stereocenters": sum(1 for _, label in centers if label == "?"),
    }


def read_ligand_file(path: Path):
    """Read one input molecule using the same file families as pinned DiffDock.

    The identity is recorded for later physical checks; an accepted path alone
    is not enough to describe which molecule the experiment actually docked.
    """
    from rdkit import Chem

    suffix = path.suffix.lower()
    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=True)
        if len(supplier) != 1:
            raise InputError("ligand input SDF must contain exactly one molecule")
        molecule = supplier[0]
    elif suffix == ".mol2":
        molecule = Chem.MolFromMol2File(str(path), sanitize=True, removeHs=True)
    elif suffix == ".pdb":
        molecule = Chem.MolFromPDBFile(str(path), sanitize=True, removeHs=True)
    elif suffix == ".pdbqt":
        block = "\n".join(line[:66] for line in path.read_text().splitlines())
        molecule = Chem.MolFromPDBBlock(block, sanitize=True, removeHs=True)
    else:
        raise InputError(f"unsupported ligand file suffix: {suffix}")
    if molecule is None or not molecule.GetNumHeavyAtoms():
        raise InputError(f"cannot read a nonempty ligand from {path}")
    return molecule


def resolve_ligand(spec: str) -> LigandInfo:
    """``spec`` is a SMILES string, a PubChem CID, or a path to a molecule file (.sdf, .mol2, .pdbqt, .pdb)."""
    spec = spec.strip()
    if not spec:
        raise InputError("empty ligand")
    p = Path(spec).expanduser()
    if is_existing_file(spec) and p.suffix.lower() in LIGAND_FILE_SUFFIXES:
        from rdkit import Chem

        molecule = read_ligand_file(p)
        details = check_smiles(Chem.MolToSmiles(molecule, isomericSmiles=True))
        return LigandInfo(description=str(p.resolve()), kind="file", source=f"file:{p.resolve()}", **details)
    if is_existing_file(spec):
        raise InputError(
            f"ligand file {spec!r}: DiffDock reads only {', '.join(LIGAND_FILE_SUFFIXES)} files (convert a .mol file to .sdf)"
        )
    if spec.lower().endswith(LIGAND_FILE_SUFFIXES):
        raise InputError(f"ligand file {spec!r} does not exist")
    if spec.isdigit():
        smiles, source = pubchem_smiles(spec), f"pubchem:{spec}"
    else:
        smiles, source = spec, "smiles"
    chk = check_smiles(smiles)
    return LigandInfo(description=smiles, kind="smiles", source=source, **chk)


@dataclass
class InputRow:
    complex_name: str
    protein_path: str
    ligand_description: str
    protein_sequence: str = ""


def write_input_csv(path: Path, rows: Sequence[InputRow]) -> Path:
    """Write the manifest in upstream's exact schema and re-validate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(CSV_COLUMNS)
        for r in rows:
            w.writerow([r.complex_name, r.protein_path, r.ligand_description, r.protein_sequence])
    validate_input_csv(path)
    return path


def read_input_csv(path: Path) -> List[InputRow]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CSV_COLUMNS:
            raise InputError(f"{path}: columns {reader.fieldnames} != expected {CSV_COLUMNS}")
        return [InputRow(**{k: (row[k] or "") for k in CSV_COLUMNS}) for row in reader]


def validate_input_csv(path: Path) -> List[InputRow]:
    """Check the schema and that every referenced file exists. Returns the parsed rows.

    Relative paths are resolved against the CSV's own folder, which is how a *portable* CSV is meant to be read.
    """
    base = Path(path).parent

    def exists(p: str) -> bool:
        pp = Path(p)
        return is_existing_file(str(pp if pp.is_absolute() else base / pp))

    rows = read_input_csv(path)
    if not rows:
        raise InputError(f"{path} has no rows")
    names = [r.complex_name for r in rows]
    if len(set(names)) != len(names):
        raise InputError(f"duplicate complex_name in {path}: {names}")
    for r in rows:
        if not COMPLEX_NAME_RE.fullmatch(r.complex_name):
            raise InputError(f"complex_name {r.complex_name!r} may only contain letters, digits, _ . -")
        if is_pandas_unsafe_name(r.complex_name):
            raise InputError(
                f"complex_name {r.complex_name!r} would be read back as a number or NaN by pandas and crash or rename the "
                f"complex inside DiffDock; use e.g. {safe_complex_name(r.complex_name)!r}"
            )
        if not r.protein_path:
            raise InputError(f"{r.complex_name}: empty protein_path (ESMFold folding is not supported here)")
        if not exists(r.protein_path):
            raise InputError(f"{r.complex_name}: receptor file not found: {r.protein_path}")
        if not r.ligand_description:
            raise InputError(f"{r.complex_name}: empty ligand_description")
        if r.ligand_description.lower().endswith(LIGAND_FILE_SUFFIXES) and not exists(r.ligand_description):
            raise InputError(f"{r.complex_name}: ligand file not found: {r.ligand_description}")
    return rows


def portable_rows(rows: Sequence[InputRow], base_dir: Path) -> List[InputRow]:
    """Copy of ``rows`` with every file path made relative to ``base_dir`` (SMILES strings are left alone)."""
    import os

    def rel(p: str) -> str:
        return Path(os.path.relpath(p, base_dir)).as_posix()

    out = []
    for r in rows:
        lig = rel(r.ligand_description) if r.ligand_description.lower().endswith(LIGAND_FILE_SUFFIXES) else r.ligand_description
        out.append(InputRow(r.complex_name, rel(r.protein_path), lig, r.protein_sequence))
    return out


def rebase_csv(portable_csv: Path, base_dir: Path, out_csv: Path) -> Path:
    """Turn a portable CSV (paths relative to a run directory) into an absolute-path CSV valid where ``base_dir`` now is.

    The CSV that DiffDock reads needs absolute paths (it runs with another working directory); the portable one is what
    survives being copied to Drive or to another machine.
    """
    rows = read_input_csv(portable_csv)
    abs_rows = []
    for r in rows:
        prot = r.protein_path if Path(r.protein_path).is_absolute() else str((base_dir / r.protein_path).resolve())
        lig = r.ligand_description
        if lig.lower().endswith(LIGAND_FILE_SUFFIXES) and not Path(lig).is_absolute():
            lig = str((base_dir / lig).resolve())
        abs_rows.append(InputRow(r.complex_name, prot, lig, r.protein_sequence))
    return write_input_csv(out_csv, abs_rows)


def describe_file(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
