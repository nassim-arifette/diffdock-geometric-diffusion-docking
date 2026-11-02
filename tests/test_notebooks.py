"""The notebooks are code too: they must be clean, compilable, and (against a fake upstream) actually run."""
import json
import os
import re
import sys
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

REPO = Path(__file__).resolve().parents[1]
NB_DOCKING = REPO / "notebooks" / "02_docking_colab.ipynb"
NB_BENCH = REPO / "notebooks" / "03_pdbbind_benchmark_prep.ipynb"
NB_GEOM = REPO / "notebooks" / "01_geometry.ipynb"
STALE = ["/tmp/input_protein_ligand.csv", "data/protein_ligand_example.csv", "PDBBind.zip?download=1 -O", "!pip", "%cd"]


def load(path):
    return nbformat.read(str(path), as_version=4)


@pytest.mark.parametrize("path", [NB_GEOM, NB_DOCKING, NB_BENCH], ids=lambda p: p.name)
def test_notebook_is_valid_clean_and_compiles(path):
    nb = load(path)
    nbformat.validate(nb)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        assert not cell.outputs and cell.execution_count is None, f"{path.name} cell {i} has saved output"
        for bad in STALE:
            assert bad not in cell.source, f"{path.name} cell {i} still contains {bad!r}"
        # IPython shell escapes (`!pip ...`) and magics (`%cd`, `%%time`), not a `% (...)` formatting continuation line
        assert not re.search(r"^\s*(![\w./-]|%{1,2}[A-Za-z])", cell.source, re.M), f"{path.name} cell {i} uses a shell/magic line"
        compile(cell.source, f"{path.name}[{i}]", "exec")


def test_docking_notebook_has_single_input_path():
    """The file that is previewed, run and exported is the one prepare_inputs wrote -- nothing else names a CSV."""
    src = "\n".join(c.source for c in load(NB_DOCKING).cells if c.cell_type == "code")
    assert src.count("prepared.csv_path") >= 2 and "preview_inputs(prepared.csv_path)" in src
    assert re.search(r"run_diffdock\(\s*WORKDIR,\s*prepared,", src)
    assert ".csv" not in re.sub(r"prepared\.csv_path", "", src)


@pytest.fixture
def kernel_env(fake_workdir, monkeypatch):
    monkeypatch.setenv("FAKE_DIFFDOCK_MODE", "ok")
    monkeypatch.setenv("DIFFDOCK_WORKDIR", str(fake_workdir))
    monkeypatch.setenv("DOCKING_TOOLS_PYTHON", sys.executable)
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    return fake_workdir


@pytest.fixture
def fake_drive(kernel_env, tmp_path, monkeypatch):
    """A mounted-looking Drive: <tmp>/mnt/MyDrive exists, notebook writes under MyDrive/DiffDock_colab."""
    mount = tmp_path / "mnt"
    (mount / "MyDrive").mkdir(parents=True)
    root = mount / "MyDrive" / "DiffDock_colab"
    monkeypatch.setenv("DOCKING_DRIVE_MOUNT", str(mount))
    monkeypatch.setenv("DOCKING_DRIVE_ROOT", str(root))
    return root


def execute(nb, cwd=REPO / "notebooks"):
    # Colab-only cells (GPU probe, environment setup) are the ones that touch the network / conda.
    nb.cells = [c for c in nb.cells if "colab-setup" not in c.metadata.get("tags", [])]
    client = nbclient.NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(cwd)}})
    client.execute()
    return nb


def errors(nb):
    return [o for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]


def stream_text(nb):
    return "\n".join(
        "".join(o.get("text", "")) for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "stream"
    )


def test_docking_notebook_runs_end_to_end_on_the_bundled_example(kernel_env):
    nb = execute(load(NB_DOCKING))
    assert not errors(nb)
    runs = sorted((kernel_env / "runs").iterdir())
    assert len(runs) == 1
    manifest = json.loads((runs[0] / "manifest.json").read_text())
    assert manifest["status"] == "ok" and manifest["complexes"][0]["complex_name"] == "1a0q"
    assert len(manifest["complexes"][0]["poses"]) == 10
    assert manifest["complexes"][0]["reference"]["provenance"]
    out = stream_text(nb)
    assert "status: ok   poses: 10" in out and "manifest verified" in out
    archives = list((kernel_env / "exports").glob("*.tar.gz"))
    assert len(archives) == 1 and archives[0].name == f"{manifest['run_id']}.tar.gz"
    # the viewer cell produced a widget, i.e. it ran against the stored poses
    widget_outputs = [
        o for c in nb.cells if c.cell_type == "code" for o in c.outputs
        if o.get("data", {}).get("application/vnd.jupyter.widget-view+json")
    ]
    assert widget_outputs


def test_docking_notebook_uses_what_the_form_says(kernel_env, pdb_file):
    nb = load(NB_DOCKING)
    form = next(c for c in nb.cells if "#@title 2. Inputs" in c.source)
    form.source = (
        form.source.replace('RECEPTOR = ""', f'RECEPTOR = r"{pdb_file}"')
        .replace('LIGAND = ""', 'LIGAND = "CC(=O)Nc1ccc(O)cc1"')
        .replace('CHAINS = ""', 'CHAINS = "A"')
        .replace("SAMPLES = 10", "SAMPLES = 3")
        .replace("SEED = 0", "SEED = 11")
    )
    execute(nb)
    (run_dir,) = list((kernel_env / "runs").iterdir())
    m = json.loads((run_dir / "manifest.json").read_text())
    c = m["complexes"][0]
    assert c["complex_name"] == "toy" and c["receptor"]["chains"] == ["A"]
    assert c["ligand"]["description"] == "CC(=O)Nc1ccc(O)cc1" and c["reference"] is None
    assert len(c["poses"]) == 3 and m["inference_parameters"]["samples_per_complex"] == 3 and m["seed"] == 11
    # exactly what ran: the command line's CSV is the one stored in the manifest, and holds the form's ligand
    csv_in_cmd = Path(m["command"][m["command"].index("--protein_ligand_csv") + 1])
    assert csv_in_cmd == (run_dir / m["input_csv"]["path"]).resolve()
    assert "CC(=O)Nc1ccc(O)cc1" in csv_in_cmd.read_text()


def test_docking_notebook_rejects_half_filled_form(kernel_env):
    nb = load(NB_DOCKING)
    form = next(c for c in nb.cells if "#@title 2. Inputs" in c.source)
    form.source = form.source.replace('LIGAND = ""', 'LIGAND = "CCO"')
    with pytest.raises(Exception, match="both RECEPTOR and LIGAND"):
        execute(nb)


def test_docking_notebook_failure_is_loud_not_silent(kernel_env, monkeypatch):
    monkeypatch.setenv("FAKE_DIFFDOCK_MODE", "silent")  # upstream exits 0 with no poses
    with pytest.raises(Exception, match="docking run failed"):
        execute(load(NB_DOCKING))
    (run_dir,) = list((kernel_env / "runs").iterdir())
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "failed"


def test_notebook_saves_dataset_run_and_archive_to_drive_automatically(kernel_env, fake_drive):
    nb = execute(load(NB_DOCKING))
    assert not errors(nb)
    (run_dir,) = list((kernel_env / "runs").iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    saved = fake_drive / "runs" / run_dir.name
    assert json.loads((saved / "manifest.json").read_text())["run_id"] == manifest["run_id"]
    assert len(list((saved / "predictions" / "1a0q").glob("rank*_confidence*.sdf"))) == 10
    assert (fake_drive / "exports" / f"{manifest['run_id']}.tar.gz").exists()
    assert (fake_drive / "datasets" / "1a0q_inputs" / "input_protein_ligand.csv").exists()
    (entry,) = json.loads((fake_drive / "index.json").read_text())
    assert entry["status"] == "ok" and entry["archive"].endswith(".tar.gz")


def test_notebook_saves_a_failed_run_to_drive_too(kernel_env, fake_drive, monkeypatch):
    monkeypatch.setenv("FAKE_DIFFDOCK_MODE", "crash")
    with pytest.raises(Exception, match="docking run failed"):
        execute(load(NB_DOCKING))
    (run_dir,) = list((kernel_env / "runs").iterdir())
    saved = fake_drive / "runs" / run_dir.name
    assert json.loads((saved / "manifest.json").read_text())["status"] == "failed"
    assert (saved / "logs" / "inference.log").read_text()
    assert json.loads((fake_drive / "index.json").read_text())[0]["status"] == "failed"
    assert not (fake_drive / "exports").exists()  # nothing to export for a failed run


def test_notebook_still_works_without_drive(kernel_env, monkeypatch, tmp_path):
    monkeypatch.setenv("DOCKING_DRIVE_MOUNT", str(tmp_path / "no-such-mount"))
    nb = execute(load(NB_DOCKING))
    assert not errors(nb) and "Drive not used" in stream_text(nb)


def test_geometry_notebook_runs_and_shows_the_fixed_numbers(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    nb = execute(load(NB_GEOM))
    assert not errors(nb)
    out = stream_text(nb)
    assert "4 torsions" in out  # ibuprofen: iPr-CH2, CH2-aryl, aryl-CH, CH-COOH
    assert "before: 1.4142" in out
    assert "old   : 1.0000" in out and "fixed : 1.4142" in out  # the integer-truncation bug and its fix
    assert "atom 3 (on the bond) moved by the old code: 1." in out and "by the fixed code: 0.000 A" in out
    assert "centre of mass" in out
    widget_outputs = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs
                      if o.get("data", {}).get("application/vnd.jupyter.widget-view+json")]
    assert len(widget_outputs) >= 4  # translation, rotation, torsion, composed update
