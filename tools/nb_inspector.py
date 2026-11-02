"""Build notebooks/04_pose_inspector.ipynb from stored run artifacts."""
import nbformat as nbf

from nbtools import bootstrap, code, finish, md


def build():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# Inspect a stored DiffDock run

Open a run directory containing `manifest.json`. This notebook reads its receptor, predicted poses and optional reference,
checks the recorded file hashes, and exports a portable HTML inspector. It does not run docking again.

If `evaluation.json` is present, its manifest hash and pose identities must match before its measurements are shown.
Without this report, validity and RMSD stay **not evaluated**. The reference overlay is a diagnostic; it is never used to
choose a pose here. The confidence score is not binding affinity.
"""),
        bootstrap(),
        md("""
## Open the artifacts

Use a local run folder, or mount Drive and point to the saved run there. The HTML embeds the molecular data, so it can be
copied without the original folder. By default its 3D renderer needs the 3Dmol.js CDN; supply a local `3Dmol-min.js` file
to embed the renderer as well and view the document entirely offline. WebGL is needed for the 3D panel.
"""),
        code('''
#@title Run to inspect
RUN_DIR = "/content/diffdock_runs/REPLACE_WITH_RUN_DIRECTORY"  #@param {type:"string"}
EVALUATION_JSON = ""  #@param {type:"string"}
THREE_DMOL_JS = ""  #@param {type:"string"}

from pathlib import Path
from docking_tools.inspector import write_inspector

run_dir = Path(RUN_DIR).expanduser()
if not (run_dir / "manifest.json").is_file():
    raise FileNotFoundError(f"Choose a saved run directory containing manifest.json: {run_dir}")
html_path = write_inspector(
    run_dir,
    evaluation=EVALUATION_JSON or None,
    three_dmol_js=THREE_DMOL_JS or None,
)
print("Saved inspector:", html_path)
'''),
        md("""
## Browse poses and recorded trajectories

Use the candidate list or the selection-variant buttons to switch poses. Refined coordinates retain their parent
candidate identity; their confidence is not recomputed. Toggle the reference overlay explicitly.

The trajectory slider uses only frames actually saved by DiffDock. The parser removes stale hydrogens and reports the
known overwritten first-step frame in the upstream layout. A refined pose has no invented diffusion trajectory.
Scores and checks continue to describe the final stored pose while an earlier frame is displayed.
"""),
        code('''
import html
from IPython.display import HTML, display

document = html_path.read_text(encoding="utf-8")
display(HTML('<iframe title="DiffDock pose inspector" style="width:100%;height:1100px;border:1px solid #ddd" '
             'srcdoc="' + html.escape(document, quote=True) + '"></iframe>'))
'''),
        md("""
## Keep or share the inspection document

Open `inspector.html` in a browser to use the full page. The document contains the molecular structures and evaluation
values you selected. The file below is the same document displayed above; no inference is launched.
"""),
        code('''
DOWNLOAD_HTML = False  #@param {type:"boolean"}
if DOWNLOAD_HTML:
    if IN_COLAB:
        from google.colab import files
        files.download(str(html_path))
    else:
        print("Open or copy:", html_path.resolve())
'''),
    ]
    return finish(nb, "04_pose_inspector.ipynb")
