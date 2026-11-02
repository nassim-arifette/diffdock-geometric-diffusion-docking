# Research website

A single English page in an interactive-article style. Sources live in `web/` (Vite + TypeScript, 3Dmol.js, D3); `npm run build` writes `site/`, which is not tracked.

## Data

All molecular content comes from one archived DiffDock-L GPU acceptance run on PDB 1A0Q (`colab_artifacts/acceptance_20261101/`, not tracked). `tools/build_web_data.py` exports it to `web/public/data/case.json`:

- the prepared receptor, the crystal ligand, its heavy-atom bonds and DiffDock torsions;
- all ten candidates with confidence, PoseBusters checks, the recorded reverse-diffusion frames (parsed with `docking_tools.frames.trajectory`);
- per-frame symmetry-aware RMSD to the crystal pose (`docking_tools.evaluation.receptor_frame_rmsd`) and closest receptor heavy atom.

The export refuses to write unless the final-frame RMSD it recomputes equals the RMSD in the run's `evaluation.json` and the final frame equals the saved pose. `case.json` is committed, so the Pages build needs neither RDKit nor the Colab artifacts.

```bash
python -m pip install -r requirements-tools.txt
python tools/build_web_data.py
cd web
npm ci
npm test        # browser geometry: torsions keep bond lengths, frames end on the saved poses
npm run build   # -> ../site
cd ..
python tools/check_site.py
```

The pilot figure states the protocol only. Pilot results will be added with their manifests, raw poses and logs once the frozen runs finish.

## Local preview

`cd web && npm run dev`, or `python -m http.server 8000 --directory site` after a build. The page loads `case.json` with `fetch`, so it must be served over HTTP.

## Publishing

`.github/workflows/pages.yml` installs, tests, builds, checks and deploys `site/`. In the repository settings choose **Pages → Source → GitHub Actions**.
