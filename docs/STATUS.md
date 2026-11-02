# Verified state — 1 November 2026

## Completed locally

- Existing setup, single-complex docking, exported manifests and three notebooks
  retained and repaired; cached downloads are compared by SHA-256, not only size.
- New headless GPU acceptance runner and resumable, fixed-candidate pilot.
- Physical evaluation using RDKit and PoseBusters, with independent graph/stereo
  checks, symmetry-aware RMSD without ligand alignment, and explicit backend errors.
- A/B/C/D selectors, optional GNINA refinement/scoring, failure accounting and
  paired uncertainty grouped by complex rather than by pose or seed.
- HTML inspector with reference overlays, exact scores, physical checks and actual
  reverse-process frames; synthetic-fixture rendering tested in desktop/mobile
  browser sizes and with an embedded local 3Dmol library while offline.
- Literature review and correction note for the historical report.

Full local suite at this checkpoint: **182 passed, 6 skipped** in 190.08 seconds.
The skips need the pinned upstream clone, unavailable locally with shell networking
blocked. The warning concerns the Windows/zmq event loop; notebook tests passed.
Command: `python -m pytest tests -q -p no:cacheprovider`, using the existing
`C:/ddv/Scripts/python.exe` environment and `DIFFDOCK_UPSTREAM_SRC=.missing-upstream`.
After that checkpoint, 21 pilot/dataset/evaluation/summary checks passed. The
documented inspector CLI was also completed and its eight tests passed. These
targeted checks do not turn the skipped upstream integration tests into passes.

The standalone research website was checked at 1280, 390 and 320 pixel widths
with no console errors or page overflow. Translation/rotation controls, the
strict 2-Angstrom boundary, a valid local JSON import, malformed import rejection
and clearing the imported data were exercised. The import used a synthetic QA
fixture, not measured docking results. Local asset and anchor checks pass.

## GPU acceptance — completed 1 November 2026

The fresh T4 runtime completed two runs of ten poses on bundled 1A0Q, seed 0.
Both runs passed manifest/pose/trajectory/export validation; physical evaluation
and the HTML inspectors completed too. This is a functional acceptance test,
not a benchmark result. No pose passed all physical checks on this example.
Independent review confirmed real receptor clashes; the native ligand passes.
See [GPU_ACCEPTANCE.md](GPU_ACCEPTANCE.md) for exact measurements and limitations.

The official PoseBusters archive and final 308-case list were downloaded and
verified. The frozen pilot contains 5 development and 20 evaluation complexes,
3 sampling seeds and 20 poses per complex. Selection precedes docking and the
manifest records exclusions and input hashes.

GNINA v1.3.3 also passed real GPU score-only/minimization checks on the native
ligand and one predicted pose. The binary and 27 libraries are pinned separately;
its 62-file bundle (5,901,132,213 bytes) is verified on Drive. See [GNINA.md](GNINA.md).

## Running / not established

The development pilot started at 22:33:27 UTC in `/content/pilot_dev_gnina_v1_3_3`.
It runs 15 complex/seed cases and keeps failures in the denominator. The first
case produced one unreadable pose and is recorded as failed, without resampling;
the remaining cases continue. No pilot improvement is established yet.

An independent finalizer saves results, logs, summaries and inspectors to Drive,
including raw outputs after an abnormal exit or postprocessing failure. It retries
temporary backup failures without needing an assistant turn. `backup_receipt.json`
and `watch_status.json` distinguish verified backup from scientific success.
See the workspace's `REPRISE_GPU.md`, also stored on Drive.

The initial WSL execution restriction was resolved after the session restart.
A fresh `cv-diffdock` Tesla T4 runtime completed the acceptance job. The
helper stack uses an isolated Python 3.11 environment provisioned by `uv`, because
the Colab system Python 3.13 could not create a seeded venv. The first failed
attempt and its source package are retained. Attempt 02 installed the pinned
Python 3.9 / Torch 1.13 upstream stack, downloaded model weights, computed the
lookup tables, passed the CUDA kernel checks and completed both docking runs.

Attempt 02 source bundle SHA-256:
`8b8cc73b7c53d9f76577a99b2a1da03f213aeddb81e476098fd6100c378e2b57`.
Project: `/content/diffdock_project_attempt02`; state and logs:
`/content/diffdock_job_attempt02`; acceptance outputs: `/content/phase1_out`.
Both GPU runtimes are connected to Drive. Sources, frozen dataset, run artifacts,
evaluations and inspectors have been copied with SHA-256 verification to
`MyDrive/Projets-CV-MVA/DiffDock/session-20261101-attempt02`. Model and table backup
is complete under `DiffDock/cache`: 14 files / 3,338,942,586 bytes, including
DiffDock and ESM weights and lookup tables. The receipt lists every SHA-256.
The local acceptance archive is also verified:
`5bcefdcd299300422804e21dfe6160dce006e38c29dfd628624d486054439352` (3,214,689 bytes).

## Next experiment

Use the frozen dataset and verified environment described in [COLAB.md](COLAB.md).
Start with development; execute the unchanged held-out protocol afterwards.
Generate `summary.json` and `summary.md`, retain full inputs/poses/logs and package
hashes, and only then add measured findings to the revision report.

The historical course PDFs remain historical; `report/revision.md` supplies the
corrected interpretation. No commits or GitHub publication were performed in this
revision session.
