# GPU acceptance — 1 November 2026

The complete pipeline ran successfully on a fresh Tesla T4: setup and CUDA
checks, two docking runs, pose/score/trajectory validation, physical evaluation,
archive export and HTML inspection. This establishes functional execution. It
does **not** establish docking accuracy or an improvement over the original method.

## Protocol and provenance

- One bundled complex, **1A0Q**, with 10 candidates per run and seed **0** for
  both runs. These are two repetitions of one input, not independent statistical
  replicates or a 20-complex benchmark.
- The acceptance runner used the bundled receptor and ligand SDF. That ligand
  SDF also supplied the native reference for evaluation. The separate pilot uses
  an isomeric SMILES as the docking input and reserves native coordinates for
  evaluation; do not infer the pilot's input policy from this smoke test.
- Upstream commit:
  [`85c49b60d3e0b0182a59ee43a34a6d7036981284`](https://github.com/gcorso/DiffDock/tree/85c49b60d3e0b0182a59ee43a34a6d7036981284),
  with the recorded seed/full-precision-score patch and v1.1 model release.
  Configuration: 20 nominal inference steps, 19 actual steps, saved trajectories.
- Inference: Python 3.9.18, Torch 1.13.1+cu117, T4 with 15,360 MiB VRAM.
  Evaluation: separate Python 3.11 environment, RDKit 2026.9.1 and
  PoseBusters 0.6.5, `dock` configuration plus input connectivity, charge and
  specified stereochemistry checks.
- RMSD: heavy atoms, graph-symmetry-aware, in the receptor frame without ligand
  alignment; success requires **RMSD < 2 Å**. Native coordinates do not enter
  physical filtering. GNINA was not used in these two acceptance evaluations.

The source bundle and each run manifest were verified by SHA-256. The manifests
also record hashes of inputs, configurations, model weights, scores and poses.

| Artifact | SHA-256 |
| --- | --- |
| Source bundle, attempt 02 | `8b8cc73b7c53d9f76577a99b2a1da03f213aeddb81e476098fd6100c378e2b57` |
| `20261101-222233_1a0q_smoke_1/manifest.json` | `1021cde43612178d41b2ce9bdf0f42946c0b5effa3cc712965d1e320a5070bf6` |
| `20261101-222434_1a0q_smoke_2/manifest.json` | `2f69afef12b1189f38a37d27aeef72738b366ee8feab12e0f2b5d9a7c8e9fe79` |
| PoseBusters configuration | `4f9a50ee208749b799998834b3104e5e50ab0d0633a038e7025183fc7daea9e7` |

Both evaluations reference the corresponding verified manifest hash. Repeating
seed 0 did not produce identical confidence arrays or pose bytes. The runner
records this diagnostic explicitly; bitwise GPU determinism was not established.

## Observed physical failures

All 20 candidates passed input connectivity/charge and stereochemistry checks.
Each failed exactly `minimum_distance_to_protein` and
`volume_overlap_with_protein`, with no backend error. The native ligand passed
those receptor checks. An independent Euclidean heavy-atom distance calculation
agreed with the PoseBusters minima to within 1e-8 Å.

| Quantity | Native ligand | Run 1 top-ranked pose | Run 2 top-ranked pose |
| --- | ---: | ---: | ---: |
| Minimum protein distance (Å) | 2.695 | 0.741 | 0.778 |
| Protein–ligand clashes | 0 | 19 | 19 |
| Protein volume overlap | 1.22% | 18.17% | 18.62% |
| Receptor-frame RMSD (Å) | — | 5.217 | 5.218 |

The recorded volume-overlap cutoff is 7.5%. The shortest contacts in the two
predictions involve ARG96/NH2. These measurements support actual steric clashes,
rather than a validator that rejects every input.

Across all candidates, RMSD ranged from 3.739 to 5.217 Å in run 1 and from 3.745
to 5.218 Å in run 2. No candidate met the 2 Å threshold; filtering or reranking
alone cannot turn these fixed pools into successful predictions. Refinement was
not evaluated here. This failure mode is consistent with the motivation of the
[PoseBusters study](https://pubs.rsc.org/en-us/content/articlehtml/2024/sc/d3sc04185a),
but one complex cannot reproduce its method comparison or support a population
accuracy estimate.

The next scientific evidence comes from the separately frozen development and
evaluation pilot described in [COLAB.md](COLAB.md), with failures retained in
the denominator. No A/B/C/D improvement is claimed from this acceptance test.
