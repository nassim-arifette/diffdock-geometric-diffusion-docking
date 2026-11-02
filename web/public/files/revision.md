# Revision of the DiffDock course project

1 November 2026. This note accompanies the preserved original PDFs. It supersedes
the interpretations below; it does not retroactively turn exploratory outputs
into a reproducible docking benchmark.

## Scientific question

The original project studied geometric diffusion over ligand translation,
rotation and torsion. The revised experiment asks how much of a fixed candidate
pool's available accuracy is lost at selection, and whether physical checks,
independent scoring or local refinement recover it at an acceptable cost.

This is an established line of work, not a new docking algorithm. The detailed
[literature review](../docs/research.md) cites DiffDock, DiffDock-L, PoseBusters,
GNINA, DiffDock-Glide, Matcha and RLDiff and distinguishes their settings.

## Corrections to the historical report

- **6AGT** is a Plasmodium falciparum lysyl-tRNA synthetase complex. Its ligand
  9X0 differs from the 4-methoxybenzonitrile used in the exploratory notebook.
  That example cannot measure redocking RMSD to 9X0. See [RCSB 6AGT](https://www.rcsb.org/structure/6AGT).
- **Confidence is not affinity or energy.** A correlation between DiffDock
  confidence and a GNINA score compares two computed quantities. It is not
  validation of binding affinity or biological activity. Archived correlations
  without their raw vectors are not reused as measured results here.
- **Pose validity is not synthesizability.** Identity, stereochemistry, strain,
  geometry and clashes concern a pose's plausibility; they do not establish
  chemical synthesis feasibility. See [PoseBusters](https://doi.org/10.1039/D3SC04185A).
- **Protein representation differs between models.** The original score model
  uses a coarse receptor representation; the confidence model uses atom-level
  information. The limitation is not uniformly “only C-alpha everywhere”.
- **RMSD must preserve the receptor frame.** Aligning each ligand onto its
  reference would hide a misplaced pose. Rounded equal scores also do not
  demonstrate equal structures or identical sampling runs.

## Protocol and evidence

The public pilot freezes a subset of the final 308-case PoseBusters benchmark,
with disjoint development/evaluation PDB entries, input hashes, exclusions and
three sampling seeds. All selectors use the same 20 raw poses per complex/seed.
The ligand's chemical identity is provided as SMILES, with reference coordinates
held back for evaluation. A/B compare confidence with validity-filtered confidence;
C adds GNINA scoring; D independently refines each candidate and revalidates it.

Success means heavy-atom RMSD strictly below 2 Angstrom AND physical validity.
Failed runs and empty selections remain failures; variants not executed remain
unmeasured. Local tests cover reference-independent selection, symmetry and frame
handling, backend failures, and preservation of candidate pools across retries.

## Results available at this revision

The engineering tests and inspector are validated locally. **The real Colab
acceptance run and the new scientific pilot have not yet been completed in this
revision.** There is therefore no supported claim that B, C or D improves A.
The terminal environment refused WSL and external sockets before a runtime could
be inspected or allocated. See [STATUS.md](../docs/STATUS.md) for exact checks and
remaining work. The notebooks and CLI produce the required artifacts when run.

## Limits that remain even after a successful pilot

A small PDB-disjoint subset does not establish generalization to unseen protein
families or exclusion of DiffDock-L/GNINA training overlap. The prepared receptor
removes waters, ions and other HETATM records, while the pilot excludes structures
with metals or LINK records; validity therefore concerns this simplified receptor.
Receptor flexibility, protonation uncertainty and specialized covalent/metal
chemistry remain outside scope. Multiple seeds do not multiply the number of
independent complexes; paired uncertainty groups seeds by complex.
