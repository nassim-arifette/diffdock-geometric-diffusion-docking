"""ESM2 per-residue embeddings for a FASTA file, in the format DiffDock's evaluation expects.

Run INSIDE the DiffDock environment (torch + fair-esm)::

    python esm_embeddings.py prepared_for_esm.fasta esm2_embeddings.pt [--truncation 4096]

Output: a ``{fasta_id: tensor[L, 1280]}`` dict saved with ``torch.save`` -- the same object upstream builds with
``scripts/extract.py`` + ``datasets/esm_embeddings_to_pt.py``, minus the intermediate one-file-per-sequence
directory and without needing a clone of facebookresearch/esm (the pip package ships no ``extract`` script).
Mirrors upstream ``utils/inference_utils.compute_ESM_embeddings`` (layer 33, ``per_tok``) but with the
4096-residue truncation used in the README's benchmark recipe instead of 1022.
"""
import argparse
import json
import sys

import torch
from esm import FastaBatchedDataset, pretrained

MODEL = "esm2_t33_650M_UR50D"
LAYER = 33


def read_fasta(path):
    labels, seqs, cur = [], [], None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                labels.append(line[1:].split()[0])
                seqs.append("")
                cur = len(seqs) - 1
            else:
                seqs[cur] += line
    return labels, seqs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("fasta")
    ap.add_argument("out")
    ap.add_argument("--truncation", type=int, default=4096)
    ap.add_argument("--toks_per_batch", type=int, default=4096)
    args = ap.parse_args(argv)

    labels, seqs = read_fasta(args.fasta)
    if not labels:
        sys.exit(f"{args.fasta} contains no sequences")
    model, alphabet = pretrained.load_model_and_alphabet(MODEL)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    dataset = FastaBatchedDataset(labels, seqs)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(args.truncation), batch_sampler=batches
    )
    out = {}
    with torch.no_grad():
        for i, (b_labels, b_strs, toks) in enumerate(loader):
            print(f"batch {i + 1}/{len(batches)} ({toks.size(0)} sequences)", flush=True)
            reps = model(toks.to(device), repr_layers=[LAYER], return_contacts=False)["representations"][LAYER].cpu()
            for j, label in enumerate(b_labels):
                n = min(args.truncation, len(b_strs[j]))
                out[label] = reps[j, 1 : n + 1].clone()

    # same completeness checks the prepared data must satisfy
    problems = []
    for label, seq in zip(labels, seqs):
        t = out.get(label)
        if t is None:
            problems.append(f"{label}: missing")
        elif t.shape != (min(args.truncation, len(seq)), 1280):
            problems.append(f"{label}: shape {tuple(t.shape)} for sequence length {len(seq)}")
        elif not torch.isfinite(t).all():
            problems.append(f"{label}: non-finite values")
    if problems:
        sys.exit("embedding check failed:\n  " + "\n  ".join(problems))
    torch.save(out, args.out)
    print(json.dumps({"n_sequences": len(out), "device": device, "out": args.out}))


if __name__ == "__main__":
    main()
