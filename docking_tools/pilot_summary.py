"""Summarize paired selectors, keeping failures and repeated seeds in their proper units."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ._util import read_json, sha256_file, write_json
from .manifest import load_manifest, verify_manifest

METRICS = ("top1_rmsd_success", "top1_valid", "top1_joint_success", "top5_joint_success", "empty_selection")
VARIANTS = ("A_confidence", "B_valid_confidence", "C_valid_gnina", "D_refined_gnina")


def aggregate_rows(rows, *, bootstrap_samples=2000, bootstrap_seed=20261101):
    if not rows:
        raise ValueError("no benchmark observations")
    identities = [(r["complex_id"], r["seed"], r["variant"]) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate complex/seed/variant observation")
    enabled = [v for v in VARIANTS if any(r["variant"] == v for r in rows)]
    by_variant = {v: [r for r in rows if r["variant"] == v] for v in enabled}
    expected = {(r["complex_id"], r["seed"]) for r in by_variant[enabled[0]]}
    if any({(r["complex_id"], r["seed"]) for r in rs} != expected for rs in by_variant.values()):
        raise ValueError("selectors were not evaluated on identical complex/seed pairs")
    tables = {}
    per_complex = {}
    for variant, observations in by_variant.items():
        n = len(observations)
        tables[variant] = {"n_complex_seed_pairs": n, "n_completed_evaluations": sum(r["evaluated"] for r in observations)}
        for key in METRICS:
            count = sum(r[key] for r in observations)
            tables[variant][key] = {"count": count, "denominator": n, "rate": count / n}
        groups = defaultdict(list)
        for row in observations:
            groups[row["complex_id"]].append(float(row["top1_joint_success"]))
        per_complex[variant] = {k: float(np.mean(v)) for k, v in groups.items()}
    baseline = per_complex["A_confidence"]
    keys = sorted(baseline)
    rng = np.random.default_rng(bootstrap_seed)
    sampled = rng.integers(0, len(keys), size=(bootstrap_samples, len(keys))) if len(keys) > 1 else None
    contrasts = {}
    for variant in enabled[1:]:
        delta = np.array([per_complex[variant][key] - baseline[key] for key in keys])
        interval = None if sampled is None else np.quantile(delta[sampled].mean(axis=1), [0.025, 0.975]).tolist()
        contrasts[variant + "_minus_A"] = {"delta_joint_success": float(delta.mean()),
                                             "paired_complex_bootstrap_95pct": interval}
    return {"n_complexes": len(keys), "n_complex_seed_pairs": len(expected), "variants": tables,
            "paired_contrasts": contrasts,
            "uncertainty": {"unit": "complex; all its seeds stay together", "bootstrap_samples": bootstrap_samples,
                            "seed": bootstrap_seed, "note": "exploratory pilot; PDB disjoint does not imply independent protein families"}}


def summarize_pilot(out):
    out = Path(out)
    state = read_json(out / "pilot_state.json")
    protocol = read_json(out / "protocol.json")
    if sha256_file(out / "protocol.json") != state["config"]["protocol_sha256"]:
        raise ValueError("frozen protocol hash changed")
    complexes = [c for c in protocol["complexes"] if state["config"].get("split") in (None, c["split"])]
    enabled = VARIANTS if state["config"].get("gnina") else VARIANTS[:2]
    rows, failures, costs = [], [], []
    for case in complexes:
        for seed in protocol["seeds"]:
            case_id = f"{case['complex_id']}_seed{seed}"
            entry = state["cases"].get(case_id)
            if not entry or entry["status"] == "running":
                raise ValueError(f"pilot is incomplete: {case_id}; do not report a partial denominator")
            evaluation = None
            if entry["status"] == "ok":
                run_dir = out / entry["run_dir"]
                manifest = load_manifest(run_dir)
                verify_manifest(run_dir, manifest)
                if sha256_file(run_dir / "manifest.json") != entry["manifest_sha256"]:
                    raise ValueError(f"changed manifest: {case_id}")
                evaluation_path = run_dir / "evaluation.json"
                if sha256_file(evaluation_path) != entry["evaluation_sha256"]:
                    raise ValueError(f"changed evaluation: {case_id}")
                evaluation = read_json(evaluation_path)
                if evaluation["status"] != "ok" or len(evaluation["complexes"]) != 1:
                    raise ValueError(f"invalid evaluation: {case_id}")
                for pose in evaluation["complexes"][0]["poses"]:
                    for artifact in (pose, pose.get("refined")):
                        if artifact and sha256_file(run_dir / artifact["path"]) != artifact["sha256"]:
                            raise ValueError(f"changed pose: {artifact['pose_id']}")
                poses = evaluation["complexes"][0]["poses"]
                costs.append({
                    "case": case_id,
                    "generation_wall_seconds": manifest["timings"]["inference_wall_seconds"],
                    "evaluation_wall_seconds": evaluation.get("evaluation_wall_seconds"),
                    "gpu_peak_memory_mib_whole_device": manifest.get("gpu_peak_memory_mib"),
                    "gnina_score_only_seconds": sum(p.get("scoring", {}).get("seconds", 0) for p in poses) if state["config"].get("gnina") else None,
                    "gnina_refinement_seconds": sum((p.get("refined") or {}).get("refinement", {}).get("seconds", 0) for p in poses) if state["config"].get("gnina") else None,
                })
            else:
                failures.append({"case": case_id, "error": entry.get("error")})
            for variant in enabled:
                metrics = evaluation["complexes"][0]["variants"][variant] if evaluation else None
                if metrics and (not metrics["available"] or any(metrics[k] is None for k in METRICS)):
                    raise ValueError(f"missing reference/scorer result: {case_id}/{variant}")
                rows.append({"complex_id": case["complex_id"], "seed": seed, "variant": variant,
                             "evaluated": evaluation is not None,
                             **{k: bool(metrics[k]) if metrics else k == "empty_selection" for k in METRICS}})
    result = {"schema_version": 1, "benchmark_id": protocol["benchmark_id"], "split": state["config"].get("split"),
              "protocol_sha256": state["config"]["protocol_sha256"], "state_sha256": sha256_file(out / "pilot_state.json"),
              **aggregate_rows(rows), "failed_cases": failures, "completed_case_costs": costs,
              "cost_note": "recorded wall time, not GPU kernel time; completed operations only; failure logs retain unsuccessful work",
              "failure_policy": "generation and evaluation failures remain failures in every enabled selector; errors are listed separately"}
    write_json(out / "summary.json", result)
    lines = ["# Pilot results", "", f"{result['n_complexes']} complexes; {result['n_complex_seed_pairs']} complex/seed pairs.",
             f"Failed cases: {len(failures)} (retained in every denominator).", "",
             "| Variant | Joint top-1 | RMSD top-1 | Valid top-1 | Joint top-5 | No selection |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, table in result["variants"].items():
        cells = [f"{table[k]['count']}/{table[k]['denominator']} ({table[k]['rate']:.1%})" for k in
                 ("top1_joint_success", "top1_rmsd_success", "top1_valid", "top5_joint_success", "empty_selection")]
        lines.append("| " + " | ".join([name] + cells) + " |")
    lines += ["", "Exploratory pilot. Confidence intervals in summary.json resample paired complexes, keeping seeds together.",
              "No independent-family or state-of-the-art claim follows from this subset.", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    args = ap.parse_args(argv)
    print(json.dumps(summarize_pilot(args.out), indent=2))


if __name__ == "__main__":
    main()
