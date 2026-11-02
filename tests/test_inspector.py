import json
import re
import sys

import pytest

from docking_tools import run
from docking_tools._util import sha256_file
from docking_tools.inspector import InspectorError, build_inspector_data, write_inspector
from docking_tools.manifest import ManifestError, load_manifest, save_manifest


@pytest.fixture
def inspectable_run(fake_workdir, pdb_file, ligand_sdf, monkeypatch):
    monkeypatch.setenv("FAKE_DIFFDOCK_MODE", "ok")
    rd = run.new_run_dir(fake_workdir / "runs", "inspect")
    prepared = run.prepare_inputs(
        rd, complex_name="toy", receptor=str(pdb_file), ligand="COc(cc1)ccc1C#N",
        reference=str(ligand_sdf), reference_note="Synthetic fixture, not an experimental structure",
    )
    run.run_diffdock(fake_workdir, prepared, python_exe=sys.executable, echo=False,
                     overrides={"samples_per_complex": 3, "save_visualisation": True})
    return rd


def _evaluation(rd):
    manifest = load_manifest(rd)
    c = manifest["complexes"][0]
    report = {
        "schema_version": 1, "status": "ok", "run_id": manifest["run_id"],
        "manifest_sha256": sha256_file(rd / "manifest.json"),
        "complexes": [{"complex_name": c["complex_name"], "poses": [
            {**p, "valid": True, "checks": {"bond_lengths": True}, "rmsd_angstrom": 1.25,
             "gnina_score": None, "errors": [], "refined": None} for p in c["poses"]
        ], "variants": {}, "oracle": {}}],
    }
    (rd / "evaluation.json").write_text(json.dumps(report), encoding="utf-8")
    return report


def test_documented_cli_exports_a_real_html_file(inspectable_run, tmp_path):
    import subprocess
    import sys

    output = tmp_path / "from_cli.html"
    result = subprocess.run(
        [sys.executable, "-m", "docking_tools.inspector", str(inspectable_run), "--out", str(output)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.is_file()
    assert "3Dmol" in output.read_text(encoding="utf-8")


def test_unevaluated_run_embeds_exact_poses_reference_and_real_frames(inspectable_run):
    rd = inspectable_run
    data = build_inspector_data(rd)
    c = data["complexes"][0]
    assert data["evaluation_status"] is None
    assert c["poses"][0]["metrics"] is None
    assert c["poses"][0]["confidence"] != c["poses"][0]["confidence_rounded"]
    manifest = load_manifest(rd)
    pose = manifest["complexes"][0]["poses"][0]
    assert c["poses"][0]["sdf"] == (rd / pose["path"]).read_text()
    assert "Synthetic fixture" in c["reference_note"]
    tr = c["poses"][0]["trajectory"]
    assert tr["frames"][0]["kind"] == "random_start"
    assert tr["frames"][-1]["kind"] == "final"
    assert any("overwrites" in n for n in tr["notes"])
    assert tr["pdb"].count("ENDMDL") == len(tr["frames"])


def test_report_provenance_is_required_not_merely_matching_pose_rank(inspectable_run):
    rd = inspectable_run
    report = _evaluation(rd)
    assert build_inspector_data(rd)["complexes"][0]["poses"][0]["metrics"]["rmsd_angstrom"] == 1.25
    report["manifest_sha256"] = "wrong-run"
    (rd / "evaluation.json").write_text(json.dumps(report))
    with pytest.raises(InspectorError, match="manifest_sha256"):
        build_inspector_data(rd)
    report["manifest_sha256"] = sha256_file(rd / "manifest.json")
    report["complexes"][0]["poses"][0]["sha256"] = "wrong-pose"
    (rd / "evaluation.json").write_text(json.dumps(report))
    with pytest.raises(InspectorError, match="pose provenance"):
        build_inspector_data(rd)


def test_refined_coordinates_are_linked_without_invented_confidence_or_frames(inspectable_run):
    rd = inspectable_run
    report = _evaluation(rd)
    p = report["complexes"][0]["poses"][0]
    refined_path = rd / "refined.sdf"
    refined_path.write_text((rd / p["path"]).read_text())
    rid = p["pose_id"] + ":refined"
    p["refined"] = {"pose_id": rid, "path": "refined.sdf", "sha256": sha256_file(refined_path),
                    "valid": False, "checks": {"bond_lengths": False}, "errors": []}
    report["complexes"][0]["variants"] = {"D_refined_gnina": {"available": True, "selected_pose_ids": [rid]}}
    (rd / "evaluation.json").write_text(json.dumps(report))
    poses = build_inspector_data(rd)["complexes"][0]["poses"]
    r = next(x for x in poses if x["pose_id"] == rid)
    assert r["parent_pose_id"] == p["pose_id"]
    assert r["confidence"] is None and r["trajectory"] is None
    assert r["parent_confidence"] == p["confidence"]
    assert r["metrics"]["valid"] is False


def test_embedded_html_treats_names_as_data_and_can_bundle_renderer(inspectable_run, tmp_path):
    rd = inspectable_run
    manifest = load_manifest(rd)
    attack = '</script><script>alert("bad")</script>'
    manifest["complexes"][0]["reference"]["provenance"] = attack
    save_manifest(rd, manifest)
    page = write_inspector(rd).read_text(encoding="utf-8")
    assert attack not in page
    payload = re.search(r'<script id="inspector-data" type="application/json">(.*?)</script>', page, re.S).group(1)
    assert json.loads(payload)["complexes"][0]["reference_note"] == attack
    assert "Rendering requires the 3Dmol.js CDN" in page
    renderer = tmp_path / "renderer.js"
    renderer.write_text("window.$3Dmol = {};")
    offline = write_inspector(rd, tmp_path / "offline.html", three_dmol_js=renderer).read_text(encoding="utf-8")
    assert 'src="data:text/javascript;base64,' in offline
    assert 'src="https://' not in offline
    assert "viewed offline" in offline


def test_stored_file_corruption_is_not_silently_displayed(inspectable_run):
    rd = inspectable_run
    pose = load_manifest(rd)["complexes"][0]["poses"][0]
    (rd / pose["path"]).write_text("broken")
    with pytest.raises(ManifestError, match="hash mismatch"):
        write_inspector(rd)


def test_manifest_cannot_embed_a_file_outside_the_run(inspectable_run):
    rd = inspectable_run
    manifest = load_manifest(rd)
    manifest["complexes"][0]["receptor"]["path"] = "../outside.pdb"
    save_manifest(rd, manifest)
    with pytest.raises(InspectorError, match="outside the run"):
        write_inspector(rd)


def test_unknown_variant_candidate_is_not_mislabeled(inspectable_run):
    rd = inspectable_run
    report = _evaluation(rd)
    report["complexes"][0]["variants"] = {"A_confidence": {"selected_pose_ids": ["unknown"]}}
    (rd / "evaluation.json").write_text(json.dumps(report))
    with pytest.raises(InspectorError, match="unknown pose"):
        build_inspector_data(rd)
