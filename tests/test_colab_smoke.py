"""The headless acceptance runner must preserve both success and failure evidence."""
import json
import sys

from docking_tools.scripts import colab_smoke


def test_headless_runner_exports_real_layout_and_repeats(fake_workdir, tmp_path, monkeypatch):
    monkeypatch.setattr(colab_smoke.setup_env, "UPSTREAM_COMMIT", "abc")
    out = tmp_path / "evidence"
    code = colab_smoke.main([
        "--workdir", str(fake_workdir), "--out", str(out), "--skip-setup", "--allow-cpu",
        "--python-exe", sys.executable, "--samples", "3", "--repeat",
    ])
    assert code == 0
    report = json.loads((out / "phase1_verification.json").read_text())
    assert report["status"] == "ok" and len(report["runs"]) == 2
    assert report["repeatability"]["byte_identical_poses"]
    assert len(list((out / "exports").glob("*.tar.gz"))) == 2


def test_failure_before_setup_still_leaves_a_report(tmp_path, monkeypatch):
    monkeypatch.setattr(colab_smoke, "gpu_info", lambda: None)
    out = tmp_path / "evidence"
    assert colab_smoke.main(["--workdir", str(tmp_path / "wd"), "--out", str(out)]) == 1
    report = json.loads((out / "phase1_verification.json").read_text())
    assert report["status"] == "failed" and "No GPU found" in report["error"]["message"]
    assert report["runs"] == []


def test_silent_docking_failure_is_not_an_acceptance_pass(fake_workdir, tmp_path, monkeypatch):
    monkeypatch.setattr(colab_smoke.setup_env, "UPSTREAM_COMMIT", "abc")
    monkeypatch.setenv("FAKE_DIFFDOCK_MODE", "silent")
    out = tmp_path / "evidence"
    assert colab_smoke.main([
        "--workdir", str(fake_workdir), "--out", str(out), "--skip-setup", "--allow-cpu",
        "--python-exe", sys.executable,
    ]) == 1
    report = json.loads((out / "phase1_verification.json").read_text())
    assert report["status"] == "failed" and not (out / "exports").exists()
    assert "no rank" in report["error"]["message"]
