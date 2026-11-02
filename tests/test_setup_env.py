"""Checks on the pinned upstream setup that need no GPU, torch or conda.

Tests marked ``upstream`` need a git clone of gcorso/DiffDock. They use $DIFFDOCK_UPSTREAM_SRC (a local clone,
fast) when set, otherwise the real GitHub URL, and are skipped when neither is reachable.
"""
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from docking_tools import setup_env
from docking_tools.setup_env import Layout


def _upstream_source():
    src = os.environ.get("DIFFDOCK_UPSTREAM_SRC", setup_env.UPSTREAM_REPO)
    ok = subprocess.run(["git", "ls-remote", "--exit-code", src, "HEAD"], capture_output=True).returncode == 0
    return src if ok else None


@pytest.fixture(scope="module")
def upstream_checkout(tmp_path_factory):
    src = _upstream_source()
    if src is None:
        pytest.skip("upstream DiffDock not reachable")
    mp = pytest.MonkeyPatch()
    mp.setattr(setup_env, "UPSTREAM_REPO", src)
    layout = Layout(tmp_path_factory.mktemp("wd"))
    setup_env.clone_upstream(layout)
    yield layout
    mp.undo()


def test_upstream_is_pinned_to_the_tested_commit(upstream_checkout):
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=upstream_checkout.repo, capture_output=True, text=True).stdout.strip()
    assert head == setup_env.UPSTREAM_COMMIT


def test_patch_applies_cleanly_and_is_idempotent(upstream_checkout):
    sha1 = setup_env.apply_patch(upstream_checkout)
    sha2 = setup_env.apply_patch(upstream_checkout)  # second call must be a no-op, not an error
    assert sha1 == sha2 and len(sha1) == 64
    text = (upstream_checkout.repo / "inference.py").read_text()
    compile(text, "inference.py", "exec")  # still valid Python
    assert "'--seed'" in text and "scores.json" in text and "traceback.format_exc()" in text
    assert "DIFFDOCK_RDKIT_SEED" in text  # --seed reaches RDKit's conformer generator, not just python/numpy/torch
    conf = (upstream_checkout.repo / "datasets" / "process_mols.py").read_text()
    compile(conf, "process_mols.py", "exec")
    assert "ps.randomSeed = int(os.environ['DIFFDOCK_RDKIT_SEED'])" in conf
    assert any(line.strip() == "import os" for line in conf.splitlines())


def test_upstream_cli_matches_what_run_py_relies_on(upstream_checkout):
    """The flags, CSV columns and output names run.py/poses.py depend on exist at the pinned commit."""
    text = (upstream_checkout.repo / "inference.py").read_text()
    for flag in ("--config", "--protein_ligand_csv", "--out_dir", "--samples_per_complex"):
        assert f"'{flag}'" in text
    for col in ("complex_name", "protein_path", "ligand_description", "protein_sequence"):
        assert f"df['{col}']" in text
    assert "rank{rank+1}_confidence{confidence[rank]:.2f}.sdf" in text
    # config values are applied after the command line (so overrides must go in the config)
    assert "arg_dict[key] = value" in text
    cfg = yaml.safe_load((upstream_checkout.repo / "default_inference_args.yaml").read_text())
    assert cfg["model_dir"] == "./workdir/v1.1/score_model" and cfg["confidence_ckpt"] == "best_model_epoch75.pt"


def _pins(lines):
    out = {}
    for item in lines:
        m = re.match(r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?==([^\s;]+)", item.strip())
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(2)
    return out


def _norm(v):
    return ".".join(str(int(x)) if x.isdigit() else x for x in v.split("."))


def test_env_pins_agree_with_upstream_environment_yml(upstream_checkout):
    ours = yaml.safe_load(setup_env.render_env_spec(cpu=False))
    theirs = yaml.safe_load((upstream_checkout.repo / "environment.yml").read_text())
    our_pip = next(d for d in ours["dependencies"] if isinstance(d, dict))["pip"]
    their_pip = [x for d in theirs["dependencies"] if isinstance(d, dict) for x in d["pip"] if isinstance(x, str)]
    our_pins, their_pins = _pins(our_pip), _pins(their_pip)
    for name, ver in our_pins.items():
        if name in their_pins:
            assert _norm(ver.split("+")[0]) == _norm(their_pins[name].split("+")[0]), name
    extras = set(our_pins) - set(their_pins)
    assert extras == set(), extras  # we add no pinned package that upstream does not pin
    ours_conda = {d.split("=")[0]: d.split("=", 1)[1] for d in ours["dependencies"] if isinstance(d, str) and "=" in d}
    theirs_conda = {d.split("==")[0].split("=")[0]: d for d in theirs["dependencies"] if isinstance(d, str)}
    assert ours_conda["python"] == "3.9.18" and "python=3.9.18" in theirs_conda.values()
    assert ours_conda["prody"] == "2.2.0" and ours_conda["scipy"] == "1.12.0"


def test_env_spec_is_valid_and_cpu_variant_has_no_cuda():
    cuda = setup_env.render_env_spec(cpu=False)
    cpu = setup_env.render_env_spec(cpu=True)
    assert "cu117" in cuda and "cu117" not in cpu and "torch==1.13.1+cpu" in cpu and "pt113cpu" in cpu
    assert "whl/cpu" in cpu and "torch-1.13.0+cpu.html" in cpu
    for text in (cuda, cpu):
        spec = yaml.safe_load(text)
        assert spec["channels"] == ["conda-forge"]
        pip = next(d for d in spec["dependencies"] if isinstance(d, dict))["pip"]
        assert not any("openfold" in p or "dllogger" in p or "gradio" in p for p in pip)


def test_layout_paths_are_under_workdir(tmp_path):
    lay = Layout(tmp_path)
    assert lay.repo == tmp_path / "DiffDock" and lay.python.parts[-2:] == ("bin", "python")
    assert lay.models_dir == tmp_path / "DiffDock" / "workdir" / "v1.1"
    assert lay.setup_info.name == "diffdock_setup.json"


def test_model_hash_check_rejects_a_corrupt_archive(tmp_path, monkeypatch):
    lay = Layout(tmp_path)
    (tmp_path).mkdir(exist_ok=True)
    (tmp_path / "diffdock_models_v1.1.zip").write_bytes(b"not the real archive")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        setup_env.download_models(lay)
    assert not (tmp_path / "diffdock_models_v1.1.zip").exists()  # bad archive removed so a retry re-downloads


def test_models_archive_layout_matches_expected_files(tmp_path, monkeypatch):
    """Build a zip shaped like the real release (score_model/, confidence_model/) and unpack it."""
    import zipfile

    archive = tmp_path / "diffdock_models_v1.1.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("score_model/model_parameters.yml", "a: 1")
        z.writestr("score_model/best_ema_inference_epoch_model.pt", b"x")
        z.writestr("confidence_model/model_parameters.yml", "a: 1")
        z.writestr("confidence_model/best_model_epoch75.pt", b"y")
    monkeypatch.setattr(setup_env, "MODELS_ZIP_SHA256", setup_env.sha256_file(archive))
    hashes = setup_env.download_models(Layout(tmp_path))
    assert set(hashes) == {
        "score_model/model_parameters.yml",
        "score_model/best_ema_inference_epoch_model.pt",
        "confidence_model/model_parameters.yml",
        "confidence_model/best_model_epoch75.pt",
    }
    assert setup_env.model_files_present(Layout(tmp_path))


def test_switching_to_a_newer_patch_restores_the_files_first(upstream_checkout):
    """The VM from an earlier session has an *older* patch applied: re-applying must not fail on it."""
    setup_env.apply_patch(upstream_checkout)
    inf = upstream_checkout.repo / "inference.py"
    kept = [line for line in inf.read_text().splitlines(keepends=True) if "DIFFDOCK_RDKIT_SEED" not in line]
    older = "".join(kept)
    assert older != inf.read_text()
    inf.write_text(older)  # what an earlier patch version (without the RDKit seed hunk) leaves behind
    assert subprocess.run(["git", "apply", "--reverse", "--check", str(setup_env.PATCH_FILE)], cwd=upstream_checkout.repo, capture_output=True).returncode != 0
    setup_env.apply_patch(upstream_checkout)
    assert "DIFFDOCK_RDKIT_SEED" in inf.read_text()
    assert subprocess.run(["git", "apply", "--reverse", "--check", str(setup_env.PATCH_FILE)], cwd=upstream_checkout.repo, capture_output=True).returncode == 0
