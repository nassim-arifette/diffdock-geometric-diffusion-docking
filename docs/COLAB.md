# Colab CLI: run the current local revision

The [official CLI](https://github.com/googlecolab/google-colab-cli) runs on Linux
and macOS. On this Windows computer, `colab.cmd` invokes `wsl colab`; PowerShell
can call it, provided its execution environment permits WSL and network access.

The commands below use a named runtime to avoid disturbing other work. Check
existing sessions first. Do not assume an October runtime or unfinished run remains.

## Package and upload (PowerShell)

From this repository, with the helper requirements installed:

```powershell
python tools/build_notebooks.py
python tools/package_colab.py
colab sessions
colab new -s cv-diffdock --gpu T4
$localBundle = (Resolve-Path 'colab_artifacts/diffdock_colab.tar.gz').Path
$wslBundle = (wsl -d Ubuntu -- wslpath -a -u $localBundle).Trim()
colab upload -s cv-diffdock $wslBundle /content/diffdock_colab.tar.gz
colab upload -s cv-diffdock /mnt/c/Users/nassi/Desktop/Projets-CV-MVA/02-DiffDock-geometric-diffusion/colab_artifacts/package_manifest.json /content/diffdock_package_manifest.json
```

This uploads local code; it does not depend on unpublished changes being on GitHub.
The package manifest records every source hash. Keep it with downloaded results.

`tools/colab_launch.py` verifies the archive and each source file, then starts
`tools/colab_acceptance.py` as an independent logged subprocess. The runner uses
`uv` to provision Python 3.11 for the helper stack and a separate pinned Python
3.9 environment for upstream inference. Colab's system Python 3.13 is not used
to install either stack. Replace the following local WSL path for another PC:

```powershell
colab exec -s cv-diffdock --file /mnt/c/Users/nassi/Desktop/Projets-CV-MVA/02-DiffDock-geometric-diffusion/tools/colab_launch.py --timeout 60
```

The default project and state paths are `/content/diffdock_project` and
`/content/diffdock_job`. Set `DIFFDOCK_PROJECT_ROOT` and `DIFFDOCK_JOB_STATE` with
CLI `--env` options for another preserved attempt. The launcher refuses to
overwrite an existing project. Check `status.json` and `launcher.log` in the
state directory; a returned PID means started, not successful.

## Persistent outputs and acceptance

Mount Drive once with `colab drivemount -s cv-diffdock` in an interactive terminal.
Complete Google authorization in the browser before pressing Enter in the CLI.
The browser flow can expire; use a new request after an expired or failed attempt.
Without a successful mount, omit `--drive-root`
and download outputs before releasing the runtime. Do not create a directory that
merely looks like a mounted Drive.

```powershell
@'
import subprocess
from pathlib import Path
root = Path('/content/diffdock_project')
cmd = [str(root/'.helper-py311/bin/python'), str(root/'docking_tools/scripts/colab_smoke.py'),
       '--workdir', '/content/diffdock_work', '--out', '/content/phase1_out', '--repeat']
if Path('/content/drive/MyDrive').is_dir():
    cmd += ['--drive-root', '/content/drive/MyDrive/Projets-CV-MVA/DiffDock']
subprocess.run(cmd, cwd=root, check=True)
'@ | colab exec -s cv-diffdock
```

First inspect `phase1_verification.json`, the real poses, the exported archive and
the inspector. A successful fake-upstream test is not acceptance of this step.
Initial setup downloads weights and computes reusable lookup tables.

## Pilot

On the VM, working in `/content/diffdock_project`:

```bash
python -m docking_tools.pilot_dataset --out /content/posebusters_pilot
python -m docking_tools.pilot --manifest /content/posebusters_pilot/pilot_manifest.json --split dev --out /content/pilot_dev --skip-setup
python -m docking_tools.pilot_summary /content/pilot_dev
```

Run the development cases first. Once the unchanged protocol is fixed, use
`--split eval --out /content/pilot_eval`, then summarize that directory.
Pass an installed GNINA binary with `--gnina` for C/D. Follow the [official GNINA
release instructions](https://github.com/gnina/gnina) for the selected GPU;
do not silently replace scoring models or binaries between cases. Hash/version
are captured. Without GNINA, report only A/B.

Optional `--drive-root /content/drive/MyDrive/Projets-CV-MVA/DiffDock` persists each run
and caches setup downloads. `--retry-failed` retries failures; completed candidate
pools are reused and their hashes checked. The dataset manifest records source
archive and final-list hashes. Download that manifest and the full result trees.

## Retrieve before releasing

Create one archive on the VM containing `/content/phase1_out`, the pilot outputs,
and the frozen dataset manifest/inputs; download it with `colab download` to the
local `colab_artifacts` directory. Inspect it before `colab stop -s cv-diffdock`.
Keep failures and logs. Report setup and inference time separately.

If the calling terminal reports `Wsl/Service/CreateInstance/E_ACCESSDENIED`, the
CLI has not reached Colab: its host execution permissions must be resolved first.
Never record that failure as a GPU experiment or claim that a runtime was launched.
