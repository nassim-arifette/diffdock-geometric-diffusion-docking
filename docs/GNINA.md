# Pinned GNINA for the fixed-candidate pilot

The optional C/D variants use the official **GNINA v1.3.3** release, commit
`6fe1ce2bb9c35c8067de9f49bb9169857dfbad70`. Its Linux x86_64 asset is
`gnina.cuda12.8.static` (2,056,131,000 bytes), with the published SHA-256
`3340c1f49cd3c7c84d8699182a1c6af13c7fa2a22448d1204640446106f72172`.
These values were checked against the [release API](https://api.github.com/repos/gnina/gnina/releases/tags/v1.3.3)
and [official asset listing](https://github.com/gnina/gnina/releases/expanded_assets/v1.3.3).
The tag, rather than the changing `latest` URL, is fixed for all pilot cases.

From the project root on the Linux runtime:

```bash
python tools/prepare_gnina.py --out /content/gnina
```

This downloads the binary, verifies its exact size and hash before execution,
then records `ldd`, `--version` and `--help`. It neither installs packages nor
changes CUDA, Torch, Python or system library paths. Existing mismatched files
are refused rather than overwritten. Every attempt writes a separate
`/content/gnina/installation-<UTC>.json` report, including any error.

The asset name does **not** guarantee a fully static executable. The
[official build recipe](https://github.com/gnina/gnina/blob/v1.3.3/docker/compat/Dockerfile)
links CUDA and cuDNN dynamically. The release family requires cuDNN 9;
CUDA 12 libraries must be available even if `nvidia-smi` advertises a CUDA 13
driver. Do not replace DiffDock's isolated Torch 1.13/CUDA 11 environment to
resolve a GNINA library error. Read the recorded `ldd` output first. The actual
machine and scoring check, not just a successful download, determine usability.

Reading the released ELF's `DT_NEEDED` entries confirms these GPU libraries:
`libcudnn.so.9`, `libcudart.so.12`, `libcusparse.so.12`, `libcufft.so.11`,
`libcublas.so.12`, `libcublasLt.so.12` and `libcusolver.so.11`. The remaining
direct dependencies are the standard C/C++/math runtime and ELF loader.
There is no dynamically linked `libtorch` dependency in this asset. Libraries
loaded indirectly at scoring time are still covered by the real smoke check.

## Colab image with CUDA 12 and CUDA 13 installed together

On the runtime used on 2026-11-01, the necessary CUDA 12 wheels were already
installed but absent from the ELF loader's library path. The original executable's
RPATH also selected a system CUDA 13 library with the same soname. For this image:

```bash
python tools/bundle_gnina_libraries.py --out /content/gnina
```

This copies the existing CUDA 12 wheel libraries into `libenv/lib`, checking
their installed versions and every library against its wheel RECORD digest.
The script keeps their licenses, records all SHA-256 values in
`runtime_manifest.json`, and creates `/content/gnina/gnina`. That launcher uses
the system ELF loader with an explicit local library path and disables the
binary's RPATH. It does not modify the original release executable, Python,
Torch, or any global environment variable. The loader report must resolve all
direct GPU dependencies from the local bundle before the launcher is accepted.

The wheel versions in this helper are intentionally fixed to the observed image;
a different image requires reviewing the pins rather than silently substituting
libraries. Keep `runtime_manifest.json` beside the launcher and original binary:
the pipeline's executable hash identifies the launcher, and this manifest also
identifies the underlying release binary and every library.

After DiffDock's acceptance run, choose one generated pose and its saved receptor
in the **same coordinate frame**, then run with the Python helper environment:

```bash
python tools/prepare_gnina.py --out /content/gnina \
  --receptor /path/to/saved/receptor.pdb --ligand /path/to/generated/pose.sdf
```

Add `--bundled-libraries` when using the isolated launcher. This verifies the
saved bundle manifest and every library hash before running the same checks
through `/content/gnina/gnina`.

This rechecks the binary and exercises both `--score_only` and `--minimize`
through the project's adapter. The check requires a finite `CNNscore` in each
output and rejects coordinate changes during score-only evaluation. The report
records input/output hashes, commands and durations; `real_scoring_validated`
stays false until both commands succeed. Scoring success is separate from a
physical-validity or docking-accuracy result. Avoid overlapping this smoke check
with a GPU-intensive DiffDock step on a small runtime.

Once the real check passes, supply `--gnina /content/gnina/gnina.cuda12.8.static`
to `docking_tools.pilot` on a machine where the default loader check succeeded,
or `--gnina /content/gnina/gnina` when using the isolated library bundle.
Keep the installation report and binary hash with the frozen pilot protocol.
The default model bundled in this fixed executable is used throughout; changing
the executable or scoring model requires a new pilot output directory.

Upstream usage and the distinction between scoring and local minimisation are
documented in the [GNINA v1.3.3 README](https://github.com/gnina/gnina/blob/v1.3.3/README.md).
