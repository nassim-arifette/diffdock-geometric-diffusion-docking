"""Regenerate the notebooks in ``notebooks/`` from their builders.

    python tools/build_notebooks.py            # all notebooks
    python tools/build_notebooks.py geometry   # one of: geometry docking benchmark inspector experiment

The notebooks are generated files (saved outputs are never committed); change the ``tools/nb_*.py`` builders instead.
Needs ``nbformat``.
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

BUILDERS = {
    "geometry": ("nb_geometry", "build"),
    "docking": ("nb_docking", "build_docking"),
    "benchmark": ("nb_docking", "build_benchmark_prep"),
    "inspector": ("nb_inspector", "build"),
    "experiment": ("nb_experiment", "build"),
}


def main(argv):
    wanted = argv or list(BUILDERS)
    for name in wanted:
        if name not in BUILDERS:
            raise SystemExit(f"unknown notebook {name!r}; choose from {sorted(BUILDERS)}")
        module, func = BUILDERS[name]
        try:
            mod = importlib.import_module(module)
        except ModuleNotFoundError:
            if argv:
                raise
            print(f"skipping {name}: {module}.py not written yet")
            continue
        getattr(mod, func)()


if __name__ == "__main__":
    main(sys.argv[1:])
