"""Small stdlib-only helpers shared by the other modules."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence


def sha256_file(path: os.PathLike | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_streaming(
    cmd: Sequence[str],
    *,
    cwd: Optional[os.PathLike | str] = None,
    env: Optional[Mapping[str, str]] = None,
    log_path: Optional[os.PathLike | str] = None,
    echo: bool = True,
) -> int:
    """Run ``cmd``, echo merged stdout/stderr live and optionally tee it to ``log_path``."""
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    log = open(log_path, "w", encoding="utf-8") if log_path else None
    proc: Optional[subprocess.Popen] = None
    try:
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd) if cwd else None,
            env=full_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if echo:
                sys.stdout.write(line)
                sys.stdout.flush()
            if log:
                log.write(line)
        return proc.wait()
    finally:  # an interrupted notebook cell (KeyboardInterrupt) must not leave docking running on the GPU
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        if log:
            log.close()


def gpu_info() -> Optional[dict]:
    """Name / total memory / driver of the first GPU, or None without nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip().splitlines()
    except (subprocess.SubprocessError, OSError):
        return None
    if not out:
        return None
    name, mem, driver = [x.strip() for x in out[0].split(",")]
    return {"name": name, "memory_total_mib": int(float(mem)), "driver_version": driver}


class GpuMemoryPoller:
    """Polls ``nvidia-smi`` in a thread and keeps the peak used memory (MiB, whole GPU)."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.peak_mib: Optional[int] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._enabled = shutil.which("nvidia-smi") is not None

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                ).stdout.strip().splitlines()
                if out:
                    used = int(float(out[0]))
                    self.peak_mib = used if self.peak_mib is None else max(self.peak_mib, used)
            except (subprocess.SubprocessError, OSError, ValueError):
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "GpuMemoryPoller":
        if self._enabled:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


def write_json(path: os.PathLike | str, obj) -> None:
    """Write JSON atomically (temp file + rename) so an interrupted write never leaves a truncated file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path: os.PathLike | str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def relpath(path: os.PathLike | str, start: os.PathLike | str) -> str:
    return Path(os.path.relpath(path, start)).as_posix()
