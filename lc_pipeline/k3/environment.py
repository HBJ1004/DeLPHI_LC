"""Immutable environment capture for the definitive K3 publication release."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .manifest import implementation_commit
from .protocol import K3_PROTOCOL_SHA256


class K3EnvironmentError(ValueError):
    """Raised when an environment snapshot cannot be captured safely."""


def _sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _valid_commit(value: str, label: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise K3EnvironmentError(f"{label} must be a full lowercase Git commit")
    return value


def _command(command: Sequence[str], *, required: bool) -> dict[str, object]:
    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        if required:
            raise K3EnvironmentError(f"cannot run {' '.join(command)}: {exc}") from exc
        return {
            "command": list(command),
            "returncode": None,
            "stdout": None,
            "stderr": None,
            "error": str(exc),
        }
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if required and result.returncode:
        raise K3EnvironmentError(
            f"{' '.join(command)} exited {result.returncode}: {stdout} {stderr}".strip()
        )
    return {
        "command": list(command),
        "returncode": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "error": None,
    }


def _immutable_text(path: Path, contents: str) -> str:
    encoded = contents.encode("utf-8")
    digest = _sha256_bytes(encoded)
    if path.exists():
        if path.read_bytes() == encoded:
            return digest
        raise K3EnvironmentError(f"refusing to overwrite environment artifact {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest


def _process_record(pid: int | None) -> dict[str, object] | None:
    if pid is None:
        return None
    process = Path("/proc") / str(pid)
    try:
        command = (process / "cmdline").read_bytes().rstrip(b"\0").replace(b"\0", b" ")
        executable = os.readlink(process / "exe")
        environment = (process / "environ").read_bytes().split(b"\0")
    except (OSError, UnicodeError) as exc:
        raise K3EnvironmentError(f"cannot inspect analysis process {pid}: {exc}") from exc
    selected_names = (
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_VISIBLE_DEVICES",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "PYTHONPATH",
        "PYTHONHASHSEED",
    )
    decoded: dict[str, str] = {}
    for item in environment:
        name, separator, value = item.partition(b"=")
        decoded_name = name.decode("utf-8", errors="replace")
        if separator and decoded_name in selected_names:
            decoded[decoded_name] = value.decode("utf-8", errors="replace")
    return {
        "pid_at_capture": pid,
        "executable": executable,
        "command_sha256": _sha256_bytes(command),
        "selected_environment": decoded,
    }


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip() or None
    except OSError:
        pass
    return platform.processor() or None


def capture_publication_environment(
    *,
    output_directory: str | Path,
    repository_root: str | Path,
    analysis_commit: str,
    analysis_pid: int | None = None,
    capture_tool_commit: str | None = None,
) -> dict[str, object]:
    """Write an exact package freeze and a hash-bound machine/software manifest."""
    analysis_commit = _valid_commit(analysis_commit, "analysis_commit")
    tool_commit = _valid_commit(
        capture_tool_commit or implementation_commit(repository_root),
        "capture_tool_commit",
    )
    freeze_record = _command(
        [sys.executable, "-m", "pip", "freeze", "--all"], required=True
    )
    freeze = str(freeze_record["stdout"]).rstrip() + "\n"
    destination = Path(output_directory)
    freeze_path = destination / "pip-freeze.txt"
    freeze_sha256 = _immutable_text(freeze_path, freeze)

    import torch

    nvidia = _command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,vbios_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        required=False,
    )
    compiler = _command(["cc", "--version"], required=False)
    payload: dict[str, object] = {
        "schema": "delphi.k3-publication-environment.v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": K3_PROTOCOL_SHA256,
        "analysis_commit": analysis_commit,
        "capture_tool_commit": tool_commit,
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "hardware": {"logical_cpu_count": os.cpu_count(), "cpu_model": _cpu_model()},
        "accelerator": {
            "torch_version": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "nvidia_smi": nvidia,
        },
        "compiler": compiler,
        "analysis_process": _process_record(analysis_pid),
        "pip_freeze": {
            "path": "pip-freeze.txt",
            "sha256": freeze_sha256,
            "line_count": len(freeze.splitlines()),
        },
    }
    encoded = json.dumps(
        payload, allow_nan=False, sort_keys=True, separators=(",", ":")
    ) + "\n"
    manifest_path = destination / "environment.json"
    manifest_sha256 = _immutable_text(manifest_path, encoded)
    return {**payload, "environment_manifest_sha256": manifest_sha256}


__all__ = [
    "K3EnvironmentError",
    "capture_publication_environment",
]
