#!/usr/bin/env python3
"""Read-only environment discovery on this machine or an existing SSH host."""

import argparse
import csv
import ctypes
from datetime import datetime, timezone
import getpass
import io
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time


SCHEMA_VERSION = 1
PROBE_MARKER = "__AUTORESEARCH_ENVIRONMENT__="
OVERALL_TIMEOUT_SECONDS = 90
COMMAND_NAMES = (
    "ssh", "tmux", "timeout", "git", "codex", "claude", "python3",
    "python", "pip", "pip3", "conda", "uv", "nvidia-smi", "sbatch",
    "srun", "rsync",
)


def build_probe_command(host=None, python_executable=None, gpu_smoke=False,
                        device=0, ssh_executable=None, workdir=None):
    """Build argv; only the remote command is interpreted by a POSIX shell."""
    if device < 0:
        raise ValueError("device must be a nonnegative CUDA logical device index")
    interpreter = python_executable or ("python3" if host else sys.executable)
    if not interpreter or "\x00" in interpreter:
        raise ValueError("python interpreter must be a nonempty executable path")
    probe_args = [interpreter, "-", "--_probe", "--device", str(device)]
    if gpu_smoke:
        probe_args.append("--gpu-smoke")
    if workdir is not None:
        workdir = os.fspath(workdir)
        if not workdir or "\x00" in workdir:
            raise ValueError("workdir must be a nonempty existing directory path")
        probe_args.append("--workdir=" + workdir)
    if host is None:
        return probe_args
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@:-]*", host):
        raise ValueError("host must be an SSH alias or hostname without whitespace or options")
    ssh_binary = ssh_executable or shutil.which("ssh")
    if not ssh_binary:
        raise FileNotFoundError("ssh is unavailable; install or select an existing SSH client")
    return [
        ssh_binary,
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=12",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        host,
        " ".join(shlex.quote(arg) for arg in probe_args),
    ]


def parse_nvidia_csv(output):
    """Parse the exact five-column nvidia-smi query, preserving device order."""
    result = []
    for row in csv.reader(io.StringIO(output), skipinitialspace=True):
        if not row or all(not value.strip() for value in row):
            continue
        if len(row) != 5:
            raise ValueError("unexpected nvidia-smi CSV row: {!r}".format(row))
        index, name, uuid, memory, driver = [value.strip() for value in row]
        result.append({
            "index": int(index), "name": name, "uuid": uuid,
            "memory_total_mib": float(memory) if memory not in ("N/A", "[N/A]") else None,
            "driver_version": driver,
        })
    return result


def _text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def collect_memory():
    """Collect bytes available to the OS; this is not a job allocation check."""
    try:
        if sys.platform.startswith("linux"):
            values = {}
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                key, _, remainder = line.partition(":")
                fields = remainder.split()
                if fields and fields[0].isdigit():
                    values[key] = int(fields[0]) * 1024
            return {"status": "PASS", "source": "/proc/meminfo",
                    "total_bytes": values.get("MemTotal"),
                    "available_bytes": values.get("MemAvailable")}
        if sys.platform == "win32":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in (
                        "total_physical", "available_physical", "total_pagefile",
                        "available_pagefile", "total_virtual", "available_virtual", "extended",
                    )
                ]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                raise OSError("GlobalMemoryStatusEx failed")
            return {"status": "PASS", "source": "GlobalMemoryStatusEx",
                    "total_bytes": status.total_physical,
                    "available_bytes": status.available_physical}
        return {"status": "UNAVAILABLE", "source": None,
                "total_bytes": None, "available_bytes": None}
    except (OSError, ValueError) as exc:
        return {"status": "FAILED", "error": str(exc), "total_bytes": None,
                "available_bytes": None}


def collect_nvidia():
    executable = shutil.which("nvidia-smi")
    result = {"available": bool(executable), "status": "UNAVAILABLE",
              "gpus": [], "returncode": None, "stdout": "", "stderr": ""}
    if not executable:
        return result
    command = [executable, "--query-gpu=index,name,uuid,memory.total,driver_version",
               "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=12)
        result.update(returncode=completed.returncode, stdout=completed.stdout,
                      stderr=completed.stderr)
        if completed.returncode != 0:
            result.update(status="FAILED", error="nvidia-smi returned a nonzero exit code")
        else:
            result.update(status="PASS", gpus=parse_nvidia_csv(completed.stdout))
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        result.update(status="FAILED", error=str(exc))
        if isinstance(exc, subprocess.TimeoutExpired):
            result.update(stdout=_text(exc.stdout), stderr=_text(exc.stderr))
    return result


def collect_torch(gpu_smoke=False, device=0):
    """Query torch; allocate and synchronize a tiny CUDA matmul only on request."""
    torch_info = {"status": "UNAVAILABLE", "version": None, "cuda_version": None,
                  "cuda_available": None, "cuda_device_count": None}
    smoke = {"status": "NOT_RUN", "requested": gpu_smoke,
             "device": "cuda:{}".format(device), "elapsed_seconds": None,
             "error": None}
    try:
        import torch
        torch_info.update(version=str(torch.__version__), cuda_version=torch.version.cuda)
        torch_info.update(cuda_available=bool(torch.cuda.is_available()),
                          cuda_device_count=int(torch.cuda.device_count()), status="PASS")
    except Exception as exc:
        torch_info.update(status="UNAVAILABLE" if isinstance(exc, ImportError) else "FAILED",
                          error="{}: {}".format(type(exc).__name__, exc))
        if gpu_smoke:
            smoke.update(status="FAILED", error=torch_info["error"])
        return torch_info, smoke
    if not gpu_smoke:
        return torch_info, smoke
    started = time.monotonic()
    try:
        if not torch_info["cuda_available"]:
            raise RuntimeError("torch reports CUDA unavailable")
        if device >= torch_info["cuda_device_count"]:
            raise RuntimeError("CUDA logical device {} is unavailable (count={})".format(
                device, torch_info["cuda_device_count"]))
        tensor = torch.ones((16, 16), dtype=torch.float32, device=smoke["device"])
        product = tensor @ tensor
        torch.cuda.synchronize(device)
        expected = torch.full_like(product, 16.0)
        if not bool(torch.allclose(product, expected)):
            raise RuntimeError("CUDA matmul produced an unexpected result")
        smoke.update(status="PASS", check="allocated_16x16_float32_matmul",
                     result_sample=float(product[0, 0].item()))
    except Exception as exc:
        smoke.update(status="FAILED", error="{}: {}".format(type(exc).__name__, exc))
    smoke["elapsed_seconds"] = round(time.monotonic() - started, 6)
    return torch_info, smoke


def collect_environment(gpu_smoke=False, device=0):
    uname = platform.uname()
    try:
        user = {"name": getpass.getuser(), "status": "PASS"}
    except (OSError, KeyError, ImportError) as exc:
        user = {"name": None, "status": "FAILED", "error": str(exc)}
    try:
        cwd = str(Path.cwd())
        disk = shutil.disk_usage(cwd)
        disk_info = {"status": "PASS", "path": cwd, "total_bytes": disk.total,
                     "used_bytes": disk.used, "free_bytes": disk.free}
    except OSError as exc:
        disk_info = {"status": "FAILED", "error": str(exc)}
    torch_info, smoke = collect_torch(gpu_smoke, device)
    return {
        "schema_version": SCHEMA_VERSION,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "probe_status": "FAILED" if smoke["status"] == "FAILED" else "PASS",
        "host": uname.node,
        "kernel": {"system": uname.system, "release": uname.release,
                   "version": uname.version, "machine": uname.machine},
        "user": user,
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "disk": disk_info,
        "memory": collect_memory(),
        "commands": {name: shutil.which(name) for name in COMMAND_NAMES},
        "gpu": {"nvidia_smi": collect_nvidia(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "torch": torch_info, "smoke": smoke},
        "allocation_guaranteed": False,
    }


def parse_probe_output(stdout):
    """Accept only the marked JSON emitted by this probe, ignoring login banners."""
    lines = [line[len(PROBE_MARKER):] for line in stdout.splitlines()
             if line.startswith(PROBE_MARKER)]
    if len(lines) != 1:
        raise ValueError("expected exactly one probe JSON record; received {}".format(len(lines)))
    data = json.loads(lines[0])
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported or malformed probe JSON schema")
    if data.get("probe_status") not in ("PASS", "FAILED"):
        raise ValueError("probe JSON has no valid probe_status")
    gpu = data.get("gpu")
    if not isinstance(gpu, dict) or not isinstance(gpu.get("smoke"), dict):
        raise ValueError("probe JSON has no valid GPU smoke record")
    smoke = gpu["smoke"]
    if smoke.get("status") not in ("PASS", "FAILED", "NOT_RUN"):
        raise ValueError("probe JSON has no valid GPU smoke status")
    if smoke["status"] == "FAILED" and data["probe_status"] == "PASS":
        raise ValueError("probe JSON reports PASS despite failed GPU smoke")
    return data


def run_probe(host=None, python_executable=None, gpu_smoke=False, device=0, workdir=None):
    """Run a bounded child process and always return a JSON-serializable report."""
    document = {
        "schema_version": SCHEMA_VERSION, "status": "FAILED",
        "target": {"mode": "ssh" if host is not None else "local", "host": host,
                   "python": python_executable or ("python3" if host else sys.executable),
                   "gpu_smoke_requested": gpu_smoke, "device": device,
                   "workdir": os.fspath(workdir) if workdir is not None else None},
        "transport": {"status": "FAILED", "returncode": None,
                      "timeout_seconds": OVERALL_TIMEOUT_SECONDS, "stderr": ""},
        "environment": None,
    }
    try:
        command = build_probe_command(host, python_executable, gpu_smoke, device,
                                      workdir=workdir)
        source = Path(__file__).read_text(encoding="utf-8")
        completed = subprocess.run(
            command, input=source, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=OVERALL_TIMEOUT_SECONDS,
        )
        document["transport"].update(returncode=completed.returncode, stderr=completed.stderr)
        try:
            data = parse_probe_output(completed.stdout)
        except (ValueError, TypeError, AttributeError) as exc:
            document["transport"].update(error_type="INVALID_RESPONSE", error=str(exc),
                                          stdout=completed.stdout)
            if completed.returncode != 0:
                document["transport"]["error_type"] = "PROCESS_FAILED"
            return document
        document["environment"] = data
        document["transport"]["status"] = "PASS"
        smoke_status = data["gpu"]["smoke"]["status"]
        expected_smoke = smoke_status in ("PASS", "FAILED") if gpu_smoke else smoke_status == "NOT_RUN"
        expected_returncode = 0 if data["probe_status"] == "PASS" else 1
        if completed.returncode == 0 and data["probe_status"] == "PASS" and expected_smoke:
            document["status"] = "PASS"
        elif not expected_smoke:
            document["transport"].update(status="FAILED", error_type="INVALID_RESPONSE",
                                          error="response does not match requested GPU smoke mode")
        elif completed.returncode != expected_returncode:
            document["transport"].update(status="FAILED", error_type="PROCESS_FAILED",
                                          error="probe exit code does not match its reported status")
    except subprocess.TimeoutExpired as exc:
        document["transport"].update(error_type="TIMEOUT", error=str(exc),
                                      stdout=_text(exc.stdout), stderr=_text(exc.stderr))
    except (OSError, ValueError) as exc:
        document["transport"].update(error_type=type(exc).__name__, error=str(exc))
    return document


def write_report(document, out_path):
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Read-only Python, memory, disk, command and GPU discovery on an existing host.",
        epilog="SSH uses existing host-key verification. GPU discovery does not guarantee an allocation; "
               "--gpu-smoke performs one tiny synchronized CUDA matmul. Nothing is installed.",
    )
    parser.add_argument("--host", help="existing SSH alias/hostname; omit to probe locally")
    parser.add_argument("--python", dest="python_executable",
                        help="Python executable path (default: local interpreter, or python3 over SSH)")
    parser.add_argument("--out", help="local UTF-8 JSON report path (required)")
    parser.add_argument("--workdir", help="existing directory on the target for disk checks; "
                        "omit to use its default working directory (never created)")
    parser.add_argument("--gpu-smoke", action="store_true", help="allocate and test a 16 x 16 CUDA matmul")
    parser.add_argument("--device", type=int, default=0,
                        help="nonnegative CUDA logical device index, respecting CUDA_VISIBLE_DEVICES (default: 0)")
    parser.add_argument("--_probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.device < 0:
        parser.error("--device must be nonnegative")
    if args._probe:
        if args.workdir is not None:
            try:
                os.chdir(args.workdir)
            except (OSError, ValueError) as exc:
                print("FAILED: workdir={!r}: {}: {}".format(
                    args.workdir, type(exc).__name__, exc), file=sys.stderr)
                return 1
        data = collect_environment(args.gpu_smoke, args.device)
        print(PROBE_MARKER + json.dumps(data, ensure_ascii=False), flush=True)
        return 0 if data["probe_status"] == "PASS" else 1
    if not args.out:
        parser.error("--out is required")
    document = run_probe(args.host, args.python_executable, args.gpu_smoke,
                         args.device, workdir=args.workdir)
    try:
        write_report(document, args.out)
    except OSError as exc:
        print("FAILED: cannot write report: {}".format(exc), file=sys.stderr)
        return 1
    environment = document["environment"]
    smoke = environment["gpu"]["smoke"]["status"] if environment else "FAILED" if args.gpu_smoke else "NOT_RUN"
    print("{}: mode={} transport={} GPU_smoke={} report={}".format(
        document["status"], document["target"]["mode"], document["transport"]["status"], smoke, args.out))
    return 0 if document["status"] == "PASS" else 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
