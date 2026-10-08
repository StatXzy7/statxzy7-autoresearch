#!/usr/bin/env python3
"""Execute one research trial and bind its result to evaluator artifacts.

The caller must snapshot source/configuration in the run's artifacts. This tool
does not infer code provenance or scientific completion from a process exit.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

RUN_KINDS = ("baseline", "reference", "upper_bound", "candidate", "final")


def write_json(path, value):
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _linux_process_info(pid):
    try:
        stat = Path("/proc", str(pid), "stat").read_text(encoding="utf-8")
        fields = stat[stat.rindex(")") + 2:].split()
        return int(fields[1]), fields[19], fields[0], int(fields[2])  # parent, starttime, state, group
    except (OSError, ValueError, IndexError):
        return None


def _linux_group_has_live_members(group):
    """Fail closed on uncertain membership; zombie-only groups are ended."""
    states, unreadable = [], False
    try:
        for path in Path("/proc").iterdir():
            if not path.name.isdigit():
                continue
            info = _linux_process_info(int(path.name))
            if info is None:
                unreadable = True
            elif info[3] == group:
                states.append(info[2])
    except OSError:
        return True
    return unreadable or not states or any(state not in ("Z", "X") for state in states)


def _visible_descendants(root_pid):
    """Snapshot Linux descendants, including children that created new sessions."""
    if not sys.platform.startswith("linux"):
        return {}
    try:
        processes = {int(path.name): _linux_process_info(int(path.name))
                     for path in Path("/proc").iterdir() if path.name.isdigit()}
    except OSError:
        return {}
    owned = {root_pid}
    while True:
        additions = {pid for pid, info in processes.items() if info and info[0] in owned} - owned
        if not additions:
            break
        owned.update(additions)
    return {pid: processes[pid][1] for pid in owned if pid != root_pid and processes.get(pid)}


def _signal_descendants(identities, sig):
    for pid, started in identities.items():
        current = _linux_process_info(pid)
        if current is None or current[1] != started:
            continue  # Never signal a recycled PID.
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def terminate_tree(process):
    """Terminate the own group plus visible Linux descendants across sessions.

    Fully detached/reparented daemons cannot be discovered by ancestry alone.
    """
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        descendants = _visible_descendants(process.pid)
        _signal_descendants(descendants, signal.SIGTERM)
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        time.sleep(0.2)
        _signal_descendants(descendants, signal.SIGKILL)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def execute(argv, cwd, stdout, stderr, timeout, env=None, input_text=None, on_started=None):
    kwargs = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    process = subprocess.Popen(argv, cwd=str(cwd), shell=False, env=env,
                               stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                               stdout=stdout, stderr=stderr, **kwargs)
    try:
        if on_started is not None:
            on_started(process.pid)
        process.communicate(input_text.encode("utf-8") if input_text is not None else None, timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        terminate_tree(process)
        if isinstance(exc, KeyboardInterrupt):
            raise
        return process.returncode, True
    except BaseException:
        # Receipt failures must not leave an untracked child running.
        terminate_tree(process)
        raise
    return process.returncode, False


def inside(run, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Artifact paths must be relative to the run directory")
    resolved = (run / relative).resolve()
    try:
        resolved.relative_to(run.resolve())
    except ValueError:
        raise ValueError("Artifact escapes the run directory: " + relative)
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ValueError("Artifact must be a nonempty file: " + relative)
    return resolved


def validate_result(run, expected, started_ns=None):
    metrics_path = inside(run, "metrics.json")
    if started_ns is not None and metrics_path.stat().st_mtime_ns < started_ns:
        raise ValueError("metrics.json predates the trial")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict):
        raise ValueError("Metrics must be a JSON object")
    fields = ("metric", "protocol", "seed", "split") if "split" in expected else ("metric", "protocol", "seed")
    for field in fields:
        if type(metrics.get(field)) is not type(expected[field]) or metrics.get(field) != expected[field]:
            raise ValueError("Metrics identity mismatch: " + field)
    value = metrics.get("value")
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Metric value must be a finite number, excluding bool")
    names = metrics.get("artifacts")
    if not isinstance(names, list) or not names:
        raise ValueError("At least one evaluator artifact is required")
    artifacts = []
    runner_files = {(run / name).resolve() for name in
                    ("metrics.json", "status.json", "command.json", "stdout.log", "stderr.log")}
    resolved_artifacts = set()
    for name in names:
        path = inside(run, name)
        if path in runner_files:
            raise ValueError("Runner receipts are not evaluator artifacts")
        if path in resolved_artifacts:
            raise ValueError("Artifact paths must identify distinct files")
        resolved_artifacts.add(path)
        artifacts.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return value, artifacts, sha256(metrics_path)


def run_trial(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.id or ""):
        raise ValueError("Run ID must be a safe single directory name")
    if not args.command or args.timeout is None or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("A command and finite positive timeout are required")
    for name in ("kind", "metric", "direction", "protocol", "seed"):
        if getattr(args, name) is None:
            raise ValueError("Missing --" + name)
    if args.kind not in RUN_KINDS or args.direction not in ("min", "max"):
        raise ValueError("Invalid run kind or direction")
    root = Path(args.root).resolve(strict=True)
    run = root / "runs" / args.id
    run.parent.mkdir(exist_ok=True)
    run.mkdir()  # Exclusive: never overwrite an earlier trial, including failures.
    started_ns = time.time_ns()
    status = {"schema_version": 1, "id": args.id, "kind": args.kind, "metric": args.metric,
              "direction": args.direction, "protocol": args.protocol, "seed": args.seed,
              "state": "RUNNING", "started_at": started_ns / 1e9, "value": None,
              "returncode": None, "artifacts": [], "metrics_sha256": None, "error": None}
    if getattr(args, "split", None) is not None:
        status["split"] = args.split
    write_json(run / "command.json", {"argv": args.command, "cwd": str(root), "timeout": args.timeout})
    write_json(run / "status.json", status)
    env = os.environ.copy()
    env.update(AUTORESEARCH_RUN_DIR=str(run), AUTORESEARCH_RESULT=str(run / "metrics.json"))
    try:
        with (run / "stdout.log").open("wb") as out, (run / "stderr.log").open("wb") as err:
            code, timed_out = execute(args.command, root, out, err, args.timeout, env=env)
        status["returncode"] = code
        if timed_out:
            status.update(state="TIMED_OUT", error="Trial exceeded its timeout")
        elif code != 0:
            status.update(state="FAILED", error="Command returned a nonzero exit code")
        else:
            value, artifacts, binding = validate_result(run, status, started_ns)
            status.update(state="SUCCESS", value=value, artifacts=artifacts, metrics_sha256=binding)
    except KeyboardInterrupt:
        status.update(state="FAILED", error="Interrupted by caller")
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        status.update(state="FAILED", error=str(exc))
    status["finished_at"] = time.time()
    status["wall_seconds"] = status["finished_at"] - status["started_at"]
    write_json(run / "status.json", status)
    return status


def summary(root):
    groups, invalid, finals = {}, [], []
    runs = Path(root).resolve() / "runs"
    for run in sorted(runs.iterdir()) if runs.exists() else []:
        if not run.is_dir():
            continue
        try:
            status = json.loads((run / "status.json").read_text(encoding="utf-8"))
            if not isinstance(status, dict):
                raise ValueError("Recorded status must be a JSON object")
            if status.get("state") != "SUCCESS":
                continue
            if (type(status.get("seed")) is not int or type(status.get("value")) not in (int, float)
                    or any(type(status.get(field)) is not str for field in ("metric", "protocol"))):
                raise ValueError("Invalid recorded result identity or value")
            value, artifacts, binding = validate_result(run, status)
            if value != status["value"] or artifacts != status["artifacts"] or binding != status["metrics_sha256"]:
                raise ValueError("Result binding changed after execution")
            if status.get("kind") not in RUN_KINDS or status.get("direction") not in ("min", "max"):
                raise ValueError("Invalid run kind or direction")
            if status["kind"] == "final":
                finals.append(run.name)
                continue
            key = (status["metric"], status["direction"], status["protocol"], status["seed"], status.get("split"))
            group = groups.setdefault(key, {"metric": key[0], "direction": key[1], "protocol": key[2],
                                            "seed": key[3], "split": key[4], "baseline_runs": [],
                                            "reference_runs": [], "upper_bound_runs": [], "candidate_runs": []})
            group[status["kind"] + "_runs"].append({"id": run.name, "value": value})
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            invalid.append({"id": run.name, "error": str(exc)})
    comparable = []
    for group in groups.values():
        if group["baseline_runs"] and group["candidate_runs"]:
            choose = min if group["direction"] == "min" else max
            group["best_candidate"] = choose(group["candidate_runs"], key=lambda item: item["value"])
            group["comparison_note"] = "Descriptive observations only; this does not establish reproducible improvement."
            comparable.append(group)
    return {"groups": comparable, "anchor_groups": list(groups.values()),
            "invalid_runs": invalid, "excluded_final_runs": finals}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--id")
    parser.add_argument("--kind", choices=RUN_KINDS)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--metric")
    parser.add_argument("--direction", choices=("min", "max"))
    parser.add_argument("--protocol")
    parser.add_argument("--split", help="Fixed evaluation split identity; required for calibration")
    parser.add_argument("--seed", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    try:
        result = summary(args.root) if args.summary else run_trial(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if args.summary or result["state"] == "SUCCESS" else 1


if __name__ == "__main__":
    sys.exit(main())
