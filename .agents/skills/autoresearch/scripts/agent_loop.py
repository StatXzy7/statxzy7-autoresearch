#!/usr/bin/env python3
"""Bounded supervisor for an explicitly supplied agent command in a container.

A successful runner process is not evidence of a model call, valid research
metrics, or research completion. No vendor CLI, credentials, or login is assumed.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

from research_run import execute, write_json, _linux_group_has_live_members


def positive(value, label):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(label + " must be finite and positive")
    return value


def child_exists(receipt):
    """Read-only liveness check; never signal or kill an unverified/reused PID."""
    pid = receipt.get("child_pid")
    if type(pid) is not int or pid <= 1:
        return None
    if os.name != "nt":
        group = receipt.get("child_process_group")
        if type(group) is not int or group <= 1:
            return None
        try:
            os.killpg(group, 0)
            return _linux_group_has_live_members(group) if sys.platform.startswith("linux") else True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False if ctypes.get_last_error() == 87 else True
    try:
        exit_code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True  # Fail closed when the process cannot be queried.
        return exit_code.value == 259  # STILL_ACTIVE
    finally:
        kernel.CloseHandle(handle)


def initialize(args):
    root = Path(args.root).resolve(strict=True)
    if args.resume:
        loop = Path(args.resume).resolve(strict=True)
        state = json.loads((loop / "state.json").read_text(encoding="utf-8"))
        if state["root"] != str(root):
            raise ValueError("Resume root differs from the original root")
        for name in ("max_rounds", "max_seconds", "round_timeout", "max_failures"):
            override = getattr(args, name)
            if override is not None and override != state[name]:
                raise ValueError("Resume must preserve the original budget: " + name)
        if args.command and args.command != state["command"]:
            raise ValueError("Resume must preserve the original agent command")
        if args.prompt and Path(args.prompt).read_text(encoding="utf-8") != (loop / "prompt.md").read_text(encoding="utf-8"):
            raise ValueError("Resume must preserve the original prompt")
        return loop, state
    if not args.prompt or not args.command:
        raise ValueError("A prompt file and explicit agent command are required")
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    budget = {"max_rounds": args.max_rounds if args.max_rounds is not None else 10,
              "max_seconds": args.max_seconds if args.max_seconds is not None else 3600,
              "round_timeout": args.round_timeout if args.round_timeout is not None else 600,
              "max_failures": args.max_failures if args.max_failures is not None else 3}
    for key, value in budget.items():
        positive(value, key)
    loop = root / "agent-loops" / (time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:8])
    loop.mkdir(parents=True)
    (loop / "prompt.md").write_text(prompt, encoding="utf-8")
    now = time.time()
    state = {"schema_version": 1, "root": str(root), "created_at": now,
             "deadline": now + budget["max_seconds"], "command": args.command, **budget,
             "rounds": 0, "failures": 0, "phase": "READY", "stop_reason": None,
             "limitation": "Agent-runner smoke is not a model test or research result validation."}
    write_json(loop / "state.json", state)
    return loop, state


def run_loop(args):
    loop, state = initialize(args)
    lock = loop / ".lock"
    try:
        fd = os.open(str(lock), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise ValueError("Loop is locked; verify the owning process before removing a stale .lock")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(str(os.getpid()) + "\n")
        # Reload after locking: another supervisor may have completed since initialize.
        state = json.loads((loop / "state.json").read_text(encoding="utf-8"))
        if state.get("active_turn"):
            previous = loop / state["active_turn"] / "status.json"
            previous.parent.mkdir(exist_ok=True)
            if previous.exists():
                receipt = json.loads(previous.read_text(encoding="utf-8"))
            else:
                receipt = {"returncode": None, "limitation": state["limitation"]}
            if receipt.get("launch_requested") or "started_at" in receipt:
                alive = child_exists(receipt)
                if alive:
                    raise ValueError("Interrupted turn's process/group still exists or cannot be verified stopped; "
                                     "inspect its recorded PID and descendants before resuming")
                # An absent parent/group cannot prove nested detached groups are gone.
                # Every interrupted launch needs independent descendant verification.
                if not getattr(args, "acknowledge_stopped_children", False):
                    raise ValueError("Cannot prove interrupted children stopped; after verifying all old children "
                                     "are stopped, resume with --acknowledge-stopped-children")
            # An interrupted supervisor did not commit the previous attempt's outcome.
            # Preserve that uncertainty and charge the reserved attempt to the retry limit.
            receipt.update(state="UNKNOWN", error="Supervisor stopped before committing the round outcome",
                           recovered_at=time.time())
            write_json(previous, receipt)
            state["failures"] += 1
            state.pop("active_turn")
        root = Path(state["root"])
        state.update(phase="ACTIVE", stop_reason=None)
        write_json(loop / "state.json", state)
        while True:
            if (loop / "STOP").exists():
                reason = "STOP_FILE"
            elif time.time() >= state["deadline"]:
                reason = "TIME_BUDGET"
            elif state["rounds"] >= state["max_rounds"]:
                reason = "ROUND_LIMIT"
            elif state["failures"] >= state["max_failures"]:
                reason = "FAILURE_LIMIT"
            else:
                reason = None
            if reason:
                break
            turn = loop / ("turn-%04d" % (state["rounds"] + 1))
            # Reserve before any turn mutation: a crash during mkdir/prompt/receipt
            # creation must remain resumable and consume the original budget.
            state["rounds"] += 1
            state["active_turn"] = turn.name
            write_json(loop / "state.json", state)
            turn.mkdir()  # Never overwrite a turn left by an interrupted supervisor.
            instruction = (loop / "prompt.md").read_text(encoding="utf-8")
            instruction += ("\n\nPerform one useful research iteration within the current budget. Read "
                            ".agents/skills/autoresearch/SKILL.md, research-plan.md, resume.md, "
                            "results.tsv and runs/*/status.json when present. Preserve evaluator "
                            "protocol and held-out boundaries. Record failures. Before exiting, "
                            "write resume.md in the research root with evidence, decisions and "
                            "the next useful action. If the objective is done or no useful progress "
                            "is possible, create STOP in AUTORESEARCH_LOOP_DIR and explain why in "
                            "resume.md. Process exit 0 alone is not research completion.\n")
            prompt_path = turn / "prompt.md"
            prompt_path.write_text(instruction, encoding="utf-8")
            template = state["command"]
            uses_file = any("{prompt_file}" in item for item in template)
            command = [item.replace("{prompt_file}", str(prompt_path)) for item in template]
            timeout = min(state["round_timeout"], max(0.001, state["deadline"] - time.time()))
            status = {"state": "RUNNING", "started_at": time.time(), "returncode": None,
                      "error": None, "limitation": state["limitation"], "launch_requested": True}
            write_json(turn / "command.json", {"argv": command, "cwd": str(root), "timeout": timeout,
                                                 "prompt_mode": "file" if uses_file else "stdin"})
            write_json(turn / "status.json", status)

            def record_child(pid):
                status.update(child_pid=pid, child_process_group=pid)
                write_json(turn / "status.json", status)

            try:
                env = os.environ.copy()
                env.update(AUTORESEARCH_LOOP_DIR=str(loop), AUTORESEARCH_TURN_DIR=str(turn))
                with (turn / "stdout.log").open("wb") as out, (turn / "stderr.log").open("wb") as err:
                    code, timed_out = execute(command, root, out, err, timeout, env=env,
                                              input_text=None if uses_file else instruction, on_started=record_child)
                status.update(returncode=code, state="TIMED_OUT" if timed_out else ("EXITED_ZERO" if code == 0 else "FAILED"))
                if timed_out:
                    status["error"] = "Agent round exceeded its timeout"
                elif code != 0:
                    status["error"] = "Agent command returned a nonzero exit code"
            except KeyboardInterrupt:
                status.update(state="FAILED", error="Interrupted by caller")
                reason = "INTERRUPTED"
            except OSError as exc:
                status.update(state="FAILED", error=str(exc))
            status["finished_at"] = time.time()
            write_json(turn / "status.json", status)
            if status["state"] != "EXITED_ZERO":
                state["failures"] += 1
            state.pop("active_turn", None)
            write_json(loop / "state.json", state)
            if reason == "INTERRUPTED":
                break
        state.update(phase="STOPPED", stop_reason=reason, updated_at=time.time())
        write_json(loop / "state.json", state)
        return {"loop_dir": str(loop), **state}
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--resume")
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--round-timeout", type=float)
    parser.add_argument("--max-failures", type=int)
    parser.add_argument("--acknowledge-stopped-children", action="store_true",
                        help="Attest all old child processes stopped after independently verifying an interrupted launch")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    try:
        result = run_loop(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
