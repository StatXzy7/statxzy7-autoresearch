import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import agent_loop
import research_run


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.prompt = self.root / "input.md"
        self.prompt.write_text("论文复现：执行一次有用迭代。", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def args(self, code="import sys; print(sys.stdin.read())", **overrides):
        values = dict(root=str(self.root), prompt=str(self.prompt), resume=None, max_rounds=2,
                      max_seconds=60, round_timeout=5, max_failures=3,
                      command=[sys.executable, "-c", code])
        values.update(overrides)
        return argparse.Namespace(**values)

    def resume_args(self, path, **overrides):
        values = dict(root=str(self.root), prompt=None, resume=str(path), max_rounds=None,
                      max_seconds=None, round_timeout=None, max_failures=None, command=[],
                      acknowledge_stopped_children=False)
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_bounded_stdin_prompt_unicode_and_no_completion_claim(self):
        result = agent_loop.run_loop(self.args())
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(result["stop_reason"], "ROUND_LIMIT")
        loop = Path(result["loop_dir"])
        stdout = (loop / "turn-0001/stdout.log").read_bytes().decode("utf-8", errors="replace")
        # The preserved prompt is independent of the child process console encoding.
        prompt = (loop / "turn-0001/prompt.md").read_text(encoding="utf-8")
        self.assertIn("论文复现", prompt)
        self.assertIn("research-plan.md", stdout)
        status = json.loads((loop / "turn-0001/status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["state"], "EXITED_ZERO")
        self.assertIn("not a model test", status["limitation"])
        self.assertFalse((loop / ".lock").exists())

    def test_failure_limit_and_timeout_receipts(self):
        failure = agent_loop.run_loop(self.args("import sys; sys.exit(4)", max_rounds=10, max_failures=2))
        self.assertEqual(failure["stop_reason"], "FAILURE_LIMIT")
        self.assertEqual(failure["rounds"], 2)
        timeout = agent_loop.run_loop(self.args("import time; time.sleep(20)", max_rounds=1, round_timeout=.1))
        receipt = json.loads((Path(timeout["loop_dir"]) / "turn-0001/status.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["state"], "TIMED_OUT")
        self.assertEqual(timeout["failures"], 1)

    def test_stop_boundary_resume_and_persistent_deadline(self):
        code = ("import sys; from pathlib import Path; p=Path(sys.argv[1]); "
                "(p.parent.parent/'STOP').touch(); print(p.read_text(encoding='utf-8'))")
        result = agent_loop.run_loop(self.args(command=[sys.executable, "-c", code, "{prompt_file}"], max_rounds=5))
        loop = Path(result["loop_dir"])
        self.assertEqual(result["stop_reason"], "STOP_FILE")
        self.assertEqual(result["rounds"], 1)
        resumed = agent_loop.run_loop(self.resume_args(loop, acknowledge_stopped_children=True))
        self.assertEqual(resumed["rounds"], 1)
        self.assertEqual(resumed["deadline"], result["deadline"])
        (loop / "STOP").unlink()
        state = json.loads((loop / "state.json").read_text(encoding="utf-8"))
        state["deadline"] = 1
        research_run.write_json(loop / "state.json", state)
        resumed = agent_loop.run_loop(self.resume_args(loop))
        self.assertEqual(resumed["stop_reason"], "TIME_BUDGET")
        self.assertEqual(resumed["rounds"], 1)

    def test_resume_cannot_expand_budget_and_concurrent_lock(self):
        result = agent_loop.run_loop(self.args(max_rounds=1))
        loop = Path(result["loop_dir"])
        with self.assertRaisesRegex(ValueError, "preserve"):
            agent_loop.run_loop(self.resume_args(loop, max_seconds=600))
        (loop / ".lock").write_text("1234\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "locked"):
            agent_loop.run_loop(self.resume_args(loop))

    def test_resume_preserves_uncertain_interrupted_attempt(self):
        result = agent_loop.run_loop(self.args(max_rounds=1))
        loop = Path(result["loop_dir"])
        state = json.loads((loop / "state.json").read_text(encoding="utf-8"))
        state["active_turn"] = "turn-0001"
        research_run.write_json(loop / "state.json", state)
        resumed = agent_loop.run_loop(self.resume_args(loop, acknowledge_stopped_children=True))
        self.assertEqual(resumed["rounds"], 1)
        self.assertEqual(resumed["failures"], 1)
        receipt = json.loads((loop / "turn-0001/status.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["state"], "UNKNOWN")

    def test_agent_can_request_stop_via_loop_environment(self):
        code = ("import os; from pathlib import Path; "
                "Path(os.environ['AUTORESEARCH_LOOP_DIR'],'STOP').write_text('no progress',encoding='utf-8')")
        result = agent_loop.run_loop(self.args(code, max_rounds=10))
        self.assertEqual(result["rounds"], 1)
        self.assertEqual(result["stop_reason"], "STOP_FILE")

    def interrupted_fixture(self):
        result = agent_loop.run_loop(self.args(max_rounds=1))
        loop = Path(result["loop_dir"])
        state = json.loads((loop / "state.json").read_text(encoding="utf-8"))
        state["active_turn"] = "turn-0001"
        research_run.write_json(loop / "state.json", state)
        return loop, state

    def test_resume_refuses_live_group_even_with_acknowledgment(self):
        loop, original = self.interrupted_fixture()
        receipt = (loop / "turn-0001/status.json").read_bytes()
        with mock.patch.object(agent_loop, "child_exists", return_value=True), mock.patch.object(agent_loop, "execute") as launch:
            with self.assertRaisesRegex(ValueError, "still exists"):
                agent_loop.run_loop(self.resume_args(loop, acknowledge_stopped_children=True))
            launch.assert_not_called()
        self.assertEqual((loop / "turn-0001/status.json").read_bytes(), receipt)
        self.assertEqual(json.loads((loop / "state.json").read_text(encoding="utf-8")), original)

    def test_missing_pid_requires_explicit_verified_recovery(self):
        loop, original = self.interrupted_fixture()
        receipt_path = loop / "turn-0001/status.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt.pop("child_pid")
        receipt.pop("child_process_group")
        research_run.write_json(receipt_path, receipt)
        with mock.patch.object(agent_loop, "execute") as launch:
            with self.assertRaisesRegex(ValueError, "acknowledge-stopped-children"):
                agent_loop.run_loop(self.resume_args(loop))
            launch.assert_not_called()
        result = agent_loop.run_loop(self.resume_args(loop, acknowledge_stopped_children=True))
        self.assertEqual(result["rounds"], original["rounds"])
        self.assertEqual(result["deadline"], original["deadline"])
        self.assertEqual(result["failures"], 1)
        self.assertEqual(json.loads(receipt_path.read_text(encoding="utf-8"))["state"], "UNKNOWN")

    def test_ended_child_can_recover_without_resetting_budget(self):
        loop, original = self.interrupted_fixture()
        with mock.patch.object(agent_loop, "child_exists", return_value=False):
            with self.assertRaisesRegex(ValueError, "acknowledge-stopped-children"):
                agent_loop.run_loop(self.resume_args(loop))
            result = agent_loop.run_loop(self.resume_args(loop, acknowledge_stopped_children=True))
        self.assertEqual(result["deadline"], original["deadline"])
        self.assertEqual(result["rounds"], original["rounds"])

    def test_crash_during_turn_setup_reserves_budget_and_resumes(self):
        for operation in ("mkdir", "write_text"):
            with self.subTest(operation=operation):
                parent = self.root / "agent-loops"
                existing = set(parent.iterdir()) if parent.exists() else set()
                original = getattr(Path, operation)

                def injected(path, *args, **kwargs):
                    if (operation == "mkdir" and path.name == "turn-0001") or (
                            operation == "write_text" and path.parent.name == "turn-0001"):
                        raise OSError("injected turn setup interruption")
                    return original(path, *args, **kwargs)

                with mock.patch.object(Path, operation, autospec=True, side_effect=injected):
                    with self.assertRaisesRegex(OSError, "injected"):
                        agent_loop.run_loop(self.args())
                loop = (set(parent.iterdir()) - existing).pop()
                state = json.loads((loop / "state.json").read_text(encoding="utf-8"))
                self.assertEqual(state["rounds"], 1)
                self.assertEqual(state["active_turn"], "turn-0001")
                deadline = state["deadline"]
                result = agent_loop.run_loop(self.resume_args(loop))
                self.assertEqual(result["rounds"], 2)
                self.assertEqual(result["failures"], 1)
                self.assertEqual(result["deadline"], deadline)
                unknown = json.loads((loop / "turn-0001/status.json").read_text(encoding="utf-8"))
                self.assertEqual(unknown["state"], "UNKNOWN")
                self.assertTrue((loop / "turn-0002/status.json").is_file())


if __name__ == "__main__":
    unittest.main()
