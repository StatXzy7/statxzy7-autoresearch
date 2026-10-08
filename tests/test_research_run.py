import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import research_run


class ResearchRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def trial(self, run_id="one", kind="baseline", value=1.0, seed=7, protocol="frozen-v1", code=None):
        if code is None:
            metrics = {"metric": "loss", "value": value, "seed": seed, "protocol": protocol, "artifacts": ["evidence.txt"]}
            code = ("import json, os; from pathlib import Path; "
                    "p=Path(os.environ['AUTORESEARCH_RUN_DIR']); "
                    "(p/'evidence.txt').write_text('actual evaluation',encoding='utf-8'); "
                    "(p/'metrics.json').write_text(" + repr(json.dumps(metrics)) + ",encoding='utf-8')")
        args = argparse.Namespace(root=str(self.root), id=run_id, kind=kind, timeout=5,
                                  metric="loss", direction="min", protocol="frozen-v1", seed=7,
                                  command=[sys.executable, "-c", code])
        return research_run.run_trial(args)

    def test_success_and_duplicate_id_preserves_receipt(self):
        status = self.trial()
        self.assertEqual(status["state"], "SUCCESS")
        self.assertEqual(status["value"], 1.0)
        self.assertEqual(len(status["metrics_sha256"]), 64)
        receipt = (self.root / "runs/one/status.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.trial()
        self.assertEqual((self.root / "runs/one/status.json").read_bytes(), receipt)

    def test_nonzero_and_missing_executable_preserve_failure(self):
        for run_id, code in (("nonzero", "import sys; print('failure evidence'); sys.exit(2)"),
                             ("missing", "")):
            if run_id == "missing":
                args = argparse.Namespace(root=str(self.root), id=run_id, kind="baseline", timeout=1,
                                          metric="loss", direction="min", protocol="p", seed=0,
                                          command=[str(self.root / "absent-executable")])
                status = research_run.run_trial(args)
            else:
                status = self.trial(run_id=run_id, code=code)
            self.assertEqual(status["state"], "FAILED")
            self.assertIsNone(status["value"])
            self.assertTrue((self.root / "runs" / run_id / "stderr.log").exists())

    def test_timeout_has_no_metric(self):
        args = argparse.Namespace(root=str(self.root), id="timeout", kind="candidate", timeout=.1,
                                  metric="loss", direction="min", protocol="p", seed=0,
                                  command=[sys.executable, "-c", "import time; time.sleep(30)"])
        status = research_run.run_trial(args)
        self.assertEqual(status["state"], "TIMED_OUT")
        self.assertIsNone(status["value"])

    def test_failed_pid_receipt_terminates_owned_child(self):
        process = mock.Mock(pid=12345)
        with mock.patch.object(research_run.subprocess, "Popen", return_value=process), \
                mock.patch.object(research_run, "terminate_tree") as terminate:
            def failed_receipt(pid):
                raise OSError("receipt write failed")
            with self.assertRaisesRegex(OSError, "receipt write failed"):
                research_run.execute(["agent"], self.root, None, None, 1, on_started=failed_receipt)
            terminate.assert_called_once_with(process)
            process.communicate.assert_not_called()

    def test_group_liveness_distinguishes_zombies_and_fails_closed(self):
        proc = mock.Mock()
        proc.iterdir.return_value = [argparse.Namespace(name="101"), argparse.Namespace(name="102")]
        with mock.patch.object(research_run, "Path", return_value=proc), \
                mock.patch.object(research_run, "_linux_process_info") as read:
            read.side_effect = [(1, "start", "Z", 42), (1, "start", "X", 42)]
            self.assertFalse(research_run._linux_group_has_live_members(42))
            read.side_effect = [(1, "start", "Z", 42), (1, "start", "S", 42)]
            self.assertTrue(research_run._linux_group_has_live_members(42))
            read.side_effect = [(1, "start", "Z", 42), None]
            self.assertTrue(research_run._linux_group_has_live_members(42))

    @unittest.skipUnless(sys.platform.startswith("linux"), "Requires Linux process ancestry and sessions")
    def test_timeout_terminates_nested_child_in_independent_session(self):
        pid_file = self.root / "nested-pid.txt"
        leaf = ("import os,time; from pathlib import Path; "
                "Path(" + repr(str(pid_file)) + ").write_text(str(os.getpid()),encoding='utf-8'); time.sleep(30)")
        outer = ("import subprocess,sys; p=subprocess.Popen([sys.executable,'-c'," + repr(leaf) +
                 "],start_new_session=True); p.wait()")
        leaf_pid = None
        try:
            with (self.root / "outer.log").open("wb") as out:
                _, timed_out = research_run.execute([sys.executable, "-c", outer], self.root, out, out, 1)
            self.assertTrue(timed_out)
            leaf_pid = int(pid_file.read_text(encoding="utf-8"))
            deadline = time.monotonic() + 2
            info = research_run._linux_process_info(leaf_pid)
            while info is not None and info[2] != "Z" and time.monotonic() < deadline:
                time.sleep(.02)
                info = research_run._linux_process_info(leaf_pid)
            self.assertTrue(info is None or info[2] == "Z", "Nested session survived the outer timeout")
        finally:
            if leaf_pid is not None:
                info = research_run._linux_process_info(leaf_pid)
                if info is not None and info[2] != "Z":
                    research_run._signal_descendants({leaf_pid: info[1]}, research_run.signal.SIGKILL)

    def test_invalid_result_values_and_identity(self):
        for index, value in enumerate((True, float("nan"), float("inf"), "0.5")):
            self.assertEqual(self.trial(run_id="value%d" % index, value=value)["state"], "FAILED")
        self.assertEqual(self.trial(run_id="seed", seed=8)["state"], "FAILED")
        self.assertEqual(self.trial(run_id="protocol", protocol="changed")["state"], "FAILED")
        self.assertEqual(self.trial(run_id="absent", code="pass")["state"], "FAILED")

    def test_stale_missing_empty_and_escape_artifacts(self):
        run = self.root / "fixture"
        run.mkdir()
        (run / "evidence").write_text("ok", encoding="utf-8")
        expected = {"metric": "loss", "protocol": "p", "seed": 0}
        metrics = {**expected, "value": 1, "artifacts": ["evidence"]}
        research_run.write_json(run / "metrics.json", metrics)
        os.utime(run / "metrics.json", (1, 1))
        with self.assertRaisesRegex(ValueError, "predates"):
            research_run.validate_result(run, expected, started_ns=2_000_000_000)
        for artifact in ("missing", "empty", "../outside"):
            (run / "empty").touch()
            (self.root / "outside").write_text("outside", encoding="utf-8")
            metrics["artifacts"] = [artifact]
            research_run.write_json(run / "metrics.json", metrics)
            with self.assertRaises(ValueError):
                research_run.validate_result(run, expected)
        (run / "stdout.log").write_text("not an evaluator artifact", encoding="utf-8")
        metrics["artifacts"] = ["./stdout.log"]
        research_run.write_json(run / "metrics.json", metrics)
        with self.assertRaisesRegex(ValueError, "receipts"):
            research_run.validate_result(run, expected)

    def test_summary_comparability_heldout_and_hash_validation(self):
        self.trial("baseline")
        self.trial("candidate", kind="candidate", value=.5)
        self.trial("final", kind="final", value=.01)
        # A successful run with a different seed must not enter this comparison.
        self.trial("other", kind="candidate", value=.1)
        status_path = self.root / "runs/other/status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["seed"] = 8
        metrics_path = self.root / "runs/other/metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["seed"] = 8
        research_run.write_json(metrics_path, metrics)
        status["metrics_sha256"] = research_run.sha256(metrics_path)
        research_run.write_json(status_path, status)
        report = research_run.summary(self.root)
        self.assertEqual(len(report["groups"]), 1)
        self.assertEqual(report["groups"][0]["best_candidate"]["id"], "candidate")
        self.assertEqual(report["excluded_final_runs"], ["final"])
        (self.root / "runs/candidate/evidence.txt").write_text("changed", encoding="utf-8")
        report = research_run.summary(self.root)
        self.assertEqual(report["groups"], [])
        self.assertEqual(report["invalid_runs"][0]["id"], "candidate")

    def test_reference_and_upper_bound_are_distinct_from_agent_candidates(self):
        self.trial("baseline")
        self.trial("reference", kind="reference", value=.3)
        self.trial("upper", kind="upper_bound", value=.01)
        self.trial("candidate", kind="candidate", value=.5)
        report = research_run.summary(self.root)
        self.assertEqual(report["groups"][0]["best_candidate"]["id"], "candidate")
        self.assertEqual(report["groups"][0]["reference_runs"], [{"id": "reference", "value": .3}])
        self.assertEqual(report["groups"][0]["upper_bound_runs"], [{"id": "upper", "value": .01}])
        self.assertEqual(report["invalid_runs"], [])

    def test_result_environment_and_split_binding(self):
        code = ("import json,os; from pathlib import Path; "
                "p=Path(os.environ['AUTORESEARCH_RUN_DIR']); "
                "assert Path(os.environ['AUTORESEARCH_RESULT'])==p/'metrics.json'; "
                "(p/'evidence.txt').write_text('预测 evidence',encoding='utf-8'); "
                "(p/'metrics.json').write_text(json.dumps(dict(metric='loss',protocol='p',seed=0,"
                "split='dev-v1',value=.5,artifacts=['evidence.txt'])),encoding='utf-8')")
        args = argparse.Namespace(root=str(self.root), id="split", kind="reference", timeout=5,
                                  metric="loss", direction="min", protocol="p", seed=0, split="dev-v1",
                                  command=[sys.executable, "-c", code])
        status = research_run.run_trial(args)
        self.assertEqual(status["state"], "SUCCESS")
        self.assertEqual(status["split"], "dev-v1")
        metrics_path = self.root / "runs/split/metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["split"] = "heldout"
        research_run.write_json(metrics_path, metrics)
        with self.assertRaisesRegex(ValueError, "split"):
            research_run.validate_result(self.root / "runs/split", status)

    def test_malformed_status_does_not_abort_other_results(self):
        self.trial("baseline")
        self.trial("candidate", kind="candidate", value=.5)
        for name, value in (("broken-array", []), ("broken-string", "broken")):
            run = self.root / "runs" / name
            run.mkdir()
            research_run.write_json(run / "status.json", value)
        report = research_run.summary(self.root)
        self.assertEqual(report["groups"][0]["best_candidate"]["id"], "candidate")
        self.assertEqual({item["id"] for item in report["invalid_runs"]},
                         {"broken-array", "broken-string"})


if __name__ == "__main__":
    unittest.main()
