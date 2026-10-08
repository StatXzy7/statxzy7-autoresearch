import argparse
from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import calibrate
import research_run


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "u-basis.md").write_text("可达目标的资源、方法和噪声依据", encoding="utf-8")
        self.spec = {"metric": "loss", "direction": "min", "protocol": "frozen-v1", "split": "dev-v1",
                     "stochastic": True, "seeds": [1, 2, 3],
                     "baseline_runs": ["b1", "b2", "b3"], "reference_runs": ["r1", "r2", "r3"],
                     "upper_bound": {"source": "estimated_attainable_upper_bound", "value": 0.0,
                                     "basis": "A known method under the allocated budget", "evidence_paths": ["u-basis.md"],
                                     "attainability": {"mode": "empirical", "run_ids": ["u1", "u2", "u3"]}}}
        for kind, prefix, values in (("baseline", "b", [9.9, 10.0, 10.1]),
                                     ("reference", "r", [4.9, 5.0, 5.1]), ("upper_bound", "u", [0, 0, 0])):
            for seed, value in zip(self.spec["seeds"], values):
                self.make_run(prefix + str(seed), kind, seed, value)

    def tearDown(self):
        self.temporary.cleanup()

    def make_run(self, run_id, kind, seed, value, direction="min", split="dev-v1", protocol="frozen-v1", code=None):
        if code is None:
            metrics = {"metric": "loss", "protocol": protocol, "split": split, "seed": seed,
                       "value": value, "artifacts": ["prediction.txt", "configuration.json"]}
            code = ("import os; from pathlib import Path; p=Path(os.environ['AUTORESEARCH_RUN_DIR']); "
                    "(p/'prediction.txt').write_text('evaluated prediction',encoding='utf-8'); "
                    "(p/'configuration.json').write_text('{\"method\":\"fixture\"}',encoding='utf-8'); "
                    "(p/'metrics.json').write_text(" + repr(json.dumps(metrics)) + ",encoding='utf-8')")
        args = argparse.Namespace(root=str(self.root), id=run_id, kind=kind, timeout=5, metric="loss",
                                  direction=direction, protocol=protocol, split=split, seed=seed,
                                  command=[sys.executable, "-c", code])
        return research_run.run_trial(args)

    def report(self, spec=None):
        return calibrate.calibrate(self.root, self.spec if spec is None else spec)

    def test_empirical_calibration_aggregates_every_seed(self):
        result = self.report()
        self.assertEqual(result["status"], "CALIBRATED")
        self.assertEqual(result["baseline"]["mean"], 10)
        self.assertEqual(result["reference"]["mean"], 5)
        self.assertAlmostEqual(result["baseline"]["sample_std"], .1)
        self.assertEqual(result["normalized"], {"baseline": 0., "reference": .5, "upper_bound": 1.})
        self.assertEqual(result["upper_bound"]["attainability"], "DEMONSTRATED")
        self.assertTrue(result["requires_semantic_review"])
        self.assertEqual([record["seed"] for record in result["reference"]["records"]], [1, 2, 3])
        self.assertEqual(len(result["upper_bound"]["evidence"][0]["sha256"]), 64)

    def test_normalization_min_max_and_no_clipping(self):
        for direction, baseline, upper, weak, beyond in (("min", 10, 0, 11, -1), ("max", 0, 10, -1, 11)):
            self.assertEqual(calibrate.normalize(baseline, baseline, upper, direction), 0)
            self.assertEqual(calibrate.normalize(upper, baseline, upper, direction), 1)
            self.assertLess(calibrate.normalize(weak, baseline, upper, direction), 0)
            self.assertGreater(calibrate.normalize(beyond, baseline, upper, direction), 1)
        for upper in (10, 11):
            with self.assertRaisesRegex(ValueError, "strictly better"):
                calibrate.normalize(5, 10, upper, "min")

    def test_max_direction_calibration(self):
        spec = copy.deepcopy(self.spec)
        spec["direction"] = "max"
        spec["upper_bound"]["value"] = 10
        for kind, prefix, values in (("baseline", "maxb", [-.1, 0, .1]),
                                     ("reference", "maxr", [4.9, 5, 5.1]), ("upper_bound", "maxu", [10, 10, 10])):
            ids = []
            for seed, value in zip(spec["seeds"], values):
                run_id = prefix + str(seed)
                ids.append(run_id)
                self.make_run(run_id, kind, seed, value, direction="max")
            if kind == "upper_bound":
                spec["upper_bound"]["attainability"]["run_ids"] = ids
            else:
                spec[kind + "_runs"] = ids
        result = self.report(spec)
        self.assertEqual(result["status"], "CALIBRATED")
        self.assertEqual(result["normalized"]["reference"], .5)

    def test_prefilled_raw_values_are_ignored(self):
        self.spec["raw_values"] = {"baseline": 1000, "reference": -1000}
        self.spec["upper_bound"]["attainability"]["raw_values"] = [-1000]
        self.assertEqual(self.report()["reference"]["mean"], 5)

    def test_missing_or_failed_run_never_yields_partial_mean(self):
        self.spec["reference_runs"][2] = "missing"
        result = self.report()
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertIsNone(result["reference"]["mean"])
        self.assertNotIn("normalized", result)
        self.make_run("failed", "reference", 3, 5, code="import sys; sys.exit(2)")
        self.spec["reference_runs"][2] = "failed"
        result = self.report()
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(next(issue["run_state"] for issue in result["issues"] if "run_state" in issue), "FAILED")

    def test_metrics_and_artifact_tampering_fail(self):
        for relative in ("prediction.txt", "metrics.json"):
            path = self.root / "runs/r1" / relative
            original = path.read_bytes()
            if relative == "metrics.json":
                value = json.loads(path.read_text(encoding="utf-8"))
                value["value"] = 1
                research_run.write_json(path, value)
            else:
                path.write_text("tampered", encoding="utf-8")
            self.assertEqual(self.report()["status"], "FAIL")
            path.write_bytes(original)
        self.assertEqual(self.report()["status"], "CALIBRATED")

    def test_malformed_success_receipts_fail(self):
        path = self.root / "runs/r1/status.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        for altered in ([], {**original, "returncode": 2}, {**original, "value": True}):
            research_run.write_json(path, altered)
            self.assertEqual(self.report()["status"], "FAIL")
        research_run.write_json(path, original)

    def test_empirical_upper_run_missing_failure_or_role_mismatch(self):
        ids = self.spec["upper_bound"]["attainability"]["run_ids"]
        ids[2] = "u-missing"
        self.assertEqual(self.report()["status"], "INCOMPLETE")
        self.make_run("u-failed", "upper_bound", 3, 0, code="import sys; sys.exit(2)")
        ids[2] = "u-failed"
        self.assertEqual(self.report()["status"], "INCOMPLETE")
        ids[2] = "r3"
        self.assertEqual(self.report()["status"], "FAIL")

    def test_protocol_split_seed_and_role_mismatch_fail(self):
        for run_id, kind, seed, protocol, split in (("badprotocol", "reference", 3, "other", "dev-v1"),
                                                   ("badsplit", "reference", 3, "frozen-v1", "test"),
                                                   ("badseed", "reference", 99, "frozen-v1", "dev-v1"),
                                                   ("badrole", "candidate", 3, "frozen-v1", "dev-v1"),
                                                   ("duplicate_seed", "reference", 1, "frozen-v1", "dev-v1")):
            self.make_run(run_id, kind, seed, 5, protocol=protocol, split=split)
            self.spec["reference_runs"][2] = run_id
            self.assertEqual(self.report()["status"], "FAIL", run_id)

    def test_empirical_u_must_match_receipts(self):
        self.spec["upper_bound"]["value"] = -1
        result = self.report()
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["upper_bound"]["attainability"], "NOT_PROVEN")

    def test_estimated_u_retains_strength_and_optional_demonstration(self):
        self.spec["upper_bound"]["attainability"] = {"mode": "estimate"}
        result = self.report()
        self.assertEqual(result["status"], "CALIBRATED")
        self.assertEqual(result["upper_bound"]["attainability"], "ESTIMATED")
        self.spec["require_demonstrated_u"] = True
        self.assertEqual(self.report()["status"], "INCOMPLETE")

    def test_metric_ceiling_and_theory_do_not_prove_reachability(self):
        upper = self.spec["upper_bound"]
        upper.update(source="metric_ceiling", estimate_unavailable="Insufficient known-method resource evidence",
                     attainability={"mode": "theoretical"})
        result = self.report()
        self.assertEqual(result["status"], "CALIBRATED")
        self.assertEqual(result["upper_bound"]["attainability"], "NOT_PROVEN")
        del upper["estimate_unavailable"]
        self.assertEqual(self.report()["status"], "FAIL")

    def test_noise_and_reference_score_gates(self):
        for seed, value in zip(self.spec["seeds"], [0, 10, 20]):
            self.make_run("noisy" + str(seed), "baseline", seed, value)
        self.spec["baseline_runs"] = ["noisy1", "noisy2", "noisy3"]
        result = self.report()
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["gates"]["reference_gain"]["passed"])
        self.assertFalse(result["gates"]["headroom"]["passed"])
        self.spec["stochastic"] = False
        self.spec["reference_score_range"] = [.6, .8]
        self.assertEqual(self.report()["status"], "FAIL")

    def test_declared_calibration_thresholds_are_applied(self):
        self.spec.update(sigma_required=1, reference_score_range=[.1, .9])
        result = self.report()
        self.assertEqual(result["status"], "CALIBRATED")
        self.assertEqual(result["gates"]["reference_score"]["range"], [.1, .9])
        self.assertAlmostEqual(result["gates"]["reference_gain"]["minimum"], .1)
        self.spec["reference_score_range"] = [.6, .9]
        self.assertEqual(self.report()["status"], "FAIL")

    def test_invalid_calibration_thresholds_are_rejected(self):
        for field, value in (("sigma_required", -1), ("sigma_required", True),
                             ("sigma_required", float("nan")), ("sigma_required", float("inf")),
                             ("reference_score_range", [-.1, 1]), ("reference_score_range", [0, 1.1]),
                             ("reference_score_range", [.5, .5]), ("reference_score_range", [1, 0])):
            spec = copy.deepcopy(self.spec)
            spec[field] = value
            result = self.report(spec)
            self.assertEqual(result["status"], "FAIL")
            self.assertIn(field, result["issues"][0]["error"])

    def test_headroom_is_remaining_reference_to_u_in_both_directions(self):
        for direction, baseline_values, reference, upper in (
                ("min", [9, 10, 11], 6, 4), ("max", [-1, 0, 1], 4, 6)):
            spec = copy.deepcopy(self.spec)
            spec["direction"] = direction
            spec["upper_bound"].update(value=upper, attainability={"mode": "estimate"})
            for kind, prefix, values in (("baseline", direction + "b", baseline_values),
                                         ("reference", direction + "r", [reference] * 3)):
                ids = []
                for seed, value in zip(spec["seeds"], values):
                    run_id = prefix + str(seed)
                    self.make_run(run_id, kind, seed, value, direction=direction)
                    ids.append(run_id)
                spec[kind + "_runs"] = ids
            result = self.report(spec)
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(result["gates"]["reference_gain"]["passed"])
            self.assertTrue(result["gates"]["reference_score"]["passed"])
            self.assertEqual(result["gates"]["headroom"]["value"], 2)
            self.assertEqual(result["gates"]["headroom"]["minimum"], 3)
            self.assertFalse(result["gates"]["headroom"]["passed"])

    def test_deterministic_single_seed_has_no_fake_noise_gate(self):
        self.spec.update(stochastic=False, seeds=[1], baseline_runs=["b1"], reference_runs=["r1"])
        self.spec["upper_bound"]["attainability"]["run_ids"] = ["u1"]
        result = self.report()
        self.assertEqual(result["status"], "CALIBRATED")
        self.assertIsNone(result["baseline"]["sample_std"])
        self.assertFalse(result["gates"]["noise"]["applied"])
        self.spec["stochastic"] = True
        self.assertEqual(self.report()["status"], "INCOMPLETE")

    def test_safe_ids_and_root_bound_evidence(self):
        self.spec["reference_runs"][2] = "../r3"
        self.assertEqual(self.report()["status"], "FAIL")
        self.spec["reference_runs"][2] = "r3"
        self.spec["upper_bound"]["evidence_paths"] = [str(self.root / "u-basis.md")]
        self.assertEqual(self.report()["status"], "FAIL")
        self.spec["upper_bound"]["evidence_paths"] = ["absent.md"]
        self.assertEqual(self.report()["status"], "INCOMPLETE")
        self.spec["upper_bound"]["evidence_paths"] = ["../../outside-absent.md"]
        self.assertEqual(self.report()["status"], "FAIL")

    def test_cli_temporary_output_cannot_overwrite_u_evidence(self):
        out = self.root / "calibration.json"
        evidence = out.with_suffix(".json.tmp")
        evidence.write_text("真实 U 依据，不得覆盖", encoding="utf-8")
        before = evidence.read_bytes()
        self.spec["upper_bound"]["evidence_paths"] = [evidence.name]
        spec_path = self.root / "spec.json"
        research_run.write_json(spec_path, self.spec)
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            calibrate.main(["--root", str(self.root), "--spec", str(spec_path), "--out", str(out)])
        self.assertEqual(evidence.read_bytes(), before)
        self.assertFalse(out.exists())

    def test_atomic_report_does_not_truncate_a_preexisting_temporary_hardlink(self):
        out = self.root / "calibration.json"
        original = self.root / "u-basis.md"
        alias = out.with_suffix(".json.tmp")
        try:
            os.link(original, alias)
        except OSError:
            self.skipTest("Filesystem does not support hard links")
        before = original.read_bytes()
        spec_path = self.root / "spec.json"
        research_run.write_json(spec_path, self.spec)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(calibrate.main(["--root", str(self.root), "--spec", str(spec_path), "--out", str(out)]), 0)
        self.assertEqual(original.read_bytes(), before)
        self.assertEqual(alias.read_bytes(), before)
        self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["status"], "CALIBRATED")

    def test_cli_preserves_input_and_run_evidence(self):
        spec_path, out = self.root / "spec.json", self.root / "calibration.json"
        research_run.write_json(spec_path, self.spec)
        before = {path: path.read_bytes() for path in (self.root / "runs").rglob("*") if path.is_file()}
        with redirect_stdout(io.StringIO()):
            self.assertEqual(calibrate.main(["--root", str(self.root), "--spec", str(spec_path), "--out", str(out)]), 0)
        self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["status"], "CALIBRATED")
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        for protected in (spec_path, self.root / "u-basis.md", self.root / "runs/b1/status.json"):
            with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                calibrate.main(["--root", str(self.root), "--spec", str(spec_path), "--out", str(protected)])


if __name__ == "__main__":
    unittest.main()
