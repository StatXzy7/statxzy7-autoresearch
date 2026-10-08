#!/usr/bin/env python3
"""Calibrate baseline B, reference R and directional target U from run receipts.

Numerical calibration is not a scientific judgment of baseline health, methods,
or attainable headroom. Estimates retain their explicit evidence strength.
"""
import argparse
import json
import math
from pathlib import Path
import re
import statistics
import sys

from research_run import inside, sha256, validate_result, write_json


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def normalize(value, baseline, upper_bound, direction):
    """B maps to zero and U to one for either direction; never clip."""
    if direction not in ("min", "max") or not all(map(finite, (value, baseline, upper_bound))):
        raise ValueError("Normalization requires finite values and min/max direction")
    sign = 1 if direction == "max" else -1
    headroom = sign * (upper_bound - baseline)
    if headroom <= 0 or not math.isfinite(headroom):
        raise ValueError("U must be strictly better than B in the declared direction")
    result = sign * (value - baseline) / headroom
    if not math.isfinite(result):
        raise ValueError("Normalized score is not finite")
    return result


def validate_spec(spec):
    if not isinstance(spec, dict):
        raise ValueError("Calibration specification must be an object")
    for field in ("metric", "protocol", "split"):
        if type(spec.get(field)) is not str or not spec[field].strip():
            raise ValueError(field + " must be a nonempty string")
    if spec.get("direction") not in ("min", "max") or type(spec.get("stochastic")) is not bool:
        raise ValueError("direction and stochastic must be declared")
    seeds = spec.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must contain distinct integers")
    for field in ("baseline_runs", "reference_runs"):
        ids = spec.get(field)
        if not isinstance(ids, list) or len(ids) != len(seeds):
            raise ValueError(field + " must declare exactly one run per seed")
        validate_ids(ids)
    if set(spec["baseline_runs"]) & set(spec["reference_runs"]):
        raise ValueError("Baseline and reference runs must be distinct")
    score_range = spec.get("reference_score_range", [0, 1])
    if (not isinstance(score_range, list) or len(score_range) != 2 or not all(map(finite, score_range))
            or not 0 <= score_range[0] < score_range[1] <= 1):
        raise ValueError("reference_score_range must be an increasing interval within [0, 1]")
    sigma = spec.get("sigma_required", 3)
    if not finite(sigma) or sigma < 0:
        raise ValueError("sigma_required must be finite and nonnegative")
    if type(spec.get("require_demonstrated_u", False)) is not bool:
        raise ValueError("require_demonstrated_u must be boolean")
    upper = spec.get("upper_bound")
    if not isinstance(upper, dict) or upper.get("source") not in ("estimated_attainable_upper_bound", "metric_ceiling") or not finite(upper.get("value")):
        raise ValueError("upper_bound must declare a supported source and finite value")
    if type(upper.get("basis")) is not str or not upper["basis"].strip():
        raise ValueError("upper_bound basis is required")
    if not isinstance(upper.get("evidence_paths"), list) or not upper["evidence_paths"]:
        raise ValueError("upper_bound supporting evidence_paths are required")
    if upper["source"] == "metric_ceiling" and (type(upper.get("estimate_unavailable")) is not str or not upper["estimate_unavailable"].strip()):
        raise ValueError("metric_ceiling requires the reason an attainable estimate is unavailable")
    attainability = upper.get("attainability")
    if not isinstance(attainability, dict) or attainability.get("mode") not in ("empirical", "estimate", "theoretical"):
        raise ValueError("upper_bound attainability mode must be declared")
    ids = attainability.get("run_ids", [])
    if not isinstance(ids, list):
        raise ValueError("upper_bound run_ids must be a list")
    validate_ids(ids)
    if attainability["mode"] != "empirical" and ids:
        raise ValueError("Only empirical attainability accepts run_ids; use evidence_paths for estimates")
    if set(ids) & (set(spec["baseline_runs"]) | set(spec["reference_runs"])):
        raise ValueError("Upper-bound runs must be distinct from baseline and reference")


def validate_ids(ids):
    if any(type(item) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", item) for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("Run IDs must be distinct safe single directory names")


def load_runs(root, ids, kind, spec, issues):
    records, complete = [], True
    for run_id in ids:
        try:
            receipt_path = inside(root, "runs/" + run_id + "/status.json")
        except ValueError as exc:
            # A missing run is absent evidence; path escapes remain invalid input.
            state = "INCOMPLETE" if not (root / "runs" / run_id / "status.json").exists() else "FAIL"
            issues.append({"status": state, "run_id": run_id, "error": str(exc)})
            complete = False
            continue
        try:
            run = receipt_path.parent
            status = json.loads(receipt_path.read_text(encoding="utf-8"))
            if not isinstance(status, dict) or status.get("id") != run_id:
                raise ValueError("Invalid run receipt identity")
            for field in ("metric", "direction", "protocol", "split"):
                if type(status.get(field)) is not str or status[field] != spec[field]:
                    raise ValueError("Run identity mismatch: " + field)
            if status.get("kind") != kind or type(status.get("seed")) is not int or status["seed"] not in spec["seeds"]:
                raise ValueError("Run role or seed does not match calibration")
            if status.get("state") != "SUCCESS":
                issues.append({"status": "INCOMPLETE", "run_id": run_id, "run_state": status.get("state"),
                               "error": "Declared anchor run did not succeed"})
                complete = False
                continue
            if not finite(status.get("value")) or type(status.get("returncode")) is not int or status["returncode"] != 0:
                raise ValueError("Successful receipt must contain a finite metric and zero exit code")
            value, artifacts, binding = validate_result(run, status)
            if value != status.get("value") or artifacts != status.get("artifacts") or binding != status.get("metrics_sha256"):
                raise ValueError("Result binding changed after execution")
            records.append({"id": run_id, "seed": status["seed"], "value": value,
                            "metrics_sha256": binding, "artifacts": artifacts, "receipt_sha256": sha256(receipt_path)})
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
            issues.append({"status": "FAIL", "run_id": run_id, "error": str(exc)})
            complete = False
    if complete and sorted(record["seed"] for record in records) != sorted(spec["seeds"]):
        issues.append({"status": "FAIL", "error": kind + " must include each declared seed exactly once"})
        complete = False
    return {"run_ids": ids, "records": records, "mean": statistics.mean(record["value"] for record in records) if complete and records else None}


def calibrate(root, spec):
    report = {"schema_version": 1, "status": "INCOMPLETE", "requires_semantic_review": True,
              "issues": [], "note": "Calibration checks numerical and evidence contracts; semantic review remains required."}
    issues = report["issues"]
    try:
        validate_spec(spec)
        root = Path(root).resolve(strict=True)
        report.update({key: spec[key] for key in ("metric", "direction", "protocol", "split", "stochastic", "seeds")})
        upper = spec["upper_bound"]
        evidence, seen = [], set()
        for relative in upper["evidence_paths"]:
            try:
                path = inside(root, relative)
                if path in seen:
                    raise ValueError("Upper-bound evidence files must be distinct")
                seen.add(path)
                evidence.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
            except (OSError, ValueError) as exc:
                missing = False
                if type(relative) is str and relative and not Path(relative).is_absolute():
                    try:
                        candidate = (root / relative).resolve()
                        candidate.relative_to(root)
                        missing = not candidate.exists()
                    except ValueError:
                        pass
                issues.append({"status": "INCOMPLETE" if missing else "FAIL", "error": str(exc)})
        mode = upper["attainability"]["mode"]
        strength = "ESTIMATED" if mode == "estimate" and upper["source"] == "estimated_attainable_upper_bound" else "NOT_PROVEN"
        report["upper_bound"] = {"value": upper["value"], "source": upper["source"], "basis": upper["basis"],
                                 "evidence": evidence, "attainability": strength, "mode": mode}
        if "estimate_unavailable" in upper:
            report["upper_bound"]["estimate_unavailable"] = upper["estimate_unavailable"]
        for name, kind in (("baseline", "baseline"), ("reference", "reference")):
            report[name] = load_runs(root, spec[name + "_runs"], kind, spec, issues)
        if mode == "empirical":
            ids = upper["attainability"].get("run_ids", [])
            if len(ids) != len(spec["seeds"]):
                issues.append({"status": "INCOMPLETE", "error": "Empirical U needs one upper_bound run for every declared seed"})
            else:
                observed = load_runs(root, ids, "upper_bound", spec, issues)
                report["upper_bound"]["observed"] = observed
                if observed["mean"] is not None:
                    if not math.isclose(observed["mean"], upper["value"], rel_tol=1e-9, abs_tol=1e-12):
                        issues.append({"status": "FAIL", "error": "Declared U differs from the mean of its empirical runs"})
                    else:
                        report["upper_bound"]["attainability"] = "DEMONSTRATED"
        if spec.get("require_demonstrated_u", False) and report["upper_bound"]["attainability"] != "DEMONSTRATED":
            issues.append({"status": "INCOMPLETE", "error": "Demonstrated U was required but is not established"})
        baseline, reference = report["baseline"]["mean"], report["reference"]["mean"]
        if baseline is not None and reference is not None:
            score = normalize(reference, baseline, upper["value"], spec["direction"])
            report["normalized"] = {"baseline": 0.0, "reference": score, "upper_bound": 1.0}
            values = [record["value"] for record in report["baseline"]["records"]]
            sigma = statistics.stdev(values) if len(values) > 1 else None
            report["baseline"]["sample_std"] = sigma
            low, high = spec.get("reference_score_range", [0, 1])
            gates = {"reference_score": {"value": score, "range": [low, high], "passed": low <= score <= high}}
            if not gates["reference_score"]["passed"]:
                issues.append({"status": "FAIL", "error": "Reference normalized score is outside the declared range"})
            if spec["stochastic"]:
                if len(values) < 3:
                    issues.append({"status": "INCOMPLETE", "error": "Stochastic calibration requires at least three baseline/reference seeds"})
                else:
                    sign = 1 if spec["direction"] == "max" else -1
                    threshold = spec.get("sigma_required", 3) * sigma
                    if not math.isfinite(threshold):
                        raise ValueError("Baseline noise threshold is not finite")
                    for name, value in (("reference_gain", sign * (reference - baseline)), ("headroom", sign * (upper["value"] - reference))):
                        if not math.isfinite(value):
                            raise ValueError("Calibration gain or headroom is not finite")
                        gates[name] = {"value": value, "minimum": threshold, "passed": value >= threshold}
                        if not gates[name]["passed"]:
                            issues.append({"status": "FAIL", "error": name + " is below the baseline noise threshold"})
            else:
                gates["noise"] = {"applied": False, "reason": "Deterministic protocol; no stochastic significance claim"}
            report["gates"] = gates
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        issues.append({"status": "FAIL", "error": str(exc)})
    report["status"] = "FAIL" if any(item["status"] == "FAIL" for item in issues) else "INCOMPLETE" if issues else "CALIBRATED"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
        result = calibrate(args.root, spec)
        destination = Path(args.out).resolve()
        # Calibration only writes a separate report, never its evidence or receipts.
        root = Path(args.root).resolve()
        protected = [Path(args.spec).resolve()] + list((root / "runs").rglob("*"))
        upper = spec.get("upper_bound", {}) if isinstance(spec, dict) else {}
        if isinstance(upper, dict) and isinstance(upper.get("evidence_paths", []), list):
            protected.extend((root / name).resolve() for name in upper.get("evidence_paths", []) if isinstance(name, str))
        protected_paths = {path.resolve() for path in protected}
        temporary = destination.with_suffix(destination.suffix + ".tmp").resolve()
        if any(path in protected_paths or root / "runs" in path.parents
               for path in (destination, temporary)):
            raise ValueError("Output must not overwrite calibration input or run evidence")
        write_json(destination, result)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] == "CALIBRATED" else 1


if __name__ == "__main__":
    sys.exit(main())
