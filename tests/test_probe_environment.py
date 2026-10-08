"""No network/GPU required: verify quoting, interpretation and failure receipts."""

import importlib.util
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch


SCRIPT = Path(__file__).resolve().parents[1] / ".agents/skills/autoresearch/scripts/probe_environment.py"
SPEC = importlib.util.spec_from_file_location("probe_environment", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def payload(smoke="NOT_RUN", status="PASS"):
    return {"schema_version": 1, "probe_status": status,
            "gpu": {"smoke": {"status": smoke}}}


def response(data, code=0, stderr=""):
    return subprocess.CompletedProcess([], code, probe.PROBE_MARKER + json.dumps(data) + "\n", stderr)


class CommandTests(unittest.TestCase):
    def test_local_command_has_no_shell(self):
        self.assertEqual(probe.build_probe_command(python_executable="/opt/env/bin/python", device=2),
                         ["/opt/env/bin/python", "-", "--_probe", "--device", "2"])

    def test_remote_command_quotes_interpreter_and_preserves_host_verification(self):
        interpreter = "/opt/env with spaces/python';$(touch danger)"
        argv = probe.build_probe_command("gpu-cloud", interpreter, True, 1, ssh_executable="/usr/bin/ssh")
        self.assertEqual(argv[:-1], [
            "/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3", "gpu-cloud",
        ])
        self.assertEqual(shlex.split(argv[-1]), [interpreter, "-", "--_probe", "--device", "1", "--gpu-smoke"])
        self.assertNotIn("StrictHostKeyChecking=no", " ".join(argv))

    def test_invalid_host_and_device_are_rejected(self):
        for host in ("-oProxyCommand=bad", "cloud host", "cloud;echo bad", ""):
            with self.subTest(host=host), self.assertRaises(ValueError):
                probe.build_probe_command(host, ssh_executable="ssh")
        with self.assertRaises(ValueError):
            probe.build_probe_command(device=-1)

    def test_workdir_is_quoted_as_one_remote_argument(self):
        workdir = "/home/project data/中文';$(touch danger)"
        argv = probe.build_probe_command("gpu-cloud", ssh_executable="ssh", workdir=workdir)
        self.assertEqual(shlex.split(argv[-1])[-1], "--workdir=" + workdir)
        self.assertEqual(probe.build_probe_command(workdir="-data")[-1], "--workdir=-data")
        for path in ("", "bad\x00path"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                probe.build_probe_command(workdir=path)

    def test_missing_ssh_is_not_a_success(self):
        with patch.object(probe.shutil, "which", return_value=None):
            report = probe.run_probe("gpu-cloud")
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["transport"]["error_type"], "FileNotFoundError")
        self.assertIsNone(report["environment"])


class ParsingTests(unittest.TestCase):
    def test_nvidia_csv_handles_quoted_device_names_and_missing_memory(self):
        data = probe.parse_nvidia_csv('0, "GPU, Example", GPU-a, 24576, 555.42\n1, Other, GPU-b, N/A, 555.42\n')
        self.assertEqual(data[0]["name"], "GPU, Example")
        self.assertEqual(data[0]["memory_total_mib"], 24576.0)
        self.assertIsNone(data[1]["memory_total_mib"])

    def test_nvidia_malformed_csv_is_rejected(self):
        with self.assertRaises(ValueError):
            probe.parse_nvidia_csv("No devices found\n")

    def test_probe_marker_ignores_login_banner(self):
        data = payload()
        self.assertEqual(probe.parse_probe_output("Welcome to GPU host\n" + response(data).stdout), data)

    def test_invalid_probe_responses_are_rejected(self):
        for output in ("", "{}", probe.PROBE_MARKER + "invalid", response(payload()).stdout * 2,
                       response({"schema_version": 9}).stdout, response(payload(smoke="UNKNOWN")).stdout,
                       response(payload(smoke="FAILED")).stdout,
                       response({"schema_version": 1, "probe_status": "PASS", "gpu": []}).stdout):
            with self.subTest(output=output), self.assertRaises(ValueError):
                probe.parse_probe_output(output)


class RunnerTests(unittest.TestCase):
    def test_probe_success_uses_stdin_utf8_and_bounded_timeout(self):
        with patch.object(probe.subprocess, "run", return_value=response(payload())) as run:
            report = probe.run_probe()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["transport"]["status"], "PASS")
        kwargs = run.call_args.kwargs
        self.assertIn("def collect_environment", kwargs["input"])
        self.assertEqual(kwargs["encoding"], "utf-8")
        self.assertEqual(kwargs["timeout"], 90)
        self.assertNotIn("shell", kwargs)

    def test_requested_workdir_is_recorded_and_passed_to_child(self):
        with patch.object(probe.subprocess, "run", return_value=response(payload())) as run:
            report = probe.run_probe(workdir="/home/project data")
        self.assertEqual(report["target"]["workdir"], "/home/project data")
        self.assertIn("--workdir=/home/project data", run.call_args.args[0])

    def test_missing_workdir_produces_failed_report_and_preserves_error(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "不存在的目录"
            report = probe.run_probe(workdir=missing)
            self.assertFalse(missing.exists())
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["transport"]["status"], "FAILED")
        self.assertEqual(report["transport"]["returncode"], 1)
        self.assertIn("FileNotFoundError", report["transport"]["stderr"])
        self.assertIn("不存在的目录", report["transport"]["stderr"])
        self.assertIsNone(report["environment"])

    def test_ssh_failure_preserves_error_and_return_code(self):
        failed = subprocess.CompletedProcess([], 255, "", "Permission denied: 连接失败\n")
        with patch.object(probe.shutil, "which", return_value="ssh"), \
                patch.object(probe.subprocess, "run", return_value=failed):
            report = probe.run_probe("gpu-cloud")
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["transport"]["returncode"], 255)
        self.assertEqual(report["transport"]["stderr"], failed.stderr)
        self.assertEqual(report["transport"]["error_type"], "PROCESS_FAILED")
        self.assertIsNone(report["environment"])

    def test_timeout_preserves_utf8_partial_error(self):
        timeout = subprocess.TimeoutExpired(["ssh"], 90, output=b"partial", stderr="超时".encode("utf-8"))
        with patch.object(probe.subprocess, "run", side_effect=timeout):
            report = probe.run_probe()
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["transport"]["error_type"], "TIMEOUT")
        self.assertEqual(report["transport"]["stderr"], "超时")

    def test_failed_smoke_has_successful_transport_but_failed_probe(self):
        with patch.object(probe.subprocess, "run", return_value=response(payload("FAILED", "FAILED"), 1)):
            report = probe.run_probe(gpu_smoke=True)
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["transport"]["status"], "PASS")
        self.assertEqual(report["environment"]["gpu"]["smoke"]["status"], "FAILED")

    def test_requested_smoke_may_not_be_silently_skipped(self):
        with patch.object(probe.subprocess, "run", return_value=response(payload())):
            report = probe.run_probe(gpu_smoke=True)
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["transport"]["error_type"], "INVALID_RESPONSE")

    def test_unexpected_process_exit_is_failed_even_with_valid_json(self):
        for returncode in (1, 255):
            with self.subTest(returncode=returncode), \
                    patch.object(probe.subprocess, "run", return_value=response(payload(), returncode)):
                report = probe.run_probe()
            self.assertEqual(report["status"], "FAILED")
            self.assertEqual(report["transport"]["status"], "FAILED")

    def test_json_written_without_bom_preserves_chinese(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "探测结果.json"
            probe.write_report({"error": "连接失败"}, output)
            content = output.read_bytes()
            self.assertFalse(content.startswith(b"\xef\xbb\xbf"))
            self.assertIn("连接失败", content.decode("utf-8"))
            self.assertEqual(json.loads(content), {"error": "连接失败"})


class SmokeTests(unittest.TestCase):
    def fake_torch(self, available=True):
        return types.SimpleNamespace(
            __version__="test-version", version=types.SimpleNamespace(cuda="test-cuda"),
            cuda=types.SimpleNamespace(is_available=MagicMock(return_value=available),
                                       device_count=MagicMock(return_value=1), synchronize=MagicMock()),
            ones=MagicMock(), full_like=MagicMock(), allclose=MagicMock(return_value=True),
            float32="float32",
        )

    def test_cuda_availability_is_never_reported_as_a_smoke(self):
        torch = self.fake_torch()
        with patch.dict(sys.modules, {"torch": torch}):
            info, smoke = probe.collect_torch()
        self.assertTrue(info["cuda_available"])
        self.assertEqual(smoke["status"], "NOT_RUN")
        torch.ones.assert_not_called()

    def test_cuda_unavailable_requested_smoke_fails(self):
        with patch.dict(sys.modules, {"torch": self.fake_torch(False)}):
            _, smoke = probe.collect_torch(True)
        self.assertEqual(smoke["status"], "FAILED")
        self.assertIn("CUDA unavailable", smoke["error"])

    def test_smoke_pass_requires_allocation_matmul_and_synchronization(self):
        torch = self.fake_torch()
        tensor = torch.ones.return_value
        product = tensor.__matmul__.return_value
        product.__getitem__.return_value.item.return_value = 16.0
        with patch.dict(sys.modules, {"torch": torch}):
            _, smoke = probe.collect_torch(True)
        self.assertEqual(smoke["status"], "PASS")
        torch.ones.assert_called_once_with((16, 16), dtype="float32", device="cuda:0")
        tensor.__matmul__.assert_called_once_with(tensor)
        torch.cuda.synchronize.assert_called_once_with(0)
        torch.allclose.assert_called_once()

    def test_allocation_failure_is_preserved(self):
        torch = self.fake_torch()
        torch.ones.side_effect = RuntimeError("CUDA out of memory: 内存不足")
        with patch.dict(sys.modules, {"torch": torch}):
            _, smoke = probe.collect_torch(True)
        self.assertEqual(smoke["status"], "FAILED")
        self.assertIn("内存不足", smoke["error"])


class WorkdirTests(unittest.TestCase):
    def test_probe_checks_actual_existing_directory_disk(self):
        original = Path.cwd()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory(prefix="autoresearch-workdir-") as directory:
            try:
                with redirect_stdout(stdout), redirect_stderr(stderr), \
                        patch.object(probe, "collect_torch", return_value=(
                            {"status": "UNAVAILABLE"}, {"status": "NOT_RUN"})), \
                        patch.object(probe, "collect_nvidia", return_value={"status": "UNAVAILABLE"}):
                    exit_code = probe.main(["--_probe", "--workdir", directory])
                data = probe.parse_probe_output(stdout.getvalue())
                self.assertEqual(exit_code, 0)
                self.assertEqual(Path(data["disk"]["path"]).resolve(), Path(directory).resolve())
                self.assertGreater(data["disk"]["total_bytes"], 0)
                self.assertEqual(data["disk"]["status"], "PASS")
                self.assertEqual(stderr.getvalue(), "")
            finally:
                os.chdir(original)


if __name__ == "__main__":
    unittest.main()
