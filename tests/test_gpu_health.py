"""scripts/gpu-health.sh: the driver verdict.

Exit code 0 means the GPUs can be used; exit code 1 means a reboot is needed. The probe is the venv
interpreter run as 'python -I -' (stub: PYTHON_STUB, state probe.seq). The kernel-log patterns are taken
from the script itself; the tests that rely on their wording are listed as interpretations in
tests/README.md.
"""

import time

from _support import ReleaseTest


class GpuHealthTest(ReleaseTest):

    def setUp(self):
        super().setUp()
        self.sb.install_venv()

    def gpu_health(self, args=(), *, env=None, timeout=120):
        return self.sb.bash("scripts/gpu-health.sh", list(args), timeout=timeout, env=env)

    def probe_calls(self):
        return [row for row in self.sb.calls("python") if row[1:3] == ["-I", "-"]]

    def test_help_exits_zero(self):
        r = self.gpu_health(["--help"], timeout=60)
        self.assertEqual(r.rc, 0, r.describe())

    def test_unknown_argument_is_refused(self):
        r = self.gpu_health(["--bogus"], timeout=60)
        self.assertEqual(r.rc, 1, r.describe())
        self.assertIn("unknown argument", r.err)

    def test_healthy_driver_exits_zero_and_probe_runs_isolated(self):
        r = self.gpu_health()
        self.assertEqual(r.rc, 0, r.describe())
        self.assertEqual(len(self.probe_calls()), 1, "the probe did not run once")

    def test_wrong_numbers_exit_one(self):
        self.sb.seq("probe", ["wrong"])
        r = self.gpu_health()
        self.assertEqual(r.rc, 1, r.describe())
        self.assertIn("probe failed", r.err)
        self.assertIn("differs from reference", r.err)

    def test_nan_exits_one(self):
        self.sb.seq("probe", ["nan"])
        r = self.gpu_health()
        self.assertEqual(r.rc, 1, r.describe())

    def test_hung_probe_exits_one_within_the_probe_limit(self):
        self.sb.seq("probe", ["hang"])
        began = time.monotonic()
        r = self.gpu_health(env={"PROBE_TIMEOUT": "2"})
        elapsed = time.monotonic() - began
        self.assertEqual(r.rc, 1, r.describe())
        self.assertIn("probe hung", r.err)
        self.assertLess(elapsed, 60, "PROBE_TIMEOUT did not limit the probe")

    def test_ring_timeout_in_kernel_log_exits_one_even_when_probe_passes(self):
        self.sb.set("journal.txt", "amdgpu 0000:03:00.0: ring gfx_0.0.0 timeout, signaled seq=1\n")
        r = self.gpu_health()
        self.assertEqual(r.rc, 1, r.describe())
        self.assertIn("kernel log", r.err)

    def test_gpu_reset_in_kernel_log_exits_one(self):
        self.sb.set("journal.txt", "amdgpu 0000:03:00.0: GPU reset begin!\n")
        r = self.gpu_health()
        self.assertEqual(r.rc, 1, r.describe())

    def test_unrelated_kernel_log_lines_do_not_fail(self):
        self.sb.set("journal.txt", "usb 1-1: new high-speed USB device number 4\n")
        r = self.gpu_health()
        self.assertEqual(r.rc, 0, r.describe())
