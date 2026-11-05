"""scripts/install-service.sh: the printed systemd unit, the refusals, and the rule that nothing is installed.

The unit fields checked are: After and Wants name network-online.target; StartLimitIntervalSec=3600;
StartLimitBurst=2; User taken from USER; SupplementaryGroups=video render; ExecStart runs serve.sh with
the preset and --foreground; ExecStop runs stop.sh; Restart=on-failure; RestartSec=60; KillSignal=SIGTERM;
SendSIGKILL=no; TimeoutStopSec=120; WantedBy=multi-user.target; RestartPreventExitStatus=2 3 4 6. The unit
must not contain After=multi-user.target. The options --print-command, --list, --help, -h and --foreground
are refused with exit code 2, and so is a CONVERT preset without weights. Stubs for systemctl and install
record any call; none is expected.
"""

import re

from _support import ReleaseTest

SCRIPT = "scripts/install-service.sh"


class InstallServiceTest(ReleaseTest):

    def setUp(self):
        super().setUp()
        self.sb.install_venv()
        self.sb.add_stub("systemctl", '#!/bin/bash\n_rt_log "$@"\nexit 0\n')
        self.sb.add_stub("install", '#!/bin/bash\n_rt_log "$@"\nexit 0\n')
        self.home_before = sorted(str(p.relative_to(self.sb.home)) for p in self.sb.home.rglob("*"))

    def run_service(self, args, *, timeout=120, env=None):
        return self.sb.bash(SCRIPT, list(args), timeout=timeout, env=env)

    def unit(self, args=("qwen3.8-27b",)):
        r = self.run_service(args)
        self.assertEqual(r.rc, 0, r.describe())
        return r.out

    def field(self, text, key):
        values = re.findall(rf"^{key}=(.*)$", text, re.M)
        return values

    # ----- unit content ----------------------------------------------------------------
    def test_unit_has_the_required_fields(self):
        text = self.unit()
        self.assertEqual(self.field(text, "After"), ["network-online.target"])
        self.assertEqual(self.field(text, "Wants"), ["network-online.target"])
        self.assertEqual(self.field(text, "StartLimitIntervalSec"), ["3600"])
        self.assertEqual(self.field(text, "StartLimitBurst"), ["2"])
        self.assertEqual(self.field(text, "User"), ["relt"], "User is not taken from USER")
        self.assertEqual(self.field(text, "SupplementaryGroups"), ["video render"])
        self.assertEqual(self.field(text, "Restart"), ["on-failure"])
        self.assertEqual(self.field(text, "RestartSec"), ["60"])
        self.assertEqual(self.field(text, "KillSignal"), ["SIGTERM"])
        self.assertEqual(self.field(text, "SendSIGKILL"), ["no"])
        self.assertEqual(self.field(text, "TimeoutStopSec"), ["120"])
        self.assertEqual(self.field(text, "WantedBy"), ["multi-user.target"])
        self.assertEqual(self.field(text, "RestartPreventExitStatus"), ["2 3 4 6"])

    def test_exec_start_and_exec_stop_point_into_the_release(self):
        text = self.unit(["qwen3.8-27b", "--ctx", "4096"])
        exec_start = self.field(text, "ExecStart")
        self.assertEqual(len(exec_start), 1)
        self.assertRegex(exec_start[0], r'serve\.sh"? "?qwen3\.8-27b"? --foreground')
        self.assertIn('"--ctx" "4096"', exec_start[0], "serve.sh options are not in ExecStart")
        exec_stop = self.field(text, "ExecStop")
        self.assertEqual(len(exec_stop), 1)
        self.assertRegex(exec_stop[0], r'stop\.sh"?$')

    def test_no_forbidden_after_multi_user(self):
        text = self.unit()
        self.assertNotIn("After=multi-user.target", text)

    def test_unit_name_contains_the_preset(self):
        text = self.unit()
        self.assertIn("vllm-rdna2-qwen3.8-27b.service", text)

    def test_installation_commands_are_printed_as_comments(self):
        text = self.unit()
        lines = text.splitlines()
        self.assertTrue(lines[0].startswith("#"))
        comments = [line for line in lines if line.startswith("#")]
        self.assertTrue(any("systemctl enable --now" in c for c in comments), comments)
        for line in lines:
            if line.strip() and not line.startswith("#"):
                self.assertRegex(line, r"^\[\w+\]$|^[A-Za-z]+=", "unit line is neither a section nor key=value")

    # ----- nothing installed -----------------------------------------------------------
    def test_nothing_is_installed_or_started(self):
        self.unit()
        self.assertEqual(self.sb.calls("systemctl"), [], "systemctl was called")
        self.assertEqual(self.sb.calls("install"), [], "install was called")
        after = sorted(str(p.relative_to(self.sb.home)) for p in self.sb.home.rglob("*"))
        self.assertEqual(after, self.home_before, "files were written under HOME")

    # ----- refusals --------------------------------------------------------------------
    def test_refused_options_are_code2(self):
        for option in ("--print-command", "--list", "--help", "-h", "--foreground"):
            with self.subTest(option=option):
                r = self.run_service(["qwen3.8-27b", option], timeout=60)
                self.assertEqual(r.rc, 2, r.describe())

    def test_preset_must_come_first_and_is_required(self):
        for args in ([], ["--ctx", "4096"]):
            with self.subTest(args=args):
                r = self.run_service(args, timeout=60)
                self.assertEqual(r.rc, 2, r.describe())

    def test_unknown_preset_is_code2(self):
        r = self.run_service(["no-such-preset"], timeout=60)
        self.assertEqual(r.rc, 2, r.describe())

    def test_invalid_serve_option_is_code2(self):
        r = self.run_service(["qwen3.8-27b", "--ctx", "0"], timeout=60)
        self.assertEqual(r.rc, 2, r.describe())

    def test_convert_preset_without_weights_is_code2(self):
        # A CONVERT preset without weights gives exit code 2, the same refusal as --print-command.
        r = self.run_service(["qwen3.6-35b-a3b"], timeout=60)
        self.assertEqual(r.rc, 2, r.describe())

    def test_convert_preset_with_weights_gives_a_unit(self):
        (self.sb.rdh / "models" / "qwen3.6-35b-a3b-ct-int4").mkdir(parents=True)
        r = self.run_service(["qwen3.6-35b-a3b"])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertIn("vllm-rdna2-qwen3.6-35b-a3b.service", r.out)
