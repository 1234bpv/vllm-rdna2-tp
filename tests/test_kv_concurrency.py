"""bench/kv-concurrency.py: run_limited(command, limit).

run_limited returns the standard output of a command that finishes within the limit, and None when
the limit expires. On expiry the child gets SIGTERM only, and the function returns after about
limit + 5 seconds without sending SIGKILL, even when the child ignores SIGTERM.

The file name has a hyphen, so the module is loaded with importlib. Signal delivery is POSIX-only:
on Windows Popen.terminate() calls TerminateProcess, which a child can neither catch nor ignore. The
tests that need a child to receive or ignore SIGTERM are skipped there with that reason. The call
sequence and the absence of kill() are checked on every platform with a Popen substitute.

The --engine option and its old spelling --rushij are checked by EngineOptionTest: main() runs with
run_measurement replaced, so no server is needed.
"""

import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import time
import unittest
from unittest import mock

from _support import BASH, RELEASE, ReleaseTest, msys

BENCH = RELEASE / "bench" / "kv-concurrency.py"
POSIX_ONLY = ("signal delivery is POSIX-only: on Windows Popen.terminate() is TerminateProcess, "
              "which a child cannot catch or ignore")
LIMIT_PLUS_FIVE_LOWER = 5.5    # the second wait is 5 s; a little slack for timer granularity
LIMIT_PLUS_FIVE_UPPER = 20.0   # generous upper bound for a loaded host; the function must not hang


def load_module():
    """Loads bench/kv-concurrency.py through importlib without writing bytecode into the release."""
    spec = importlib.util.spec_from_file_location("kv_concurrency", BENCH)
    module = importlib.util.module_from_spec(spec)
    saved = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = saved
    return module


class FakeChild:
    """Stands in for subprocess.Popen. It records every call; communicate raises TimeoutExpired
    while the child is 'running', and keeps raising after terminate() when it ignores SIGTERM."""

    def __init__(self, events, *, out="", hang_first=False, ignore_term=False):
        self.events = events
        self.out = out
        self.hang_first = hang_first
        self.ignore_term = ignore_term
        self.terminated = False

    def communicate(self, timeout=None):
        self.events.append(("communicate", timeout))
        if self.terminated and self.ignore_term:
            raise subprocess.TimeoutExpired(["fake-child"], timeout)
        if self.hang_first and not self.terminated:
            raise subprocess.TimeoutExpired(["fake-child"], timeout)
        return self.out, None

    def terminate(self):
        self.events.append(("terminate",))
        self.terminated = True

    def kill(self):
        self.events.append(("kill",))


class RunLimitedWithSubstituteTest(ReleaseTest):
    """Runs on every platform: the child is a FakeChild, so no signal is ever sent."""

    @classmethod
    def setUpClass(cls):
        cls.kv = load_module()

    def run_with(self, child, command, limit):
        with mock.patch.object(subprocess, "Popen", return_value=child) as popen:
            out = self.kv.run_limited(command, limit)
        return out, popen

    def test_output_inside_the_limit_is_returned_and_nothing_is_signalled(self):
        events = []
        child = FakeChild(events, out="card0,1,2\n")
        out, popen = self.run_with(child, ["rocm-smi"], 9)
        self.assertEqual(out, "card0,1,2\n")
        self.assertEqual(events, [("communicate", 9)], "a child that finished was waited for twice or signalled")
        self.assertEqual(popen.call_args.args[0], ["rocm-smi"])

    def test_child_that_exits_on_sigterm_returns_none_after_one_term_and_no_kill(self):
        events = []
        child = FakeChild(events, hang_first=True)
        out, _ = self.run_with(child, ["rocm-smi"], 9)
        self.assertIsNone(out, "an expired command must return None, not partial output")
        self.assertEqual(events, [("communicate", 9), ("terminate",), ("communicate", 5)])
        self.assertNotIn(("kill",), events, "SIGKILL was sent")

    def test_child_that_ignores_sigterm_returns_none_after_limit_plus_five_without_kill(self):
        events = []
        child = FakeChild(events, hang_first=True, ignore_term=True)
        out, _ = self.run_with(child, ["rocm-smi"], 9)
        self.assertIsNone(out)
        self.assertEqual(events, [("communicate", 9), ("terminate",), ("communicate", 5)],
                         "the function must wait 5 s after SIGTERM and then return")
        self.assertNotIn(("kill",), events, "SIGKILL was sent to a child that ignores SIGTERM")

    def test_the_limit_is_the_first_wait_and_the_grace_is_five_seconds(self):
        for limit in (1, 37):
            with self.subTest(limit=limit):
                events = []
                child = FakeChild(events, hang_first=True, ignore_term=True)
                self.run_with(child, ["x"], limit)
                self.assertEqual(events[0], ("communicate", limit))
                self.assertEqual(events[-1], ("communicate", 5))


class RunLimitedRealProcessTest(ReleaseTest):
    """Real child processes. Each one is started by the function under test, never by a test."""

    @classmethod
    def setUpClass(cls):
        cls.kv = load_module()

    def test_stdout_of_a_command_inside_the_limit_is_returned(self):
        out = self.kv.run_limited([sys.executable, "-c", "print('done')"], 60)
        self.assertIsNotNone(out, "a command inside the limit returned None")
        self.assertEqual(out.strip(), "done")

    def test_non_zero_exit_inside_the_limit_still_returns_its_stdout(self):
        out = self.kv.run_limited([sys.executable, "-c", "import sys; print('partial'); sys.exit(3)"], 60)
        self.assertIsNotNone(out)
        self.assertEqual(out.strip(), "partial")

    def test_command_over_the_limit_returns_none_within_limit_plus_five(self):
        began = time.monotonic()
        out = self.kv.run_limited([sys.executable, "-c", "import time; time.sleep(60)"], 1)
        elapsed = time.monotonic() - began
        self.assertIsNone(out, "an expired command must return None")
        self.assertGreaterEqual(elapsed, 0.9, "the function returned before the limit")
        self.assertLess(elapsed, 1 + 5 + 15, "the function did not return after the limit")

    @unittest.skipIf(os.name == "nt", POSIX_ONLY)
    def test_child_receives_sigterm_and_is_not_killed_first(self):
        # The handler writes a marker. SIGKILL cannot run a handler, so the marker proves SIGTERM.
        marker = self.sb.root / "term-marker.txt"
        ready = self.sb.root / "ready-marker.txt"
        code = (
            "import pathlib, signal, sys, time\n"
            "marker, ready = sys.argv[1], sys.argv[2]\n"
            "def on_term(signum, frame):\n"
            "    pathlib.Path(marker).write_text('TERM', encoding='utf-8')\n"
            "    sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, on_term)\n"
            "pathlib.Path(ready).write_text('ready', encoding='utf-8')\n"
            "time.sleep(60)\n"
        )
        out = self.kv.run_limited([sys.executable, "-c", code, str(marker), str(ready)], 3)
        self.assertIsNone(out)
        self.assertTrue(ready.exists(), "the child never installed its SIGTERM handler")
        self.assertEqual(marker.read_text(encoding="utf-8") if marker.exists() else "",
                         "TERM", "the child did not receive SIGTERM before the function returned")

    @unittest.skipIf(os.name == "nt", POSIX_ONLY)
    def test_child_that_ignores_sigterm_returns_after_about_limit_plus_five_and_survives(self):
        # The child ignores SIGTERM and waits for a stop file, bounded at 60 s by its own loop.
        # After run_limited returns it must still be alive: no SIGKILL was sent.
        pid_file = self.sb.root / "stubborn.pid"
        stop_file = self.sb.root / "stubborn.stop"
        body = ('echo $$ > "$1"; trap "" TERM; n=0; '
                'while [ ! -e "$2" ] && [ "$n" -lt 600 ]; do sleep 0.1; n=$((n + 1)); done')
        self.addCleanup(lambda: stop_file.touch())
        began = time.monotonic()
        out = self.kv.run_limited([BASH, "-c", body, "bash", msys(pid_file), msys(stop_file)], 1)
        elapsed = time.monotonic() - began
        self.assertIsNone(out)
        self.assertGreaterEqual(elapsed, LIMIT_PLUS_FIVE_LOWER, "returned before the 5 s grace")
        self.assertLess(elapsed, LIMIT_PLUS_FIVE_UPPER, "the function hung")
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        self.assertTrue(self.sb.bash_alive(pid), "the child that ignores SIGTERM was killed")


class EngineOptionTest(unittest.TestCase):
    """--engine selects the server that is measured; its old spelling --rushij is accepted too.

    main() runs with run_measurement replaced, so no server is needed. Each call loads the module
    again, so the global ENGINE set by main() does not carry over from one test to the next.
    """

    ARGS = ["--base-url", "http://127.0.0.1:1", "--streams", "1", "--length", "10", "--blocks", "0",
            "--output", "unused.jsonl"]

    def run_main(self, extra):
        module = load_module()
        with mock.patch.object(sys, "argv", ["kv-concurrency.py"] + self.ARGS + list(extra)), \
                mock.patch.object(module, "run_measurement", return_value=0) as run:
            module.main()
        return module, run.call_args[0][0]

    def test_engine_llamacpp_selects_the_llamacpp_metric_names(self):
        module, args = self.run_main(["--engine", "llamacpp"])
        self.assertEqual(args.engine, "llamacpp")
        self.assertEqual(module.ENGINE, "llamacpp")
        self.assertEqual(module.METRIC_NAMES, module.METRIC_NAMES_BY_ENGINE["llamacpp"])

    def test_old_spelling_rushij_gives_the_same_engine(self):
        module, args = self.run_main(["--rushij", "llamacpp"])
        self.assertEqual(args.engine, "llamacpp")
        self.assertEqual(module.ENGINE, "llamacpp")
        self.assertEqual(module.METRIC_NAMES, module.METRIC_NAMES_BY_ENGINE["llamacpp"])

    def test_engine_defaults_to_vllm(self):
        module, args = self.run_main([])
        self.assertEqual(args.engine, "vllm")
        self.assertEqual(module.ENGINE, "vllm")

    def test_unknown_engine_is_refused_before_any_run(self):
        module = load_module()
        argv = ["kv-concurrency.py"] + self.ARGS + ["--engine", "ollama"]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(module, "run_measurement", return_value=0) as run, \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                module.main()
        self.assertEqual(caught.exception.code, 2)
        run.assert_not_called()
