"""serve.sh: launch, readiness, the probe request, retries, the driver gate, and the stop of a previous server.

Exit codes: 0 success, 3 no install marker, 4 driver check failed, 5 every attempt failed, 6 a previous server
did not stop. Also covered: --foreground, the /dev/shm cleanup, and the PATH of the server process (the
venv and ROCm first, ahead of the caller's PATH, so that ninja and hipcc resolve for the .hip kernels).
The vllm, curl, journalctl, python and sleep stubs decide every outcome; a test changes only the state
files (vllm.seq, chat.seq, probe.seq). A test that starts a server stops it again in cleanup.
"""

import json
import os
import pathlib
import re
import time
import unittest

from _support import ReleaseTest, msys, write_script

PRESET = "qwen3.8-27b"
RESERVED_PORT = "8100"
# A live stand-in for a previous server: stop.sh recognises it by bin/vllm in its command line.
VLLM_ARGV0 = "/opt/rdna2/venv/bin/vllm serve stub-model"


def starts(sb):
    """Number of vllm serve launches (the --help call is not a launch)."""
    return sum(1 for row in sb.calls("vllm") if len(row) > 1 and row[1] == "serve")


def probes(sb):
    """gpu-health probes: the venv interpreter run as 'python -I -'."""
    return [row for row in sb.calls("python") if row[1:3] == ["-I", "-"]]


def chats(sb):
    return [row for row in sb.calls("curl") if any("/v1/chat/completions" in a for a in row[1:])]


def models_polls(sb):
    return [row for row in sb.calls("curl") if any(a.endswith("/v1/models") for a in row[1:])]


def event_order(sb):
    """Ordered list of launches ('start'), driver probes ('probe') and chat requests ('chat')."""
    order = []
    for row in sb.calls():
        if row[0] == "vllm" and len(row) > 1 and row[1] == "serve":
            order.append("start")
        elif row[0] == "python" and row[1:3] == ["-I", "-"]:
            order.append("probe")
        elif row[0] == "curl" and any("/v1/chat/completions" in a for a in row[1:]):
            order.append("chat")
    return order


class ServeLaunchTest(ReleaseTest):

    def setUp(self):
        super().setUp()
        self.sb.install_venv()
        self.pid_file = self.sb.rdh / "run" / "vllm.pid"
        self.addCleanup(self.stop_leftovers)

    def stop_leftovers(self):
        if self.pid_file.exists():
            self.sb.bash("stop.sh", ["--timeout", "5"], timeout=60)

    def serve(self, extra=(), *, timeout=150, env=None, preset=PRESET):
        return self.sb.bash("serve.sh", [preset] + list(extra), timeout=timeout, env=env)

    def assert_server_path(self):
        """Checks the server process as the vllm stub saw it at its last launch.

        PATH is the venv bin, then the ROCm bin, then the PATH serve.sh was given. The .hip kernels are
        built at first use, so ninja (venv) and hipcc (ROCm) must resolve from this PATH.
        """
        seen = {}
        for line in (self.sb.state / "vllm-serve.env").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            seen[key] = value
        venv_bin = msys(self.sb.rdh / "venv" / "bin")
        rocm_bin = msys(self.sb.rocm / "bin")
        self.assertEqual(seen.get("PATH"), f"{venv_bin}:{rocm_bin}:{self.sb.env()['PATH']}",
                         "the server PATH is not venv, ROCm, then the previous PATH")
        self.assertEqual(seen.get("ninja"), f"{venv_bin}/ninja", "ninja does not resolve to the venv")
        self.assertEqual(seen.get("hipcc"), f"{rocm_bin}/hipcc", "hipcc does not resolve to ROCm")

    # ----- success path ----------------------------------------------------------------
    def test_success_prints_endpoint_root_kv_and_log(self):
        r = self.serve(["--tries", "1", "--start-timeout", "20"])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertIn("127.0.0.1:8100", r.out)
        self.assertIn("stub-root", r.out, "model root from /v1/models is not printed")
        self.assertIn("GPU KV cache size", r.out)
        logs = list((self.sb.rdh / "logs").glob("serve-qwen3.8-27b-*.log"))
        self.assertEqual(len(logs), 1, "expected one serve log logs/serve-<preset>-<time>.log")
        self.assertEqual(starts(self.sb), 1)

    def test_pid_file_names_the_running_server(self):
        r = self.serve(["--tries", "1", "--start-timeout", "20"])
        self.assertEqual(r.rc, 0, r.describe())
        text = self.pid_file.read_text().strip()
        self.assertTrue(text.isdigit(), f"run/vllm.pid is not a PID: {text!r}")
        self.assertTrue(self.sb.bash_alive(int(text)), "the PID in run/vllm.pid is not alive")

    def test_readiness_polls_models_with_five_second_limit(self):
        self.serve(["--tries", "1", "--start-timeout", "20"])
        polls = models_polls(self.sb)
        self.assertGreaterEqual(len(polls), 1, "/v1/models was not polled")
        row = polls[0]
        self.assertIn("-m", row)
        self.assertEqual(row[row.index("-m") + 1], "5", row)
        self.assertTrue("-sf" in row or ("-s" in row and "-f" in row), row)
        self.assertIn(f"http://127.0.0.1:{RESERVED_PORT}/v1/models", row)

    def test_probe_request_matches_spec(self):
        # The probe is POST /v1/chat/completions with the model id, one user message "Reply with
        # one word.", max_tokens 8 and temperature 0. curl runs with -sf and a 300 s limit.
        # A real vLLM reports the served name as the model id; the stub does the same.
        self.sb.set("models.json", '{"object":"list","data":[{"id":"qwen3.8-27b","root":"stub-root"}]}')
        r = self.serve(["--tries", "1", "--start-timeout", "20"])
        self.assertEqual(r.rc, 0, r.describe())
        rows = chats(self.sb)
        self.assertEqual(len(rows), 1, "expected exactly one probe request")
        row = rows[0]
        self.assertEqual(row[row.index("-m") + 1], "300", row)
        self.assertTrue("-sf" in row or ("-s" in row and "-f" in row), row)
        body = next(a for a in row[1:] if a.startswith("{"))
        payload = json.loads(body)
        self.assertEqual(payload["model"], "qwen3.8-27b")
        self.assertEqual(payload["messages"][0]["content"], "Reply with one word.")
        self.assertEqual(payload["max_tokens"], 8)
        self.assertEqual(payload["temperature"], 0)
        self.assertRegex(r.out, re.compile(r"probe.*\d"), "the probe time is not printed")

    def test_no_probe_skips_the_chat_request(self):
        r = self.serve(["--tries", "1", "--start-timeout", "20", "--no-probe"])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertEqual(chats(self.sb), [], "--no-probe still sent a chat request")

    def test_port_option_reaches_the_endpoint_and_probe(self):
        r = self.serve(["--tries", "1", "--start-timeout", "20", "--port", "9100"])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertTrue(any("http://127.0.0.1:9100/v1/models" in row for row in models_polls(self.sb)))
        self.assertTrue(any("http://127.0.0.1:9100/v1/chat/completions" in " ".join(row) for row in chats(self.sb)))

    def test_host_0_0_0_0_prints_authentication_warning(self):
        r = self.serve(["--tries", "1", "--start-timeout", "20", "--host", "0.0.0.0"])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertIn("0.0.0.0", r.both)
        self.assertRegex(r.both, re.compile(r"authentic|автентифік", re.I))

    # ----- failed attempts and recovery ------------------------------------------------
    def test_probe_error_fails_attempt_and_next_attempt_follows_driver_check(self):
        self.sb.seq("chat", ["fail", "ok"])
        r = self.serve(["--tries", "2", "--start-timeout", "20"])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertEqual(starts(self.sb), 2)
        order = event_order(self.sb)
        first_start = order.index("start")
        second_start = order.index("start", first_start + 1)
        self.assertIn("probe", order[first_start + 1:second_start],
                      "no driver check between the failed attempt and the next one")

    def test_empty_choices_is_a_failed_attempt(self):
        self.sb.seq("chat", ["nochoices", "ok"])
        r = self.serve(["--tries", "2", "--start-timeout", "20"])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertEqual(starts(self.sb), 2)

    def test_illegal_memory_access_in_log_fails_attempt(self):
        self.sb.seq("vllm", ["illegal", "serve"])
        r = self.serve(["--tries", "2", "--start-timeout", "20"])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertEqual(starts(self.sb), 2)

    def test_engine_fatal_error_in_log_fails_attempt(self):
        self.sb.seq("vllm", ["fatal", "serve"])
        r = self.serve(["--tries", "2", "--start-timeout", "20"])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertEqual(starts(self.sb), 2)

    def test_process_exit_fails_attempt(self):
        self.sb.seq("vllm", ["exit1", "serve"])
        r = self.serve(["--tries", "2", "--start-timeout", "20"])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertEqual(starts(self.sb), 2)

    def test_start_timeout_fails_attempt_and_is_honoured(self):
        self.sb.seq("vllm", ["silent"])
        began = time.monotonic()
        r = self.serve(["--tries", "1", "--start-timeout", "1"])
        elapsed = time.monotonic() - began
        self.assertEqual(r.rc, 5, r.describe())
        self.assertLess(elapsed, 60, "--start-timeout 1 did not limit the wait")

    def test_all_attempts_failing_is_code5_with_three_tries_by_default(self):
        self.sb.seq("vllm", ["exit1"])
        r = self.serve(["--start-timeout", "20"])
        self.assertEqual(r.rc, 5, r.describe())
        self.assertEqual(starts(self.sb), 3, "default --tries is 3")

    def test_tries_option_limits_attempts(self):
        self.sb.seq("vllm", ["exit1"])
        r = self.serve(["--tries", "2", "--start-timeout", "20"])
        self.assertEqual(r.rc, 5, r.describe())
        self.assertEqual(starts(self.sb), 2)

    # ----- driver gate: code 4 ---------------------------------------------------------
    def test_driver_failure_before_start_is_code4_and_nothing_starts(self):
        self.sb.seq("probe", ["wrong"])
        r = self.serve(["--tries", "1", "--start-timeout", "20"])
        self.assertEqual(r.rc, 4, r.describe())
        self.assertEqual(starts(self.sb), 0, "a server was started on a failing driver")

    def test_driver_failure_after_failed_attempt_is_code4(self):
        self.sb.seq("probe", ["ok", "nan"])
        self.sb.seq("vllm", ["exit1"])
        r = self.serve(["--tries", "3", "--start-timeout", "20"])
        self.assertEqual(r.rc, 4, r.describe())
        self.assertEqual(starts(self.sb), 1, "the next attempt ran after a failed driver check")

    # ----- install guard: code 3 -------------------------------------------------------
    def test_missing_install_marker_is_code3(self):
        self.sb.marker().unlink()
        r = self.serve(["--tries", "1", "--start-timeout", "20"])
        self.assertEqual(r.rc, 3, r.describe())
        self.assertEqual(starts(self.sb), 0)

    # ----- previous server still running: code 6 --------------------------------------
    def test_stale_pid_file_is_ignored_and_server_starts(self):
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text("2999999\n")
        r = self.serve(["--tries", "1", "--start-timeout", "20"])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertEqual(starts(self.sb), 1)

    def test_live_previous_server_is_stopped_before_start(self):
        pid, popen, _ = self.sb.spawn("true", name="old-server", argv0=VLLM_ARGV0)
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(f"{pid}\n")
        r = self.serve(["--tries", "1", "--start-timeout", "20"])
        self.assertEqual(r.rc, 0, r.describe())
        popen.wait(timeout=30)  # reap the stopped process before testing its PID
        self.assertFalse(self.sb.bash_alive(pid), "the previous server was not stopped")
        self.assertEqual(starts(self.sb), 1)

    def test_pid_file_of_an_unrelated_live_process_gets_no_signal_and_server_starts(self):
        # The PID file can outlive its server, and the number can belong to another process by then.
        # serve.sh must not signal that process: the stub exits on SIGTERM, so it stays alive only if
        # no signal was sent.
        pid, popen, _ = self.sb.spawn("true", name="unrelated")
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(f"{pid}\n")
        r = self.serve(["--tries", "1", "--start-timeout", "20"])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertTrue(self.sb.bash_alive(pid), "serve.sh signalled a process that is not vLLM")
        self.assertEqual(starts(self.sb), 1)

    def test_previous_server_that_survives_stop_is_code6_and_nothing_starts(self):
        # A previous server that does not stop gives exit code 6, and no new server is started.
        # serve.sh calls stop.sh with --timeout 60 here, so this test takes about one minute.
        pid, popen, _ = self.sb.spawn("true", name="stuck-server", ignore_term=True, argv0=VLLM_ARGV0)
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(f"{pid}\n")
        r = self.serve(["--tries", "1", "--start-timeout", "20"], timeout=200)
        self.assertEqual(r.rc, 6, r.describe())
        self.assertEqual(starts(self.sb), 0, "a new server started next to a live one")
        self.assertTrue(self.sb.bash_alive(pid), "the test's survivor was removed; the test is invalid")

    # ----- --kill-stuck ----------------------------------------------------------------
    def stuck_after_failed_attempt(self, *flags):
        """The attempt fails, its server ignores SIGTERM, and a rank named by the pgrep stub is stuck."""
        pid, popen, _ = self.sb.spawn("true", name="stuck-rank", ignore_term=True, argv0="VLLM::Worker_TP1")
        write_script(self.sb.stubs / "pgrep", f'case "$*" in *VLLM::Worker*) echo {pid} ;; *) exit 1 ;; esac\n')
        kfd = self.sb.root / "kfd"
        (kfd / "proc" / str(pid)).mkdir(parents=True)
        (kfd / "proc" / str(pid) / "vram_100").write_text("4096\n")
        (kfd / "topology" / "nodes" / "1").mkdir(parents=True)
        (kfd / "topology" / "nodes" / "1" / "gpu_id").write_text("100\n")
        (kfd / "topology" / "nodes" / "1" / "properties").write_text("location_id 1792\ndomain 0\n")
        (self.sb.state / "journal.txt").write_text("kernel: boot line\n", encoding="utf-8")
        self.sb.seq("vllm", ["ignoreterm"])
        self.addCleanup(lambda: (self.sb.state / "stop-now").touch())
        r = self.serve(["--tries", "2", "--start-timeout", "1", *flags], timeout=500,
                       env={"RT_MAX_LIFE": "400", "RR_KFD_ROOT": msys(kfd)})
        return r, pid, popen

    def test_kill_stuck_forces_the_rank_between_attempts(self):
        r, pid, popen = self.stuck_after_failed_attempt("--kill-stuck")
        self.assertIn("running kill-stuck-rank.sh --force", r.both, r.describe())
        popen.wait(timeout=30)
        self.assertFalse(self.sb.bash_alive(pid), "--kill-stuck left the rank alive")

    def test_without_kill_stuck_the_rank_stays(self):
        r, pid, _ = self.stuck_after_failed_attempt()
        self.assertEqual(r.rc, 6, r.describe())
        self.assertNotIn("kill-stuck-rank.sh --force", r.both)
        self.assertTrue(self.sb.bash_alive(pid), "a rank was removed without --kill-stuck")

    # ----- foreground for systemd -----------------------------------------------------
    def test_foreground_runs_once_without_probe_or_driver_check(self):
        r = self.serve(["--foreground"], timeout=60, env={"RT_MAX_LIFE": "2"})
        self.assertEqual(r.rc, 0, r.describe())
        self.assertEqual(starts(self.sb), 1)
        self.assertEqual(chats(self.sb), [], "--foreground sent a probe")
        self.assertEqual(probes(self.sb), [], "--foreground ran the driver check")
        text = self.pid_file.read_text().strip()
        self.assertTrue(text.isdigit(), f"run/vllm.pid is not a PID: {text!r}")

    def test_foreground_failure_is_not_retried(self):
        self.sb.seq("vllm", ["exit1"])
        r = self.serve(["--foreground"], timeout=60, env={"RT_MAX_LIFE": "2"})
        self.assertNotEqual(r.rc, 0, r.describe())
        self.assertEqual(starts(self.sb), 1, "--foreground retried a failed start")

    # ----- PATH of the server process --------------------------------------------------
    def test_background_server_sees_venv_and_rocm_first_on_path(self):
        r = self.serve(["--tries", "1", "--start-timeout", "20"])
        self.assertEqual(r.rc, 0, r.describe())
        self.assert_server_path()

    def test_foreground_server_sees_venv_and_rocm_first_on_path(self):
        r = self.serve(["--foreground"], timeout=60, env={"RT_MAX_LIFE": "2"})
        self.assertEqual(r.rc, 0, r.describe())
        self.assert_server_path()

    # ----- /dev/shm cleanup (real directory) ------------------------------------------
    @unittest.skipIf(os.name == "nt",
                     "clean_shm works on the real /dev/shm; on Windows (Git Bash) that is a "
                     "shared directory of the host, so the test is skipped here")
    def test_unowned_shm_regions_are_removed_and_held_ones_kept(self):
        shm = pathlib.Path("/dev/shm")
        own = shm / f"vllm_offload_rt-own-{os.getpid()}.mmap"
        kept = shm / f"vllm_host_ar_rt-keep-{os.getpid()}"
        own.write_bytes(b"x")
        kept.write_bytes(b"x")
        self.addCleanup(lambda: [p.unlink() for p in (own, kept) if p.exists()])
        r = self.serve(["--tries", "1", "--start-timeout", "20"])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertFalse(own.exists(), "a region that nothing holds was not removed")
        self.assertTrue(kept.exists(), "a region that is held was removed")

