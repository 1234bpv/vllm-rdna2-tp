"""stop.sh: exit codes, SIGTERM to the server and its process group, the wait limit, no SIGKILL, and the
check that the PID belongs to a vLLM process before any signal is sent.

The server PID comes from run/vllm.pid. Exit code 0 when no server is left running; 1 when a process
survives the wait, in which case its PID is printed with pointers to docs/TROUBLESHOOTING.md and
scripts/kill-stuck-rank.sh; 2 for a bad call. The default wait is 60 seconds, as --help shows. A process
that ignores SIGTERM is a test stub: SIGKILL cannot be trapped, so a stub that survives stop.sh and is
still alive afterwards shows that no SIGKILL was sent within the limit.

Before it signals, stop.sh reads /proc/PID/cmdline and the process group (pgrep -a -g). A PID whose
command line names bin/vllm or VLLM:: is a vLLM process; any other live PID gets no signal, the PID file
is removed, and the exit code is 0. The stubs that stand for a server or a rank set argv0 with exec -a,
so their /proc/PID/cmdline carries those names.

A process group that is alive, whose leader is gone, and whose members cannot be read (pgrep finds none)
is refused with exit 1: the PID file stays and nothing is signalled. A vLLM process of another user is
refused the same way, with "of another user". The second case can only be reproduced as root on POSIX.
"""

import os
import re
import shutil
import signal
import subprocess
import time
import unittest

from _support import BASH, RELEASE, ReleaseTest, msys

VLLM_ARGV0 = "/opt/rdna2/venv/bin/vllm serve stub-model"
WORKER_ARGV0 = "VLLM::Worker_TP0"


class StopTest(ReleaseTest):

    def setUp(self):
        super().setUp()
        self.pid_file = self.sb.rdh / "run" / "vllm.pid"

    def write_pid(self, pid):
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(f"{pid}\n")

    def stop(self, args=(), *, timeout=120):
        return self.sb.bash("stop.sh", list(args), timeout=timeout)

    # ----- usage -----------------------------------------------------------------------
    def test_help_exits_zero(self):
        r = self.stop(["--help"], timeout=60)
        self.assertEqual(r.rc, 0, r.describe())
        self.assertIn("--timeout", r.out)

    def test_bad_usage_exits_2(self):
        bad = [["--timeout", "0"], ["--timeout", "abc"], ["--timeout", "-5"], ["--timeout"],
               ["--bogus"], ["extra-word"]]
        for args in bad:
            with self.subTest(args=args):
                r = self.stop(args, timeout=60)
                self.assertEqual(r.rc, 2, r.describe())

    # ----- nothing to stop -------------------------------------------------------------
    def test_no_pid_file_exits_zero(self):
        r = self.stop(timeout=60)
        self.assertEqual(r.rc, 0, r.describe())

    def test_stale_pid_file_exits_zero(self):
        self.write_pid(2999999)
        r = self.stop(timeout=60)
        self.assertEqual(r.rc, 0, r.describe())

    # ----- the PID must belong to a vLLM process ----------------------------------------
    def test_foreign_live_process_gets_no_signal_and_the_pid_file_is_removed(self):
        # The stub exits on SIGTERM (trap 'exit 0' TERM), so a signal would end it: staying alive shows
        # that no signal was sent.
        pid, popen, _ = self.sb.spawn("true", name="foreign")
        self.write_pid(pid)
        r = self.stop(["--timeout", "10"], timeout=60)
        self.assertEqual(r.rc, 0, r.describe())
        self.assertIn("is not a vLLM process", r.out)
        self.assertTrue(self.sb.bash_alive(pid), "a process that is not vLLM was signalled")
        self.assertFalse(self.pid_file.exists(), "the PID file of a non-vLLM PID was kept")

    def test_foreign_process_that_ignores_sigterm_is_not_waited_for(self):
        # If stop.sh took this PID for a server, it would wait the full --timeout and exit 1.
        pid, popen, _ = self.sb.spawn("true", name="foreign-stuck", ignore_term=True)
        self.write_pid(pid)
        began = time.monotonic()
        r = self.stop(["--timeout", "30"], timeout=60)
        elapsed = time.monotonic() - began
        self.assertEqual(r.rc, 0, r.describe())
        self.assertLess(elapsed, 20, "stop.sh waited for a process that is not vLLM")
        self.assertTrue(self.sb.bash_alive(pid), "a process that is not vLLM was signalled")
        self.assertFalse(self.pid_file.exists())

    @unittest.skipIf(os.name == "nt" or not shutil.which("pgrep"),
                     "needs pgrep and process groups: procps on Linux; Git Bash has neither")
    def test_group_member_named_vllm_makes_the_group_a_vllm_process(self):
        # The parent's own command line has no vLLM name; a child in its process group does. The
        # group check (pgrep -a -g) must find it, so the parent and the child both get SIGTERM.
        body = "( exec -a '/opt/rdna2/venv/bin/vllm serve group' \"$RT_REAL_SLEEP\" 1000 ) & echo $! > \"$RT_STATE/child.pid\""
        pid, popen, _ = self.sb.spawn(body, name="group-probe")
        child = None
        for _ in range(200):
            path = self.sb.state / "child.pid"
            text = path.read_text().strip() if path.exists() else ""
            if text.isdigit():
                child = int(text)
                break
            time.sleep(0.05)
        self.assertIsNotNone(child, "the child process did not start")
        self.write_pid(pid)
        r = self.stop(["--timeout", "10"], timeout=60)
        self.assertEqual(r.rc, 0, r.describe())
        self.assertIn("stopped PID", r.out)
        popen.wait(timeout=30)
        self.assertFalse(self.sb.bash_alive(child), "the vLLM-named group member survived stop.sh")

    # ----- a group that cannot be read, and a process of another user ------------------
    @unittest.skipIf(os.name == "nt",
                     "a group whose leader has exited needs POSIX process groups; Git Bash does not "
                     "reproduce that state")
    def test_group_without_its_leader_or_members_is_code1_and_keeps_the_pid_file(self):
        # The leader exits and a child stays in its group: the group is alive, the PID has no process, and
        # pgrep (stubbed to find nothing) gives no members. stop.sh cannot tell what the group is, so it
        # must neither signal nor drop the PID file.
        self.sb.add_stub("pgrep", "exit 1\n")
        child_file = self.sb.state / "orphan-child.pid"
        leader = subprocess.Popen(
            [BASH, "-c", '"$RT_REAL_SLEEP" 1000 & echo $! > "$1"; exit 0', "bash", msys(child_file)],
            env=self.sb.env(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)
        leader.wait(timeout=30)
        child = int(child_file.read_text().strip())
        self.addCleanup(self._end_process, child)
        self.write_pid(leader.pid)
        r = self.stop(["--timeout", "10"], timeout=60)
        self.assertEqual(r.rc, 1, r.describe())
        self.assertIn("its members cannot be read", r.err)
        self.assertTrue(self.pid_file.exists(), "the PID file of a live group was removed")
        self.assertTrue(self.sb.bash_alive(child), "a member of the group was signalled")

    @staticmethod
    def _end_process(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass

    @unittest.skipIf(os.name == "nt" or not hasattr(os, "geteuid") or os.geteuid() != 0,
                     "reproducing 'of another user' needs root on POSIX: root starts a vLLM-named process, and "
                     "an unprivileged account runs stop.sh, because kill -0 of that process fails for it")
    def test_vllm_process_of_another_user_is_code1_and_keeps_the_pid_file(self):
        pid, popen, _ = self.sb.spawn("true", name="other-user", argv0=VLLM_ARGV0)
        self.write_pid(pid)
        os.chmod(self.sb.root, 0o755)   # the unprivileged account must reach the PID file and the release

        def unprivileged():
            os.setgid(65534)
            os.setuid(65534)

        proc = subprocess.run([BASH, str(RELEASE / "stop.sh"), "--timeout", "10"], env=self.sb.env(),
                              cwd=str(self.sb.root), capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=60, preexec_fn=unprivileged)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("of another user", proc.stderr)
        self.assertTrue(self.pid_file.exists(), "the PID file of another user's process was removed")
        self.assertTrue(self.sb.bash_alive(pid), "the other user's process was signalled")

    # ----- stopping a live server ------------------------------------------------------
    def test_sigterm_stops_a_live_server(self):
        pid, popen, _ = self.sb.spawn("true", name="server", argv0=VLLM_ARGV0)
        self.write_pid(pid)
        r = self.stop(["--timeout", "10"], timeout=60)
        self.assertEqual(r.rc, 0, r.describe())
        popen.wait(timeout=30)
        self.assertFalse(self.sb.bash_alive(pid), "the server is still alive after stop.sh exited 0")

    def test_vllm_worker_rank_is_stopped_with_sigterm(self):
        # Ranks rename themselves VLLM::Worker*; stop.sh must treat them as vLLM processes.
        pid, popen, _ = self.sb.spawn("true", name="rank", argv0=WORKER_ARGV0)
        self.write_pid(pid)
        r = self.stop(["--timeout", "10"], timeout=60)
        self.assertEqual(r.rc, 0, r.describe())
        self.assertIn("stopped PID", r.out)
        popen.wait(timeout=30)
        self.assertFalse(self.sb.bash_alive(pid), "the VLLM::Worker rank is still alive after stop.sh")

    def test_group_members_are_stopped_too(self):
        # A background child in the same process group must end with its parent.
        body = '"$RT_REAL_SLEEP" 1000 & echo $! > "$RT_STATE/child.pid"'
        pid, popen, _ = self.sb.spawn(body, name="group-parent", argv0=VLLM_ARGV0)
        child = None
        for _ in range(200):
            text = (self.sb.state / "child.pid").read_text().strip() if (self.sb.state / "child.pid").exists() else ""
            if text.isdigit():
                child = int(text)
                break
            time.sleep(0.05)
        self.assertIsNotNone(child, "the child process did not start")
        self.write_pid(pid)
        r = self.stop(["--timeout", "10"], timeout=60)
        self.assertEqual(r.rc, 0, r.describe())
        popen.wait(timeout=30)
        self.assertFalse(self.sb.bash_alive(child), "a member of the group survived stop.sh")

    def test_surviving_process_is_code1_with_pointers_and_no_sigkill(self):
        pid, popen, _ = self.sb.spawn("true", name="stuck", ignore_term=True, argv0=VLLM_ARGV0)
        self.write_pid(pid)
        began = time.monotonic()
        r = self.stop(["--timeout", "2"], timeout=60)
        elapsed = time.monotonic() - began
        self.assertEqual(r.rc, 1, r.describe())
        self.assertIn(str(pid), r.both, "the surviving PID is not printed")
        self.assertIn("docs/TROUBLESHOOTING.md", r.both)
        self.assertIn("scripts/kill-stuck-rank.sh", r.both)
        self.assertTrue(self.sb.bash_alive(pid), "the process was removed, so SIGKILL was used")
        self.assertLess(elapsed, 30, "--timeout 2 did not limit the wait")

    def test_default_timeout_is_sixty_seconds_in_help(self):
        r = self.stop(["--help"], timeout=60)
        self.assertRegex(r.out, re.compile(r"60"))
