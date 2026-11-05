"""stop.sh, function alive(): a zombie leader, the members of its process group, and a ps that does not answer.

kill -0 succeeds on a zombie until its parent collects the exit code, and a container without an init never
collects it. alive() therefore reads "ps -A -o pid=,pgid=,stat=": a process whose stat starts with Z has exited,
a zombie leader with live threads (stat Zl) has not, and a live member of the group keeps the group alive.
Without a usable ps, the answer of kill -0 stands.

The cases follow the specification:
  a) SIGTERM ends the leader, which stays a zombie, and nothing else is alive in its group: exit 0 at once,
     "stopped PID <pid>", PID file removed.
  b) the leader is a zombie and a member of its group ignores SIGTERM: the wait runs to the limit, exit 1 with
     "still running", PID file kept, and the member gets no signal beyond SIGTERM.
  c) the leader is a zombie with live threads (ps stat "Zl"): alive, exit 1 at the limit.
  d) ps does not answer (stub, exit 127): kill -0 decides. A live process stays alive; a zombie leader counts as
     alive too, because kill -0 succeeds on it.

Cases a), b) and the zombie part of d) need a real zombie and POSIX process groups. They skip themselves on
Windows (Git Bash keeps no zombies) and on a POSIX host where ps does not report an exited child as stat Z.
Cases c) and the live part of d) stub ps and run on every platform: the process is alive and ignores SIGTERM,
so the outcome depends on the answer of ps (or on kill -0), not on how a zombie is reaped.

Processes of this file and their end. Every background bash process is started by StopCase.start(), and its
cleanup (end_process) is registered before the process runs, so it runs when the test passes, fails or raises.
The cleanup creates the stop file, waits up to STOP_WAIT seconds for the process to end, then sends SIGHUP to the
process and its group (the members ignore only SIGTERM, and SIGKILL is never sent), and it reports a leftover as
a test error when the process still runs after that. A process also ends on its own: its loop stops when its
stop file appears, when the sandbox state directory is removed, or after LIFE_SECONDS, so a cleanup that never
runs cannot leave the process behind for longer than that.
"""

import os
import subprocess
import time

from _support import BASH, ReleaseTest, msys

VLLM_ARGV0 = "/opt/rdna2/venv/bin/vllm serve stub-model"
NO_ZOMBIES = ("a real zombie needs POSIX process groups and zombie states; Git Bash on Windows keeps no "
              "exited process as a zombie, so the state cannot be produced there")

# Longest a helper process lives on its own (seconds). A test runs for well under this.
LIFE_SECONDS = 120
# Seconds the cleanup gives a process to end on its stop file before it sends SIGHUP.
STOP_WAIT = 10

# start() runs the body after the SIGTERM trap is set, so the ready file means the leader can take SIGTERM.
# The file must not be empty: wait_file() reads a non-empty file as "written".
LEADER_BODY = 'echo ready > "$RT_STATE/leader.ready"'
# A member of the leader's process group that ignores SIGTERM and lives until member.stop exists or its loop
# runs out (see loop()). It keeps the group alive after the leader has exited. exec -a gives it a vLLM command
# line, so stop.sh treats the group as vLLM. The bash path comes from bash.path, not from $BASH: under exec -a
# the leader's $BASH is its argv[0] (the vLLM name), and exec of that name fails with rc 127.
MEMBER_BODY = r'''( trap '' TERM; exec -a '/opt/rdna2/venv/bin/vllm serve member' "$(cat "$RT_STATE/bash.path")" -c '@CHILD_LOOP@' bash "$RT_STATE/member.stop" ) &
echo $! > "$RT_STATE/member.pid"
echo ready > "$RT_STATE/leader.ready"'''
# Leader that stays alive after SIGTERM but leaves a flag. With a stubbed ps the flag turns its row into Z, which
# is the process table that a real zombie would show, on a host without zombies (Windows).
TERM_FLAG_BODY = r'''trap 'echo 1 > "$RT_STATE/leader.term"' TERM
echo ready > "$RT_STATE/leader.ready"'''
# Every ps call is logged to RT_CALLS, so a test can show that alive() asked ps at all.
PS_LOG = r'''printf 'ps\t%s\n' "$*" >> "$RT_CALLS"'''
PS_NOT_AVAILABLE = "echo 'ps: command not found (stub)' >&2\nexit 127\n"

_probe_reason = None


def loop(stop_arg):
    """Bash loop that ends when the file named by stop_arg exists, when the sandbox state directory is removed,
    or when LIFE_SECONDS have passed. stop_arg is the shell expression of the stop file, e.g. "$2"."""
    return ('while [ ! -e "' + stop_arg + '" ] && [ -d "$RT_STATE" ] && [ "$SECONDS" -lt ' + str(LIFE_SECONDS)
            + ' ]; do "$RT_REAL_SLEEP" 0.1; done')


def member_body():
    return MEMBER_BODY.replace("@CHILD_LOOP@", loop("$1"))


def read_pid(path):
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def stat_of(pid):
    """The stat column of pid in the real ps. "" when the row is missing; None when ps cannot run."""
    try:
        proc = subprocess.run(["ps", "-A", "-o", "pid=,pgid=,stat="], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[0] == str(pid):
            return fields[2]
    return ""


def zombie_probe_reason():
    """"" when this host reports an exited, uncollected child with stat Z; otherwise the reason to skip."""
    global _probe_reason
    if _probe_reason is None:
        _probe_reason = _run_zombie_probe()
    return _probe_reason


def _run_zombie_probe():
    if os.name == "nt":
        return NO_ZOMBIES
    child = subprocess.Popen([BASH, "-c", "exit 0"], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        stat = ""
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            stat = stat_of(child.pid)
            if stat is None:
                return "ps -A -o pid=,pgid=,stat= does not run on this host"
            if stat.startswith("Z"):
                return ""
            time.sleep(0.05)
        return f"ps does not report an exited child as stat Z on this host (last stat: {stat!r})"
    finally:
        # Collected only after the probe has read the state; the probe never signals anything.
        child.wait(timeout=30)


class StopCase(ReleaseTest):

    def setUp(self):
        super().setUp()
        self.pid_file = self.sb.rdh / "run" / "vllm.pid"

    def write_pid(self, pid):
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(f"{pid}\n")

    def wait_file(self, path, what, limit=20.0):
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            if path.exists() and path.read_text().strip():
                return path.read_text().strip()
            time.sleep(0.05)
        self.fail(f"{what} did not write {path.name} within {limit:.0f} s")

    def timed_stop(self, args, timeout=90):
        began = time.monotonic()
        r = self.sb.bash("stop.sh", list(args), timeout=timeout)
        return r, time.monotonic() - began

    def stub_ps(self, body):
        self.sb.add_stub("ps", PS_LOG + "\n" + body)

    # ----- helper processes ------------------------------------------------------------
    def start(self, body, name, *, argv0=None, ignore_term=False):
        """Starts a bash process of this test; returns (pid, popen).

        The process writes its PID to <name>.pid and ends when <name>.stop exists, when the sandbox state
        directory is removed, or after LIFE_SECONDS. argv0 gives it a command line that stop.sh reads as vLLM.
        Its cleanup is registered before it starts (see end_process).
        """
        stop_file = self.sb.state / f"{name}.stop"
        pid_file = self.sb.state / f"{name}.pid"
        prologue = "trap '' TERM; " if ignore_term else "trap 'exit 0' TERM; "
        inner = f'echo $$ > "$1"; {prologue}{body}; {loop("$2")}'
        if argv0:
            # The outer bash keeps its own $BASH; the inner one gets argv0 and keeps the same PID.
            argv = [BASH, "-c", 'exec -a "$0" "$BASH" -c "$1" bash "$2" "$3"', argv0, inner,
                    msys(pid_file), msys(stop_file)]
        else:
            argv = [BASH, "-c", inner, "bash", msys(pid_file), msys(stop_file)]
        popen = subprocess.Popen(argv, env=self.sb.env(), stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 start_new_session=(os.name != "nt"))
        self.addCleanup(self.end_process, name, popen)
        return self.wait_pid(name, popen), popen

    def wait_pid(self, name, popen):
        pid_file = self.sb.state / f"{name}.pid"
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            pid = read_pid(pid_file)
            if pid is not None:
                return pid
            if popen.poll() is not None:
                self.fail(f"process {name} exited with rc {popen.returncode} before it wrote its PID")
            time.sleep(0.05)
        self.fail(f"process {name} did not write {pid_file.name} within 20 s")

    def process_alive(self, pid):
        if not self.sb.bash_alive(pid):
            return False
        return os.name == "nt" or not (stat_of(pid) or "").startswith("Z")

    def wait_for_end(self, pid, popen, seconds):
        deadline = time.monotonic() + seconds
        while True:
            if popen is not None:
                if popen.poll() is not None:
                    return True
            elif pid is None or not self.process_alive(pid):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.2)

    def end_process(self, name, popen=None):
        """Cleanup of a process of this test; runs whether the test passed, failed or raised.

        1. The stop file is created, so the loop of the process ends at its next check. If the state directory
           is already gone, the process ends by its own directory check instead.
        2. The cleanup waits STOP_WAIT seconds for the process to end. popen is None for a process that is not a
           child of this test (a member of a group), and its end is checked by PID.
        3. A process that still runs gets SIGHUP, to its PID and to its process group. Members ignore only
           SIGTERM, and SIGKILL is never sent.
        4. A process that still runs 5 s later fails the cleanup and is reported as a leftover.
        """
        try:
            (self.sb.state / f"{name}.stop").touch()
        except OSError:
            pass
        pid = read_pid(self.sb.state / f"{name}.pid")
        if popen is None and pid is None:
            return   # the process never started (the test failed before it)
        if self.wait_for_end(pid, popen, STOP_WAIT):
            return
        if pid is not None:
            self.sb.run([BASH, "-c", 'kill -HUP -- "-$1" 2>/dev/null; kill -HUP "$1" 2>/dev/null; exit 0',
                         "bash", str(pid)], timeout=20)
        if not self.wait_for_end(pid, popen, 5):
            self.fail(f"leftover: process {name} (PID {pid}) still runs after the cleanup; SIGHUP did not end it")


class ZombieTest(StopCase):
    """Real zombies. Skipped where the host cannot produce them (see the module docstring)."""

    def setUp(self):
        super().setUp()
        reason = zombie_probe_reason()
        if reason:
            self.skipTest(reason)

    def test_zombie_leader_without_group_members_stops_at_once_with_exit_0(self):
        # The leader exits on SIGTERM and nobody collects it: kill -0 succeeds on it from then on.
        pid, _ = self.start(LEADER_BODY, "leader", argv0=VLLM_ARGV0)
        self.wait_file(self.sb.state / "leader.ready", "the leader")
        self.write_pid(pid)
        r, elapsed = self.timed_stop(["--timeout", "30"])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertIn(f"stopped PID {pid}", r.out)
        self.assertFalse(self.pid_file.exists(), "the PID file was kept after stop.sh reported the stop")
        self.assertLess(elapsed, 15.0, f"stop.sh took {elapsed:.1f} s for a zombie leader (--timeout 30)")
        self.assertTrue((stat_of(pid) or "").startswith("Z"),
                        "the leader is not a zombie after stop.sh: the case under test did not occur")

    def test_zombie_leader_with_live_group_member_waits_to_the_limit_and_exits_1(self):
        # The member is not a child of this test; its cleanup is registered here, so it runs after the leader's.
        self.addCleanup(self.end_process, "member")
        (self.sb.state / "bash.path").write_text(msys(BASH) + "\n")   # read by MEMBER_BODY, see there
        pid, _ = self.start(member_body(), "leader", argv0=VLLM_ARGV0)
        member = int(self.wait_file(self.sb.state / "member.pid", "the group member"))
        self.wait_file(self.sb.state / "leader.ready", "the leader")
        self.write_pid(pid)
        r, elapsed = self.timed_stop(["--timeout", "4"])
        self.assertEqual(r.rc, 1, r.describe())
        self.assertIn("still running", r.both)
        self.assertNotIn("stopped PID", r.out)
        self.assertTrue(self.pid_file.exists(), "the PID file was removed while a member of the group is alive")
        self.assertTrue(self.sb.bash_alive(member),
                        "the group member is gone: it got a signal beyond SIGTERM (SIGKILL?)")
        self.assertGreaterEqual(elapsed, 3.0, f"stop.sh returned after {elapsed:.1f} s, before the 4 s limit")
        self.assertTrue((stat_of(pid) or "").startswith("Z"),
                        "the leader is not a zombie: the case under test did not occur")

    def test_ps_unavailable_zombie_leader_counts_as_alive_by_kill0(self):
        # ps fails, so kill -0 decides, and kill -0 succeeds on the zombie leader.
        pid, _ = self.start(LEADER_BODY, "leader", argv0=VLLM_ARGV0)
        self.wait_file(self.sb.state / "leader.ready", "the leader")
        self.stub_ps(PS_NOT_AVAILABLE)
        self.write_pid(pid)
        r, elapsed = self.timed_stop(["--timeout", "3"])
        self.assertEqual(r.rc, 1, r.describe())
        self.assertIn("still running", r.both)
        self.assertTrue(self.pid_file.exists(), "the PID file was removed although kill -0 finds the process")
        self.assertGreaterEqual(len(self.sb.calls("ps")), 1, "the ps stub was not run: the fallback was not tested")
        self.assertGreaterEqual(elapsed, 2.0, f"stop.sh returned after {elapsed:.1f} s, before the 3 s limit")
        self.assertTrue((stat_of(pid) or "").startswith("Z"),
                        "the leader is not a zombie: the case under test did not occur")


class PsStubTest(StopCase):
    """ps stubbed; the process is alive and ignores SIGTERM. Runs on every platform."""

    def start_leader(self):
        pid, _ = self.start(LEADER_BODY, "leader", argv0=VLLM_ARGV0, ignore_term=True)
        self.wait_file(self.sb.state / "leader.ready", "the leader")
        return pid

    def start_term_flag_leader(self):
        pid, _ = self.start(TERM_FLAG_BODY, "leader", argv0=VLLM_ARGV0)
        self.wait_file(self.sb.state / "leader.ready", "the leader")
        return pid

    def test_leader_that_ps_reports_as_zombie_after_sigterm_stops_at_once_with_exit_0(self):
        # Stub form of case a): after SIGTERM, ps lists the leader as Z and no group member exists.
        # kill -0 still succeeds on it, so a stop.sh that trusts kill -0 alone would wait the whole limit.
        pid = self.start_term_flag_leader()
        self.stub_ps(f'if [ -e "$RT_STATE/leader.term" ]; then echo "{pid} {pid} Z"; '
                     f'else echo "{pid} {pid} S"; fi\n')
        self.write_pid(pid)
        r, elapsed = self.timed_stop(["--timeout", "30"])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertIn(f"stopped PID {pid}", r.out)
        self.assertFalse(self.pid_file.exists(), "the PID file was kept after stop.sh reported the stop")
        self.assertTrue((self.sb.state / "leader.term").exists(), "SIGTERM never reached the leader")
        self.assertGreater(len(self.sb.calls("ps")), 0, "the ps stub was not run: alive() was not tested")
        self.assertLess(elapsed, 15.0, f"stop.sh took {elapsed:.1f} s for a zombie leader (--timeout 30)")

    def test_zombie_leader_with_a_live_member_of_its_group_waits_to_the_limit_and_exits_1(self):
        # Stub form of case b): the leader is Z after SIGTERM, a member with pgid = leader PID ignores SIGTERM.
        # Only the member keeps the group alive, so the check must read the pgid column, not the leader row.
        pid = self.start_term_flag_leader()
        member, _ = self.start("true", "stubmember", ignore_term=True)
        self.stub_ps(f'if [ -e "$RT_STATE/leader.term" ]; then echo "{pid} {pid} Z"; '
                     f'else echo "{pid} {pid} S"; fi\n'
                     f'echo "{member} {pid} S"\n')
        self.write_pid(pid)
        r, elapsed = self.timed_stop(["--timeout", "4"])
        self.assertEqual(r.rc, 1, r.describe())
        self.assertIn("still running", r.both)
        self.assertNotIn("stopped PID", r.out)
        self.assertTrue(self.pid_file.exists(), "the PID file was removed while a group member is alive")
        self.assertTrue(self.sb.bash_alive(member), "the group member is gone: it got a signal beyond SIGTERM")
        self.assertGreaterEqual(elapsed, 3.0, f"stop.sh returned after {elapsed:.1f} s, before the 4 s limit")

    def test_zombie_leader_with_live_threads_stat_Zl_is_alive(self):
        pid = self.start_leader()
        self.stub_ps(f"echo '{pid} {pid} Zl'\n")
        self.write_pid(pid)
        r, elapsed = self.timed_stop(["--timeout", "3"])
        self.assertEqual(r.rc, 1, r.describe())
        self.assertIn("still running", r.both)
        self.assertNotIn("stopped PID", r.out)
        self.assertTrue(self.pid_file.exists(), "the PID file was removed although ps reports the leader as Zl")
        self.assertGreater(len(self.sb.calls("ps")), 0, "the ps stub was not run: alive() was not tested")
        self.assertTrue(self.sb.bash_alive(pid),
                        "the leader is gone: it was ended by a signal other than SIGTERM (SIGKILL?)")
        self.assertGreaterEqual(elapsed, 1.5, f"stop.sh returned after {elapsed:.1f} s, before the 3 s limit")

    def test_ps_unavailable_live_process_counts_as_alive_by_kill0(self):
        pid = self.start_leader()
        self.stub_ps(PS_NOT_AVAILABLE)
        self.write_pid(pid)
        r, elapsed = self.timed_stop(["--timeout", "3"])
        self.assertEqual(r.rc, 1, r.describe())
        self.assertIn("still running", r.both)
        self.assertNotIn("stopped PID", r.out)
        self.assertTrue(self.pid_file.exists(), "the PID file was removed although kill -0 finds the process")
        self.assertGreater(len(self.sb.calls("ps")), 0, "the ps stub was not run: the fallback was not tested")
        self.assertTrue(self.sb.bash_alive(pid),
                        "the process is gone: it was ended by a signal other than SIGTERM (SIGKILL?)")
        self.assertGreaterEqual(elapsed, 1.5, f"stop.sh returned after {elapsed:.1f} s, before the 3 s limit")
