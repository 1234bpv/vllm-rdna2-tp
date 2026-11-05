"""scripts/kill-stuck-rank.sh: gates, the single SIGTERM, the 30-second wait and no SIGKILL.

Exit codes: 0 when the process is gone after SIGTERM; 1 when the GPU check fails or the process survives
the wait; 2 for a bad call or a refused process. Without --force SIGKILL is never sent: a survivor is left
to a person, as stop.sh leaves it. With --force a rank that survives gets SIGKILL only if the checks pass. The stub processes are started with exec -a, so /proc/PID/cmdline shows the vLLM
names the script looks for: VLLM::Worker*, VLLM::EngineCore*, bin/vllm serve. A stub that ignores SIGTERM
stays alive after the script, which shows that no SIGKILL was sent.
"""

import subprocess
import time

from _support import BASH, RELEASE, Result, ReleaseTest, msys

RANK = "VLLM::Worker_TP0"
ENGINE = "VLLM::EngineCore"
SCRIPT = RELEASE / "scripts" / "kill-stuck-rank.sh"


class KillStuckRankTest(ReleaseTest):

    def setUp(self):
        super().setUp()
        self.sb.install_venv()
        self.victims = []

    def fake_kfd(self, pid):
        """A KFD tree where the rank holds its memory on the card 0000:07:00.0."""
        root = self.sb.root / "kfd"
        (root / "proc" / str(pid)).mkdir(parents=True, exist_ok=True)
        (root / "proc" / str(pid) / "vram_100").write_text("4096\n")
        (root / "proc" / str(pid) / "vram_200").write_text("0\n")
        for node, gpu_id, loc in (("1", "100", 0x0700), ("2", "200", 0x0400)):
            (root / "topology" / "nodes" / node).mkdir(parents=True, exist_ok=True)
            (root / "topology" / "nodes" / node / "gpu_id").write_text(gpu_id + "\n")
            (root / "topology" / "nodes" / node / "properties").write_text(f"location_id {loc}\ndomain 0\n")
        return msys(root)

    def journal(self, text):
        (self.sb.state / "journal.txt").write_text(text, encoding="utf-8")

    def run_script(self, args, *, timeout=120, env=None):
        """Runs the script while reaping the victims: an exited child stays visible in /proc
        until its parent collects it, and the script decides by /proc."""
        argv = [BASH, msys(SCRIPT)] + [str(a) for a in args]
        popen = subprocess.Popen(argv, env=self.sb.env(env), stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, encoding="utf-8", errors="replace")
        deadline = time.monotonic() + timeout
        while popen.poll() is None:
            for victim, _ in self.victims:
                victim.poll()
            if time.monotonic() > deadline:
                popen.kill()
                out, err = popen.communicate()
                raise AssertionError(f"time limit of {timeout} s reached\n--- stdout ---\n{out}\n--- stderr ---\n{err}")
            time.sleep(0.05)
        out, err = popen.communicate()
        return Result(argv, popen.returncode, out, err)

    def victim(self, argv0, *, name, ignore_term=False):
        pid, popen, _ = self.sb.spawn("true", name=name, argv0=argv0, ignore_term=ignore_term)
        self.victims.append((popen, pid))
        return pid, popen

    def alive(self, pid):
        return self.sb.bash_alive(pid)

    # ----- usage -----------------------------------------------------------------------
    def test_help_exits_zero(self):
        r = self.sb.bash("scripts/kill-stuck-rank.sh", ["--help"], timeout=60)
        self.assertEqual(r.rc, 0, r.describe())

    def test_bad_usage_is_code2(self):
        for args in ([], ["abc"], ["123", "456"], ["-5"]):
            with self.subTest(args=args):
                r = self.sb.bash("scripts/kill-stuck-rank.sh", args, timeout=60)
                self.assertEqual(r.rc, 2, r.describe())

    def test_missing_process_is_code2(self):
        r = self.sb.bash("scripts/kill-stuck-rank.sh", ["2999999"], timeout=60)
        self.assertEqual(r.rc, 2, r.describe())
        self.assertIn("no such process", r.both)

    # ----- gates -----------------------------------------------------------------------
    def test_non_vllm_process_is_refused_and_not_signalled(self):
        pid, _ = self.victim("bash", name="other", ignore_term=False)
        r = self.run_script([str(pid)])
        self.assertEqual(r.rc, 2, r.describe())
        self.assertIn("not a vLLM process", r.both)
        self.assertTrue(self.alive(pid), "a non-vLLM process was signalled")

    def test_rank_with_unhealthy_gpu_before_signal_is_code1_and_not_signalled(self):
        pid, _ = self.victim(RANK, name="rank-unhealthy")
        self.sb.seq("probe", ["wrong"])
        r = self.run_script([str(pid)])
        self.assertEqual(r.rc, 1, r.describe())
        self.assertTrue(self.alive(pid), "a process was signalled although the GPU check failed")

    def test_engine_with_live_rank_is_refused_and_nothing_is_signalled(self):
        engine_pid, _ = self.victim(ENGINE, name="engine-busy")
        rank_pid, _ = self.victim(RANK, name="rank-alive")
        r = self.run_script([str(engine_pid)])
        self.assertEqual(r.rc, 2, r.describe())
        self.assertIn("live rank", r.both)
        self.assertTrue(self.alive(engine_pid))
        self.assertTrue(self.alive(rank_pid))

    # ----- signalling -------------------------------------------------------------------
    def test_rank_that_exits_on_sigterm_is_code0(self):
        pid, _ = self.victim(RANK, name="rank-exits")
        r = self.run_script([str(pid)])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertIn("exited after SIGTERM", r.both)
        self.assertFalse(self.alive(pid))

    def test_engine_without_ranks_is_stopped(self):
        pid, _ = self.victim(ENGINE, name="engine-alone")
        r = self.run_script([str(pid)])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertFalse(self.alive(pid))

    def test_api_server_command_line_counts_as_engine(self):
        pid, _ = self.victim("/opt/x/bin/vllm serve qwen", name="api-server")
        r = self.run_script([str(pid)])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertFalse(self.alive(pid))

    def test_process_that_survives_sigterm_is_code1_and_not_killed(self):
        pid, _ = self.victim(RANK, name="rank-stuck", ignore_term=True)
        r = self.run_script([str(pid)])
        self.assertEqual(r.rc, 1, r.describe())
        self.assertIn("still alive 30 s after SIGTERM", r.both)
        self.assertTrue(self.alive(pid), "the process is gone: SIGKILL or another hard stop was used")

    # ----- --force --------------------------------------------------------------------
    def test_force_kills_a_rank_that_survives_sigterm(self):
        pid, _ = self.victim(RANK, name="rank-force", ignore_term=True)
        self.journal("kernel: boot line\n")
        r = self.run_script(["--force", str(pid)], env={"RR_KFD_ROOT": self.fake_kfd(pid)})
        self.assertEqual(r.rc, 0, r.describe())
        self.assertIn("sending SIGKILL (--force)", r.both)
        self.assertFalse(self.alive(pid), "--force left the rank alive")

    def test_force_does_not_kill_a_rank_that_exits_on_sigterm(self):
        pid, _ = self.victim(RANK, name="rank-force-term")
        self.journal("kernel: boot line\n")
        r = self.run_script(["--force", str(pid)], env={"RR_KFD_ROOT": self.fake_kfd(pid)})
        self.assertEqual(r.rc, 0, r.describe())
        self.assertNotIn("sending SIGKILL (--force)", r.both)
        self.assertFalse(self.alive(pid))

    def test_force_is_refused_with_a_page_fault_on_the_card_of_the_rank(self):
        pid, _ = self.victim(RANK, name="rank-fault-card", ignore_term=True)
        self.journal("amdgpu 0000:07:00.0: [gfxhub] page fault (src_id:0 ring:40)\n")
        r = self.run_script(["--force", str(pid)], env={"RR_KFD_ROOT": self.fake_kfd(pid)})
        self.assertEqual(r.rc, 1, r.describe())
        self.assertIn("page fault on the card", r.both)
        self.assertTrue(self.alive(pid), "SIGKILL was sent despite a fault on the card of the rank")

    def test_force_is_refused_with_a_page_fault_from_the_process(self):
        pid, _ = self.victim(RANK, name="rank-fault-pid", ignore_term=True)
        self.journal(f"amdgpu 0000:04:00.0: [gfxhub] page fault (pid {pid} thread x)\n")
        r = self.run_script(["--force", str(pid)], env={"RR_KFD_ROOT": self.fake_kfd(pid)})
        self.assertEqual(r.rc, 1, r.describe())
        self.assertIn("page fault from this process", r.both)
        self.assertTrue(self.alive(pid), "SIGKILL was sent despite a fault from the process")

    def test_force_goes_ahead_with_a_page_fault_on_the_other_card(self):
        # The case of KI-1: the rank on the faulting card dies and the rank on the other card stays.
        pid, _ = self.victim(RANK, name="rank-other-card", ignore_term=True)
        self.journal("amdgpu 0000:04:00.0: [gfxhub] page fault (pid 1 thread x)\n")
        r = self.run_script(["--force", str(pid)], env={"RR_KFD_ROOT": self.fake_kfd(pid)})
        self.assertEqual(r.rc, 0, r.describe())
        self.assertFalse(self.alive(pid))

    def test_force_is_refused_when_the_kernel_log_is_empty(self):
        # journalctl without rights prints nothing and exits 0: that must not read as "no faults".
        pid, _ = self.victim(RANK, name="rank-no-log", ignore_term=True)
        self.journal("")
        r = self.run_script(["--force", str(pid)], env={"RR_KFD_ROOT": self.fake_kfd(pid)})
        self.assertEqual(r.rc, 1, r.describe())
        self.assertIn("kernel log cannot be read", r.both)
        self.assertTrue(self.alive(pid))

    def test_force_is_refused_when_the_card_is_unknown(self):
        pid, _ = self.victim(RANK, name="rank-no-card", ignore_term=True)
        self.journal("kernel: boot line\n")
        r = self.run_script(["--force", str(pid)], env={"RR_KFD_ROOT": msys(self.sb.root / "no-kfd")})
        self.assertEqual(r.rc, 1, r.describe())
        self.assertIn("card of this rank cannot be determined", r.both)
        self.assertTrue(self.alive(pid))

    def test_force_applies_to_ranks_only(self):
        pid, _ = self.victim(ENGINE, name="engine-force", ignore_term=True)
        self.journal("kernel: boot line\n")
        r = self.run_script(["--force", str(pid)], env={"RR_KFD_ROOT": self.fake_kfd(pid)})
        self.assertEqual(r.rc, 1, r.describe())
        self.assertIn("applies to ranks only", r.both)
        self.assertTrue(self.alive(pid))

    def test_force_without_pid_is_code2(self):
        r = self.sb.bash("scripts/kill-stuck-rank.sh", ["--force"], timeout=60)
        self.assertEqual(r.rc, 2, r.describe())

    def test_unhealthy_gpu_after_signal_is_code1(self):
        pid, _ = self.victim(RANK, name="rank-then-bad")
        self.sb.seq("probe", ["ok", "nan"])
        r = self.run_script([str(pid)])
        self.assertEqual(r.rc, 1, r.describe())
        self.assertIn("not 0 after the signal", r.both)
        self.assertFalse(self.alive(pid))
