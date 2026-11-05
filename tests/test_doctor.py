"""scripts/doctor.sh: each check, the thresholds, --pre, --json and exit codes.

The text output has one "PASS|WARN|FAIL name: detail" line per check. Exit code 0 means no check
failed, 1 means a check failed, and 2 means a bad call. --pre skips the venv and vllm checks, and
--json prints one object with the checks and the counts of FAIL and WARN. Stubs replace rocminfo,
hipconfig, hipcc, git, g++, make, id, uname, df, python3.12 and the venv interpreter. Two checks read
real host files that no stub can replace (/etc/os-release for os, /proc/meminfo for ram); for them the
test computes the expected status from the same file, so the test is correct on any host.
"""

import glob
import json
import pathlib
import re
import shutil
import subprocess

from _support import BASH, PYTHON_STUB, ReleaseTest, msys

GIB_KIB = 1024 * 1024
LINE = re.compile(r"^(PASS|WARN|FAIL) [a-z0-9-]+: ")
PATH_WITHOUT_HOST_PYTHON = "/usr/bin:/bin"


def host_os_is_ubuntu_2404():
    path = pathlib.Path("/etc/os-release")
    if not path.exists():
        return False
    values = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"')
    return values.get("ID") == "ubuntu" and values.get("VERSION_ID") == "24.04"


def host_mem_total_kib():
    """MemTotal as bash sees /proc/meminfo: on Git Bash that is the MSYS view, which
    Python on Windows cannot open."""
    result = subprocess.run([BASH, "-c", "cat /proc/meminfo"], capture_output=True, text=True,
                            timeout=20)
    for line in result.stdout.splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1])
    raise AssertionError(f"MemTotal is not in /proc/meminfo (rc={result.returncode})")


def host_cards_in_error_state():
    """amdgpu cards whose runtime_status reads "error" on this host: the same sysfs path that
    doctor.sh reads when no gfx1030 is listed. Empty on a host without the amdgpu driver."""
    found = []
    for path in glob.glob("/sys/bus/pci/drivers/amdgpu/*/power/runtime_status"):
        try:
            if pathlib.Path(path).read_text(encoding="utf-8").strip() == "error":
                found.append(path)
        except OSError:
            continue
    return found


class DoctorTest(ReleaseTest):

    def setUp(self):
        super().setUp()
        self.kfd = self.sb.root / "kfd"
        self.kfd.write_text("kfd")

    def doctor(self, args=("--pre", "--json"), *, env=None, timeout=120):
        merged = {"DOCTOR_KFD": msys(self.kfd)}
        merged.update(env or {})
        return self.sb.bash("scripts/doctor.sh", list(args), timeout=timeout, env=merged)

    def checks(self, args=("--pre", "--json"), env=None):
        r = self.doctor(args, env=env)
        self.assertIn(r.rc, (0, 1), r.describe())
        data = json.loads(r.out)
        return r, {c["name"]: c for c in data["checks"]}, data

    def status(self, name, **kwargs):
        _, checks, _ = self.checks(**kwargs)
        self.assertIn(name, checks, f"check {name} is missing")
        return checks[name]["status"], checks[name]["detail"]

    # ----- usage and output format -----------------------------------------------------
    def test_help_exits_zero(self):
        r = self.sb.bash("scripts/doctor.sh", ["--help"], timeout=60)
        self.assertEqual(r.rc, 0, r.describe())

    def test_bad_usage_exits_2(self):
        r = self.sb.bash("scripts/doctor.sh", ["--bogus"], timeout=60)
        self.assertEqual(r.rc, 2, r.describe())

    def test_text_lines_have_status_name_and_detail(self):
        r = self.doctor(("--pre",))
        self.assertIn(r.rc, (0, 1), r.describe())
        lines = [line for line in r.out.splitlines() if line.strip()]
        self.assertGreater(len(lines), 5, r.out)
        for line in lines:
            with self.subTest(line=line):
                self.assertRegex(line, LINE)

    def test_json_is_one_object_and_counts_match(self):
        r, checks, data = self.checks()
        self.assertEqual(set(data), {"checks", "fail", "warn"}, data.keys())
        self.assertEqual(data["fail"], sum(1 for c in data["checks"] if c["status"] == "FAIL"))
        self.assertEqual(data["warn"], sum(1 for c in data["checks"] if c["status"] == "WARN"))
        for item in data["checks"]:
            self.assertEqual(set(item), {"name", "status", "detail"})

    def test_exit_code_is_zero_without_fail_and_one_with_fail(self):
        r = self.doctor(("--pre", "--json"))
        self.assertEqual(r.rc, 0, r.describe())
        (self.sb.rocm / "bin" / "rocminfo").unlink()
        r = self.doctor(("--pre", "--json"))
        self.assertEqual(r.rc, 1, r.describe())

    def test_pre_skips_venv_and_vllm_checks(self):
        _, checks, _ = self.checks(("--pre", "--json"))
        self.assertNotIn("venv", checks)
        self.assertNotIn("vllm", checks)

    def test_full_run_checks_venv_and_vllm(self):
        _, checks, _ = self.checks(("--json",))
        self.assertIn("venv", checks)
        self.assertIn("vllm", checks)
        self.assertEqual(checks["venv"]["status"], "FAIL", "no venv was installed")
        self.sb.install_venv()
        self.assertEqual(self.status("venv", args=("--json",))[0], "PASS")

    # ----- system checks ---------------------------------------------------------------
    def test_os_status_follows_host_release_file(self):
        self.sb.set("uname.s", "Linux")
        status, detail = self.status("os")
        expected = "PASS" if host_os_is_ubuntu_2404() else "WARN"
        self.assertEqual(status, expected, detail)

    def test_non_linux_is_fail(self):
        self.sb.set("uname.s", "Darwin")
        status, detail = self.status("os")
        self.assertEqual(status, "FAIL", detail)

    def test_kfd_present_and_missing(self):
        self.assertEqual(self.status("kfd")[0], "PASS")
        self.kfd.unlink()
        self.assertEqual(self.status("kfd")[0], "FAIL")

    def test_groups_render_or_video(self):
        self.sb.set("user.groups", "users video")
        self.assertEqual(self.status("groups")[0], "PASS")
        self.sb.set("user.groups", "users render")
        self.assertEqual(self.status("groups")[0], "PASS")
        self.sb.set("user.groups", "users")
        self.assertEqual(self.status("groups")[0], "WARN")

    def test_rocminfo_missing_is_fail(self):
        (self.sb.rocm / "bin" / "rocminfo").unlink()
        self.assertEqual(self.status("rocm")[0], "FAIL")

    def test_hipcc_missing_is_fail(self):
        (self.sb.rocm / "bin" / "hipcc").unlink()
        self.assertEqual(self.status("hipcc")[0], "FAIL")

    def test_rocm_version_7_2_passes_other_warns(self):
        self.sb.set("hip.version", "7.2.4")
        self.assertEqual(self.status("rocm-version")[0], "PASS")
        self.sb.set("hip.version", "6.4.0")
        status, detail = self.status("rocm-version")
        self.assertEqual(status, "WARN")
        self.assertIn("tested with ROCm 7.2.4", detail)

    def test_gpu_count_thresholds(self):
        self.sb.set("gpus.txt", "gfx1030\ngfx1030\n")
        self.assertEqual(self.status("gpu-count")[0], "PASS")
        self.sb.set("gpus.txt", "gfx1030\n")
        status, detail = self.status("gpu-count")
        self.assertEqual(status, "WARN")
        self.assertIn("one card", detail)
        self.sb.set("gpus.txt", "")
        self.assertEqual(self.status("gpu-count")[0], "FAIL")

    def test_other_gfx103x_next_to_gfx1030_is_warn_untested(self):
        # Another gfx103x card next to gfx1030 cards gives WARN that names it as untested. A host
        # whose cards are all gfx1031 is not fixed by any test: its expected status is not settled
        # (see Interpretations in tests/README.md).
        self.sb.set("gpus.txt", "gfx1030\ngfx1030\ngfx1031\n")
        status, detail = self.status("gpu-count")
        self.assertEqual(status, "WARN", detail)
        self.assertIn("untested", detail)

    # The check reads the absolute sysfs path of host_cards_in_error_state(); no stub can replace it.
    # The no-card tests need that path to hold no card in state error, so they are skipped otherwise.
    def skip_if_host_card_in_error(self):
        found = host_cards_in_error_state()
        if found:
            self.skipTest(f"host card in state error ({found[0]}): the no-card case cannot be isolated")

    def test_no_gfx1030_with_hsa_error_on_stderr_is_fail_naming_the_error(self):
        # rocminfo writes its HSA error to stderr, in colour; the check must find the name in it.
        self.skip_if_host_card_in_error()
        self.sb.set("gpus.txt", "")
        self.sb.set("rocminfo.rc", "1")
        for name in ("HSA_STATUS_ERROR_OUT_OF_RESOURCES", "HSA_STATUS_ERROR_INVALID_ARGUMENT"):
            with self.subTest(name=name):
                self.sb.set("rocminfo.stderr", f"\x1b[31m{name}\x1b[0m\n")
                status, detail = self.status("gpu-count")
                self.assertEqual(status, "FAIL", detail)
                self.assertIn(f"rocminfo failed with {name}", detail)

    def test_no_gfx1030_and_no_rocminfo_error_is_fail_with_the_plain_message(self):
        self.skip_if_host_card_in_error()
        self.sb.set("gpus.txt", "")
        status, detail = self.status("gpu-count")
        self.assertEqual(status, "FAIL", detail)
        self.assertEqual(detail, "no gfx1030 GPU device in rocminfo")

    def test_gfx1030_found_keeps_pass_even_if_rocminfo_wrote_an_hsa_error(self):
        self.sb.set("gpus.txt", "gfx1030\ngfx1030\n")
        self.sb.set("rocminfo.stderr", "\x1b[31mHSA_STATUS_ERROR_OUT_OF_RESOURCES\x1b[0m\n")
        status, detail = self.status("gpu-count")
        self.assertEqual(status, "PASS", detail)
        self.assertEqual(detail, "2 gfx1030 devices")

    def test_python_3_12_is_pass_and_python3_of_3_12_is_pass(self):
        # A python3.12 on PATH gives PASS, and so does a generic python3 that reports 3.12.
        self.assertEqual(self.status("python")[0], "PASS")
        (self.sb.stubs / "python3.12").unlink()
        self.sb.add_stub("python3", PYTHON_STUB)
        self.sb.set("pyver.python3", "3.12")
        status, detail = self.status("python", env={"PATH": self._restricted_path()})
        self.assertEqual(status, "PASS", detail)

    def test_named_python3_11_only_is_warn(self):
        # A python3.11 found under its versioned name, and no python3.12, gives WARN.
        (self.sb.stubs / "python3.12").unlink()
        self.sb.add_stub("python3.11", PYTHON_STUB)
        self.sb.set("pyver.python3.11", "3.11")
        status, detail = self.status("python", env={"PATH": self._restricted_path()})
        self.assertEqual(status, "WARN", detail)

    def test_generic_python3_of_3_11_is_warn(self):
        # A generic python3 reporting 3.11 is another Python of the 3.10 to 3.14 range, so it gives WARN.
        (self.sb.stubs / "python3.12").unlink()
        self.sb.add_stub("python3", PYTHON_STUB)
        self.sb.set("pyver.python3", "3.11")
        status, detail = self.status("python", env={"PATH": self._restricted_path()})
        self.assertEqual(status, "WARN", detail)

    def test_no_python_is_fail(self):
        (self.sb.stubs / "python3.12").unlink()
        status, detail = self.status("python", env={"PATH": self._restricted_path()})
        self.assertEqual(status, "FAIL", detail)

    def _restricted_path(self):
        """The stub directory and system tools only: no host python3 can be found."""
        for name in ("python3", "python3.12"):
            host = shutil.which(name, path=PATH_WITHOUT_HOST_PYTHON)
            if host:
                self.skipTest(f"{name} is in {PATH_WITHOUT_HOST_PYTHON}; the FAIL case cannot be isolated")
        return msys(self.sb.stubs) + ":" + PATH_WITHOUT_HOST_PYTHON

    def test_build_tools_present_and_missing(self):
        self.assertEqual(self.status("build-tools")[0], "PASS")
        if shutil.which("g++", path=PATH_WITHOUT_HOST_PYTHON):
            self.skipTest("a real g++ is in /usr/bin; the missing-tool case cannot be isolated")
        (self.sb.stubs / "g++").unlink()
        self.assertEqual(self.status("build-tools", env={"PATH": self._restricted_path()})[0], "FAIL")

    def test_ram_follows_host_meminfo(self):
        expected = "PASS" if host_mem_total_kib() >= 32 * GIB_KIB else "WARN"
        status, detail = self.status("ram")
        self.assertEqual(status, expected, detail)

    def test_shm_threshold_is_8_gib(self):
        self.sb.set("df.kb", str(8 * GIB_KIB))
        self.assertEqual(self.status("shm")[0], "PASS")
        self.sb.set("df.kb", str(8 * GIB_KIB - 1))
        self.assertEqual(self.status("shm")[0], "WARN")

    def test_disk_threshold_is_40_gib(self):
        self.sb.set("df.kb", str(40 * GIB_KIB))
        self.assertEqual(self.status("disk")[0], "PASS")
        self.sb.set("df.kb", str(40 * GIB_KIB - 1))
        self.assertEqual(self.status("disk")[0], "WARN")

    def test_p2p_is_informational_pass_with_exact_text(self):
        self.sb.install_venv()
        self.sb.set("p2p.out", "False")
        status, detail = self.status("p2p", args=("--json",))
        self.assertEqual(status, "PASS")
        self.assertEqual(detail, "p2p=False (expected on most consumer systems; not required)")

    def test_p2p_true_is_never_a_failure(self):
        self.sb.install_venv()
        self.sb.set("p2p.out", "True")
        status, _ = self.status("p2p", args=("--json",))
        self.assertEqual(status, "PASS")

    def test_venv_present_and_vllm_import_checks(self):
        self.sb.install_venv()
        self.assertEqual(self.status("venv", args=("--json",))[0], "PASS")
        self.assertEqual(self.status("vllm", args=("--json",))[0], "PASS")
        self.sb.set("import.rc", "1")
        self.assertEqual(self.status("vllm", args=("--json",))[0], "FAIL")

    # ----- Python headers (python-dev) --------------------------------------------------
    # The python stub answers the sysconfig query from state/pyinclude/Python.h. The default sandbox has
    # that file (python3.12-dev installed); a test removes it to make the check fail.
    def headers(self):
        return self.sb.state / "pyinclude" / "Python.h"

    def test_python_dev_passes_when_python_h_is_present(self):
        self.assertTrue(self.headers().is_file())
        status, detail = self.status("python-dev")
        self.assertEqual(status, "PASS", detail)
        self.assertEqual(detail, "python3.12 has Python.h")

    def test_python_dev_fails_with_the_apt_command_when_python_h_is_missing(self):
        self.headers().unlink()
        status, detail = self.status("python-dev")
        self.assertEqual(status, "FAIL", detail)
        self.assertEqual(detail, "python3.12 has no Python.h (sudo apt-get install python3.12-dev)")

    def test_python_dev_asks_the_interpreter_through_sysconfig_for_python_h(self):
        self.status("python-dev")
        queries = [row for row in self.sb.calls("python3.12")
                   if row[1:3] == ["-I", "-c"] and len(row) > 3 and "sysconfig" in row[3]]
        self.assertEqual(len(queries), 1, "python-dev did not ask sysconfig exactly once")
        self.assertIn("Python.h", queries[0][3])
        self.assertIn('sysconfig.get_path("include")', queries[0][3])

    def test_python_dev_uses_python3_when_python312_is_absent(self):
        (self.sb.stubs / "python3.12").unlink()
        self.sb.add_stub("python3", PYTHON_STUB)
        self.sb.set("pyver.python3", "3.12")
        status, detail = self.status("python-dev", env={"PATH": self._restricted_path()})
        self.assertEqual(status, "PASS", detail)
        self.assertEqual(detail, "python3 has Python.h")

    def test_python_dev_takes_python312_first_and_does_not_ask_python3(self):
        self.sb.add_stub("python3", PYTHON_STUB)
        self.sb.set("pyver.python3", "3.12")
        self.status("python-dev")
        asked = [row for row in self.sb.calls("python3") if "sysconfig" in " ".join(row)]
        self.assertEqual(asked, [], "python3 was asked for Python.h although python3.12 exists")

    def test_python_dev_line_is_absent_when_no_interpreter_is_found(self):
        (self.sb.stubs / "python3.12").unlink()
        _, checks, _ = self.checks(env={"PATH": self._restricted_path()})
        self.assertNotIn("python-dev", checks)
        self.assertEqual(checks["python"]["status"], "FAIL")

    # ----- ninja in the venv --------------------------------------------------------------
    def test_ninja_in_the_venv_passes(self):
        self.sb.install_venv()
        status, detail = self.status("ninja", args=("--json",))
        self.assertEqual(status, "PASS", detail)
        self.assertEqual(detail, msys(self.sb.rdh / "venv" / "bin" / "ninja"))

    def test_ninja_missing_from_the_venv_fails_and_names_the_fix(self):
        self.sb.install_venv()
        (self.sb.rdh / "venv" / "bin" / "ninja").unlink()
        status, detail = self.status("ninja", args=("--json",))
        self.assertEqual(status, "FAIL", detail)
        self.assertIn("install.sh --force", detail)

    def test_ninja_check_needs_the_venv_python_and_is_absent_with_pre(self):
        _, checks, _ = self.checks(("--json",))
        self.assertIn("venv", checks)
        self.assertNotIn("ninja", checks, "ninja was checked without a venv python")
        self.sb.install_venv()
        _, checks, _ = self.checks(("--pre", "--json"))
        self.assertNotIn("ninja", checks, "ninja was checked with --pre")
