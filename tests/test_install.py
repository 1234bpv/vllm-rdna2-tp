"""install.sh: the install steps, the git and pip calls, the build environment, exit codes, dry run,
--force, and the Python version gate.

Every external command is a stub: git, curl, pip, venv/bin/python, hf, vllm. The doctor
check runs for real with stubbed ROCm tools, DOCTOR_KFD pointing at a file, and uname,
id and df stubs, so the outcome is the same on Linux and on Git Bash.

InstallSuccessTest runs one successful install and only reads its results. The other
classes use their own sandboxes, because each one changes a precondition.
"""

import re
import shutil
import unittest

from _support import BASH, PYTHON_STUB, RELEASE, ReleaseTest, Sandbox, msys

COMMIT = "c296851a7d173fa89d2eefbca0243be42ae9b5e0"
ROCM_HELP = "amdgpu-install --usecase=rocm --no-dkms -y"


def git_subcommands(sb):
    """git calls as argument lists: the -C dir and -c key=value options are removed."""
    rows = []
    for row in sb.calls("git"):
        args = row[1:]
        while args and args[0] in ("-C", "-c"):
            args = args[2:]
        rows.append(args)
    return rows


def pip_calls(sb):
    return [row for row in sb.calls("python") if "-m" in row and "pip" in row]


def install_args(prefix, *, python="python3.12", jobs="2", extra=()):
    return ["--prefix", msys(prefix), "--python", python, "--jobs", jobs] + list(extra)


class InstallSuccessTest(unittest.TestCase):
    """One successful install, shared by the tests of this class (read-only)."""

    @classmethod
    def setUpClass(cls):
        cls.sb = Sandbox()
        kfd = cls.sb.root / "kfd"
        kfd.write_text("kfd")
        cls.prefix = cls.sb.root / "prefix"
        try:
            cls.caller_path = "/opt/rt-custom/bin:" + cls.sb.env()["PATH"]
            cls.result = cls.sb.bash("install.sh", install_args(cls.prefix, jobs="3"),
                                     timeout=300, env={"DOCTOR_KFD": msys(kfd), "PATH": cls.caller_path})
        except BaseException:
            cls.sb.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.sb.cleanup()

    def test_success_exit_code_is_zero(self):
        self.assertEqual(self.result.rc, 0, self.result.describe())

    def test_seven_steps_are_printed_in_order(self):
        out = self.result.out
        positions = [out.find(f"[{n}/7]") for n in range(1, 8)]
        self.assertTrue(all(pos >= 0 for pos in positions), out)
        self.assertEqual(positions, sorted(positions), "steps out of order")

    def test_marker_and_log_are_written(self):
        marker = self.prefix / ".installed-0.1.0"
        self.assertTrue(marker.is_file(), "no marker written")
        self.assertIn("version 0.1.0", marker.read_text(encoding="utf-8"))
        logs = list((self.prefix / "logs").glob("install-*.log"))
        self.assertEqual(len(logs), 1, "expected one install log in logs/")
        self.assertIn("next steps:", self.result.out)
        self.assertRegex(self.result.out, re.compile(r"installed: vLLM 0\.1\.0 in .+"))

    def test_venv_is_created_on_the_requested_python(self):
        self.assertTrue((self.prefix / "venv" / "bin" / "python").exists())
        creations = [row for row in self.sb.calls("python3.12") if row[1:3] == ["-m", "venv"]]
        self.assertEqual(len(creations), 1, creations)

    def test_git_sequence_matches_spec(self):
        seq = [args for args in git_subcommands(self.sb)
               if not (args[:1] == ["remote"] and args[1:2] == ["get-url"])]
        self.assertEqual(seq[0], ["init"])
        self.assertEqual(seq[1], ["remote", "add", "origin", "https://github.com/vllm-project/vllm.git"])
        self.assertEqual(seq[2], ["fetch", "--depth", "1", "origin", COMMIT])
        self.assertEqual(seq[3], ["checkout", "--force", "FETCH_HEAD"])
        self.assertEqual(seq[4], ["clean", "-fdx"], "clean must follow checkout: it removes the files of an earlier run")
        self.assertEqual(seq[5][:2], ["apply", "--whitespace=nowarn"])
        self.assertTrue(seq[5][2].endswith("patches/vllm-rdna2-tp-0.1.0.patch"), seq[5])
        self.assertEqual(len(seq), 6, seq)

    def test_every_writing_git_call_disables_autocrlf(self):
        # Every git command that writes into the source tree runs with -c core.autocrlf=false, so
        # that the patch stays byte-exact. "git remote get-url" only reads and is exempt.
        offenders = [row[1:] for row in self.sb.calls("git")
                     if row[1:3] != ["-c", "core.autocrlf=false"] and "get-url" not in row]
        self.assertEqual(offenders, [], "writing git calls without -c core.autocrlf=false")

    def test_dependency_pip_call_uses_pip_install_and_spec_pins(self):
        deps = [row for row in pip_calls(self.sb) if "setuptools-scm" in row]
        self.assertEqual(len(deps), 1, pip_calls(self.sb))
        args = deps[0]
        self.assertEqual(args[1:4], ["-m", "pip", "install"], "step 3 must be pip install, not pip wheel")
        for needle in ["setuptools>=77.0.3,<80.0.0", "setuptools-rust", "cmake>=3.26.1,<4",
                       "ninja", "packaging", "jinja2", "wheel", "-c"]:
            with self.subTest(needle=needle):
                self.assertIn(needle, args)

    def test_torch_pip_call_matches_spec(self):
        torch = [row for row in pip_calls(self.sb) if "--extra-index-url" in row]
        self.assertEqual(len(torch), 1)
        args = torch[0]
        self.assertIn("https://download.pytorch.org/whl/rocm7.2", args)
        for name in ["torch", "torchvision", "triton", "-c"]:
            with self.subTest(name=name):
                self.assertIn(name, args)

    def test_amdsmi_is_installed_from_a_writable_copy(self):
        amd = [row for row in pip_calls(self.sb) if any("amd_smi" in a for a in row)]
        self.assertEqual(len(amd), 1, pip_calls(self.sb))
        self.assertNotIn(msys(self.sb.rocm / "share" / "amd_smi"), amd[0],
                         "pip ran on the read-only ROCm source")
        copied = [a for a in amd[0] if "amd_smi" in a][0]
        self.assertIn(msys(self.sb.tmp), copied)

    def test_requirements_and_editable_build_calls(self):
        reqs = [row for row in pip_calls(self.sb) if "requirements/rocm.txt" in row]
        self.assertEqual(len(reqs), 1)
        builds = [row for row in pip_calls(self.sb) if "-e" in row]
        self.assertEqual(len(builds), 1)
        for needle in ["-e", ".", "--no-build-isolation", "--no-deps"]:
            with self.subTest(needle=needle):
                self.assertIn(needle, builds[0])

    def test_build_step_environment_matches_spec(self):
        values = dict(line.split("=", 1) for line in (self.sb.state / "last-pip.env").read_text().splitlines())
        self.assertEqual(values["tag"], "build", "last pip call is not the editable build")
        path_parts = values["PATH"].split(":")
        self.assertTrue(path_parts[0].endswith("venv/bin"), values["PATH"])
        self.assertTrue(path_parts[1].endswith("rocm/bin"), values["PATH"])
        self.assertEqual(values["ROCM_PATH"], msys(self.sb.rocm))
        self.assertEqual(values["HIP_PATH"], msys(self.sb.rocm))
        self.assertEqual(values["PYTORCH_ROCM_ARCH"], "gfx1030")
        self.assertEqual(values["VLLM_TARGET_DEVICE"], "rocm")
        self.assertEqual(values["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM"], "0.0.0")
        self.assertEqual(values["CMAKE_BUILD_PARALLEL_LEVEL"], "3", "--jobs 3 must reach the build")
        self.assertEqual(values["MAX_JOBS"], "3")
        self.assertTrue(values["PWD"].endswith("src/vllm"), values["PWD"])

    def test_build_step_path_is_venv_then_rocm_then_the_callers_path(self):
        # The command that runs gets the real PATH: the venv, the ROCm bin, then every caller entry.
        text = (self.sb.state / "last-pip.env").read_text(encoding="utf-8")
        values = dict(line.split("=", 1) for line in text.splitlines())
        expected = f"{msys(self.prefix / 'venv' / 'bin')}:{msys(self.sb.rocm / 'bin')}:{self.caller_path}"
        self.assertEqual(values["PATH"], expected)

    def test_post_install_check_covers_spec_items(self):
        checks = [row for row in self.sb.calls("python") if "-I" in row and "-c" in row
                  and any("is_available" in a for a in row)]
        self.assertEqual(len(checks), 1, "no post-install import check")
        code = checks[0][checks[0].index("-c") + 1]
        for needle in ["import vllm", "import torch", "torch.cuda.is_available()",
                       "gfx1030", "get_arch_list", "host_all_reduce"]:
            with self.subTest(needle=needle):
                self.assertIn(needle, code)
        self.assertTrue(any(row[1:] == ["--help"] for row in self.sb.calls("vllm")),
                        "vllm --help was not run")

    def test_ninja_version_is_run_from_the_venv_after_install(self):
        # Step 7 runs the venv's ninja, because the .hip kernels are built with that ninja at first use.
        invoked = (self.sb.state / "ninja.invoked").read_text(encoding="utf-8").splitlines()
        self.assertEqual(invoked, [msys(self.prefix / "venv" / "bin" / "ninja")],
                         "step 7 did not run the venv's ninja")
        self.assertEqual(self.sb.calls("ninja"), [["ninja", "--version"]])


class InstallDryRunTest(ReleaseTest):
    """--dry-run, doctor gate and version checks; no state is changed by the run."""

    def setUp(self):
        super().setUp()
        self.kfd = self.sb.root / "kfd"
        self.kfd.write_text("kfd")

    def run_install(self, args, *, env=None, timeout=240, prefix=None):
        merged = {"DOCTOR_KFD": msys(self.kfd)}
        merged.update(env or {})
        if prefix is not None:
            args = install_args(prefix) + list(args)
        return self.sb.bash("install.sh", args, timeout=timeout, env=merged)

    def test_help_exits_zero_and_describes_exit_codes(self):
        r = self.sb.bash("install.sh", ["--help"], timeout=60)
        self.assertEqual(r.rc, 0, r.describe())
        self.assertIn("--dry-run", r.out)
        self.assertIn("Exit: 0 installed", r.out)

    def test_bad_calls_exit_2(self):
        bad = [["--bogus"], ["--jobs", "0"], ["--jobs", "x"], ["--prefix"], ["--python"],
               ["--prefix", ""]]
        for args in bad:
            with self.subTest(args=args):
                r = self.sb.bash("install.sh", args, timeout=60)
                self.assertEqual(r.rc, 2, r.describe())
                self.assertIn("(see --help)", r.err)

    def test_default_home_comes_from_VLLM_RDNA2_HOME(self):
        r = self.run_install(["--dry-run", "--python", "python3.12", "--jobs", "2"])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertIn(msys(self.sb.rdh / "venv"), r.out)

    def test_dry_run_prints_expanded_commands_and_creates_nothing(self):
        prefix = self.sb.root / "never-created"
        r = self.run_install(["--dry-run"], prefix=prefix)
        self.assertEqual(r.rc, 0, r.describe())
        self.assertFalse(prefix.exists(), "dry run created the install home")
        plus_lines = [line for line in r.out.splitlines() if line.startswith("+ ")]
        self.assertGreaterEqual(len(plus_lines), 12, r.out)
        needles = [
            "core.autocrlf=false init",
            "remote add origin https://github.com/vllm-project/vllm.git",
            f"fetch --depth 1 origin {COMMIT}",
            "checkout --force FETCH_HEAD",
            "clean -fdx",
            "apply --whitespace=nowarn",
            "--extra-index-url https://download.pytorch.org/whl/rocm7.2",
            "setuptools-scm",
            "-r requirements/rocm.txt",
            "pip install -e . --no-build-isolation --no-deps",
            "PYTORCH_ROCM_ARCH=gfx1030",
            "VLLM_TARGET_DEVICE=rocm",
            "MAX_JOBS=2",
            "dry run complete: nothing was changed",
        ]
        for needle in needles:
            with self.subTest(needle=needle):
                self.assertIn(needle, r.out)
        for line in plus_lines:
            # PATH is the one value printed as a reference, "$PATH" (rr_run in scripts/lib.sh); the others are expanded.
            self.assertNotIn("$", line.replace("$PATH", ""), f"command not expanded: {line}")

    def test_dry_run_prints_clean_between_checkout_and_apply(self):
        r = self.run_install(["--dry-run"], prefix=self.sb.root / "dry")
        self.assertEqual(r.rc, 0, r.describe())
        needles = ("checkout --force FETCH_HEAD", "clean -fdx", "apply --whitespace=nowarn")
        order = [r.out.find(needle) for needle in needles]
        self.assertTrue(all(pos >= 0 for pos in order), r.out)
        self.assertEqual(order, sorted(order), f"out of order: {order}")

    def test_dry_run_makes_no_network_or_build_calls(self):
        r = self.run_install(["--dry-run"], prefix=self.sb.root / "dry")
        self.assertEqual(r.rc, 0, r.describe())
        self.assertEqual(self.sb.calls("curl"), [], "curl was called in a dry run")
        self.assertEqual(self.sb.calls("git"), [], "git was called in a dry run")
        self.assertEqual(pip_calls(self.sb), [], "pip was called in a dry run")
        self.assertEqual(self.sb.calls("vllm"), [])

    def test_dry_run_python_versions_3_10_to_3_14_warn_except_3_12(self):
        for minor in ("3.10", "3.11", "3.13", "3.14"):
            name = "python" + minor
            with self.subTest(python=name):
                self.sb.add_stub(name, PYTHON_STUB)
                self.sb.set("pyver." + name, minor)
                r = self.run_install(["--dry-run", "--prefix", msys(self.sb.root / "dry"),
                                      "--python", name, "--jobs", "2"])
                self.assertEqual(r.rc, 0, r.describe())
                self.assertIn("tested only with 3.12", r.both)

    def test_python_outside_3_10_to_3_14_is_code3(self):
        for minor in ("3.9", "3.15"):
            name = "python" + minor
            with self.subTest(python=name):
                self.sb.add_stub(name, PYTHON_STUB)
                self.sb.set("pyver." + name, minor)
                r = self.run_install(["--dry-run", "--prefix", msys(self.sb.root / "dry"),
                                      "--python", name, "--jobs", "2"])
                self.assertEqual(r.rc, 3, r.describe())
                self.assertIn("needs Python 3.10 to 3.14", r.err)

    def test_missing_python_is_code3_but_dry_run_continues(self):
        missing = "/nonexistent/bin/python3.12"
        r = self.run_install(["--prefix", msys(self.sb.root / "p"), "--python", missing, "--jobs", "2"])
        self.assertEqual(r.rc, 3, r.describe())
        self.assertIn("does not run", r.err)
        r = self.run_install(["--dry-run", "--prefix", msys(self.sb.root / "p"),
                              "--python", missing, "--jobs", "2"])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertIn("a real install would stop", r.both)

    def test_doctor_fail_is_code3_and_prints_rocm_commands(self):
        (self.sb.rocm / "bin" / "rocminfo").unlink()
        prefix = self.sb.root / "prefix"
        r = self.run_install([], prefix=prefix)
        self.assertEqual(r.rc, 3, r.describe())
        self.assertIn("https://repo.radeon.com/amdgpu-install/7.2.4/ubuntu/noble/", r.out)
        self.assertIn("sudo apt-get install -y ./amdgpu-install_7.2.4.70204-1_all.deb", r.out)
        self.assertIn(ROCM_HELP, r.out)
        self.assertEqual(pip_calls(self.sb), [], "pip ran after a failed system check")
        self.assertEqual(self.sb.calls("git"), [], "git ran after a failed system check")

    def test_doctor_fail_dry_run_is_code3(self):
        (self.sb.rocm / "bin" / "rocminfo").unlink()
        r = self.run_install(["--dry-run"])
        self.assertEqual(r.rc, 3, r.describe())
        self.assertIn(ROCM_HELP, r.out)

    def test_skip_checks_turns_fail_into_warning(self):
        (self.sb.rocm / "bin" / "rocminfo").unlink()
        prefix = self.sb.root / "prefix"
        r = self.run_install(["--skip-checks"], prefix=prefix)
        self.assertEqual(r.rc, 0, r.describe())
        self.assertIn("continuing because of --skip-checks", r.err)
        self.assertTrue((prefix / ".installed-0.1.0").is_file())

    def test_existing_marker_is_a_noop(self):
        prefix = self.sb.root / "prefix"
        prefix.mkdir()
        marker = prefix / ".installed-0.1.0"
        marker.write_text("version 0.1.0\ninstalled earlier\n")
        r = self.run_install([], prefix=prefix)
        self.assertEqual(r.rc, 0, r.describe())
        self.assertIn("already installed", r.out)
        self.assertEqual(marker.read_text(), "version 0.1.0\ninstalled earlier\n")
        self.assertFalse((prefix / "logs").exists(), "a no-op run created logs/")
        self.assertEqual(self.sb.calls("git"), [])
        self.assertEqual(pip_calls(self.sb), [])

    def test_force_keeps_models_and_logs_and_removes_venv_src_and_markers(self):
        prefix = self.sb.root / "prefix"
        for path, text in [
            (prefix / ".installed-0.0.9", "old\n"),
            (prefix / ".installed-0.1.0", "old\n"),
            (prefix / "venv" / "bin" / "old-marker", "venv\n"),
            (prefix / "src" / "vllm" / "old.txt", "src\n"),
            (prefix / "models" / "m" / "weights.bin", "keep-me"),
            (prefix / "logs" / "old.log", "keep-log\n"),
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        r = self.run_install(["--force"], prefix=prefix)
        self.assertEqual(r.rc, 0, r.describe())
        self.assertFalse((prefix / "venv" / "bin" / "old-marker").exists(), "venv not removed")
        self.assertFalse((prefix / "src" / "vllm" / "old.txt").exists(), "src/vllm not removed")
        self.assertFalse((prefix / ".installed-0.0.9").exists(), "old marker kept")
        self.assertEqual((prefix / "models" / "m" / "weights.bin").read_text(), "keep-me")
        self.assertEqual((prefix / "logs" / "old.log").read_text(), "keep-log\n")
        self.assertNotEqual((prefix / ".installed-0.1.0").read_text(), "old\n")
        self.assertIn("--force", r.out)

    def test_dry_run_prints_the_ninja_check_and_does_not_run_ninja(self):
        r = self.run_install(["--dry-run"], prefix=self.sb.root / "dry")
        self.assertEqual(r.rc, 0, r.describe())
        self.assertIn("+ ninja --version", r.out, "the ninja check is not printed in a dry run")
        self.assertEqual(self.sb.calls("ninja"), [], "ninja was run in a dry run")
        self.assertFalse((self.sb.state / "ninja.invoked").exists(), "ninja was run in a dry run")

    def test_missing_patch_is_code5_before_any_download(self):
        copy = self.sb.release_copy()
        (copy / "patches" / "vllm-rdna2-tp-0.1.0.patch").unlink()
        prefix = self.sb.root / "prefix"
        r = self.sb.run([BASH, msys(copy / "install.sh")] + install_args(prefix),
                        timeout=120, env={"DOCTOR_KFD": msys(self.kfd)})
        self.assertEqual(r.rc, 5, r.describe())
        self.assertIn("patch is missing", r.err)
        self.assertEqual(self.sb.calls("git"), [])
        self.assertEqual(self.sb.calls("curl"), [])

    def test_dry_run_prints_path_as_a_reference_and_never_the_callers_entries(self):
        marker = "/opt/rt-custom/bin"
        base = self.sb.env()["PATH"]
        r = self.run_install(["--dry-run"], prefix=self.sb.root / "dry", env={"PATH": marker + ":" + base})
        self.assertEqual(r.rc, 0, r.describe())
        self.assertNotIn(marker, r.out, "an entry of the caller's PATH is in the dry-run output")
        self.assertNotIn(base, r.out, "the caller's PATH is printed expanded")
        venv_rocm = f"{msys(self.sb.root / 'dry' / 'venv' / 'bin')}:{msys(self.sb.rocm / 'bin')}"
        self.assertIn(f"{venv_rocm}:$PATH", r.out, "the build step does not print PATH as a reference")

    def add_python_without_python_h(self):
        # doctor.sh checks python3.12 only, so the interpreter given to --python is a separate stub whose
        # sysconfig query finds no Python.h. The stub is the python stub with another state file.
        self.sb.add_stub("python3.13", PYTHON_STUB.replace("pyinclude/Python.h", "pyinclude/no-python-h"))
        self.sb.set("pyver.python3.13", "3.13")

    def test_real_install_with_a_python_without_python_h_is_code3_and_creates_no_venv(self):
        self.add_python_without_python_h()
        prefix = self.sb.root / "noh"
        r = self.run_install(["--python", "python3.13"], prefix=prefix)
        self.assertEqual(r.rc, 3, r.describe())
        self.assertIn("python3.13 has no Python.h", r.err)
        self.assertEqual([row for row in self.sb.calls("python3.13") if row[1:3] == ["-m", "venv"]], [],
                         "a venv was created without Python.h")
        self.assertFalse((prefix / ".installed-0.1.0").exists(), "marker written after a failure")

    def test_force_with_a_python_without_python_h_is_code3_and_keeps_venv_src_and_markers(self):
        # --force removes the old venv and source only after the interpreter checks; a failed check
        # must leave the existing install in place.
        self.add_python_without_python_h()
        prefix = self.sb.root / "prefix"
        kept = {
            prefix / "venv" / "bin" / "old-marker": "venv\n",
            prefix / "src" / "vllm" / "old.txt": "src\n",
            prefix / ".installed-0.0.9": "old\n",
            prefix / ".installed-0.1.0": "version 0.1.0\ninstalled earlier\n",
        }
        for path, text in kept.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        r = self.run_install(["--force", "--python", "python3.13"], prefix=prefix)
        self.assertEqual(r.rc, 3, r.describe())
        self.assertIn("python3.13 has no Python.h", r.err)
        for path, text in kept.items():
            with self.subTest(path=path.relative_to(prefix).as_posix()):
                self.assertTrue(path.is_file(), f"{path.name} was removed although the checks failed")
                self.assertEqual(path.read_text(), text)

    def test_dry_run_with_a_python_without_python_h_warns_and_succeeds(self):
        self.add_python_without_python_h()
        r = self.run_install(["--dry-run", "--python", "python3.13"], prefix=self.sb.root / "noh-dry")
        self.assertEqual(r.rc, 0, r.describe())
        self.assertIn("python3.13 has no Python.h; a real install would stop", r.both)


class InstallFailureTest(ReleaseTest):
    """Each test sets one failure in the stubs and checks the exit code and the marker."""

    def setUp(self):
        super().setUp()
        self.kfd = self.sb.root / "kfd"
        self.kfd.write_text("kfd")
        self.prefix = self.sb.root / "prefix"

    def run_install(self, timeout=240):
        return self.sb.bash("install.sh", install_args(self.prefix), timeout=timeout,
                            env={"DOCTOR_KFD": msys(self.kfd)})

    def assert_no_marker(self):
        self.assertFalse((self.prefix / ".installed-0.1.0").exists(), "marker written after a failure")

    def test_fetch_failure_is_code5(self):
        self.sb.set("git.fetch.rc", "1")
        r = self.run_install()
        self.assertEqual(r.rc, 5, r.describe())
        self.assertIn("fetching commit", r.err)
        self.assertFalse(any(a[:1] == ["apply"] for a in git_subcommands(self.sb)))
        self.assert_no_marker()

    def test_patch_that_does_not_apply_is_code5(self):
        self.sb.set("git.apply.rc", "1")
        r = self.run_install()
        self.assertEqual(r.rc, 5, r.describe())
        self.assertIn("the patch does not apply", r.err)
        self.assert_no_marker()

    def test_clean_failure_is_code5_and_apply_is_not_run(self):
        self.sb.set("git.clean.rc", "1")
        r = self.run_install()
        self.assertEqual(r.rc, 5, r.describe())
        self.assertIn("git clean failed", r.err)
        self.assertFalse(any(a[:1] == ["apply"] for a in git_subcommands(self.sb)),
                         "apply ran after a failed clean")
        self.assert_no_marker()

    def test_dependency_failures_are_code4(self):
        for tag in ("deps", "torch", "amdsmi", "reqs"):
            with self.subTest(step=tag):
                self.sb.set(f"fail.{tag}", "1")
                r = self.run_install()
                self.assertEqual(r.rc, 4, r.describe())
                self.assert_no_marker()
                (self.sb.state / f"fail.{tag}").unlink()
                shutil.rmtree(self.prefix, ignore_errors=True)

    def test_build_failure_is_code6(self):
        self.sb.set("fail.build", "1")
        r = self.run_install()
        self.assertEqual(r.rc, 6, r.describe())
        self.assertIn("the build failed", r.err)
        self.assert_no_marker()

    def test_post_install_import_failure_is_code7(self):
        self.sb.set("postcheck.rc", "1")
        r = self.run_install()
        self.assertEqual(r.rc, 7, r.describe())
        self.assertIn("import and GPU check failed", r.err)
        self.assert_no_marker()

    def test_vllm_help_failure_is_code7(self):
        self.sb.set("vllm_help.rc", "1")
        r = self.run_install()
        self.assertEqual(r.rc, 7, r.describe())
        self.assertIn("vllm --help failed", r.err)
        self.assert_no_marker()

    def test_ninja_missing_after_install_is_code7_and_no_marker(self):
        # The build-dependency step leaves no ninja behind: the post-install check must catch it.
        self.sb.set("no-ninja", "1")
        r = self.run_install()
        self.assertEqual(r.rc, 7, r.describe())
        self.assertIn("ninja is missing from the venv", r.err)
        self.assertFalse((self.prefix / "venv" / "bin" / "ninja").exists(),
                         "the stub created ninja although no-ninja was set")
        self.assert_no_marker()


class InstallLeftoverTest(ReleaseTest):
    """A re-run after an install that stopped once the patch was applied: the files the patch
    added are in the tree as untracked files. The git stub refuses apply while any of them exists."""

    def setUp(self):
        super().setUp()
        self.kfd = self.sb.root / "kfd"
        self.kfd.write_text("kfd")
        self.prefix = self.sb.root / "prefix"
        self.src = self.prefix / "src" / "vllm"

    def leave(self, *rel_paths):
        self.sb.set("git.leftover", "\n".join(rel_paths) + "\n")
        for rel in rel_paths:
            path = self.src / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("added by the patch of an earlier run\n")

    def run_install(self):
        return self.sb.bash("install.sh", install_args(self.prefix), timeout=240,
                            env={"DOCTOR_KFD": msys(self.kfd)})

    def test_untracked_leftover_is_removed_by_clean_and_apply_runs_after_it(self):
        self.leave("vllm/rdna2_tp.py")
        r = self.run_install()
        self.assertEqual(r.rc, 0, r.describe())
        self.assertNotIn("already exists", r.both)
        names = [args[0] for args in git_subcommands(self.sb)]
        self.assertLess(names.index("checkout"), names.index("clean"), names)
        self.assertLess(names.index("clean"), names.index("apply"), names)
        self.assertFalse((self.src / "vllm" / "rdna2_tp.py").exists(), "clean did not remove the leftover")

    def test_ignored_hip_leftover_is_removed_only_because_clean_has_x(self):
        # upstream's .gitignore covers *.hip: without -x that file stays and apply refuses to run.
        self.leave("csrc/rdna2_tp_kernels.hip")
        r = self.run_install()
        self.assertEqual(r.rc, 0, r.describe())
        clean = [args for args in git_subcommands(self.sb) if args[:1] == ["clean"]]
        self.assertEqual(clean, [["clean", "-fdx"]], "clean is not called with -x")
        self.assertFalse((self.src / "csrc" / "rdna2_tp_kernels.hip").exists())
