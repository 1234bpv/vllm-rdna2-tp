"""Release-wide readiness rules that apply to every tool.

Every shell script passes bash -n, and every Python file compiles. No SIGKILL is sent or enabled in a tool,
including the install-service.sh unit (SendSIGKILL=no, KillSignal=SIGTERM). No sudo runs at a command
position; sudo in printed text is allowed. The Python tools smoke-test.py, add-mtp-head.py and
make_tables.py use the standard library only, and every shell script has its executable bit in the git
index. The scans skip tests/. The shell scans and the unit-file scan also skip src/, the vLLM fork, while
the Python scans include it. Each scanner has a planted self-check, so that a scanner which finds nothing
is not taken for a clean tree.
"""

import ast
import py_compile
import re
import shutil
import subprocess
import sys
import unittest

from _support import BASH, PYTHON, RELEASE, ReleaseTest, msys

EXCLUDED_TOP = {"tests", ".git", "__pycache__"}
SCANNED_OUTSIDE_FORK = {"src"}

SIGKILL_CMD = re.compile(
    r"\bkill\s+(-9|-KILL|-SIGKILL|-s\s+(KILL|SIGKILL|9))\b"
    r"|\bpkill\s+-(9|KILL|SIGKILL)\b"
    r"|\bkillall\s+-(9|KILL|SIGKILL)\b"
    r"|\btimeout\s+(--signal[=\s]+|-s\s+)(KILL|SIGKILL|9)\b"
    r"|\bSendSIGKILL\s*=\s*yes\b"
    r"|\bKillSignal\s*=\s*SIGKILL\b"
    r"|\bsignal\.SIGKILL\b",
    re.IGNORECASE)
SUDO_COMMAND = re.compile(r"(^|[;&|(]|\bthen\b|\bdo\b|\belse\b)\s*sudo\s")
QUOTED = re.compile(r"\"(?:\\.|[^\"\\])*\"|'[^']*'")
HEREDOC = re.compile(r"<<-?\s*'?\"?(\w+)'?\"?")


def unquoted(line):
    """The line with quoted text emptied: a sudo inside a message string is printed text, not a command."""
    return QUOTED.sub('""', line)


def release_files(suffixes, *, fork=False):
    for path in sorted(RELEASE.rglob("*")):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        top = path.relative_to(RELEASE).parts[0]
        if top in EXCLUDED_TOP:
            continue
        if top == "src" and not fork:
            continue
        yield path


def code_lines(path):
    """Yields (number, line) for lines that are code: not comments, not inside a heredoc body."""
    delimiter = None
    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if delimiter is not None:
            if line.strip() == delimiter:
                delimiter = None
            continue
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        yield number, line
        match = HEREDOC.search(line)
        if match and "<<" in line:
            delimiter = match.group(1)


def scan_sigkill(path):
    return [f"{path.relative_to(RELEASE).as_posix()}:{n}: {line.strip()}"
            for n, line in code_lines(path) if SIGKILL_CMD.search(line)]


def scan_sudo(path):
    return [f"{path.relative_to(RELEASE).as_posix()}:{n}: {line.strip()}"
            for n, line in code_lines(path) if SUDO_COMMAND.search(unquoted(line))]


def third_party_imports(path):
    tree = ast.parse(path.read_bytes(), filename=str(path))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    return sorted(m for m in modules if m not in sys.stdlib_module_names)


class ReadinessTest(ReleaseTest):

    def shell(self):
        return sorted(release_files({".sh"}))

    def python(self):
        return sorted(release_files({".py"}, fork=True))

    def test_every_shell_script_passes_bash_syntax_check(self):
        scripts = self.shell()
        names = {p.relative_to(RELEASE).as_posix() for p in scripts}
        self.assertIn("bench/run-matrix.sh", names)
        self.assertIn("scripts/lib.sh", names)
        self.assertGreaterEqual(len(scripts), 12, "fewer shell scripts than expected")
        for path in scripts:
            with self.subTest(script=path.relative_to(RELEASE).as_posix()):
                r = self.sb.run([BASH, "-n", msys(path)], timeout=60)
                self.assertEqual(r.rc, 0, r.describe())

    def test_every_python_file_compiles(self):
        files = self.python()
        names = {p.relative_to(RELEASE).as_posix() for p in files}
        self.assertIn("scripts/smoke-test.py", names)
        self.assertIn("results/make_tables.py", names)
        cache = self.sb.root / "pycache"
        cache.mkdir()
        failures = []
        for index, path in enumerate(files):
            try:
                py_compile.compile(str(path), cfile=str(cache / f"{index}.pyc"), doraise=True)
            except py_compile.PyCompileError as exc:
                failures.append(f"{path.relative_to(RELEASE).as_posix()}: {exc.msg}")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_no_sigkill_in_release_tools(self):
        hits = []
        for path in self.shell() + self.python() + sorted(release_files({".service", ".conf"})):
            hits.extend(scan_sigkill(path))
        # The one permitted place: --force, behind the checks that test_kill_stuck_rank.py covers.
        hits = [hit for hit in hits if not (hit.startswith("scripts/kill-stuck-rank.sh:")
                                            and hit.endswith('kill -KILL "$PID" 2>/dev/null'))]
        self.assertEqual(hits, [], "SIGKILL is sent or enabled:\n" + "\n".join(hits))

    def test_no_sudo_at_command_position_in_tools(self):
        hits = []
        for path in self.shell():
            hits.extend(scan_sudo(path))
        self.assertEqual(hits, [], "sudo is run by a tool:\n" + "\n".join(hits))

    def test_python_tools_use_only_the_standard_library(self):
        for rel in ("scripts/smoke-test.py", "scripts/add-mtp-head.py", "results/make_tables.py"):
            with self.subTest(tool=rel):
                outside = third_party_imports(RELEASE / rel)
                self.assertEqual(outside, [], f"{rel} imports non-standard modules")

    def test_shell_scripts_are_executable_in_git_index(self):
        git = shutil.which("git")
        if not git:
            self.skipTest("git is not on PATH")
        tracked = subprocess.run([git, "ls-files", "--stage", "--", "."], cwd=str(RELEASE),
                                 capture_output=True, text=True, timeout=120)
        if tracked.returncode != 0 or not tracked.stdout.strip():
            self.skipTest("release/ has no files in git yet (untracked in the repository); "
                          "the executable bit cannot be checked in the index")
        modes = {}
        for line in tracked.stdout.splitlines():
            meta, path = line.split("\t", 1)
            modes[path] = meta.split()[0]
        bad = []
        for path in self.shell():
            rel = path.relative_to(RELEASE).as_posix()
            if rel not in modes:
                bad.append(f"{rel}: not in git index")
            elif modes[rel] != "100755":
                bad.append(f"{rel}: mode {modes[rel]}")
        self.assertEqual(bad, [], "\n".join(bad))

    # ----- self-checks of the scanners -----------------------------------------------------
    def test_sigkill_scanner_detects_planted_forms(self):
        planted = [
            "kill -9 $PID", "kill -KILL 123", "kill -s KILL 123", "pkill -9 vllm", "killall -KILL vllm",
            "timeout --signal=KILL 5 cmd", "timeout -s KILL 5 cmd", "SendSIGKILL=yes", "KillSignal=SIGKILL",
            "os.kill(pid, signal.SIGKILL)",
        ]
        for text in planted:
            with self.subTest(text=text):
                self.assertTrue(SIGKILL_CMD.search(text), "scanner misses a SIGKILL form")

    def test_sigkill_scanner_ignores_allowed_forms(self):
        allowed = ["kill -0 $PID", "kill -TERM $PID", "SendSIGKILL=no", "KillSignal=SIGTERM",
                   "SIGKILL is never sent", "rr_error \"SIGKILL is not sent\""]
        for text in allowed:
            with self.subTest(text=text):
                self.assertFalse(SIGKILL_CMD.search(text), "scanner flags an allowed form")

    def test_sudo_scanner_detects_command_and_ignores_printed_text(self):
        def hit(line):
            return bool(SUDO_COMMAND.search(unquoted(line)))
        self.assertTrue(hit("sudo apt-get install -y git"))
        self.assertTrue(hit("  if true; then sudo systemctl restart x; fi"))
        self.assertTrue(hit("cat x | sudo tee y"))
        self.assertFalse(hit('echo "run: sudo apt-get update"'))
        self.assertFalse(hit('die 3 "$PY does not run: install (sudo apt-get install python3.12)"'))
        self.assertFalse(hit('add build-tools FAIL "missing (sudo apt-get install git g++ make)"'))
        self.assertFalse(hit("# sudo systemctl enable"))

    def test_heredoc_bodies_are_not_code(self):
        probe = self.sb.root / "probe.sh"
        probe.write_text("cat <<'EOF'\nsudo cp unit /etc/systemd/system/\nEOF\necho ok\n", encoding="utf-8")
        self.assertEqual(scan_sudo(probe), [], "a heredoc body was scanned as code")

    def test_stdlib_checker_detects_planted_import(self):
        probe = self.sb.root / "probe.py"
        probe.write_text("import json\nimport requests\n", encoding="utf-8")
        self.assertEqual(third_party_imports(probe), ["requests"])
