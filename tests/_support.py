"""Shared helpers for the release test suite. Standard library only.

Rules that every helper here keeps:

- every external run has a time limit (subprocess timeout); a run that exceeds it
  fails the test with its partial output, it is never read as silence;
- every test works in its own sandbox (HOME, VLLM_RDNA2_HOME, ROCM_PATH, TMPDIR and
  a stub directory placed first on PATH); nothing is written outside it;
- stubs are small bash scripts: each call is logged to RT_CALLS, and the answer comes
  from files in RT_STATE, so a test decides the outcome of every external command;
- nothing reaches the network: curl, git and pip are stubs, and the add-mtp-head
  test replaces urllib in its own process;
- no helper sends SIGKILL. Background stubs exit on a stop file, and a watchdog in
  the vllm stub ends them after RT_MAX_LIFE seconds even when a test fails.
- the vllm stub records, at each serve launch, the PATH it received and what ninja and
  hipcc resolve to on that PATH (state/vllm-serve.env); the venv's ninja answers --version
  and records the path it was run from (state/ninja.invoked). Setting state/no-ninja makes
  the pip build-dependency step leave no ninja behind.
- the python stub answers the sysconfig query of doctor.sh from state/pyinclude/Python.h,
  which exists by default (python3.12-dev installed); a test removes it to get a FAIL.
"""

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

TESTS = pathlib.Path(__file__).resolve().parent
RELEASE = TESTS.parent
PYTHON = sys.executable
ORIG_PATH = os.environ.get("PATH", "")
# Captured before any PATH change; the sleep stub calls this binary for the short waits.
REAL_SLEEP = shutil.which("sleep") or "/usr/bin/sleep"


def _find_bash():
    override = os.environ.get("RELEASE_TESTS_BASH")
    if override:
        return override
    if os.name != "nt":
        return shutil.which("bash") or "/bin/bash"
    # On Windows the first bash on PATH may be the WSL launcher in System32, which
    # cannot run these scripts. Git Bash is the one that does.
    for directory in ORIG_PATH.split(os.pathsep):
        candidate = os.path.join(directory, "bash.exe")
        if os.path.isfile(candidate) and "system32" not in candidate.lower():
            return candidate
    return "bash"


BASH = _find_bash()


def msys(path):
    """Path as Git Bash sees it: C:\\x\\y becomes /c/x/y. Other systems: unchanged."""
    text = str(path)
    if os.name != "nt":
        return text
    match = re.match(r"^([A-Za-z]):[\\/]*(.*)$", text)
    if match:
        return "/" + match.group(1).lower() + "/" + match.group(2).replace("\\", "/")
    return text.replace("\\", "/")


# Variables of the developer's shell that change release behaviour; removed so that
# a test sees only what it sets.
_DROP_PREFIXES = ("VLLM_", "HIP_", "PYTORCH_", "REASONEND", "PROBE_TIMEOUT", "DOCTOR_",
                  "RR_", "CMAKE_", "MAX_JOBS", "SETUPTOOLS_SCM", "OMP_", "SAFETENSORS_",
                  "HF_", "CUDA_", "ROCR_", "HSA_", "TMPDIR_", "RT_")


class Result:
    def __init__(self, argv, returncode, stdout, stderr):
        self.argv = argv
        self.rc = returncode
        self.out = stdout
        self.err = stderr

    @property
    def both(self):
        return self.out + self.err

    def describe(self):
        return f"rc={self.rc}\n--- stdout ---\n{self.out}\n--- stderr ---\n{self.err}"


_STUB_COMMON = r'''
_rt_log() {
  local a
  { printf '%s' "$(basename -- "$0")"; for a in "$@"; do a=${a//$'\n'/\\n}; a=${a//$'\t'/\\t}; printf '\t%s' "$a"; done; printf '\n'; } >> "$RT_CALLS"
}
_one() { local v; v=$(cat "$RT_STATE/$1" 2>/dev/null); printf '%s' "${v:-$2}"; }
_seq() {
  local n v
  n=$(cat "$RT_STATE/$1.n" 2>/dev/null); n=$(( ${n:-0} + 1 )); echo "$n" > "$RT_STATE/$1.n"
  v=$(sed -n "${n}p" "$RT_STATE/$1.seq" 2>/dev/null)
  [ -n "$v" ] || v=$(tail -n 1 "$RT_STATE/$1.seq" 2>/dev/null)
  printf '%s' "${v:-$2}"
}
'''

GIT_STUB = _STUB_COMMON + r'''
_rt_log "$@"
# git.leftover lists files that an earlier run left in the source tree (paths relative to it,
# the working directory). clean removes the untracked ones; upstream's .gitignore covers *.hip,
# so those stay unless -x is given. apply refuses to run while any leftover exists.
_git_clean() {
  local a x=0 f rc
  for a in "$@"; do case $a in -[!-]*) case $a in *x*) x=1 ;; esac ;; esac; done
  rc=$(_one git.clean.rc 0)
  [ "$rc" = 0 ] || exit "$rc"
  for f in $(cat "$RT_STATE/git.leftover" 2>/dev/null); do
    case $f in *.hip) [ "$x" = 1 ] || continue ;; esac
    rm -f -- "$f"
  done
  exit 0
}
while [ $# -gt 0 ]; do case $1 in -C|-c) shift 2 ;; *) break ;; esac; done
sub=$1; shift
case $sub in
  init) mkdir -p .git ;;
  remote)
    case $1 in
      get-url) if [ -e "$RT_STATE/git.remote" ]; then cat "$RT_STATE/git.remote"; else exit 2; fi ;;
      add|set-url) printf '%s\n' "$3" > "$RT_STATE/git.remote" ;;
    esac ;;
  fetch) exit "$(_one git.fetch.rc 0)" ;;
  checkout) mkdir -p requirements && : > requirements/rocm.txt; exit "$(_one git.checkout.rc 0)" ;;
  clean) _git_clean "$@" ;;
  apply)
    for f in $(cat "$RT_STATE/git.leftover" 2>/dev/null); do
      if [ -e "$f" ]; then echo "error: $f: already exists in working directory" >&2; exit 1; fi
    done
    exit "$(_one git.apply.rc 0)" ;;
esac
exit 0
'''

CURL_STUB = _STUB_COMMON + r'''
_rt_log "$@"
url=
for a in "$@"; do case $a in http://*|https://*) url=$a ;; esac; done
case $url in
  */v1/models)
    [ -e "$RT_STATE/ready" ] || exit 7
    printf '%s\n' "$(_one models.json '{"object":"list","data":[{"id":"stub-model","root":"stub-root"}]}')"
    exit 0 ;;
  */v1/chat/completions)
    case $(_seq chat ok) in
      ok) printf '%s\n' '{"choices":[{"message":{"role":"assistant","content":"One."}}]}' ;;
      nochoices) printf '%s\n' '{"choices":[]}' ;;
      fail) exit 22 ;;
    esac
    exit 0 ;;
esac
exit 22
'''

JOURNAL_STUB = _STUB_COMMON + r'''
_rt_log "$@"
cat "$RT_STATE/journal.txt" 2>/dev/null
exit "$(_one journal.rc 0)"
'''

SLEEP_STUB = r'''#!/bin/bash
# Fast sleep for the tests: the scripts wait in seconds, the tests cannot.
exec "$RT_REAL_SLEEP" 0.02
'''

LSOF_STUB = r'''#!/bin/bash
# Reports every /dev/shm region as held, except the test's own regions (name contains
# rt-own-). serve.sh removes only the regions that nothing holds, so a host's real
# vLLM regions are never removed by a test.
case "$*" in *rt-own-*) exit 1 ;; esac
echo 1
'''

UNAME_STUB = r'''#!/bin/bash
case $1 in
  -s) printf '%s\n' "$(cat "$RT_STATE/uname.s" 2>/dev/null || echo Linux)" ;;
  *) command -p uname "$@" ;;
esac
'''

ID_STUB = r'''#!/bin/bash
case $1 in
  -un) printf '%s\n' "$(cat "$RT_STATE/user.name" 2>/dev/null || echo relt)" ;;
  -Gn) printf '%s\n' "$(cat "$RT_STATE/user.groups" 2>/dev/null || echo users)" ;;
  *) command -p id "$@" ;;
esac
'''

DF_STUB = r'''#!/bin/bash
# Free space is set by the test (df.kb, in KiB); the output format is df -P.
printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\n'
printf 'stub 1000000000 1000 %s 1%% %s\n' "$(cat "$RT_STATE/df.kb" 2>/dev/null || echo 100000000000)" "${!#}"
'''

BUILD_STUB = r'''#!/bin/bash
# Existence-only stub for doctor.sh (g++, make) and similar build tools.
printf '%s\n' "$(basename -- "$0")" >> "$RT_CALLS"
exit 0
'''

HIPCONFIG_STUB = _STUB_COMMON + r'''
_rt_log "$@"
printf '%s\n' "$(_one hip.version 7.2.4)"
exit 0
'''

HIPCC_STUB = _STUB_COMMON + r'''
_rt_log "$@"
exit 0
'''

ROCMINFO_STUB = _STUB_COMMON + r'''
_rt_log "$@"
echo "ROCk module is loaded"
echo "HSA System Attributes"
g=0
while IFS= read -r name; do
  [ -n "$name" ] || continue
  printf 'Agent %d\n' "$g"
  printf '  Name:                    %s\n' "$name"
  printf '  Marketing Name:          AMD Radeon RX 6800\n'
  printf '      Name:                    amdgcn-amd-amdhsa--%s\n' "$name"
  g=$((g + 1))
done < "$RT_STATE/gpus.txt"
# rocminfo.stderr: text for stderr, for example an HSA error with ANSI colours.
if [ -e "$RT_STATE/rocminfo.stderr" ]; then cat "$RT_STATE/rocminfo.stderr" >&2; fi
exit "$(_one rocminfo.rc 0)"
'''

NINJA_STUB = _STUB_COMMON + r'''
_rt_log "$@"
printf '%s\n' "$0" >> "$RT_STATE/ninja.invoked"
case ${1:-} in
  --version) echo "1.11.1 (stub)" ;;
esac
exit 0
'''

PIP_STUB = _STUB_COMMON + r'''
_rt_log "$@"
exit 0
'''

HF_STUB = _STUB_COMMON + r'''
_rt_log "$@"
if [ "${1:-}" = download ]; then
  dir=
  while [ $# -gt 0 ]; do
    if [ "$1" = --local-dir ]; then dir=$2; shift; fi
    shift
  done
  rc=$(_one hf.rc 0)
  [ "$rc" = 0 ] || exit "$rc"
  mkdir -p "$dir" && printf '%s\n' '{"quantization_config":{"ignore":[]}}' > "$dir/config.json"
fi
exit 0
'''

# One Python stub serves as the venv interpreter, as python3.12 on PATH, and as the
# interpreter named by --python. It behaves by the arguments it gets.
PYTHON_STUB = _STUB_COMMON + r'''
_rt_log "$@"
me=$(basename -- "$0")
ver=$(_one "pyver.$me" "$(_one pyver 3.12)")
case $1 in
  -m)
    case $2 in
      venv)
        dir=$3
        mkdir -p "$dir/bin" || exit 1
        cp "$RT_TPL/python" "$dir/bin/python" && cp "$RT_TPL/pip" "$dir/bin/pip" \
          && cp "$RT_TPL/vllm" "$dir/bin/vllm" && cp "$RT_TPL/hf" "$dir/bin/hf" || exit 1
        exit "$(_one venv.rc 0)" ;;
      pip)
        shift 2
        tag=other
        case " $* " in
          *" setuptools-scm "*) tag=deps ;;
          *" --extra-index-url "*) tag=torch ;;
          *amd_smi*) tag=amdsmi ;;
          *requirements/rocm.txt*) tag=reqs ;;
          *" -e "*) tag=build ;;
        esac
        {
          printf 'tag=%s\n' "$tag"
          printf 'PATH=%s\n' "$PATH"
          printf 'ROCM_PATH=%s\n' "$ROCM_PATH"
          printf 'HIP_PATH=%s\n' "$HIP_PATH"
          printf 'PYTORCH_ROCM_ARCH=%s\n' "$PYTORCH_ROCM_ARCH"
          printf 'VLLM_TARGET_DEVICE=%s\n' "$VLLM_TARGET_DEVICE"
          printf 'SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=%s\n' "$SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM"
          printf 'CMAKE_BUILD_PARALLEL_LEVEL=%s\n' "$CMAKE_BUILD_PARALLEL_LEVEL"
          printf 'MAX_JOBS=%s\n' "$MAX_JOBS"
          printf 'PWD=%s\n' "$PWD"
        } > "$RT_STATE/last-pip.env"
        rc=$(_one "fail.$tag" 0)
        # The build dependencies bring ninja into the venv, as pip does for real.
        if [ "$tag" = deps ] && [ "$rc" = 0 ] && [ ! -e "$RT_STATE/no-ninja" ]; then
          cp "$RT_TPL/ninja" "$(dirname -- "$0")/ninja" || exit 1
        fi
        exit "$rc" ;;
      *) exit 0 ;;
    esac ;;
  -I)
    case $2 in
      -)
        cat > /dev/null
        case $(_seq probe ok) in
          ok) echo ok ;;
          hang) exec "$RT_REAL_SLEEP" 30 ;;
          wrong) echo "sum 1.0 differs from reference 2.0" >&2; exit 1 ;;
          nan) echo "GPU 0: not a finite number (nan)" >&2; exit 1 ;;
        esac
        exit 0 ;;
      -c)
        code=$3
        case $code in
          *is_available*) printf 'vllm 0.0.0 torch 2.13.0+rocm7.2\n'; exit "$(_one postcheck.rc 0)" ;;
          *sysconfig*) [ -e "$RT_STATE/pyinclude/Python.h" ]; exit $? ;;
          *sys.version_info*) printf '%s\n' "$ver"; exit 0 ;;
          *can_device_access_peer*) printf '%s\n' "$(_one p2p.out False)"; exit 0 ;;
          *host_all_reduce*)
            rc=$(_one import.rc 0)
            if [ "$rc" != 0 ]; then printf '%s\n' "$(_one import.err 'ModuleNotFoundError: No module named vllm')" >&2; fi
            exit "$rc" ;;
        esac
        exit 0 ;;
      *)
        case $2 in
          */awq-to-ct-check.py) exit "$(_one fail.check 0)" ;;
          */awq-to-ct.py) exit "$(_one fail.convert 0)" ;;
        esac
        exit 0 ;;
    esac ;;
esac
exit 0
'''

# vllm stub: serve mode prints the KV line, touches the ready file (the curl stub
# answers /v1/models only then), and waits. vllm.seq gives one mode per start.
VLLM_STUB = _STUB_COMMON + r'''
_rt_log "$@"
if [ "${1:-}" = --help ]; then echo "usage: vllm (stub)"; exit "$(_one vllm_help.rc 0)"; fi
if [ "${1:-}" = serve ]; then
  { printf 'PATH=%s\n' "$PATH"; printf 'ninja=%s\n' "$(command -v ninja)"; printf 'hipcc=%s\n' "$(command -v hipcc)"; } > "$RT_STATE/vllm-serve.env"
fi
mode=$(_seq vllm serve)
case $mode in
  exit1) echo "Traceback (stub): simulated start failure"; exit 1 ;;
  serve) echo "INFO: GPU KV cache size: 1,234,567 tokens (stub)" ;;
  illegal) echo "HIP error: illegal memory access (stub)" ;;
  fatal) echo "EngineCore encountered a fatal error (stub)" ;;
  ignoreterm) trap '' TERM ;;
  silent) ;;
esac
trap 'rm -f "$RT_STATE/ready"' EXIT
[ "$mode" = serve ] && touch "$RT_STATE/ready"
[ "$mode" = ignoreterm ] || trap 'exit 0' TERM INT
deadline=$(( $(date +%s) + ${RT_MAX_LIFE:-120} ))
while [ ! -e "$RT_STATE/stop-now" ] && [ "$(date +%s)" -lt "$deadline" ]; do "$RT_REAL_SLEEP" 0.1; done
exit 0
'''


def write_text(path, text):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_script(path, body):
    write_text(path, body if body.startswith("#!") else "#!/bin/bash\n" + body)
    os.chmod(path, 0o755)


class Sandbox:
    """Private tree for one test: HOME, install home, ROCm, stubs, state and calls log."""

    def __init__(self):
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="relt-"))
        self.home = self.root / "home"
        self.rdh = self.root / "rdh"            # VLLM_RDNA2_HOME
        self.rocm = self.root / "rocm"          # ROCM_PATH
        self.stubs = self.root / "stubs"        # first on PATH
        self.tpl = self.root / "tpl"            # templates copied into a stub venv
        self.state = self.root / "state"
        self.tmp = self.root / "tmp"
        self.calls_file = self.root / "calls.log"
        for directory in (self.home, self.rdh, self.rocm / "bin", self.stubs, self.tpl,
                          self.state, self.tmp):
            directory.mkdir(parents=True, exist_ok=True)
        self.calls_file.touch()
        self.procs = []
        self._write_stubs()
        self.set("gpus.txt", "gfx1030\ngfx1030\n")
        self.set("hip.version", "7.2.4")
        self.set("pyver", "3.12")
        self.set("df.kb", "100000000000")
        self.set("user.groups", "users video")
        write_text(self.state / "pyinclude" / "Python.h", "")

    # ----- stubs ---------------------------------------------------------------------
    def _write_stubs(self):
        for name, body in (("python", PYTHON_STUB), ("pip", PIP_STUB), ("vllm", VLLM_STUB),
                           ("hf", HF_STUB), ("ninja", NINJA_STUB)):
            write_script(self.tpl / name, body)
        for name, body in (("git", GIT_STUB), ("curl", CURL_STUB), ("journalctl", JOURNAL_STUB),
                           ("sleep", SLEEP_STUB), ("lsof", LSOF_STUB), ("uname", UNAME_STUB),
                           ("id", ID_STUB), ("df", DF_STUB), ("g++", BUILD_STUB),
                           ("make", BUILD_STUB), ("python3.12", PYTHON_STUB)):
            write_script(self.stubs / name, body)
        write_script(self.rocm / "bin" / "rocminfo", ROCMINFO_STUB)
        write_script(self.rocm / "bin" / "hipconfig", HIPCONFIG_STUB)
        write_script(self.rocm / "bin" / "hipcc", HIPCC_STUB)
        (self.rocm / "share" / "amd_smi").mkdir(parents=True, exist_ok=True)
        write_text(self.rocm / "share" / "amd_smi" / "setup.py", "# amdsmi source (stub)\n")

    def add_stub(self, name, body):
        write_script(self.stubs / name, body)

    # ----- state ----------------------------------------------------------------------
    def set(self, name, value):
        write_text(self.state / name, str(value))

    def seq(self, name, values):
        """Modes for successive calls of a stub; the last one repeats."""
        write_text(self.state / (name + ".seq"), "\n".join(values) + "\n")
        (self.state / (name + ".n")).unlink(missing_ok=True)

    def install_venv(self, version="0.1.0"):
        venv_bin = self.rdh / "venv" / "bin"
        venv_bin.mkdir(parents=True, exist_ok=True)
        for name in ("python", "pip", "vllm", "hf", "ninja"):
            shutil.copy(self.tpl / name, venv_bin / name)
            os.chmod(venv_bin / name, 0o755)
        write_text(self.rdh / f".installed-{version}", f"version {version}\n")

    def marker(self, version="0.1.0"):
        return self.rdh / f".installed-{version}"

    # ----- environment and runs -------------------------------------------------------
    def env(self, extra=None):
        env = {k: v for k, v in os.environ.items()
               if not k.upper().startswith(_DROP_PREFIXES) and k.upper() != "PATH"}
        dirs = [msys(self.stubs)] + [msys(p) for p in ORIG_PATH.split(os.pathsep) if p]
        env.update({
            "PATH": ":".join(dirs),
            "HOME": msys(self.home),
            "VLLM_RDNA2_HOME": msys(self.rdh),
            "ROCM_PATH": msys(self.rocm),
            "TMPDIR": msys(self.tmp),
            "USER": "relt",
            "LC_ALL": "C",
            "LANG": "C",
            "TERM": "dumb",
            "RT_STATE": msys(self.state),
            "RT_CALLS": msys(self.calls_file),
            "RT_REAL_SLEEP": msys(REAL_SLEEP),
            "RT_TPL": msys(self.tpl),
            "RT_MAX_LIFE": "120",
        })
        if extra:
            env.update(extra)
        return env

    def run(self, argv, *, timeout, env=None, cwd=None, stdin=None):
        """Runs argv (already in bash form) under a time limit and returns a Result."""
        argv = [str(a) for a in argv]
        try:
            proc = subprocess.run(
                argv, env=self.env(env), cwd=str(cwd or self.root), input=stdin,
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=timeout, start_new_session=(os.name != "nt"))
        except subprocess.TimeoutExpired as exc:
            partial_out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            partial_err = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            raise AssertionError(
                f"time limit of {timeout} s reached: {argv[:3]}\n--- partial stdout ---\n{partial_out}"
                f"\n--- partial stderr ---\n{partial_err}")
        return Result(argv, proc.returncode, proc.stdout, proc.stderr)

    def bash(self, script, args=(), *, timeout=60, env=None, cwd=None, stdin=None):
        """Runs a release script (path relative to the release root) with bash."""
        return self.run([BASH, msys(RELEASE / script)] + list(args), timeout=timeout,
                        env=env, cwd=cwd, stdin=stdin)

    def bash_snippet(self, snippet, args=(), *, timeout=30, env=None):
        return self.run([BASH, "-c", snippet, "bash"] + list(args), timeout=timeout, env=env)

    def calls(self, name=None):
        """Logged calls as lists: [command, arg1, ...]. Filtered by command name."""
        rows = []
        text = self.calls_file.read_text(encoding="utf-8") if self.calls_file.exists() else ""
        for line in text.splitlines():
            parts = line.split("\t")
            if name is None or parts[0] == name:
                rows.append(parts)
        return rows

    def count(self, name, *needles):
        return sum(1 for row in self.calls(name)
                   if all(any(needle in arg for arg in row[1:]) for needle in needles))

    # ----- background processes -------------------------------------------------------
    def spawn(self, body, *, name="proc", argv0=None, ignore_term=False):
        """Starts a bash process that writes its PID to <name>.pid and waits until
        <name>.stop exists. With argv0, its /proc/PID/cmdline starts with that text.
        Returns (pid, popen, stop_path). Its PID is the bash process's own $$."""
        pid_file = self.state / f"{name}.pid"
        stop_file = self.state / f"{name}.stop"
        prologue = "trap '' TERM; " if ignore_term else "trap 'exit 0' TERM; "
        inner = f'echo $$ > "$1"; {prologue}{body}; while [ ! -e "$2" ]; do "$RT_REAL_SLEEP" 0.1; done'
        if argv0:
            # The outer bash replaces itself with a bash that carries the wanted argv[0];
            # exec keeps the PID, so $$ inside the body is the PID the test reads.
            argv = [BASH, "-c", 'exec -a "$0" "$BASH" -c "$1" bash "$2" "$3"',
                    argv0, inner, msys(pid_file), msys(stop_file)]
        else:
            argv = [BASH, "-c", inner, "bash", msys(pid_file), msys(stop_file)]
        popen = subprocess.Popen(argv, env=self.env(), stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 start_new_session=(os.name != "nt"))
        self.procs.append((popen, stop_file))
        for _ in range(400):
            if pid_file.exists() and pid_file.read_text().strip().isdigit():
                return int(pid_file.read_text().strip()), popen, stop_file
            if popen.poll() is not None:
                break
            time.sleep(0.05)
        raise AssertionError(f"background process {name} did not start: rc={popen.poll()}")

    def bash_alive(self, pid):
        """True when the MSYS/Linux process with this PID still exists."""
        result = self.run([BASH, "-c", 'kill -0 "$1" 2>/dev/null', "bash", str(pid)], timeout=20)
        return result.rc == 0

    def release_copy(self):
        """Copy of the release tree (without tests/) for tests that change files."""
        dst = self.root / "release"
        shutil.copytree(RELEASE, dst, ignore=shutil.ignore_patterns("tests", ".git", "__pycache__"))
        return dst

    def stop_background(self):
        for _, stop_file in self.procs:
            try:
                stop_file.touch()
            except OSError:
                pass
        (self.state / "stop-now").touch()
        for popen, _ in self.procs:
            try:
                popen.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    def cleanup(self):
        self.stop_background()
        shutil.rmtree(self.root, ignore_errors=True)


class ReleaseTest(unittest.TestCase):
    """Base class: one sandbox per test, removed afterwards."""

    def setUp(self):
        self.sb = Sandbox()
        self.addCleanup(self.sb.cleanup)


def parse_printed_command(text):
    """Parses the output of serve.sh --print-command.

    Returns (env, argv): env maps the leading KEY=value words to values, argv is the
    vllm command word list. Lines are joined on backslash-newline before splitting.
    """
    import shlex
    joined = re.sub(r"\\\r?\n", " ", text)
    words = shlex.split(joined)
    env = {}
    index = 0
    while index < len(words) and re.match(r"^[A-Z_][A-Z0-9_]*=", words[index]):
        key, _, value = words[index].partition("=")
        env[key] = value
        index += 1
    return env, words[index:]


def flag_value(argv, flag):
    """Value after the last occurrence of flag (argparse keeps the last), or None."""
    value = None
    for index, word in enumerate(argv[:-1]):
        if word == flag:
            value = argv[index + 1]
    return value


def flag_positions(argv, flag):
    return [index for index, word in enumerate(argv) if word == flag]


def json_loads_strict(text):
    return json.loads(text)
