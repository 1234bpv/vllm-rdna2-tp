"""Command-line flags: the help text names the flags the parser accepts, and the documented flags exist.

SHELL_TOOLS and PYTHON_TOOLS below list the documented flags of each tool. Two kinds of check.
(1) Each tool's --help names the flags listed for it. (2) Consistency: the flags the parser accepts
(case labels of a shell tool, add_argument calls of a Python tool) are the flags its --help names; a
flag the program accepts but the help does not mention is a gap between the program and its
documentation, and a flag the help names but the program does not accept is the reverse gap.
"""

import re
import unittest

from _support import PYTHON, RELEASE, ReleaseTest

CASE_LABEL = re.compile(r"^\s*((?:-[-A-Za-z][\w-]*)(?:\|-[-A-Za-z][\w-]*)*)\)")
HELP_FLAG = re.compile(r"(?<![\w-])(--?[A-Za-z][\w-]*)")
ADD_ARGUMENT = re.compile(r'add_argument\(\s*"(--[\w-]+)"')

SHELL_TOOLS = {
    "install.sh": ["--prefix", "--python", "--jobs", "--dry-run", "--skip-checks", "--force", "--help"],
    "stop.sh": ["--timeout", "--help"],
    "scripts/doctor.sh": ["--pre", "--json", "--help"],
    "scripts/download-model.sh": ["--dir", "--dry-run", "--help"],
    "scripts/gpu-health.sh": ["--help"],
    "scripts/kill-stuck-rank.sh": ["--help"],
    "serve.sh": ["--model", "--mtp", "--no-mtp", "--ctx", "--streams", "--kv", "--util", "--host", "--port",
                 "--gpus", "--tools", "--no-tools", "--vision", "--no-vision", "--offload-gb", "--tries",
                 "--start-timeout", "--no-probe", "--foreground", "--print-command", "--list", "--help"],
}
PYTHON_TOOLS = {
    "scripts/add-mtp-head.py": ["--repo", "--timeout"],
    "scripts/smoke-test.py": ["--url", "--model", "--timeout", "--needle", "--json"],
}


def help_flags(text):
    """Every flag the help mentions, including those in descriptions such as "(--max-model-len)"."""
    return {m.group(1) for m in HELP_FLAG.finditer(text)}


def option_rows(text):
    """Flags that head an option row of the help: the left column, before two spaces.

    A description may mention a flag of vLLM itself, as serve.sh does with --max-model-len; that
    mention is documentation of a mapping, not a flag the tool accepts, so only rows count.
    """
    flags = set()
    for line in text.splitlines():
        if not re.match(r"^\s{1,4}-", line):
            continue
        head = re.split(r"\s{2,}", line.strip(), maxsplit=1)[0]
        flags.update(part for part in re.split(r"[,\s|]+", head) if part.startswith("-"))
    return flags


def shell_parser_flags(path):
    flags = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        m = CASE_LABEL.match(line)
        if m:
            flags.update(m.group(1).split("|"))
    flags.discard("-h")
    flags.discard("--")
    return flags


def python_parser_flags(path):
    return set(ADD_ARGUMENT.findall(path.read_text(encoding="utf-8")))


class FlagsTest(ReleaseTest):

    def help_text(self, tool, *, python=False):
        if python:
            r = self.sb.run([PYTHON, "-I", "-B", str(RELEASE / tool), "--help"], timeout=60)
        else:
            r = self.sb.bash(tool, ["--help"], timeout=60)
        self.assertEqual(r.rc, 0, r.describe())
        self.assertTrue(r.out.strip(), f"{tool} --help printed nothing on stdout")
        return r.out

    def test_shell_tools_help_names_the_documented_flags(self):
        for tool, flags in SHELL_TOOLS.items():
            with self.subTest(tool=tool):
                named = help_flags(self.help_text(tool))
                missing = sorted(set(flags) - named)
                self.assertEqual(missing, [], f"{tool} --help does not name {missing}")

    def test_python_tools_help_names_the_documented_flags(self):
        for tool, flags in PYTHON_TOOLS.items():
            with self.subTest(tool=tool):
                named = help_flags(self.help_text(tool, python=True))
                missing = sorted(set(flags) - named)
                self.assertEqual(missing, [], f"{tool} --help does not name {missing}")

    def test_shell_parser_accepts_exactly_the_flags_in_help(self):
        for tool in SHELL_TOOLS:
            with self.subTest(tool=tool):
                accepted = shell_parser_flags(RELEASE / tool)
                self.assertTrue(accepted, f"no flags found in the parser of {tool}")
                text = self.help_text(tool)
                rows = option_rows(text) | ({"--help"} if "--help" in help_flags(text) else set())
                undocumented = sorted(accepted - rows)
                self.assertEqual(undocumented, [], f"{tool} accepts flags its option table does not list")

    def test_help_option_table_names_no_flag_the_shell_parser_rejects(self):
        for tool in SHELL_TOOLS:
            with self.subTest(tool=tool):
                accepted = shell_parser_flags(RELEASE / tool) | {"-h", "--help"}
                rows = option_rows(self.help_text(tool))
                phantom = sorted(rows - accepted)
                self.assertEqual(phantom, [], f"{tool} option table lists flags the parser rejects")

    def test_option_table_extraction_skips_descriptions(self):
        sample = ("Options:\n  --model PATH|HF_ID   weights (default: --max-model-len is the vLLM name)\n"
                  "  --mtp, --no-mtp      speculative decoding\n")
        self.assertEqual(option_rows(sample), {"--model", "--mtp", "--no-mtp"})

    def test_python_parser_accepts_exactly_the_flags_in_help(self):
        for tool in PYTHON_TOOLS:
            with self.subTest(tool=tool):
                accepted = python_parser_flags(RELEASE / tool)
                self.assertTrue(accepted, f"no add_argument flags found in {tool}")
                named = help_flags(self.help_text(tool, python=True))
                undocumented = sorted(accepted - named)
                self.assertEqual(undocumented, [], f"{tool} accepts flags its --help does not name")

    def test_checker_detects_a_planted_gap(self):
        # The checker must be able to fail: a parser label with no help line must be found.
        planted = self.sb.root / "planted.sh"
        planted.write_text('case $1 in\n  --secret)\n    ;;\n  --help)\n    ;;\nesac\n', encoding="utf-8")
        accepted = shell_parser_flags(planted)
        self.assertEqual(accepted, {"--secret", "--help"})
        self.assertEqual(sorted(accepted - help_flags("Usage: x [--help]")), ["--secret"])
