"""Documentation checks: links, anchors, generic forbidden forms, the version, and the tests README link.

Markdown links and HTML href and src targets must resolve inside the release, and no link may point at
a local path. Anchors follow the GitHub slug rule: lowercase, punctuation removed except "-" and "_",
spaces replaced by "-", and a repeated heading gets "-1", "-2". No release text file may contain a
generic private form: a home directory path, a carrier-grade or private network address, or a path with
a drive letter. The forbidden-form check knows only these generic forms; names of people and machines
are not matched, and tests/README.md lists that as not covered. VERSION must be 0.1.0, and docs/INSTALL.md
and docs/README.md must link to tests/README.md.
"""

import re
import unittest
from urllib.parse import unquote

from _support import RELEASE, ReleaseTest

MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HTML_LINK = re.compile(r"""(?:href|src)=["']([^"']+)["']""")
FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")

FORBIDDEN = [
    ("home directory", re.compile(r"/home/[A-Za-z0-9._-]+")),
    ("carrier-grade address", re.compile(r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b")),
    ("private address, class A", re.compile(r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
    ("private address, class C", re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b")),
    ("private address, class B", re.compile(r"\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b")),
    ("Windows drive path", re.compile(r"\b[A-Za-z]:[\\/][A-Za-z_]")),
    ("MSYS drive path", re.compile(r"/[a-z]/Users")),
    ("macOS home path", re.compile(r"/Users/[A-Za-z]")),
]


def prose_lines(text):
    """(number, line) for lines outside fenced code blocks."""
    fenced = False
    for number, line in enumerate(text.splitlines(), 1):
        if FENCE.match(line):
            fenced = not fenced
            continue
        if not fenced:
            yield number, line


def slugify(heading):
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading).strip()
    text = re.sub(r"[^\w\- ]", "", text.lower())
    return text.replace(" ", "-")


def anchors_of(text):
    seen = {}
    found = set()
    for _, line in prose_lines(text):
        match = HEADING.match(line)
        if not match:
            continue
        base = slugify(match.group(1))
        count = seen.get(base, 0)
        found.add(base if count == 0 else f"{base}-{count}")
        seen[base] = count + 1
    return found


def markdown_files():
    return sorted(p for p in RELEASE.rglob("*.md") if ".git" not in p.parts and "__pycache__" not in p.parts)


def text_files():
    for path in sorted(RELEASE.rglob("*")):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        yield path


def forbidden_hits(path):
    data = path.read_bytes()
    if b"\0" in data[:8000]:
        return []
    hits = []
    rel = path.relative_to(RELEASE).as_posix()
    for number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
        for label, pattern in FORBIDDEN:
            if pattern.search(line):
                hits.append(f"{rel}:{number}: {label}: {line.strip()[:120]}")
    return hits


class DocsTest(ReleaseTest):

    def test_relative_links_and_anchors_resolve_inside_the_release(self):
        broken = []
        for md in markdown_files():
            rel = md.relative_to(RELEASE).as_posix()
            text = md.read_text(encoding="utf-8")
            for number, line in prose_lines(text):
                for target in MD_LINK.findall(line) + HTML_LINK.findall(line):
                    if target.startswith(("http://", "https://", "mailto:", "file:")):
                        continue
                    path_part, _, anchor = target.partition("#")
                    if path_part.startswith("/"):
                        broken.append(f"{rel}:{number}: {target}: root-relative link")
                        continue
                    dest = md if not path_part else (md.parent / unquote(path_part))
                    dest = dest.resolve()
                    try:
                        dest.relative_to(RELEASE.resolve())
                    except ValueError:
                        broken.append(f"{rel}:{number}: {target}: points outside the release")
                        continue
                    if not dest.exists():
                        broken.append(f"{rel}:{number}: {target}: missing file")
                        continue
                    if anchor and dest.suffix == ".md":
                        if anchor not in anchors_of(dest.read_text(encoding="utf-8")):
                            broken.append(f"{rel}:{number}: {target}: anchor not found")
        self.assertEqual(broken, [], f"{len(broken)} broken links:\n" + "\n".join(broken))

    def test_no_file_links_to_local_checkouts_in_release_docs(self):
        # A file:// link or a root-relative link points outside the release, where the public tree
        # cannot resolve it. Release docs refer to paths only through variables.
        local = []
        for md in markdown_files():
            rel = md.relative_to(RELEASE).as_posix()
            for number, line in prose_lines(md.read_text(encoding="utf-8")):
                for target in MD_LINK.findall(line) + HTML_LINK.findall(line):
                    if target.startswith("file:") or target.startswith("/"):
                        local.append(f"{rel}:{number}: {target}")
        self.assertEqual(local, [], f"{len(local)} links to local paths:\n" + "\n".join(local[:20]))

    def test_no_forbidden_strings_in_release_files(self):
        hits = []
        for path in text_files():
            hits.extend(forbidden_hits(path))
        self.assertEqual(hits, [], f"{len(hits)} forbidden occurrences:\n" + "\n".join(hits))

    def test_forbidden_scanner_detects_planted_terms(self):
        # The sample is assembled from pieces, so that no forbidden form appears verbatim in this file.
        parts = [
            "see " + "/ho" + "me/alice/x",
            ".".join(("100", "100", "1", "2")),
            ".".join(("10", "1", "2", "3")),
            ".".join(("192", "168", "0", "5")),
            ".".join(("172", "20", "1", "1")),
            "D" + ":/work",
            "/c/" + "Users/x",
            "/" + "Users/alice",
        ]
        sample = ", ".join(parts)
        found = {label for label, pattern in FORBIDDEN if pattern.search(sample)}
        self.assertEqual(found, {label for label, _ in FORBIDDEN},
                         "a forbidden form is not detected by its pattern")

    def test_forbidden_scanner_accepts_loopback_and_variables(self):
        sample = "listen 127.0.0.1:8100 on 0.0.0.0, path $VLLM_RDNA2_HOME/models, url http://localhost:8100/v1"
        found = [label for label, pattern in FORBIDDEN if pattern.search(sample)]
        self.assertEqual(found, [])

    def test_version_file_is_0_1_0(self):
        self.assertEqual((RELEASE / "VERSION").read_text(encoding="utf-8").strip(), "0.1.0")

    def test_tests_readme_exists_and_is_linked_from_install_and_docs(self):
        target = RELEASE / "tests" / "README.md"
        self.assertTrue(target.is_file(), "tests/README.md is missing")
        for doc in ("docs/INSTALL.md", "docs/README.md"):
            with self.subTest(doc=doc):
                text = (RELEASE / doc).read_text(encoding="utf-8")
                links = [t for _, line in prose_lines(text) for t in MD_LINK.findall(line)
                         if t.partition("#")[0].endswith("tests/README.md")]
                self.assertTrue(links, f"{doc} has no link to tests/README.md")
                resolved = [(RELEASE / doc).parent / t.partition("#")[0] for t in links]
                self.assertTrue(any(p.resolve() == target.resolve() for p in resolved),
                                f"{doc} links to tests/README.md through a wrong path: {links}")

    def test_install_doc_apt_line_installs_python312_dev(self):
        lines = (RELEASE / "docs" / "INSTALL.md").read_text(encoding="utf-8").splitlines()
        apt = [line for line in lines if "apt-get install" in line and "python3.12" in line]
        self.assertTrue(apt, "no apt-get install line names python3.12")
        self.assertTrue(any("python3.12-dev" in line for line in apt),
                        f"no apt-get line installs python3.12-dev: {apt}")

    def test_install_doc_check_table_names_python_dev_and_ninja(self):
        lines = (RELEASE / "docs" / "INSTALL.md").read_text(encoding="utf-8").splitlines()
        rows = [line for line in lines if line.startswith("|")]
        self.assertTrue(any(line.startswith("| `python-dev` |") for line in rows),
                        "the check table has no python-dev row")
        self.assertTrue(any("`ninja`" in line for line in rows), "the check table does not name ninja")

    def test_configuration_doc_says_parallel_sizes_are_last_and_double_dash_cannot_change_them(self):
        text = (RELEASE / "docs" / "CONFIGURATION.md").read_text(encoding="utf-8")
        sentences = [line for _, line in prose_lines(text)
                     if "--pipeline-parallel-size" in line and "--tensor-parallel-size" in line]
        self.assertTrue(any("always puts them last" in s and "cannot change them" in s for s in sentences),
                        f"no sentence says the parallel sizes come last and -- cannot change them: {sentences}")

    def test_configuration_doc_names_path_as_passed_value_in_venv_then_rocm_order(self):
        text = (RELEASE / "docs" / "CONFIGURATION.md").read_text(encoding="utf-8")
        self.assertIn("PATH=<venv>/bin:<ROCm>/bin:$PATH", text)

    def test_bench_readme_warns_that_torchrun_sends_sigkill_after_30_seconds(self):
        # The sentence is wrapped across lines in the file, so the check joins the words first.
        text = " ".join((RELEASE / "bench" / "README.md").read_text(encoding="utf-8").split())
        self.assertIn("`torchrun` sends SIGKILL to a worker that survives SIGTERM for 30 s", text)

    def test_bench_readme_asks_for_the_venv_and_rocm_bin_on_path_for_torchrun(self):
        text = " ".join((RELEASE / "bench" / "README.md").read_text(encoding="utf-8").split())
        self.assertIn("with `<venv>/bin` and `<ROCm>/bin` on `PATH`", text)

    def test_performance_pipeline_command_starts_with_venv_then_rocm_on_path(self):
        text = (RELEASE / "docs" / "PERFORMANCE.md").read_text(encoding="utf-8")
        self.assertIn("PATH=<venv>/bin:/opt/rocm/bin:$PATH HIP_VISIBLE_DEVICES=0,1,2", text)

    def test_performance_llamacpp_command_uses_the_engine_option(self):
        lines = (RELEASE / "docs" / "PERFORMANCE.md").read_text(encoding="utf-8").splitlines()
        commands = [line for line in lines if "bench/kv-concurrency.py" in line]
        self.assertTrue(any("--engine llamacpp" in line for line in commands),
                        f"no llama.cpp command with --engine llamacpp: {commands}")
