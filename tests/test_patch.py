"""patches/vllm-rdna2-tp-0.1.0.patch: byte form, manifest, src/ copies, and whether it applies.

install.sh applies the patch to the vLLM base commit with git apply --whitespace=nowarn; a missing or
non-applying patch is exit code 5. The patch must stay byte-exact: it contains no CR bytes, and every git
command that writes into the tree runs with core.autocrlf=false. The upstream files are read from the
repository named by VLLM_BASE_REPO with git archive only; nothing in that repository is changed.

The byte check counts CR bytes in Python. A grep in Git Bash can report a false CRLF finding, so this
test does not use grep.
"""

import os
import pathlib
import re
import shutil
import subprocess
import tarfile
import unittest

from _support import RELEASE, ReleaseTest

PATCH = RELEASE / "patches" / "vllm-rdna2-tp-0.1.0.patch"
MANIFEST = RELEASE / "patches" / "MODIFIED_FILES.txt"
BASE_COMMIT = "c296851a7d173fa89d2eefbca0243be42ae9b5e0"
BASE_REPO = os.environ.get("VLLM_BASE_REPO", "")
DIFF_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)$")


def read_patch():
    return PATCH.read_bytes()


def parse_patch(data):
    """Returns {path: {"new": bool, "plus": [lines]}} for every diff block."""
    blocks = {}
    current = None
    for line in data.decode("utf-8", errors="surrogateescape").split("\n"):
        match = DIFF_HEADER.match(line)
        if match:
            current = match.group(2)
            blocks[current] = {"new": False, "plus": [], "headers": []}
            continue
        if current is None:
            continue
        if line.startswith("new file mode"):
            blocks[current]["new"] = True
        if line.startswith("+") and not line.startswith("+++ "):
            blocks[current]["plus"].append(line[1:])
        blocks[current]["headers"].append(line)
    return blocks


def read_manifest():
    entries = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        kind, path = line.split("\t", 1)
        entries[path] = kind
    return entries


class PatchFormTest(ReleaseTest):

    def test_patch_has_no_carriage_returns(self):
        data = read_patch()
        cr = data.count(b"\r")
        self.assertEqual(cr, 0, f"the patch has {cr} CR bytes ({data.count(b'\r\n')} CRLF); it must be LF only")

    def test_patch_is_nonempty_and_ends_with_newline(self):
        data = read_patch()
        self.assertGreater(len(data), 0)
        self.assertEqual(data[-1:], b"\n")

    def test_gitattributes_keep_the_patch_byte_exact(self):
        text = (RELEASE / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("*.patch -text", text.splitlines())
        self.assertIn("* text=auto eol=lf", text.splitlines())

    def test_every_diff_block_has_file_headers(self):
        data = read_patch().decode("utf-8", errors="surrogateescape")
        blocks = parse_patch(read_patch())
        self.assertGreater(len(blocks), 0)
        for path, block in blocks.items():
            with self.subTest(path=path):
                joined = "\n".join(block["headers"])
                if block["new"] and not block["plus"]:
                    # git writes no ---/+++ lines for an empty new file; the index line names
                    # the empty blob e69de29 (git's hash of zero bytes).
                    self.assertIn("e69de29bb2d1d6434b8b29ae775ad8c2e48c5391", joined)
                    continue
                if block["new"]:
                    self.assertIn("--- /dev/null", joined)
                else:
                    self.assertIn(f"--- a/{path}", joined)
                self.assertIn(f"+++ b/{path}", joined)
        self.assertEqual(data.count("\ndiff --git ") + (1 if data.startswith("diff --git ") else 0), len(blocks))


class PatchManifestTest(ReleaseTest):

    def test_manifest_lists_exactly_the_patched_files(self):
        blocks = parse_patch(read_patch())
        manifest = read_manifest()
        self.assertEqual(sorted(blocks), sorted(manifest),
                         "MODIFIED_FILES.txt and the patch name different files")

    def test_added_and_modified_marks_match_the_patch(self):
        blocks = parse_patch(read_patch())
        manifest = read_manifest()
        for path, block in blocks.items():
            with self.subTest(path=path):
                expected = "A" if block["new"] else "M"
                self.assertEqual(manifest.get(path), expected,
                                 f"manifest says {manifest.get(path)}, patch says {expected}")

    def test_manifest_has_comment_header_and_only_known_marks(self):
        lines = MANIFEST.read_text(encoding="utf-8").splitlines()
        self.assertTrue(lines[0].startswith("#"), "no comment header")
        for line in lines:
            if line.strip() and not line.startswith("#"):
                self.assertRegex(line, r"^[AM]\t\S")

    def test_src_copies_match_added_files_in_the_patch(self):
        blocks = parse_patch(read_patch())
        src = RELEASE / "src"
        copies = [p for p in src.rglob("*") if p.is_file()] if src.exists() else []
        self.assertGreater(len(copies), 0, "src/ has no files to compare")
        for copy in copies:
            rel = copy.relative_to(src).as_posix()
            with self.subTest(file=rel):
                self.assertIn(rel, blocks, "src/ file is not added by the patch")
                self.assertTrue(blocks[rel]["new"], "src/ file is not a new file in the patch")
                expected = "\n".join(blocks[rel]["plus"]) + "\n"
                actual = copy.read_bytes().decode("utf-8")
                self.assertEqual(actual, expected, "src/ copy differs from the patch's added lines")


class PatchAppliesTest(ReleaseTest):

    def setUp(self):
        super().setUp()
        self.git = shutil.which("git")

    def base_ready(self):
        if not self.git:
            self.skipTest("git is not on PATH")
        if not BASE_REPO:
            self.skipTest("VLLM_BASE_REPO is not set; the base commit is needed for git apply --check")
        if not pathlib.Path(BASE_REPO).exists():
            self.skipTest(f"base repository {BASE_REPO} is not present (set VLLM_BASE_REPO)")
        result = subprocess.run([self.git, "-C", BASE_REPO, "cat-file", "-t", BASE_COMMIT],
                                capture_output=True, text=True, timeout=60)
        if result.returncode != 0 or result.stdout.strip() != "commit":
            self.skipTest(f"commit {BASE_COMMIT} is not in {BASE_REPO}")

    def extract_modified(self, dst):
        """Writes the upstream files the patch modifies into dst (read-only git archive)."""
        blocks = parse_patch(read_patch())
        modified = sorted(p for p, b in blocks.items() if not b["new"])
        archive = subprocess.run([self.git, "-C", BASE_REPO, "archive", "--format=tar", BASE_COMMIT, "--"] + modified,
                                 capture_output=True, timeout=300)
        self.assertEqual(archive.returncode, 0, archive.stderr.decode("utf-8", "replace"))
        import io
        with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
            tar.extractall(dst)
        return modified

    def apply_check(self, tree, patch_path):
        return subprocess.run([self.git, "apply", "--check", "--whitespace=nowarn", str(patch_path)],
                              cwd=str(tree), capture_output=True, text=True, timeout=300)

    def test_patch_applies_to_the_base_commit(self):
        self.base_ready()
        tree = self.sb.root / "base"
        tree.mkdir()
        self.extract_modified(tree)
        result = self.apply_check(tree, PATCH)
        self.assertEqual(result.returncode, 0, f"git apply --check failed:\n{result.stderr}")

    def test_check_detects_a_corrupted_patch(self):
        # The check must be able to fail: a patch with one removed line altered must not apply.
        self.base_ready()
        tree = self.sb.root / "base"
        tree.mkdir()
        self.extract_modified(tree)
        data = read_patch().decode("utf-8")
        lines = data.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("-") and not line.startswith("---"):
                lines[i] = line + "X"
                break
        else:
            self.fail("no removed line to corrupt")
        bad = self.sb.root / "bad.patch"
        bad.write_bytes("\n".join(lines).encode("utf-8"))
        result = self.apply_check(tree, bad)
        self.assertNotEqual(result.returncode, 0, "a corrupted patch passed git apply --check")
