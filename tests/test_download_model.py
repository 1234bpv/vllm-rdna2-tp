"""scripts/download-model.sh: downloads through the venv's hf, the conversion chain, exit codes and dry run.

Downloads go only through $VENV/bin/hf. A missing installation or a missing hf is exit code 3, and hf is
not called. --dry-run does not check the installation and creates nothing. Exit codes: 0 success, 1 failure
(download, conversion or check), 2 wrong call, 3 not installed. A CONVERT preset is downloaded, converted
with awq-to-ct.py, checked with awq-to-ct-check.py, and its final directory is <PRESET>-ct-int4. The default
directory is models/<PRESET> under the install home. The hf and converter stubs record their arguments.
"""

import pathlib
import re
import unittest

from _support import RELEASE, ReleaseTest, msys

SCRIPT = RELEASE / "scripts" / "download-model.sh"
PRESET_HF = {
    "qwen3.8-27b": "philbert440/Qwen3.8-27B-W4A16-AWQ",
    "qwen3.5-9b": "atalsharma/Qwen3.5-9B-AWQ",
    "qwen3.5-35b-a3b": "cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit",
    "qwen3.6-35b-a3b": "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
    "gemma-4-26b-a4b": "cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4",
}


class DownloadModelTest(ReleaseTest):

    def setUp(self):
        super().setUp()

    def install(self):
        self.sb.install_venv()

    def download(self, args, *, timeout=120):
        return self.sb.bash("scripts/download-model.sh", list(args), timeout=timeout)

    def hf_calls(self):
        return [row for row in self.sb.calls("hf") if len(row) > 1 and row[1] == "download"]

    def converter_calls(self, name):
        return [row for row in self.sb.calls("python") if any(name in a for a in row[1:])]

    # ----- usage -----------------------------------------------------------------------
    def test_help_exits_zero(self):
        r = self.download(["--help"], timeout=60)
        self.assertEqual(r.rc, 0, r.describe())
        self.assertIn("--dry-run", r.out)

    def test_bad_calls_exit_2(self):
        bad = [[], ["no-such-preset"], ["qwen3.8-27b", "--dir"], ["qwen3.8-27b", "--bogus"],
               ["qwen3.8-27b", "extra-word"]]
        for args in bad:
            with self.subTest(args=args):
                r = self.download(args, timeout=60)
                self.assertEqual(r.rc, 2, r.describe())

    # ----- dry run and install guard ---------------------------------------------------
    def test_dry_run_prints_hf_command_and_creates_nothing_without_install(self):
        r = self.download(["qwen3.8-27b", "--dry-run"], timeout=60)
        self.assertEqual(r.rc, 0, r.describe())
        self.assertIn("philbert440/Qwen3.8-27B-W4A16-AWQ", r.both)
        self.assertIn("--local-dir", r.both)
        self.assertFalse((self.sb.rdh / "models").exists(), "dry run created the models directory")
        self.assertEqual(self.hf_calls(), [], "dry run called hf")

    def test_dry_run_hf_ids_match_the_spec_table(self):
        for preset, hf_id in PRESET_HF.items():
            with self.subTest(preset=preset):
                r = self.download([preset, "--dry-run"], timeout=60)
                self.assertEqual(r.rc, 0, r.describe())
                self.assertIn(hf_id, r.both)

    def test_missing_install_is_code3_and_hf_is_not_called(self):
        r = self.download(["qwen3.8-27b"], timeout=60)
        self.assertEqual(r.rc, 3, r.describe())
        self.assertEqual(self.hf_calls(), [])

    def test_missing_hf_in_venv_is_code3(self):
        self.install()
        (self.sb.rdh / "venv" / "bin" / "hf").unlink()
        r = self.download(["qwen3.8-27b"], timeout=60)
        self.assertEqual(r.rc, 3, r.describe())

    # ----- plain download ----------------------------------------------------------------
    def test_download_calls_hf_download_with_hf_id_and_default_dir(self):
        self.install()
        r = self.download(["qwen3.8-27b"])
        self.assertEqual(r.rc, 0, r.describe())
        rows = self.hf_calls()
        self.assertEqual(len(rows), 1, rows)
        args = rows[0][1:]
        self.assertEqual(args[:2], ["download", "philbert440/Qwen3.8-27B-W4A16-AWQ"], args)
        self.assertEqual(args[2], "--local-dir")
        self.assertTrue(args[3].endswith("models/qwen3.8-27b"), args[3])
        self.assertTrue((self.sb.rdh / "models" / "qwen3.8-27b" / "config.json").is_file())

    def test_dir_option_is_used_as_local_dir(self):
        self.install()
        target = self.sb.root / "elsewhere"
        r = self.download(["qwen3.8-27b", "--dir", msys(target)])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertEqual(self.hf_calls()[0][4], msys(target))

    def test_download_failure_is_code1(self):
        self.install()
        self.sb.set("hf.rc", "1")
        r = self.download(["qwen3.8-27b"])
        self.assertEqual(r.rc, 1, r.describe())
        self.assertIn("download of", r.err)

    def test_download_timeout_is_twelve_hours(self):
        # The download time limit is 12 hours. The script passes it as the timeout of rr_run; the
        # test reads the constant from the script, because the hf stub cannot observe that timeout.
        text = SCRIPT.read_text(encoding="utf-8")
        match = re.search(r"^DOWNLOAD_TIMEOUT=(\d+)", text, re.M)
        self.assertIsNotNone(match, "DOWNLOAD_TIMEOUT is not set")
        self.assertEqual(int(match.group(1)), 12 * 3600)

    def test_only_hf_is_used_for_downloads(self):
        text = SCRIPT.read_text(encoding="utf-8")
        code = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
        self.assertFalse(any("huggingface-cli" in line for line in code),
                         "download-model.sh calls huggingface-cli (only comments may mention it)")
        self.assertIn('"$RR_VENV/bin/hf" download', text)

    # ----- CONVERT presets -------------------------------------------------------------
    def test_convert_preset_downloads_converts_and_checks(self):
        self.install()
        r = self.download(["qwen3.6-35b-a3b"])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertEqual(len(self.hf_calls()), 1)
        convert = self.converter_calls("awq-to-ct.py")
        check = self.converter_calls("awq-to-ct-check.py")
        self.assertEqual(len(convert), 1, convert)
        self.assertEqual(len(check), 1, check)
        args = convert[0][1:]
        self.assertEqual(args[0], "-I", args)
        self.assertTrue(args[1].endswith("scripts/awq-to-ct.py"), args)
        self.assertTrue(args[args.index("--src") + 1].endswith("models/qwen3.6-35b-a3b"), args)
        self.assertTrue(args[args.index("--dst") + 1].endswith("models/qwen3.6-35b-a3b-ct-int4"), args)
        self.assertIn("qwen3.6-35b-a3b-ct-int4", r.both, "the final directory is not reported")

    def test_failed_conversion_is_code1_and_no_check_runs(self):
        self.install()
        self.sb.set("fail.convert", "1")
        r = self.download(["qwen3.6-35b-a3b"])
        self.assertEqual(r.rc, 1, r.describe())
        self.assertEqual(self.converter_calls("awq-to-ct-check.py"), [])

    def test_failed_check_is_code1(self):
        self.install()
        self.sb.set("fail.check", "1")
        r = self.download(["qwen3.6-35b-a3b"])
        self.assertEqual(r.rc, 1, r.describe())
        self.assertIn("conversion check failed", r.err)

    def test_convert_dry_run_names_the_final_directory(self):
        r = self.download(["qwen3.6-35b-a3b", "--dry-run"], timeout=60)
        self.assertEqual(r.rc, 0, r.describe())
        self.assertIn("qwen3.6-35b-a3b-ct-int4", r.both)
