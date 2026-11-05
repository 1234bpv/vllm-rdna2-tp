"""scripts/add-mtp-head.py: download of the MTP head through HTTP ranges, checks, codes.

The script takes MODEL_DIR and --repo, --timeout and --help. It reads byte ranges over HTTP and needs
exactly 15 mtp.* tensors; any other count writes nothing and exits 1. It writes
MODEL_DIR/model-mtp.safetensors, copies config.json to config.json.before-mtp (only when that copy does
not exist yet), adds the eight modules to quantization_config.ignore without duplicates, and is idempotent
(exit 0 without changes). Exit codes: 0 success, 1 failure, 2 bad call. A 4xx answer is not retried;
a 5xx answer is retried a bounded number of times.

No network: a runner process replaces urllib.request.urlopen and time.sleep before it runs
the script with runpy. The fake repository is built in the test from real safetensors bytes.
"""

import base64
import json
import pathlib
import struct
import unittest

from _support import PYTHON, RELEASE, ReleaseTest

SCRIPT = RELEASE / "scripts" / "add-mtp-head.py"
REPO = "Qwen/Qwen3.5-9B"
SHARDS = [
    "model.safetensors-00002-of-00004.safetensors",
    "model.safetensors-00003-of-00004.safetensors",
    "model.safetensors-00004-of-00004.safetensors",
]
IGNORE_ADD = [
    "mtp.fc",
    "mtp.layers.0.mlp.down_proj",
    "mtp.layers.0.mlp.gate_proj",
    "mtp.layers.0.mlp.up_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.o_proj",
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.v_proj",
]

# Runs the script with urllib and sleep replaced. The JSON input file (argv[1]) holds the shard
# bytes, the plan of answers per shard (an int status, "urlerror", or 200), and the request log.
RUNNER = r'''
import base64, json, io, runpy, sys, time, urllib.error, urllib.request

spec = json.load(open(sys.argv[1], encoding="utf-8"))
shards = {k: base64.b64decode(v) for k, v in spec["shards"].items()}
plan = {k: list(v) for k, v in spec["plan"].items()}
log = open(spec["log"], "a", encoding="utf-8")
time.sleep = lambda seconds: None

class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False

def fake_urlopen(req, timeout=None):
    url = req.full_url
    rng = req.get_header("Range")
    name = url.rsplit("/", 1)[-1]
    log.write(json.dumps({"url": url, "range": rng, "timeout": timeout}) + "\n")
    log.flush()
    step = plan.get(name, [200]).pop(0) if plan.get(name) else 200
    if step == "urlerror":
        raise urllib.error.URLError("network unreachable (fake)")
    if isinstance(step, int) and step != 200:
        raise urllib.error.HTTPError(url, step, "fake status", {}, None)
    blob = shards[name]
    if rng:
        begin, end = rng.split("=", 1)[1].split("-")
        blob = blob[int(begin):int(end) + 1]
    return FakeResponse(blob)

urllib.request.urlopen = fake_urlopen
sys.argv = [spec["script"]] + spec["argv"]
try:
    runpy.run_path(spec["script"], run_name="__main__")
    code = 0
except SystemExit as exc:
    code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
sys.exit(code)
'''


def safetensors_bytes(tensors):
    """tensors: name -> (dtype, shape, data). Returns the file bytes."""
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    order = sorted(tensors)
    for name in order:
        dtype, shape, data = tensors[name]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)
    body = b"".join(tensors[name][2] for name in order)
    return struct.pack("<Q", len(blob)) + blob + body


def tensor(seed, n=16):
    return ("BF16", [n // 2], bytes((seed * 7 + i) % 251 for i in range(n)))


class AddMtpHeadTest(ReleaseTest):

    def setUp(self):
        super().setUp()
        self.model = self.sb.root / "model"
        self.model.mkdir()
        self.config = {"model_type": "qwen", "quantization_config": {"quant_method": "awq",
                                                                    "ignore": ["lm_head"]}}
        self.write_config(self.config)
        self.log = self.sb.root / "requests.log"
        self.log.touch()
        self.plan = {}
        self.shard_bytes = self.make_shards(per_shard=5, extra=True)

    def write_config(self, config):
        (self.model / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    def make_shards(self, *, per_shard=5, extra=True):
        """Three shards, each with per_shard mtp.* tensors and one ordinary tensor.
        Keeps the tensor dictionaries in self.tensors and returns the file bytes."""
        self.tensors = {}
        index = 0
        for shard in SHARDS:
            tensors = {}
            for _ in range(per_shard):
                index += 1
                tensors[f"mtp.layers.0.w{index}.weight"] = tensor(index)
            if extra:
                tensors[f"model.layers.{index}.weight"] = tensor(100 + index)
            self.tensors[shard] = tensors
        return {shard: safetensors_bytes(tensors) for shard, tensors in self.tensors.items()}

    def run_script(self, argv, *, timeout=120):
        runner = self.sb.root / "runner.py"
        runner.write_text(RUNNER, encoding="utf-8")
        spec = self.sb.root / "spec.json"
        spec.write_text(json.dumps({
            "script": str(SCRIPT),
            "argv": [str(a) for a in argv],
            "shards": {k: base64.b64encode(v).decode("ascii") for k, v in self.shard_bytes.items()},
            "plan": self.plan,
            "log": str(self.log),
        }), encoding="utf-8")
        return self.sb.run([PYTHON, "-I", "-B", str(runner), str(spec)], timeout=timeout)

    def requests(self):
        return [json.loads(line) for line in self.log.read_text().splitlines() if line.strip()]

    def head_path(self):
        return self.model / "model-mtp.safetensors"

    # ----- usage -----------------------------------------------------------------------
    def test_help_exits_zero(self):
        r = self.run_script(["--help"])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertIn("--repo", r.out)
        self.assertIn("--timeout", r.out)

    def test_bad_calls_exit_2(self):
        bad = [[], [str(self.sb.root / "missing")], [str(self.model), "--timeout", "0"],
               [str(self.model), "--bogus"]]
        for argv in bad:
            with self.subTest(argv=argv):
                r = self.run_script(argv)
                self.assertEqual(r.rc, 2, r.describe())

    def test_missing_config_json_exits_2(self):
        (self.model / "config.json").unlink()
        r = self.run_script([str(self.model)])
        self.assertEqual(r.rc, 2, r.describe())

    # ----- success ---------------------------------------------------------------------
    def test_success_writes_head_backup_and_ignore_entries(self):
        r = self.run_script([str(self.model)])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertTrue(self.head_path().is_file(), "model-mtp.safetensors was not written")
        self.assertTrue((self.model / "config.json.before-mtp").is_file(), "no config backup")
        backup = json.loads((self.model / "config.json.before-mtp").read_text(encoding="utf-8"))
        self.assertEqual(backup, self.config, "the backup is not the original config.json")
        config = json.loads((self.model / "config.json").read_text(encoding="utf-8"))
        ignore = config["quantization_config"]["ignore"]
        self.assertEqual(ignore[0], "lm_head", "existing ignore entries must be kept")
        for name in IGNORE_ADD:
            with self.subTest(name=name):
                self.assertEqual(ignore.count(name), 1, f"{name} must appear exactly once")
        self.assertEqual(config["model_type"], "qwen", "other config keys changed")

    def test_head_file_holds_15_tensors_with_source_bytes(self):
        r = self.run_script([str(self.model)])
        self.assertEqual(r.rc, 0, r.describe())
        data = self.head_path().read_bytes()
        (length,) = struct.unpack("<Q", data[:8])
        header = json.loads(data[8:8 + length])
        mtp = sorted(k for k in header if k.startswith("mtp."))
        self.assertEqual(len(mtp), 15, mtp)
        source = {}
        for shard, blob in self.shard_bytes.items():
            (src_len,) = struct.unpack("<Q", blob[:8])
            src_header = json.loads(blob[8:8 + src_len])
            for name, info in src_header.items():
                if name.startswith("mtp."):
                    begin, end = info["data_offsets"]
                    source[name] = blob[8 + src_len + begin:8 + src_len + end]
        data_start = 8 + length
        for name in mtp:
            with self.subTest(name=name):
                begin, end = header[name]["data_offsets"]
                self.assertEqual(data[data_start + begin:data_start + end], source[name])

    def test_second_run_is_a_no_op_without_requests(self):
        self.assertEqual(self.run_script([str(self.model)]).rc, 0)
        config_before = (self.model / "config.json").read_bytes()
        head_before = self.head_path().read_bytes()
        self.log.write_text("")
        r = self.run_script([str(self.model)])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertIn("nothing to do", r.out)
        self.assertEqual(self.requests(), [], "the second run made network requests")
        self.assertEqual((self.model / "config.json").read_bytes(), config_before)
        self.assertEqual(self.head_path().read_bytes(), head_before)

    def test_config_without_ignore_list_exits_1_and_writes_nothing(self):
        self.write_config({"model_type": "qwen"})
        r = self.run_script([str(self.model)])
        self.assertEqual(r.rc, 1, r.describe())
        self.assertFalse(self.head_path().exists(), "a head was written for an unpatchable config")

    def test_repo_option_reaches_the_url(self):
        # The fake repository answers for any name, so the run succeeds; only the URLs matter.
        r = self.run_script([str(self.model), "--repo", "example/other-repo"])
        self.assertEqual(r.rc, 0, r.describe())
        urls = [row["url"] for row in self.requests()]
        self.assertTrue(urls, "no request was made")
        self.assertTrue(all(u.startswith("https://huggingface.co/example/other-repo/resolve/main/") for u in urls), urls)

    def test_timeout_is_passed_to_every_request(self):
        r = self.run_script([str(self.model), "--timeout", "7"])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertTrue(self.requests())
        self.assertTrue(all(row["timeout"] == 7 for row in self.requests()))

    def test_ranges_are_used_not_whole_files(self):
        r = self.run_script([str(self.model)])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertTrue(all(row["range"] for row in self.requests()), "a request without a Range header")

    # ----- head that is incomplete or missing ------------------------------------------
    def test_fourteen_tensors_is_exit_1_and_nothing_is_written(self):
        # Rebuild the last shard without one mtp tensor: a complete, valid shard with 14 in all.
        last = SHARDS[-1]
        tensors = self.tensors[last]
        del tensors[sorted(k for k in tensors if k.startswith("mtp."))[0]]
        self.shard_bytes[last] = safetensors_bytes(tensors)
        config_before = (self.model / "config.json").read_bytes()
        r = self.run_script([str(self.model)])
        self.assertEqual(r.rc, 1, r.describe())
        self.assertIn("expected 15", r.err)
        self.assertFalse(self.head_path().exists(), "a partial head was written")
        self.assertEqual((self.model / "config.json").read_bytes(), config_before)

    def test_client_error_is_final_and_not_retried(self):
        self.plan = {SHARDS[1]: [404]}
        r = self.run_script([str(self.model)])
        self.assertEqual(r.rc, 1, r.describe())
        hits = [row for row in self.requests() if row["url"].endswith(SHARDS[1])]
        self.assertEqual(len(hits), 1, f"a 4xx answer was retried: {len(hits)} requests")
        self.assertFalse(self.head_path().exists())

    def test_server_error_is_retried_then_succeeds(self):
        self.plan = {SHARDS[0]: [503, 200]}
        r = self.run_script([str(self.model)])
        self.assertEqual(r.rc, 0, r.describe())

    def test_server_error_every_time_gives_up_with_exit_1(self):
        self.plan = {SHARDS[0]: [503, 503, 503, 503, 503]}
        r = self.run_script([str(self.model)])
        self.assertEqual(r.rc, 1, r.describe())
        hits = [row for row in self.requests() if row["url"].endswith(SHARDS[0])]
        self.assertGreater(len(hits), 1, "a 5xx answer was not retried")
        self.assertLessEqual(len(hits), 3 * 2, "retries are not bounded")
        self.assertIn("nothing was written", r.err)

    def test_network_error_is_exit_1_and_nothing_is_written(self):
        self.plan = {SHARDS[2]: ["urlerror"] * 5}
        r = self.run_script([str(self.model)])
        self.assertEqual(r.rc, 1, r.describe())
        self.assertIn("nothing was written", r.err)
        self.assertFalse(self.head_path().exists())

    def test_complete_head_with_missing_ignore_entries_updates_config_only(self):
        # Head already complete: the script must not download again.
        self.assertEqual(self.run_script([str(self.model)]).rc, 0)
        config = json.loads((self.model / "config.json").read_text(encoding="utf-8"))
        config["quantization_config"]["ignore"] = ["lm_head"]
        self.write_config(config)
        (self.model / "config.json.before-mtp").unlink()
        self.log.write_text("")
        r = self.run_script([str(self.model)])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertEqual(self.requests(), [], "a complete head was downloaded again")
        restored = json.loads((self.model / "config.json").read_text(encoding="utf-8"))
        for name in IGNORE_ADD:
            self.assertIn(name, restored["quantization_config"]["ignore"])

    def test_standard_library_only(self):
        import ast
        import sys
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules.add(node.module.split(".")[0])
        outside = sorted(m for m in modules if m not in sys.stdlib_module_names)
        self.assertEqual(outside, [], "non-stdlib imports")
