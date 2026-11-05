"""scripts/smoke-test.py: the end-to-end check of a live server, against a local fake server.

The script sends GET /v1/models, then POST /v1/chat/completions with stream set, stream_options
include_usage, temperature 0 and max_tokens 64. The answer must not be empty, usage must be reported,
and completion_tokens must be greater than 0. With --needle the prompt hides a six-digit access code near
its start, and the answer must contain it. Exit codes: 0 pass, 1 a check or an HTTP request failed, 2 a bad
call, 3 /v1/models unreachable. Every request has a time limit. The fake server is an http.server thread on
127.0.0.1; nothing leaves the machine.
"""

import http.server
import json
import re
import socket
import threading
import time
import unittest

from _support import PYTHON, RELEASE, ReleaseTest

SCRIPT = RELEASE / "scripts" / "smoke-test.py"
MODEL_ID = "fake-model"
CODE_RE = re.compile(r"access code is (\d{6})")


class FakeHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def _json(self, status, obj):
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        srv = self.server
        srv.requests.append(("GET", self.path, None))
        if srv.models_mode == "http500":
            self.send_error(500, "fake failure")
            return
        if srv.models_mode == "empty":
            self._json(200, {"object": "list", "data": []})
            return
        self._json(200, {"object": "list", "data": [{"id": MODEL_ID}]})

    def do_POST(self):
        srv = self.server
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        srv.requests.append(("POST", self.path, body))
        if srv.chat_mode == "http500":
            self.send_error(500, "fake failure")
            return
        if srv.chat_mode == "sleep":
            time.sleep(srv.sleep_s)
            return
        prompt = body["messages"][0]["content"]
        if srv.answer_mode == "needle":
            found = CODE_RE.search(prompt)
            pieces = [found.group(1) if found else "unknown"]
        elif srv.answer_mode == "empty":
            pieces = []
        else:
            pieces = ["The sky is blue", " because of scattering."]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for piece in pieces:
            self._sse({"choices": [{"delta": {"content": piece}}]})
        if srv.answer_mode != "nousage":
            completion = 0 if srv.answer_mode == "zero" else 12
            self._sse({"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": completion,
                                                "total_tokens": 20 + completion}})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _sse(self, obj):
        self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode("utf-8"))
        self.wfile.flush()


class FakeServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class SmokeTest(ReleaseTest):

    def setUp(self):
        super().setUp()
        self.server = FakeServer(("127.0.0.1", 0), FakeHandler)
        self.server.requests = []
        self.server.models_mode = "ok"
        self.server.chat_mode = "ok"
        self.server.answer_mode = "plain"
        self.server.sleep_s = 10
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.stop_server, thread)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop_server(self, thread):
        self.server.shutdown()
        self.server.server_close()
        thread.join(timeout=5)

    def smoke(self, extra=(), *, timeout=90):
        argv = [PYTHON, "-I", "-B", str(SCRIPT), "--url", self.url, "--timeout", "20"] + list(extra)
        return self.sb.run(argv, timeout=timeout)

    def posts(self):
        return [body for method, path, body in self.server.requests if method == "POST"]

    # ----- usage -----------------------------------------------------------------------
    def test_help_exits_zero(self):
        r = self.sb.run([PYTHON, "-I", "-B", str(SCRIPT), "--help"], timeout=60)
        self.assertEqual(r.rc, 0, r.describe())
        for flag in ("--url", "--model", "--timeout", "--needle", "--json"):
            self.assertIn(flag, r.out)

    def test_bad_calls_exit_2(self):
        bad = [["--url", "ftp://x"], ["--timeout", "0"], ["--timeout", "abc"],
               ["--needle", "0"], ["--needle", "abc"], ["--bogus"]]
        for extra in bad:
            with self.subTest(extra=extra):
                r = self.smoke(extra, timeout=60)
                self.assertEqual(r.rc, 2, r.describe())

    # ----- success and the request ------------------------------------------------------
    def test_success_exits_zero_and_json_reports_pass(self):
        r = self.smoke(["--json"])
        self.assertEqual(r.rc, 0, r.describe())
        data = json.loads(r.out)
        self.assertTrue(data["passed"], r.out)
        names = [c["name"] for c in data["checks"]]
        self.assertIn("answer is not empty", names)
        self.assertIn("usage is reported", names)
        self.assertIn("completion_tokens > 0", names)
        self.assertIn("usage", data)
        self.assertIn("ttft_s", data)
        self.assertIn("decode_tok_s", data)

    def test_request_body_matches_spec(self):
        r = self.smoke()
        self.assertEqual(r.rc, 0, r.describe())
        bodies = self.posts()
        self.assertEqual(len(bodies), 1)
        body = bodies[0]
        self.assertIs(body["stream"], True)
        self.assertIs(body["stream_options"]["include_usage"], True)
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["max_tokens"], 64)
        self.assertEqual(body["model"], MODEL_ID, "the model id is not taken from /v1/models")
        self.assertEqual(body["messages"][0]["role"], "user")

    def test_model_option_overrides_models_list(self):
        r = self.smoke(["--model", "forced-name"])
        self.assertEqual(r.rc, 0, r.describe())
        self.assertEqual(self.posts()[0]["model"], "forced-name")

    def test_models_endpoint_is_asked_first(self):
        self.smoke()
        paths = [path for method, path, _ in self.server.requests]
        self.assertEqual(paths[0], "/v1/models")
        self.assertEqual(paths[1], "/v1/chat/completions")

    # ----- needle -----------------------------------------------------------------------
    def test_needle_answered_correctly_passes(self):
        self.server.answer_mode = "needle"
        r = self.smoke(["--needle", "300", "--json"])
        self.assertEqual(r.rc, 0, r.describe())
        checks = {c["name"]: c for c in json.loads(r.out)["checks"]}
        self.assertTrue(checks["answer contains the number from the start"]["ok"])

    def test_needle_answered_wrongly_is_exit_1(self):
        self.server.answer_mode = "plain"
        r = self.smoke(["--needle", "300"])
        self.assertEqual(r.rc, 1, r.describe())

    def test_needle_code_sits_at_the_start_of_the_prompt(self):
        self.server.answer_mode = "needle"
        r = self.smoke(["--needle", "300"])
        self.assertEqual(r.rc, 0, r.describe())
        prompt = self.posts()[0]["messages"][0]["content"]
        found = CODE_RE.search(prompt)
        self.assertIsNotNone(found, "the prompt has no needle")
        self.assertLess(found.start(), 100, "the needle is not at the start of the prompt")

    # ----- answer checks ----------------------------------------------------------------
    def test_empty_answer_is_exit_1(self):
        self.server.answer_mode = "empty"
        r = self.smoke(["--json"])
        self.assertEqual(r.rc, 1, r.describe())
        checks = {c["name"]: c for c in json.loads(r.out)["checks"]}
        self.assertFalse(checks["answer is not empty"]["ok"])

    def test_missing_usage_is_exit_1(self):
        self.server.answer_mode = "nousage"
        r = self.smoke(["--json"])
        self.assertEqual(r.rc, 1, r.describe())
        checks = {c["name"]: c for c in json.loads(r.out)["checks"]}
        self.assertFalse(checks["usage is reported"]["ok"])

    def test_zero_completion_tokens_is_exit_1(self):
        self.server.answer_mode = "zero"
        r = self.smoke(["--json"])
        self.assertEqual(r.rc, 1, r.describe())
        checks = {c["name"]: c for c in json.loads(r.out)["checks"]}
        self.assertFalse(checks["completion_tokens > 0"]["ok"])

    # ----- server errors ----------------------------------------------------------------
    def test_models_http_error_is_exit_1(self):
        self.server.models_mode = "http500"
        r = self.smoke()
        self.assertEqual(r.rc, 1, r.describe())

    def test_models_list_without_model_is_exit_1(self):
        self.server.models_mode = "empty"
        r = self.smoke()
        self.assertEqual(r.rc, 1, r.describe())

    def test_chat_http_error_is_exit_1(self):
        self.server.chat_mode = "http500"
        r = self.smoke()
        self.assertEqual(r.rc, 1, r.describe())

    def test_unreachable_server_is_exit_3(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        argv = [PYTHON, "-I", "-B", str(SCRIPT), "--url", f"http://127.0.0.1:{port}", "--timeout", "5"]
        r = self.sb.run(argv, timeout=60)
        self.assertEqual(r.rc, 3, r.describe())

    def test_slow_chat_is_bounded_by_timeout(self):
        self.server.chat_mode = "sleep"
        self.server.sleep_s = 15
        argv = [PYTHON, "-I", "-B", str(SCRIPT), "--url", self.url, "--timeout", "2"]
        began = time.monotonic()
        r = self.sb.run(argv, timeout=60)
        elapsed = time.monotonic() - began
        self.assertEqual(r.rc, 1, r.describe())
        self.assertLess(elapsed, 30, "the request was not limited by --timeout")

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
