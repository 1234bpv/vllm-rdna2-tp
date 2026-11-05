"""scripts/smoke-test.py, function chat(): the thinking switch in the request body, and a reasoning-only answer.

Checked on the bytes that the fake server received, not on the code:
  - the body of the chat request carries "chat_template_kwargs": {"enable_thinking": false}, in the plain mode
    and in the needle mode; the value must be JSON false (Python False, not 0 or the string "false");
  - the literal pair `"chat_template_kwargs": {"enable_thinking": false}` appears in the raw body text;
  - an answer with only reasoning_content and no text is not passed: "answer is not empty" fails with the detail
    "empty (reasoning text: 22 characters)" (the summed length of the reasoning chunks), and passed is false;
  - an answer with text and reasoning passes, and the reasoning does not become the answer text.

The fake server is an http.server thread on 127.0.0.1 that keeps the raw request bytes, as in
test_smoke_test.py. Nothing leaves the machine.
"""

import http.server
import json
import re
import threading

from _support import PYTHON, RELEASE, ReleaseTest

SCRIPT = RELEASE / "scripts" / "smoke-test.py"
MODEL_ID = "fake-model"
LITERAL_PAIR = '"chat_template_kwargs": {"enable_thinking": false}'
CODE_RE = re.compile(r"access code is (\d{6})")
REASONING = ["Let me think", " about it."]      # 12 + 10 characters
REASONING_CHARS = 22
USAGE_CHUNK = {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 64, "total_tokens": 84}}


class ChatHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.requests.append(("GET", self.path, b""))
        raw = json.dumps({"object": "list", "data": [{"id": MODEL_ID}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        self.server.requests.append(("POST", self.path, raw))
        prompt = json.loads(raw.decode("utf-8"))["messages"][0]["content"]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in self.server.answer_chunks(prompt):
            self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class ChatServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, mode):
        super().__init__(("127.0.0.1", 0), ChatHandler)
        self.mode = mode
        self.requests = []

    def answer_chunks(self, prompt):
        if self.mode == "reasoning_only":
            pieces = [{"choices": [{"delta": {"reasoning_content": piece}}]} for piece in REASONING]
        elif self.mode == "text_and_reasoning":
            pieces = [{"choices": [{"delta": {"reasoning_content": REASONING[0]}}]},
                      {"choices": [{"delta": {"content": "The sky is blue."}}]}]
        elif self.mode == "needle":
            found = CODE_RE.search(prompt)
            pieces = [{"choices": [{"delta": {"content": found.group(1) if found else "unknown"}}]}]
        else:
            pieces = [{"choices": [{"delta": {"content": "The sky is blue"}}]}]
        return pieces + [USAGE_CHUNK]


class SmokeThinkingTest(ReleaseTest):

    def serve(self, mode):
        server = ChatServer(mode)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self._close, server, thread)
        self.fake = server
        return f"http://127.0.0.1:{server.server_address[1]}"

    @staticmethod
    def _close(server, thread):
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    def smoke(self, url, extra=(), timeout=90):
        argv = [PYTHON, "-I", "-B", str(SCRIPT), "--url", url, "--timeout", "20"] + list(extra)
        return self.sb.run(argv, timeout=timeout)

    def one_chat_body(self):
        """Raw bytes of the single chat request the fake server received."""
        bodies = [raw for method, _, raw in self.fake.requests if method == "POST"]
        self.assertEqual(len(bodies), 1, f"expected one chat request, the server got {len(bodies)}")
        return bodies[0]

    # ----- the request body --------------------------------------------------------------
    def test_plain_request_body_switches_thinking_off_with_json_false(self):
        url = self.serve("plain")
        r = self.smoke(url)
        self.assertEqual(r.rc, 0, r.describe())
        body = json.loads(self.one_chat_body().decode("utf-8"))
        self.assertIn("chat_template_kwargs", body, "the chat request has no chat_template_kwargs")
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False},
                         f"chat_template_kwargs is {body['chat_template_kwargs']!r}")
        self.assertIs(body["chat_template_kwargs"]["enable_thinking"], False,
                      "enable_thinking is not JSON false (0 or a string would compare equal or differ)")

    def test_request_body_text_contains_the_literal_pair(self):
        url = self.serve("plain")
        r = self.smoke(url)
        self.assertEqual(r.rc, 0, r.describe())
        raw = self.one_chat_body().decode("utf-8")
        self.assertIn(LITERAL_PAIR, raw, "the raw request body does not contain the literal pair")

    def test_needle_request_body_switches_thinking_off_too(self):
        url = self.serve("needle")
        r = self.smoke(url, ["--needle", "300"])
        self.assertEqual(r.rc, 0, r.describe())
        body = json.loads(self.one_chat_body().decode("utf-8"))
        self.assertIn("chat_template_kwargs", body, "needle request: no chat_template_kwargs")
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False},
                         f"needle request: chat_template_kwargs is {body['chat_template_kwargs']!r}")
        self.assertIs(body["chat_template_kwargs"]["enable_thinking"], False)

    # ----- a reasoning-only answer -------------------------------------------------------
    def test_reasoning_only_answer_is_not_passed_with_the_reasoning_character_count(self):
        url = self.serve("reasoning_only")
        r = self.smoke(url, ["--json"])
        self.assertEqual(r.rc, 1, r.describe())
        data = json.loads(r.out)
        self.assertFalse(data["passed"], r.out)
        self.assertEqual(data["text"], "", "reasoning text was taken as the answer")
        checks = {c["name"]: c for c in data["checks"]}
        check = checks["answer is not empty"]
        self.assertFalse(check["ok"], r.out)
        self.assertEqual(check["detail"], f"empty (reasoning text: {REASONING_CHARS} characters)", r.out)

    def test_reasoning_only_answer_text_report_shows_the_same_detail(self):
        url = self.serve("reasoning_only")
        r = self.smoke(url)
        self.assertEqual(r.rc, 1, r.describe())
        self.assertIn("output: (empty)", r.out)
        self.assertIn(f"  FAIL answer is not empty: empty (reasoning text: {REASONING_CHARS} characters)", r.out)
        self.assertIn("RESULT: FAILED", r.out)

    def test_reasoning_with_a_text_answer_passes_and_is_not_the_answer(self):
        url = self.serve("text_and_reasoning")
        r = self.smoke(url, ["--json"])
        self.assertEqual(r.rc, 0, r.describe())
        data = json.loads(r.out)
        self.assertTrue(data["passed"], r.out)
        self.assertEqual(data["text"], "The sky is blue.", "the reasoning text leaked into the answer")
        checks = {c["name"]: c for c in data["checks"]}
        self.assertTrue(checks["answer is not empty"]["ok"], r.out)
