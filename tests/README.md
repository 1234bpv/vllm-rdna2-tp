# Tests of the release tools

These tests check the scripts of the release (`install.sh`, `serve.sh`, `stop.sh`, `scripts/`,
`results/make_tables.py`, `bench/kv-concurrency.py`, `patches/`, `docs/`) without GPUs and without network access. Each
tool runs as a real process inside a temporary sandbox. The commands that touch hardware or
the system (`rocminfo`, `hipconfig`, `hipcc`, `git`, `curl`, `journalctl`, `sleep`, the venv's
`python`, `pip`, `vllm`, `hf`) are stubs placed first on `PATH`; they record their calls and
return the answers the test sets up. `smoke-test.py` talks to a fake HTTP server on
`127.0.0.1` that runs in a thread of the test process.

## Run

From the release root:

    timeout 900 python3 -B -I -m unittest discover -s tests -v

On Windows the interpreter is usually named `python`; use that name in the same command.
The tests use the standard library only. Bash is required for the shell tools; on Windows
the harness finds the bash that ships with Git for Windows. Every subprocess has a time limit
in its test, so a tool that hangs fails its test instead of stopping the run.

## Reference run

Version 0.1.0, measured on 2026-11-05.

On the reference machine (Kubuntu 24.04, Python 3.12), with no vLLM server running, twice:

    Ran 369 tests in 166.851s
    OK (skipped=10)

    Ran 369 tests in 165.679s
    OK (skipped=10)

These two runs predate `kill-stuck-rank.sh --force` and `serve.sh --kill-stuck`, and their 11 new tests.
The full current set, 380 tests, was run only on Windows (below).

On Linux, the `--force` code with the previous tests gave 3 failures out of 369. Two were expected, a missing
flag table entry and the ban on SIGKILL, and the new tests cover them.

Two tests in `test_serve_launch.py` can be flaky on Linux. The cause is not established:

- `test_process_exit_fails_attempt` failed once (`1 != 2` starts). It passed in 3 of 3 runs alone and in both
  full runs above.
- `test_driver_failure_after_failed_attempt_is_code4` failed once (`0 != 1` starts), in the run with 3 failures.

On Windows 11 with Python 3.12.6 and the bash of Git for Windows, with the command above and
`VLLM_BASE_REPO` set to a clone of the base repository that contains the base commit:

    Ran 380 tests in 616.122s
    OK (skipped=9)

The run prints one `DeprecationWarning` from the `tarfile` call in `test_patch.py`; the test passes.
Do not run the full set while a vLLM server is running on the same machine.

## Layout

| File | Covers |
|---|---|
| `_support.py` | Harness: sandbox, stubs, process helpers. Not a test module. |
| `test_install.py` | `install.sh`: steps, git calls (`checkout`, `clean -fdx`, `apply`), leftovers of an earlier run, build environment and its PATH, ninja check after install, `--python` without `Python.h`, failure codes, dry run (PATH printed as `$PATH`), `--force`, Python versions |
| `test_serve_args.py` | `serve.sh`: presets, options, `--print-command` (PATH line as `"$PATH"`), reasoning and tool settings, parallel sizes after `--` |
| `test_serve_launch.py` | `serve.sh`: startup, PATH of the started server, probe, retries, driver checks, PID file, `--foreground`, `clean_shm` |
| `test_stop.py` | `stop.sh`: process identity (vLLM by command line, worker rank, process group), a group that cannot be read, a process of another user, SIGTERM only, timeout, survivors |
| `test_stop_zombie.py` | `stop.sh`, `alive()`: a zombie leader after SIGTERM, a zombie leader with a live group member, a zombie with live threads (stat `Zl`), a `ps` that does not answer (falls back to `kill -0`) |
| `test_doctor.py` | `scripts/doctor.sh`: each check, python-dev and ninja checks, thresholds, gpu-count without a gfx1030 device (rocminfo error on stderr), JSON output, exit codes |
| `test_gpu_health.py` | `scripts/gpu-health.sh`: probe result, hang, kernel log patterns |
| `test_kill_stuck_rank.py` | `scripts/kill-stuck-rank.sh`: gates, refusals, SIGTERM by default, survivors, `--force` and its checks |
| `test_download_model.py` | `scripts/download-model.sh`: dry run, `hf` call, conversion chain, timeout |
| `test_install_service.py` | `scripts/install-service.sh`: printed unit fields, refusals, nothing installed |
| `test_add_mtp_head.py` | `scripts/add-mtp-head.py`: Range requests, retries, header check, config backup |
| `test_smoke_test.py` | `scripts/smoke-test.py`: request body, checks, exit codes, timeouts |
| `test_smoke_thinking.py` | `scripts/smoke-test.py`: `chat_template_kwargs` in the request body the server receives (plain and needle), a reasoning-only answer and its character count |
| `test_tables.py` | `results/make_tables.py`: `--check`, rewrite, markers, `summary.csv`; no `point=profile` record in the concurrency measurements; on synthetic fixtures the `older` runs (chains, cycles, unknown ids, order in `runs.json`), the marks of older points, the all-reduce A/B mean and gain, the rows of `summary.csv`; on the committed data the `older` links of the release runs, valid release-tree numbers, no host path, LF line endings under `--check` |
| `test_kv_concurrency.py` | `bench/kv-concurrency.py`: `run_limited` (time limit, SIGTERM, five-second grace, `None` on expiry); `--engine` and its old spelling `--rushij` |
| `test_patch.py` | `patches/vllm-rdna2-tp-0.1.0.patch`: bytes, manifest, `src/` copies, `git apply --check` |
| `test_docs.py` | Documentation: relative links and anchors, generic forbidden forms, version, link to this file, install and configuration statements |
| `test_flags.py` | Command-line flags: help text against the option parser of each tool |
| `test_readiness.py` | Release-wide: `bash -n`, compilation, no SIGKILL outside `kill-stuck-rank.sh --force`, no `sudo` in commands, stdlib-only Python tools, executable bits |

## Coverage by tool

| Tool or area | Test file | Tests |
|---|---|---|
| `install.sh` | `test_install.py` | 44 |
| `serve.sh`: arguments, presets, `--print-command` (PATH line), MTP head, `--list`, `--help` | `test_serve_args.py` | 30 |
| `serve.sh`: launch, PATH of the server, readiness, probe, retries, driver gate, `--foreground` | `test_serve_launch.py` | 27 |
| `stop.sh` | `test_stop.py` | 14 |
| `stop.sh`: zombie leader, its group, `Zl`, `ps` unavailable | `test_stop_zombie.py` | 7 |
| `scripts/doctor.sh` | `test_doctor.py` | 39 |
| `scripts/gpu-health.sh` | `test_gpu_health.py` | 9 |
| `scripts/kill-stuck-rank.sh` | `test_kill_stuck_rank.py` | 20 |
| `scripts/download-model.sh` | `test_download_model.py` | 15 |
| `scripts/install-service.sh` | `test_install_service.py` | 12 |
| `scripts/add-mtp-head.py` | `test_add_mtp_head.py` | 17 |
| `scripts/smoke-test.py` | `test_smoke_test.py` | 18 |
| `scripts/smoke-test.py`: thinking off in the request body, reasoning-only answer | `test_smoke_thinking.py` | 6 |
| `results/make_tables.py` and `results/` measurement files | `test_tables.py` | 64 |
| `bench/kv-concurrency.py`: `run_limited`, `--engine` | `test_kv_concurrency.py` | 13 |
| `patches/vllm-rdna2-tp-0.1.0.patch` | `test_patch.py` | 10 |
| Documentation: links, anchors, generic forbidden forms, version, install, configuration and bench/PERFORMANCE statements | `test_docs.py` | 15 |
| Command-line flags of all tools | `test_flags.py` | 7 |
| Release-wide rules: syntax, no SIGKILL outside `--force`, no sudo, stdlib only, executable bits | `test_readiness.py` | 11 |

The release-wide checks in `test_readiness.py` scan the shell and Python files of the release
outside `tests/`. The shell scans and the unit-file scan skip `src/`; the Python scans include it.

## Skipped tests and why

| Test | Reason |
|---|---|
| `test_serve_launch.py::test_unowned_shm_regions_are_removed_and_held_ones_kept` | `clean_shm` works on the real `/dev/shm`. On Windows (Git Bash) that directory is shared with the host, so the test is skipped there. It runs on Linux. |
| `test_stop.py::test_group_member_named_vllm_makes_the_group_a_vllm_process` | Needs `pgrep` and process groups: procps on Linux; Git Bash has neither. It runs on Linux when `pgrep` is installed. |
| `test_stop.py::test_group_without_its_leader_or_members_is_code1_and_keeps_the_pid_file` | A group whose leader has exited needs POSIX process groups; Git Bash does not reproduce that state. It runs on Linux. |
| `test_stop.py::test_vllm_process_of_another_user_is_code1_and_keeps_the_pid_file` | Needs root on POSIX: root starts a vLLM-named process and an unprivileged account runs `stop.sh`, because `kill -0` of that process fails for that account. Skipped in every other case. |
| `test_kv_concurrency.py::test_child_receives_sigterm_and_is_not_killed_first` | Signal delivery is POSIX-only: on Windows `Popen.terminate()` is `TerminateProcess`, which a child cannot catch or ignore. It runs on Linux. |
| `test_kv_concurrency.py::test_child_that_ignores_sigterm_returns_after_about_limit_plus_five_and_survives` | Same reason as above. |
| `test_patch.py::test_patch_applies_to_the_base_commit` | Needs the upstream repository at the base commit. Set `VLLM_BASE_REPO` to a clone of vLLM that contains commit `c296851a7d173fa89d2eefbca0243be42ae9b5e0`. |
| `test_patch.py::test_check_detects_a_corrupted_patch` | Same requirement as above. |
| `test_readiness.py::test_shell_scripts_are_executable_in_git_index` | The test reads the mode of every shell script from the git index, so it runs in a git clone of the release. Outside a git repository, or when the release files are not in the index, it is skipped. |
| `test_stop_zombie.py`: `test_zombie_leader_without_group_members_stops_at_once_with_exit_0`, `test_zombie_leader_with_live_group_member_waits_to_the_limit_and_exits_1`, `test_ps_unavailable_zombie_leader_counts_as_alive_by_kill0` | A real zombie needs POSIX process groups and zombie states. Git Bash on Windows keeps no exited process as a zombie. On POSIX the tests also skip when `ps` does not report an exited child as stat `Z`. They run on Linux. |

## Interpretations

Where the tests have to choose a value, the choice is fixed as follows. A choice can be changed in
its test when the behaviour is changed on purpose.

- `--tries 0` and `--start-timeout 0` are usage errors (exit 2) for `serve.sh`.
- `--kv int4` and `--kv int8` are passed to vLLM as `int4_per_token_head` and `int8_per_token_head`;
  `--kv auto` is passed through unchanged.
- The offload option is checked through `--offload-gb` only. The name of the byte-size key in the
  connector configuration is not checked.
- The log lines of `kill-stuck-rank.sh` and the kernel-log patterns of `gpu-health.sh` are the wording
  found in those scripts. The tests check that wording, not a text fixed elsewhere.
- `smoke-test.py` exits 1 when `--timeout` expires during the chat request, and 3 only when
  `/v1/models` is unreachable.
- `serve.sh --model` accepts a local directory or a Hugging Face identifier; the served name in the
  probe is the id returned by `/v1/models`.
- A `CONVERT` preset without weights is refused with exit 2 by `serve.sh --print-command` and by
  `install-service.sh`.
- `serve.sh --print-command` prints the caller's PATH as `"$PATH"` after the venv and ROCm directories, so
  no entry of the caller's PATH appears in the output (this output goes into bug reports). The launched
  server gets the expanded PATH. `rr_run` in `scripts/lib.sh` prints the PATH of a command as `$PATH` the
  same way; the command that runs gets the real PATH.
- `install.sh --python BIN` checks the `Python.h` of that interpreter itself. `doctor.sh` checks `python3.12`
  (or `python3` when `python3.12` is absent) and not `--python`, so a system interpreter without `Python.h`
  fails the system check whatever `--python` says.
- Message texts are checked exactly only where a test quotes them. Other message texts (for example
  "the patch does not apply" and "the build failed") are checked as substrings.
- `run_limited` in `bench/kv-concurrency.py` returns `None` when its time limit expires. It sends
  SIGTERM, waits five seconds, and does not send SIGKILL. The POSIX tests check the signal; the tests
  with a substituted `Popen` check the order of the calls.
- `alive()` in `stop.sh` reads `ps -A -o pid=,pgid=,stat=`. A process whose stat starts with `Z` has exited, except a
  zombie leader whose stat contains `l` (live threads), which is alive. A live member of the leader's process group
  keeps the group alive. When `ps` does not run, `kill -0` decides, so a zombie counts as alive. The stub cases of
  `test_stop_zombie.py` set `ps` to those rows; the real-zombie cases need a POSIX host.
- The helper processes of `test_stop_zombie.py` end in one of three ways: their stop file appears, their sandbox
  directory is removed, or `LIFE_SECONDS` (120) pass. The cleanup of each test is registered before its process
  starts, so it runs when the test fails too. It creates the stop file, waits 10 s, sends SIGHUP to the process
  and its group, and reports a process that still runs as a test error. SIGKILL is never sent.
- The thinking check in `test_smoke_thinking.py` compares the body's text with the layout that `json.dumps` writes by
  default (`"chat_template_kwargs": {"enable_thinking": false}`); the parsed JSON is checked as well, so a different
  layout with the same meaning fails only the text check.

## Not covered

- Real GPUs, ROCm, the driver and kernel behaviour: nothing here loads a model or runs a kernel.
- A real `vllm serve`, real `hf download` from the network, real weight conversion.
- `systemd`: the unit file is checked as text, it is never installed or started.
- A real GPU rank stuck in the driver: `kill-stuck-rank.sh` is tested with ordinary processes
  whose command lines carry the `VLLM::Worker`, `VLLM::EngineCore` and `vllm serve` names.
- Names of people and machines. The forbidden-form check in `test_docs.py` matches only generic
  forms: home directory paths, private and carrier-grade network addresses, and paths with a drive
  letter or a macOS home prefix. It does not know any particular name of a person or a machine, so a
  release review must check for those.
- The `gpu-count` check of `doctor.sh` for a host whose only cards are gfx1031: no test fixes its
  status, because the expected status is not settled.
- The `gpu-count` branch for a card whose `runtime_status` reads `error` (the FAIL that names the cards,
  "failed to wake up", their PCI addresses and KI-38). The check reads the absolute path
  `/sys/bus/pci/drivers/amdgpu/*/power/runtime_status`, which no stub can replace, and `doctor.sh` has no
  setting that points it elsewhere. The no-card tests of `test_doctor.py` skip themselves on a host where
  such a card exists; on a host without the amdgpu driver (Windows, Git Bash) this branch is not run.
- Behaviour of the `contrib/` scripts beyond `bash -n` and the release-wide scans. Of `bench/`, only
  `run_limited` in `bench/kv-concurrency.py` is tested.
- The execute bit of `venv/bin/ninja`. `doctor.sh` and `install.sh` test it with `-x`, but in Git Bash on
  Windows `-x` is true for every regular file, so a file without the execute bit cannot be made there.
  No test covers this case; on a POSIX host a test can set mode 0644 on the venv's `ninja` and expect
  a FAIL that names `install.sh --force`.
- The two `stop.sh` refusals for a group without its leader and for another user's process run only on
  POSIX, and the second only as root. On Windows both tests are skipped, so no test checks those branches
  there.
- The kernels under `kernels/` and the C++/HIP sources under `src/` (only `src/` file copies
  against the patch are checked).
- Whether the full build of vLLM succeeds: `install.sh` is tested with stubbed `pip`, `git`
  and build commands, so no real build runs.

- A zombie with live threads (stat `Zl`) is only stubbed in `ps`: no real multi-threaded zombie is created.
- A `ps` that runs, exits 0 and prints no rows is not specified for `stop.sh` and is not tested.
- On Windows the zombie rules of `stop.sh` are covered through a stubbed `ps` only; the real-zombie tests of
  `test_stop_zombie.py` run on Linux.

## Known limits of the harness

- On Windows `shutil.which` does not find MSYS tools through `/usr/bin`, so the restricted-`PATH`
  checks in `test_doctor.py` rely on the stubs and are weaker there.
- Git Bash has no `pgrep`, so `stop.sh` finds no process group members there; the process-group test
  is skipped (see "Skipped tests and why"). Git Bash also cannot make a process group whose leader has
  exited, so the leaderless-group test is skipped there.
- Inside the sandbox `sleep` is a stub that waits 0.02 s, so the polling loops of the tools finish
  quickly. The sandbox does not change the tools' own time limits; the tests check those through the
  values the tools print or pass on.
