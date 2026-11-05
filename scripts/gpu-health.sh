#!/usr/bin/env bash
# Can the GPUs do work right now?
#
# Usage: gpu-health.sh [--help]
#   Environment: PROBE_TIMEOUT (seconds, default 60)
# Exit: 0 the probe passed on every card, work can proceed.
#       1 the driver needs a reboot: the probe hung or gave wrong numbers, or the
#         kernel log reports ring timeouts or GPU resets in the last three minutes.
#
# The probe is the verdict. It runs a matmul on every card and compares the sum
# with a CPU reference. The kernel log is a second signal and can only add a failure.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/lib.sh"

PROBE_TIMEOUT=${PROBE_TIMEOUT:-60}
PY=$RR_VENV/bin/python

case ${1:-} in
    -h|--help)
        cat <<'EOF'
Usage: gpu-health.sh [--help]

Runs a functional probe on every visible GPU and checks the kernel log.
Exit 0: work can proceed. Exit 1: the driver needs a reboot.
Environment: PROBE_TIMEOUT (seconds, default 60).
EOF
        exit 0
        ;;
    "") ;;
    *)
        rr_error "unknown argument: $1 (see --help)"
        exit 1
        ;;
esac

if [ ! -x "$PY" ]; then
    rr_error "gpu-health: no Python in the venv ($PY); cannot probe"
    exit 1
fi

ERR=$(mktemp) || { rr_error "gpu-health: mktemp failed"; exit 1; }
trap 'rm -f "$ERR"' EXIT

# The probe must check numbers, not just "not NaN". A sum of fp16 products of
# random data is near the fp16 ceiling, and a card that returns zeros or garbage
# would pass a NaN-only test.
probe() {
    timeout --signal=TERM "$PROBE_TIMEOUT" "$PY" -I - 2>"$ERR" <<'PY'
import sys
import torch

n = torch.cuda.device_count()
if n == 0:
    sys.exit("no GPU visible")
torch.manual_seed(0)
a_cpu = torch.randn(512, 512, dtype=torch.float32) * 0.05
ref = (a_cpu @ a_cpu).sum().item()
for i in range(n):
    torch.cuda.set_device(i)
    a = a_cpu.to(f"cuda:{i}", dtype=torch.float16)
    got = (a @ a).sum().item()
    if got != got or abs(got) == float("inf"):
        sys.exit(f"GPU {i}: not a finite number ({got})")
    if abs(ref) > 1e-6 and abs(got - ref) / abs(ref) > 0.05:
        sys.exit(f"GPU {i}: sum {got:.3f} differs from reference {ref:.3f}")
    del a
torch.cuda.synchronize()
print("ok")
PY
}

out=$(probe)
rc=$?
if [ "$rc" = 124 ]; then
    rr_error "GPU probe hung for ${PROBE_TIMEOUT}s: the driver is wedged"
    exit 1
fi
if [ "$out" != ok ]; then
    rr_error "GPU probe failed: ${out:-no answer}"
    tail -n 3 "$ERR" >&2
    exit 1
fi

if ! command -v journalctl >/dev/null 2>&1; then
    rr_warn "gpu-health: journalctl not found; the kernel log was not checked"
    exit 0
fi
log=$(timeout 30 journalctl -k --since "3 min ago" --no-pager 2>/dev/null)
jrc=$?
if [ "$jrc" != 0 ]; then
    rr_warn "gpu-health: kernel log unavailable (journalctl exit $jrc); it was not checked"
    exit 0
fi
hard=$(printf '%s\n' "$log" | grep -icE "ring [a-z0-9_.]+ timeout|GPU reset|amdgpu: probing|Fatal error during GPU init")
if [ "${hard:-0}" -gt 0 ]; then
    rr_error "kernel log reports ring timeouts or GPU resets ($hard line(s)): the driver needs a reboot"
    exit 1
fi
exit 0
