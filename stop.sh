#!/usr/bin/env bash
# Stops the vLLM server recorded in run/vllm.pid, with SIGTERM only.
#
# Usage: stop.sh [--timeout S] [--help]
#   --timeout S  seconds to wait after SIGTERM (default 60)
#
# Exit: 0 stopped, or nothing was running; 1 processes still alive after the
#       timeout; 2 bad usage.
#
# SIGKILL is never sent. A process that still holds unfinished GPU address faults
# can hang the amdgpu driver when it is killed hard (kfd_process_wq_release), so a
# survivor is reported, not killed.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/scripts/lib.sh"

TIMEOUT=60

usage() {
    cat <<'EOF'
Usage: stop.sh [--timeout S] [--help]

Stops the vLLM server whose PID is in run/vllm.pid: SIGTERM to the process and
to its process group, then waits.

  --timeout S  seconds to wait after SIGTERM (default 60)
  --help       show this text

Exit: 0 stopped or nothing was running; 1 processes still alive; 2 bad usage.
SIGKILL is never sent.
EOF
}

while [ $# -gt 0 ]; do
    case $1 in
        --timeout)
            if [ $# -lt 2 ] || ! rr_is_int "$2"; then
                rr_error "--timeout needs a positive number of seconds"
                exit 2
            fi
            TIMEOUT=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            rr_error "unknown argument: $1 (see --help)"
            exit 2
            ;;
    esac
done

PIDFILE=$RR_RUN/vllm.pid
if [ ! -e "$PIDFILE" ]; then
    echo "nothing to stop: $PIDFILE does not exist"
    exit 0
fi
PID=$(tr -d '[:space:]' < "$PIDFILE" 2>/dev/null)
if ! [[ $PID =~ ^[0-9]+$ ]]; then
    rr_error "cannot read a PID from $PIDFILE (content: '${PID:-empty}')"
    exit 1
fi

# A process or its group is alive if either still exists. A zombie has exited: kill -0
# succeeds on it until its parent collects it, and in a container without an init nobody
# does. A zombie leader with live threads ("Zl") has not exited. Without a usable ps the
# answer of kill -0 stands.
alive() {
    local rows
    kill -0 "$PID" 2>/dev/null || kill -0 -- "-$PID" 2>/dev/null || return 1
    rows=$(timeout 5 ps -A -o pid=,pgid=,stat= 2>/dev/null) || return 0
    printf '%s\n' "$rows" | awk -v id="$PID" '
        ($1 == id || $2 == id) && ($3 !~ /^Z/ || $3 ~ /l/) { found = 1 }
        END { exit !found }'
}

# A PID file can outlive its server: after a reboot the number may belong to an
# unrelated process, which must get no signal. Ranks rename themselves VLLM::*.
is_vllm() {
    local cmd
    cmd=$(tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null; timeout 5 pgrep -a -g "$PID" 2>/dev/null)
    case $cmd in
        *bin/vllm*|*VLLM::*) return 0 ;;
    esac
    return 1
}

if ! alive; then
    # kill -0 fails for a process of another user too: serve.sh must not start a second server.
    if [ -e "/proc/$PID" ] && is_vllm; then
        rr_error "PID $PID is a vLLM process of another user: it cannot be stopped from this account"
        exit 1
    fi
    rm -f -- "$PIDFILE"
    echo "nothing to stop: PID $PID is not running (stale $PIDFILE removed)"
    exit 0
fi

if ! is_vllm; then
    if [ ! -e "/proc/$PID" ] && [ -z "$(timeout 5 pgrep -g "$PID" 2>/dev/null)" ]; then
        rr_error "process group $PID is alive, but its members cannot be read (no pgrep, or /proc is restricted)"
        exit 1
    fi
    rm -f -- "$PIDFILE"
    echo "nothing to stop: PID $PID is not a vLLM process and got no signal (stale $PIDFILE removed)"
    exit 0
fi

echo "stopping PID $PID and its process group with SIGTERM (wait up to ${TIMEOUT}s)"
kill -TERM "$PID" 2>/dev/null
kill -TERM -- "-$PID" 2>/dev/null

deadline=$(( $(date +%s) + TIMEOUT ))
while alive && [ "$(date +%s)" -lt "$deadline" ]; do
    sleep 1
done

if alive; then
    rr_error "PID $PID is still running after ${TIMEOUT}s; SIGKILL is not sent"
    echo "see docs/TROUBLESHOOTING.md; for a stuck GPU rank: scripts/kill-stuck-rank.sh $PID" >&2
    exit 1
fi

rm -f -- "$PIDFILE"
echo "stopped PID $PID"
exit 0
