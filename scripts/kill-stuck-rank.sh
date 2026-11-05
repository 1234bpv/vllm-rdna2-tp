#!/usr/bin/env bash
# Sends SIGTERM to one stuck vLLM process (a GPU rank, the engine core or the API
# server) after the GPU driver has been confirmed healthy. Last resort after a
# crash of another rank, for when stop.sh reports survivors.
#
# Usage: kill-stuck-rank.sh [--force] PID | --help
#
# Gates, all required before any signal is sent:
#   - PID is a vLLM process: a rank (VLLM::Worker*), the engine core
#     (VLLM::EngineCore*) or the API server (bin/vllm serve);
#   - for an engine process, no live rank exists;
#   - scripts/gpu-health.sh exits 0.
# The kernel log's page-fault count since the process started is recorded for
# diagnosis. It does not block the signal.
#
# Without --force, SIGKILL is never sent: a process that does not exit within 30 s
# after SIGTERM is reported and left for a person, with a reboot as the final step.
#
# With --force, a rank that survives SIGTERM gets SIGKILL, because after a crash of
# the other rank it cannot leave its call into the GPU runtime and only a reboot
# would remove it otherwise. SIGKILL on a process with unfinished GPU page faults can
# freeze the machine (KI-2), so --force sends it only if all of these hold:
#   - the process is a rank;
#   - the kernel log can be read;
#   - since the rank started, that log has no page fault from this PID or on its card.
# A failed condition leaves the process alive, with exit code 1.
#
# Exit: 0 the process is gone and the GPU is healthy; 1 the GPU check failed before
#       the signal (nothing was signalled), or the process is still alive, or the GPU
#       check failed afterwards; 2 bad call, or the PID is not a vLLM process or has
#       live ranks (nothing was signalled).
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/lib.sh"

LOG=$RR_LOGS/kill-stuck-rank.log

usage() {
    cat <<'EOF'
Usage: kill-stuck-rank.sh [--force] PID
       kill-stuck-rank.sh --help

Sends SIGTERM to one stuck vLLM process after gates pass: the process is a vLLM
rank, engine core or API server; no live rank exists for an engine process; and
gpu-health.sh exits 0. Sends no SIGKILL by default.

  --force   a rank that survives SIGTERM for 30 s gets SIGKILL, if the kernel log
            can be read and has no page fault from this PID or on the card of the
            rank since it started. SIGKILL can freeze a machine with unfinished
            GPU page faults; these checks reduce that risk and do not remove it.

Exit: 0 gone and GPU healthy; 1 GPU check failed, or still alive or GPU unhealthy
after the signal; 2 bad call or not a vLLM process (nothing signalled).
EOF
}

say() {
    local line
    line="$(date -Iseconds) pid=${PID:-?} $*"
    printf '%s\n' "$line"
    mkdir -p "$RR_LOGS" 2>/dev/null && printf '%s\n' "$line" >> "$LOG" 2>/dev/null \
        || rr_warn "cannot write $LOG"
}
refuse() {
    say "REFUSED: $2"
    exit "$1"
}

FORCE=0
case ${1:-} in
    --force)
        FORCE=1
        shift
        ;;
esac
case ${1:-} in
    -h|--help)
        usage
        exit 0
        ;;
    "")
        rr_error "PID is required (see --help)"
        exit 2
        ;;
esac
if [ $# -gt 1 ]; then
    rr_error "unexpected argument: $2 (see --help)"
    exit 2
fi
PID=$1
if ! [[ $PID =~ ^[0-9]+$ ]]; then
    rr_error "PID must be a number: $PID"
    exit 2
fi
[ -d "/proc/$PID" ] || refuse 2 "no such process"

name=$(cat "/proc/$PID/comm" 2>/dev/null)
cmd=$(tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null)
ppid=$(awk '{ print $4 }' "/proc/$PID/stat" 2>/dev/null)

mode=rank
if [[ $cmd == VLLM::Worker* || $name == VLLM::Worker* ]]; then
    mode=rank
elif [[ $cmd == VLLM::EngineCore* || $name == VLLM::EngineCore* || $cmd == *"bin/vllm serve"* ]]; then
    mode=engine
    workers=""
    for q in /proc/[0-9]*; do
        if tr '\0' ' ' < "$q/cmdline" 2>/dev/null | grep -q '^VLLM::Worker'; then
            workers="$workers ${q#/proc/}"
        fi
    done
    [ -z "$workers" ] || refuse 2 "live rank processes exist (PIDs:$workers); stop them first"
else
    refuse 2 "not a vLLM process: '${cmd:-$name}'"
fi

since=$(date -d "$(LC_ALL=C ps -o lstart= -p "$PID" 2>/dev/null)" '+%Y-%m-%d %H:%M:%S' 2>/dev/null)
faults="unavailable"
if [ -n "$since" ] && command -v journalctl >/dev/null 2>&1; then
    faults=$(timeout 30 journalctl -k --since "$since" --no-pager 2>/dev/null | grep -ci "page fault")
fi
say "mode=$mode name='${name:-?}' ppid=$ppid started='${since:-?}' kernel page faults since start: $faults"

if ! timeout 300 "$RR_RELEASE_DIR/scripts/gpu-health.sh" >/dev/null 2>&1; then
    refuse 1 "gpu-health.sh did not exit 0; the driver needs a reboot"
fi

# rank_card: PCI address of the card where the rank holds the most VRAM, from the KFD entries.
rank_card() {
    local root=${RR_KFD_ROOT:-/sys/class/kfd/kfd} gpu_id node props loc dom
    gpu_id=$(for f in "$root/proc/$PID"/vram_*; do
        printf '%s %s\n' "$(cat "$f" 2>/dev/null)" "${f##*vram_}"
    done | sort -n | tail -1 | awk '{ print $2 }')
    [[ $gpu_id =~ ^[0-9]+$ ]] || return 1
    node=$(grep -lx "$gpu_id" "$root"/topology/nodes/*/gpu_id 2>/dev/null | head -1)
    [ -n "$node" ] || return 1
    props=${node%/gpu_id}/properties
    loc=$(awk '$1 == "location_id" { print $2 }' "$props" 2>/dev/null)
    dom=$(awk '$1 == "domain" { print $2 }' "$props" 2>/dev/null)
    [[ $loc =~ ^[0-9]+$ && $dom =~ ^[0-9]+$ ]] || return 1
    printf '%04x:%02x:%02x.%x\n' "$dom" $((loc >> 8)) $(((loc >> 3) & 31)) $((loc & 7))
}

# force_refusal: prints why SIGKILL must not be sent. Empty output means it may be sent.
force_refusal() {
    local card log
    if [ "$mode" != rank ]; then
        echo "--force applies to ranks only"
        return
    fi
    if [ -z "$since" ] || ! command -v journalctl >/dev/null 2>&1; then
        echo "the start time of the process or journalctl is not available"
        return
    fi
    # Without rights journalctl prints nothing and exits 0, which would read as "no faults".
    if [ -z "$(timeout 30 journalctl -k -q -n 1 --no-pager 2>/dev/null)" ]; then
        echo "the kernel log cannot be read"
        return
    fi
    card=$(rank_card) || card=""
    if [ -z "$card" ]; then
        echo "the card of this rank cannot be determined"
        return
    fi
    log=$(timeout 30 journalctl -k --since "$since" --no-pager 2>/dev/null | grep -i "page fault")
    if grep -qi "$card" <<< "$log"; then
        echo "the kernel log has a page fault on the card of this rank ($card)"
    elif grep -q "pid $PID[ )]" <<< "$log"; then
        echo "the kernel log has a page fault from this process"
    fi
}

if [ "$FORCE" = 1 ]; then
    say "sending SIGTERM (SIGKILL follows only if it has no effect and the checks pass)"
else
    say "sending SIGTERM (no SIGKILL is used)"
fi
kill -TERM "$PID" 2>/dev/null
for _ in $(seq 1 30); do
    [ -d "/proc/$PID" ] || break
    sleep 1
done

rc=0
if [ -d "/proc/$PID" ] && [ "$FORCE" = 1 ]; then
    refusal=$(force_refusal)
    if [ -n "$refusal" ]; then
        say "--force refused: $refusal; not killed"
    else
        say "still alive 30 s after SIGTERM; sending SIGKILL (--force)"
        kill -KILL "$PID" 2>/dev/null
        for _ in $(seq 1 60); do
            [ -d "/proc/$PID" ] || break
            sleep 1
        done
    fi
fi
if [ -d "/proc/$PID" ]; then
    say "still alive 30 s after SIGTERM (state: $(awk '{ print $3 }' "/proc/$PID/stat" 2>/dev/null)); not killed"
    rc=1
elif [ "$FORCE" = 1 ]; then
    say "process is gone"
else
    say "exited after SIGTERM"
fi

if timeout 300 "$RR_RELEASE_DIR/scripts/gpu-health.sh" >/dev/null 2>&1; then
    say "gpu-health.sh: 0"
else
    say "gpu-health.sh: not 0 after the signal; the driver needs a reboot"
    rc=1
fi
exit $rc
