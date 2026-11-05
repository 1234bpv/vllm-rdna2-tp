#!/usr/bin/env bash
# Starts a vLLM server for one model preset on RDNA2 (gfx1030) cards.
#
# Usage: serve.sh PRESET [options] [-- extra vLLM serve arguments]
#        serve.sh --list | --help
#
# Without --foreground the server runs in the background. A started server must
# answer /v1/models and then one short chat request (the probe). A failed attempt
# is stopped with stop.sh (never SIGKILL), the driver is probed, and the next
# attempt starts. --foreground execs the server once, with no retries, for systemd.
#
# Exit: 0  the server answers (or --list, --print-command, --help)
#       2  bad call, unknown preset, or a preset requirement is missing
#       3  not installed (no .installed-* marker; run install.sh)
#       4  the driver probe failed (reboot needed)
#       5  no attempt produced a working server
#       6  a previous server did not stop; nothing new is started
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/scripts/lib.sh"

ARG_MODEL=
ARG_MTP=
ARG_CTX=
ARG_STREAMS=
ARG_KV=
ARG_UTIL=
ARG_HOST=
ARG_PORT=
ARG_GPUS=
ARG_TOOLS=
ARG_VISION=
ARG_OFFLOAD=
KILL_STUCK=0
TRIES=3
START_TIMEOUT=1200
FOREGROUND=0
PRINT_ONLY=0
LIST=0
PROBE=1
PRESET=
EXTRA=()

usage() {
    cat <<'EOF'
Usage: serve.sh PRESET [options] [-- extra vLLM serve arguments]
       serve.sh --list
       serve.sh --help

Starts the vLLM server for PRESET (a file in presets/). Options override the preset.

  --model PATH|HF_ID   weights: a local directory or a Hugging Face id
                       (default: the local directory of the preset, else HF_ID)
  --mtp, --no-mtp      speculative decoding with the preset's MTP head
  --ctx N              context window in tokens (--max-model-len)
  --streams N          concurrent sequences (--max-num-seqs)
  --kv int4|int8|auto  KV cache type
  --util F             GPU memory fraction, for example 0.92
  --host H             listen address (default 127.0.0.1; 0.0.0.0 has no authentication)
  --port P             listen port (default 8100)
  --gpus 0,1           GPU indices, ascending; sets HIP_VISIBLE_DEVICES and TP
  --tools, --no-tools  automatic tool choice (default on)
  --vision, --no-vision  vision inputs (default from the preset)
  --offload-gb N       KV offload to host RAM, N GiB (0 = off)
  --tries N            attempts before giving up (default 3)
  --kill-stuck         between attempts, remove a rank that survives stop.sh with
                       kill-stuck-rank.sh --force (SIGKILL, see its --help). Off by default
  --start-timeout S    seconds one attempt may take to answer (default 1200)
  --no-probe           do not send the first chat request after startup
  --foreground         exec the server once, no retries (for systemd)
  --print-command      print the environment and the command; start nothing
  --list               list presets
  --help               show this text

Exit: 0 running (or listing/printing/help); 2 bad call; 3 not installed;
4 driver probe failed; 5 no attempt worked; 6 previous server did not stop.
EOF
}

need_value() {
    if [ $# -lt 2 ]; then
        rr_error "$1 needs a value (see --help)"
        exit 2
    fi
}

bad_value() {
    rr_error "$1: $2 (see --help)"
    exit 2
}

# gpus_ascending LIST: true when the comma-separated indices strictly increase.
gpus_ascending() {
    local -a parts
    local prev=-1 g
    IFS=',' read -r -a parts <<< "$1"
    for g in "${parts[@]}"; do
        [ "$g" -gt "$prev" ] || return 1
        prev=$g
    done
}

while [ $# -gt 0 ]; do
    case $1 in
        --model)
            need_value "$@"
            ARG_MODEL=$2
            shift 2
            ;;
        --mtp) ARG_MTP=1; shift ;;
        --no-mtp) ARG_MTP=0; shift ;;
        --ctx)
            need_value "$@"
            rr_is_int "$2" || bad_value --ctx "must be a positive integer"
            ARG_CTX=$2
            shift 2
            ;;
        --streams)
            need_value "$@"
            rr_is_int "$2" || bad_value --streams "must be a positive integer"
            ARG_STREAMS=$2
            shift 2
            ;;
        --kv)
            need_value "$@"
            case $2 in
                int4|int8|auto) ;;
                *) bad_value --kv "must be int4, int8 or auto" ;;
            esac
            ARG_KV=$2
            shift 2
            ;;
        --util)
            need_value "$@"
            rr_is_util "$2" || bad_value --util "must be a fraction such as 0.92"
            ARG_UTIL=$2
            shift 2
            ;;
        --host)
            need_value "$@"
            ARG_HOST=$2
            shift 2
            ;;
        --port)
            need_value "$@"
            [[ $2 =~ ^[0-9]+$ ]] && [ "$2" -ge 1 ] && [ "$2" -le 65535 ] \
                || bad_value --port "must be 1-65535"
            ARG_PORT=$2
            shift 2
            ;;
        --gpus)
            need_value "$@"
            [[ $2 =~ ^[0-9]+(,[0-9]+)*$ ]] || bad_value --gpus "must be a list such as 0,1"
            gpus_ascending "$2" || bad_value --gpus "indices must be ascending, for example 0,1: a reversed order (1,0) hangs the first decode step"
            ARG_GPUS=$2
            shift 2
            ;;
        --tools) ARG_TOOLS=1; shift ;;
        --no-tools) ARG_TOOLS=0; shift ;;
        --vision) ARG_VISION=1; shift ;;
        --no-vision) ARG_VISION=0; shift ;;
        --offload-gb)
            need_value "$@"
            [[ $2 =~ ^[0-9]+$ ]] || bad_value --offload-gb "must be a non-negative integer"
            ARG_OFFLOAD=$2
            shift 2
            ;;
        --kill-stuck)
            KILL_STUCK=1
            shift
            ;;
        --tries)
            need_value "$@"
            rr_is_int "$2" || bad_value --tries "must be a positive integer"
            TRIES=$2
            shift 2
            ;;
        --start-timeout)
            need_value "$@"
            rr_is_int "$2" || bad_value --start-timeout "must be a positive number of seconds"
            START_TIMEOUT=$2
            shift 2
            ;;
        --no-probe) PROBE=0; shift ;;
        --foreground) FOREGROUND=1; shift ;;
        --print-command) PRINT_ONLY=1; shift ;;
        --list) LIST=1; shift ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA=("$@")
            break
            ;;
        -*)
            rr_error "unknown option: $1 (see --help)"
            exit 2
            ;;
        *)
            if [ -n "$PRESET" ]; then
                rr_error "unexpected argument: $1 (see --help)"
                exit 2
            fi
            PRESET=$1
            shift
            ;;
    esac
done

list_presets() {
    local f name dir state status=0
    printf '%-18s %-44s %7s %-22s %-3s %-10s %s\n' PRESET HF_ID CTX KV MTP CONVERT LOCAL
    for f in "$RR_RELEASE_DIR"/presets/*.env; do
        name=$(basename "$f" .env)
        if ! rr_preset_load "$name" >/dev/null; then
            status=2
            continue
        fi
        dir=$RR_MODELS/$name
        [ -n "$P_CONVERT" ] && dir=$RR_MODELS/$name-ct-int4
        state=missing
        [ -d "$dir" ] && state=present
        printf '%-18s %-44s %7s %-22s %-3s %-10s %s\n' \
            "$name" "$P_HF_ID" "$P_CTX" "$P_KV" "$P_MTP" "${P_CONVERT:--}" "$state"
    done
    return $status
}

if [ "$LIST" = 1 ]; then
    list_presets
    exit $?
fi
if [ -z "$PRESET" ]; then
    rr_error "PRESET is required (see --list and --help)"
    exit 2
fi
rr_preset_load "$PRESET" || exit 2
PY_JSON=$(rr_python) || PY_JSON=

# Effective settings: command-line options override the preset.
GPUS=${ARG_GPUS:-0,1}
GPU_SEPARATORS=${GPUS//[^,]/}
TP=$(( ${#GPU_SEPARATORS} + 1 ))
CTX=${ARG_CTX:-$P_CTX}
MAXSEQ=${ARG_STREAMS:-$P_MAXSEQ}
case ${ARG_KV:-} in
    int4) KV=int4_per_token_head ;;
    int8) KV=int8_per_token_head ;;
    auto) KV=auto ;;
    *) KV=$P_KV ;;
esac
UTIL=${ARG_UTIL:-$P_UTIL}
HOST=${ARG_HOST:-127.0.0.1}
PORT=${ARG_PORT:-8100}
MTP=${ARG_MTP:-$P_MTP}
TOOLS=${ARG_TOOLS:-1}
VISION=${ARG_VISION:-$P_VISION_DEFAULT}
OFFLOAD=${ARG_OFFLOAD:-0}

if [ "$MTP" = 1 ] && [ -z "$P_MTP_METHOD" ]; then
    rr_error "preset $PRESET has no MTP (MTP_METHOD is empty); drop --mtp"
    exit 2
fi

# REASONEND in the environment, even if empty, overrides the preset. Otherwise the
# preset's REASON_END applies, and the shared default when the preset has none.
if [ "${REASONEND+x}" = x ]; then
    REASON_END=$REASONEND
elif [ "$P_REASON_END_SET" = 1 ]; then
    REASON_END=$P_REASON_END
else
    REASON_END=$REASON_END_DEFAULT
fi
case $REASON_END in
    *'</think>')
        if [ "$P_REASON_GRACE" != 0 ]; then
            rr_warn "REASON_END ends with '</think>', so REASON_GRACE=$P_REASON_GRACE has no effect; remove '</think>' from the phrase or set REASON_GRACE=0 in the preset"
        fi
        ;;
esac

if [ "$PRINT_ONLY" = 0 ]; then
    if ! marker=$(rr_installed_marker); then
        rr_error "not installed: no $RR_HOME/.installed-* marker; run install.sh first"
        exit 3
    fi
fi

# Model: an explicit --model, else the preset's local directory, else HF_ID.
LOCAL_DIR=$RR_MODELS/$PRESET
[ -n "$P_CONVERT" ] && LOCAL_DIR=$RR_MODELS/$PRESET-ct-int4
if [ -n "$ARG_MODEL" ]; then
    MODEL=$ARG_MODEL
elif [ -d "$LOCAL_DIR" ]; then
    MODEL=$LOCAL_DIR
elif [ -n "$P_CONVERT" ]; then
    rr_error "no converted weights at $LOCAL_DIR; run: $RR_RELEASE_DIR/scripts/download-model.sh $PRESET"
    exit 2
else
    MODEL=$P_HF_ID
fi
MODEL_IS_DIR=0
if [ -d "$MODEL" ]; then
    MODEL_IS_DIR=1
    MODEL=$(cd -- "$MODEL" && pwd)
fi

if [ "$MTP" = 1 ] && [ "$P_MTP_NEEDS_HEAD" = 1 ]; then
    if [ "$MODEL_IS_DIR" = 0 ]; then
        case $MODEL in
            /*|.*|~*)
                rr_error "preset $PRESET needs its MTP head, and $MODEL is not an existing directory"
                ;;
            *)
                rr_error "preset $PRESET needs its MTP head, and $MODEL is a Hugging Face id; first run: $RR_RELEASE_DIR/scripts/download-model.sh $PRESET"
                ;;
        esac
        exit 2
    fi
    if [ ! -e "$MODEL/model-mtp.safetensors" ] && [ ! -e "$MODEL/model_mtp.safetensors" ]; then
        rr_error "$MODEL has no MTP head (model-mtp.safetensors); run: $RR_RELEASE_DIR/scripts/add-mtp-head.py $MODEL"
        exit 2
    fi
fi

# Command pieces, built once and used by --print-command and by the launch.
ENV_PAIRS=()
SERVE_ARGS=()
build_command() {
    local -a preset_env=()
    local rcfg
    if [ -n "$REASON_END" ] && [ -z "$PY_JSON" ]; then
        rr_error "no Python interpreter (python3 or python) found, needed for --reasoning-config"
        exit 2
    fi
    ENV_PAIRS=(
        # The .hip kernels are built at first use: that needs ninja and hipcc on PATH.
        "PATH=$RR_VENV/bin:$RR_ROCM_PATH/bin:$PATH"
        "HIP_VISIBLE_DEVICES=$GPUS"
        "ROCM_PATH=$RR_ROCM_PATH"
        "PYTORCH_ROCM_ARCH=gfx1030"
        "SAFETENSORS_FAST_GPU=1"
        "OMP_NUM_THREADS=4"
    )
    read -r -a preset_env <<< "$P_PRESET_ENV"
    ENV_PAIRS+=("${preset_env[@]}")
    if [ "$MTP" = 1 ]; then
        ENV_PAIRS+=("VLLM_ROCM_INT4_DECODE_QLEN_MAX=3")
    else
        ENV_PAIRS+=("VLLM_ROCM_INT4_DECODE_QLEN_MAX=1")
    fi
    ENV_PAIRS+=("VLLM_REASONEND_GRACE=$P_REASON_GRACE")

    SERVE_ARGS=("$RR_VENV/bin/vllm" serve "$MODEL"
        --served-model-name "$P_SERVED_NAME" --disable-custom-all-reduce --dtype float16
        --max-model-len "$CTX" --kv-cache-dtype "$KV" --gpu-memory-utilization "$UTIL"
        --max-num-seqs "$MAXSEQ" --max-num-batched-tokens "$P_BATCHTOK"
        --long-prefill-token-threshold "$P_LONGPREFILL" --attention-backend TRITON_ATTN
        --host "$HOST" --port "$PORT" --shutdown-timeout 60)

    if [ "$TOOLS" = 1 ]; then
        SERVE_ARGS+=(--enable-auto-tool-choice --tool-call-parser "$P_TOOL_PARSER")
    fi
    if [ -n "$P_REASONING_PARSER" ]; then
        SERVE_ARGS+=(--reasoning-parser "$P_REASONING_PARSER")
    fi
    if [ -n "$REASON_END" ]; then
        rcfg=$(REASONEND="$REASON_END" "$PY_JSON" -I -c 'import json, os; print(json.dumps({"reasoning_end_str": os.environ["REASONEND"]}))') \
            || { rr_error "cannot build --reasoning-config: $PY_JSON failed"; exit 2; }
        SERVE_ARGS+=(--reasoning-config "$rcfg")
    fi

    # Vision: on with a processor limit only for presets that take max_pixels;
    # off means no image or video inputs at all.
    if [ "$VISION" = 1 ]; then
        if [ "$P_VISION_KWARGS" = 1 ]; then
            SERVE_ARGS+=(--mm-processor-kwargs '{"max_pixels":1048576}')
        fi
    else
        SERVE_ARGS+=(--limit-mm-per-prompt '{"image":0,"video":0}')
    fi

    if [ "$MTP" = 1 ]; then
        SERVE_ARGS+=(
            --speculative-config "{\"method\":\"$P_MTP_METHOD\",\"num_speculative_tokens\":2}"
            --compilation-config "{\"cudagraph_capture_sizes\":$P_MTP_CAPTURE_SIZES}"
        )
    fi
    if [ "$OFFLOAD" != 0 ]; then
        SERVE_ARGS+=(--kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$((OFFLOAD * 1073741824))}}")
    fi

    # Extra arguments go before the parallel sizes: argparse keeps the last value.
    SERVE_ARGS+=("${EXTRA[@]}")
    SERVE_ARGS+=(--pipeline-parallel-size 1 --tensor-parallel-size "$TP")
}

print_command() {
    local tok q out="" idx=0
    for tok in "${ENV_PAIRS[@]}"; do
        # The caller's PATH stays a reference: this output goes into bug reports.
        if [[ $tok == PATH=* ]]; then
            printf 'PATH=%s:"$PATH" \\\n' "$(rr_quote_one "$RR_VENV/bin:$RR_ROCM_PATH/bin")"
            continue
        fi
        printf '%s \\\n' "$(rr_quote_one "$tok")"
    done
    for tok in "${SERVE_ARGS[@]}"; do
        q=$(rr_quote_one "$tok")
        if [ "$idx" = 0 ]; then
            out=$q
        elif [[ $tok == --* ]]; then
            out+=" \\"$'\n'"  $q"
        else
            out+=" $q"
        fi
        idx=$((idx + 1))
    done
    printf '%s\n' "$out"
}

build_command

if [ "$PRINT_ONLY" = 1 ]; then
    print_command
    exit 0
fi

server_url() {
    local h=$HOST
    [ "$h" = 0.0.0.0 ] && h=127.0.0.1
    printf 'http://%s:%s' "$h" "$PORT"
}

fatal_in_log() {
    grep -a -q -E 'illegal memory access|hipErrorIllegalAddress|EngineCore encountered a fatal error' \
        "$LOG_FILE" 2>/dev/null
}

# clean_shm: remove vLLM offload and all-reduce regions in /dev/shm that no live
# process maps. Left behind by a killed server; the next start fails without them.
clean_shm() {
    local f base n=0
    if ! command -v lsof >/dev/null 2>&1; then
        rr_warn "lsof not found: /dev/shm cleanup checks /proc maps only"
    fi
    for f in /dev/shm/vllm_offload_*.mmap /dev/shm/vllm_host_ar_*; do
        [ -e "$f" ] || continue
        base=$(basename "$f")
        if grep -ql -- "$base" /proc/[0-9]*/maps 2>/dev/null; then
            continue
        fi
        if command -v lsof >/dev/null 2>&1 && [ -n "$(timeout 30 lsof -t -- "$f" 2>/dev/null)" ]; then
            continue
        fi
        if rm -f -- "$f"; then
            n=$((n + 1))
        else
            rr_warn "cannot remove $f"
        fi
    done
    [ "$n" -gt 0 ] && echo "removed $n orphaned /dev/shm region(s)"
    return 0
}

start_attempt() {
    local wrap=()
    LOG_FILE=$RR_LOGS/serve-$PRESET-$(date +%Y%m%d-%H%M%S).log
    if ! mkdir -p "$RR_LOGS" "$RR_RUN"; then
        rr_error "cannot create $RR_LOGS or $RR_RUN"
        exit 5
    fi
    command -v setsid >/dev/null 2>&1 && wrap+=(setsid)
    wrap+=(nohup)
    env "${ENV_PAIRS[@]}" "${wrap[@]}" "${SERVE_ARGS[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
    LAUNCH_PID=$!
    printf '%s\n' "$LAUNCH_PID" > "$RR_RUN/vllm.pid"
}

# wait_serving: poll /v1/models until it answers, the process dies, the log shows a
# fatal error, or START_TIMEOUT passes.
wait_serving() {
    local url deadline
    url=$(server_url)
    deadline=$(( $(date +%s) + START_TIMEOUT ))
    while :; do
        if timeout 10 curl -sf -m 5 "$url/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        if fatal_in_log; then
            FAIL_REASON="fatal error in the server log"
            return 1
        fi
        if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
            FAIL_REASON="the server process exited before it answered"
            return 1
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            FAIL_REASON="no answer after ${START_TIMEOUT}s"
            return 1
        fi
        sleep 5
    done
}

# probe_serving: one short chat request. Long prefill and first compilation are
# silent, so the limit is 300 s with no retry inside the attempt.
probe_serving() {
    local url model body out rc t0 t1
    url=$(server_url)
    model=$(timeout 10 curl -sf -m 5 "$url/v1/models" \
        | "$PY_JSON" -I -c 'import json, sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
    if [ -z "$model" ]; then
        PROBE_FAIL="model name could not be read from /v1/models"
        return 1
    fi
    body=$(MODEL_NAME="$model" "$PY_JSON" -I -c 'import json, os; print(json.dumps({"model": os.environ["MODEL_NAME"], "messages": [{"role": "user", "content": "Reply with one word."}], "max_tokens": 8, "temperature": 0}))') \
        || { PROBE_FAIL="cannot build the probe request"; return 1; }
    t0=$(date +%s%N)
    out=$(timeout 330 curl -sf -m 300 -H 'Content-Type: application/json' -d "$body" \
        "$url/v1/chat/completions")
    rc=$?
    t1=$(date +%s%N)
    if [ "$rc" != 0 ]; then
        PROBE_FAIL="request failed (curl exit $rc)"
        return 1
    fi
    if ! printf '%s' "$out" | "$PY_JSON" -I -c 'import json, sys; sys.exit(0 if json.load(sys.stdin).get("choices") else 1)' 2>/dev/null; then
        PROBE_FAIL="the response has no choices"
        return 1
    fi
    PROBE_MS=$(( (t1 - t0) / 1000000 ))
    return 0
}

probe_and_check() {
    if ! probe_serving; then
        FAIL_REASON="probe request: $PROBE_FAIL"
        return 1
    fi
    if fatal_in_log; then
        FAIL_REASON="fatal error in the server log after the probe"
        return 1
    fi
    if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
        FAIL_REASON="the server process exited after the probe"
        return 1
    fi
    return 0
}

# force_stuck_ranks: --kill-stuck, for the ranks that stop.sh could not stop.
force_stuck_ranks() {
    local pid
    for pid in $(pgrep -u "$(id -u)" -f '^VLLM::Worker'); do
        rr_warn "rank $pid did not stop; running kill-stuck-rank.sh --force"
        timeout 800 "$RR_RELEASE_DIR/scripts/kill-stuck-rank.sh" --force "$pid"
    done
}

# recover_after_failure N: stop what the failed attempt left, then probe the driver.
recover_after_failure() {
    local rc
    rr_error "attempt $1 failed: $FAIL_REASON (log: $LOG_FILE)"
    timeout 150 "$RR_RELEASE_DIR/stop.sh" --timeout 60
    rc=$?
    if [ "$rc" != 0 ] && [ "$KILL_STUCK" = 1 ]; then
        force_stuck_ranks
        timeout 150 "$RR_RELEASE_DIR/stop.sh" --timeout 60
        rc=$?
    fi
    if [ "$rc" != 0 ]; then
        rr_error "the previous server did not stop; no new attempt is started"
        exit 6
    fi
    sleep 10
    if ! timeout 300 "$RR_RELEASE_DIR/scripts/gpu-health.sh"; then
        rr_error "driver probe failed after attempt $1; the driver needs a reboot"
        exit 4
    fi
}

report_success() {
    local url root kvline
    url=$(server_url)
    root=$(timeout 10 curl -sf -m 5 "$url/v1/models" \
        | "$PY_JSON" -I -c 'import json, sys; print(json.load(sys.stdin)["data"][0]["root"])' 2>/dev/null) \
        || root="(unknown: /v1/models did not parse)"
    kvline=$(grep -a -m 1 "GPU KV cache size" "$LOG_FILE" 2>/dev/null)
    echo "serving at $url (PID $LAUNCH_PID)"
    echo "model root: $root"
    echo "${kvline:-GPU KV cache size: not reported in the log}"
    if [ "$PROBE" = 1 ]; then
        echo "probe answered in $(awk -v ms="$PROBE_MS" 'BEGIN { printf "%.1f", ms / 1000 }') s"
    fi
    echo "log: $LOG_FILE"
}

print_summary() {
    echo "preset:    $PRESET (served as $P_SERVED_NAME)"
    echo "model:     $MODEL"
    echo "GPUs:      $GPUS (tensor parallel $TP)"
    echo "window:    $CTX tokens, $MAXSEQ streams, KV $KV, utilization $UTIL"
    if [ "$MTP" = 1 ]; then
        echo "MTP:       on ($P_MTP_METHOD, 2 draft tokens)"
    else
        echo "MTP:       off"
    fi
    if [ "$VISION" = 1 ]; then echo "vision:    on"; else echo "vision:    off"; fi
    echo "endpoint:  $(server_url)"
}

if [ "$HOST" = 0.0.0.0 ]; then
    rr_warn "listening on 0.0.0.0: the server has no authentication"
fi

if [ "$FOREGROUND" = 1 ]; then
    # systemd runs this process: no probe, no retries. The PID file lets stop.sh
    # find the server, and exec keeps the PID.
    clean_shm
    mkdir -p "$RR_RUN" || { rr_error "cannot create $RR_RUN"; exit 5; }
    printf '%s\n' "$$" > "$RR_RUN/vllm.pid"
    exec env "${ENV_PAIRS[@]}" "${SERVE_ARGS[@]}"
fi

print_summary

if ! timeout 300 "$RR_RELEASE_DIR/scripts/gpu-health.sh"; then
    rr_error "driver probe failed before start; the driver needs a reboot"
    exit 4
fi

if [ -e "$RR_RUN/vllm.pid" ]; then
    if ! timeout 150 "$RR_RELEASE_DIR/stop.sh" --timeout 60; then
        rr_error "a previous server did not stop; nothing new is started"
        exit 6
    fi
fi
clean_shm

attempt=1
while [ "$attempt" -le "$TRIES" ]; do
    echo "attempt $attempt of $TRIES at $(date +%H:%M:%S)"
    start_attempt
    if wait_serving && { [ "$PROBE" = 0 ] || probe_and_check; }; then
        report_success
        exit 0
    fi
    recover_after_failure "$attempt"
    attempt=$((attempt + 1))
done

rr_error "no attempt produced a working server after $TRIES tries"
exit 5
