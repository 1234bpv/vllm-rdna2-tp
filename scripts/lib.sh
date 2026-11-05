#!/usr/bin/env bash
# Shared helpers for the release scripts. Sourced by the other scripts, never run.
#
# Holds the release root and install paths, message helpers, shell quoting for
# printed commands, the preset loader, and the time-limited command runner.
# Preset values are data: they are read line by line and passed as arguments,
# never evaluated.

RR_LIB_FILE=$(readlink -f "${BASH_SOURCE[0]}")
RR_RELEASE_DIR=$(dirname "$(dirname "$RR_LIB_FILE")")
RR_VERSION=$(tr -d '[:space:]' < "$RR_RELEASE_DIR/VERSION" 2>/dev/null)
RR_ROCM_PATH=${ROCM_PATH:-/opt/rocm}
RR_DRY_RUN=${RR_DRY_RUN:-0}
RR_LOG_FILE=${RR_LOG_FILE:-}

# Default reasoning-end phrase.
# It has an apostrophe, so it is never written into a preset file or a shell
# string; it reaches vLLM only through the environment (see serve.sh).
REASON_END_DEFAULT="Okay, I've spent enough time on this. Let me wrap up my reasoning and give the answer."

# Install paths. rr_set_home recomputes them when --prefix changes the home.
rr_set_home() {
    RR_HOME=$1
    RR_VENV=$RR_HOME/venv
    RR_SRC=$RR_HOME/src/vllm
    RR_MODELS=$RR_HOME/models
    RR_LOGS=$RR_HOME/logs
    RR_RUN=$RR_HOME/run
}
rr_set_home "${VLLM_RDNA2_HOME:-${HOME:-.}/.local/share/vllm-rdna2}"

# Messages go to stdout or stderr and, when RR_LOG_FILE is set, to the log.
rr_logline() {
    [ -n "$RR_LOG_FILE" ] || return 0
    printf '%s\n' "$1" >> "$RR_LOG_FILE" 2>/dev/null \
        || printf 'WARN: cannot write log %s\n' "$RR_LOG_FILE" >&2
    return 0
}
rr_say()   { printf '%s\n' "$*"; rr_logline "$*"; }
rr_warn()  { printf 'WARN: %s\n' "$*" >&2; rr_logline "WARN: $*"; }
rr_error() { printf 'ERROR: %s\n' "$*" >&2; rr_logline "ERROR: $*"; }

# rr_quote_one ARG: print ARG as one shell word, single-quoted unless it is plain,
# so that a printed command can be pasted into a shell.
rr_quote_one() {
    local s=$1
    if [[ $s =~ ^[A-Za-z0-9_./:=,@%+-]+$ ]]; then
        printf '%s' "$s"
    else
        printf "'%s'" "${s//\'/\'\\\'\'}"
    fi
}

# rr_quote ARG...: print the arguments as shell words, space-separated.
rr_quote() {
    local a first=1
    for a in "$@"; do
        [ $first = 1 ] || printf ' '
        rr_quote_one "$a"
        first=0
    done
}

# rr_show ARG...: print a command as "+ command" without running it.
rr_show() { rr_say "+ $(rr_quote "$@")"; }

# rr_run SECONDS [-C DIR] [NAME=VALUE...] COMMAND [ARG...]
# Runs one external command under a time limit. The limit sends SIGTERM and never
# SIGKILL; status 124 means the limit expired. The command is printed first as
# "+ ...". In dry-run mode (RR_DRY_RUN=1) it is only printed. Output also goes to
# RR_LOG_FILE when it is set.
rr_run() {
    local secs=$1 dir="" rc=0 shown
    shift
    if [ "${1:-}" = -C ]; then
        dir=$2
        shift 2
    fi
    local -a envs=()
    while [ $# -gt 1 ] && [[ $1 =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; do
        envs+=("$1")
        shift
    done
    local -a cmd=(timeout --signal=TERM "$secs" env "${envs[@]}" "$@")
    shown=$(rr_quote "${envs[@]}" "$@")
    # The log goes into bug reports: the caller's PATH stays a reference.
    shown=${shown//"$PATH"/'$PATH'}
    if [ -n "$dir" ]; then
        rr_say "+ (cd $(rr_quote_one "$dir") && $shown)"
    else
        rr_say "+ $shown"
    fi
    [ "$RR_DRY_RUN" = 1 ] && return 0

    (
        if [ -n "$dir" ]; then cd -- "$dir" || exit 1; fi
        if [ -n "$RR_LOG_FILE" ]; then
            "${cmd[@]}" 2>&1 | tee -a -- "$RR_LOG_FILE"
            exit "${PIPESTATUS[0]}"
        fi
        exec "${cmd[@]}"
    ) || rc=$?

    if [ "$rc" = 124 ]; then
        rr_error "time limit of ${secs}s reached: $shown"
    fi
    return "$rc"
}

# rr_python: print the first interpreter that actually runs (python3, then python).
# A Windows Store stub named python3 exists but does not run, so "command -v" alone
# is not enough.
rr_python() {
    local c
    for c in python3 python; do
        if command -v "$c" >/dev/null 2>&1 && "$c" -I -c 'pass' >/dev/null 2>&1; then
            printf '%s' "$c"
            return 0
        fi
    done
    return 1
}

rr_is_int()  { [[ ${1:-} =~ ^[1-9][0-9]*$ ]]; }
rr_is_util() { [[ ${1:-} =~ ^(0?\.[0-9]+|1(\.0+)?)$ ]]; }

# rr_installed_marker: print the install marker path and succeed, or fail if none.
rr_installed_marker() {
    local m
    for m in "$RR_HOME"/.installed-*; do
        if [ -e "$m" ]; then
            printf '%s' "$m"
            return 0
        fi
    done
    return 1
}

# rr_preset_load NAME: read presets/NAME.env into P_* variables. Only KEY=value
# lines are accepted. Returns 2 on a bad name, a missing file or a bad value.
rr_preset_load() {
    local name=$1 file line n=0 key val
    P_HF_ID= P_CONVERT= P_SERVED_NAME= P_CTX= P_KV= P_UTIL= P_MAXSEQ= P_MTP=
    P_MTP_METHOD= P_MTP_CAPTURE_SIZES= P_MTP_NEEDS_HEAD= P_TOOL_PARSER=
    P_REASONING_PARSER= P_VISION_KWARGS= P_VISION_DEFAULT= P_PRESET_ENV=
    P_REASON_END= P_REASON_END_SET=0 P_REASON_GRACE= P_BATCHTOK= P_LONGPREFILL=

    if ! [[ $name =~ ^[a-z0-9][a-z0-9._-]*$ ]]; then
        rr_error "invalid preset name: $name"
        return 2
    fi
    file=$RR_RELEASE_DIR/presets/$name.env
    if [ ! -r "$file" ]; then
        rr_error "unknown preset: $name (see --list)"
        return 2
    fi
    while IFS= read -r line || [ -n "$line" ]; do
        n=$((n + 1))
        case $line in
            ''|'#'*) continue ;;
        esac
        if ! [[ $line =~ ^([A-Z][A-Z0-9_]*)=(.*)$ ]]; then
            rr_error "$file:$n: expected KEY=value"
            return 2
        fi
        key=${BASH_REMATCH[1]}
        val=${BASH_REMATCH[2]}
        case $key in
            HF_ID) P_HF_ID=$val ;;
            CONVERT) P_CONVERT=$val ;;
            SERVED_NAME) P_SERVED_NAME=$val ;;
            CTX) P_CTX=$val ;;
            KV) P_KV=$val ;;
            UTIL) P_UTIL=$val ;;
            MAXSEQ) P_MAXSEQ=$val ;;
            MTP) P_MTP=$val ;;
            MTP_METHOD) P_MTP_METHOD=$val ;;
            MTP_CAPTURE_SIZES) P_MTP_CAPTURE_SIZES=$val ;;
            MTP_NEEDS_HEAD) P_MTP_NEEDS_HEAD=$val ;;
            TOOL_PARSER) P_TOOL_PARSER=$val ;;
            REASONING_PARSER) P_REASONING_PARSER=$val ;;
            VISION_KWARGS) P_VISION_KWARGS=$val ;;
            VISION_DEFAULT) P_VISION_DEFAULT=$val ;;
            PRESET_ENV) P_PRESET_ENV=$val ;;
            REASON_END) P_REASON_END=$val; P_REASON_END_SET=1 ;;
            REASON_GRACE) P_REASON_GRACE=$val ;;
            BATCHTOK) P_BATCHTOK=$val ;;
            LONGPREFILL) P_LONGPREFILL=$val ;;
            *)
                rr_error "$file:$n: unknown key $key"
                return 2
                ;;
        esac
    done < "$file"
    rr_preset_check "$name" "$file"
}

# rr_preset_check NAME FILE: validate the loaded preset and fill shared defaults.
rr_preset_check() {
    local name=$1 file=$2 var tok
    local -a envs=()
    for var in P_HF_ID P_SERVED_NAME P_CTX P_KV P_UTIL P_MAXSEQ P_MTP P_TOOL_PARSER \
               P_REASONING_PARSER P_VISION_KWARGS P_VISION_DEFAULT P_PRESET_ENV; do
        if [ -z "${!var}" ]; then
            rr_error "$file: missing or empty key ${var#P_}"
            return 2
        fi
    done
    if ! rr_is_int "$P_CTX"; then rr_error "$file: CTX must be a positive integer"; return 2; fi
    if ! rr_is_int "$P_MAXSEQ"; then rr_error "$file: MAXSEQ must be a positive integer"; return 2; fi
    if ! rr_is_util "$P_UTIL"; then rr_error "$file: UTIL must be a fraction such as 0.92"; return 2; fi
    case $P_KV in
        int4_per_token_head|int8_per_token_head|auto) ;;
        *) rr_error "$file: KV must be int4_per_token_head, int8_per_token_head or auto"; return 2 ;;
    esac
    case $P_CONVERT in
        ''|awq-to-ct) ;;
        *) rr_error "$file: CONVERT must be empty or awq-to-ct"; return 2 ;;
    esac
    case $P_MTP in 0|1) ;; *) rr_error "$file: MTP must be 0 or 1"; return 2 ;; esac
    case $P_VISION_KWARGS in 0|1) ;; *) rr_error "$file: VISION_KWARGS must be 0 or 1"; return 2 ;; esac
    case $P_VISION_DEFAULT in 0|1) ;; *) rr_error "$file: VISION_DEFAULT must be 0 or 1"; return 2 ;; esac
    P_MTP_NEEDS_HEAD=${P_MTP_NEEDS_HEAD:-0}
    case $P_MTP_NEEDS_HEAD in 0|1) ;; *) rr_error "$file: MTP_NEEDS_HEAD must be 0 or 1"; return 2 ;; esac
    if [ "$P_MTP" = 1 ]; then
        if [ -z "$P_MTP_METHOD" ] || [ -z "$P_MTP_CAPTURE_SIZES" ]; then
            rr_error "$file: MTP=1 needs MTP_METHOD and MTP_CAPTURE_SIZES"
            return 2
        fi
    fi
    if [ -n "$P_MTP_CAPTURE_SIZES" ] && ! [[ $P_MTP_CAPTURE_SIZES =~ ^\[[0-9,]+\]$ ]]; then
        rr_error "$file: MTP_CAPTURE_SIZES must be a JSON list of integers"
        return 2
    fi
    P_REASON_GRACE=${P_REASON_GRACE:-2048}
    if ! [[ $P_REASON_GRACE =~ ^[0-9]+$ ]]; then rr_error "$file: REASON_GRACE must be an integer"; return 2; fi
    P_BATCHTOK=${P_BATCHTOK:-8192}
    P_LONGPREFILL=${P_LONGPREFILL:-2048}

    # PRESET_ENV is exported to the server. Keys the scripts compute themselves
    # or that must never be set are refused here.
    read -r -a envs <<< "$P_PRESET_ENV"
    for tok in "${envs[@]}"; do
        if ! [[ $tok =~ ^[A-Z][A-Z0-9_]*=[^[:space:]]*$ ]]; then
            rr_error "$file: PRESET_ENV entry '$tok' is not KEY=value"
            return 2
        fi
        case ${tok%%=*} in
            PYTORCH_HIP_ALLOC_CONF)
                rr_error "$file: PYTORCH_HIP_ALLOC_CONF breaks the allocator and is never set"
                return 2
                ;;
            VLLM_ROCM_INT4_DECODE_QLEN_MAX|VLLM_REASONEND_GRACE)
                rr_error "$file: ${tok%%=*} is computed by serve.sh and must not be in PRESET_ENV"
                return 2
                ;;
        esac
    done
    return 0
}
