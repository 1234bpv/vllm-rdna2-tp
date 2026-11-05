#!/usr/bin/env bash
# Checks that this machine can run vLLM on RDNA2 (gfx1030) cards.
#
# Usage: doctor.sh [--pre] [--json] [--help]
#   --pre   skip the checks that need an installed venv and vLLM (run before install)
#   --json  print one JSON object instead of one line per check
#
# Output lines: PASS|WARN|FAIL <name>: <detail>
# Exit: 0 no FAIL; 1 at least one FAIL; 2 bad usage.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/lib.sh"

PRE=0
JSON=0
NAMES=()
STATUSES=()
DETAILS=()

usage() {
    cat <<'EOF'
Usage: doctor.sh [--pre] [--json] [--help]

Checks that this machine can run vLLM on RDNA2 (gfx1030) cards.

  --pre    skip the checks that need an installed venv and vLLM
  --json   print one JSON object: {"checks":[...],"fail":N,"warn":N}
  --help   show this text

Exit: 0 no FAIL; 1 at least one FAIL; 2 bad usage.
EOF
}

add() {
    NAMES+=("$1")
    STATUSES+=("$2")
    DETAILS+=("$3")
}

gib() {
    awk -v k="$1" 'BEGIN { printf "%.1f GiB", k / 1048576 }'
}

check_os() {
    local kernel distro
    kernel=$(uname -s)
    if [ "$kernel" != Linux ]; then
        add os FAIL "$kernel is not Linux; the release targets Ubuntu 24.04"
        return
    fi
    distro=$( (. /etc/os-release && printf '%s %s' "${ID:-unknown}" "${VERSION_ID:-unknown}") 2>/dev/null)
    if [ "$distro" = "ubuntu 24.04" ]; then
        add os PASS "Ubuntu 24.04"
    else
        add os WARN "${distro:-unknown distribution}: tested on Ubuntu 24.04"
    fi
}

check_kfd() {
    local kfd=${DOCTOR_KFD:-/dev/kfd}
    if [ ! -e "$kfd" ]; then
        add kfd FAIL "$kfd is missing: the amdgpu driver is not loaded"
    elif [ -r "$kfd" ] && [ -w "$kfd" ]; then
        add kfd PASS "$kfd is readable and writable"
    else
        add kfd FAIL "$kfd is not readable and writable by $(id -un): add the user to the render and video groups"
    fi
}

check_groups() {
    local g
    for g in $(id -Gn 2>/dev/null); do
        case $g in
            render|video)
                add groups PASS "$(id -un) is in group $g"
                return
                ;;
        esac
    done
    add groups WARN "$(id -un) is in neither render nor video: sudo usermod -aG render,video $(id -un)"
}

check_rocm() {
    local bin=$RR_ROCM_PATH/bin/rocminfo
    if [ -x "$bin" ]; then
        add rocm PASS "$bin"
    else
        add rocm FAIL "$bin is missing or not executable: install ROCm 7.2 (see install.sh)"
    fi
}

check_rocm_version() {
    local bin=$RR_ROCM_PATH/bin/hipconfig v
    if [ ! -x "$bin" ]; then
        add rocm-version WARN "hipconfig not found: tested with ROCm 7.2.4"
        return
    fi
    v=$(timeout 30 "$bin" --version 2>/dev/null | head -n 1)
    case $v in
        7.2*) add rocm-version PASS "ROCm $v" ;;
        *) add rocm-version WARN "${v:-version not reported}: tested with ROCm 7.2.4" ;;
    esac
}

check_hipcc() {
    local bin=$RR_ROCM_PATH/bin/hipcc
    if [ -x "$bin" ]; then
        add hipcc PASS "$bin"
    else
        add hipcc FAIL "$bin is missing or not executable: kernels are compiled at run time"
    fi
}

check_gpu_count() {
    local info counts hsa_error status_file asleep="" n=0 other=0
    info=$(timeout 60 "$RR_ROCM_PATH/bin/rocminfo" 2>&1)
    # Count GPU devices by their "Name: gfx..." line. The ISA lines
    # ("amdgcn-amd-amdhsa--gfx1030") contain the same token and must not count.
    counts=$(printf '%s\n' "$info" | awk '
        /^[[:space:]]*Name:[[:space:]]+gfx[0-9a-f]+[[:space:]]*$/ {
            if ($2 == "gfx1030") g++
            else if ($2 ~ /^gfx103/) o++
        }
        END { print g + 0, o + 0 }')
    read -r n other <<< "$counts"
    if [ "$n" = 0 ]; then
        hsa_error=$(printf '%s\n' "$info" | grep -o 'HSA_STATUS_ERROR[A-Z_]*' | head -n 1)
        for status_file in /sys/bus/pci/drivers/amdgpu/*/power/runtime_status; do
            if [ "$(cat "$status_file" 2>/dev/null)" = error ]; then
                asleep="$asleep $(basename "$(dirname "$(dirname "$status_file")")")"
            fi
        done
        if [ -n "$asleep" ]; then
            add gpu-count FAIL "cards failed to wake up:$asleep (${hsa_error:-no rocminfo error}): reboot, see KI-38"
        elif [ -n "$hsa_error" ]; then
            add gpu-count FAIL "rocminfo failed with $hsa_error"
        else
            add gpu-count FAIL "no gfx1030 GPU device in rocminfo"
        fi
    elif [ "$other" != 0 ]; then
        add gpu-count WARN "$n gfx1030 and $other other gfx103x device(s): untested"
    elif [ "$n" = 1 ]; then
        add gpu-count WARN "$n gfx1030 device; one card: TP=1 only"
    else
        add gpu-count PASS "$n gfx1030 devices"
    fi
}

check_python() {
    local c v
    for c in python3.12 python3; do
        command -v "$c" >/dev/null 2>&1 || continue
        v=$("$c" -I -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null) || continue
        if [ "$v" = 3.12 ]; then
            add python PASS "$c is Python 3.12"
            return
        fi
    done
    for c in python3.10 python3.11 python3.13 python3.14 python3; do
        command -v "$c" >/dev/null 2>&1 || continue
        v=$("$c" -I -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null) || continue
        case $v in
            3.1[0-4])
                add python WARN "python3.12 not found; $c is Python $v (tested only with 3.12)"
                return
                ;;
        esac
    done
    add python FAIL "no Python 3.10-3.14 found: install python3.12 and python3.12-venv"
}

check_build_tools() {
    local t missing=""
    for t in git g++ make; do
        command -v "$t" >/dev/null 2>&1 || missing="$missing $t"
    done
    if [ -z "$missing" ]; then
        add build-tools PASS "git, g++ and make found"
    else
        add build-tools FAIL "missing:$missing (sudo apt-get install git g++ make)"
    fi
    check_python_dev
}

# The build of vLLM needs the Python headers (cmake: Development.Module).
check_python_dev() {
    local c
    for c in python3.12 python3; do
        command -v "$c" >/dev/null 2>&1 || continue
        if timeout 30 "$c" -I -c 'import os, sys, sysconfig; sys.exit(not os.path.exists(os.path.join(sysconfig.get_path("include"), "Python.h")))' 2>/dev/null; then
            add python-dev PASS "$c has Python.h"
        else
            add python-dev FAIL "$c has no Python.h (sudo apt-get install python3.12-dev)"
        fi
        return
    done
}

check_ram() {
    local kb
    kb=$(awk '/^MemTotal:/ { print $2 }' /proc/meminfo 2>/dev/null)
    if [ -z "$kb" ]; then
        add ram WARN "cannot read MemTotal from /proc/meminfo"
    elif [ "$kb" -ge 33554432 ]; then
        add ram PASS "$(gib "$kb") RAM"
    else
        add ram WARN "$(gib "$kb") RAM: 32 GiB recommended"
    fi
}

check_shm() {
    local kb
    kb=$(df -P -k /dev/shm 2>/dev/null | awk 'NR == 2 { print $4 }')
    if [ -z "$kb" ]; then
        add shm WARN "cannot read free space of /dev/shm"
    elif [ "$kb" -ge 8388608 ]; then
        add shm PASS "$(gib "$kb") free in /dev/shm"
    else
        add shm WARN "$(gib "$kb") free in /dev/shm: 8 GiB needed for all-reduce buffers and KV offload"
    fi
}

check_disk() {
    local d=$RR_HOME kb
    while [ ! -e "$d" ] && [ "$d" != / ] && [ "$d" != . ]; do
        d=$(dirname "$d")
    done
    kb=$(df -P -k "$d" 2>/dev/null | awk 'NR == 2 { print $4 }')
    if [ -z "$kb" ]; then
        add disk WARN "cannot read free space at $d"
    elif [ "$kb" -ge 41943040 ]; then
        add disk PASS "$(gib "$kb") free at $d"
    else
        add disk WARN "$(gib "$kb") free at $d: 40 GiB recommended"
    fi
}

check_venv() {
    local py=$RR_VENV/bin/python
    if [ -e "$py" ]; then
        add venv PASS "$py"
    else
        add venv FAIL "$py is missing: run install.sh"
    fi
    # The .hip kernels are built at first use with the ninja of the venv.
    if [ -e "$py" ]; then
        if [ -x "$RR_VENV/bin/ninja" ]; then
            add ninja PASS "$RR_VENV/bin/ninja"
        else
            add ninja FAIL "$RR_VENV/bin/ninja is missing: run install.sh --force"
        fi
    fi
}

check_vllm() {
    local py=$RR_VENV/bin/python out rc=0
    if [ ! -e "$py" ]; then
        add vllm FAIL "venv is missing, so vLLM was not imported"
        return
    fi
    out=$(timeout 300 "$py" -I -c 'import vllm; import vllm.distributed.device_communicators.host_all_reduce' 2>&1) || rc=$?
    if [ "$rc" = 0 ]; then
        add vllm PASS "vllm and host_all_reduce import"
    else
        add vllm FAIL "import failed (exit $rc): $(printf '%s' "$out" | tail -n 1)"
    fi
}

# Informational only: a PCIe peer link is not required for the all-reduce path.
check_p2p() {
    local py=$RR_VENV/bin/python out
    if [ ! -e "$py" ]; then
        add p2p PASS "p2p=unknown (venv is missing; not required)"
        return
    fi
    out=$(timeout 120 "$py" -I -c 'import torch
if torch.cuda.device_count() < 2:
    print("skip")
else:
    print(torch.cuda.can_device_access_peer(0, 1))' 2>/dev/null | tail -n 1)
    case $out in
        True) add p2p PASS "p2p=True" ;;
        False) add p2p PASS "p2p=False (expected on most consumer systems; not required)" ;;
        *) add p2p PASS "p2p=unknown (probe did not report; not required)" ;;
    esac
}

print_text() {
    local i
    for i in "${!NAMES[@]}"; do
        printf '%s %s: %s\n' "${STATUSES[i]}" "${NAMES[i]}" "${DETAILS[i]}"
    done
}

json_escape() {
    local s=$1
    s=${s//\\/\\\\}
    s=${s//\"/\\\"}
    s=${s//$'\n'/\\n}
    s=${s//$'\t'/\\t}
    printf '%s' "$s"
}

print_json() {
    local i sep="" fail=0 warn=0
    for i in "${!NAMES[@]}"; do
        case ${STATUSES[i]} in
            FAIL) fail=$((fail + 1)) ;;
            WARN) warn=$((warn + 1)) ;;
        esac
    done
    printf '{"checks":['
    for i in "${!NAMES[@]}"; do
        printf '%s{"name":"%s","status":"%s","detail":"%s"}' \
            "$sep" "${NAMES[i]}" "${STATUSES[i]}" "$(json_escape "${DETAILS[i]}")"
        sep=","
    done
    printf '],"fail":%d,"warn":%d}\n' "$fail" "$warn"
}

main() {
    local i fail=0
    while [ $# -gt 0 ]; do
        case $1 in
            --pre) PRE=1 ;;
            --json) JSON=1 ;;
            -h|--help) usage; exit 0 ;;
            *)
                rr_error "unknown argument: $1 (see --help)"
                exit 2
                ;;
        esac
        shift
    done

    check_os
    check_kfd
    check_groups
    check_rocm
    check_rocm_version
    check_hipcc
    check_gpu_count
    check_python
    check_build_tools
    check_ram
    check_shm
    check_disk
    if [ "$PRE" = 0 ]; then
        check_venv
        check_vllm
        check_p2p
    fi

    for i in "${!NAMES[@]}"; do
        [ "${STATUSES[i]}" = FAIL ] && fail=$((fail + 1))
    done
    if [ "$JSON" = 1 ]; then
        print_json
    else
        print_text
    fi
    [ "$fail" = 0 ]
}

main "$@"
