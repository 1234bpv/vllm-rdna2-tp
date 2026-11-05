#!/usr/bin/env bash
# Installs vLLM for RDNA2 (gfx1030) cards: checks the system, creates a virtual
# environment, installs PyTorch for ROCm 7.2 at the pinned versions, fetches the
# pinned vLLM commit, applies the RDNA2 patch, and builds vLLM in place.
#
# Usage: install.sh [--prefix DIR] [--python BIN] [--jobs N] [--dry-run]
#                   [--skip-checks] [--force] [--help]
#
# Exit: 0 installed, or already installed; 2 bad call; 3 the system does not meet
#       the requirements; 4 a dependency install failed; 5 the source or the patch
#       could not be obtained or applied; 6 the build failed; 7 the check after
#       installation failed.
#
# This script never runs sudo. System packages (ROCm, Python venv, build tools) are
# printed as commands for you to run.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/scripts/lib.sh"

VLLM_REPO=https://github.com/vllm-project/vllm.git
VLLM_COMMIT=c296851a7d173fa89d2eefbca0243be42ae9b5e0
TOTAL=7
DEFAULT_HOME=${HOME:-.}/.local/share/vllm-rdna2

PY=python3.12
JOBS=
PREFIX=
FORCE=0
SKIP_CHECKS=0

usage() {
    cat <<'EOF'
Usage: install.sh [--prefix DIR] [--python BIN] [--jobs N] [--dry-run]
                  [--skip-checks] [--force] [--help]

Installs vLLM for RDNA2 (gfx1030) cards into $VLLM_RDNA2_HOME (default
~/.local/share/vllm-rdna2): venv, PyTorch for ROCm 7.2, vLLM at a pinned commit with
the RDNA2 patch, built in place. PyTorch alone is a 6.2 GB download; the native build
took about 10 minutes on 48 threads.

  --prefix DIR      install home (default: $VLLM_RDNA2_HOME or ~/.local/share/vllm-rdna2)
  --python BIN      Python for the venv (default python3.12; 3.10 to 3.14 accepted)
  --jobs N          parallel build jobs (default: number of CPUs)
  --dry-run         print every command; change nothing, create nothing, use no network
                    (the doctor check still runs)
  --skip-checks     turn a failed system check into a warning
  --force           remove the venv and the source tree first, then reinstall
                    (models/ and logs/ are kept)
  --help            show this text

Exit: 0 installed or already installed; 2 bad call; 3 system requirements not met;
4 dependency install failed; 5 source or patch failed; 6 build failed; 7 check after
installation failed.
EOF
}

while [ $# -gt 0 ]; do
    case $1 in
        --prefix)
            if [ $# -lt 2 ] || [ -z "$2" ]; then
                rr_error "--prefix needs a directory (see --help)"
                exit 2
            fi
            PREFIX=$2
            shift 2
            ;;
        --python)
            if [ $# -lt 2 ] || [ -z "$2" ]; then
                rr_error "--python needs a command (see --help)"
                exit 2
            fi
            PY=$2
            shift 2
            ;;
        --jobs)
            if [ $# -lt 2 ] || ! rr_is_int "$2"; then
                rr_error "--jobs needs a positive number (see --help)"
                exit 2
            fi
            JOBS=$2
            shift 2
            ;;
        --dry-run)
            RR_DRY_RUN=1
            shift
            ;;
        --skip-checks)
            SKIP_CHECKS=1
            shift
            ;;
        --force)
            FORCE=1
            shift
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

if [ -n "$PREFIX" ]; then
    case $PREFIX in
        /*) ;;
        *) PREFIX=$PWD/$PREFIX ;;
    esac
    rr_set_home "$PREFIX"
fi

if [ -z "$JOBS" ]; then
    JOBS=$(timeout 10 nproc 2>/dev/null)
    if [ -z "$JOBS" ]; then
        rr_warn "cannot count CPUs; using 4 build jobs (set --jobs to change)"
        JOBS=4
    fi
fi

VPY=$RR_VENV/bin/python
PATCH=$RR_RELEASE_DIR/patches/vllm-rdna2-tp-$RR_VERSION.patch
CONSTRAINTS=$RR_RELEASE_DIR/constraints/rocm-7.2.txt
MARKER=$RR_HOME/.installed-$RR_VERSION

step() { rr_say ""; rr_say "[$1/$TOTAL] $2"; }
die() {
    local code=$1
    shift
    rr_error "$*"
    exit "$code"
}

# gitx SECONDS ARGS...: git in the source tree, with CRLF conversion off so that the
# patch applies byte for byte.
gitx() {
    local secs=$1
    shift
    rr_run "$secs" -C "$RR_SRC" git -c core.autocrlf=false "$@"
}

# quiet_run SECONDS DESCRIPTION CMD...: runs CMD under a time limit, printing only
# DESCRIPTION. Its output goes to the log, and to the screen only on failure.
quiet_run() {
    local secs=$1 desc=$2 out rc
    shift 2
    rr_say "+ $desc"
    [ "$RR_DRY_RUN" = 1 ] && return 0
    out=$(timeout --signal=TERM "$secs" "$@" 2>&1)
    rc=$?
    rr_logline "$out"
    if [ "$rc" != 0 ]; then
        [ "$rc" = 124 ] && rr_error "time limit of ${secs}s reached: $desc"
        printf '%s\n' "$out" | tail -n 20 >&2
    fi
    return "$rc"
}

# python_version BIN: prints MAJOR.MINOR of BIN, or nothing if it does not run.
python_version() {
    timeout 30 "$1" -I -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null
}

# run_doctor: the system check. Prints the ROCm commands on failure. Returns 0 when
# the install may continue, 1 when it must stop.
run_doctor() {
    local out rc
    rr_say "+ $RR_RELEASE_DIR/scripts/doctor.sh --pre"
    out=$(timeout 120 "$RR_RELEASE_DIR/scripts/doctor.sh" --pre 2>&1)
    rc=$?
    [ "$rc" = 124 ] && rr_error "doctor.sh did not finish within 120 s"
    rr_say "$out"
    if [ "$rc" = 0 ]; then
        return 0
    fi
    if [ "$SKIP_CHECKS" = 1 ]; then
        rr_warn "doctor.sh reported FAIL; continuing because of --skip-checks"
        return 0
    fi
    rr_say ""
    rr_say "ROCm 7.2 is not ready, or a FAIL line above needs fixing. Run these commands yourself;"
    rr_say "this script never uses sudo:"
    rr_say ""
    rr_say "  curl -LO https://repo.radeon.com/amdgpu-install/7.2.4/ubuntu/noble/amdgpu-install_7.2.4.70204-1_all.deb"
    rr_say "  sudo apt-get install -y ./amdgpu-install_7.2.4.70204-1_all.deb"
    rr_say "  sudo apt-get update"
    rr_say "  sudo amdgpu-install --usecase=rocm --no-dkms -y"
    rr_say "  sudo usermod -aG render,video \"\$USER\"    # then log out and back in"
    return 1
}

# Already installed: nothing to do. Checked first, so a re-run changes nothing.
if [ -e "$MARKER" ] && [ "$FORCE" = 0 ]; then
    rr_say "already installed: $MARKER exists; nothing changed (use --force to reinstall)"
    exit 0
fi

# The patch and the pinned constraints ship with the release; check them here, so
# that a damaged checkout is found before anything is downloaded.
[ -f "$CONSTRAINTS" ] || die 4 "constraints file is missing: $CONSTRAINTS"
[ -f "$PATCH" ] || die 5 "patch is missing: $PATCH"

if [ "$RR_DRY_RUN" = 0 ]; then
    mkdir -p -- "$RR_LOGS" || die 3 "cannot create $RR_LOGS"
    RR_LOG_FILE=$RR_LOGS/install-$(date +%Y%m%d-%H%M%S).log
    : > "$RR_LOG_FILE" || die 3 "cannot create the log $RR_LOG_FILE"
    rr_say "log: $RR_LOG_FILE"
else
    rr_say "dry run: nothing will be changed"
fi

step 1 "system check: doctor.sh --pre"
run_doctor || exit 3

step 2 "virtual environment in $RR_VENV"
pyver=$(python_version "$PY")
if [ -z "$pyver" ]; then
    if [ "$RR_DRY_RUN" = 1 ]; then
        rr_warn "$PY does not run here; a real install would stop (the dry run continues)"
    else
        die 3 "$PY does not run: install python3.12, python3.12-dev and python3.12-venv (sudo apt-get install -y python3.12 python3.12-dev python3.12-venv) or pass --python BIN"
    fi
else
    case $pyver in
        3.1[0-4]) ;;
        *) die 3 "$PY is Python $pyver; this release needs Python 3.10 to 3.14" ;;
    esac
    if [ "$pyver" != 3.12 ]; then
        rr_warn "tested only with 3.12; $PY is $pyver"
    fi
fi
# cmake needs the headers of this very interpreter: better now than deep into the build.
if [ -n "$pyver" ] && ! timeout 30 "$PY" -I -c 'import os, sys, sysconfig; sys.exit(not os.path.exists(os.path.join(sysconfig.get_path("include"), "Python.h")))' 2>/dev/null; then
    if [ "$RR_DRY_RUN" = 1 ]; then
        rr_warn "$PY has no Python.h; a real install would stop (the dry run continues)"
    else
        die 3 "$PY has no Python.h: install its development package (on Ubuntu: sudo apt-get install -y python$pyver-dev)"
    fi
fi
# Only now: an interpreter that fails the checks above must not cost the old venv.
if [ "$FORCE" = 1 ]; then
    rr_say "--force: removing $RR_VENV and $RR_SRC and the install markers (models/ and logs/ are kept)"
    rr_run 600 rm -rf -- "$RR_VENV" "$RR_SRC" || die 3 "cannot remove the old venv or source tree"
    shopt -s nullglob
    markers=("$RR_HOME"/.installed-*)
    shopt -u nullglob
    if [ ${#markers[@]} -gt 0 ]; then
        rr_run 60 rm -f -- "${markers[@]}" || die 3 "cannot remove the old install markers"
    fi
fi
rr_run 60 mkdir -p -- "$RR_HOME" || die 3 "cannot create $RR_HOME"
rr_run 300 "$PY" -m venv "$RR_VENV" || die 3 "cannot create the venv at $RR_VENV with $PY (on Ubuntu: sudo apt-get install -y python3.12-venv)"

step 3 "build dependencies"
rr_run 1800 "$VPY" -m pip install -c "$CONSTRAINTS" \
    "setuptools>=77.0.3,<80.0.0" setuptools-scm setuptools-rust "cmake>=3.26.1,<4" \
    ninja packaging jinja2 wheel || die 4 "installing the build dependencies failed (see the log)"

step 4 "PyTorch for ROCm 7.2, and amdsmi"
rr_run 7200 "$VPY" -m pip install --extra-index-url https://download.pytorch.org/whl/rocm7.2 \
    -c "$CONSTRAINTS" torch torchvision triton || die 4 "installing torch, torchvision and triton failed (see the log)"
AMDSMI_SRC=$RR_ROCM_PATH/share/amd_smi
if [ "$RR_DRY_RUN" = 1 ]; then
    tmp='<tmpdir>'
else
    tmp=$(mktemp -d "${TMPDIR:-/tmp}/amd-smi.XXXXXX") || die 4 "cannot create a temporary directory"
fi
# The amdsmi sources sit in /opt, which is read-only for the user; pip needs a copy.
cleanup_tmp() { rr_run 60 rm -rf -- "$tmp"; }
rr_run 300 cp -r -- "$AMDSMI_SRC" "$tmp/amd_smi" || { cleanup_tmp; die 4 "cannot copy $AMDSMI_SRC (is ROCm installed?)"; }
rr_run 1200 "$VPY" -m pip install -c "$CONSTRAINTS" "$tmp/amd_smi" || { cleanup_tmp; die 4 "installing amdsmi failed (see the log)"; }
cleanup_tmp

step 5 "vLLM source at commit $VLLM_COMMIT, with the RDNA2 patch"
rr_run 60 mkdir -p -- "$RR_SRC" || die 5 "cannot create $RR_SRC"
gitx 60 init || die 5 "git init failed in $RR_SRC"
if [ "$RR_DRY_RUN" = 0 ] && timeout 30 git -C "$RR_SRC" remote get-url origin >/dev/null 2>&1; then
    gitx 60 remote set-url origin "$VLLM_REPO" || die 5 "git remote set-url failed"
else
    gitx 60 remote add origin "$VLLM_REPO" || die 5 "git remote add failed"
fi
gitx 1800 fetch --depth 1 origin "$VLLM_COMMIT" || die 5 "fetching commit $VLLM_COMMIT from $VLLM_REPO failed (network?)"
# A re-run after an unfinished install finds the patch already applied, and this tree
# belongs to the installer. checkout --force restores the tracked files; clean removes the
# files the patch added. -x is needed: upstream's .gitignore covers the patch's .hip files.
gitx 300 checkout --force FETCH_HEAD || die 5 "checkout of FETCH_HEAD failed"
gitx 300 clean -fdx || die 5 "git clean failed in $RR_SRC"
gitx 120 apply --whitespace=nowarn "$PATCH" || die 5 "the patch does not apply to $VLLM_COMMIT: $PATCH"

step 6 "build vLLM in place (editable)"
if [ "$RR_DRY_RUN" = 0 ] && [ ! -f "$RR_SRC/requirements/rocm.txt" ]; then
    die 5 "$RR_SRC/requirements/rocm.txt is missing: the pinned commit or the patch is wrong"
fi
rr_run 3600 -C "$RR_SRC" "$VPY" -m pip install -c "$CONSTRAINTS" -r requirements/rocm.txt \
    || die 4 "installing requirements/rocm.txt failed (see the log)"
# PATH puts the venv first so that cmake and ninja from the venv are the ones used.
rr_run 10800 -C "$RR_SRC" \
    ROCM_PATH="$RR_ROCM_PATH" HIP_PATH="$RR_ROCM_PATH" \
    PATH="$RR_VENV/bin:$RR_ROCM_PATH/bin:$PATH" \
    PYTORCH_ROCM_ARCH=gfx1030 VLLM_TARGET_DEVICE=rocm \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.0.0 \
    CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" MAX_JOBS="$JOBS" \
    "$VPY" -m pip install -e . --no-build-isolation --no-deps \
    || die 6 "the build failed (see the log; the last lines are the cause)"

step 7 "checks after installation"
POST_CHECK='import sys
import torch
import vllm
print("vllm", getattr(vllm, "__version__", "?"), "torch", torch.__version__)
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False: the driver shows no GPU")
if "gfx1030" not in torch.cuda.get_arch_list():
    sys.exit("gfx1030 is not in torch.cuda.get_arch_list(): " + str(torch.cuda.get_arch_list()))
import vllm.distributed.device_communicators.host_all_reduce
print("host_all_reduce: import ok")'
quiet_run 600 "python -I -c <import vllm, torch; CUDA visible; gfx1030 in arch list; host_all_reduce>" \
    "$VPY" -I -c "$POST_CHECK" || die 7 "the import and GPU check failed"
quiet_run 300 "vllm --help" "$RR_VENV/bin/vllm" --help || die 7 "vllm --help failed"
quiet_run 60 "ninja --version" "$RR_VENV/bin/ninja" --version \
    || die 7 "ninja is missing from the venv: the .hip kernels cannot be built"

if [ "$RR_DRY_RUN" = 0 ]; then
    printf 'version %s\ninstalled %s\n' "$RR_VERSION" "$(date -Iseconds)" > "$MARKER" \
        || die 7 "cannot write the marker $MARKER"
    rr_say ""
    rr_say "installed: vLLM $RR_VERSION in $RR_HOME"
else
    rr_say ""
    rr_say "dry run complete: nothing was changed"
fi
rr_say "next steps:"
rr_say "  $RR_RELEASE_DIR/scripts/download-model.sh PRESET   fetch the weights of a preset"
rr_say "  $RR_RELEASE_DIR/serve.sh PRESET                    start the server for a preset"
rr_say "  $RR_RELEASE_DIR/serve.sh --list                    list the presets"
if [ "$RR_HOME" != "$DEFAULT_HOME" ]; then
    rr_say "  export VLLM_RDNA2_HOME=$RR_HOME   (this install is not in the default place)"
fi
exit 0
