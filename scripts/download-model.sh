#!/usr/bin/env bash
# Downloads the weights of one preset with hf (huggingface_hub), then converts them when
# the preset needs it (CONVERT=awq-to-ct) and checks the result.
#
# Usage: download-model.sh PRESET [--dir DIR] [--dry-run] [--help]
#   --dir DIR   target directory (default: $VLLM_RDNA2_HOME/models/PRESET)
#   --dry-run   print the commands only; touch nothing, use no network
#
# A converted preset ends in DIR-ct-int4, which serve.sh reads by default.
# Exit: 0 ready; 1 download, conversion or check failed; 2 bad call; 3 not installed.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/lib.sh"

DIR=
PRESET=
DOWNLOAD_TIMEOUT=43200
CHECK_TIMEOUT=3600

usage() {
    cat <<'EOF'
Usage: download-model.sh PRESET [--dir DIR] [--dry-run] [--help]

Downloads the weights of PRESET with hf from the venv, then converts
them if the preset has CONVERT=awq-to-ct and checks the result.

  --dir DIR   target directory (default: $VLLM_RDNA2_HOME/models/PRESET)
  --dry-run   print the commands only; nothing is created and no network is used
  --help      show this text

A converted preset ends in DIR-ct-int4 (the default is models/PRESET-ct-int4).
Exit: 0 ready; 1 download, conversion or check failed; 2 bad call; 3 not installed.
EOF
}

while [ $# -gt 0 ]; do
    case $1 in
        --dir)
            if [ $# -lt 2 ] || [ -z "$2" ]; then
                rr_error "--dir needs a directory (see --help)"
                exit 2
            fi
            DIR=$2
            shift 2
            ;;
        --dry-run)
            RR_DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
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

if [ -z "$PRESET" ]; then
    rr_error "PRESET is required (see --help)"
    exit 2
fi
rr_preset_load "$PRESET" || exit 2

[ -n "$DIR" ] || DIR=$RR_MODELS/$PRESET
case $DIR in
    /*) ;;
    *) DIR=$PWD/$DIR ;;
esac

# A dry run is allowed before installation: it only shows what would run.
if [ "$RR_DRY_RUN" = 0 ]; then
    if ! rr_installed_marker >/dev/null; then
        rr_error "not installed: no $RR_HOME/.installed-* marker; run install.sh first"
        exit 3
    fi
    if [ ! -x "$RR_VENV/bin/hf" ]; then
        rr_error "hf is missing from $RR_VENV/bin; reinstall with install.sh --force"
        exit 3
    fi
fi

OUT=$DIR
[ -n "$P_CONVERT" ] && OUT=$DIR-ct-int4

rr_run 60 mkdir -p -- "$(dirname "$DIR")" || { rr_error "cannot create $(dirname "$DIR")"; exit 1; }
# huggingface_hub 1.x, pinned in constraints/, keeps huggingface-cli only as a stub that
# refuses to work.
if ! rr_run "$DOWNLOAD_TIMEOUT" "$RR_VENV/bin/hf" download "$P_HF_ID" --local-dir "$DIR"; then
    rr_error "download of $P_HF_ID failed"
    exit 1
fi

if [ -n "$P_CONVERT" ]; then
    if ! rr_run "$DOWNLOAD_TIMEOUT" "$RR_VENV/bin/python" -I "$RR_RELEASE_DIR/scripts/awq-to-ct.py" \
            --src "$DIR" --dst "$OUT"; then
        rr_error "conversion from AWQ to compressed-tensors failed"
        exit 1
    fi
    if ! rr_run "$CHECK_TIMEOUT" "$RR_VENV/bin/python" -I "$RR_RELEASE_DIR/scripts/awq-to-ct-check.py" \
            --src "$DIR" --dst "$OUT"; then
        rr_error "conversion check failed for $OUT"
        exit 1
    fi
fi

if [ "$RR_DRY_RUN" = 1 ]; then
    rr_say "dry run: the model would be ready at $OUT"
else
    rr_say "ready: $OUT"
fi
exit 0
