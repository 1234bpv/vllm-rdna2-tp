#!/usr/bin/env bash
# run-matrix.sh PRESET [--points "LENGTH:STREAMS ..."] [--url URL] [--output FILE]
#                      [--label L] [--point-timeout S] [--blocks N]
#
# Measures a server that is already running. Start it with serve.sh first: this script
# does not start, stop or change the server. For each LENGTH:STREAMS point it runs
# kv-concurrency.py once and appends the record to the output JSONL.
#
# Defaults: --url http://127.0.0.1:8100, --label PRESET, --point-timeout 3000 (seconds),
# --blocks 0, --points "300:1 300:2 300:4 32768:1 32768:2 32768:4 125000:1 125000:2",
# --output results/local/PRESET-YYYY-MM-DD.jsonl (relative to the release root).
#
# --blocks N is the block count from the server log line 'GPU KV cache size'. 0 means
# not given: the record's blocks_peak is then null and is not compared.
#
# Environment:
#   PYTHON   interpreter (default: python3).
#
# Exit status: 0 all points succeeded; 1 at least one point failed (the remaining points
# still ran); 2 bad arguments; 3 the server did not answer /v1/models.

set -u

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
PYTHON=${PYTHON:-python3}
MAX_TOKENS=256
CHECK_TIMEOUT_S=20
DEFAULT_POINTS="300:1 300:2 300:4 32768:1 32768:2 32768:4 125000:1 125000:2"

usage() {
    echo "usage: run-matrix.sh PRESET [--points \"LENGTH:STREAMS ...\"] [--url URL] [--output FILE] [--label L] [--point-timeout S] [--blocks N]" >&2
}

if [ $# -lt 1 ]; then
    usage
    exit 2
fi
case $1 in
    -h|--help) usage; exit 0 ;;
esac

PRESET=$1
shift
POINTS=$DEFAULT_POINTS
URL=http://127.0.0.1:8100
OUTPUT=
LABEL=$PRESET
POINT_TIMEOUT_S=3000
BLOCKS=0

while [ $# -gt 0 ]; do
    if [ $# -lt 2 ]; then
        echo "run-matrix: $1 needs a value" >&2
        exit 2
    fi
    case $1 in
        --points) POINTS=$2 ;;
        --url) URL=$2 ;;
        --output) OUTPUT=$2 ;;
        --label) LABEL=$2 ;;
        --point-timeout) POINT_TIMEOUT_S=$2 ;;
        --blocks) BLOCKS=$2 ;;
        *) echo "run-matrix: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
    shift 2
done

# The preset name becomes part of a file name, so only plain characters are allowed.
case $PRESET in
    ''|*[!A-Za-z0-9._-]*) echo "run-matrix: PRESET may contain only letters, digits, '.', '_' and '-'" >&2; exit 2 ;;
esac
case $POINT_TIMEOUT_S in
    ''|*[!0-9.]*) echo "run-matrix: --point-timeout must be a number of seconds" >&2; exit 2 ;;
esac
case $BLOCKS in
    ''|*[!0-9]*) echo "run-matrix: --blocks must be a non-negative integer" >&2; exit 2 ;;
esac

point_count=0
for point in $POINTS; do
    if ! [[ $point =~ ^[0-9]+:[0-9]+$ ]]; then
        echo "run-matrix: bad point '$point', expected LENGTH:STREAMS" >&2
        exit 2
    fi
    point_count=$((point_count + 1))
done
if [ "$point_count" -eq 0 ]; then
    echo "run-matrix: no points to measure" >&2
    exit 2
fi

[ -n "$OUTPUT" ] || OUTPUT="$ROOT/results/local/$PRESET-$(date +%Y-%m-%d).jsonl"
if ! mkdir -p "$(dirname "$OUTPUT")"; then
    echo "run-matrix: cannot create $(dirname "$OUTPUT")" >&2
    exit 2
fi

models=$(timeout "$CHECK_TIMEOUT_S" curl -sf "$URL/v1/models")
rc=$?
case $models in
    *'"data"'*) ;;
    *) rc=3 ;;
esac
if [ "$rc" -ne 0 ]; then
    echo "run-matrix: the server at $URL did not answer /v1/models; start it with serve.sh" >&2
    exit 3
fi

failed=0
for point in $POINTS; do
    length=${point%%:*}
    streams=${point##*:}
    echo "run-matrix: point length=$length streams=$streams"
    timeout "$POINT_TIMEOUT_S" "$PYTHON" -I "$HERE/kv-concurrency.py" \
        --base-url "$URL" --streams "$streams" --length "$length" --blocks "$BLOCKS" \
        --output "$OUTPUT" --label "$LABEL point=L${length}x${streams}" --max-tokens "$MAX_TOKENS"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        failed=$((failed + 1))
        if [ "$rc" -eq 124 ]; then
            reason="time limit ${POINT_TIMEOUT_S} s"
        else
            reason="exit $rc"
        fi
        echo "run-matrix: point length=$length streams=$streams FAILED ($reason); continuing" >&2
    fi
done

echo "run-matrix: $((point_count - failed)) of $point_count points succeeded; records in $OUTPUT"
[ "$failed" -eq 0 ]
