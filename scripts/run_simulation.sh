#!/usr/bin/env bash
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"

if [ -z "${OMNETPP_ROOT:-}" ] && [ -n "${OMNETPP_HOME:-}" ]; then
  OMNETPP_ROOT="$OMNETPP_HOME"
fi

if [ -z "${OMNETPP_ROOT:-}" ] && [ -d "/workspaces/omnetpp" ]; then
  OMNETPP_ROOT="/workspaces/omnetpp"
fi

if [ -n "${OMNETPP_ROOT:-}" ] && [ -f "$OMNETPP_ROOT/setenv" ]; then
  # shellcheck source=/dev/null
  set +u
  source "$OMNETPP_ROOT/setenv" -q
  set -u
fi

if ! command -v opp_configfilepath >/dev/null 2>&1; then
  echo "OMNeT++ build tools not found (opp_configfilepath missing)." >&2
  echo "Set OMNETPP_ROOT and source scripts/omnetpp_env.sh first." >&2
  exit 1
fi

CONFIG_NAME="${CONFIG_NAME:-QuickTest}"

case "$CONFIG_NAME" in
  QuickTest|OfflineDqnPolicy|OfflineDqnMlp|General)
    ;;
  *)
    echo "Unsupported CONFIG_NAME: $CONFIG_NAME" >&2
    echo "Allowed values: QuickTest, OfflineDqnPolicy, OfflineDqnMlp, General" >&2
    exit 1
    ;;
esac

cd "$ROOT_DIR"

make

if [ ! -x "$ROOT_DIR/uav-wsn-rl1" ]; then
  echo "Simulation executable not found at $ROOT_DIR/uav-wsn-rl1" >&2
  exit 1
fi

exec "$ROOT_DIR/uav-wsn-rl1" -u Cmdenv -c "$CONFIG_NAME" -n .:src omnetpp.ini "$@"
