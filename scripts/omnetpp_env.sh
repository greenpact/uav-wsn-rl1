#!/usr/bin/env bash
# Source this file to configure OMNeT++ tools and runtime libraries.

set -eu

if [ -z "${OMNETPP_ROOT:-}" ] && [ -n "${OMNETPP_HOME:-}" ]; then
  OMNETPP_ROOT="$OMNETPP_HOME"
fi

if [ -z "${OMNETPP_ROOT:-}" ]; then
  cat >&2 <<'EOF'
OMNETPP_ROOT is not set.
Set it to your OMNeT++ installation path, for example:
  export OMNETPP_ROOT=/opt/omnetpp-6.0
Then source this script again.
EOF
  return 1 2>/dev/null || exit 1
fi

if [ ! -f "$OMNETPP_ROOT/setenv" ]; then
  echo "OMNeT++ setenv file not found at: $OMNETPP_ROOT/setenv" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck source=/dev/null
set +u
source "$OMNETPP_ROOT/setenv" -q
set -u

echo "OMNeT++ environment loaded from: $OMNETPP_ROOT"
command -v opp_run >/dev/null 2>&1 && echo "opp_run: $(command -v opp_run)"
command -v opp_configfilepath >/dev/null 2>&1 && echo "opp_configfilepath: $(command -v opp_configfilepath)"
