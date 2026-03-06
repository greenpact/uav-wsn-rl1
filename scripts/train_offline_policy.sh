#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/.."

if [ -f ".venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  # Activation scripts may reference optional shell vars; relax nounset briefly.
  set +u
  . ".venv/bin/activate"
  set -u
fi

python3 training/train_offline_dqn.py \
  --episodes "${EPISODES:-20000}" \
  --max-steps "${MAX_STEPS:-240}" \
  --output-dir "${OUTPUT_DIR:-models}" \
  "$@"
