#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
INI_FILE="${INI_FILE:-omnetpp_offline.ini}"
RUNS="${RUNS:-30}"
CONFIGS_CSV="${CONFIGS:-OfflineData_BehaviorUtility,OfflineData_BehaviorStochastic,OfflineData_AblationNoUav}"
EXTRA_SIM_ARGS="${EXTRA_SIM_ARGS:-}"

IFS=',' read -r -a CONFIGS <<< "$CONFIGS_CSV"

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "No configs provided in CONFIGS" >&2
  exit 1
fi

for config in "${CONFIGS[@]}"; do
  config="$(echo "$config" | xargs)"
  [[ -z "$config" ]] && continue
  echo "[offline-collect] running config=$config runs=$RUNS ini=$INI_FILE"

  case "$config" in
    OfflineData_BehaviorUtility) EXP_DIR="behavior-utility" ;;
    OfflineData_BehaviorStochastic) EXP_DIR="behavior-stochastic" ;;
    OfflineData_AblationNoUav) EXP_DIR="ablation-no-uav" ;;
    *) EXP_DIR="" ;;
  esac

  for ((r=0; r<RUNS; r++)); do
    echo "[offline-collect] config=$config repetition=$r"
    (
      cd "$ROOT_DIR"
      if [[ -n "$EXP_DIR" ]]; then
        rm -rf "results/offline_dataset/${EXP_DIR}/run-${r}" || true
      fi
      export UAVWSN_GIT_COMMIT="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
      export UAVWSN_OMNET_VERSION="$(opp_run -V 2>/dev/null | head -n 1 || echo unknown)"
      export UAVWSN_CONFIG_HASH="$(sha256sum "$INI_FILE" | awk '{print $1}')"
      INI_FILE="$INI_FILE" CONFIG_NAME="$config" ./scripts/run_simulation.sh -r "$r" ${EXTRA_SIM_ARGS}
    )
  done
done

echo "[offline-collect] done"
