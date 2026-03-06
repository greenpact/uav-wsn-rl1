# UAV-WSN-RL1 Simulation Project

This repository contains an OMNeT++ simulation for UAV-assisted wireless sensor network routing with TR-LAR and optional offline DQN policy inference.

## What was missing and fixed

The project code is present, but simulation startup typically fails in fresh environments because OMNeT++ runtime/build environment variables are not loaded.

Added setup/run helpers:

- `scripts/omnetpp_env.sh`: validates and sources OMNeT++ `setenv`
- `scripts/run_simulation.sh`: builds and runs a selected OMNeT++ config in `Cmdenv`
- `simulations/run`: now attempts to load `OMNETPP_ROOT/setenv` and prints actionable diagnostics

## Prerequisites

1. OMNeT++ installed locally (6.x recommended)
2. `OMNETPP_ROOT` set to installation path
3. C++ build tools (`g++`, `make`)

Example:

```bash
export OMNETPP_ROOT=/opt/omnetpp-6.0
. "$OMNETPP_ROOT/setenv"
```

## Build and run

From repository root:

```bash
. scripts/omnetpp_env.sh
make
./uav-wsn-rl1 -u Cmdenv -c QuickTest -n .:src omnetpp.ini
```

Or with the helper script:

```bash
. scripts/omnetpp_env.sh
CONFIG_NAME=QuickTest scripts/run_simulation.sh
```

The project uses a persistent virtual environment at `.venv` for training and plotting:

```bash
. .venv/bin/activate
```

Supported `CONFIG_NAME` values:

- `QuickTest`
- `General`
- `OfflineDqnPolicy`
- `OfflineDqnMlp`

## Offline policy training

```bash
python3 -m pip install -r training/requirements.txt
scripts/train_offline_policy.sh --episodes 200 --max-steps 80
```

## Plot simulation results

```bash
. .venv/bin/activate
python scripts/plot_results.py
```

Plots are written to `results/plots/`.

Generated policy artifacts are saved under `models/` and can be selected via `omnetpp.ini` configs.

## Troubleshooting

- `opp_configfilepath: No such file or directory`
  - OMNeT++ toolchain is not on `PATH`; source `scripts/omnetpp_env.sh`
- `liboppcmdenv.so: cannot open shared object file`
  - OMNeT++ runtime library path is not loaded; source OMNeT++ `setenv`
- NED type resolution errors
  - Ensure `-n .:src` is present (already included in provided scripts)
