# Offline DQN Training (Base Station)

This folder contains the centralized offline training pipeline for UAV-assisted WSN routing.

## What it does

- Builds a digital-twin routing environment with domain randomization.
- Trains a DQN policy centrally at the base station (no on-node learning).
- Exports two artifacts for simulation-time inference:
  - `offline_dqn_policy.mdl` (MLP model)
  - `offline_qtable.qtab` (discretized Q-table)

Runtime sensor nodes only perform inference (`Q(s,a) + bias`) and do not run gradient updates.

## Install dependencies

```bash
python3 -m pip install -r training/requirements.txt
```

## Run training

```bash
python3 training/train_offline_dqn.py \
  --episodes 20000 \
  --max-steps 240 \
  --output-dir models
```

Quick smoke run:

```bash
python3 training/train_offline_dqn.py \
  --episodes 200 \
  --max-steps 80 \
  --output-dir models
```

## Main hyperparameters

- `--gamma` default `0.95`
- `--learning-rate` default `1e-3`
- `--replay-size` default `100000`
- `--batch-size` default `64`
- `--target-update-steps` default `1000`
- `--qtable-bins` default `10,10,11,10,10,10,10,10`

## Publication-grade workflow

For a comprehensive offline-DQN data, training, validation, and integration protocol
with strict module boundaries between training and runtime routing, see:

- `training/OFFLINE_DQN_WORKFLOW.md`

## Export format

### MLP model (`.mdl`)

Text format consumed by `src/routing/RLAgentInterface`:

- `layers <n>`
- repeated per layer:
  - `layer <out> <in>`
  - `weights ...`
  - `bias ...`
- optional normalization block:
  - `norm <dim>`
  - `mean ...`
  - `std ...`

### Q-table model (`.qtab`)

Text format:

- `format qtable_v1`
- `dims <d>`
- `bins ...`
- `state_min ...`
- `state_max ...`
- `default <q>`
- `entries <n>`
- repeated lines: `<i0> <i1> ... <id-1> <q>`

## Integrate with OMNeT++

Use config `OfflineDqnPolicy` in `omnetpp.ini`, which points to:

- `**.routing[*].modelFile = "models/offline_qtable.qtab"`

You can also point to the MLP export (`offline_dqn_policy.mdl`) if desired.

