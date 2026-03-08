# Offline DQN Comprehensive Workflow (Training + Validation)

This document defines the publication-grade workflow for offline DQN in `uav-wsn-rl1`.

It is designed to satisfy scientific rigor, reproducibility, and clean separation between:
- training/data pipeline (`training/` + analysis scripts), and
- runtime routing protocol (`src/routing/` in OMNeT++).

## 1. Scope and Objective

Goal: train a routing Q-function offline and deploy it in OMNeT++ inference-only mode, then validate claims with robust statistics.

This workflow assumes an 8-dimensional feature vector:
- `[energy, queue, density, tau, linkQuality, neighborEnergy, neighborFailure, uavDistance]`

All components must keep this feature order and dimensionality identical.

## 2. Boundary and Integration Contract

## 2.1 Training/Data Module Responsibilities

Training/data module (Python side) is responsible for:
- collecting fixed offline datasets from OMNeT++ runs,
- building train/val/test splits,
- training offline Q model from dataset only (no online env stepping during optimization),
- exporting model artifacts in runtime formats (`.mdl` and `.qtab`),
- generating metadata and validation reports.

Training/data module must not:
- call or modify runtime forwarding decisions during simulation,
- perform gradient updates inside OMNeT++ runtime.

## 2.2 Runtime Routing Module Responsibilities

Runtime module (`TrlarRouting` + `RLAgentInterface`) is responsible for:
- producing decision-time feature vectors,
- loading exported model artifacts,
- scoring candidate actions at inference time,
- logging transitions/decisions for dataset generation,
- never training or updating NN weights.

Runtime module must not:
- retrain the model online,
- silently accept incompatible model schema/dimensions.

## 2.3 Interface Contract (Hard Requirements)

1. Feature dimension and ordering are versioned and immutable per schema version.
2. Exported model metadata includes:
- `schema_version`
- `feature_dim`
- `feature_order`
- training dataset ID/hash
- code commit hash
3. Runtime loader validates `feature_dim == runtime_feature_dim`.
4. On mismatch, runtime must fail fast in evaluation mode (do not silently fallback for scientific runs).

## 3. Data Contract for Offline Dataset

Use JSONL transitions with one line per decision transition.

Required fields:
- `run_id`: string
- `seed`: int
- `config_name`: string
- `t`: float
- `node_id`: int
- `packet_id`: string
- `state`: array[8]
- `candidates`: array[array[8]]
- `action_index`: int
- `action_features`: array[8]
- `reward`: float
- `next_state_context`: array[8] or null
- `next_candidates`: array[array[8]]
- `done`: bool
- `event_quality`: object (ack, delay, drop, retries)

Recommended extra fields:
- `scenario`: object (numNodes, area, traffic interval, UAV params)
- `policy_tag`: string (`baseline`, `explore_eps_0.2`, etc.)
- `schema_version`: int

Example (single line):
```json
{"run_id":"general_seed_07","seed":7,"config_name":"General","t":42.13,"node_id":133,"packet_id":"12-334","state":[0.73,0.19,0.31,0.55,0.82,0.70,0.12,0.44],"candidates":[[...],[...]],"action_index":1,"action_features":[0.73,0.19,0.31,0.55,0.79,0.66,0.21,0.49],"reward":0.84,"next_state_context":[0.72,0.22,0.30,0.52,0.00,0.00,0.00,0.46],"next_candidates":[[...],[...]],"done":false,"event_quality":{"ack":1,"delay":0.11,"drop":0,"retries":0},"policy_tag":"baseline","schema_version":1}
```

## 4. End-to-End Workflow

## Phase A: Instrument and Freeze

1. Record environment and code state:
- `git rev-parse HEAD`
- OMNeT++ version
- Python package versions
2. Freeze feature schema:
- document 8 features and order
- add schema version constant in training + runtime
3. Add strict validation checks in runtime loader.

Gate A (must pass):
- feature schema documented and identical across modules.

## Phase B: Offline Dataset Collection (From OMNeT++)

1. Run behavior policies to generate broad support data:
- baseline heuristic (`General`)
- exploratory variants (stochastic tie-breaking/epsilon-like behavior)
2. Collect >= 30 seeds minimum per condition; more for final paper.
3. Save raw transitions with scenario metadata.

Gate B:
- no missing required fields,
- non-zero rewards and action diversity,
- coverage report per feature dimension and action count.

## Phase C: Dataset QA and Split

1. Build deterministic split by seed/scenario:
- train seeds
- validation seeds
- held-out test seeds
2. Prevent leakage across splits.
3. Publish dataset manifest:
- sample counts
- feature min/max/mean/std
- policy composition percentages

Gate C:
- leakage check = pass,
- split manifest saved,
- dataset hash recorded.

## Phase D: Offline Training (Dataset-Only)

1. Train Q(s,a) from fixed dataset without live env stepping.
2. Use conservative offline regularization (recommended):
- CQL-style penalty against unseen actions.
3. Tune hyperparameters on validation split only.
4. Save checkpoints and training curves.

Gate D:
- stable training curves,
- no NaN/infinite values,
- validation objective improves.

## Phase E: Artifact Export and Compatibility Tests

1. Export:
- `offline_dqn_policy.mdl`
- `offline_qtable.qtab`
- `offline_training_summary.json`
2. Add metadata fields:
- schema version
- feature dim/order
- dataset hash
- training commit hash
3. Run compatibility unit test:
- runtime can load and score one test vector.

Gate E:
- dimensions match runtime exactly,
- loader test passes,
- metadata complete.

## Phase F: OMNeT++ Integration and Scientific Evaluation

1. Compare at least:
- Baseline (`General`)
- OfflineDqnPolicy
- OfflineDqnMlp
2. Keep all non-policy parameters fixed.
3. Use paired seeds across methods.
4. Collect primary metrics:
- PDR
- mean/median/p95 delay
- first-node-death time
- drop rate

Gate F:
- all runs completed,
- no missing result files,
- per-seed paired results table available.

## Phase G: Statistics and Reporting

1. Report per metric and condition:
- mean, std, median, IQR, 95% CI
2. Inferential tests:
- paired t-test if assumptions hold, otherwise Wilcoxon signed-rank
3. Report effect size (not p-value only).
4. Correct for multiple comparisons.

Gate G:
- statistical report reproducible from scripts,
- claim statements tied to tested hypotheses.

## Phase H: Sensitivity and Ablation

Required sensitivity sweeps:
- node count
- traffic intensity
- UAV speed/contact window

Required ablations:
- disable local bias adaptation
- remove UAV-distance feature (or other key feature)
- no conservative penalty in offline training

Gate H:
- conclusions remain directionally consistent.

## 5. Implementation Mapping for This Repository

Recommended additions:
- `training/build_offline_dataset.py`
- `training/train_offline_dqn_from_dataset.py`
- `training/evaluate_offline_policy.py`
- `training/stats_report.py`
- `training/schemas/transition_schema_v1.json`

Recommended runtime updates:
- `src/routing/TrlarRouting.cc`: enrich transition logging to full transition tuples
- `src/routing/RLAgentInterface.cc`: strict schema/dimension checks and explicit scientific-mode failure

Recommended outputs:
- `models/offline_training_summary.json` (extended metadata)
- `results/analysis/metrics_by_seed.csv`
- `results/analysis/stats_report.md`

## 6. Publication-Readiness Checklist

1. Model and runtime feature dimensions match exactly (8D).
2. Offline training uses fixed OMNeT++ dataset, not online env interaction.
3. Train/val/test split is seed-disjoint and documented.
4. >= 30 seeds per condition, paired across methods.
5. Baselines include at least one strong heuristic beyond no-model.
6. Statistical tests + effect sizes + multiple-comparison handling included.
7. Sensitivity and ablation studies completed.
8. Full reproducibility package archived (commands, seeds, hashes, artifacts).

## 7. Command Skeleton (Operational)

```bash
# A) collect trajectories
CONFIG_NAME=General scripts/run_simulation.sh
# (repeat across seeds/conditions, output transitions per run)

# B) build offline dataset
python3 training/build_offline_dataset.py \
  --input results/transitions_raw \
  --output results/datasets/offline_dataset_v1.jsonl \
  --split-manifest results/datasets/split_manifest_v1.json

# C) train from fixed dataset
python3 training/train_offline_dqn_from_dataset.py \
  --dataset results/datasets/offline_dataset_v1.jsonl \
  --split-manifest results/datasets/split_manifest_v1.json \
  --output-dir models

# D) run policy evaluation configs
CONFIG_NAME=General scripts/run_simulation.sh
CONFIG_NAME=OfflineDqnPolicy scripts/run_simulation.sh
CONFIG_NAME=OfflineDqnMlp scripts/run_simulation.sh

# E) statistical report
python3 training/stats_report.py \
  --results-root results \
  --output results/analysis/stats_report.md
```

## 8. Non-Negotiable Scientific Rules

1. No changing model between compared policy runs.
2. No cherry-picking seeds or runs.
3. No silent fallback to heuristic in scientific evaluation mode.
4. Any protocol change increments schema/version and triggers rerun.

---

If this workflow is followed, the DQN module and routing protocol stay cleanly decoupled while remaining fully integrated through explicit, testable interfaces and reproducible artifacts.
