# Single-Model CQL

## Architecture

A **single unified Q-network** handles all three interventions. The intervention
identity is passed as a 3-dim one-hot vector concatenated to the state, so the
network learns a shared representation across all decision points while still
producing intervention-specific Q-values.

| Dimension | Value |
|-----------|-------|
| State input | 16 (5 base features + 11 activity counts) |
| Intervention encoding | 3-dim one-hot concatenated to state |
| Total input | 19 |
| Output | 3 Q-values (masked to valid actions per intervention) |
| Hidden layers | [256, 256] with ReLU + LayerNorm |

Valid actions per intervention: `[2, 2, 3]` — invalid logits are set to `-inf`
before the argmax so the network never selects an out-of-range action.

## File layout

```
singleModelCQL/
├── generate_data.py   # RCT or confounded data generation
├── convert_data.py    # Raw events → flat transition table
├── train.py           # Unified Q-network training
└── evaluate.py        # Simulator-based evaluation
```

## Quick-start

```bash
# RCT
python singleModelCQL/generate_data.py --n_cases 10000 --seed 42
python singleModelCQL/convert_data.py  --n_cases 10000
python singleModelCQL/train.py         --n_cases 10000 --seed 42 --epochs 50
python singleModelCQL/evaluate.py      --n_cases 10000 --seed 9942

# Confounded
python singleModelCQL/generate_data.py --n_cases 10000 --seed 42 --confounded
python singleModelCQL/convert_data.py  --n_cases 10000 --confounded
python singleModelCQL/train.py         --n_cases 10000 --seed 42 --confounded --epochs 50
python singleModelCQL/evaluate.py      --n_cases 10000 --confounded --seed 9942

# Full five-seed run
./run_experiments.sh --n_cases 10000
./run_experiments.sh --n_cases 10000 --confounded
```

## Data files produced

| File | Contents |
|------|----------|
| `data/single_cql_{RCT\|CONF}_{n}_raw.pkl` | Raw event log DataFrame |
| `data/single_cql_{RCT\|CONF}_{n}_params.pkl` | SimBank simulation params |
| `data/single_cql_{RCT\|CONF}_{n}_trans_train.pkl` | Training transitions |
| `data/single_cql_{RCT\|CONF}_{n}_trans_val.pkl` | Validation transitions |
| `models/single_cql_{RCT\|CONF}_{n}.pth` | Checkpoint (model + scaler) |
