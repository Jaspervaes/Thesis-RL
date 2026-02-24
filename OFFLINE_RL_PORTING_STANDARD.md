# Offline RL Porting Standard

Universal rulebook for integrating any offline RL algorithm into the SimBank
benchmark. Algorithm-specific implementation notes live in each method's own
`README.md`. This document covers only standards that apply to every method.

---

## Recent Bug Fixes & Changes

Historical record of decisions that shaped the rules below.

### Fix: `print_results` key mismatch (`shared/evaluation.py`)
`evaluate.py` stores the bank baseline under the capitalised key `'Bank'`, but
`print_results` was looking for lowercase `'bank'`, so the relative-gain column was
always empty.

**Change:** In `print_results`, `'bank'` → `'Bank'` for both the lookup and the
comparison guard.

**Rule for porting:** Always use `'Bank'` (capital B) as the key when building the
results dict passed to `print_results`:
```python
results = {'Bank': bank_res, 'Random': random_res, 'YourMethod': method_res}
print_results(results)
```

---

### Fix: Redundant terminal-state zeroing removed (`singleModelCQL/train.py`)
`convert_data.py` already initialises every terminal `next_state` to
`np.zeros(STATE_DIM, dtype=np.float32)` at extraction time.
The identical loop that ran after scaling in `train.py` was doing nothing useful
while iterating over the entire dataset a second time.

**Change:** Removed the post-scaling zeroing loop from `train.py`.

**Rule for porting:** Initialise terminal `next_state` to zeros once, at extraction
time in `convert_data.py`. Do not repeat the operation in `train.py`.

---

### New: `run_experiments.sh` — automated multi-seed pipeline
Automates the full four-step pipeline over the five standard seeds.

```bash
./run_experiments.sh                        # RCT, 10 000 cases
./run_experiments.sh --confounded           # confounded, 10 000 cases
./run_experiments.sh --n_cases 50000        # custom size
./run_experiments.sh --n_cases 50000 --confounded
```

**Eval-seed convention:** The test seed is `99{train_seed}` (e.g. train=42 → eval=9942).
This ensures the evaluation set is always distinct from the training distribution.

---

## Part 1 — Standardisation Rules (apply to every new method)

### 1.1 Shared module — what to import

```
shared/
├── experiment_config.py   # Constants
├── data_utils.py          # Data helpers
├── evaluation.py          # Baselines and evaluation loop
└── __init__.py            # Re-exports everything below
```

```python
from shared import (
    # Constants
    SEEDS, SIZES, DELTAS, STATE_DIM,
    BASE_FEATURES, TRACKED_ACTIVITIES, INTERVENTION_INFO,
    # Data
    get_simbank_params, generate_rct_data, generate_confounded_data,
    split_train_val, count_activities, extract_state, get_ir_action,
    save_pickle, load_pickle,
    # Evaluation
    bank_policy, random_policy, evaluate_policy,
    print_results, print_action_dist,
)
```

---

### 1.2 Standard constants

```python
SEEDS  = [42, 123, 456, 789, 1024]          # always use all five
SIZES  = {'small': 5000, 'medium': 10000,
           'large': 50000, 'full': 100000}
DELTAS = {'rct': 0.0, 'low': 0.5,
           'medium': 0.8, 'high': 0.95}
STATE_DIM = 16                               # 5 base + 11 activity counts
```

---

### 1.3 State representation

**Base features (5 dims) — order matters:**
```python
BASE_FEATURES = ['amount', 'est_quality', 'unc_quality', 'cum_cost', 'elapsed_time']
```

**Activity counts (11 dims) — order matters:**
```python
TRACKED_ACTIVITIES = [
    'initiate_application', 'start_standard',   'start_priority',
    'call_customer',         'email_customer',   'validate_application',
    'contact_headquarters',  'skip_contact',     'calculate_offer',
    'cancel_application',    'receive_acceptance',
]
```

Use the shared helpers — do not reimplement them:
```python
from shared import extract_state, count_activities

state = extract_state(event_row, count_activities(case_group, up_to_index))
# returns np.float32 array of length STATE_DIM (16)
```

---

### 1.4 Data generation

```python
from shared import generate_rct_data, generate_confounded_data, split_train_val

# Pure random-action data
df, params = generate_rct_data(n_cases=10000, seed=42)

# Confounded: 95 % bank policy + 5 % RCT exploration
df, params = generate_confounded_data(n_cases=10000, seed=42, delta=0.95)

# Split 80 / 20 by case_nr (NOT by event row)
df_train, df_val = split_train_val(df, val_ratio=0.2, seed=42)
```

Standard CLI arguments every `generate_data.py` must accept:
```python
parser.add_argument('--n_cases',    type=int,  default=10000)
parser.add_argument('--seed',       type=int,  default=42)
parser.add_argument('--confounded', action='store_true')
```

---

### 1.5 Transition extraction (`convert_data.py`)

#### Transition schema — one flat table per split
```python
{
    'state':             np.float32[16],   # state at decision point
    'action':            int,              # action taken (0/1 or 0/1/2)
    'reward':            float,            # 0.0 intermediate, outcome at terminal
    'next_state':        np.float32[16],   # zeros if terminal
    'terminal':          bool,
    'intervention':      int,              # 0, 1, or 2 (which decision point)
    'next_intervention': int,              # 0/1/2 for next, -1 if terminal
}
```

#### Trajectory types
```
1. Full:          Int0 → Int1 → Int2 → Terminal
2. Skip Int1:     Int0 → Int2 → Terminal  (priority path skips contact_HQ)
3. Terminal@Int1: Int0 → Int1 → Terminal
4. Terminal@Int0: Int0 → Terminal
```

#### Intervention indexing
| Index | Name | Actions |
|-------|------|---------|
| 0 | choose_procedure | 0=standard, 1=priority |
| 1 | contact_headquarters | 0=contact, 1=skip |
| 2 | set_ir_3_levels | 0=7%, 1=8%, 2=9% |

#### Terminal `next_state` must be zeroed at extraction time
```python
terminal = np.zeros(STATE_DIM, dtype=np.float32)

transitions.append({
    ...,
    'next_state': terminal,   # zeros here, not in train.py
    'terminal': True,
    'next_intervention': -1,
})
```

**Do not repeat this zeroing in `train.py`.**

#### Interest-rate action helper
```python
from shared import get_ir_action
action3 = get_ir_action(row.get('interest_rate', 0.08))
# 0.07 → 0,  0.08 → 1,  0.09 → 2
```

---

### 1.6 Scaling (`train.py`)

Fit the scaler on training states only, then apply to all splits:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_states = np.vstack(df_train['state'].values)
scaler.fit(train_states)

df_train['state']      = list(scaler.transform(train_states))
df_train['next_state'] = list(scaler.transform(np.vstack(df_train['next_state'].values)))
df_val['state']        = list(scaler.transform(np.vstack(df_val['state'].values)))
df_val['next_state']   = list(scaler.transform(np.vstack(df_val['next_state'].values)))
```

Save the fitted scaler alongside the model weights so evaluation can reuse it:
```python
torch.save({'model': model.state_dict(), 'scaler': scaler, 'config': {...}}, path)
```

---

### 1.7 Evaluation (`evaluate.py`)

```python
from shared import bank_policy, random_policy, evaluate_policy, print_results

bank_res   = evaluate_policy(bank_policy,   n_episodes, params, seed)
random_res = evaluate_policy(random_policy, n_episodes, params, seed)
method_res = evaluate_policy(
    your_policy, n_episodes, params, seed,
    use_prefix=True, reset_fn=your_policy.reset,
)

# Key names are case-sensitive — 'Bank' must be capitalised
results = {'Bank': bank_res, 'Random': random_res, 'YourMethod': method_res}
print_results(results)
print_action_dist(results)
```

**Eval seed must differ from train seed.**
Convention: `eval_seed = int("99" + str(train_seed))` → e.g. 42 → 9942.

Standard CLI arguments every `evaluate.py` must accept:
```python
parser.add_argument('--n_cases',    type=int, default=10000)
parser.add_argument('--confounded', action='store_true')
parser.add_argument('--n_episodes', type=int, default=1000)
parser.add_argument('--seed',       type=int, default=1042)  # eval seed
```

---

### 1.8 Hyperparameters

Keep all method-specific hyperparameters in `train.py` argparse. Tune on the
validation set only — never on the test simulation.

If two methods use the same model type (e.g. both MLP), use the same search
space for fair comparison.

---

### 1.9 Multi-seed automation

Use `run_experiments.sh` as the template for any new method's automation script.
Key points:
- Loop over `SEEDS=(42 123 456 789 1024)`
- Pass `--seed $SEED` to `generate_data.py` and `train.py`
- Pass `--seed 99$SEED` to `evaluate.py`
- Accept `--n_cases` and `--confounded` flags at the top level
- Use `set -euo pipefail` so failures are not silently swallowed

---

## Part 2 — Coding Style & Bloat Reduction Guidelines

These rules apply to every new method ported to the benchmark. The goal is code
that is strictly business: minimal, readable, and free of tutorial-style scaffolding.

### 2.1 Architectural preservation vs. code bloat

**Preserve the RL methodology exactly.** Never alter network architecture, training
order, or loss functions to make the code shorter. If a method requires three
Q-networks and reverse-order training, keep all three networks and the reverse order.

**Aggressively remove structural bloat.** Consolidate all logic into the standard
4-file pipeline. Deprecate any redundant config files, monolithic pipeline scripts,
or helper modules whose functions are already covered by `shared/`.

```
your_method/
├── generate_data.py   # data only — no transitions
├── convert_data.py    # transitions only — no training
├── train.py           # training only — no evaluation
└── evaluate.py        # evaluation only — no training
```

---

### 2.2 Comments and documentation

Strip all verbose, tutorial-style inline comments and large block-comment banners.
Use one-line docstrings; let the code speak for itself.

```python
# Bad — explains what the next line obviously does
# Soft update target network
for p, tp in zip(q.parameters(), qt.parameters()):
    tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

# Good — no comment needed; the code is self-evident
for p, tp in zip(q.parameters(), qt.parameters()):
    tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
```

```python
# Bad — multi-line docstring for a trivial function
def make_loader(df, int_idx, batch_size, shuffle=True):
    """
    Create a DataLoader for a specific intervention index.
    Filters the dataframe, wraps it in a Dataset, and returns a DataLoader.
    Returns None if the subset is empty.
    """

# Good
def make_loader(df, int_idx, batch_size, shuffle=True):
    """DataLoader for one intervention subset, or None if empty."""
```

Remove section banners entirely:
```python
# Bad
# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

# Good — just write the function
```

---

### 2.3 Clean terminal output

Remove step-by-step phase announcements, trajectory-type counters, action
distribution tables printed during training, and "Next: python ..." hints.

Terminal output during training should contain only:
- A single header line summarising key hyperparameters
- Per-phase identifier (e.g. `[Q3]`, `[Q2]`, `[Q1]`)
- Epoch loss lines, printed every N epochs: `[epoch/total] train=X  val=Y`
- A final `[OK] path/to/model.pth` confirmation

```python
# Bad — noisy training output
print(f"\n[Phase 1] Training Q3 (Int2 — interest rate, {args.epochs} epochs)...")
print(f"  Best Q3 val loss: {bv3:.4f}")
print(f"  No Int2 transitions — skipping Q3 training.")

# Good
print("\n[Q3]")
```

---

### 2.4 Leverage `shared/` — never reimplement

If a function exists in `shared/`, import it. Do not write a local version.

| Forbidden reimplementation | Import instead |
|----------------------------|----------------|
| Custom `bank_policy` | `from shared import bank_policy` |
| Custom `random_policy` | `from shared import random_policy` |
| Custom `evaluate_policy` loop | `from shared import evaluate_policy` |
| Custom `print_actions` / results table | `from shared import print_results, print_action_dist` |
| Custom `extract_base_features` + activity list | `from shared import extract_state, count_activities` |
| Custom data generation with manual `simulation` calls | `from shared import generate_rct_data, generate_confounded_data` |
| Custom train/val split | `from shared import split_train_val` |

Custom logic should be strictly reserved for the specific algorithm's internal
mechanics: network architecture, loss function, and training loop structure.

---

## Part 3 — Porting Checklist

When adding a new offline RL method, verify every item below:

**Data & transitions**
- [ ] Import `generate_rct_data` / `generate_confounded_data` from `shared`
- [ ] Split 80/20 by `case_nr` (not by row) via `split_train_val`
- [ ] Only extract transitions for interventions that actually occurred
- [ ] Intermediate rewards = 0; terminal reward = outcome
- [ ] Terminal `next_state` zeroed in `convert_data.py` (not `train.py`)
- [ ] `next_intervention` = 0/1/2 for next step, −1 at terminal
- [ ] Use `extract_state` + `count_activities` from `shared` for consistent 16-dim states

**Training**
- [ ] Scaler fitted on train states only, applied to train + val `next_state` too
- [ ] Scaler saved with model checkpoint
- [ ] `--seed` argument sets `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`
- [ ] Hyperparameters tunable via argparse; tuned on val loss

**Evaluation**
- [ ] Results dict uses `'Bank'` (capital B) as key
- [ ] Eval seed ≠ train seed (convention: `99{train_seed}`)
- [ ] Compare against `bank_policy` and `random_policy` from `shared`
- [ ] Call `print_results` and `print_action_dist`

**Style**
- [ ] No section-comment banners
- [ ] No tutorial-style inline comments
- [ ] One-line module and function docstrings
- [ ] No "Next: python ..." print hints
- [ ] No post-training action-distribution prints in `train.py`

**Reproducibility**
- [ ] Accepts `--n_cases`, `--confounded`, `--seed` arguments
- [ ] Five seeds: 42, 123, 456, 789, 1024
- [ ] Report mean ± std over all five seeds

---

## Part 4 — Quick-Start Commands

```bash
# Single run (RCT)
python singleModelCQL/generate_data.py --n_cases 10000 --seed 42
python singleModelCQL/convert_data.py  --n_cases 10000
python singleModelCQL/train.py         --n_cases 10000 --seed 42 --epochs 50
python singleModelCQL/evaluate.py      --n_cases 10000 --seed 9942

# Single run (confounded)
python singleModelCQL/generate_data.py --n_cases 10000 --seed 42 --confounded
python singleModelCQL/convert_data.py  --n_cases 10000 --confounded
python singleModelCQL/train.py         --n_cases 10000 --seed 42 --confounded --epochs 50
python singleModelCQL/evaluate.py      --n_cases 10000 --confounded --seed 9942

# Full five-seed run
chmod +x run_experiments.sh
./run_experiments.sh --n_cases 10000
./run_experiments.sh --n_cases 10000 --confounded
```
