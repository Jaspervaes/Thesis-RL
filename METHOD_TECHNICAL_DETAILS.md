# Technical Reference: All Offline RL Methods for Prescriptive Process Monitoring

Exhaustive technical detail per method, intended as raw material for thesis writing.
All details are extracted directly from the codebase. Project root: `SimBank-main/`.

---

## Repository Layout

```
SimBank-main/
├── methods/
│   ├── shared/                  — shared data utilities, evaluation harness, LSTM building blocks
│   ├── K-Means-FQI/             — tabular offline RL (K-means + FQI)
│   ├── LSTM-DQN/                — sequential offline DQN baseline
│   ├── RIMS-DQN/                — online DQN in a learned simulator
│   ├── CQL-SN/                  — single-network conservative Q-learning
│   ├── CQL-MN/                  — multi-network conservative Q-learning
│   ├── LSTM-DQN-GBR/            — causal hybrid: GBR S-learner + LSTM-DQN
│   ├── LSTM-DQN-SLearner/       — causal hybrid: LSTM S-learner + LSTM-DQN
│   ├── LSTM-DQN-TabPFN/         — causal hybrid: TabPFN S-learner + LSTM-DQN
│   └── LSTM-DQN-DragonNet/      — causal hybrid: DragonNet + LSTM-DQN
├── data/                        — generated data files (gitignored)
├── models/                      — trained checkpoints (gitignored)
├── results/                     — JSON result files + figures
├── generate_data.py             — root-level entry point → delegates to methods/shared/generate_data.py
├── run_all_steps.py             — orchestrate all 9 methods × steps × conditions × seeds
├── run_seeds.py                 — run one method × all 5 seeds
├── retrain_all_3step.py         — retrain all 9 methods for --steps 3 specifically
├── run_intervention_combos.py   — LSTM-DQN joint vs subset training comparison
├── plot_results.py              — generate thesis figures from all_results.json
└── SimBank-main/SimBank/        — SimBank process simulator (vendored)
```

Each method directory has four scripts: `generate_data.py`, `convert_data.py`, `train.py`, `evaluate.py`.

---

## Shared Infrastructure

### State Representations

Two state representations are used across the codebase. Most methods use only the sequential one; the flat one appears only in K-Means-FQI (which has no LSTM at all) and as the S-learner input of two causal hybrids (LSTM-DQN-GBR, LSTM-DQN-TabPFN). In every causal hybrid the downstream DQN that selects actions is sequential — so policy inference is sequential everywhere except K-Means-FQI.

**Prefix-based (sequential) state — used by the policy/Q-network in every method except K-Means-FQI** (LSTM-DQN, RIMS-DQN, CQL-SN, CQL-MN, and the Phase-3 DQN of all four causal hybrids):
- A variable-length sequence of events from the start of a case up to (not including) the intervention point
- Each timestep has 7 features: 6 continuous features + 1 activity identifier
- Continuous features (`FEATURE_COLS` in `methods/shared/lstm_utils.py`): `amount`, `est_quality`, `unc_quality`, `interest_rate`, `cum_cost`, `elapsed_time`
- Activity identifier: either integer index (integer encoding, default) or one-hot vector (onehot encoding)
- Sequences are padded to `max_len` (set from training-set max prefix length) and packed with `pack_padded_sequence` for efficient LSTM processing
- Per-feature normalisation: mean/std computed from training prefixes; `feat_means` and `feat_stds` saved in the checkpoint via `build_vocab_and_stats()`

**Flat state vector — K-Means-FQI policy + the causal S-learner input of LSTM-DQN-GBR and LSTM-DQN-TabPFN** (not used as DQN input in any method):
- Fixed 17-dimensional vector: 5 base features + 12 activity-count features
- Base features (5, `BASE_FEATURES` in `methods/shared/experiment_config.py`): `amount`, `est_quality`, `unc_quality`, `cum_cost`, `elapsed_time`
- Activity counts (12, `TRACKED_ACTIVITIES`): `initiate_application`, `start_standard`, `start_priority`, `call_customer`, `email_customer`, `validate_application`, `contact_headquarters`, `skip_contact`, `calculate_offer`, `cancel_application`, `receive_acceptance`, `receive_refusal`
- `STATE_DIM = len(BASE_FEATURES) + len(TRACKED_ACTIVITIES) = 5 + 12 = 17`
- Built by `extract_state(event, activity_counts)` in `methods/shared/data_utils.py`

Note: `FEATURE_COLS` (per-timestep features for LSTM methods) includes `interest_rate` and uses `[amount, est_quality, unc_quality, interest_rate, cum_cost, elapsed_time]` (6 features). The flat state intentionally drops `interest_rate` and replaces it with `elapsed_time` as a base feature, because at intervention 0 / intervention 1 the interest rate is undefined; intervention 2's interest-rate decision is the action itself.

| Method | Policy/Q-network state | Causal model state |
|--------|------------------------|--------------------|
| LSTM-DQN, RIMS-DQN | sequential prefix | — |
| K-Means-FQI | flat 17-dim | — |
| CQL-SN, CQL-MN | sequential prefix | — |
| LSTM-DQN-GBR | sequential prefix (DQN) | flat 17-dim (GBR) |
| LSTM-DQN-SLearner | sequential prefix (DQN) | sequential prefix (LSTM S-learner) |
| LSTM-DQN-TabPFN | sequential prefix (DQN) | flat 17-dim (TabPFN) |
| LSTM-DQN-DragonNet | sequential prefix (DQN) | sequential prefix (LSTM-DragonNet) |

### Intervention Points (SimBank)

Three sequential intervention points in every case:
- **Intervention 0** — `choose_procedure`: 2 actions (`start_standard`=0, `start_priority`=1)
- **Intervention 1** — `time_contact_HQ`: 2 actions (`contact_headquarters`=0, `skip_contact`=1)
- **Intervention 2** — `set_ir_3_levels`: 3 actions (low IR=0, medium IR=1, high IR=2)

`N_ACTIONS = [2, 2, 3]`

Not every case reaches all three interventions. Conversion scripts handle all branching paths: only-int0, int0→int1, int0→int2, int0→int1→int2.

### Backward TD Bootstrapping

Methods that use Q-learning train in reverse intervention order: Q₃ first, then Q₂ using Q₃ targets, then Q₁ using Q₂/Q₃ targets. This is necessary because rewards are only observed at the final intervention (terminal transition); intermediate transitions have reward=0.

The Q₁ target depends on the `next_intervention` field:
- If next is intervention 1: `γ · max_a Q₂(s', a)`
- If next is intervention 2: `γ · max_a Q₃(s', a)`
- If terminal: `norm(r)` (no bootstrap)

### Data Generation — quick reference

- `generate_rct_data(n_cases, seed)` — RCT: actions assigned uniformly at random by the simulator
- `generate_confounded_data(n_cases, seed, delta)` — confounded: mixes bank-policy and RCT logs at fraction delta (default 0.95) via `confounding_level.set_delta()`
- Both functions are in `methods/shared/data_utils.py`
- Output: raw DataFrame of event-log rows + params dict
- Saved to: `data/simbank_{RCT|CONF}_{n_cases}_raw.pkl` and `data/simbank_{RCT|CONF}_{n_cases}_params.pkl`

---

## Method 1: LSTM-DQN (`methods/LSTM-DQN/`)

### RL Paradigm
Offline Q-learning (DQN variant) with separate Q-networks per intervention point. Uses backward TD bootstrapping (Q₃→Q₂→Q₁). Core sequence-based offline RL baseline.

### Network Architecture (LSTM_DQN in `methods/shared/lstm_utils.py`)
```
Input per timestep: [activity_embedding (32-dim)] + [6 continuous features] = 38-dim  (integer mode)
                    [activity_onehot (n_activities-dim)] + [6 continuous features]      (onehot mode)

LSTM: input_size=38, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2 (between layers)
  → packed sequences, returns final hidden state h_n[-1] (last layer): 128-dim

FC head:
  Linear(128 → 128) → ReLU → Dropout(0.2) → Linear(128 → n_actions)
  n_actions = 2 for interventions 0 and 1; 3 for intervention 2
```

Three separate `LSTM_DQN` instances (Q₁, Q₂, Q₃), each with its own target network and replay buffer.

### Target Networks
Soft (Polyak) updates after every training step:
```
θ_target ← τ · θ_online + (1 − τ) · θ_target     τ = 0.005
```
Target networks are set to `.eval()` mode permanently (disables dropout during target computation).

### Data Conversion (`methods/LSTM-DQN/convert_data.py`)
Each case produces (prefix, action, reward, next_prefix, terminal, intervention, next_intervention) tuples. `prefix` is the list of event dicts from case start up to (not including) the intervention row. `reward = outcome` for terminal transitions, `0.0` otherwise. The train/val split (80/20) is by `case_nr` via `split_train_val(df, val_ratio=0.2, seed)`.

### Training Procedure
```
Optimizer: Adam, lr=1e-3, weight_decay=1e-5
Loss: MSE(Q_predicted, Q_target)
LR scheduler: ReduceLROnPlateau(factor=0.5, patience=5)
Batch size: 256
Training order: Q₃ first → Q₂ → Q₁ (backward TD)
Target update: soft Polyak, τ=0.005, after every step
Gamma (discount): 0.99
Reward normalisation: norm(r) = (r − r_mean) / (r_std + 1e-8), computed from terminal rewards
Early stopping: patience=10 epochs on validation loss, es_delta=1e-4
Epochs: up to 50
Grad clip: L2-norm 1.0
```

Q-target computation:
```python
# Q₃ (terminal reward only):
target = norm(reward)

# Q₂:
next_q = Q₃t(next_prefix).max(dim=1).values
target = term * norm(r) + (1 − term) * γ * next_q

# Q₁ (routes by next_intervention):
m1 = non_terminal & (ni == 1)
m2 = non_terminal & (ni == 2)
t = term * norm(r)
t[m1] = γ * Q₂t(next_prefix[m1]).max(1)[0]
t[m2] = γ * Q₃t(next_prefix[m2]).max(1)[0]
```

### Evaluation
Selects `argmax Q_i(prefix)` at each intervention point. Prefix re-encoded from the running event sequence via `encode_prefix()` in `methods/shared/lstm_utils.py`. Evaluated over 1000 SimBank episodes.

### Activity Encoding
Controlled by `--activity_enc` argument (default: `integer`):
- **integer**: activity mapped to integer index, through `nn.Embedding(n_activities, emb_dim=32)`
- **onehot**: activity one-hot encoded, concatenated directly with continuous features

### Key Hyperparameters
| Parameter | Value |
|-----------|-------|
| emb_dim | 32 |
| hidden | 128 |
| n_layers | 2 |
| dropout | 0.2 |
| lr | 1e-3 |
| batch_size | 256 |
| gamma | 0.99 |
| tau | 0.005 |
| epochs | 50 |
| patience | 10 |
| activity_enc | integer |

---

## Method 2: RIMS-DQN (`methods/RIMS-DQN/`)

### RL Paradigm
Online DQN (epsilon-greedy) trained inside a **learned simulator** built from historical data. RIMS first mines a process simulator from the event log, then runs standard online RL inside it. The only method that performs online RL; all others are fully offline.

### Phase 1: Simulator Mining (`methods/RIMS-DQN/convert_data.py`)
Two LSTM models trained from the event log:
- **P_T (Processing Time Model)**: predicts `log(duration_seconds + 1)` for the next event given the current prefix. `LSTM(emb_dim=32, hidden=64, n_layers=1) → Linear(64→1)`. Trained with MSE loss.
- **P_C (Control Flow Model)**: predicts which activity comes next given the current prefix. Same architecture, output size = n_activities. Trained with cross-entropy loss.

Additional components mined from data:
- **Transition matrix**: empirical probability of moving from one activity to the next
- **Acceptance model**: logistic regression predicting case acceptance/rejection probability from final state features
- **Initial prefix distribution**: set of real case prefixes used as starting states for rollouts

### Phase 2: Online RL in Simulator (`methods/RIMS-DQN/train.py`)
Q-networks (LSTM_DQN, same architecture as LSTM-DQN) trained with epsilon-greedy exploration inside the learned simulator. At intervention points the Q-network selects an action; the simulator generates the remainder of the trajectory via P_C and P_T. The simulator class `LearnedSimBankEnv` is defined in `methods/RIMS-DQN/simulator.py` and imported by `train.py` at runtime via `from simulator import LearnedSimBankEnv` (hyphenated directory name prevents package-style import).

### Simulator Domain Knowledge
- **COSTS dict**: hardcoded per-activity cost values used to compute `cum_cost` during rollout
- **IR_LEVELS = [0.07, 0.08, 0.09]**: maps action indices to interest rate values
- **INTERVENTION_ACTIONS dict**: maps activity names to action spaces
- **`_calc_outcome`**: replicates SimBank's reward function (acceptance probability + loan profit)

### Network Architecture (Q-networks)
Identical to LSTM-DQN: three separate `LSTM_DQN` instances (Q₁, Q₂, Q₃) with `(emb_dim=32, hidden=128, n_layers=2, dropout=0.2)`.

Simulator networks:
```
P_T: LSTM(emb_dim=32, hidden=64, n_layers=1) → Linear(64 → 1) → scalar (log duration)
P_C: LSTM(emb_dim=32, hidden=64, n_layers=1) → Linear(64 → n_activities) → softmax
```

### Training Procedure (Online RL)
```
Epsilon-greedy: eps_start=1.0, eps_end=0.05, eps_decay=0.00005
  ε = eps_end + (eps_start − eps_end) · exp(−steps · eps_decay)
Replay buffer: capacity=50000, reward clipped to [−5000, 10000] / 1000
Optimizer: Adam, lr=1e-3
Batch size: 128
Gamma: 0.99, Tau: 0.005
Validation: every 500 episodes; early stopping patience=10 checks
Training order: Q₃ → Q₂ → Q₁ (backward TD)
```

### Key Hyperparameters
| Parameter | Value |
|-----------|-------|
| emb_dim | 32 |
| hidden | 128 |
| n_layers | 2 |
| dropout | 0.2 |
| lr | 1e-3 |
| batch_size | 128 |
| gamma | 0.99 |
| tau | 0.005 |
| replay_capacity | 50000 |
| eps_start | 1.0 |
| eps_end | 0.05 |
| eps_decay | 0.00005 |
| patience | 10 |
| val_every | 500 episodes |
| Simulator P_T/P_C hidden | 64 |
| Simulator n_layers | 1 |

---

## Method 3: K-Means-FQI (`methods/K-Means-FQI/`)

### RL Paradigm
Tabular offline Q-learning using K-means state abstraction (Fitted Q-Iteration variant). No neural network in the Q-function; states are discretised into clusters, Q-values stored in a table. Single-step FQI.

### State Representation
Flat 17-dimensional vector:
- 5 base features (`BASE_FEATURES`): `amount`, `est_quality`, `unc_quality`, `cum_cost`, `elapsed_time`
- 12 activity counts (`TRACKED_ACTIVITIES`): one count per tracked activity accumulated up to the intervention point

No sequence encoding; no LSTM.

### K-means Clustering
```
n_clusters (k): 50 per intervention (configurable via --k)
Features: 17-dim state vector, standardised with StandardScaler per intervention
Algorithm: sklearn KMeans(n_clusters=k, random_state=seed, n_init=10)
One KMeans model per intervention point (3 total)
```

### Q-Table Construction (Backward FQI)
```
1. Fit K-means on all training states for intervention i
2. Assign each transition to its nearest cluster
3. Q₃: Q[cluster, action] = mean(reward) over all matching terminal transitions
4. Q₂: Q[cluster, action] = mean(reward + γ · max_a Q₃[next_cluster, a]) for non-terminal
5. Q₁: routes to Q₂ or Q₃ by next_intervention
```
No gradient descent; Q-table computed directly from averaged returns.

### Data Conversion (`methods/K-Means-FQI/convert_data.py`)
Extracts flat (state, action, reward, next_state, terminal, intervention, next_intervention) tuples. `state = extract_state(event, activity_counts)`, a 17-dim vector.

### Evaluation
```python
cluster = kmeans[int_idx].predict(scaler[int_idx].transform([state]))[0]
action  = argmax(Q_table[int_idx][cluster])
```
`reset()` clears running `activity_counts` between episodes.

### Key Hyperparameters
| Parameter | Value |
|-----------|-------|
| k (n_clusters) | 50 |
| gamma | 0.99 |
| steps | 3 |

---

## Method 4: CQL-SN — Single-Network CQL (`methods/CQL-SN/`)

### RL Paradigm
Conservative Q-Learning (CQL) with a **single shared LSTM-DQN** handling all three intervention points. Invalid actions per intervention are masked to −∞ in both the TD target and the CQL logsumexp before argmax/loss computation.

### Network Architecture
```
Single LSTM_DQN(n_activities, n_features=6, n_act=MAX_ACTIONS=3,
                emb_dim=32, hidden=128, n_layers=2, dropout=0.2)
  Output: 3 Q-values; per-intervention masking sets invalid actions to −∞
    Int 0/1: Q-values for actions ≥ 2 masked to −∞
    Int 2:   all 3 used
```
Single online network + single target network, soft Polyak updates (τ=0.005).

### CQL Loss (with action masking)
```python
q       = model(prefix)              # (B, 3)
q_taken = q.gather(1, action)        # (B,)

# TD target — next-state Q masked by next_intervention
nq = target(next_prefix); nq_masked = nq.clone()
for j in {0,1,2}: nq_masked[next_int==j, N_ACTIONS[j]:] = −∞
max_nq = nq_masked.max(1).values
target = term * norm(r) + (1−term) * γ * max_nq
td_loss = MSE(q_taken, target)

# CQL penalty over masked current Q
q_masked = q.clone()
for j in {0,1,2}: q_masked[int==j, N_ACTIONS[j]:] = −∞
cql_loss = (logsumexp(q_masked, dim=1) − q_taken).mean()

total = td_loss + α · cql_loss       # α = 1.0 default
```

### Training Procedure
```
Optimizer: Adam, lr=1e-3, weight_decay=1e-5
Scheduler: ReduceLROnPlateau(factor=0.5, patience=5)
Batch size: 256, Alpha (CQL): 1.0, Gamma: 0.99, Tau: 0.005
Epochs: 50, Patience: 10, es_delta: 1e-4, Grad clip: L2-norm 1.0
```

### Key Hyperparameters
| Parameter | Value |
|-----------|-------|
| emb_dim | 32 |
| hidden | 128 |
| n_layers | 2 |
| dropout | 0.2 |
| lr | 1e-3 |
| batch_size | 256 |
| alpha (CQL) | 1.0 |
| gamma | 0.99 |
| tau | 0.005 |
| epochs | 50 |
| patience | 10 |

---

## Method 5: CQL-MN — Multi-Network CQL (`methods/CQL-MN/`)

### RL Paradigm
CQL with **three separate LSTM-DQN networks**, one per intervention. Same CQL penalty as CQL-SN, but each network is specialised to its own intervention and only sees transitions from that intervention. Structurally identical to LSTM-DQN plus a per-step CQL term.

### Network Architecture
```
Q₁: LSTM_DQN(n_act=2, emb_dim=32, hidden=128, n_layers=2, dropout=0.2)
Q₂: LSTM_DQN(n_act=2, emb_dim=32, hidden=128, n_layers=2, dropout=0.2)
Q₃: LSTM_DQN(n_act=3, emb_dim=32, hidden=128, n_layers=2, dropout=0.2)
```
Three online networks + three target networks, soft Polyak updates (τ=0.005).

### CQL Loss (per network, no masking needed)
```
td_loss  = MSE(Q_i(s)[a_taken], target_i)
cql_loss = (logsumexp(Q_i(s), dim=1) − Q_i(s)[a_taken]).mean()
total    = td_loss + α · cql_loss          (α = 1.0)
```

### Backward TD Routing
- Q₃ target: `norm(r)` (terminal only)
- Q₂ target: `term·norm(r) + (1−term)·γ·max(Q₃t(s'))`
- Q₁ target: routes by `next_intervention` to `max(Q₂t(s'))` or `max(Q₃t(s'))`

### Training Procedure
Identical to CQL-SN: Adam lr=1e-3, batch=256, α=1.0, γ=0.99, τ=0.005, 50 epochs, patience=10, backward TD order Q₃→Q₂→Q₁.

### Key Hyperparameters
Same as CQL-SN.

---

## Method 6: LSTM-DQN-GBR (`methods/LSTM-DQN-GBR/`)

### RL Paradigm
Hybrid: **GBR S-learner (causal outcome model) + offline LSTM-DQN**. Three sequential phases. The S-learner estimates causal treatment effects to replace potentially confounded observed rewards; the downstream DQN trains on these causally-corrected rewards using backward TD (same as LSTM-DQN).

### Three-Phase Pipeline

**Phase 1 — Train GBR S-learner per intervention:**
`GradientBoostingRegressor` fitted as an S-learner: a single model `f([17-dim state | action]) → normalised outcome`, one per intervention. Fitted on all (flat state, action, case_outcome) triplets from the training set.

**Phase 2 — Counterfactual augmentation:**
For every terminal transition the S-learner provides predicted outcomes under each valid action. The transition table is augmented with one synthetic terminal row per (transition, candidate-action) pair, where `action` is the candidate and `reward` is the denormalised S-learner prediction. The factual row's reward is also rewritten using the model's prediction for the observed action. Non-terminal rows are not augmented.

Effect on dataset size per intervention:
```
new_size ≈ #terminal × N_ACTIONS[int_idx] + #non-terminal
```

**Phase 3 — Train LSTM-DQN on causal rewards:**
Standard LSTM-DQN (identical architecture and backward TD to Method 1) trained on the Phase-2 augmented transition table.

### State Representations
- **S-learner (Phase 1 & 2)**: Flat 17-dim state vector (`extract_state()`)
- **DQN (Phase 3)**: Prefix-based sequential state (same as LSTM-DQN)

### Model Architecture — Phase 1 (GBR S-learner)
```
sklearn GradientBoostingRegressor
  n_estimators: 500
  max_depth: 5
  learning_rate: 0.05
  subsample: 0.8
Input: [17-dim StandardScaler-normalised state | 1-dim action] = 18-dim
Output: scalar predicted outcome (normalised)
```

### Model Architecture — Phase 3 (LSTM-DQN)
Identical to Method 1: Q₁, Q₂, Q₃ with LSTM encoder (hidden=128, n_layers=2) + FC head.

### Training Procedure
```
Phase 1:
  X = [[state | action]] per transition
  y = normalised case_outcome
  GBR_i.fit(X, y)  — one model per intervention

Phase 2:
  Augment terminal transitions with counterfactual rows (see above)

Phase 3 (DQN):
  Optimizer: Adam, dqn_lr=1e-3
  Batch size: 256
  Epochs: 50, patience: 10
  Backward TD: Q₃ → Q₂ → Q₁
  Tau: 0.005, Gamma: 0.99
```

### Key Hyperparameters
| Parameter | Value |
|-----------|-------|
| n_estimators | 500 |
| max_depth | 5 |
| gbr_lr | 0.05 |
| subsample | 0.8 |
| dqn_lr | 1e-3 |
| dqn_epochs | 50 |
| dqn_patience | 10 |
| emb_dim | 32 |
| hidden | 128 |
| n_layers | 2 |
| dropout | 0.2 |
| tau | 0.005 |
| gamma | 0.99 |

---

## Method 7: LSTM-DQN-SLearner (`methods/LSTM-DQN-SLearner/`)

### RL Paradigm
Hybrid: **sequence-aware LSTM S-learner + offline LSTM-DQN**. Same three-phase pipeline as LSTM-DQN-GBR, but replaces the flat-state GBR with a prefix-aware LSTM S-learner. The entire pipeline is then fully sequence-aware.

### Three-Phase Pipeline

**Phase 1 — Train LSTM S-learner per intervention:**
`LSTM_SLearner` trained on (prefix, action, case_outcome) triplets via MSE regression with early stopping.

**Phase 2 — Counterfactual augmentation:**
Same protocol as Method 6 Phase 2. Because the S-learner is sequence-aware, both factual and counterfactual reward estimates condition on the full temporal process history.

**Phase 3 — Train LSTM-DQN on causal rewards:**
Identical to Method 6 Phase 3.

### Network Architecture — Phase 1 (LSTM_SLearner)
```
Activity embedding: nn.Embedding(n_activities, emb_dim=32)
LSTM: input=(emb_dim+6)=38, hidden=128, n_layers=2, dropout=0.2
  → prefix encoding h_n[-1]: 128-dim

Action embedding: nn.Embedding(max_actions=3, action_emb_dim=16)

Fusion: concat([prefix (128), action_emb (16)]) → 144-dim
FC head: Linear(144 → 128) → ReLU → Dropout(0.2) → Linear(128 → 1)
Output: scalar predicted outcome (normalised)
```
Unlike LSTM_DQN which outputs Q-values for all actions, LSTM_SLearner takes a specific action and outputs a single scalar for that (prefix, action) pair.

### Network Architecture — Phase 3 (LSTM-DQN)
Identical to Method 1.

### Training Procedure
```
Phase 1 (S-learner):
  Optimizer: Adam, slearner_lr=1e-3
  Loss: MSE(predicted_outcome, normalised_case_outcome)
  Batch size: 256
  Epochs: 150, patience: 10

Phase 3 (DQN):
  Identical to Method 1 (lr=1e-3, batch=256, epochs=50, patience=10, τ=0.005, γ=0.99)
```

### Key Hyperparameters
| Parameter | Value |
|-----------|-------|
| emb_dim | 32 |
| action_emb_dim | 16 |
| hidden | 128 |
| n_layers | 2 |
| dropout | 0.2 |
| slearner_lr | 1e-3 |
| slearner_epochs | 150 |
| slearner_patience | 10 |
| dqn_lr | 1e-3 |
| dqn_epochs | 50 |
| dqn_patience | 10 |
| tau | 0.005 |
| gamma | 0.99 |

---

## Method 8: LSTM-DQN-TabPFN (`methods/LSTM-DQN-TabPFN/`)

### RL Paradigm
Hybrid: **TabPFN S-learner + offline LSTM-DQN**. Same three-phase pipeline as LSTM-DQN-GBR, but replaces the GBR with a **TabPFN regressor** — a pretrained transformer that performs in-context learning, requiring no gradient-based training in Phase 1.

### Phase 1 — TabPFN S-learner
`TabPFNRegressor` fitted per intervention on (flat 17-dim state, action, case_outcome) triplets. `fit()` stores the data; `predict()` runs a transformer forward pass over it (in-context learning, no gradient descent). A `StandardScaler` is fitted on state features. If the training set exceeds `max_samples` (default 10000), a random subset is used.

### Phase 2 — Counterfactual augmentation
Same protocol as Method 6 Phase 2. `predict_with_tabpfn` queries the model with concatenated `[state | action]` as a flat feature vector.

### Phase 3 — Train LSTM-DQN on causal rewards
Standard LSTM-DQN with backward TD, identical to Method 1.

### State Representations
- **S-learner**: Flat 17-dim state (`extract_state()`) + 1-dim action = 18-dim input to TabPFN
- **DQN**: Prefix-based sequential state (same as LSTM-DQN)

### Training Procedure
```
Phase 1:
  X = [StandardScaler(state) | action]  (18-dim)
  y = normalised case_outcome
  TabPFN_i.fit(X, y)   — no gradient descent

Phase 3 (DQN):
  Identical to Method 1 (lr=1e-3, batch=256, epochs=50, patience=10, τ=0.005, γ=0.99)
```

### Key Hyperparameters
| Parameter | Value |
|-----------|-------|
| tabpfn_max_samples | 10000 |
| dqn_lr | 1e-3 |
| dqn_epochs | 50 |
| dqn_patience | 10 |
| emb_dim | 32 |
| hidden | 128 |
| n_layers | 2 |
| dropout | 0.2 |
| batch_size | 256 |
| tau | 0.005 |
| gamma | 0.99 |

### Checkpoint Keys
`tabpfn_0/1/2` (pickled), `scaler_0/1/2` (pickled), `outcome_mean_i`, `outcome_std_i`, `Q1`, `Q2`, `Q3`, `config`

---

## Method 9: LSTM-DQN-DragonNet (`methods/LSTM-DQN-DragonNet/`)

### RL Paradigm
Hybrid: **LSTM-DragonNet causal outcome model + offline LSTM-DQN**. Same three-phase pipeline as LSTM-DQN-SLearner, but replaces the plain LSTM S-learner with a **DragonNet** architecture that adds a propensity head and targeted regularisation loss (Shi et al., 2019), improving causal reward quality under confounding.

### Phase 1 — LSTM-DragonNet
`LSTM_DragonNet` trained per intervention on (prefix, action, case_outcome) triplets. The model jointly optimises a factual outcome loss, a propensity loss (predicting which action was taken), and a targeted regularisation term.

### Phase 2 — Counterfactual augmentation
Same protocol as other causal hybrids. For each terminal transition, the factual reward is rewritten as `head_{a_obs}(Φ(prefix))` (denormalised) and one synthetic row is added per remaining candidate action with reward `head_a(Φ(prefix))`. The propensity head and `eps` parameter are not used at this stage.

### Phase 3 — Train LSTM-DQN on causal rewards
Standard LSTM-DQN with backward TD, identical to Method 1.

### Network Architecture — Phase 1 (LSTM_DragonNet)
```
Shared LSTM trunk Φ:
  Activity embedding: nn.Embedding(n_activities, emb_dim=32)
  LSTM: input=38, hidden=128, n_layers=2, dropout=0.2
  Output: representation r = Φ(prefix), 128-dim

Per-action outcome heads (one per action for this intervention):
  head_a: Linear(128→64) → ReLU → Dropout(0.2) → Linear(64→1)
  Int. 0 & 1: 2 heads; Int. 2: 3 heads.

Propensity head:
  Linear(128→64) → ReLU → Dropout(0.2) → Linear(64→n_actions)
  Outputs logits for P(action | prefix)

Targeted regularisation scalar:
  self.eps: nn.Parameter (scalar, learned, one per intervention model)
```

### DragonNet Loss
```
factual_loss    = MSE(head_a(r), normalised_outcome)
propensity_loss = CrossEntropy(prop_logits, observed_action)
targeted_loss   = MSE(outcome_pred + eps / max(p_obs, 1e-6), normalised_outcome)
  where p_obs   = softmax(prop_logits)[observed_action]

total = factual_loss + α_prop · propensity_loss + α_targeted · targeted_loss
```

### Training Procedure
```
Phase 1 (DragonNet):
  Optimizer: Adam, lr=1e-3, weight_decay=1e-5
  LR scheduler: ReduceLROnPlateau(factor=0.5, patience=5)
  Batch size: 256
  Epochs: 150, patience: 15
  Early stopping criterion: validation factual MSE only

Phase 3 (DQN):
  Identical to Method 1 (lr=1e-3, batch=256, epochs=50, patience=10, τ=0.005, γ=0.99)
```

### Key Hyperparameters
| Parameter | Value |
|-----------|-------|
| emb_dim | 32 |
| hidden | 128 |
| n_layers | 2 |
| dropout | 0.2 |
| slearner_lr | 1e-3 |
| slearner_epochs | 150 |
| patience (DragonNet) | 15 |
| alpha_prop | 1.0 |
| alpha_targeted | 1.0 |
| dqn_lr | 1e-3 |
| dqn_epochs | 50 |
| dqn_patience | 10 |
| batch_size | 256 |
| tau | 0.005 |
| gamma | 0.99 |

### Checkpoint Keys
`dragonnet_1/2/3` (state dicts), `Q1`, `Q2`, `Q3`, `config`

---

## Cross-Method Comparison

| Aspect | LSTM-DQN | RIMS-DQN | K-Means-FQI | CQL-SN | CQL-MN | LSTM-DQN-GBR | LSTM-DQN-SLearner | LSTM-DQN-TabPFN | LSTM-DQN-DragonNet |
|--------|----------|----------|-------------|--------|--------|--------------|-------------------|-----------------|-------------------|
| **RL type** | Offline DQN | Online DQN | Offline FQI | Offline CQL | Offline CQL | Causal+Offline DQN | Causal+Offline DQN | Causal+Offline DQN | Causal+Offline DQN |
| **State (policy)** | Sequential | Sequential | Flat 17-dim | Sequential | Sequential | Sequential | Sequential | Sequential | Sequential |
| **State (causal model)** | — | — | — | — | — | Flat 17-dim | Sequential | Flat 17-dim | Sequential |
| **Causal model** | — | — | — | — | — | GBR S-learner | LSTM S-learner | TabPFN S-learner | LSTM DragonNet |
| **Causal model trains?** | — | — | — | — | — | Yes (GBR) | Yes (LSTM) | No (pretrained) | Yes (LSTM) |
| **Confounding mechanism** | None | None | None | None | None | S-learner | S-learner | S-learner | Propensity + targeted |
| **CQL penalty** | No | No | No | Yes α=1.0 | Yes α=1.0 | No | No | No | No |
| **Backward TD** | Yes | Yes | Yes | Yes | Yes | Yes (DQN) | Yes (DQN) | Yes (DQN) | Yes (DQN) |
| **Target network τ** | 0.005 | 0.005 | No | 0.005 | 0.005 | 0.005 | 0.005 | 0.005 | 0.005 |
| **Reward signal** | Observed | Simulated | Observed | Observed | Observed | Causal | Causal | Causal | Causal |
| **Env interaction** | None | Simulator | None | None | None | None | None | None | None |

---

## Reproducibility: Seeding Strategy

All methods implement a three-layer seeding strategy.

**1. Global seed** (controls weight initialisation and all random ops):
```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
```

**2. DataLoader seed** (controls shuffle order per epoch):
```python
g = torch.Generator()
g.manual_seed(seed + int_idx)   # per-intervention offset avoids identical shuffle orders
DataLoader(..., worker_init_fn=seed_worker, generator=g)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
```

**3. sklearn seed** (for K-Means-FQI and LSTM-DQN-GBR):
```python
KMeans(random_state=seed, n_init=10)
GradientBoostingRegressor(random_state=seed)
```

The 5 experimental seeds (`SEEDS = [42, 123, 456, 789, 1024]` in `methods/shared/experiment_config.py`) are applied identically across all methods.

**Per-seed train/val split:** `split_train_val(df, val_ratio=0.2, seed=seed)` performs an 80/20 case-level split using the training seed. The split must be regenerated with the matching seed before training each model; using a fixed split across all seeds produces systematically incorrect early stopping and biased model selection. `run_intervention_combos.py` enforces this by calling `convert_data.py --seed {seed}` before each training run.

---

## Data Generation, Conversion and Loading Pipeline

### 1. Raw event-log generation (`methods/shared/generate_data.py`)

Entry points:
- `python generate_data.py` (root-level delegator)
- `python methods/{Method}/generate_data.py` (per-method delegator)
- `python methods/shared/generate_data.py` (called directly by `run_intervention_combos.py`)

All three delegate to `shared.generate_data.main()`.

CLI arguments:

| Flag | Default | Meaning |
|------|---------|---------|
| `--n_cases` | 10000 | Number of independent cases to simulate |
| `--confounded` | off | Use confounded (bank+RCT) generation |
| `--delta` | 0.95 | Bank-policy fraction in the mix (only with `--confounded`) |
| `--seed` | 42 | Master RNG seed |
| `--active` | all | Comma-separated active intervention indices (for subset data generation) |

Key functions in `methods/shared/data_utils.py`:

- **`get_simbank_params(n_cases, seed, rct)`** builds the SimBank `params` dict:
  - `simulation_start = datetime(2024, 3, 20, 8, 0)`
  - `intervention_info`: three interventions, action spaces `[2, 2, 3]`, `RCT_timing = [1000, 1000, 1000]`
  - `policies_info`: bank-policy thresholds
  - `log_cols`: `[case_nr, activity, timestamp, elapsed_time, cum_cost, est_quality, unc_quality, amount, interest_rate, discount_factor, outcome, quality, noc, nor, min_interest_rate]`

- **`generate_rct_data(n_cases, seed)`**: `RCT=True`, uniform random actions at all interventions.
- **`generate_confounded_data(n_cases, seed, delta=0.95)`**:
  1. Bank-policy simulation (`RCT=False`, seed `seed`) → `df_bank`
  2. RCT simulation (seed `seed*10`, `simulation_start = bank_run.simulation_end`) → `df_rct`
  3. Mix via `confounding_level.set_delta(df_bank, df_rct, delta)` → ~95% bank, ~5% RCT

- **`generate_confounded_data_subset(n_cases, seed, active, delta=0.95)`** and **`generate_rct_data_subset(n_cases, seed, active)`**: variants where only interventions in `active` are randomised; inactive interventions follow bank policy. Used by `run_intervention_combos.py` for subset experiments.

Outputs:
- `data/simbank_{RCT|CONF}_{n_cases}_raw.pkl`
- `data/simbank_{RCT|CONF}_{n_cases}_params.pkl`

### 2. Method-specific transition extraction (`methods/{Method}/convert_data.py`)

**Sequential prefix transitions** (LSTM-DQN, RIMS-DQN Q-network input, CQL-SN, CQL-MN, LSTM-DQN-SLearner, LSTM-DQN-GBR, LSTM-DQN-TabPFN, LSTM-DQN-DragonNet):
- Per case: walk events, identify intervention rows by activity name
  - `prefix` = list of event dicts from case start up to (not including) the intervention
  - `action` = integer; for intervention 2 derived via `get_ir_action(interest_rate)` (0.07→0, 0.08→1, 0.09→2)
  - `reward` = `outcome` for terminal, `0.0` for non-terminal
  - `next_prefix`, `terminal`, `intervention ∈ {0,1,2}`, `next_intervention ∈ {1,2,−1}`
- Branching paths handled: only-int0, int0→int1, int0→int2, int0→int1→int2
- Train/val split by `case_nr` via `split_train_val(df, val_ratio=0.2, seed)` with the training seed

**Flat-state transitions** (K-Means-FQI; LSTM-DQN-GBR and LSTM-DQN-TabPFN additionally keep a flat `state` column for the S-learner):
- `state = extract_state(prev_event, activity_counts)`, 17-dim vector
- `next_state = extract_state(...)` at the next intervention, or zeros for terminal

**RIMS simulator artefacts** (`methods/RIMS-DQN/convert_data.py`): trains P_T, P_C, empirical transition matrix, logistic acceptance model; saves as `data/rims_{suffix}_{n}_simulator.pkl`.

Per-method output paths:
```
data/lstm_{suffix}_{n}_trans_{train|val}.pkl                (prefix sequences)
data/rims_{suffix}_{n}_simulator.pkl                        (P_T, P_C, transition matrix, acceptance model)
data/kmeans_{suffix}_{n}_trans_{train|val}.pkl              (flat 17-dim states)
data/single_cql_{suffix}_{n}_trans_{train|val}.pkl          (prefix sequences)
data/multi_cql_{suffix}_{n}_trans_{train|val}.pkl           (prefix sequences)
data/procause_econml_{suffix}_{n}_trans_{train|val}.pkl     (prefix + flat 17-dim state column)
data/procause_lstm_{suffix}_{n}_trans_{train|val}.pkl       (prefix sequences)
data/lstm_dqn_tabpfn_{suffix}_{n}_trans_{train|val}.pkl     (prefix + flat 17-dim state column)
data/lstm_dqn_dragonnet_{suffix}_{n}_trans_{train|val}.pkl  (prefix sequences)
```

`--steps {1,2,3}` truncates transitions to the first `s` interventions and adds a `_steps{s}` suffix.

### 3. Vocabulary, normalisation and DataLoader construction

- **`build_vocab_and_stats(df_train)`** in `methods/shared/lstm_utils.py`: walks all events in `prefix` + `next_prefix`, returns `(activity_to_idx, feat_means, feat_stds)`. Activity strings mapped to unique integers ≥ 1 (0 = padding). Sorted unique activities for run-to-run hash-order stability.
- **`encode(prefixes, activity_to_idx, feat_means, feat_stds, max_len)`**: produces `(acts, feats, lens)` numpy arrays, normalising continuous features per `feat_means/feat_stds`.
- **`encode_prefix(prefix, cfg)`**: single-prefix variant for evaluation; reads vocab and stats from the saved checkpoint `cfg` dict.
- DataLoaders seeded with `torch.Generator().manual_seed(seed + int_idx)` for reproducible epoch shuffling.

### 4. Loading at training time

Each `methods/{Method}/train.py`:
```python
df_train = load_pickle(f"data/{method}_{suffix}_{n}_trans_train.pkl")
df_val   = load_pickle(f"data/{method}_{suffix}_{n}_trans_val.pkl")
activity_to_idx, feat_means, feat_stds = build_vocab_and_stats(df_train)
```
Config dict (vocab, stats, max_len, network shape, `n_actions`) saved alongside the model state dict in the `.pth` checkpoint.

---

## Evaluation: Technical Details

### Shared evaluation harness (`methods/shared/evaluation.py`)

All methods evaluate via `evaluate_policy(policy_fn, n_episodes, params, seed, ...)`:
```python
gen = simulation.PresProcessGenerator(params, seed=seed)
for i in range(n_episodes):
    if reset_fn: reset_fn()
    prefix_list = gen.start_simulation_inference(seed_to_add=i)
    while gen.int_points_available:
        prefix     = prefix_list[0][:-1]
        prev_event = prefix[-1]
        int_idx    = gen.current_int_index
        action     = policy_fn(prev_event, int_idx, prefix)   # use_prefix=True for LSTM-based
        prefix_list = gen.continue_simulation_inference(action)
    outcome = float(pd.DataFrame(gen.end_simulation_inference())["outcome"].iloc[-1])
```

A fresh `PresProcessGenerator` is built from the saved `params` dict. `seed_to_add=i` gives paired, reproducible episodes across methods. Default: `n_episodes=1000`, eval seed `1042`.

### Baselines

`bank_policy(prev_event, int_idx)` replicates the bank's heuristic from SimBank's `extra_flow_conditions.py`:
- **Int 0**: `priority` if `amount > 50000` and `est_quality ≥ 5`, else `standard`
- **Int 1**: `contact_HQ` if `noc < 2`, `unc_quality == 0`, `amount > 10000`, `est_quality ≥ 2`; else `skip`
- **Int 2**: `7%` if `amount > 60000`; `8%` if `amount > 30000`; else `9%`

`random_policy(prev_event, int_idx)` returns `np.random.randint(0, [2, 2, 3][int_idx])`.

### Method-specific policy wrappers

| Method | Policy class | Inputs to Q/table |
|--------|-------------|-------------------|
| LSTM-DQN | `LSTMPolicy` | `encode_prefix(prefix, cfg)` → `argmax Q_i[:N_ACTIONS[i]]` |
| RIMS-DQN | `RIMSPolicy` | same as LSTM-DQN |
| K-Means-FQI | `KMeansPolicy` | `extract_state(prev_event, counts)`; cluster via scaler+KMeans; `argmax Q_table[i][cluster]`; `reset()` clears activity counts |
| CQL-SN | `SingleCQLPolicy` | `encode_prefix`; mask invalid slots; argmax over `N_ACTIONS[i]` |
| CQL-MN | `MultiCQLPolicy` | `encode_prefix`; per-intervention Q-network |
| All causal hybrids | `LSTMPolicy`-style | `encode_prefix`; only DQN heads queried at eval; causal model unused |

Beyond the trained `--steps` boundary every policy falls back to `bank_policy`.

### Result aggregation (`run_seeds.py`, `run_all_steps.py`, `retrain_all_3step.py`)

`run_seeds.py` runs train + evaluate once per seed in `SEEDS = [42, 123, 456, 789, 1024]`, aggregates mean ± std across seeds. Available for: K-Means-FQI, LSTM-DQN, RIMS-DQN, CQL-SN, CQL-MN.

`run_all_steps.py` orchestrates all 9 methods × steps {1,2,3} × conditions {RCT, CONF} × seeds and writes `results/all_results.json`.

`retrain_all_3step.py` performs a clean retrain of all 9 methods × 5 seeds × {CONF, RCT} specifically for `--steps 3`, writing into `results/all_results.json`.

Reported metric: mean per-case `outcome` (and std); also reported as `% gain over Bank Policy` = `(avg / bank_avg − 1) × 100`.

---

## Intervention Subset Experiment (`run_intervention_combos.py`)

### Purpose
Empirically tests whether joint optimisation over all intervention points outperforms training on subsets. LSTM-DQN is trained separately on each of the 7 non-empty subsets of {int0, int1, int2} and evaluated against the joint 3-step baseline.

### Subsets
```
{0}, {1}, {2}, {0,1}, {0,2}, {1,2}, {0,1,2}
```
For each subset, inactive interventions are controlled by bank policy at both training and deployment time — ensuring the comparison is between purely subset-trained models, not partial deployment of a joint model.

### Data Generation for Subsets
Each subset uses its own training data in which only the active interventions are randomised:
- `generate_confounded_data_subset(n_cases, seed, active)` / `generate_rct_data_subset(n_cases, seed, active)` in `methods/shared/data_utils.py`
- `SimBank-main/SimBank/activity_execution.py` gates interest-rate randomisation on a per-name check: `if intervention_info["RCT"] and "set_ir_3_levels" in intervention_info["name"]` (line 204), consistent with how interventions 0 and 1 are gated via `policies_to_ignore`
- Exception: the `{0,1,2}` subset reuses the standard CONF/RCT data (no separate generation needed)

### Model Architecture
`LSTM_DQN` per active intervention. Keys saved as `Q_int{i}` (not `Q1/Q2/Q3`) to avoid collision with `--steps` mode checkpoints. Model filename suffix: `_active{IDS}` (e.g., `_active02` for {0,2}).

### Training (backward TD for active subset)
```python
# Models initialised in sorted(active) order for consistent RNG consumption
models_init = {i: make_model(N_ACTIONS[i]) for i in active_sorted}

# Train in reverse order
for i in reversed(active_sorted):
    if i == active_sorted[-1]:
        target_fn = lambda b: norm(b['reward'])   # terminal reward only
    else:
        later_targets = {j: Q_target[j] for j in active_sorted if j > i}
        target_fn = make_td_target(later_targets)  # TD routing by next_intervention
    best = train_q(Q_i, Qt_i, optimizer, tr_loader, va_loader, target_fn, args)
```

`make_td_target` routes by `next_intervention` index: for each active downstream network, computes `γ · max Q_j(next_state)` and applies it to the matching transitions.

### Seeding for Correctness
`run_intervention_combos.py` calls `methods/LSTM-DQN/convert_data.py --seed {seed}` before each training run, regenerating the train/val split for that specific seed. This matches the behaviour of `retrain_all_3step.py`, which also regenerates per seed. All 5 seeds produce independent estimates; the `{0,1,2}` sanity check should agree with the 3-step joint baseline within stochastic noise.

### Results Files
- `results/lstm_joint_vs_subset.json` — CONF results
- `results/lstm_joint_vs_subset_rct.json` — RCT results

Format per entry:
```json
"lstm_CONF_Int012": {"mean": ..., "std": ..., "per_seed": {"42": ..., "123": ..., ...}}
```
The joint baseline is loaded from `results/all_results.json` (key `lstm_{suffix}_3`) and written as `lstm_{suffix}_joint` in the subset file.

---

## Data Pipeline Summary

```
methods/shared/generate_data.py  (or root generate_data.py)
  └─ data/simbank_{RCT|CONF}_{n_cases}_raw.pkl
     data/simbank_{RCT|CONF}_{n_cases}_params.pkl

methods/{Method}/convert_data.py   (reads shared raw, writes method-specific transitions)
  └─ K-Means-FQI:         data/kmeans_{suffix}_{n}_trans_{train|val}.pkl
     LSTM-DQN:             data/lstm_{suffix}_{n}_trans_{train|val}.pkl
     RIMS-DQN:             data/rims_{suffix}_{n}_simulator.pkl
     CQL-SN:               data/single_cql_{suffix}_{n}_trans_{train|val}.pkl
     CQL-MN:               data/multi_cql_{suffix}_{n}_trans_{train|val}.pkl
     LSTM-DQN-GBR:         data/procause_econml_{suffix}_{n}_trans_{train|val}.pkl
     LSTM-DQN-SLearner:    data/procause_lstm_{suffix}_{n}_trans_{train|val}.pkl
     LSTM-DQN-TabPFN:      data/lstm_dqn_tabpfn_{suffix}_{n}_trans_{train|val}.pkl
     LSTM-DQN-DragonNet:   data/lstm_dqn_dragonnet_{suffix}_{n}_trans_{train|val}.pkl

methods/{Method}/train.py
  └─ models/{prefix}_{suffix}_{n}_s{seed}.{pkl|pth}

methods/{Method}/evaluate.py  (loads model + params, runs SimBank episodes, writes JSON)
  └─ per-seed result → aggregated into results/all_results.json
```

Subset experiment (LSTM-DQN only):
```
methods/shared/generate_data.py --active {ids}
  └─ data/simbank_{suffix}_{n}_active{ids}_raw.pkl   (omitted for {0,1,2} — reuses standard)

methods/LSTM-DQN/convert_data.py --active {ids} --seed {seed}
  └─ data/lstm_{suffix}_{n}_active{ids}_trans_{train|val}.pkl   (omitted for {0,1,2})

methods/LSTM-DQN/train.py --active {ids} --seed {seed}
  └─ models/lstm_{suffix}_{n}_s{seed}_active{ids}.pth

methods/LSTM-DQN/evaluate.py --active {ids}
  └─ results/lstm_joint_vs_subset[_rct].json
```
