# Technical Reference: All Offline RL Methods for Prescriptive Process Monitoring

This document provides exhaustive technical detail per method, intended as raw material for thesis writing.
All details are extracted directly from the codebase.

---

## Shared Infrastructure

### State Representations

Two state representations are used across the codebase. Most methods use only the sequential one; the flat one appears only in K-means (which has no LSTM at all) and as the *S-learner input* of two causal hybrids (EconML, TabPFN). In every causal hybrid the **downstream DQN that selects actions is sequential** — so policy inference is sequential everywhere except K-means.

**Prefix-based (sequential) state — used by the policy/Q-network in every method except K-means** (LSTM, RIMS, Single-Model CQL, Multi-Model CQL, and the Phase-3 DQN of ProCause-EconML, ProCause-LSTM, TabPFN, DragonNet):
- A variable-length sequence of events from the start of a case up to (not including) the intervention point
- Each timestep has 7 features: 6 continuous features + 1 activity identifier
- Continuous features (`FEATURE_COLS` in `shared/lstm_utils.py`): `amount`, `est_quality`, `unc_quality`, `interest_rate`, `cum_cost`, `elapsed_time`
- Activity identifier: either integer index (integer encoding) or one-hot vector (onehot encoding)
- Sequences are padded to `max_len` (typically 10, set from training-set max prefix length) and packed with `pack_padded_sequence` for efficient LSTM processing
- Per-feature normalisation: mean/std computed from training prefixes; `feat_means` and `feat_stds` are saved with the checkpoint via `build_vocab_and_stats()`

**Flat state vector — K-means policy + the *causal S-learner input* of EconML and TabPFN** (NOT used as the DQN input in any method):
- Fixed 17-dimensional vector: 5 base features + 12 activity-count features
- Base features (5, `BASE_FEATURES` in `shared/experiment_config.py`): `amount`, `est_quality`, `unc_quality`, `cum_cost`, `elapsed_time`
- Activity counts (12, `TRACKED_ACTIVITIES`): `initiate_application`, `start_standard`, `start_priority`, `call_customer`, `email_customer`, `validate_application`, `contact_headquarters`, `skip_contact`, `calculate_offer`, `cancel_application`, `receive_acceptance`, `receive_refusal`
- `STATE_DIM = len(BASE_FEATURES) + len(TRACKED_ACTIVITIES) = 17`
- Built by `extract_state(event, activity_counts)` in `shared/data_utils.py`

For ProCause-LSTM and DragonNet the S-learner is *also* sequential (it shares the same prefix encoder as the DQN); only EconML's GBR S-learner and TabPFN's transformer S-learner consume the flat 17-dim state.

Note: the LSTM `FEATURE_COLS` (per-timestep features) include `interest_rate` and use `[amount, est_quality, unc_quality, interest_rate, cum_cost, elapsed_time]` (6 features). The flat state intentionally drops `interest_rate` because at int 0 / int 1 it is undefined; intervention 2's interest-rate decision is the action itself.

Quick map:

| Method | Policy/Q-network state | Causal model state |
|--------|------------------------|--------------------|
| LSTM-DQN, RIMS | sequential prefix | — |
| K-means | flat 17-dim | — |
| Single CQL, Multi CQL | sequential prefix | — |
| ProCause EconML | sequential prefix (DQN) | flat 17-dim (GBR) |
| ProCause LSTM | sequential prefix (DQN) | sequential prefix (LSTM S-learner) |
| TabPFN | sequential prefix (DQN) | flat 17-dim (TabPFN) |
| DragonNet | sequential prefix (DQN) | sequential prefix (LSTM-DragonNet) |

### Intervention Points (SimBank)
Three sequential intervention points in every case:
- **Intervention 0** — `choose_procedure`: 2 actions (`start_standard`=0, `start_priority`=1)
- **Intervention 1** — `time_contact_HQ`: 2 actions (`contact_headquarters`=0, `skip_contact`=1)
- **Intervention 2** — `set_ir_3_levels`: 3 actions (low IR=0, medium IR=1, high IR=2)

`N_ACTIONS = [2, 2, 3]`

Not every case reaches all three interventions. Convert_data handles all branching paths (only int0, int0→int1, int0→int2, int0→int1→int2).

### Backward TD Bootstrapping
Methods that use Q-learning train in reverse intervention order: Q3 first, then Q2 using Q3 targets, then Q1 using Q2/Q3 targets. This is necessary because rewards are only observed at the final intervention (terminal transition); intermediate transitions have reward=0.

The Q1 target depends on the `next_intervention` field:
- If next is intervention 1: `r + gamma * max(Q2(s'))`
- If next is intervention 2: `r + gamma * max(Q3(s'))`
- If terminal: `r` (no bootstrap)

### Data Generation (SimBank) — quick reference
- `generate_rct_data(n_cases, seed)` — RCT: actions assigned uniformly at random by the simulator
- `generate_confounded_data(n_cases, seed, delta)` — confounded: mixes bank-policy and RCT logs at fraction delta (default 0.95) via `confounding_level.set_delta()`
- Output: raw DataFrame of event-log rows + params dict
- Saved to: `data/simbank_{RCT|CONF}_{n_cases}_raw.pkl` and `data/simbank_{RCT|CONF}_{n_cases}_params.pkl`

A complete walk-through of generation, conversion and loading is given in the **Data Generation, Conversion and Loading Pipeline** section below.

---

## Method 1: LSTM-DQN (lstm/)

### RL Paradigm
Offline Q-learning (DQN variant) with separate Q-networks per intervention point. Uses backward TD bootstrapping (Q3→Q2→Q1). This is the core sequence-based offline RL baseline.

### How Historical Data Is Used
Historical event logs are converted to (prefix_sequence, action, reward, next_prefix_sequence, terminal, intervention_idx, next_intervention_idx) tuples. These tuples are stored in replay buffers — one per intervention — and sampled in mini-batches during training. No environment simulator is queried; the agent only sees the offline transitions.

### State Representation
Prefix-based sequential state. The prefix is the sequence of events from case start to the event immediately before the intervention decision. At intervention 0 there is always exactly 1 event (`initiate_application`). At intervention 1 there are always exactly 2 events. At intervention 2 the prefix length varies from 2 to 10 events.

### Activity Encoding
Controlled by `--activity_enc` argument (default: `integer`):
- **integer**: each activity is mapped to a unique integer index; passed through `nn.Embedding(n_activities, emb_dim=32)` → embedding vector
- **onehot**: activity is one-hot encoded; no embedding layer; directly concatenated with continuous features

### Network Architecture (LSTM_DQN)
```
Input per timestep: [activity_embedding (32-dim)] + [6 continuous features] = 38-dim  (integer mode)
                    [activity_onehot (n_activities-dim)] + [6 continuous features]      (onehot mode)

LSTM: input_size=38, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2 (between layers)
  → takes packed sequences, returns final hidden state h_n[-1] (last layer, last timestep): 128-dim

FC head:
  Linear(128 → 128) → ReLU → Dropout(0.2) → Linear(128 → n_actions)
  n_actions = 2 for interventions 0 and 1; 3 for intervention 2
```

Three separate `LSTM_DQN` instances: Q1, Q2, Q3. Each has its own replay buffer and target network.

### Target Networks
Each Q-network has a corresponding target network (Q1t, Q2t, Q3t) with identical architecture. Target networks are updated via **soft (Polyak) updates** after every training step:
```
θ_target ← τ * θ_online + (1 - τ) * θ_target     τ = 0.005
```
Target networks are set to `.eval()` mode at creation and kept there permanently (disables dropout during target computation — Jakob's Bug 2 fix).

### Data Conversion (lstm/convert_data.py)
For each case, the script identifies intervention rows by activity name, then extracts:
- `prefix`: list of event dicts from case start up to (not including) the intervention row
- `action`: integer action taken
- `reward`: 0.0 for non-terminal transitions; float `outcome` for terminal transition
- `next_prefix`: prefix at the next intervention (or empty for terminal)
- `terminal`: True/False
- `intervention`: 0, 1, or 2
- `next_intervention`: 1, 2, or -1 (terminal)

Continuous features in the prefix are normalised using per-feature mean and std computed from the training set. Normalisation stats and activity vocabulary are saved in the checkpoint.

### Training Procedure
```
Optimizer: Adam, lr=1e-3
Loss: MSE(Q_predicted, Q_target)
Batch size: 64
Replay buffer capacity: 10000 per intervention
Min samples before training: 64
Training order: Q3 first → Q2 → Q1 (backward TD)
Target update: soft update every step, τ=0.005
Gamma (discount): 0.99
Early stopping: patience=10 epochs on validation loss
Epochs: up to 50
```

**Q-target computation:**
```python
# For Q3 (always terminal in full 3-step):
target = reward  # no bootstrap

# For Q2:
with torch.no_grad():
    Q3t.eval()
    next_q = Q3t(next_prefix).max(dim=1).values
target = reward + gamma * (1 - terminal) * next_q

# For Q1 (routes to Q2 or Q3 depending on next_intervention):
mask2 = (next_intervention == 1)
mask3 = (next_intervention == 2)
next_q = torch.zeros(batch)
next_q[mask2] = Q2t(next_prefix[mask2]).max(dim=1).values
next_q[mask3] = Q3t(next_prefix[mask3]).max(dim=1).values
target = reward + gamma * (1 - terminal) * next_q
```

### Evaluation
The trained policy selects `argmax Q_i(prefix)` at each intervention point. The prefix is re-encoded from the running event sequence using `encode_prefix()`. Evaluated over 1000 SimBank episodes. Results compared to bank_policy and random_policy baselines (% gain over bank).

### Key Hyperparameters (defaults)
| Parameter | Value |
|-----------|-------|
| emb_dim | 32 |
| hidden | 128 |
| n_layers | 2 |
| dropout | 0.2 |
| lr | 1e-3 |
| batch_size | 64 |
| gamma | 0.99 |
| tau | 0.005 |
| replay_capacity | 10000 |
| epochs | 50 |
| patience | 10 |
| max_len | 10 |
| activity_enc | integer |
| target_calc | standard |

---

## Method 2: RIMS (rims/)

### RL Paradigm
Online DQN (epsilon-greedy) trained inside a **learned simulator** built from historical data. RIMS first mines a process simulator from the event log, then runs standard online RL inside that simulator. This is the only method that performs online RL; all others are fully offline.

### How Historical Data Is Used — Two Phases

**Phase 1: Simulator Mining (rims/convert_data.py)**
The historical event log is used to train two LSTM models that together simulate the process:
- **P_T (Processing Time Model)**: predicts `log(duration_seconds + 1)` for the next event given the current prefix. Architecture: `LSTM(emb_dim=32, hidden=64, n_layers=1) → Linear → scalar`. Trained with MSE loss.
- **P_C (Control Flow Model)**: predicts which activity comes next given the current prefix (multi-class classification). Same architecture but output size = n_activities. Trained with cross-entropy loss.

Additional components mined from data:
- **Transition matrix**: empirical probability of moving from one activity to the next
- **Acceptance model**: logistic regression predicting case acceptance/rejection probability from final state features
- **Initial prefix distribution**: set of real case prefixes used as starting states for simulation rollouts

**Phase 2: Online RL in Simulator (rims/train.py)**
The Q-networks (LSTM_DQN, identical architecture to lstm/) are trained with epsilon-greedy exploration inside the learned simulator. The simulator generates full case trajectories by sequentially sampling next activities and durations from P_C and P_T. At intervention points, the Q-network selects an action. The reward is the simulated outcome.

### Network Architecture
Q-networks: identical to LSTM-DQN (see Method 1). Three separate Q-networks (Q1, Q2, Q3) with target networks, same `(emb_dim=32, hidden=128, n_layers=2, dropout=0.2)` defaults.

Simulator networks:
```
P_T: LSTM(emb_dim=32, hidden=64, n_layers=1) → Linear(64 → 1) → scalar (log duration)
P_C: LSTM(emb_dim=32, hidden=64, n_layers=1) → Linear(64 → n_activities) → softmax
```

### Training Procedure (Online RL)
```
Epsilon-greedy: eps_start=1.0, eps_end=0.05, eps_decay=0.00005
  ε decays as: ε = eps_end + (eps_start - eps_end) * exp(-steps * eps_decay)

Replay buffer: capacity=50000, reward clipping to [-5000, 10000] / 1000
Optimizer: Adam, lr=1e-3
Batch size: 128
Gamma: 0.99
Tau: 0.005 (soft target updates)
Validation: every 500 episodes on held-out simulator rollouts
Early stopping: patience=10 validation checks
Max episodes: configurable (default ~5000+)
```

Training order: Q3 → Q2 → Q1 (same backward TD as lstm/).
Target networks updated with soft Polyak updates. Target networks kept in `.eval()` mode (dropout disabled during target computation).

### Simulator Domain Knowledge
The RIMS simulator requires hardcoded domain knowledge not needed by other methods:
- **COSTS dict**: hardcoded per-activity cost values used to compute `cum_cost` during rollout
- **IR_LEVELS**: maps action indices to actual interest rate values `[0.07, 0.08, 0.09]` (mirrors SimBank's `set_ir_3_levels` actions in `shared/experiment_config.INTERVENTION_INFO`)
- **INTERVENTION_ACTIONS dict**: maps activity names to action spaces
- **Outcome formula**: `_calc_outcome` replicates SimBank's reward function using acceptance probability + loan profit calculation

This is necessary because RIMS *generates* new trajectories (it must compute costs and outcomes), whereas offline methods only *read* pre-computed transitions from the log.

### Evaluation
Identical to lstm/evaluate.py. RIMSPolicy loads Q1, Q2, Q3 from checkpoint, applies to SimBank episodes.

### Key Hyperparameters (defaults)
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

## Method 3: K-means Offline RL (kmeans/)

### RL Paradigm
Tabular offline Q-learning using K-means state abstraction (Fitted Q-Iteration variant). No neural network in the Q-function; states are discretised into clusters, and Q-values are stored in a table. One of the simplest offline RL baselines.

### How Historical Data Is Used
Historical transitions are converted to flat state vectors. K-means is fitted to cluster these states per intervention. Each cluster-action pair accumulates observed rewards (or bootstrapped TD targets). The Q-table is the mean reward per (cluster, action) cell — a single-step FQI with K-means state abstraction.

### State Representation
Flat 16-dimensional state vector (same as CQL):
- 5 base features: `amount`, `est_quality`, `unc_quality`, `interest_rate`, `cum_cost`
- 11 activity counts: one count per tracked activity, accumulated up to the intervention point

No sequence encoding; no LSTM. The entire process history is compressed into a fixed-size activity-count vector.

### K-means Clustering
```
n_clusters (k): 50 per intervention (configurable via --k_clusters)
Features: 16-dim state vector, standardised with StandardScaler per intervention
Algorithm: sklearn KMeans with random_state=seed
One KMeans model per intervention point (3 total)
```

### Q-Table Construction (Backward TD)
```
1. Fit K-means on all training states for intervention i
2. Assign each transition to its nearest cluster
3. Q3: terminal transitions → Q[cluster, action] = mean(reward) across all matching transitions
4. Q2: Q[cluster, action] = mean(reward + gamma * max_a'(Q3[next_cluster, a'])) for non-terminal
5. Q1: same with routing to Q2 or Q3 depending on next_intervention
```

No gradient descent; the Q-table is computed directly from averaged returns. This is essentially one-step Fitted Q-Iteration.

### Data Conversion (kmeans/convert_data.py)
Extracts flat (state, action, reward, next_state, terminal, intervention, next_intervention) tuples. Same transition extraction logic as multiModelCQL. No prefix sequences needed.

### Evaluation
```python
# Policy
cluster = kmeans[int_idx].predict(scaler[int_idx].transform([state]))[0]
action = argmax(Q_table[int_idx][cluster])
```

State is reconstructed from the current SimBank event by calling `extract_state(prev_event, activity_counts)`. Activity counts are accumulated during the episode.

### Key Hyperparameters (defaults)
| Parameter | Value |
|-----------|-------|
| k_clusters | 50 |
| gamma | 0.99 |
| steps | 3 |
| seed | 42 |

---

## Method 4: Single-Model CQL (singleModelCQL/)

### RL Paradigm
Conservative Q-Learning (CQL) with a **single LSTM-DQN** that handles all three intervention points simultaneously. The intervention point is identified implicitly via the prefix length and content; the network outputs `MAX_ACTIONS=3` Q-values, and invalid actions per intervention are masked to `-inf` in both the TD target and the CQL logsumexp before argmax/loss computation.

### How Historical Data Is Used
Sequential prefix transitions (same conversion as `lstm/`) — but stored in `data/single_cql_*_trans_{train|val}.pkl`. The CQL penalty discourages the Q-network from assigning high values to out-of-distribution actions by penalising `logsumexp(Q(s)) − Q(s, a_taken)` over the masked, valid actions.

### State Representation
Identical to LSTM-DQN: prefix-based sequential state encoded with the shared `LSTM_DQN` module (`shared/lstm_utils.py`). No flat state, no one-hot intervention vector — the model uses one shared LSTM trunk with `n_act=MAX_ACTIONS=3` outputs and per-intervention action masking.

### Network Architecture
```
Single LSTM_DQN(n_activities, n_features=6, n_act=3,
                emb_dim=32, hidden=128, n_layers=2, dropout=0.2)
  Input: same as Method 1
  Output: 3 Q-values; per-intervention masking sets invalid actions to -inf
    → Int 0/1: Q-values for actions ≥ 2 masked
    → Int 2:    all 3 used
```
Single online network + single target network, soft Polyak updates (τ=0.005).

### CQL Loss (with action masking)
```python
q          = model(prefix)                  # (B, 3)
q_taken    = q.gather(action)               # (B,)

# TD target (next-state Q masked by next_intervention)
nq = target(next_prefix); nq_masked = nq.clone()
for j in 0,1,2:  nq_masked[next_int==j, N_ACTIONS[j]:] = -inf
max_nq  = nq_masked.max(1).values
target  = terminal * norm(reward) + (1-terminal) * gamma * max_nq

td_loss  = MSE(q_taken, target)

# CQL penalty over masked current Q
q_masked = q.clone()
for j in 0,1,2:  q_masked[int==j, N_ACTIONS[j]:] = -inf
cql_loss = (logsumexp(q_masked, dim=1) - q_taken).mean()

total = td_loss + alpha * cql_loss            # alpha = 1.0 default
```

### Reward Normalisation
Terminal-reward mean/std computed from the training set; `norm(r) = (r - r_mean) / (r_std + 1e-8)`.

### Data Conversion (singleModelCQL/convert_data.py)
Reads from `data/simbank_*_raw.pkl`, extracts (prefix, action, reward, next_prefix, terminal, intervention, next_intervention) tuples, splits 80/20 by `case_nr` via `split_train_val()` and saves `_trans_train.pkl` / `_trans_val.pkl` plus a `--steps {1,2,3}` variant tag.

### Training Procedure (defaults from `singleModelCQL/train.py`)
```
Optimizer:    Adam, lr=1e-3, weight_decay=1e-5
Scheduler:    ReduceLROnPlateau(factor=0.5, patience=5)
Batch size:   256
Alpha (CQL):  1.0
Gamma:        0.99
Tau:          0.005
Epochs:       50
Patience:     10 (early stopping on validation TD-MSE)
es_delta:     1e-4
Grad clip:    norm 1.0
```

### Evaluation
At each intervention, encode the running prefix with `encode_prefix(cfg)`, run `Q(prefix)`, slice the first `N_ACTIONS[int_idx]` outputs, take argmax. Beyond the trained `--steps` boundary, fall back to `bank_policy`.

### Key Hyperparameters (defaults)
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
| steps | 3 |

---

## Method 5: Multi-Model CQL (multiModelCQL/)

### RL Paradigm
Conservative Q-Learning (CQL) with **three separate LSTM-DQN networks**, one per intervention point. Same CQL penalty as single-model, but each network is specialised to its own intervention and only ever sees transitions from that intervention. Structurally identical to LSTM-DQN (Method 1) plus a CQL conservative term in the per-step loss.

### How Historical Data Is Used
Sequential prefix transitions (same as Method 1). Filtered per intervention via `df[df['intervention']==int_idx]` inside each `make_loader()` call. Stored in `data/multi_cql_*_trans_{train|val}.pkl`.

### State Representation
Prefix-based sequential state — **identical for all three Q-networks**. Each Q_i is a separate `LSTM_DQN(n_activities, len(FEATURE_COLS)=6, n_act=N_ACTIONS[i], …)` instance trained only on intervention-i prefixes.

### Network Architecture
```
Q1: LSTM_DQN(n_act=2, emb_dim=32, hidden=128, n_layers=2, dropout=0.2)
Q2: LSTM_DQN(n_act=2, emb_dim=32, hidden=128, n_layers=2, dropout=0.2)
Q3: LSTM_DQN(n_act=3, emb_dim=32, hidden=128, n_layers=2, dropout=0.2)
```
Three online networks + three target networks. All updated with soft Polyak updates (τ=0.005, after every step).

### CQL Loss (per network)
```
td_loss  = MSE(Q_i(s)[a_taken], target_i)
cql_loss = (logsumexp(Q_i(s), dim=1) - Q_i(s)[a_taken]).mean()
total    = td_loss + alpha * cql_loss            # alpha = 1.0
```
Gradient clipped to L2-norm 1.0; `weight_decay=1e-5`.

### Reward Normalisation
Per-network `r_mean, r_std` computed from terminal rewards in the training set; `norm(r) = (r - r_mean) / (r_std + 1e-8)`.

### Backward TD Routing
- Q3 target: `norm(r)` (terminal only)
- Q2 target: `term*norm(r) + (1-term)*gamma * max(Q3t(s'))`
- Q1 target: routes by `next_intervention` to `max(Q2t(s'))` or `max(Q3t(s'))`; for unknown next intervention, takes the max over both.

### Training Procedure (defaults from `multiModelCQL/train.py`)
```
Optimizer:    Adam, lr=1e-3, weight_decay=1e-5
Scheduler:    ReduceLROnPlateau(factor=0.5, patience=5)
Batch size:   256
Alpha (CQL):  1.0
Gamma:        0.99
Tau:          0.005
Epochs:       50
Patience:     10 (early stopping on validation TD-MSE per network)
es_delta:     1e-4
Grad clip:    norm 1.0
Training order: Q3 → Q2 → Q1 (backward TD)
```

### Key Hyperparameters (defaults)
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

## Method 6: ProCause EconML S-learner (procause/econml_slearner/)

### RL Paradigm
Hybrid method: **causal reward estimation (S-learner) + offline Q-learning (LSTM-DQN)**. Operates in three sequential phases. The S-learner estimates causal treatment effects to replace potentially confounded observed rewards; the downstream DQN then trains on these causally-corrected rewards using the same backward TD procedure as Method 1.

### Three-Phase Pipeline

**Phase 1 — Train GBR S-learner (causal outcome model):**
A GradientBoostingRegressor is trained per intervention as an S-learner: a single model `f(state, action) → outcome` fitted on all (flat 16-dim state, action, case_outcome) triplets. The case outcome is the final result of the entire case — not the intermediate step reward. Three separate GBR models, one per intervention.

**Phase 2 — Counterfactual-augmented causal rewards (shared protocol across Methods 6–9, since 2026-04-21):**
For every terminal transition the S-learner provides a predicted outcome under each *valid* action at that intervention. The training/validation transition tables are augmented with one synthetic terminal row per (transition, candidate-action) pair, in which `action` is overwritten with the candidate and `reward` is the (denormalised) S-learner prediction for that counterfactual. The *factual* row is also rewritten using the model's prediction for the observed action (replacing the logged outcome). Non-terminal rows are not augmented; their reward stays at 0.

Effect on dataset size, per intervention:
```
new_size  ≈  #terminal × N_ACTIONS[int_idx]   +   #non-terminal
```
Source: `lstm_dqn_dragonnet/train.py` Phase 2 block (the original "factual replacement only" code path is preserved, commented out, for rollback). The same augmentation logic is applied in `procause/econml_slearner/train.py`, `procause/lstm_slearner/train.py`, and `lstm_dqn_tabpfn/train.py`.

**Phase 3 — Train LSTM-DQN on causal rewards:**
A standard LSTM-DQN (identical architecture and backward TD procedure to Method 1) is trained using the causal rewards from Phase 2 in place of raw observed rewards. The flat-state S-learner and the prefix-sequence DQN thus operate on different state representations within the same pipeline.

### State Representation
- **S-learner (Phase 1 & 2)**: Flat 16-dim state vector — same `extract_state()` as K-means and CQL
- **DQN (Phase 3)**: Prefix-based sequential state — same as LSTM-DQN (Method 1)

### Model Architecture (Phase 1 — GBR S-learner)
```
Estimator: sklearn GradientBoostingRegressor
n_estimators: 500
max_depth: 5
learning_rate: 0.05
subsample: 0.8

Input: [16-dim normalised state] + [1-dim action] = 17-dim
Output: scalar predicted outcome (normalised)
```

### Model Architecture (Phase 3 — LSTM-DQN)
Identical to Method 1 (LSTM-DQN): three Q-networks (Q1, Q2, Q3) with LSTM encoder (hidden=128, n_layers=2) + FC head, target networks with soft Polyak updates (τ=0.005).

### Training Procedure
```
Phase 1:
  For each intervention i:
    X = [[state | action] for each transition at intervention i]
    y = [case_outcome (normalised) for each transition]
    GBR_i.fit(X, y)

Phase 2:
  For each terminal transition at intervention i:
    causal_reward = GBR_i.predict([state | action_taken])
    replace observed reward with causal_reward

Phase 3 (DQN — identical to Method 1):
  Optimizer: Adam, dqn_lr=1e-3
  Batch size: 256
  Epochs: 50, patience: 10
  Backward TD: Q3 → Q2 → Q1
  Tau: 0.005, Gamma: 0.99
```

### Key Hyperparameters (defaults)
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

## Method 7: ProCause LSTM S-learner (procause/lstm_slearner/)

### RL Paradigm
Hybrid method: **sequence-aware causal reward estimation (LSTM S-learner) + offline Q-learning (LSTM-DQN)**. Same three-phase pipeline as Method 6, but replaces the flat-state GBR S-learner with a sequence-aware LSTM S-learner that operates on the same variable-length event prefixes used by the downstream DQN. This makes the entire pipeline fully sequence-aware.

### Three-Phase Pipeline

**Phase 1 — Train LSTM S-learner (causal outcome model):**
An LSTM_SLearner is trained per intervention: the model encodes the event prefix with an LSTM, embeds the action, and jointly predicts the scalar case outcome. Trained via MSE regression on (prefix, action, case_outcome) triplets with early stopping.

**Phase 2 — Counterfactual-augmented causal rewards:**
Same protocol as Method 6 Phase 2 (factual rewriting + one synthetic row per counterfactual action per terminal transition). Because the S-learner is sequence-aware, both the factual and counterfactual reward estimates condition on the full temporal process history rather than a flat state vector.

**Phase 3 — Train LSTM-DQN on causal rewards:**
Identical to Method 6 Phase 3 and Method 1: standard LSTM-DQN with backward TD (Q3→Q2→Q1) trained on the causally-corrected rewards.

### State Representation
Both the S-learner and the DQN use prefix-based sequential state — identical encoding (6 continuous features + activity embedding, padded to max_len=10). This distinguishes Method 7 from Method 6 where the S-learner uses a flat state.

### Network Architecture (Phase 1 — LSTM_SLearner)
```
Activity embedding: nn.Embedding(n_activities, emb_dim=32)
LSTM: input=(emb_dim+6)=38, hidden=128, n_layers=2, dropout=0.2
  → final hidden state h_n[-1]: 128-dim (prefix encoding)

Action embedding: nn.Embedding(max_actions=3, action_emb_dim=16)

Fusion: concat([prefix_encoding (128), action_embedding (16)]) → 144-dim
FC head: Linear(144 → 128) → ReLU → Dropout(0.2) → Linear(128 → 1)
Output: scalar predicted outcome (normalised)
```

Unlike LSTM_DQN which outputs Q-values for all actions simultaneously, LSTM_SLearner takes a specific action as input and outputs a single scalar outcome prediction for that (prefix, action) pair.

### Network Architecture (Phase 3 — LSTM-DQN)
Identical to Method 1: three Q-networks (Q1, Q2, Q3), hidden=128, n_layers=2, target networks τ=0.005.

### Training Procedure
```
Phase 1 (S-learner):
  Optimizer: Adam, slearner_lr=1e-3
  Loss: MSE(predicted_outcome, normalised_case_outcome)
  Batch size: 256
  Epochs: 150, patience: 10
  DataLoader: seeded for reproducibility

Phase 2:
  For each terminal transition at intervention i:
    causal_reward = SLearner_i.predict(prefix, action_taken)
    replace observed reward with causal_reward (de-normalised)

Phase 3 (DQN — identical to Method 1):
  Optimizer: Adam, dqn_lr=1e-3
  Batch size: 256
  Epochs: 50, patience: 10
  Backward TD: Q3 → Q2 → Q1
  Tau: 0.005, Gamma: 0.99
```

### Key Hyperparameters (defaults)
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
| max_len | 10 |

---

## Method 8: LSTM-DQN-TabPFN (lstm_dqn_tabpfn/)

### RL Paradigm
Hybrid method: **TabPFN causal S-learner + offline Q-learning (LSTM-DQN)**. Same three-phase pipeline as ProCause EconML (Method 6), but replaces the GradientBoostingRegressor with a **TabPFN regressor** — a pretrained transformer that performs in-context learning over the training set at inference time, requiring no gradient-based training in Phase 1.

### Three-Phase Pipeline

**Phase 1 — Train TabPFN S-learner:**
A `TabPFNRegressor` is fitted per intervention on (flat state, action, case_outcome) triplets. TabPFN is a pretrained transformer model that uses the training set as context at prediction time (in-context learning); `fit()` stores the data and `predict()` runs a transformer forward pass over it. No gradient descent occurs. A `StandardScaler` is fitted on the state features for normalisation. Outcomes are normalised (mean/std) before fitting.

**Phase 2 — Counterfactual-augmented causal rewards:**
Same protocol as Method 6 Phase 2 — factual reward rewritten with `TabPFN_i.predict([state | a_obs])` (denormalised) and one synthetic row per remaining candidate action per terminal transition.

**Phase 3 — Train LSTM-DQN on causal rewards:**
Standard LSTM-DQN with backward TD (Q3→Q2→Q1), identical to Method 1.

### State Representation
- **S-learner (Phase 1 & 2)**: Flat 16-dim state vector (same as K-means, CQL, ProCause EconML). Stored in the `state` column of the transition DataFrame during convert_data.
- **DQN (Phase 3)**: Prefix-based sequential state — same LSTM encoding as Method 1.

### Model Architecture (Phase 1 — TabPFN)
```
Estimator: TabPFNRegressor (pretrained transformer, in-context learning)
  device: cuda if available, else cpu
  random_state: seed + int_idx (for reproducibility)

Input:  [16-dim StandardScaler-normalised state] + [1-dim action] = 17-dim
Output: scalar predicted outcome (normalised), then denormalised for reward

Subsampling: if training set > max_samples (default 10000), a random subset
  is used for fitting (TabPFN has a practical limit on context size)
```

No explicit hyperparameters to tune for Phase 1 — TabPFN is pretrained and used as-is.

### Model Architecture (Phase 3 — LSTM-DQN)
Identical to Method 1: three Q-networks (Q1, Q2, Q3), hidden=128, n_layers=2, target networks τ=0.005.

### Training Procedure
```
Phase 1 (TabPFN):
  For each intervention i:
    states = flat 16-dim state vectors (StandardScaler normalised)
    X = [states | action_column]    # 17-dim
    y = normalised case_outcome
    TabPFN_i.fit(X, y)              # no gradient descent

Phase 2:
  For each terminal transition at intervention i:
    causal_reward = TabPFN_i.predict([state | action]) * outcome_std + outcome_mean
    replace observed reward with causal_reward

Phase 3 (DQN — identical to Method 1):
  Optimizer: Adam, dqn_lr=1e-3
  Batch size: 256
  Epochs: 50, patience: 10
  Backward TD: Q3 → Q2 → Q1
  Tau: 0.005, Gamma: 0.99
```

### Key Hyperparameters (defaults)
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

## Method 9: LSTM-DQN-DragonNet (lstm_dqn_dragonnet/)

### RL Paradigm
Hybrid method: **LSTM-DragonNet causal outcome model + offline Q-learning (LSTM-DQN)**. Same three-phase pipeline as ProCause LSTM (Method 7), but replaces the plain LSTM S-learner with a **DragonNet** architecture that adds a propensity head and targeted regularisation loss (Shi et al., 2019). The targeted regularisation encourages outcome predictions to be locally invariant to propensity perturbations, improving causal reward quality under confounding.

### Three-Phase Pipeline

**Phase 1 — Train LSTM-DragonNet:**
An `LSTM_DragonNet` is trained per intervention on (prefix, action, case_outcome) triplets. The model jointly optimises a factual outcome loss, a propensity loss (predicting which action was taken), and a targeted regularisation term. Only the outcome heads are used downstream.

**Phase 2 — Counterfactual-augmented causal rewards:**
Same protocol as Method 6 Phase 2. For each terminal transition, the factual reward is rewritten as `head_{a_obs}(Φ(prefix))` (denormalised) and one synthetic row is added per other candidate action `a` with reward `head_a(Φ(prefix))` (denormalised). The propensity head and `eps` parameter are not used at this stage.

**Phase 3 — Train LSTM-DQN on causal rewards:**
Standard LSTM-DQN with backward TD (Q3→Q2→Q1), identical to Method 1.

### State Representation
Both the DragonNet and the DQN use prefix-based sequential state — identical LSTM encoding (emb_dim=32, hidden=128, n_layers=2). This is the same as ProCause LSTM (Method 7).

### Network Architecture (Phase 1 — LSTM_DragonNet)
```
Shared LSTM trunk Φ:
  Activity embedding: nn.Embedding(n_activities, emb_dim=32)
  LSTM: input=(emb_dim+6)=38, hidden=128, n_layers=2, dropout=0.2
  Output: representation r = Φ(prefix), 128-dim

Per-action outcome heads (one per action for this intervention):
  head_a: Linear(128→64) → ReLU → Dropout(0.2) → Linear(64→1)
  Predicts scalar outcome if action a were taken.
  Int. 0 & 1: 2 heads. Int. 2: 3 heads.

Propensity head:
  Linear(128→64) → ReLU → Dropout(0.2) → Linear(64→n_actions)
  Outputs logits → softmax gives P(action | prefix)

Targeted regularisation scalar:
  self.eps: nn.Parameter (scalar, one per intervention model, learned)
```

### DragonNet Loss
```
factual_loss    = MSE(head_a(r), observed_outcome_normalised)

propensity_loss = CrossEntropy(prop_logits, observed_action)

targeted_loss   = MSE(outcome_pred + eps * (1 / p_obs), observed_outcome_normalised)
  where p_obs = softmax(prop_logits)[observed_action], clamped >= 1e-6

total_loss = factual_loss + alpha_prop * propensity_loss + alpha_targeted * targeted_loss
```

### Training Procedure
```
Phase 1 (DragonNet):
  Optimizer: Adam, lr=1e-3, weight_decay=1e-5
  Loss: DragonNet loss (factual + propensity + targeted)
  Batch size: 256
  Epochs: 150, patience: 15
  Early stopping criterion: validation factual MSE only
  LR scheduler: ReduceLROnPlateau, factor=0.5, patience=5

Phase 2:
  For each terminal transition at intervention i:
    causal_reward = head_action(Φ(prefix)) * outcome_std + outcome_mean
    replace observed reward with causal_reward
    (propensity head and eps not used here)

Phase 3 (DQN — identical to Method 1):
  Optimizer: Adam, dqn_lr=1e-3
  Batch size: 256
  Epochs: 50, patience: 10
  Backward TD: Q3 → Q2 → Q1
  Tau: 0.005, Gamma: 0.99
```

### Key Hyperparameters (defaults)
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

### Relationship to CFRNet
DragonNet supersedes the earlier CFRNet implementation. CFRNet used MMD/IPM loss to balance representations across treatment groups, which hurt performance on confounded offline data by destroying confounder information in the shared encoder. DragonNet instead models confounding explicitly via the propensity head, leaving the representation free to retain all predictive signal.

---

## Cross-Method Comparison Table

| Aspect | LSTM-DQN | RIMS | K-means | Single CQL | Multi CQL | ProCause EconML | ProCause LSTM | TabPFN | DragonNet |
|--------|----------|------|---------|------------|-----------|-----------------|---------------|--------|-----------|
| **RL type** | Offline DQN | Online DQN | Offline FQI | Offline CQL | Offline CQL | Causal + Offline DQN | Causal + Offline DQN | Causal + Offline DQN | Causal + Offline DQN |
| **State (policy)** | Seq. prefix | Seq. prefix | Flat 17-dim | Seq. prefix | Seq. prefix | Seq. prefix | Seq. prefix | Seq. prefix | Seq. prefix |
| **State (causal)** | — | — | — | — | — | Flat 17-dim | Seq. prefix | Flat 17-dim | Seq. prefix |
| **Causal model** | — | — | — | — | — | GBR S-learner | LSTM S-learner | TabPFN S-learner | LSTM DragonNet |
| **Network** | LSTM+FC | LSTM+FC | KMeans+table | LSTM-DQN (masked) | 3×LSTM-DQN | GBR+LSTM-DQN | LSTM+LSTM-DQN | TabPFN+LSTM-DQN | DragonNet+LSTM-DQN |
| **Causal model trains?** | — | — | — | — | — | Yes (GBR) | Yes (LSTM) | No (pretrained) | Yes (LSTM) |
| **Confounding mechanism** | None | None | None | None | None | S-learner | S-learner | S-learner | Propensity + targeted reg. |
| **Backward TD** | Yes | Yes | Yes | Yes | Yes | Yes (DQN) | Yes (DQN) | Yes (DQN) | Yes (DQN) |
| **Target network** | Yes τ=0.005 | Yes τ=0.005 | No | Yes τ=0.005 | Yes τ=0.005 | Yes τ=0.005 | Yes τ=0.005 | Yes τ=0.005 | Yes τ=0.005 |
| **CQL penalty** | No | No | No | Yes α=1.0 | Yes α=1.0 | No | No | No | No |
| **Reward signal** | Observed | Simulated | Observed | Observed | Observed | Causal reward | Causal reward | Causal reward | Causal reward |
| **Loss fn** | MSE | MSE | Mean agg. | MSE+CQL | MSE+CQL | MSE | MSE | MSE | MSE+CE+targeted |
| **Env interaction** | None | Simulator | None | None | None | None | None | None | None |

---

## Reproducibility: Seeding Strategy

All methods implement a 3-layer seeding strategy:

**1. Global seed** (controls weight initialisation + all random ops):
```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
```

**2. DataLoader seed** (controls shuffle order per epoch):
```python
g = torch.Generator()
g.manual_seed(seed)
DataLoader(..., worker_init_fn=seed_worker, generator=g)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
```

**3. sklearn seed** (for K-means and GBR):
```python
KMeans(random_state=seed)
GradientBoostingRegressor(random_state=seed)
```

All 5 experimental seeds (defined in `shared/experiment_config.SEEDS = [42, 123, 456, 789, 1024]`, also re-listed in `run_seeds.py`) are applied identically across all methods, enabling paired statistical comparison.

---

## Data Generation, Conversion and Loading Pipeline

### 1. Raw event-log generation (`shared/generate_data.py`)

The single shared entry point is `shared.generate_data.main()` (also exposed as `generate_data.py` at the project root). CLI arguments:

| Flag | Default | Meaning |
|------|---------|---------|
| `--n_cases` | 10000 | Number of independent cases to simulate |
| `--confounded` | off | Use confounded (bank+RCT) generation |
| `--delta` | 0.95 | Bank-policy fraction in the mix (only with `--confounded`) |
| `--seed` | 42 | Master RNG seed |

The script delegates to `shared.data_utils`:

- **`get_simbank_params(n_cases, seed, rct)`** builds the SimBank `params` dict (the simulator config). Key fields:
  - `simulation_start = datetime(2024, 3, 20, 8, 0)`
  - `intervention_info`: defines the three interventions, their action spaces (`[2, 2, 3]`) and `RCT_timing = [1000, 1000, 1000]` (RCT randomisation budget per intervention).
  - `policies_info`: bank-policy thresholds (`amount`, `est_quality`, `min_quality`, `max_noc`, `max_nor`, `min_amount_contact_cust`, etc.)
  - `log_cols`: `[case_nr, activity, timestamp, elapsed_time, cum_cost, est_quality, unc_quality, amount, interest_rate, discount_factor, outcome, quality, noc, nor, min_interest_rate]`
- **`generate_rct_data(n_cases, seed)`**: instantiates `simulation.PresProcessGenerator(params, seed)` with `RCT=True`, calls `run_simulation_normal(n_cases)`, returns `(df, params)`.
- **`generate_confounded_data(n_cases, seed, delta=0.95)`**:
  1. Run a bank-policy simulation (`RCT=False`) with seed `seed` → `df_bank`.
  2. Run an RCT simulation with seed `seed * 10` whose `simulation_start` equals the bank run's `simulation_end` (so timestamps don't overlap) → `df_rct`.
  3. Mix via `confounding_level.set_delta(df_bank, df_rct, delta)`. With `delta=0.95`, ~95% of cases come from the bank policy and ~5% from the RCT, producing the confounded log used in confounded experiments.
- **`save_pickle(obj, path)`** / **`load_pickle(path)`** are thin `pickle` wrappers used everywhere.

Outputs (under `data/`):
- `simbank_{RCT|CONF}_{n_cases}_raw.pkl` — the event-log DataFrame
- `simbank_{RCT|CONF}_{n_cases}_params.pkl` — the SimBank `params` dict (also used at evaluation time to instantiate a fresh simulator)

### 2. Method-specific transition extraction (`{method}/convert_data.py`)

Each method reads the shared raw file and emits its own transition table. There are two shapes:

**(a) Sequential prefix transitions** (LSTM, RIMS Q-network input, Single CQL, Multi CQL, ProCause LSTM, EconML, TabPFN, DragonNet):
- For each case, walk events. When the activity matches an intervention activity:
  - `prefix` = list of preceding event dicts (variable length)
  - `action` = integer action; for intervention 2 derived via `get_ir_action(interest_rate)` (`0.07→0`, `0.08→1`, `0.09→2`)
  - `reward` = `outcome` for terminal, `0.0` otherwise
  - `next_prefix` = prefix at next intervention or `[]` if terminal
  - `terminal`, `intervention ∈ {0,1,2}`, `next_intervention ∈ {1,2,-1}`
- Branching paths handled: only-int0, int0→int1, int0→int2, int0→int1→int2.
- Train/val split is by `case_nr` via `split_train_val(df, val_ratio=0.2, seed)` so that all transitions of a case live in the same split.

**(b) Flat-state transitions** (K-means; ProCause-EconML / TabPFN keep both shapes — flat for the S-learner, sequential for the DQN):
- `state` = `extract_state(prev_event, activity_counts)`, a 17-dim vector built from `BASE_FEATURES` + `TRACKED_ACTIVITIES` counts up to the intervention point.
- `next_state` is `extract_state` at the next intervention, or zeros for terminal.

**(c) RIMS simulator artefacts** (`rims/convert_data.py`): trains the two LSTM simulator models (P_T, P_C), fits the empirical transition matrix, fits the acceptance logistic regression, collects initial prefixes. Bundle saved as `data/rims_{suffix}_{n}_simulator.pkl`.

Per-method output paths (suffix = `RCT` or `CONF`):
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

`--steps {1,2,3}` truncates the transitions to the first `s` interventions and adds a `_steps{s}` suffix to the filenames.

### 3. Vocabulary, normalisation and DataLoader construction

- **`build_vocab_and_stats(df_train)`** in `shared/lstm_utils.py` walks all events in `prefix` + `next_prefix` and returns `(activity_to_idx, feat_means, feat_stds)`, mapping each activity string to a unique integer ≥ 1 (0 is reserved for padding) and computing per-feature mean/std for `FEATURE_COLS`.
- **`encode(prefixes, activity_to_idx, feat_means, feat_stds, max_len, n_activities)`** turns a list of prefixes into `(acts, feats, lens)` numpy arrays:
  - `acts ∈ ℤ^{N×max_len}` (integer encoding) or one-hot `ℤ^{N×max_len×n_activities}` (onehot encoding)
  - `feats ∈ ℝ^{N×max_len×6}`, normalised per feature
  - `lens ∈ ℤ^{N}`, the true (unpadded) length, used by `pack_padded_sequence`
- **`encode_prefix(prefix, cfg)`** is the single-prefix variant used at evaluation time; reads the same `feat_means/stds` and `activity_to_idx` from the saved checkpoint config so train- and eval-time encodings agree exactly.
- DataLoaders are built with `worker_init_fn=seed_worker` and `generator=torch.Generator().manual_seed(seed)` so shuffle order is reproducible.

### 4. Loading at training time

Each `{method}/train.py` does:
```python
df_train = load_pickle(f"data/{method}_{suffix}_{n}_trans_train{step_tag}.pkl")
df_val   = load_pickle(f"data/{method}_{suffix}_{n}_trans_val{step_tag}.pkl")
activity_to_idx, feat_means, feat_stds = build_vocab_and_stats(df_train)   # for LSTM-based methods
```
and packs everything used at inference (vocab, stats, max_len, network shape, `n_actions`) into a `cfg` dict that is saved alongside the model state dict in the `.pth` checkpoint.

---

## Evaluation: Technical Details

### Shared evaluation harness (`shared/evaluation.py`)

All methods evaluate via a single shared driver, `evaluate_policy(policy_fn, n_episodes, params, seed, ...)`:

```python
gen = simulation.PresProcessGenerator(params, seed=seed)
for i in range(n_episodes):
    if reset_fn: reset_fn()                              # e.g. clear running activity_counts
    prefix_list = gen.start_simulation_inference(seed_to_add=i)
    while gen.int_points_available:
        prefix     = prefix_list[0][:-1]                 # all events up to (not incl.) intervention
        prev_event = prefix[-1]
        int_idx    = gen.current_int_index
        action     = policy_fn(prev_event, int_idx[, prefix])   # use_prefix toggles arity
        prefix_list = gen.continue_simulation_inference(action)
    outcome = float(pd.DataFrame(gen.end_simulation_inference())["outcome"].iloc[-1])
```

Key points:
- A **fresh `PresProcessGenerator`** is built from the saved `params` dict so evaluation uses the exact simulator configuration as data generation.
- `seed_to_add=i` perturbs the per-episode random stream, so episodes are independent but reproducible (paired across methods given the same `seed`).
- Default evaluation: `n_episodes = 1000`, eval seed `seed=1042` (`--seed`), training seed re-loaded via `--train_seed` (default 42).
- Action counts per intervention are tracked (`action_counts[int_idx][action] += 1`) and reported alongside the outcome.

### Baselines (`shared/evaluation.bank_policy`, `random_policy`)

`bank_policy(prev_event, int_idx)` is a verbatim re-implementation of the bank's heuristic from SimBank's `extra_flow_conditions.py`:
- **Int 0 (`choose_procedure`)**: `priority` if `amount > 50000` and `est_quality ≥ 5`, else `standard`.
- **Int 1 (`time_contact_HQ`)**: `contact_headquarters` if `noc < 2`, `unc_quality == 0`, `amount > 10000`, `est_quality ≥ 2`, else `skip_contact`.
- **Int 2 (`set_ir_3_levels`)**: `7%` if `amount > 60000`, `8%` if `amount > 30000`, else `9%`. Action indices align with `IR_LEVELS = [0.07, 0.08, 0.09]`.

`random_policy(prev_event, int_idx)` is `np.random.randint(0, [2, 2, 3][int_idx])`.

### Method-specific policy wrappers

Each `{method}/evaluate.py` defines a thin policy class implementing `__call__(prev_event, int_idx, prefix=None)` and an optional `reset()`:

| Method | Policy class | Inputs to Q | Notes |
|--------|-------------|-------------|-------|
| LSTM | `LSTMPolicy` | `encode_prefix(prefix, cfg)` → `(acts, feats, lens)` | `argmax(Q[int_idx][:N_ACTIONS[int_idx]])` |
| RIMS | `RIMSPolicy` | same as LSTM | shares the LSTM-DQN inference path |
| K-means | `KMeansPolicy` | `extract_state(prev_event, counts)`; cluster via fitted `scaler` + `KMeans` | `argmax(Q_table[int_idx][cluster])`; `reset()` clears `activity_counts` |
| Single CQL | `SingleCQLPolicy` | `encode_prefix(prefix, cfg)` | masks invalid action slots; argmax over `N_ACTIONS[int_idx]` |
| Multi CQL | `MultiCQLPolicy` | `encode_prefix(prefix, cfg)` | per-intervention Q-network |
| ProCause / TabPFN / DragonNet | `LSTMPolicy`-style | `encode_prefix(prefix, cfg)` | only the DQN heads are queried at eval; the causal model is unused |

Beyond the trained `--steps` boundary, every policy falls back to `bank_policy(prev_event, int_idx)` so partial-step models still produce well-defined action choices.

### Result aggregation (`run_seeds.py`)

`run_seeds.py` runs `train.py` and `evaluate.py` once per seed in `SEEDS = [42, 123, 456, 789, 1024]` and writes a JSON line per seed to a temporary `--results_file`, then aggregates mean ± std across seeds. CLI:
```
python run_seeds.py --method {kmeans|lstm|rims|singleModelCQL|multiModelCQL} \
                    --n_cases 10000 [--confounded] [--n_episodes 1000]
                    [-- <extra_train_args>]
```
Causal hybrids (TabPFN, DragonNet, ProCause variants) are run from their per-method scripts under `scripts/` plus `run_seeds.py` patterns; the same paired-seed protocol applies.

The reported metric in `print_results()` is mean per-case `outcome` (and its std); each non-Bank policy is also reported as `% gain over Bank` (`(avg/bank_avg − 1) × 100`).

`print_action_dist()` prints the per-intervention action-mix percentages — these are the inputs to all action-distribution figures in `generate_performance_graphs.py` / `plot_results.py`.

---

## Data Pipeline Summary

```
generate_data.py  (or shared/generate_data.py)
  └─ generates: data/simbank_{RCT|CONF}_{n_cases}_raw.pkl
                data/simbank_{RCT|CONF}_{n_cases}_params.pkl

{method}/convert_data.py   (reads shared raw file, writes method-specific transitions)
  └─ lstm:         data/lstm_{suffix}_{n}_trans_{train|val}.pkl     (prefix sequences)
  └─ rims:         data/rims_{suffix}_{n}_simulator.pkl             (P_T, P_C, transition matrix, acceptance model)
  └─ kmeans:       data/kmeans_{suffix}_{n}_trans_{train|val}.pkl   (flat vectors)
  └─ singleCQL:    data/single_cql_{suffix}_{n}_trans_{train|val}.pkl
  └─ multiCQL:     data/multi_cql_{suffix}_{n}_trans_{train|val}.pkl
  └─ econml:       data/procause_econml_{suffix}_{n}_trans_{train|val}.pkl
  └─ lstm_sl:      data/procause_lstm_{suffix}_{n}_trans_{train|val}.pkl
  └─ tabpfn:       data/lstm_dqn_tabpfn_{suffix}_{n}_trans_{train|val}.pkl  (prefix + flat state)
  └─ dragonnet:    data/lstm_dqn_dragonnet_{suffix}_{n}_trans_{train|val}.pkl  (prefix sequences)

{method}/train.py  (reads converted data, writes model)
  └─ models/{method}_{suffix}_{n}_s{seed}.pkl (or .pt)

{method}/evaluate.py  (loads model + params, runs SimBank episodes)
  └─ results/{method}_{suffix}_{n}_seed{seed}_eval.pkl
```
