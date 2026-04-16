# Technical Reference: All Offline RL Methods for Prescriptive Process Monitoring

This document provides exhaustive technical detail per method, intended as raw material for thesis writing.
All details are extracted directly from the codebase.

---

## Shared Infrastructure

### State Representations

Two fundamentally different state representations are used across methods:

**Prefix-based (sequential) state — LSTM, RIMS, ProCause LSTM, DragonNet (DQN phase):**
- A variable-length sequence of events from the start of a case up to (not including) the intervention point
- Each timestep in the sequence has 7 features: 6 continuous features + 1 activity identifier
- Continuous features (FEATURE_COLS): `amount`, `est_quality`, `unc_quality`, `interest_rate`, `cum_cost`, `elapsed_time`
- Activity identifier: either integer index (integer encoding) or one-hot vector (onehot encoding)
- Sequences are padded to `max_len=10` and packed with `pack_padded_sequence` for efficient LSTM processing

**Flat state vector — K-means, CQL (both), ProCause EconML, TabPFN (causal phase), DragonNet (causal phase):**
- Fixed 16-dimensional vector: 5 base features + 11 activity-count features
- Base features (5): `amount`, `est_quality`, `unc_quality`, `interest_rate`, `cum_cost`
- Activity counts (11): cumulative occurrence count for each of the 11 tracked activities up to the intervention point
- `STATE_DIM = 16`

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

### Data Generation (SimBank)
- `generate_rct_data(n_cases, seed)` — randomised controlled trial: actions assigned uniformly at random
- `generate_confounded_data(n_cases, seed, delta)` — confounded: bank's heuristic policy influences action selection with probability delta (default 0.95)
- Output: raw DataFrame of event-log rows + params dict (SimBank simulator parameters)
- Saved to: `data/simbank_{RCT|CONF}_{n_cases}_raw.pkl` and `data/simbank_{RCT|CONF}_{n_cases}_params.pkl`

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
- **IR_LEVELS**: maps action indices to actual interest rate values (0.05, 0.07, 0.10)
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
Conservative Q-Learning (CQL) with a **single neural network** that handles all three intervention points simultaneously. The intervention identity is injected via one-hot encoding concatenated to the state vector. Uses backward TD bootstrapping + CQL penalty.

### How Historical Data Is Used
Same as K-means: flat state vectors from offline transitions. The CQL penalty discourages the Q-network from assigning high values to out-of-distribution actions by penalising the logsumexp of Q-values over all actions.

### State Representation
Flat state vector: `[16-dim state] + [3-dim one-hot intervention ID]` = **19-dim input**.

The one-hot intervention encoding allows a single network to distinguish which decision point it is at:
- Intervention 0: [1, 0, 0]
- Intervention 1: [0, 1, 0]
- Intervention 2: [0, 0, 1]

This is what distinguishes Single-Model from Multi-Model CQL — one network for all interventions vs. separate networks per intervention.

### Network Architecture
```
Input: 19-dim (16 state + 3 one-hot intervention)
Hidden layers: Linear(19 → 256) → ReLU → Linear(256 → 256) → ReLU → Linear(256 → 3)
Output: 3 Q-values (for actions 0, 1, 2)
  → For interventions 0 and 1: only first 2 Q-values are used (2 valid actions)
  → For intervention 2: all 3 Q-values are used
```

One network, one target network. Target updated with soft updates (τ=0.005).

### CQL Loss
```
Q_pred = Q_net(state)[action_taken]   # predicted Q for taken action
Q_target = reward + gamma * (1-terminal) * max(Q_target_net(next_state))  # TD target

TD_loss = MSE(Q_pred, Q_target)

CQL_penalty = logsumexp(Q_net(state), dim=-1).mean() - Q_net(state)[action_taken].mean()
  (penalises large Q-values for actions not in the data)

Total_loss = TD_loss + alpha * CQL_penalty     alpha = 1.0 (default)
```

### Reward Normalisation
Before training, rewards from terminal transitions are normalised: `reward = (reward - mean) / std`. This is computed per-intervention from the training set.

### State Normalisation
Per-intervention `StandardScaler` fitted on training states. Three scalers (one per intervention), but the same network sees all three (with different one-hot).

### Data Conversion (singleModelCQL/convert_data.py)
Transition extraction identical to multiModelCQL, but `convert_data.py` reads from the shared raw file and does the train/val split (80/20) internally. Saves `_trans_train.pkl` and `_trans_val.pkl`.

### Backward TD Routing
When computing Q1 targets, `next_intervention` is used to route to the correct next Q-value:
- next_intervention=1: `max(Q_net(next_state | one_hot=[0,1,0]))`
- next_intervention=2: `max(Q_net(next_state | one_hot=[0,0,1]))`

### Training Procedure
```
Optimizer: Adam, lr=1e-4
Batch size: 256
Alpha (CQL penalty weight): 1.0
Gamma: 0.99
Tau: 0.005
Epochs: 200
Patience: 20 (early stopping on validation loss)
```

### Evaluation
At each intervention point, construct `[state | one_hot_intervention]`, pass through the single network, mask invalid actions, take argmax.

### Key Hyperparameters (defaults)
| Parameter | Value |
|-----------|-------|
| hidden_dim | 256 |
| lr | 1e-4 |
| batch_size | 256 |
| alpha (CQL) | 1.0 |
| gamma | 0.99 |
| tau | 0.005 |
| epochs | 200 |
| patience | 20 |
| steps | 3 |

---

## Method 5: Multi-Model CQL (multiModelCQL/)

### RL Paradigm
Conservative Q-Learning (CQL) with **separate neural networks per intervention point**. Same CQL penalty as single-model but each network is specialised to one intervention. Structurally closer to the LSTM-DQN approach.

### How Historical Data Is Used
Same offline transition tuples as single-model CQL. No one-hot intervention encoding needed because each network only sees transitions from its own intervention.

### State Representation
Differs by intervention:
- **Intervention 0 (Q1)**: 5-dim (base features only: `amount`, `est_quality`, `unc_quality`, `interest_rate`, `cum_cost`)
- **Interventions 1 and 2 (Q2, Q3)**: 16-dim (base features + 11 activity counts)

Rationale: at intervention 0, no process activities have yet occurred, so activity counts are all zero and add no information.

### Network Architecture
```
Q1: Linear(5 → 256) → ReLU → Linear(256 → 256) → ReLU → Linear(256 → 2)
Q2: Linear(16 → 256) → ReLU → Linear(256 → 256) → ReLU → Linear(256 → 2)
Q3: Linear(16 → 256) → ReLU → Linear(256 → 256) → ReLU → Linear(256 → 3)
```

Three networks, three target networks. All updated with soft Polyak updates (τ=0.005).

### CQL Loss (per network)
Identical formula to single-model CQL:
```
Total_loss = MSE(Q_pred, Q_target) + alpha * (logsumexp(Q(state)) - Q(state)[action])
```

### State Normalisation
Per-intervention `StandardScaler`. Q1 scaler fitted on 5-dim states, Q2/Q3 on 16-dim states.

### Reward Normalisation
Same as single-model: per-intervention mean/std normalisation of terminal rewards.

### Backward TD
Same routing logic as single-model, but routing uses separate networks:
- Q1 target routes to `max(Q2(s'))` or `max(Q3(s'))` depending on `next_intervention`
- Q2 target always routes to `max(Q3(s'))`

### Training Procedure
```
Optimizer: Adam, lr=1e-4
Batch size: 256
Alpha (CQL): 1.0
Gamma: 0.99
Tau: 0.005
Epochs: 200
Patience: 20
```

### Key Hyperparameters (defaults)
| Parameter | Value |
|-----------|-------|
| hidden_dim | 256 |
| lr | 1e-4 |
| batch_size | 256 |
| alpha (CQL) | 1.0 |
| gamma | 0.99 |
| tau | 0.005 |
| epochs | 200 |
| patience | 20 |

---

## Method 6: ProCause EconML S-learner (procause/econml_slearner/)

### RL Paradigm
Hybrid method: **causal reward estimation (S-learner) + offline Q-learning (LSTM-DQN)**. Operates in three sequential phases. The S-learner estimates causal treatment effects to replace potentially confounded observed rewards; the downstream DQN then trains on these causally-corrected rewards using the same backward TD procedure as Method 1.

### Three-Phase Pipeline

**Phase 1 — Train GBR S-learner (causal outcome model):**
A GradientBoostingRegressor is trained per intervention as an S-learner: a single model `f(state, action) → outcome` fitted on all (flat 16-dim state, action, case_outcome) triplets. The case outcome is the final result of the entire case — not the intermediate step reward. Three separate GBR models, one per intervention.

**Phase 2 — Compute causal rewards:**
The trained S-learners replace the observed terminal reward of each transition with a causally-estimated reward: `GBR_i.predict([state | action_taken])`. This reward reshaping is intended to reduce confounding bias present in the logged rewards under the bank's historical policy. Non-terminal rewards remain zero.

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

**Phase 2 — Compute causal rewards:**
Identical in purpose to Method 6 Phase 2: the S-learner predictions replace observed terminal rewards with causally-estimated rewards. Because the S-learner is sequence-aware, the causal reward estimate conditions on the full temporal process history rather than just the flat state vector.

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

**Phase 2 — Compute causal rewards:**
Identical contract to ProCause EconML: terminal transition rewards are replaced with `TabPFN_i.predict([state | action_taken])`, denormalised back to the original outcome scale. Non-terminal rewards remain 0.0.

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

**Phase 2 — Compute causal rewards:**
Terminal transition rewards are replaced with the factual outcome head prediction: `head_action(Φ(prefix))`, denormalised. The propensity head and `eps` parameter are not used at this stage.

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
| **State (policy)** | Seq. prefix | Seq. prefix | Flat 16-dim | Flat 19-dim | Flat 5/16-dim | Seq. prefix | Seq. prefix | Seq. prefix | Seq. prefix |
| **State (causal)** | — | — | — | — | — | Flat 16-dim | Seq. prefix | Flat 16-dim | Seq. prefix |
| **Causal model** | — | — | — | — | — | GBR S-learner | LSTM S-learner | TabPFN S-learner | LSTM DragonNet |
| **Network** | LSTM+FC | LSTM+FC | KMeans+table | MLP 256×2 | 3×MLP 256×2 | GBR+LSTM-DQN | LSTM+LSTM-DQN | TabPFN+LSTM-DQN | DragonNet+LSTM-DQN |
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

All 5 experimental seeds (default: `SEEDS = [42, 123, 456, 789, 1011]`) are applied identically across all methods, enabling paired statistical comparison.

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
