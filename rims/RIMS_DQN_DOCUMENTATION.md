# RIMS-DQN: Category 5 — Learned Simulator + Online RL

## Overview

RIMS-DQN is a Category 5 offline RL approach that:
1. Builds a process simulator from historical event logs
2. Trains an RL agent online inside that simulator via epsilon-greedy exploration
3. Evaluates the learned policy in the real SimBank environment

The key hypothesis: online exploration in the learned simulator generates **counterfactual state-action pairs** (actions not present in the training data), enabling better policy learning than purely offline methods that are limited to the logged behavior.

---

## Pipeline

```
Historical logs (generate_data.py)
        |
        v
  +-----------+     Mine structure      +------------------+
  | Raw event |  ===================>   | Simulator artifact|
  |   logs    |   (convert_data.py)     | (.pkl)           |
  +-----------+                         +------------------+
                                               |
                                               v
                                    +---------------------+
                                    | LearnedSimBankEnv   |
                                    | (simulator.py)      |
                                    +---------------------+
                                               |
                                    Online training (train.py)
                                    20k episodes, eps-greedy
                                               |
                                               v
                                    +---------------------+
                                    | LSTM-DQN Q-networks |
                                    | (Q1, Q2, Q3)        |
                                    +---------------------+
                                               |
                                    Evaluate in REAL SimBank
                                    (evaluate.py)
                                               |
                                               v
                                         Results
```

---

## Component Classification: What Is Learned vs What Is Not

This is the critical distinction for interpreting the results. The simulator is a mix of:
- **Learned from data** (ML models and mined rules)
- **Domain knowledge** (hardcoded from SimBank source code)

### A. LEARNED FROM HISTORICAL LOGS (purely data-driven)

These components are extracted solely from the offline event logs. No SimBank source code knowledge is used.

#### A1. P_T — Processing Time Model
- **What**: LSTM predicting `log(duration_seconds + 1)` of the next event given a prefix
- **Architecture**: `Embedding(n_act, 32) -> LSTM(38, 64, 1 layer) -> Linear(64, 1)`
- **Training**: MSE loss on log-duration targets, from all consecutive event pairs in the log
- **File**: `convert_data.py` lines 32-45
- **Usage**: Called in simulator whenever a new event is generated, to predict how much time passes
- **Impact on results**: Low — processing times mainly affect `elapsed_time` feature, which has minor influence on reward

#### A2. P_C — Control Flow Model
- **What**: LSTM classifier predicting the next activity given a prefix
- **Architecture**: `Embedding(n_act, 32) -> LSTM(38, 64, 1 layer) -> Linear(64, n_activities)`
- **Training**: Cross-entropy loss on next-activity targets, from all consecutive event pairs
- **File**: `convert_data.py` lines 48-61
- **Usage**: Called in simulator at non-intervention branching points (e.g., after `validate_application`, decides whether `email_customer` or `call_customer` happens next). Outputs are masked by the transition matrix (A4).
- **NOT used for**: Intervention decisions (agent's choice), post-offer accept/refuse (handled by A3)
- **Impact on results**: Medium — determines the non-intervention process flow between agent decisions

#### A3. Acceptance Model — Logistic Regression
- **What**: Predicts customer accept vs refuse/cancel after `calculate_offer`
- **Architecture**: `LogisticRegression` on 4 features: `interest_rate`, `min_interest_rate`, `amount`, `elapsed_time`
- **Training**: Fitted on all post-`calculate_offer` transitions in the log (89.2% accuracy on 10k cases)
- **File**: `convert_data.py` lines 120-174
- **Usage**: After `calculate_offer`, replaces P_C for the accept/refuse decision. Also implements the mined cancellation rule: if `interest_rate < min_interest_rate`, the application is cancelled.
- **Why needed**: P_C cannot learn the sharp threshold relationship between interest rate and acceptance probability from limited data. This is the single most important stochastic decision in the process — it directly determines whether the bank earns money or loses money.
- **Impact on results**: Critical — without this, the agent cannot learn meaningful interest rate decisions

#### A4. Transition Matrix — Mined Process Structure
- **What**: A dict mapping each activity to its set of observed successor activities
- **Mining**: Scan all consecutive activity pairs in the event log
- **File**: `convert_data.py` lines 177-195
- **Usage**: Masks P_C's softmax output — logits for unobserved transitions are set to `-1e9`, so P_C can only choose among activities that actually followed the current activity in the historical data
- **Example**: `validate_application -> {contact_headquarters, skip_contact, email_customer, call_customer}`
- **Why needed**: Without this, P_C could generate impossible activity sequences (e.g., `calculate_offer` before `validate_application`), wasting training episodes on unrealistic trajectories
- **Impact on results**: Medium — prevents nonsensical process flows, but doesn't change probabilities within valid transitions

#### A5. Initial Prefixes — Starting States
- **What**: Real case prefixes from the log, up to the first intervention point
- **Mining**: For each case, extract events before `start_standard` / `start_priority`
- **File**: `convert_data.py` lines 198-212
- **Usage**: `env.reset()` samples a random prefix to start each training episode
- **Impact on results**: Low-Medium — determines the distribution of starting states the agent trains on

#### A6. Feature Normalization Statistics
- **What**: Per-feature mean and standard deviation computed from the log
- **Features**: `amount`, `est_quality`, `unc_quality`, `interest_rate`, `cum_cost`, `elapsed_time`
- **File**: `convert_data.py` lines 86-90
- **Usage**: All feature inputs are normalized as `(value - mean) / std` before feeding to any LSTM
- **Impact on results**: Low — standard preprocessing, same as all other methods

---

### B. DOMAIN KNOWLEDGE (hardcoded from SimBank source)

These components encode knowledge about the SimBank process that is NOT learned from data. They come from reading the SimBank source code (`activity_execution.py`, `extra_flow_conditions.py`).

#### B1. Cost Structure — `COSTS` dict
```python
COSTS = {
    "initiate_application": 0,    "start_standard": 10,
    "start_priority": 5000,       "validate_application": 20,
    "contact_headquarters": 3000, "skip_contact": 0,
    "email_customer": 10,         "call_customer": 20,
    "calculate_offer": 400,       "cancel_application": 30,
    "receive_acceptance": 10,     "receive_refusal": 10,
    "stop_application": 0,
}
```
- **Source**: `SimBank/activity_execution.py` lines 16-28
- **File**: `simulator.py` lines 12-26
- **Usage**: Each event adds `COSTS[activity]` to `cum_cost`. Special case: `contact_headquarters` cost is dynamic: `unc_quality * 1000 + 1000`
- **Why hardcoded**: Costs are deterministic business rules — they don't vary between cases and could technically be mined from the log by looking at `cum_cost` differences, but hardcoding is simpler and more reliable

#### B2. Reward Function — `_calc_outcome()`
```python
if accepted:
    risk_factor = (10 - quality) / 200
    df = 0.03 + risk_factor
    future_earnings = amount * (1 + ir) ** 10
    discounted = future_earnings / (1 + df) ** 10
    reward = discounted - cum_cost - amount - 100
else:
    reward = -cum_cost - 100
```
- **Source**: `SimBank/activity_execution.py` lines 55-68
- **File**: `simulator.py` lines 217-229
- **Why hardcoded**: The reward formula is the objective function — the agent optimizes this. It must match reality exactly, otherwise the agent optimizes for the wrong thing. This is standard in RL: the reward function defines the task.

#### B3. Feature Update Rules — `_make_event()`
For each activity, deterministic rules update the case state:
- **`contact_headquarters`**: `cum_cost += unc_quality * 1000 + 1000`, `unc_quality = 0`, `est_quality = true_quality` (reveals true quality)
- **`call_customer`**: `unc_quality -= 3` (clamped >= 0), `noc += 1`, `est_quality` re-estimated with reduced uncertainty
- **`email_customer`**: `unc_quality -= 2` (clamped >= 0), `noc += 1`, `est_quality` re-estimated
- **`calculate_offer`**: `interest_rate`, `discount_factor`, `min_interest_rate` set based on state (handled in `step()`)
- **`receive_acceptance`**: `outcome` computed via reward formula
- **`receive_refusal`**: `nor += 1` (number of refusals)
- **`cancel_application`**: `outcome` computed (negative)

- **Source**: `SimBank/activity_execution.py` lines 286-374
- **File**: `simulator.py` lines 161-215
- **Why hardcoded**: These are deterministic business rules that define how the process state evolves. They could theoretically be mined from data (observe how features change per activity), but they are exact and deterministic — learning them would add noise without benefit.

#### B4. Interest Rate Levels
```python
IR_LEVELS = [0.07, 0.08, 0.09]
```
- **Source**: `SimBank/activity_execution.py`
- **File**: `simulator.py` line 36
- **Why hardcoded**: These define the agent's action space at intervention 2. The action space must match the real environment.

#### B5. Intervention-Action Mapping
```python
INTERVENTION_ACTIONS = {
    0: {0: 'start_standard', 1: 'start_priority'},
    1: {0: 'contact_headquarters', 1: 'skip_contact'},
    2: 'calculate_offer',
}
```
- **File**: `simulator.py` lines 30-34
- **Why hardcoded**: Defines what agent actions mean at each intervention point. This is the task definition, not something to learn.

#### B6. Terminal Activities
```python
TERMINAL_ACTIVITIES = {'cancel_application', 'receive_acceptance', 'stop_application'}
```
- **File**: `simulator.py` line 28
- **Note**: `receive_refusal` is NOT terminal — it loops back to `calculate_offer` for a re-offer attempt

#### B7. Minimum Interest Rate Formula
```python
best_case_costs = event['cum_cost'] + COSTS['receive_acceptance'] + FIXED_COST
min_ir = ((best_case_costs / amount + 1) ** (1/10)) * (1 + df_rate) - 1
min_ir = ceil(min_ir * 100) / 100
```
- **Source**: `SimBank/activity_execution.py`
- **File**: `simulator.py` lines 264-266
- **Why hardcoded**: Determines the profitability threshold — if `interest_rate < min_interest_rate`, the bank cancels the application (would lose money). This is a business rule.

#### B8. Constants
- `FIXED_COST = 100` — per-application fixed overhead
- `LOAN_LENGTH = 10` — 10-year loans
- **Source**: `SimBank/activity_execution.py`

---

### C. Summary Classification Table

| Component | Source | Type | Impact |
|-----------|--------|------|--------|
| P_T (processing time) | Historical logs | LSTM (learned) | Low |
| P_C (control flow) | Historical logs | LSTM (learned) | Medium |
| Acceptance model | Historical logs | Logistic regression (mined) | Critical |
| Transition matrix | Historical logs | Exact mining | Medium |
| Initial prefixes | Historical logs | Exact extraction | Low-Medium |
| Feature normalization | Historical logs | Statistics | Low |
| Cost structure | SimBank source | Hardcoded | Medium |
| Reward function | SimBank source | Hardcoded | Critical |
| Feature update rules | SimBank source | Hardcoded | Medium |
| Interest rate levels | SimBank source | Hardcoded (action space) | — |
| Intervention mapping | SimBank source | Hardcoded (task definition) | — |
| Terminal activities | SimBank source | Hardcoded | Low |
| Min interest rate formula | SimBank source | Hardcoded | Medium |

**Bottom line**: The simulator is ~50% learned from data, ~50% hardcoded domain knowledge. The learned parts handle stochastic behavior (timing, branching, acceptance). The hardcoded parts handle deterministic business rules (costs, rewards, state transitions). The agent (LSTM-DQN) is fully learned via online RL.

---

## How the Simulator Works (Technical)

### Episode lifecycle

1. **`reset()`**: Sample a random initial prefix from the training log. Set intervention index to 0.

2. **`step(action)`** at intervention `k`:
   a. Map `action` to an activity (e.g., action=1 at int0 -> `start_priority`)
   b. Predict duration using P_T
   c. Create event with deterministic feature updates (`_make_event`)
   d. If int2 (`calculate_offer`): set `interest_rate = IR_LEVELS[action]`, compute `min_interest_rate`
   e. Append event to prefix
   f. **Roll forward** with P_C/P_T until next intervention or terminal:
      - If last activity is `calculate_offer`: use acceptance model (not P_C) for accept/refuse
      - Otherwise: use P_C (masked by transition matrix) to predict next activity
      - If next activity is an intervention: stop, return to agent
      - If next activity is terminal: compute reward, done=True
      - If `receive_refusal`: loop back to intervention 2 (agent gets to make another offer)

3. **Steps ablation**: If the next intervention is beyond the `steps` limit, auto-complete remaining interventions using bank policy (`_auto_complete_bank`).

### Transition mask mechanics
```
P_C outputs logits: [0.1, 2.3, -0.5, 1.2, 0.8, ...]  (one per activity)
Transition mask for current activity: [2, 5, 7]        (valid successor indices)
Masked logits:      [-1e9, -1e9, 2.3, -1e9, -1e9, 0.8, -1e9, 1.2, ...]
Softmax -> sample from valid successors only
```

### Acceptance model mechanics
```
After calculate_offer:
  1. Check cancellation: if interest_rate < min_interest_rate -> cancel_application
  2. Extract features: [ir, min_ir, amount, elapsed_time]
  3. Normalize with saved scaler (mean/std from training data)
  4. Logistic regression: logit = X @ coef + intercept
  5. accept_prob = sigmoid(logit)
  6. Sample: uniform() < accept_prob -> receive_acceptance, else receive_refusal
```

---

## How the Agent Trains (Technical)

### Architecture
- **3 separate LSTM-DQN networks** (Q1, Q2, Q3), one per intervention point
- Each: `Embedding(n_act, 32) -> LSTM(38, 128, 2 layers, dropout=0.2) -> FC(128,128) -> ReLU -> FC(128, n_actions)`
- Same architecture as the offline LSTM-DQN method (shared code in `shared/lstm_utils.py`)

### Training loop (20,000 episodes)
```
for each episode:
    prefix = env.reset()
    while not done and steps < 10:
        action = epsilon_greedy(Q[int_idx], prefix)
        next_prefix, reward, done = env.step(action)
        replay[int_idx].push(prefix, action, reward, next_prefix, done)

    for each intervention i:
        sample batch from replay[i]
        compute target: reward/1000 + gamma * (1-done) * max(Q_target[i+1](next_state))
        MSE loss, gradient step, soft target update (tau=0.005)
```

### Backward-chaining Q-learning
- **Q3** (interest rate): terminal intervention, target = `reward / 1000`
- **Q2** (contact HQ): bootstraps from Q3_target
- **Q1** (procedure): bootstraps from Q2_target

This chain means Q1 implicitly learns the long-term value of procedure choice by considering downstream decisions.

### Epsilon schedule
- Linear decay from 1.0 to 0.05 over 20,000 episodes (decay rate = 5e-5)
- Full exploration at start, ~5% exploration at end

### Reward scaling
- Rewards clipped to [-5000, 10000] and divided by 1000
- This keeps Q-values in a manageable range (~[-5, 10]) for stable gradient descent

---

## Debugging History — Issues Encountered and Resolved

### Issue 1: P_T NaN Loss
- **Cause**: Raw logs contained NaN timestamps, producing NaN durations
- **Fix**: Skip samples with non-finite or negative durations in `prepare_sim_data()`

### Issue 2: P_C CrossEntropyLoss Type Error
- **Cause**: CrossEntropyLoss requires LongTensor targets, but SimDataset returned FloatTensor
- **Fix**: Added `target_dtype='long'` parameter to SimDataset

### Issue 3: Simulator NaN Probabilities
- **Cause**: Initial prefixes contain NaN feature values (e.g., `interest_rate=nan` before any offer is made). NaN features -> NaN LSTM output -> NaN softmax
- **Fix**: NaN sanitization in `_encode_prefix()` and uniform fallback in `_predict_next_activity()`

### Issue 4: NaN Q-Network Weights (THE MAJOR BUG)
- **Cause**: `shared/lstm_utils.py` `encode()` and `encode_prefix()` did `float(e.get(col, 0))` which preserves NaN. Every batch contained NaN features -> NaN gradients -> NaN weights after first update
- **Fix**: Added `try/except + np.isfinite()` guards in both `encode()` and `encode_prefix()`

### Issue 5: Q-Target Formula Inverted
- **Cause**: `target = dones_t * reward + (1 - dones_t) * gamma * next_q` — this multiplies reward by `done` (0 for non-terminal), dropping the reward signal entirely for non-terminal transitions
- **Fix**: `target = reward + (1 - dones_t) * gamma * next_q` — standard Bellman equation
- **Impact**: Without this fix, Q1 and Q2 received zero reward signal and could not learn

### Issue 6: Degenerate Policy (Std=100%, Contact=100%, 7%=100%)
- **Cause**: Combination of issues 4 and 5 — NaN weights meant all Q-values were NaN, and `argmax(NaN)` = index 0
- **Fix**: Resolved by fixing issues 4 and 5

### Issue 7: `receive_refusal` Classified as Terminal
- **Cause**: In real SimBank, a refusal leads to a re-offer at a different rate. The simulator incorrectly ended the episode on refusal.
- **Fix**: Removed `receive_refusal` from `TERMINAL_ACTIVITIES`, added loop-back to intervention 2

### Issue 8: Contact HQ Missing Quality Revelation
- **Cause**: `_make_event()` charged the HQ cost but didn't set `unc_quality=0` or reveal true quality
- **Fix**: Added `event['unc_quality'] = 0` and `event['est_quality'] = true_quality`

### Issue 9: `min_interest_rate` Double-Counting
- **Cause**: `best_case_costs` included `COSTS['calculate_offer']` (400) on top of `cum_cost` which already included it
- **Fix**: Use `event['cum_cost']` (post-offer) instead of `prev_event['cum_cost'] + COSTS['calculate_offer']`

---

## Results

### Final Performance (RCT, 10k cases, 3-step, seed=42)

| Policy | Average Reward | Std | Gain vs Bank |
|--------|---------------|-----|-------------|
| Bank | 433.20 | 2964.64 | — |
| Random | -106.06 | 6582.10 | -124.5% |
| **RIMS-DQN** | **3599.03** | **8091.94** | **+730.8%** |

### Learned Action Distribution
```
Procedure:  Std=60%, Pri=40%   (Bank: Std=93%, Pri=7%)
Contact:    Contact=14%, Skip=86% (Bank: Contact=0%, Skip=100%)
Interest:   7%=0%, 8%=1%, 9%=99% (Bank: 7%=70%, 8%=30%, 9%=0%)
```

### Comparison to All Methods (RCT, 3-step, 10k cases)

| Method | Category | Avg Reward | Gain vs Bank |
|--------|----------|-----------|-------------|
| Bank (baseline) | — | 438.2 | — |
| Random | — | -95.6 | -121.8% |
| LSTM-DQN | 3 (offline RL) | -1960.9 | -547.5% |
| KMeans | 1 (rule-based) | 4386.9 | +901.0% |
| CQL-MM | 4 (offline RL + conservatism) | 3567.2 | +714.0% |
| CQL-SM | 4 (offline RL + conservatism) | 3607.0 | +723.1% |
| **RIMS-DQN** | **5 (learned sim + online RL)** | **3599.0** | **+730.8%** |

### Key Observations

1. **RIMS-DQN >> LSTM-DQN**: +730.8% vs -547.5%. Both use LSTM-DQN architecture, but RIMS trains online in a simulator while LSTM trains offline on fixed transitions. This directly supports the Category 5 hypothesis.

2. **RIMS-DQN ~ CQL methods**: Comparable to CQL-MM (+714%) and CQL-SM (+723%). The simulator-based approach achieves similar performance to conservative offline RL methods.

3. **KMeans still leads**: The simple clustering approach (+901%) outperforms all RL methods on this dataset, suggesting that for this problem size, the optimal policy may be simpler than what deep RL captures.

4. **Interest rate strategy**: The agent learned to almost always offer 9% (the highest rate), while the bank offers 7% most of the time. Higher rates maximize profit per accepted loan, and the acceptance model learned that most customers still accept at 9%.

5. **Priority procedure**: 40% priority (vs bank's 7%) — the agent invests in faster processing for high-value loans where the upfront cost is offset by higher acceptance rates and faster turnaround.

---

## Files

| File | Purpose | Lines |
|------|---------|-------|
| `rims/generate_data.py` | Generate raw SimBank event logs | Unchanged from shared |
| `rims/convert_data.py` | Train P_T, P_C; mine transition matrix & acceptance model | ~400 |
| `rims/simulator.py` | Gymnasium-style learned environment | ~425 |
| `rims/train.py` | Online LSTM-DQN training with replay buffer | ~280 |
| `rims/evaluate.py` | Evaluate in real SimBank | ~100 |
| `shared/lstm_utils.py` | Shared LSTM-DQN architecture, encoding functions | ~100 |
