# Offline RL Implementation Guide for SimBank

This document outlines the structural changes that should be applied consistently across all offline RL methods (CQL, BCQ, IQL, etc.) for the SimBank environment.

---

## 1. Skipped Intervention Handling

### Problem
In SimBank, the **priority path skips Intervention 2** (contact_headquarters). Previously, fake transitions were fabricated for skipped interventions, which corrupts the learning signal.

### Solution
**Only include transitions for interventions that actually occurred.**

### Trajectory Types
```
1. Full Trajectory:      Int1 → Int2 → Int3 → Terminal
2. Skip Int2 (Priority): Int1 → Int3 → Terminal
3. Terminal at Int2:     Int1 → Int2 → Terminal
4. Terminal at Int1:     Int1 → Terminal
```

### Transition Structure
```python
transition = {
    'state': [...],              # State BEFORE the intervention
    'action': 0/1/2,             # Action taken at this intervention
    'next_state': [...],         # State at NEXT intervention (None if terminal)
    'reward': 0.0 or outcome,    # 0 for intermediate, outcome for terminal
    'terminal': True/False,      # Is this the last intervention?
    'next_intervention': 2/3/-1  # Which intervention comes next (-1 if terminal)
}
```

### Key Rules
1. **Intermediate reward = 0** (only terminal states get the outcome)
2. **next_intervention** tracks what comes next for proper target computation
3. **Never fabricate transitions** for interventions that didn't happen

### Code Template: Extract Transitions
```python
def extract_transitions(df):
    """Extract transitions with proper skipped intervention handling."""
    transitions_int1 = []
    transitions_int2 = []
    transitions_int3 = []

    for case_nr, case_group in df.groupby('case_nr'):
        case_group = case_group.sort_values('timestamp').reset_index(drop=True)
        final_reward = float(case_group['outcome'].iloc[-1])

        # Find interventions
        int1_rows = case_group[case_group['activity'].isin(['start_standard', 'start_priority'])]
        int2_rows = case_group[case_group['activity'].isin(['contact_headquarters', 'skip_contact'])]
        int3_rows = case_group[case_group['activity'] == 'calculate_offer']

        has_int1 = not int1_rows.empty
        has_int2 = not int2_rows.empty
        has_int3 = not int3_rows.empty

        if not has_int1:
            continue  # Skip cases without Int1

        # Get Int1 details
        int1_idx = int1_rows.index[0]
        action1 = 1 if case_group.loc[int1_idx, 'activity'] == 'start_priority' else 0
        state1 = get_state_before(case_group, int1_idx)

        # === FULL TRAJECTORY ===
        if has_int2 and has_int3:
            int2_idx = int2_rows.index[0]
            int3_idx = int3_rows.index[0]

            state2 = get_state_before(case_group, int2_idx)
            state3 = get_state_before(case_group, int3_idx)
            action2 = 0 if case_group.loc[int2_idx, 'activity'] == 'contact_headquarters' else 1
            action3 = get_ir_action(case_group.loc[int3_idx, 'interest_rate'])

            transitions_int1.append({
                'state': state1, 'action': action1, 'next_state': state2,
                'reward': 0.0, 'terminal': False, 'next_intervention': 2
            })
            transitions_int2.append({
                'state': state2, 'action': action2, 'next_state': state3,
                'reward': 0.0, 'terminal': False, 'next_intervention': 3
            })
            transitions_int3.append({
                'state': state3, 'action': action3, 'next_state': None,
                'reward': final_reward, 'terminal': True, 'next_intervention': -1
            })

        # === SKIP INT2 (Priority Path) ===
        elif not has_int2 and has_int3:
            int3_idx = int3_rows.index[0]
            state3 = get_state_before(case_group, int3_idx)
            action3 = get_ir_action(case_group.loc[int3_idx, 'interest_rate'])

            transitions_int1.append({
                'state': state1, 'action': action1, 'next_state': state3,
                'reward': 0.0, 'terminal': False, 'next_intervention': 3  # Skip to Int3!
            })
            # NO Int2 transition - it didn't happen!
            transitions_int3.append({
                'state': state3, 'action': action3, 'next_state': None,
                'reward': final_reward, 'terminal': True, 'next_intervention': -1
            })

        # === TERMINAL AT INT2 ===
        elif has_int2 and not has_int3:
            int2_idx = int2_rows.index[0]
            state2 = get_state_before(case_group, int2_idx)
            action2 = 0 if case_group.loc[int2_idx, 'activity'] == 'contact_headquarters' else 1

            transitions_int1.append({
                'state': state1, 'action': action1, 'next_state': state2,
                'reward': 0.0, 'terminal': False, 'next_intervention': 2
            })
            transitions_int2.append({
                'state': state2, 'action': action2, 'next_state': None,
                'reward': final_reward, 'terminal': True, 'next_intervention': -1
            })
            # NO Int3 transition

        # === TERMINAL AT INT1 ===
        else:
            transitions_int1.append({
                'state': state1, 'action': action1, 'next_state': None,
                'reward': final_reward, 'terminal': True, 'next_intervention': -1
            })
            # NO Int2 or Int3 transitions

    return transitions_int1, transitions_int2, transitions_int3
```

### Target Computation (Backward Training: Q3 → Q2 → Q1)
```python
# Q3: Always terminal
target_q3 = normalize(reward)

# Q2: Either goes to Q3 or is terminal
if terminal:
    target_q2 = normalize(reward)
else:
    target_q2 = gamma * max(Q3(next_state))

# Q1: Can go to Q2, Q3 (skip), or be terminal
if terminal:
    target_q1 = normalize(reward)
elif next_intervention == 2:
    target_q1 = gamma * max(Q2(next_state))
elif next_intervention == 3:  # Skipped Int2
    target_q1 = gamma * max(Q3(next_state))
```

---

## 2. Confounded Data Generation

### Problem
Using `RCT=False` gives 100% bank policy data with no exploration, which limits learning.

### Solution
Mix bank policy data with RCT data using `confounding_level.set_delta()`.

### Code Template
```python
from SimBank import simulation
from SimBank import confounding_level
from copy import deepcopy

def generate_confounded_data(n_cases, seed, delta=0.95):
    """
    Generate confounded data: delta% bank policy + (1-delta)% RCT.

    Args:
        n_cases: Number of cases to generate
        seed: Random seed
        delta: Confounding level (0.95 = 95% bank policy, 5% RCT)
    """

    # Step 1: Generate bank policy data (RCT=False)
    params_bank = get_dataset_params(n_cases, seed, rct=False)
    gen_bank = simulation.PresProcessGenerator(params_bank, seed)
    data_bank = gen_bank.run_simulation_normal(n_cases)
    df_bank = pd.DataFrame(data_bank)

    # Step 2: Generate RCT data (RCT=True)
    params_rct = get_dataset_params(n_cases, seed * 10, rct=True)
    params_rct["simulation_start"] = deepcopy(gen_bank.simulation_end)
    gen_rct = simulation.PresProcessGenerator(params_rct, seed * 10)
    data_rct = gen_rct.run_simulation_normal(n_cases)
    df_rct = pd.DataFrame(data_rct)

    # Step 3: Mix with delta
    df_combined = confounding_level.set_delta(
        data=df_bank,
        data_RCT=df_rct,
        delta=delta  # 0.95 = 95% bank, 5% RCT
    )

    return df_combined

def generate_rct_data(n_cases, seed):
    """Generate pure RCT data (100% random actions)."""
    params = get_dataset_params(n_cases, seed, rct=True)
    gen = simulation.PresProcessGenerator(params, seed)
    data = gen.run_simulation_normal(n_cases)
    return pd.DataFrame(data)
```

### Dataset Parameters Template
```python
def get_dataset_params(n_cases, seed, rct=True):
    return {
        "train_size": n_cases,
        "test_size": int(n_cases * 0.1),
        "simulation_start": datetime(2024, 3, 20, 8, 0),
        "random_seed_train": seed,
        "intervention_info": {
            "name": ["choose_procedure", "time_contact_HQ", "set_ir_3_levels"],
            "activities": [
                ["start_standard", "start_priority"],
                ["contact_headquarters", "skip_contact"],
                ["calculate_offer"]
            ],
            "actions": [
                ["start_standard", "start_priority"],
                ["contact_headquarters", "skip_contact"],
                [0.07, 0.08, 0.09]
            ],
            "action_depth": [1, 1, 1],
            "action_width": [2, 2, 3],
            "RCT": rct,  # True for random, False for bank policy
            "RCT_timing": [1000, 1000, 1000]
        },
        "policies_info": {
            "general": "real",
            "choose_procedure": {"amount": 50000, "est_quality": 5},
            "time_contact_HQ": "real",
            "set_ir_3_levels": "real",
        }
    }
```

---

## 3. Bank Policy Baseline

### Problem
Hardcoded bank policies don't match SimBank's actual policy, leading to unfair comparisons.

### Solution
Use the exact policy from `SimBank/extra_flow_conditions.py`.

### Code Template
```python
def bank_policy(prev_event, intervention_index):
    """
    Exact bank policy from SimBank's extra_flow_conditions.py.
    Use this for evaluation baseline.
    """

    if intervention_index == 0:  # choose_procedure
        # Priority if: amount > 50000 AND est_quality >= 5
        amount = prev_event.get("amount", 0)
        est_quality = prev_event.get("est_quality", 0)

        if amount > 50000 and est_quality >= 5:
            return 1  # start_priority
        else:
            return 0  # start_standard

    elif intervention_index == 1:  # time_contact_HQ
        # Contact if: noc < 2 AND unc_quality == 0 AND amount > 10000 AND est_quality >= 2
        noc = prev_event.get("noc", 0)
        unc_quality = prev_event.get("unc_quality", 1)
        amount = prev_event.get("amount", 0)
        est_quality = prev_event.get("est_quality", 0)

        contact_condition = (
            noc < 2 and
            unc_quality == 0 and
            amount > 10000 and
            est_quality >= 2
        )

        if contact_condition:
            return 0  # contact_headquarters
        else:
            return 1  # skip_contact

    elif intervention_index == 2:  # set_ir_3_levels
        # Interest rate based on amount thresholds
        amount = prev_event.get("amount", 0)

        if amount > 60000:
            return 0  # 7% (0.07)
        elif amount > 30000:
            return 1  # 8% (0.08)
        else:
            return 2  # 9% (0.09)
```

### Evaluation Template
```python
def evaluate_policy(policy_func, n_cases, dataset_params, seed):
    """Evaluate a policy through SimBank simulator."""

    case_gen = simulation.PresProcessGenerator(dataset_params, seed)
    total_performance = 0

    for case_nr in range(n_cases):
        prefix_list = case_gen.start_simulation_inference(seed_to_add=case_nr)

        while case_gen.int_points_available:
            prefix_without_int = prefix_list[0][:-1]
            prev_event = prefix_without_int[-1]
            current_int_index = case_gen.current_int_index

            action = policy_func(prev_event, current_int_index)
            prefix_list = case_gen.continue_simulation_inference(action)

        full_case = pd.DataFrame(case_gen.end_simulation_inference())
        outcome = full_case["outcome"].iloc[-1]
        total_performance += outcome

    return total_performance / n_cases
```

---

## 4. State Representation

### Base Features (5 dimensions)
```python
BASE_FEATURES = ['amount', 'est_quality', 'unc_quality', 'cum_cost', 'elapsed_time']

def extract_base_features(event):
    """Extract 5 base features from an event."""
    return [
        float(event.get('amount', 0)),
        float(event.get('est_quality', 0)),
        float(event.get('unc_quality', 0)),
        float(event.get('cum_cost', 0)),
        float(event.get('elapsed_time', 0))
    ]
```

### Activity Counts (11 dimensions)
Activity counts encode the process position, helping the model understand where it is in the workflow.

```python
ALL_ACTIVITIES = [
    'initiate_application',
    'start_standard',
    'start_priority',
    'collect_documents',
    'assess_quality',
    'contact_headquarters',
    'skip_contact',
    'calculate_offer',
    'contact_customer',
    'submit_file',
    'file_submitted'
]

def count_activities(case_group, up_to_index):
    """Count activity occurrences up to (but not including) the given index."""
    counts = {act: 0 for act in ALL_ACTIVITIES}

    for idx in range(up_to_index):
        activity = case_group.loc[idx, 'activity']
        if activity in counts:
            counts[activity] += 1

    return [counts[act] for act in ALL_ACTIVITIES]
```

### State Dimensions by Intervention
| Intervention | State Dim | Features |
|--------------|-----------|----------|
| Int1 (choose_procedure) | 5 | Base features only |
| Int2 (contact_HQ) | 16 | Base (5) + Activity counts (11) |
| Int3 (set_ir) | 16 | Base (5) + Activity counts (11) |

### Full State Extraction
```python
def get_state(event, case_group, event_index, intervention_index):
    """Get state representation for a given intervention."""

    base_features = extract_base_features(event)

    if intervention_index == 0:
        # Int1: Only base features
        return base_features  # 5 dims
    else:
        # Int2/Int3: Base + activity counts
        activity_counts = count_activities(case_group, event_index)
        return base_features + activity_counts  # 16 dims
```

---

## 5. Summary Checklist

When implementing a new offline RL method for SimBank:

- [ ] **Transitions**: Only include transitions for interventions that actually happened
- [ ] **Rewards**: Intermediate = 0, terminal = outcome
- [ ] **next_intervention**: Track what comes next for target computation
- [ ] **Data Generation**: Use `set_delta()` for confounded data (95% bank + 5% RCT)
- [ ] **Bank Policy**: Use exact policy from `extra_flow_conditions.py`
- [ ] **State Representation**: 5 base features + 11 activity counts for Int2/Int3
- [ ] **Backward Training**: Train Q3 first, then Q2 using Q3, then Q1 using Q2/Q3
- [ ] **Handle Skip**: Q1 target uses Q3 directly when Int2 is skipped

---

## 6. File Structure Template

```
your_method/
├── generate_data.py    # Data generation (RCT and confounded)
├── convert_data.py     # Extract transitions with proper handling
├── train.py            # Training with backward order
├── evaluate.py         # Evaluation against bank baseline
└── config.py           # Shared constants and utilities
```
