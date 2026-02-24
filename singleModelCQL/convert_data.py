"""
Convert SimBank data to CQL transitions.
"""
import sys
import os
import argparse
import pandas as pd
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
os.chdir(project_root)

from shared import load_pickle, save_pickle, count_activities, extract_state, get_ir_action, STATE_DIM


def extract_transitions(df):
    """Extract transitions handling skipped interventions."""
    transitions = []
    stats = {'full': 0, 'skip_int2': 0, 'end_int2': 0, 'end_int1': 0, 'skip': 0}

    for _, group in df.groupby('case_nr'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        outcome = float(group['outcome'].iloc[-1])

        # Find Int1
        int1_rows = group[group['activity'].isin(['start_standard', 'start_priority'])]
        if int1_rows.empty or int1_rows.index[0] == 0:
            stats['skip'] += 1
            continue

        int1_idx = int1_rows.index[0]
        action1 = 1 if group.loc[int1_idx, 'activity'] == 'start_priority' else 0
        state1 = extract_state(group.loc[int1_idx - 1], count_activities(group, int1_idx))

        # Find Int2, Int3
        int2_rows = group[group['activity'].isin(['contact_headquarters', 'skip_contact']) & (group.index > int1_idx)]
        int3_rows = group[(group['activity'] == 'calculate_offer') & (group.index > int1_idx)]
        has_int2, has_int3 = not int2_rows.empty, not int3_rows.empty
        terminal = np.zeros(STATE_DIM, dtype=np.float32)

        if has_int2 and has_int3:
            stats['full'] += 1
            int2_idx, int3_idx = int2_rows.index[0], int3_rows.index[0]
            action2 = 0 if group.loc[int2_idx, 'activity'] == 'contact_headquarters' else 1
            action3 = get_ir_action(group.loc[int3_idx].get('interest_rate', 0.08))
            state2 = extract_state(group.loc[int2_idx - 1], count_activities(group, int2_idx))
            state3 = extract_state(group.loc[int3_idx - 1], count_activities(group, int3_idx))

            transitions.append({'state': state1, 'action': action1, 'reward': 0.0, 'next_state': state2, 'terminal': False, 'intervention': 0, 'next_intervention': 1})
            transitions.append({'state': state2, 'action': action2, 'reward': 0.0, 'next_state': state3, 'terminal': False, 'intervention': 1, 'next_intervention': 2})
            transitions.append({'state': state3, 'action': action3, 'reward': outcome, 'next_state': terminal, 'terminal': True, 'intervention': 2, 'next_intervention': -1})

        elif not has_int2 and has_int3:
            stats['skip_int2'] += 1
            int3_idx = int3_rows.index[0]
            action3 = get_ir_action(group.loc[int3_idx].get('interest_rate', 0.08))
            state3 = extract_state(group.loc[int3_idx - 1], count_activities(group, int3_idx))

            transitions.append({'state': state1, 'action': action1, 'reward': 0.0, 'next_state': state3, 'terminal': False, 'intervention': 0, 'next_intervention': 2})
            transitions.append({'state': state3, 'action': action3, 'reward': outcome, 'next_state': terminal, 'terminal': True, 'intervention': 2, 'next_intervention': -1})

        elif has_int2:
            stats['end_int2'] += 1
            int2_idx = int2_rows.index[0]
            action2 = 0 if group.loc[int2_idx, 'activity'] == 'contact_headquarters' else 1
            state2 = extract_state(group.loc[int2_idx - 1], count_activities(group, int2_idx))

            transitions.append({'state': state1, 'action': action1, 'reward': 0.0, 'next_state': state2, 'terminal': False, 'intervention': 0, 'next_intervention': 1})
            transitions.append({'state': state2, 'action': action2, 'reward': outcome, 'next_state': terminal, 'terminal': True, 'intervention': 1, 'next_intervention': -1})
        else:
            stats['end_int1'] += 1
            transitions.append({'state': state1, 'action': action1, 'reward': outcome, 'next_state': terminal, 'terminal': True, 'intervention': 0, 'next_intervention': -1})

    return transitions, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cases', type=int, default=10000)
    parser.add_argument('--confounded', action='store_true')
    args = parser.parse_args()

    suffix = "CONF" if args.confounded else "RCT"
    base = f"data/single_cql_{suffix}_{args.n_cases}"

    print(f"\n{'='*50}")
    print(f"Converting {suffix} data to transitions")
    print('='*50)

    # Convert train
    df_train = pd.DataFrame(load_pickle(f"{base}_train.pkl"))
    train_trans, train_stats = extract_transitions(df_train)
    print(f"\nTrain: {len(train_trans)} transitions")
    print(f"  full={train_stats['full']}, skip_int2={train_stats['skip_int2']}")

    # Convert val
    df_val = pd.DataFrame(load_pickle(f"{base}_val.pkl"))
    val_trans, val_stats = extract_transitions(df_val)
    print(f"Val: {len(val_trans)} transitions")

    # Save
    save_pickle(pd.DataFrame(train_trans), f"{base}_trans_train.pkl")
    save_pickle(pd.DataFrame(val_trans), f"{base}_trans_val.pkl")
    save_pickle({'state_dim': STATE_DIM, 'n_actions': [2, 2, 3], 'train_stats': train_stats}, f"{base}_meta.pkl")

    print(f"\n[OK] Saved transitions")
    print(f"Next: python singleModelCQL/train.py --n_cases {args.n_cases} {'--confounded' if args.confounded else ''}")


if __name__ == "__main__":
    main()
