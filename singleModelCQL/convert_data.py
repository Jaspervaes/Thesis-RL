"""Convert SimBank data to Single-Model CQL transitions."""
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


def extract_transitions(df, steps=3):
    rows = []
    z = np.zeros(STATE_DIM, dtype=np.float32)

    for _, group in df.groupby('case_nr'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        outcome = float(group['outcome'].iloc[-1])

        int1_rows = group[group['activity'].isin(['start_standard', 'start_priority'])]
        if int1_rows.empty or int1_rows.index[0] == 0:
            continue

        i1 = int1_rows.index[0]
        a1 = 1 if group.loc[i1, 'activity'] == 'start_priority' else 0
        s1 = extract_state(group.loc[i1 - 1], count_activities(group, i1))

        if steps == 1:
            rows.append({'state': s1, 'action': a1, 'reward': outcome,
                         'next_state': z.copy(), 'terminal': True, 'intervention': 0, 'next_intervention': -1})
            continue

        int2_rows = group[
            group['activity'].isin(['contact_headquarters', 'skip_contact']) & (group.index > i1)
        ]
        int3_rows = group[(group['activity'] == 'calculate_offer') & (group.index > i1)]
        has2, has3 = not int2_rows.empty, not int3_rows.empty

        if steps == 2:
            if has2:
                i2 = int2_rows.index[0]
                a2 = 0 if group.loc[i2, 'activity'] == 'contact_headquarters' else 1
                s2 = extract_state(group.loc[i2 - 1], count_activities(group, i2))
                rows += [
                    {'state': s1, 'action': a1, 'reward': 0.0, 'next_state': s2,
                     'terminal': False, 'intervention': 0, 'next_intervention': 1},
                    {'state': s2, 'action': a2, 'reward': outcome,
                     'next_state': z.copy(), 'terminal': True, 'intervention': 1, 'next_intervention': -1},
                ]
            else:
                rows.append({'state': s1, 'action': a1, 'reward': outcome,
                             'next_state': z.copy(), 'terminal': True, 'intervention': 0, 'next_intervention': -1})
            continue

        # steps == 3
        if has2 and has3:
            i2, i3 = int2_rows.index[0], int3_rows.index[0]
            a2 = 0 if group.loc[i2, 'activity'] == 'contact_headquarters' else 1
            a3 = get_ir_action(group.loc[i3].get('interest_rate', 0.08))
            s2 = extract_state(group.loc[i2 - 1], count_activities(group, i2))
            s3 = extract_state(group.loc[i3 - 1], count_activities(group, i3))
            rows += [
                {'state': s1, 'action': a1, 'reward': 0.0, 'next_state': s2,
                 'terminal': False, 'intervention': 0, 'next_intervention': 1},
                {'state': s2, 'action': a2, 'reward': 0.0, 'next_state': s3,
                 'terminal': False, 'intervention': 1, 'next_intervention': 2},
                {'state': s3, 'action': a3, 'reward': outcome,
                 'next_state': z.copy(), 'terminal': True, 'intervention': 2, 'next_intervention': -1},
            ]
        elif not has2 and has3:
            i3 = int3_rows.index[0]
            a3 = get_ir_action(group.loc[i3].get('interest_rate', 0.08))
            s3 = extract_state(group.loc[i3 - 1], count_activities(group, i3))
            rows += [
                {'state': s1, 'action': a1, 'reward': 0.0, 'next_state': s3,
                 'terminal': False, 'intervention': 0, 'next_intervention': 2},
                {'state': s3, 'action': a3, 'reward': outcome,
                 'next_state': z.copy(), 'terminal': True, 'intervention': 2, 'next_intervention': -1},
            ]
        elif has2:
            i2 = int2_rows.index[0]
            a2 = 0 if group.loc[i2, 'activity'] == 'contact_headquarters' else 1
            s2 = extract_state(group.loc[i2 - 1], count_activities(group, i2))
            rows += [
                {'state': s1, 'action': a1, 'reward': 0.0, 'next_state': s2,
                 'terminal': False, 'intervention': 0, 'next_intervention': 1},
                {'state': s2, 'action': a2, 'reward': outcome,
                 'next_state': z.copy(), 'terminal': True, 'intervention': 1, 'next_intervention': -1},
            ]
        else:
            rows.append({'state': s1, 'action': a1, 'reward': outcome,
                         'next_state': z.copy(), 'terminal': True, 'intervention': 0, 'next_intervention': -1})

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cases',    type=int, default=10000)
    parser.add_argument('--confounded', action='store_true')
    parser.add_argument('--seed',       type=int, default=42)  # unused; data already split
    parser.add_argument('--steps',      type=int, default=3, choices=[1, 2, 3])
    args = parser.parse_args()

    suffix   = "CONF" if args.confounded else "RCT"
    base     = f"data/single_cql_{suffix}_{args.n_cases}"
    step_tag = "" if args.steps == 3 else f"_steps{args.steps}"

    df_train = pd.DataFrame(load_pickle(f"{base}_train.pkl"))
    df_val   = pd.DataFrame(load_pickle(f"{base}_val.pkl"))

    train_rows = extract_transitions(df_train, args.steps)
    val_rows   = extract_transitions(df_val,   args.steps)

    save_pickle(pd.DataFrame(train_rows), f"{base}_trans_train{step_tag}.pkl")
    save_pickle(pd.DataFrame(val_rows),   f"{base}_trans_val{step_tag}.pkl")
    print(f"Train: {len(train_rows)}, Val: {len(val_rows)} transitions (steps={args.steps})")
    print(f"[OK] Saved to {base}_trans_*{step_tag}.pkl")


if __name__ == "__main__":
    main()
