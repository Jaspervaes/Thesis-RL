"""Convert raw SimBank data to LSTM transitions with case_outcome for S-learner training."""
import sys
import os
import argparse
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
os.chdir(project_root)

from shared import load_pickle, save_pickle, split_train_val, get_ir_action


def extract_transitions(df, steps=3):
    """Extract transitions with prefix sequences and case_outcome for all interventions."""
    rows = []

    for _, group in df.groupby('case_nr'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        outcome = float(group['outcome'].iloc[-1])
        events  = group.to_dict('records')

        int0_rows = group[group['activity'].isin(['start_standard', 'start_priority'])]
        if int0_rows.empty or int0_rows.index[0] == 0:
            continue

        i0 = int0_rows.index[0]
        a0 = 1 if group.loc[i0, 'activity'] == 'start_priority' else 0
        p0 = events[:i0]

        if steps == 1:
            rows.append({'prefix': p0, 'action': a0, 'reward': outcome,
                         'next_prefix': [], 'terminal': True, 'intervention': 0,
                         'next_intervention': -1, 'case_outcome': outcome})
            continue

        int1_rows = group[group['activity'].isin(['contact_headquarters', 'skip_contact']) & (group.index > i0)]
        int2_rows = group[(group['activity'] == 'calculate_offer') & (group.index > i0)]
        has1, has2 = not int1_rows.empty, not int2_rows.empty

        if steps == 2:
            if has1:
                i1 = int1_rows.index[0]
                a1 = 0 if group.loc[i1, 'activity'] == 'contact_headquarters' else 1
                p1 = events[:i1]
                rows += [
                    {'prefix': p0, 'action': a0, 'reward': 0.0, 'next_prefix': p1,
                     'terminal': False, 'intervention': 0, 'next_intervention': 1,
                     'case_outcome': outcome},
                    {'prefix': p1, 'action': a1, 'reward': outcome, 'next_prefix': [],
                     'terminal': True, 'intervention': 1, 'next_intervention': -1,
                     'case_outcome': outcome},
                ]
            else:
                rows.append({'prefix': p0, 'action': a0, 'reward': outcome,
                             'next_prefix': [], 'terminal': True, 'intervention': 0,
                             'next_intervention': -1, 'case_outcome': outcome})
            continue

        # steps == 3
        if has1 and has2:
            i1, i2 = int1_rows.index[0], int2_rows.index[0]
            a1 = 0 if group.loc[i1, 'activity'] == 'contact_headquarters' else 1
            a2 = get_ir_action(group.loc[i2].get('interest_rate', 0.08))
            p1, p2 = events[:i1], events[:i2]
            rows += [
                {'prefix': p0, 'action': a0, 'reward': 0.0, 'next_prefix': p1,
                 'terminal': False, 'intervention': 0, 'next_intervention': 1,
                 'case_outcome': outcome},
                {'prefix': p1, 'action': a1, 'reward': 0.0, 'next_prefix': p2,
                 'terminal': False, 'intervention': 1, 'next_intervention': 2,
                 'case_outcome': outcome},
                {'prefix': p2, 'action': a2, 'reward': outcome, 'next_prefix': [],
                 'terminal': True, 'intervention': 2, 'next_intervention': -1,
                 'case_outcome': outcome},
            ]
        elif not has1 and has2:
            i2 = int2_rows.index[0]
            a2 = get_ir_action(group.loc[i2].get('interest_rate', 0.08))
            p2 = events[:i2]
            rows += [
                {'prefix': p0, 'action': a0, 'reward': 0.0, 'next_prefix': p2,
                 'terminal': False, 'intervention': 0, 'next_intervention': 2,
                 'case_outcome': outcome},
                {'prefix': p2, 'action': a2, 'reward': outcome, 'next_prefix': [],
                 'terminal': True, 'intervention': 2, 'next_intervention': -1,
                 'case_outcome': outcome},
            ]
        elif has1:
            i1 = int1_rows.index[0]
            a1 = 0 if group.loc[i1, 'activity'] == 'contact_headquarters' else 1
            p1 = events[:i1]
            rows += [
                {'prefix': p0, 'action': a0, 'reward': 0.0, 'next_prefix': p1,
                 'terminal': False, 'intervention': 0, 'next_intervention': 1,
                 'case_outcome': outcome},
                {'prefix': p1, 'action': a1, 'reward': outcome, 'next_prefix': [],
                 'terminal': True, 'intervention': 1, 'next_intervention': -1,
                 'case_outcome': outcome},
            ]
        else:
            rows.append({'prefix': p0, 'action': a0, 'reward': outcome,
                         'next_prefix': [], 'terminal': True, 'intervention': 0,
                         'next_intervention': -1, 'case_outcome': outcome})

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cases',    type=int, default=10000)
    parser.add_argument('--confounded', action='store_true')
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--steps',      type=int, default=3, choices=[1, 2, 3])
    args = parser.parse_args()

    suffix = "CONF" if args.confounded else "RCT"
    base = f"data/procause_lstm_{suffix}_{args.n_cases}"
    step_tag = "" if args.steps == 3 else f"_steps{args.steps}"

    df = load_pickle(f"data/simbank_{suffix}_{args.n_cases}_raw.pkl")
    df_train, df_val = split_train_val(df, val_ratio=0.2, seed=args.seed)

    train_rows = extract_transitions(df_train, args.steps)
    val_rows   = extract_transitions(df_val,   args.steps)

    save_pickle(pd.DataFrame(train_rows), f"{base}_trans_train{step_tag}.pkl")
    save_pickle(pd.DataFrame(val_rows),   f"{base}_trans_val{step_tag}.pkl")
    print(f"Train: {len(train_rows)}, Val: {len(val_rows)} transitions (steps={args.steps})")
    print(f"[OK] Saved to {base}_trans_*{step_tag}.pkl")


if __name__ == "__main__":
    main()
