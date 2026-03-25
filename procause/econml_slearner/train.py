"""Train ProCause EconML S-learner: causal reward estimation via GBR."""
import sys
import os
import argparse
import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
os.chdir(project_root)

from shared import load_pickle, N_ACTIONS


def train_econml_slearner(states, actions, outcomes, n_actions):
    """Train S-learner via GradientBoosting. Returns model, scaler, outcome stats."""
    state_scaler = StandardScaler()
    states_norm = state_scaler.fit_transform(states)
    outcome_mean, outcome_std = outcomes.mean(), outcomes.std() + 1e-8
    outcomes_norm = (outcomes - outcome_mean) / outcome_std

    X_train = np.column_stack([states_norm, actions.reshape(-1, 1)])
    model = GradientBoostingRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42)
    model.fit(X_train, outcomes_norm)

    return model, state_scaler, outcome_mean, outcome_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cases',    type=int,   default=10000)
    parser.add_argument('--confounded', action='store_true')
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--steps',      type=int,   default=3, choices=[1, 2, 3])
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    suffix = "CONF" if args.confounded else "RCT"
    base   = f"data/procause_econml_{suffix}_{args.n_cases}"
    step_tag = "" if args.steps == 3 else f"_steps{args.steps}"
    print(f"Training ProCause EconML — {suffix} | steps={args.steps}")

    df_train = load_pickle(f"{base}_trans_train{step_tag}.pkl")
    df_val   = load_pickle(f"{base}_trans_val{step_tag}.pkl")
    print(f"Train: {len(df_train)}, Val: {len(df_val)} transitions")

    cfg = {
        'n_actions':    N_ACTIONS,
        'steps':        args.steps,
    }

    save_dict = {'config': cfg}
    active_interventions = list(range(args.steps))

    for int_idx in active_interventions:
        n_act = N_ACTIONS[int_idx]
        sub_train = df_train[df_train['intervention'] == int_idx]
        sub_val   = df_val[df_val['intervention'] == int_idx]

        if sub_train.empty:
            print(f"  [skip] No training data for intervention {int_idx}")
            continue

        # Check action diversity
        unique_actions = sub_train['action'].nunique()
        if unique_actions < 2:
            print(f"  [skip] Int.{int_idx}: only action {sub_train['action'].iloc[0]} in data, cannot estimate CATE")
            continue

        # Print per-action stats
        for a in sorted(sub_train['action'].unique()):
            mask = sub_train['action'] == a
            m = sub_train.loc[mask, 'case_outcome'].mean()
            print(f"  Int.{int_idx} action={a}: n={mask.sum()}, outcome_mean={m:.1f}")

        # Phase 1: Train GBR S-learner on state vectors
        print(f"\n[GBR S-learner Int.{int_idx}]")
        tr_states = np.stack(sub_train['state'].tolist())
        tr_actions = np.array(sub_train['action'].tolist())
        tr_outcomes = np.array(sub_train['case_outcome'].tolist(), dtype=np.float64)

        va_states = np.stack(sub_val['state'].tolist())
        va_actions = np.array(sub_val['action'].tolist())
        va_outcomes = np.array(sub_val['case_outcome'].tolist(), dtype=np.float64)

        gbr_model, state_scaler, outcome_mean, outcome_std = train_econml_slearner(
            tr_states, tr_actions, tr_outcomes, n_act)

        # Validate: predict for each action on val set
        va_states_norm = state_scaler.transform(va_states)
        val_preds = []
        for a in range(n_act):
            X_val = np.column_stack([va_states_norm, np.full(len(va_states), a)])
            val_preds.append(gbr_model.predict(X_val) * outcome_std + outcome_mean)
        val_preds = np.stack(val_preds, axis=1)
        print(f"  Val pred means per action: {np.mean(val_preds, axis=0)}")

        # Save GBR S-learner artifacts
        import pickle
        save_dict[f'gbr_{int_idx}'] = pickle.dumps(gbr_model)
        save_dict[f'scaler_{int_idx}'] = pickle.dumps(state_scaler)
        save_dict[f'outcome_mean_{int_idx}'] = outcome_mean
        save_dict[f'outcome_std_{int_idx}'] = outcome_std
        save_dict[f'n_actions_{int_idx}'] = n_act

    os.makedirs("models", exist_ok=True)
    model_path = f"models/procause_econml_{suffix}_{args.n_cases}_s{args.seed}{step_tag}.pth"
    torch.save(save_dict, model_path)
    print(f"\n[OK] {model_path}")


if __name__ == "__main__":
    main()
