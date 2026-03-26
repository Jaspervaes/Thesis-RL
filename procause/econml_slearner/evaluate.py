"""Evaluate ProCause EconML S-learner against bank and random baselines."""
import sys
import os
import argparse
import pickle
import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
os.chdir(project_root)

from shared import (
    load_pickle, bank_policy, random_policy, evaluate_policy,
    print_results, print_action_dist, extract_state, N_ACTIONS,
)
from shared.experiment_config import TRACKED_ACTIVITIES


def count_activities_from_list(events, up_to_idx):
    """Count activity occurrences from a list of event dicts."""
    counts = {act: 0 for act in TRACKED_ACTIVITIES}
    for i in range(min(up_to_idx, len(events))):
        activity = events[i].get('activity', '').lower()
        for tracked in TRACKED_ACTIVITIES:
            if tracked in activity:
                counts[tracked] += 1
    return counts


class ProCauseEconMLPolicy:
    """ProCause policy: use GBR S-learner directly to pick best action per case."""

    def __init__(self, gbr_models, cfg, steps=3):
        self.gbr_models = gbr_models  # {int_idx: (model, scaler, n_actions)}
        self.cfg = cfg
        self.steps = steps

    def reset(self):
        pass

    def __call__(self, prev_event, int_idx, prefix=None):
        if int_idx >= self.steps or int_idx not in self.gbr_models:
            return bank_policy(prev_event, int_idx)

        model, scaler, n_act = self.gbr_models[int_idx]

        # Extract state from prefix (same as convert_data)
        if prefix and len(prefix) > 0:
            activity_counts = count_activities_from_list(prefix, len(prefix) - 1)
            state = extract_state(prev_event, activity_counts)
        else:
            state = np.zeros(16)

        state_norm = scaler.transform(state.reshape(1, -1))

        # Predict outcome for each action, pick best
        preds = []
        for a in range(n_act):
            X = np.column_stack([state_norm, np.array([[a]])])
            preds.append(model.predict(X)[0])
        return int(np.argmax(preds))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cases',      type=int, default=10000)
    parser.add_argument('--confounded',   action='store_true')
    parser.add_argument('--n_episodes',   type=int, default=1000)
    parser.add_argument('--seed',         type=int, default=1042)
    parser.add_argument('--train_seed',   type=int, default=42)
    parser.add_argument('--steps',        type=int, default=3, choices=[1, 2, 3])
    parser.add_argument('--results_file', type=str, default=None)
    args = parser.parse_args()

    suffix = "CONF" if args.confounded else "RCT"
    step_tag = "" if args.steps == 3 else f"_steps{args.steps}"
    ckpt   = torch.load(f"models/procause_econml_{suffix}_{args.n_cases}_s{args.train_seed}{step_tag}.pth",
                        map_location='cpu', weights_only=False)
    cfg    = ckpt['config']
    params = load_pickle(f"data/simbank_{suffix}_{args.n_cases}_params.pkl")

    gbr_models = {}
    for int_idx in range(args.steps):
        key = f'gbr_{int_idx}'
        if key in ckpt:
            model = pickle.loads(ckpt[key])
            scaler = pickle.loads(ckpt[f'scaler_{int_idx}'])
            n_act = ckpt[f'n_actions_{int_idx}']
            gbr_models[int_idx] = (model, scaler, n_act)

    policy = ProCauseEconMLPolicy(gbr_models, cfg, steps=args.steps)
    label  = f'ProCause-EconML {suffix} ({args.steps}-step)'

    print(f"Evaluating ProCause EconML — {suffix} | steps={args.steps}")
    bank_res   = evaluate_policy(bank_policy,   args.n_episodes, params, args.seed)
    random_res = evaluate_policy(random_policy, args.n_episodes, params, args.seed)
    pc_res     = evaluate_policy(policy, args.n_episodes, params, args.seed,
                                 use_prefix=True, reset_fn=policy.reset)

    results = {'Bank': bank_res, 'Random': random_res, label: pc_res}
    print_results(results)
    print_action_dist(results)

    gain = ((pc_res['avg'] / bank_res['avg']) - 1) * 100
    print(f"\nProCause-EconML {'beats' if gain > 0 else 'underperforms'} Bank by {gain:+.1f}%")

    if args.results_file:
        import json
        os.makedirs(os.path.dirname(os.path.abspath(args.results_file)), exist_ok=True)
        with open(args.results_file, 'w') as f:
            json.dump({'Bank': bank_res['avg'], label: pc_res['avg'], 'Random': random_res['avg']}, f)


if __name__ == "__main__":
    main()
