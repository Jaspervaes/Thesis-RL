"""Evaluate Multi-Model CQL against bank and random baselines."""
import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
os.chdir(project_root)

from shared import (
    load_pickle, bank_policy, random_policy, evaluate_policy,
    print_results, print_action_dist,
    BASE_FEATURES, TRACKED_ACTIVITIES, STATE_DIM,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden):
        super().__init__()
        layers, d = [], state_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.LayerNorm(h)]
            d = h
        layers.append(nn.Linear(d, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CQLPolicy:
    def __init__(self, nets, scalers, steps=3):
        self.nets    = nets
        self.scalers = scalers
        self.steps   = steps
        self.counts  = {a: 0 for a in TRACKED_ACTIVITIES}

    def reset(self):
        self.counts = {a: 0 for a in TRACKED_ACTIVITIES}

    def __call__(self, prev_event, int_idx, prefix=None):
        if int_idx >= self.steps:
            return bank_policy(prev_event, int_idx)

        if prefix:
            self.counts = {a: 0 for a in TRACKED_ACTIVITIES}
            for e in prefix:
                act = e.get('activity', '').lower()
                for t in TRACKED_ACTIVITIES:
                    if t in act:
                        self.counts[t] += 1

        state = [float(prev_event.get(f, 0)) for f in BASE_FEATURES]
        if int_idx > 0:
            state += [float(self.counts.get(a, 0)) for a in TRACKED_ACTIVITIES]

        state = self.scalers[int_idx].transform(np.array(state, dtype=np.float32).reshape(1, -1))[0]
        with torch.no_grad():
            return self.nets[int_idx](torch.FloatTensor(state).unsqueeze(0).to(device)).argmax(1).item()


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

    suffix   = "CONF" if args.confounded else "RCT"
    step_tag = "" if args.steps == 3 else f"_steps{args.steps}"
    ckpt = torch.load(f"models/multi_cql_{suffix}_{args.n_cases}_s{args.train_seed}{step_tag}.pth",
                      map_location=device, weights_only=False)
    cfg   = ckpt['config']
    steps = cfg.get('steps', 3)

    def load_net(i):
        q = QNetwork(cfg['state_dims'][i], cfg['n_actions'][i], cfg['hidden']).to(device)
        q.load_state_dict(ckpt[f'Q{i+1}'])
        q.eval()
        return q

    nets    = {i: load_net(i) for i in range(steps)}
    scalers = {i: ckpt[f'scaler{i+1}'] for i in range(steps)}
    policy  = CQLPolicy(nets, scalers, steps=steps)
    params  = load_pickle(f"data/multi_cql_{suffix}_{args.n_cases}_params.pkl")
    label   = f'CQL-MM {suffix} ({steps}-step)'

    print(f"Evaluating Multi-Model CQL — {suffix} | steps={steps}")
    bank_res   = evaluate_policy(bank_policy,   args.n_episodes, params, args.seed)
    random_res = evaluate_policy(random_policy, args.n_episodes, params, args.seed)
    cql_res    = evaluate_policy(policy, args.n_episodes, params, args.seed,
                                 use_prefix=True, reset_fn=policy.reset)

    results = {'Bank': bank_res, 'Random': random_res, label: cql_res}
    print_results(results)
    print_action_dist(results)

    gain = ((cql_res['avg'] / bank_res['avg']) - 1) * 100
    print(f"\nCQL-MM {'beats' if gain > 0 else 'underperforms'} Bank by {gain:+.1f}%")

    if args.results_file:
        import json
        os.makedirs(os.path.dirname(os.path.abspath(args.results_file)), exist_ok=True)
        with open(args.results_file, 'w') as f:
            json.dump({'Bank': bank_res['avg'], label: cql_res['avg'], 'Random': random_res['avg']}, f)


if __name__ == "__main__":
    main()
