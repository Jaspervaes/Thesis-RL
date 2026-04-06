"""Evaluate Single-Model CQL against bank and random baselines."""
import sys
import os
import argparse
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
os.chdir(project_root)

from shared import (
    load_pickle, bank_policy, random_policy, evaluate_policy,
    print_results, print_action_dist, N_ACTIONS, LSTM_DQN, encode_prefix,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CQLPolicy:
    """CQL-SN policy: single shared LSTM-DQN for all interventions, masked per intervention."""

    def __init__(self, model, cfg, steps=3):
        self.model = model
        self.cfg   = cfg
        self.steps = steps

    def reset(self):
        pass

    def __call__(self, prev_event, int_idx, prefix=None):
        if int_idx >= self.steps:
            return bank_policy(prev_event, int_idx)
        acts, feats, lens = encode_prefix(prefix or [], self.cfg)
        with torch.no_grad():
            q = self.model(acts, feats, lens)
        return q[0, :N_ACTIONS[int_idx]].argmax().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cases',      type=int,  default=10000)
    parser.add_argument('--confounded',   action='store_true')
    parser.add_argument('--steps',        type=int,  default=3, choices=[1, 2, 3])
    parser.add_argument('--n_episodes',   type=int,  default=1000)
    parser.add_argument('--seed',         type=int,  default=1042)
    parser.add_argument('--train_seed',   type=int,  default=42)
    parser.add_argument('--results_file', type=str,  default=None)
    args = parser.parse_args()

    suffix     = "CONF" if args.confounded else "RCT"
    step_tag   = "" if args.steps == 3 else f"_steps{args.steps}"
    model_path = f"models/single_cql_{suffix}_{args.n_cases}_s{args.train_seed}{step_tag}.pth"

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg  = ckpt['config']

    model = LSTM_DQN(cfg['n_activities'], cfg['n_features'], cfg['max_actions'],
                     cfg['emb_dim'], cfg['hidden'], cfg['n_layers'], cfg['dropout']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    label  = f'CQL-SN {suffix} ({args.steps}-step)'
    policy = CQLPolicy(model, cfg, steps=args.steps)
    params = load_pickle(f"data/simbank_{suffix}_{args.n_cases}_params.pkl")

    print(f"Evaluating Single-Model CQL — {suffix} | steps={args.steps}")
    bank_res   = evaluate_policy(bank_policy,   args.n_episodes, params, args.seed)
    random_res = evaluate_policy(random_policy, args.n_episodes, params, args.seed)
    cql_res    = evaluate_policy(policy, args.n_episodes, params, args.seed,
                                 use_prefix=True, reset_fn=policy.reset)

    results = {'Bank': bank_res, 'Random': random_res, label: cql_res}
    print_results(results)
    print_action_dist(results)

    gain = ((cql_res['avg'] / bank_res['avg']) - 1) * 100
    print(f"\nCQL-SN {'beats' if gain > 0 else 'underperforms'} Bank by {gain:+.1f}%")

    if args.results_file:
        import json
        os.makedirs(os.path.dirname(os.path.abspath(args.results_file)), exist_ok=True)
        with open(args.results_file, 'w') as f:
            json.dump({'Bank': bank_res['avg'], label: cql_res['avg'], 'Random': random_res['avg']}, f)


if __name__ == "__main__":
    main()
