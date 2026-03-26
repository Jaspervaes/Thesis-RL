"""Evaluate LSTM-DQN offline RL against bank and random baselines."""
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
    print_results, print_action_dist,
    N_ACTIONS, LSTM_DQN, encode_prefix,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMPolicy:
    """LSTM-DQN policy with separate Q-networks per trained intervention."""

    def __init__(self, models, cfg, steps=3):
        self.models = models  # {0: Q1, 1: Q2, 2: Q3} — only trained interventions
        self.cfg    = cfg
        self.steps  = steps

    def reset(self):
        pass

    def __call__(self, prev_event, int_idx, prefix=None):
        if int_idx >= self.steps:
            return bank_policy(prev_event, int_idx)
        acts, feats, lens = encode_prefix(prefix or [], self.cfg)
        with torch.no_grad():
            q = self.models[int_idx](acts, feats, lens)
        return q[0, :N_ACTIONS[int_idx]].argmax().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cases',      type=int, default=10000)
    parser.add_argument('--confounded',   action='store_true')
    parser.add_argument('--n_episodes',   type=int, default=1000)
    parser.add_argument('--seed',         type=int, default=1042)
    parser.add_argument('--train_seed',   type=int, default=42)
    parser.add_argument('--steps',        type=int, default=3, choices=[1, 2, 3])
    parser.add_argument('--results_file', type=str, default=None)
    parser.add_argument('--target_calc', default='normal', choices=['normal', 'torch.max'])
    parser.add_argument('--activity_enc', default='integer', choices=['onehot', 'integer'])
    args = parser.parse_args()

    suffix = "CONF" if args.confounded else "RCT"
    suffix_res = suffix + f'_target_{args.target_calc}' + f'_actenc_{args.activity_enc}'
    step_tag = "" if args.steps == 3 else f"_steps{args.steps}"
    ckpt   = torch.load(f"models/lstm_{suffix}_{args.n_cases}_s{args.train_seed}{step_tag}.pth",
                        map_location=device, weights_only=False)
    cfg    = ckpt['config']
    params = load_pickle(f"data/simbank_{suffix}_{args.n_cases}_params.pkl")

    def load_net(key, n_act):
        m = LSTM_DQN(cfg['n_activities'], cfg['n_features'], n_act,
                     cfg['emb_dim'], cfg['hidden'], cfg['n_layers'], cfg['dropout'],
                     activity_enc=cfg.get('activity_enc', 'integer')).to(device)
        m.load_state_dict(ckpt[key])
        m.eval()
        return m

    models = {}
    models[0] = load_net('Q1', N_ACTIONS[0])
    if args.steps >= 2:
        models[1] = load_net('Q2', N_ACTIONS[1])
    if args.steps >= 3:
        models[2] = load_net('Q3', N_ACTIONS[2])

    policy = LSTMPolicy(models, cfg, steps=args.steps)
    label  = f'LSTM {suffix} ({args.steps}-step)'

    print(f"Evaluating LSTM-DQN — {suffix} | steps={args.steps}")
    bank_res   = evaluate_policy(bank_policy,   args.n_episodes, params, args.seed)
    random_res = evaluate_policy(random_policy, args.n_episodes, params, args.seed)
    lstm_res   = evaluate_policy(policy, args.n_episodes, params, args.seed,
                                 use_prefix=True, reset_fn=policy.reset)

    results = {'Bank': bank_res, 'Random': random_res, label: lstm_res}
    print_results(results)
    print_action_dist(results)

    gain = ((lstm_res['avg'] / bank_res['avg']) - 1) * 100
    print(f"\nLSTM {'beats' if gain > 0 else 'underperforms'} Bank by {gain:+.1f}%")

    if args.results_file:
        import json
        os.makedirs(os.path.dirname(os.path.abspath(args.results_file)), exist_ok=True)
        with open(args.results_file, 'w') as f:
            json.dump({'Bank': bank_res['avg'], label: lstm_res['avg'], 'Random': random_res['avg']}, f)


if __name__ == "__main__":
    main()
