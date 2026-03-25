"""Evaluate ProCause LSTM S-learner against bank and random baselines."""
import sys
import os
import argparse
import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
os.chdir(project_root)

from shared import (
    load_pickle, bank_policy, random_policy, evaluate_policy,
    print_results, print_action_dist, N_ACTIONS, encode_prefix,
)
from procause.lstm_slearner.train import LSTM_SLearner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProCauseLSTMPolicy:
    """ProCause policy: use S-learner directly to pick best action per case."""

    def __init__(self, slearners, cfg, steps=3):
        self.slearners = slearners
        self.cfg = cfg
        self.steps = steps

    def reset(self):
        pass

    def __call__(self, prev_event, int_idx, prefix=None):
        if int_idx >= self.steps or int_idx not in self.slearners:
            return bank_policy(prev_event, int_idx)
        acts, feats, lens = encode_prefix(prefix or [], self.cfg)
        slearner = self.slearners[int_idx]
        n_act = slearner.n_actions
        with torch.no_grad():
            preds = []
            for a in range(n_act):
                action_t = torch.LongTensor([a]).to(device)
                pred = slearner(acts, feats, lens, action_t).item()
                preds.append(pred)
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
    ckpt   = torch.load(f"models/procause_lstm_{suffix}_{args.n_cases}_s{args.train_seed}{step_tag}.pth",
                        map_location=device, weights_only=False)
    cfg    = ckpt['config']
    params = load_pickle(f"data/procause_lstm_{suffix}_{args.n_cases}_params.pkl")

    action_emb_dim = cfg.get('action_emb_dim', 16)

    def load_slearner(key, n_act):
        m = LSTM_SLearner(cfg['n_activities'], cfg['n_features'], n_act,
                          cfg['emb_dim'], action_emb_dim, cfg['hidden'],
                          cfg['n_layers'], cfg['dropout']).to(device)
        m.load_state_dict(ckpt[key])
        m.eval()
        return m

    slearners = {}
    if 'S1' in ckpt:
        slearners[0] = load_slearner('S1', N_ACTIONS[0])
    if args.steps >= 2 and 'S2' in ckpt:
        slearners[1] = load_slearner('S2', N_ACTIONS[1])
    if args.steps >= 3 and 'S3' in ckpt:
        slearners[2] = load_slearner('S3', N_ACTIONS[2])

    policy = ProCauseLSTMPolicy(slearners, cfg, steps=args.steps)
    label  = f'ProCause-LSTM {suffix} ({args.steps}-step)'

    print(f"Evaluating ProCause LSTM — {suffix} | steps={args.steps}")
    bank_res   = evaluate_policy(bank_policy,   args.n_episodes, params, args.seed)
    random_res = evaluate_policy(random_policy, args.n_episodes, params, args.seed)
    pc_res     = evaluate_policy(policy, args.n_episodes, params, args.seed,
                                 use_prefix=True, reset_fn=policy.reset)

    results = {'Bank': bank_res, 'Random': random_res, label: pc_res}
    print_results(results)
    print_action_dist(results)

    gain = ((pc_res['avg'] / bank_res['avg']) - 1) * 100
    print(f"\nProCause-LSTM {'beats' if gain > 0 else 'underperforms'} Bank by {gain:+.1f}%")

    if args.results_file:
        import json
        os.makedirs(os.path.dirname(os.path.abspath(args.results_file)), exist_ok=True)
        with open(args.results_file, 'w') as f:
            json.dump({'Bank': bank_res['avg'], label: pc_res['avg'], 'Random': random_res['avg']}, f)


if __name__ == "__main__":
    main()
