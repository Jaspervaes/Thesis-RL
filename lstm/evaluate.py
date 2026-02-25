"""Evaluate LSTM-DQN offline RL against bank and random baselines."""
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
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_ACTIONS = [2, 2, 3]


class LSTM_DQN(nn.Module):
    def __init__(self, n_activities, n_features, n_actions, emb_dim, hidden, n_layers, dropout):
        super().__init__()
        self.emb  = nn.Embedding(n_activities, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim + n_features, hidden, n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc   = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, acts, feats, lens):
        x = torch.cat([self.emb(acts), feats], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.fc(h[-1])


def encode_prefix(prefix, cfg):
    """Encode a single prefix into padded tensors (batch size 1)."""
    max_len = cfg['max_len']
    a2i     = cfg['activity_to_idx']
    means   = cfg['feat_means']
    stds    = cfg['feat_stds']
    cols    = cfg['feature_cols']

    acts  = np.zeros((1, max_len), dtype=np.int64)
    feats = np.zeros((1, max_len, len(cols)), dtype=np.float32)
    seq_len = max(min(len(prefix), max_len), 1)

    for j, e in enumerate(prefix[:seq_len]):
        acts[0, j] = a2i.get(e.get('activity', ''), 0)
        for k, col in enumerate(cols):
            feats[0, j, k] = (float(e.get(col, 0)) - means[col]) / stds[col]

    return torch.LongTensor(acts).to(device), torch.FloatTensor(feats).to(device), torch.LongTensor([seq_len])


class LSTMPolicy:
    """LSTM-DQN policy with three separate Q-networks."""

    def __init__(self, models, cfg):
        self.models = models  # {0: Q1, 1: Q2, 2: Q3}
        self.cfg    = cfg

    def reset(self):
        pass

    def __call__(self, prev_event, int_idx, prefix=None):
        acts, feats, lens = encode_prefix(prefix or [], self.cfg)
        with torch.no_grad():
            q = self.models[int_idx](acts, feats, lens)
        return q[0, :N_ACTIONS[int_idx]].argmax().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cases',    type=int, default=10000)
    parser.add_argument('--confounded', action='store_true')
    parser.add_argument('--n_episodes',   type=int, default=1000)
    parser.add_argument('--seed',         type=int, default=1042)
    parser.add_argument('--train_seed',   type=int, default=42)
    parser.add_argument('--results_file', type=str, default=None)
    args = parser.parse_args()

    suffix = "CONF" if args.confounded else "RCT"
    ckpt   = torch.load(f"models/lstm_{suffix}_{args.n_cases}_s{args.train_seed}.pth", map_location=device, weights_only=False)
    cfg    = ckpt['config']
    params = load_pickle(f"data/lstm_{suffix}_{args.n_cases}_params.pkl")

    def load_net(key, n_act):
        m = LSTM_DQN(cfg['n_activities'], cfg['n_features'], n_act,
                     cfg['emb_dim'], cfg['hidden'], cfg['n_layers'], cfg['dropout']).to(device)
        m.load_state_dict(ckpt[key])
        m.eval()
        return m

    models = {0: load_net('Q1', N_ACTIONS[0]), 1: load_net('Q2', N_ACTIONS[1]), 2: load_net('Q3', N_ACTIONS[2])}
    policy = LSTMPolicy(models, cfg)

    print(f"Evaluating LSTM-DQN — {suffix}")
    bank_res   = evaluate_policy(bank_policy,   args.n_episodes, params, args.seed)
    random_res = evaluate_policy(random_policy, args.n_episodes, params, args.seed)
    lstm_res   = evaluate_policy(policy, args.n_episodes, params, args.seed,
                                 use_prefix=True, reset_fn=policy.reset)

    results = {'Bank': bank_res, 'Random': random_res, f'LSTM {suffix}': lstm_res}
    print_results(results)
    print_action_dist(results)

    gain = ((lstm_res['avg'] / bank_res['avg']) - 1) * 100
    print(f"\nLSTM {'beats' if gain > 0 else 'underperforms'} Bank by {gain:+.1f}%")

    if args.results_file:
        import json
        os.makedirs(os.path.dirname(os.path.abspath(args.results_file)), exist_ok=True)
        with open(args.results_file, 'w') as f:
            json.dump({'Bank': bank_res['avg'], f'LSTM {suffix}': lstm_res['avg'], 'Random': random_res['avg']}, f)


if __name__ == "__main__":
    main()
