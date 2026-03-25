"""
Shared LSTM utilities for prefix-based methods (lstm, rims, procause).
"""
import random
import numpy as np
import torch
import torch.nn as nn


def seed_worker(worker_id):
    """Ensure reproducible DataLoader workers by seeding from PyTorch's RNG."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = ['amount', 'est_quality', 'unc_quality', 'interest_rate', 'cum_cost', 'elapsed_time']
N_ACTIONS = [2, 2, 3]


class LSTM_DQN(nn.Module):
    """LSTM encoder + Q-head for one intervention."""

    def __init__(self, n_activities, n_features, n_actions, emb_dim=32, hidden=128, n_layers=2, dropout=0.2,
                 activity_enc='integer'):
        super().__init__()
        self.activity_enc = activity_enc
        if self.activity_enc == 'integer':
            self.emb = nn.Embedding(n_activities, emb_dim, padding_idx=0)
            lstm_in_dim = emb_dim + n_features
        else:
            self.emb = None
            lstm_in_dim = n_activities + n_features
        self.lstm = nn.LSTM(lstm_in_dim, hidden, n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc   = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, acts, feats, lens):
        if self.activity_enc == 'integer':
            act_repr = self.emb(acts)
        else:
            act_repr = acts
        x = torch.cat([act_repr, feats], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.fc(h[-1])


def build_vocab_and_stats(df):
    """Build activity vocab and feature normalization from all prefixes."""
    all_events = []
    for prefixes in [df['prefix'], df['next_prefix']]:
        for p in prefixes:
            all_events.extend(p)

    # Use sorted unique activities to avoid run-to-run hash-order variation.
    activities = sorted({e.get('activity', '') for e in all_events})
    activity_to_idx = {a: i + 1 for i, a in enumerate(activities)}
    activity_to_idx[''] = 0

    def _vals(c):
        return [float(v) for e in all_events if not np.isnan(v := float(e.get(c, 0) or 0))]
    feat_means = {c: (np.mean(_vals(c)) if _vals(c) else 0.0) for c in FEATURE_COLS}
    feat_stds  = {c: max(np.std(_vals(c))  if _vals(c) else 0.0, 1e-8) for c in FEATURE_COLS}

    return activity_to_idx, feat_means, feat_stds


def encode(prefixes, activity_to_idx, feat_means, feat_stds, max_len, activity_enc='integer', n_activities=None):
    """Encode a list of prefix sequences to padded tensors."""
    n = len(prefixes)
    if n_activities is None:
        n_activities = max(activity_to_idx.values(), default=0) + 1
    if activity_enc == 'onehot':
        acts = np.zeros((n, max_len, n_activities), dtype=np.float32)
    else:
        acts = np.zeros((n, max_len), dtype=np.int64)
    feats = np.zeros((n, max_len, len(FEATURE_COLS)), dtype=np.float32)
    lens = np.ones(n, dtype=np.int64)

    for i, p in enumerate(prefixes):
        seq_len = min(len(p), max_len)
        lens[i] = max(seq_len, 1)
        for j, e in enumerate(p[:seq_len]):
            a_idx = activity_to_idx.get(e.get('activity', ''), 0)
            if activity_enc == 'onehot':
                acts[i, j, a_idx] = 1.0
            else:
                acts[i, j] = a_idx
            for k, col in enumerate(FEATURE_COLS):
                v = float(e.get(col, 0) or 0)
                feats[i, j, k] = 0.0 if np.isnan(v) else (v - feat_means[col]) / feat_stds[col]

    return acts, feats, lens


def encode_prefix(prefix, cfg):
    """Encode a single prefix into padded tensors (batch size 1)."""
    max_len = cfg['max_len']
    a2i     = cfg['activity_to_idx']
    means   = cfg['feat_means']
    stds    = cfg['feat_stds']
    cols    = cfg['feature_cols']
    activity_enc = cfg.get('activity_enc', 'integer')
    n_activities = cfg.get('n_activities', max(a2i.values(), default=0) + 1)

    seq_len = max(min(len(prefix), max_len), 1)

    if activity_enc == 'onehot':
        acts = np.zeros((1, max_len, n_activities), dtype=np.float32)
    else:
        acts = np.zeros((1, max_len), dtype=np.int64)
    feats = np.zeros((1, max_len, len(cols)), dtype=np.float32)

    for j, e in enumerate(prefix[:seq_len]):
        a_idx = a2i.get(e.get('activity', ''), 0)
        if activity_enc == 'onehot':
            acts[0, j, a_idx] = 1.0
        else:
            acts[0, j] = a_idx
        for k, col in enumerate(cols):
            v = float(e.get(col, 0) or 0)
            feats[0, j, k] = 0.0 if np.isnan(v) else (v - means[col]) / stds[col]

    if activity_enc == 'onehot':
        return torch.FloatTensor(acts).to(device), torch.FloatTensor(feats).to(device), torch.LongTensor([seq_len])
    return torch.LongTensor(acts).to(device), torch.FloatTensor(feats).to(device), torch.LongTensor([seq_len])
