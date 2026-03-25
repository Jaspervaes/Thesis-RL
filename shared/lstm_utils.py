"""
Shared LSTM utilities for prefix-based methods (lstm, rims).
"""
import numpy as np
import torch
import torch.nn as nn


def seed_worker(worker_id):
    """Ensure reproducible DataLoader workers by seeding from PyTorch's RNG."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = ['amount', 'est_quality', 'unc_quality', 'interest_rate', 'cum_cost', 'elapsed_time']
N_ACTIONS = [2, 2, 3]


class LSTM_DQN(nn.Module):
    """LSTM encoder + Q-head for one intervention."""

    def __init__(self, n_activities, n_features, n_actions, emb_dim=32, hidden=128, n_layers=2, dropout=0.2):
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


def build_vocab_and_stats(df):
    """Build activity vocab and feature normalization from all prefixes."""
    all_events = []
    for prefixes in [df['prefix'], df['next_prefix']]:
        for p in prefixes:
            all_events.extend(p)

    activities = list({e.get('activity', '') for e in all_events})
    activity_to_idx = {a: i + 1 for i, a in enumerate(activities)}
    activity_to_idx[''] = 0

    feat_means = {c: np.nanmean([float(e.get(c, 0)) for e in all_events]) for c in FEATURE_COLS}
    feat_stds  = {c: max(np.nanstd([float(e.get(c, 0)) for e in all_events]), 1e-8) for c in FEATURE_COLS}

    return activity_to_idx, feat_means, feat_stds


def encode(prefixes, activity_to_idx, feat_means, feat_stds, max_len):
    """Encode a list of prefix sequences to padded tensors."""
    n = len(prefixes)
    acts = np.zeros((n, max_len), dtype=np.int64)
    feats = np.zeros((n, max_len, len(FEATURE_COLS)), dtype=np.float32)
    lens = np.ones(n, dtype=np.int64)

    for i, p in enumerate(prefixes):
        seq_len = min(len(p), max_len)
        lens[i] = max(seq_len, 1)
        for j, e in enumerate(p[:seq_len]):
            acts[i, j] = activity_to_idx.get(e.get('activity', ''), 0)
            for k, col in enumerate(FEATURE_COLS):
                val = e.get(col, 0)
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    val = 0.0
                if not np.isfinite(val):
                    val = 0.0
                feats[i, j, k] = (val - feat_means[col]) / feat_stds[col]

    return acts, feats, lens


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
            val = e.get(col, 0)
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = 0.0
            if not np.isfinite(val):
                val = 0.0
            feats[0, j, k] = (val - means[col]) / stds[col]

    return torch.LongTensor(acts).to(device), torch.FloatTensor(feats).to(device), torch.LongTensor([seq_len])
