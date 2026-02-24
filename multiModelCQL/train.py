"""Train Multi-Model CQL: Q3 → Q2 → Q1."""
import sys
import os
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
os.chdir(project_root)

from shared import load_pickle, STATE_DIM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN     = [256, 256]
STATE_DIMS = [5, STATE_DIM, STATE_DIM]
N_ACTIONS  = [2, 2, 3]


class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        layers, d = [], state_dim
        for h in HIDDEN:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.LayerNorm(h)]
            d = h
        layers.append(nn.Linear(d, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TransitionDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return {
            'state':             torch.FloatTensor(r['state']),
            'action':            torch.LongTensor([r['action']]),
            'reward':            torch.FloatTensor([r['reward']]),
            'next_state':        torch.FloatTensor(r['next_state']),
            'terminal':          torch.FloatTensor([float(r['terminal'])]),
            'next_intervention': torch.LongTensor([r['next_intervention']]),
        }


def make_loader(df, int_idx, batch_size, shuffle=True):
    sub = df[df['intervention'] == int_idx].copy()
    if sub.empty:
        return None
    return DataLoader(TransitionDataset(sub), batch_size=batch_size, shuffle=shuffle)


def scale_col(df, col, mask, scaler):
    idx = df.index[mask]
    df.loc[idx, col] = list(scaler.transform(np.vstack(df.loc[idx, col].values)))


def train_q(q, qt, opt, tr, va, target_fn, args):
    best_val, best_state = float('inf'), copy.deepcopy(q.state_dict())
    for epoch in range(args.epochs):
        q.train()
        tl = 0.0
        for b in tr:
            s, a = b['state'].to(device), b['action'].squeeze(1).to(device)
            with torch.no_grad():
                tgt = target_fn(b)
            qa = q(s)
            q_taken = qa.gather(1, a.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(q_taken, tgt) + args.alpha * (torch.logsumexp(qa, 1).mean() - q_taken.mean())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 1.0)
            opt.step()
            for p, tp in zip(q.parameters(), qt.parameters()):
                tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)
            tl += loss.item()

        q.eval()
        vl = 0.0
        with torch.no_grad():
            for b in va:
                s, a = b['state'].to(device), b['action'].squeeze(1).to(device)
                q_taken = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
                vl += F.mse_loss(q_taken, target_fn(b)).item()
        vl /= max(len(va), 1)
        if vl < best_val:
            best_val, best_state = vl, copy.deepcopy(q.state_dict())
        if (epoch + 1) % 10 == 0:
            print(f"  [{epoch+1:3d}/{args.epochs}] train={tl/len(tr):.4f}  val={vl:.4f}")

    return best_state, best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cases',    type=int,   default=10000)
    parser.add_argument('--confounded', action='store_true')
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--batch_size', type=int,   default=256)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--alpha',      type=float, default=1.0)
    parser.add_argument('--gamma',      type=float, default=0.99)
    parser.add_argument('--tau',        type=float, default=0.005)
    parser.add_argument('--seed',       type=int,   default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    suffix = "CONF" if args.confounded else "RCT"
    base = f"data/multi_cql_{suffix}_{args.n_cases}"
    print(f"Training Multi-Model CQL — {suffix} | lr={args.lr} α={args.alpha} epochs={args.epochs}")

    df_train = load_pickle(f"{base}_trans_train.pkl")
    df_val   = load_pickle(f"{base}_trans_val.pkl")
    print(f"Train: {len(df_train)}, Val: {len(df_val)} transitions")

    # Fit one scaler per intervention on training states
    scalers = [
        StandardScaler().fit(np.vstack(df_train.loc[df_train['intervention'] == i, 'state'].values))
        for i in range(3)
    ]

    # Apply scalers: states by intervention, next_states by next_intervention
    for df in [df_train, df_val]:
        for i, sc in enumerate(scalers):
            scale_col(df, 'state', df['intervention'] == i, sc)
        for ni, sc in [(1, scalers[1]), (2, scalers[2])]:
            scale_col(df, 'next_state', df['next_intervention'] == ni, sc)

    term_r = df_train.loc[df_train['terminal'] == True, 'reward'].values
    r_mean, r_std = float(term_r.mean()), float(term_r.std()) + 1e-8

    def norm(r):
        return (r - r_mean) / r_std

    bs = args.batch_size
    tr0, va0 = make_loader(df_train, 0, bs), make_loader(df_val, 0, bs, False)
    tr1, va1 = make_loader(df_train, 1, bs), make_loader(df_val, 1, bs, False)
    tr2, va2 = make_loader(df_train, 2, bs), make_loader(df_val, 2, bs, False)

    def make_net(i):
        q  = QNetwork(STATE_DIMS[i], N_ACTIONS[i]).to(device)
        qt = QNetwork(STATE_DIMS[i], N_ACTIONS[i]).to(device)
        qt.load_state_dict(q.state_dict())
        return q, qt

    Q1, Q1t = make_net(0)
    Q2, Q2t = make_net(1)
    Q3, Q3t = make_net(2)
    opt1 = optim.Adam(Q1.parameters(), lr=args.lr)
    opt2 = optim.Adam(Q2.parameters(), lr=args.lr)
    opt3 = optim.Adam(Q3.parameters(), lr=args.lr)

    print("\n[Q3]")
    best3, _ = train_q(Q3, Q3t, opt3, tr2, va2,
                       lambda b: norm(b['reward'].squeeze(1).to(device)), args)
    Q3.load_state_dict(best3)
    Q3t.load_state_dict(best3)

    print("\n[Q2]")
    def tgt_q2(b):
        ns, r, term = b['next_state'].to(device), b['reward'].squeeze(1).to(device), b['terminal'].squeeze(1).to(device)
        return (1 - term) * args.gamma * Q3t(ns).max(1)[0] + term * norm(r)
    best2, _ = train_q(Q2, Q2t, opt2, tr1, va1, tgt_q2, args)
    Q2.load_state_dict(best2)
    Q2t.load_state_dict(best2)

    print("\n[Q1]")
    def tgt_q1(b):
        ns   = b['next_state'].to(device)
        r    = b['reward'].squeeze(1).to(device)
        term = b['terminal'].squeeze(1).to(device)
        ni   = b['next_intervention'].squeeze(1).to(device)
        t    = torch.zeros_like(r)
        t[term == 1] = norm(r[term == 1])
        m1, m2 = (ni == 1) & (term == 0), (ni == 2) & (term == 0)
        if m1.any(): t[m1] = args.gamma * Q2t(ns[m1]).max(1)[0]
        if m2.any(): t[m2] = args.gamma * Q3t(ns[m2]).max(1)[0]
        return t
    best1, _ = train_q(Q1, Q1t, opt1, tr0, va0, tgt_q1, args)
    Q1.load_state_dict(best1)

    os.makedirs("models", exist_ok=True)
    model_path = f"models/multi_cql_{suffix}_{args.n_cases}.pth"
    torch.save({
        'Q1': Q1.state_dict(), 'Q2': Q2.state_dict(), 'Q3': Q3.state_dict(),
        'scaler1': scalers[0], 'scaler2': scalers[1], 'scaler3': scalers[2],
        'config': {'state_dims': STATE_DIMS, 'n_actions': N_ACTIONS, 'hidden': HIDDEN},
        'reward_stats': {'mean': r_mean, 'std': r_std},
    }, model_path)
    print(f"\n[OK] {model_path}")


if __name__ == "__main__":
    main()
