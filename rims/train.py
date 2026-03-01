"""Train RIMS offline RL: multi-task LSTM (timing + outcome + Q) per intervention, Q3 → Q2 → Q1."""
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

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
os.chdir(project_root)

from shared import load_pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = ['amount', 'est_quality', 'unc_quality', 'interest_rate', 'cum_cost', 'elapsed_time']
N_ACTIONS    = [2, 2, 3]


class RIMS_Model(nn.Module):
    """
    Shared LSTM encoder with three heads:
    timing (proc_time, rem_time), outcome (per action), Q-values (per action).
    """

    def __init__(self, n_activities, n_features, n_actions, emb_dim=32, hidden=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.emb  = nn.Embedding(n_activities, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim + n_features, hidden, n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.timing_head  = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 2))
        self.outcome_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_actions))
        self.q_head       = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_actions))

    def encode(self, acts, feats, lens):
        x = torch.cat([self.emb(acts), feats], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return h[-1]

    def forward(self, acts, feats, lens):
        h = self.encode(acts, feats, lens)
        return self.timing_head(h), self.outcome_head(h), self.q_head(h)

    def q_values(self, acts, feats, lens):
        return self.q_head(self.encode(acts, feats, lens))


def build_vocab_and_stats(df):
    """Build activity vocab and feature normalization stats."""
    all_events = []
    for prefixes in [df['prefix'], df['next_prefix']]:
        for p in prefixes:
            all_events.extend(p)

    activities = list({e.get('activity', '') for e in all_events})
    activity_to_idx = {a: i + 1 for i, a in enumerate(activities)}
    activity_to_idx[''] = 0

    def _vals(c):
        return [float(v) for e in all_events if not np.isnan(v := float(e.get(c, 0) or 0))]
    feat_means = {c: (np.mean(_vals(c)) if _vals(c) else 0.0) for c in FEATURE_COLS}
    feat_stds  = {c: max(np.std(_vals(c))  if _vals(c) else 0.0, 1e-8) for c in FEATURE_COLS}

    return activity_to_idx, feat_means, feat_stds


def encode(prefixes, activity_to_idx, feat_means, feat_stds, max_len):
    """Encode list of prefix sequences to padded tensors."""
    n = len(prefixes)
    acts  = np.zeros((n, max_len), dtype=np.int64)
    feats = np.zeros((n, max_len, len(FEATURE_COLS)), dtype=np.float32)
    lens  = np.ones(n, dtype=np.int64)
    for i, p in enumerate(prefixes):
        seq_len = min(len(p), max_len)
        lens[i] = max(seq_len, 1)
        for j, e in enumerate(p[:seq_len]):
            acts[i, j] = activity_to_idx.get(e.get('activity', ''), 0)
            for k, col in enumerate(FEATURE_COLS):
                v = float(e.get(col, 0) or 0)
                feats[i, j, k] = 0.0 if np.isnan(v) else (v - feat_means[col]) / feat_stds[col]
    return acts, feats, lens


class RIMSDataset(Dataset):
    def __init__(self, acts, feats, lens, n_acts, n_feats, n_lens,
                 actions, rewards, terminals, proc_times, rem_times):
        self.acts, self.feats, self.lens = acts, feats, lens
        self.n_acts, self.n_feats, self.n_lens = n_acts, n_feats, n_lens
        self.actions   = actions
        self.rewards   = rewards
        self.terminals = terminals
        self.proc_times = proc_times
        self.rem_times  = rem_times

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, i):
        return {
            'acts':    torch.LongTensor(self.acts[i]),
            'feats':   torch.FloatTensor(self.feats[i]),
            'lens':    torch.LongTensor([self.lens[i]]),
            'n_acts':  torch.LongTensor(self.n_acts[i]),
            'n_feats': torch.FloatTensor(self.n_feats[i]),
            'n_lens':  torch.LongTensor([self.n_lens[i]]),
            'action':    torch.LongTensor([self.actions[i]]),
            'reward':    torch.FloatTensor([self.rewards[i]]),
            'terminal':  torch.FloatTensor([self.terminals[i]]),
            'proc_time': torch.FloatTensor([self.proc_times[i]]),
            'rem_time':  torch.FloatTensor([self.rem_times[i]]),
        }


def make_loader(df, int_idx, activity_to_idx, feat_means, feat_stds, max_len,
                proc_mean, proc_std, rem_mean, rem_std, batch_size, shuffle=True):
    """DataLoader for one intervention subset, or None if empty."""
    sub = df[df['intervention'] == int_idx]
    if sub.empty:
        return None

    acts,   feats,   lens   = encode(sub['prefix'].tolist(),      activity_to_idx, feat_means, feat_stds, max_len)
    n_acts, n_feats, n_lens = encode(sub['next_prefix'].tolist(), activity_to_idx, feat_means, feat_stds, max_len)

    proc = [(t - proc_mean) / proc_std for t in sub['proc_time'].tolist()]
    rem  = [(t - rem_mean)  / rem_std  for t in sub['rem_time'].tolist()]

    ds = RIMSDataset(acts, feats, lens, n_acts, n_feats, n_lens,
                     sub['action'].tolist(), sub['reward'].tolist(),
                     [float(t) for t in sub['terminal'].tolist()], proc, rem)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_rims(model, target, opt, tr, va, target_fn, args, w_t, w_o, w_q, norm_fn=None):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    best_val, best_state = float('inf'), copy.deepcopy(model.state_dict())
    patience_count = 0
    for epoch in range(args.epochs):
        model.train()
        tl = 0.0
        for b in tr:
            a   = b['action'].squeeze(1).to(device)
            r   = b['reward'].squeeze(1).to(device)
            r_n = norm_fn(r) if norm_fn is not None else r
            pt  = b['proc_time'].squeeze(1).to(device)
            rt_ = b['rem_time'].squeeze(1).to(device)

            timing, outcome, q = model(b['acts'].to(device), b['feats'].to(device), b['lens'].squeeze(1))

            loss_t = F.mse_loss(timing, torch.stack([pt, rt_], dim=1))
            loss_o = F.mse_loss(outcome.gather(1, a.unsqueeze(1)).squeeze(1), r_n)
            with torch.no_grad():
                tgt = target_fn(b)
            loss_q = F.mse_loss(q.gather(1, a.unsqueeze(1)).squeeze(1), tgt)

            loss = w_t * loss_t + w_o * loss_o + w_q * loss_q
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for p, tp in zip(model.parameters(), target.parameters()):
                tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)
            tl += loss.item()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for b in va:
                a  = b['action'].squeeze(1).to(device)
                _, _, q = model(b['acts'].to(device), b['feats'].to(device), b['lens'].squeeze(1))
                q_taken = q.gather(1, a.unsqueeze(1)).squeeze(1)
                vl += F.mse_loss(q_taken, target_fn(b)).item()
        vl /= max(len(va), 1)
        scheduler.step(vl)
        if vl < best_val - args.es_delta:
            best_val, best_state = vl, copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
        if (epoch + 1) % 10 == 0:
            print(f"  [{epoch+1:3d}/{args.epochs}] train={tl/len(tr):.4f}  val={vl:.4f}")
        if patience_count >= args.patience:
            print(f"  [early stop] epoch {epoch+1}, best_val={best_val:.4f}")
            break

    return best_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cases',    type=int,   default=10000)
    parser.add_argument('--confounded', action='store_true')
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=256)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--gamma',      type=float, default=0.99)
    parser.add_argument('--tau',        type=float, default=0.005)
    parser.add_argument('--w_timing',   type=float, default=0.3)
    parser.add_argument('--w_outcome',  type=float, default=0.3)
    parser.add_argument('--w_q',        type=float, default=0.4)
    parser.add_argument('--emb_dim',    type=int,   default=32)
    parser.add_argument('--hidden',     type=int,   default=128)
    parser.add_argument('--n_layers',   type=int,   default=2)
    parser.add_argument('--dropout',    type=float, default=0.2)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--patience',   type=int,   default=10)
    parser.add_argument('--es_delta',   type=float, default=1e-4)
    parser.add_argument('--steps',      type=int,   default=3, choices=[1, 2, 3])
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    suffix = "CONF" if args.confounded else "RCT"
    base   = f"data/rims_{suffix}_{args.n_cases}"
    step_tag = "" if args.steps == 3 else f"_steps{args.steps}"
    print(f"Training RIMS — {suffix} | lr={args.lr} w=(t={args.w_timing},o={args.w_outcome},q={args.w_q}) steps={args.steps}")

    df_train = load_pickle(f"{base}_trans_train{step_tag}.pkl")
    df_val   = load_pickle(f"{base}_trans_val{step_tag}.pkl")
    print(f"Train: {len(df_train)}, Val: {len(df_val)} transitions")

    activity_to_idx, feat_means, feat_stds = build_vocab_and_stats(df_train)
    n_activities = len(activity_to_idx)
    max_len = max((len(p) for p in list(df_train['prefix']) + list(df_train['next_prefix'])), default=1)

    all_proc = df_train['proc_time'].values
    all_rem  = df_train['rem_time'].values
    proc_mean, proc_std = float(all_proc.mean()), float(all_proc.std()) + 1e-8
    rem_mean,  rem_std  = float(all_rem.mean()),  float(all_rem.std())  + 1e-8

    term_r = df_train.loc[df_train['terminal'] == True, 'reward'].values
    r_mean = float(term_r.mean())
    r_std  = float(term_r.std()) + 1e-8
    def norm(r): return (r - r_mean) / r_std

    def loader(df, int_idx, shuffle=True):
        return make_loader(df, int_idx, activity_to_idx, feat_means, feat_stds, max_len,
                           proc_mean, proc_std, rem_mean, rem_std, args.batch_size, shuffle)

    def make_model(n_act):
        m  = RIMS_Model(n_activities, len(FEATURE_COLS), n_act, args.emb_dim, args.hidden, args.n_layers, args.dropout).to(device)
        mt = RIMS_Model(n_activities, len(FEATURE_COLS), n_act, args.emb_dim, args.hidden, args.n_layers, args.dropout).to(device)
        mt.load_state_dict(m.state_dict())
        return m, mt

    wt, wo, wq = args.w_timing, args.w_outcome, args.w_q

    cfg = {
        'n_activities':   n_activities,
        'n_features':     len(FEATURE_COLS),
        'feature_cols':   FEATURE_COLS,
        'activity_to_idx': activity_to_idx,
        'feat_means':     feat_means,
        'feat_stds':      feat_stds,
        'max_len':        max_len,
        'emb_dim':        args.emb_dim,
        'hidden':         args.hidden,
        'n_layers':       args.n_layers,
        'dropout':        args.dropout,
        'n_actions':      N_ACTIONS,
        'proc_mean':      proc_mean, 'proc_std': proc_std,
        'rem_mean':       rem_mean,  'rem_std':  rem_std,
        'steps':          args.steps,
    }

    if args.steps == 1:
        R1, R1t = make_model(N_ACTIONS[0])
        tr0 = loader(df_train, 0); va0 = loader(df_val, 0, False)
        print("\n[Q1]")
        best1 = train_rims(R1, R1t, optim.Adam(R1.parameters(), args.lr, weight_decay=1e-5), tr0, va0,
                           lambda b: norm(b['reward'].squeeze(1).to(device)), args, wt, wo, wq, norm)
        R1.load_state_dict(best1)
        save_dict = {'R1': R1.state_dict(), 'config': cfg}

    elif args.steps == 2:
        R1, R1t = make_model(N_ACTIONS[0])
        R2, R2t = make_model(N_ACTIONS[1])
        tr0 = loader(df_train, 0); va0 = loader(df_val, 0, False)
        tr1 = loader(df_train, 1); va1 = loader(df_val, 1, False)

        print("\n[Q2]")
        best2 = train_rims(R2, R2t, optim.Adam(R2.parameters(), args.lr, weight_decay=1e-5), tr1, va1,
                           lambda b: norm(b['reward'].squeeze(1).to(device)), args, wt, wo, wq, norm)
        R2.load_state_dict(best2); R2t.load_state_dict(best2)

        print("\n[Q1]")
        def tgt1_2step(b):
            r, term = b['reward'].squeeze(1).to(device), b['terminal'].squeeze(1).to(device)
            with torch.no_grad():
                nq2 = R2t.q_values(b['n_acts'].to(device), b['n_feats'].to(device), b['n_lens'].squeeze(1))
            return term * norm(r) + (1 - term) * args.gamma * nq2.max(1)[0]
        best1 = train_rims(R1, R1t, optim.Adam(R1.parameters(), args.lr, weight_decay=1e-5), tr0, va0, tgt1_2step, args, wt, wo, wq, norm)
        R1.load_state_dict(best1)
        save_dict = {'R1': R1.state_dict(), 'R2': R2.state_dict(), 'config': cfg}

    else:  # steps == 3
        R1, R1t = make_model(N_ACTIONS[0])
        R2, R2t = make_model(N_ACTIONS[1])
        R3, R3t = make_model(N_ACTIONS[2])
        tr0 = loader(df_train, 0); va0 = loader(df_val, 0, False)
        tr1 = loader(df_train, 1); va1 = loader(df_val, 1, False)
        tr2 = loader(df_train, 2); va2 = loader(df_val, 2, False)

        print("\n[Q3]")
        best3 = train_rims(R3, R3t, optim.Adam(R3.parameters(), args.lr, weight_decay=1e-5), tr2, va2,
                           lambda b: norm(b['reward'].squeeze(1).to(device)), args, wt, wo, wq, norm)
        R3.load_state_dict(best3); R3t.load_state_dict(best3)

        print("\n[Q2]")
        def tgt2(b):
            r, term = b['reward'].squeeze(1).to(device), b['terminal'].squeeze(1).to(device)
            with torch.no_grad():
                nq = R3t.q_values(b['n_acts'].to(device), b['n_feats'].to(device), b['n_lens'].squeeze(1))
            return term * norm(r) + (1 - term) * args.gamma * nq.max(1)[0]
        best2 = train_rims(R2, R2t, optim.Adam(R2.parameters(), args.lr, weight_decay=1e-5), tr1, va1, tgt2, args, wt, wo, wq, norm)
        R2.load_state_dict(best2); R2t.load_state_dict(best2)

        print("\n[Q1]")
        def tgt1(b):
            r, term = b['reward'].squeeze(1).to(device), b['terminal'].squeeze(1).to(device)
            with torch.no_grad():
                nq2 = R2t.q_values(b['n_acts'].to(device), b['n_feats'].to(device), b['n_lens'].squeeze(1))
                nq3 = R3t.q_values(b['n_acts'].to(device), b['n_feats'].to(device), b['n_lens'].squeeze(1))
            t = term * norm(r)
            m = (1 - term).bool()
            if m.any():
                t[m] = args.gamma * torch.max(nq2[m].max(1)[0], nq3[m].max(1)[0])
            return t
        best1 = train_rims(R1, R1t, optim.Adam(R1.parameters(), args.lr, weight_decay=1e-5), tr0, va0, tgt1, args, wt, wo, wq, norm)
        R1.load_state_dict(best1)
        save_dict = {'R1': R1.state_dict(), 'R2': R2.state_dict(), 'R3': R3.state_dict(), 'config': cfg}

    os.makedirs("models", exist_ok=True)
    model_path = f"models/rims_{suffix}_{args.n_cases}_s{args.seed}{step_tag}.pth"
    torch.save(save_dict, model_path)
    print(f"\n[OK] {model_path}")


if __name__ == "__main__":
    main()
