"""Train Single-Model CQL with LSTM-DQN: one shared network for all interventions + CQL penalty."""
import random
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

from shared import (
    load_pickle, FEATURE_COLS, N_ACTIONS, LSTM_DQN,
    build_vocab_and_stats, encode, seed_worker,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_ACTIONS = max(N_ACTIONS)


class SeqDataset(Dataset):
    def __init__(self, acts, feats, lens, n_acts, n_feats, n_lens, actions, rewards, terminals,
                 interventions, next_interventions):
        self.acts, self.feats, self.lens = acts, feats, lens
        self.n_acts, self.n_feats, self.n_lens = n_acts, n_feats, n_lens
        self.actions, self.rewards, self.terminals = actions, rewards, terminals
        self.interventions = interventions
        self.next_interventions = next_interventions

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, i):
        return {
            'acts':  torch.LongTensor(self.acts[i]),
            'feats': torch.FloatTensor(self.feats[i]),
            'lens':  torch.LongTensor([self.lens[i]]),
            'n_acts':  torch.LongTensor(self.n_acts[i]),
            'n_feats': torch.FloatTensor(self.n_feats[i]),
            'n_lens':  torch.LongTensor([self.n_lens[i]]),
            'action':  torch.LongTensor([self.actions[i]]),
            'reward':  torch.FloatTensor([self.rewards[i]]),
            'terminal': torch.FloatTensor([self.terminals[i]]),
            'intervention': torch.LongTensor([self.interventions[i]]),
            'next_intervention': torch.LongTensor([self.next_interventions[i]]),
        }


def make_loader(df, activity_to_idx, feat_means, feat_stds, max_len, batch_size, steps,
                shuffle=True, seed=42, n_activities=None):
    sub = df[df['intervention'] < steps]
    if sub.empty:
        return None

    acts, feats, lens = encode(sub['prefix'].tolist(), activity_to_idx, feat_means, feat_stds, max_len,
                               n_activities=n_activities)
    n_acts, n_feats, n_lens = encode(sub['next_prefix'].tolist(), activity_to_idx, feat_means, feat_stds, max_len,
                                     n_activities=n_activities)

    ds = SeqDataset(acts, feats, lens, n_acts, n_feats, n_lens,
                    sub['action'].tolist(), sub['reward'].tolist(),
                    [float(t) for t in sub['terminal'].tolist()],
                    [int(i) for i in sub['intervention'].tolist()],
                    [int(ni) for ni in sub['next_intervention'].tolist()])
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, worker_init_fn=seed_worker, generator=g)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cases',    type=int,   default=10000)
    parser.add_argument('--confounded', action='store_true')
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=256)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--alpha',      type=float, default=1.0)
    parser.add_argument('--gamma',      type=float, default=0.99)
    parser.add_argument('--tau',        type=float, default=0.005)
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
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    suffix = "CONF" if args.confounded else "RCT"
    base   = f"data/single_cql_{suffix}_{args.n_cases}"
    step_tag = "" if args.steps == 3 else f"_steps{args.steps}"
    print(f"Training Single-Model CQL — {suffix} | lr={args.lr} alpha={args.alpha} steps={args.steps}")

    df_train = load_pickle(f"{base}_trans_train{step_tag}.pkl")
    df_val   = load_pickle(f"{base}_trans_val{step_tag}.pkl")
    print(f"Train: {len(df_train)}, Val: {len(df_val)} transitions")

    activity_to_idx, feat_means, feat_stds = build_vocab_and_stats(df_train)
    n_activities = max(activity_to_idx.values(), default=0) + 1

    all_prefixes = list(df_train['prefix']) + list(df_train['next_prefix'])
    max_len = max((len(p) for p in all_prefixes), default=1)

    term_r = df_train.loc[df_train['terminal'] == True, 'reward'].values
    r_mean = float(term_r.mean())
    r_std  = float(term_r.std()) + 1e-8
    def norm(r): return (r - r_mean) / r_std

    model  = LSTM_DQN(n_activities, len(FEATURE_COLS), MAX_ACTIONS,
                      args.emb_dim, args.hidden, args.n_layers, args.dropout).to(device)
    target = LSTM_DQN(n_activities, len(FEATURE_COLS), MAX_ACTIONS,
                      args.emb_dim, args.hidden, args.n_layers, args.dropout).to(device)
    target.load_state_dict(model.state_dict())
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

    tr = make_loader(df_train, activity_to_idx, feat_means, feat_stds, max_len,
                     args.batch_size, args.steps, shuffle=True, seed=args.seed, n_activities=n_activities)
    va = make_loader(df_val, activity_to_idx, feat_means, feat_stds, max_len,
                     args.batch_size, args.steps, shuffle=False, seed=args.seed, n_activities=n_activities)

    cfg = {
        'n_activities': n_activities,
        'n_features':   len(FEATURE_COLS),
        'feature_cols': FEATURE_COLS,
        'activity_to_idx': activity_to_idx,
        'feat_means':   feat_means,
        'feat_stds':    feat_stds,
        'max_len':      max_len,
        'emb_dim':      args.emb_dim,
        'hidden':       args.hidden,
        'n_layers':     args.n_layers,
        'dropout':      args.dropout,
        'n_actions':    N_ACTIONS,
        'max_actions':  MAX_ACTIONS,
        'steps':        args.steps,
    }

    best_val, best_state = float('inf'), copy.deepcopy(model.state_dict())
    patience_count = 0

    for epoch in range(args.epochs):
        model.train()
        tl = 0.0
        for b in tr:
            a    = b['action'].squeeze(1).to(device)
            int_id = b['intervention'].squeeze(1).to(device)
            ni   = b['next_intervention'].squeeze(1).to(device)
            r    = b['reward'].squeeze(1).to(device)
            term = b['terminal'].squeeze(1).to(device)

            q = model(b['acts'].to(device), b['feats'].to(device), b['lens'].squeeze(1))
            q_taken = q.gather(1, a.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                nq = target(b['n_acts'].to(device), b['n_feats'].to(device), b['n_lens'].squeeze(1))
                # Mask invalid actions for next intervention
                nq_masked = nq.clone()
                for j in range(3):
                    mask = (ni == j)
                    if mask.any():
                        nq_masked[mask, N_ACTIONS[j]:] = float('-inf')
                # Terminal or unknown next_intervention: max over all
                mask_term = (term == 1) | (ni < 0)
                if mask_term.any():
                    nq_masked[mask_term] = nq[mask_term]
                max_nq = nq_masked.max(1)[0]
                targets = term * norm(r) + (1 - term) * args.gamma * max_nq

            td_loss = F.mse_loss(q_taken, targets)

            # CQL penalty: mask invalid actions for current intervention before logsumexp
            q_masked = q.clone()
            for j in range(3):
                mask = (int_id == j)
                if mask.any():
                    q_masked[mask, N_ACTIONS[j]:] = float('-inf')
            cql_loss = (torch.logsumexp(q_masked, 1) - q_taken).mean()

            loss = td_loss + args.alpha * cql_loss
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
                a = b['action'].squeeze(1).to(device)
                q = model(b['acts'].to(device), b['feats'].to(device), b['lens'].squeeze(1))
                q_taken = q.gather(1, a.unsqueeze(1)).squeeze(1)

                ni   = b['next_intervention'].squeeze(1).to(device)
                r    = b['reward'].squeeze(1).to(device)
                term = b['terminal'].squeeze(1).to(device)
                nq = target(b['n_acts'].to(device), b['n_feats'].to(device), b['n_lens'].squeeze(1))
                nq_masked = nq.clone()
                for j in range(3):
                    mask = (ni == j)
                    if mask.any():
                        nq_masked[mask, N_ACTIONS[j]:] = float('-inf')
                mask_term = (term == 1) | (ni < 0)
                if mask_term.any():
                    nq_masked[mask_term] = nq[mask_term]
                max_nq = nq_masked.max(1)[0]
                targets = term * norm(r) + (1 - term) * args.gamma * max_nq
                vl += F.mse_loss(q_taken, targets).item()

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

    model.load_state_dict(best_state)
    os.makedirs("models", exist_ok=True)
    model_path = f"models/single_cql_{suffix}_{args.n_cases}_s{args.seed}{step_tag}.pth"
    torch.save({'model': model.state_dict(), 'config': cfg}, model_path)
    print(f"\n[OK] {model_path}")


if __name__ == "__main__":
    main()
