"""Train ProCause LSTM S-learner: causal reward estimation, no backward TD."""
import sys
import os
import argparse
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
os.chdir(project_root)

from shared import load_pickle, FEATURE_COLS, N_ACTIONS, build_vocab_and_stats, encode, seed_worker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM_SLearner(nn.Module):
    """LSTM encoder with action embedding for outcome prediction."""

    def __init__(self, n_activities, n_features, n_actions, emb_dim=32, action_emb_dim=16,
                 hidden=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.n_actions = n_actions
        self.emb  = nn.Embedding(n_activities, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim + n_features, hidden, n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.action_emb = nn.Embedding(n_actions, action_emb_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden + action_emb_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, acts, feats, lens, actions):
        x = torch.cat([self.emb(acts), feats], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h_last = h[-1]
        a_emb = self.action_emb(actions)
        return self.head(torch.cat([h_last, a_emb], dim=-1)).squeeze(-1)


class SLearnerDataset(Dataset):
    """Dataset for S-learner: (prefix, action) -> normalized outcome."""

    def __init__(self, acts, feats, lens, actions, outcomes):
        self.acts, self.feats, self.lens = acts, feats, lens
        self.actions = actions
        self.outcomes = outcomes

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, i):
        return {
            'acts': torch.LongTensor(self.acts[i]),
            'feats': torch.FloatTensor(self.feats[i]),
            'lens': torch.LongTensor([self.lens[i]]),
            'action': torch.LongTensor([self.actions[i]]),
            'outcome': torch.FloatTensor([self.outcomes[i]]),
        }


def train_slearner(model, train_loader, val_loader, args):
    """Train S-learner with early stopping."""
    opt = optim.Adam(model.parameters(), lr=args.slearner_lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    best_val, best_state = float('inf'), copy.deepcopy(model.state_dict())
    patience_count = 0

    for epoch in range(args.slearner_epochs):
        model.train()
        tl = 0.0
        for b in train_loader:
            pred = model(b['acts'].to(device), b['feats'].to(device),
                         b['lens'].squeeze(1), b['action'].squeeze(1).to(device))
            loss = F.mse_loss(pred, b['outcome'].squeeze(1).to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for b in val_loader:
                pred = model(b['acts'].to(device), b['feats'].to(device),
                             b['lens'].squeeze(1), b['action'].squeeze(1).to(device))
                vl += F.mse_loss(pred, b['outcome'].squeeze(1).to(device)).item()
        vl /= max(len(val_loader), 1)
        scheduler.step(vl)

        if vl < best_val - args.es_delta:
            best_val, best_state = vl, copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
        if (epoch + 1) % 10 == 0:
            print(f"  [{epoch+1:3d}/{args.slearner_epochs}] train={tl/len(train_loader):.4f}  val={vl:.4f}")
        if patience_count >= args.patience:
            print(f"  [early stop] epoch {epoch+1}, best_val={best_val:.4f}")
            break

    model.load_state_dict(best_state)
    return model


def compute_cate(slearner, acts, feats, lens, n_actions, batch_size=1024):
    """Compute CATE for all actions relative to action 0, in batches."""
    slearner.eval()
    n = len(lens)
    cate = np.zeros((n, n_actions), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            acts_t = torch.LongTensor(acts[start:end]).to(device)
            feats_t = torch.FloatTensor(feats[start:end]).to(device)
            lens_t = torch.LongTensor(lens[start:end])
            bs = end - start

            preds = []
            for a in range(n_actions):
                action_t = torch.full((bs,), a, dtype=torch.long, device=device)
                pred = slearner(acts_t, feats_t, lens_t, action_t).cpu().numpy()
                preds.append(pred)

            baseline = preds[0]
            for a in range(n_actions):
                cate[start:end, a] = preds[a] - baseline

    return cate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cases',        type=int,   default=10000)
    parser.add_argument('--confounded',     action='store_true')
    parser.add_argument('--batch_size',     type=int,   default=256)
    parser.add_argument('--emb_dim',        type=int,   default=32)
    parser.add_argument('--hidden',         type=int,   default=128)
    parser.add_argument('--n_layers',       type=int,   default=2)
    parser.add_argument('--dropout',        type=float, default=0.2)
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--patience',       type=int,   default=15)
    parser.add_argument('--es_delta',       type=float, default=1e-4)
    parser.add_argument('--steps',          type=int,   default=3, choices=[1, 2, 3])
    parser.add_argument('--slearner_epochs', type=int,  default=150)
    parser.add_argument('--slearner_lr',    type=float, default=1e-3)
    parser.add_argument('--action_emb_dim', type=int,   default=16)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    suffix = "CONF" if args.confounded else "RCT"
    base   = f"data/procause_lstm_{suffix}_{args.n_cases}"
    step_tag = "" if args.steps == 3 else f"_steps{args.steps}"
    print(f"Training ProCause LSTM — {suffix} | steps={args.steps}")

    df_train = load_pickle(f"{base}_trans_train{step_tag}.pkl")
    df_val   = load_pickle(f"{base}_trans_val{step_tag}.pkl")
    print(f"Train: {len(df_train)}, Val: {len(df_val)} transitions")

    activity_to_idx, feat_means, feat_stds = build_vocab_and_stats(df_train)
    n_activities = len(activity_to_idx)

    all_prefixes = list(df_train['prefix']) + list(df_train['next_prefix'])
    max_len = max((len(p) for p in all_prefixes), default=1)

    # Global outcome normalization (used for S-learner targets)
    all_outcomes = df_train['case_outcome'].values.astype(np.float64)
    outcome_mean = float(np.mean(all_outcomes))
    outcome_std  = float(np.std(all_outcomes)) + 1e-8
    print(f"Outcome stats: mean={outcome_mean:.1f}, std={outcome_std:.1f}")

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
        'steps':        args.steps,
        'action_emb_dim': args.action_emb_dim,
    }

    save_dict = {'config': cfg}
    active_interventions = list(range(args.steps))

    for int_idx in active_interventions:
        n_act = N_ACTIONS[int_idx]
        sub_train = df_train[df_train['intervention'] == int_idx]
        sub_val   = df_val[df_val['intervention'] == int_idx]

        if sub_train.empty:
            print(f"  [skip] No training data for intervention {int_idx}")
            continue

        # Check action diversity
        unique_actions = sub_train['action'].nunique()
        if unique_actions < 2:
            print(f"  [skip] Int.{int_idx}: only action {sub_train['action'].iloc[0]} in data, cannot estimate CATE")
            continue

        # Print per-action outcome stats
        for a in sorted(sub_train['action'].unique()):
            mask = sub_train['action'] == a
            m = sub_train.loc[mask, 'case_outcome'].mean()
            print(f"  Int.{int_idx} action={a}: n={mask.sum()}, outcome_mean={m:.1f}")

        # Encode prefixes
        tr_acts, tr_feats, tr_lens = encode(sub_train['prefix'].tolist(), activity_to_idx, feat_means, feat_stds, max_len)
        va_acts, va_feats, va_lens = encode(sub_val['prefix'].tolist(), activity_to_idx, feat_means, feat_stds, max_len)

        # Normalize outcomes for S-learner training
        tr_outcomes_norm = [(o - outcome_mean) / outcome_std for o in sub_train['case_outcome'].tolist()]
        va_outcomes_norm = [(o - outcome_mean) / outcome_std for o in sub_val['case_outcome'].tolist()]

        # Phase 1: Train S-learner on normalized outcomes
        print(f"\n[S-learner Int.{int_idx}]")
        slearner = LSTM_SLearner(n_activities, len(FEATURE_COLS), n_act,
                                 args.emb_dim, args.action_emb_dim, args.hidden,
                                 args.n_layers, args.dropout).to(device)

        tr_sl_ds = SLearnerDataset(tr_acts, tr_feats, tr_lens,
                                   sub_train['action'].tolist(), tr_outcomes_norm)
        va_sl_ds = SLearnerDataset(va_acts, va_feats, va_lens,
                                   sub_val['action'].tolist(), va_outcomes_norm)
        g = torch.Generator()
        g.manual_seed(args.seed + int_idx)
        tr_sl_loader = DataLoader(tr_sl_ds, batch_size=args.batch_size, shuffle=True,
                                  worker_init_fn=seed_worker, generator=g)
        va_sl_loader = DataLoader(va_sl_ds, batch_size=args.batch_size, shuffle=False)

        slearner = train_slearner(slearner, tr_sl_loader, va_sl_loader, args)

        # Save S-learner (used directly at eval time for action selection)
        save_dict[f'S{int_idx+1}'] = slearner.state_dict()
        save_dict[f'n_actions_{int_idx}'] = n_act

    os.makedirs("models", exist_ok=True)
    model_path = f"models/procause_lstm_{suffix}_{args.n_cases}_s{args.seed}{step_tag}.pth"
    torch.save(save_dict, model_path)
    print(f"\n[OK] {model_path}")


if __name__ == "__main__":
    main()
