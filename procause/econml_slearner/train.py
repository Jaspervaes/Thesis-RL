"""Train ProCause EconML S-learner + LSTM-DQN: causal reward estimation, no backward TD."""
import sys
import os
import argparse
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
os.chdir(project_root)

from shared import load_pickle, STATE_DIM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = ['amount', 'est_quality', 'unc_quality', 'interest_rate', 'cum_cost', 'elapsed_time']
N_ACTIONS    = [2, 2, 3]


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

    def _vals(c):
        return [float(v) for e in all_events if not np.isnan(v := float(e.get(c, 0) or 0))]
    feat_means = {c: (np.mean(_vals(c)) if _vals(c) else 0.0) for c in FEATURE_COLS}
    feat_stds  = {c: max(np.std(_vals(c))  if _vals(c) else 0.0, 1e-8) for c in FEATURE_COLS}

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
                v = float(e.get(col, 0) or 0)
                feats[i, j, k] = 0.0 if np.isnan(v) else (v - feat_means[col]) / feat_stds[col]

    return acts, feats, lens


class CATEDataset(Dataset):
    """Dataset for Q-network: (prefix) -> CATE targets for all actions."""

    def __init__(self, acts, feats, lens, cate_targets):
        self.acts, self.feats, self.lens = acts, feats, lens
        self.cate_targets = cate_targets

    def __len__(self):
        return len(self.cate_targets)

    def __getitem__(self, i):
        return {
            'acts': torch.LongTensor(self.acts[i]),
            'feats': torch.FloatTensor(self.feats[i]),
            'lens': torch.LongTensor([self.lens[i]]),
            'cate': torch.FloatTensor(self.cate_targets[i]),
        }


def train_econml_slearner(states, actions, outcomes, n_actions):
    """Train S-learner via GradientBoosting. Returns model, scaler, outcome stats."""
    state_scaler = StandardScaler()
    states_norm = state_scaler.fit_transform(states)
    outcome_mean, outcome_std = outcomes.mean(), outcomes.std() + 1e-8
    outcomes_norm = (outcomes - outcome_mean) / outcome_std

    X_train = np.column_stack([states_norm, actions.reshape(-1, 1)])
    model = GradientBoostingRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42)
    model.fit(X_train, outcomes_norm)

    return model, state_scaler, outcome_mean, outcome_std


def train_q_on_cate(model, train_loader, val_loader, args):
    """Train Q-network on CATE targets (full-information MSE, no bootstrapping)."""
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    best_val, best_state = float('inf'), copy.deepcopy(model.state_dict())
    patience_count = 0

    for epoch in range(args.epochs):
        model.train()
        tl = 0.0
        for b in train_loader:
            q = model(b['acts'].to(device), b['feats'].to(device), b['lens'].squeeze(1))
            target = b['cate'].to(device)
            loss = F.mse_loss(q, target)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for b in val_loader:
                q = model(b['acts'].to(device), b['feats'].to(device), b['lens'].squeeze(1))
                vl += F.mse_loss(q, b['cate'].to(device)).item()
        vl /= max(len(val_loader), 1)
        scheduler.step(vl)

        if vl < best_val - args.es_delta:
            best_val, best_state = vl, copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
        if (epoch + 1) % 10 == 0:
            print(f"  [{epoch+1:3d}/{args.epochs}] train={tl/len(train_loader):.4f}  val={vl:.4f}")
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
    parser.add_argument('--emb_dim',    type=int,   default=32)
    parser.add_argument('--hidden',     type=int,   default=128)
    parser.add_argument('--n_layers',   type=int,   default=2)
    parser.add_argument('--dropout',    type=float, default=0.2)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--patience',   type=int,   default=15)
    parser.add_argument('--es_delta',   type=float, default=1e-4)
    parser.add_argument('--steps',      type=int,   default=3, choices=[1, 2, 3])
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    suffix = "CONF" if args.confounded else "RCT"
    base   = f"data/procause_econml_{suffix}_{args.n_cases}"
    step_tag = "" if args.steps == 3 else f"_steps{args.steps}"
    print(f"Training ProCause EconML — {suffix} | steps={args.steps}")

    df_train = load_pickle(f"{base}_trans_train{step_tag}.pkl")
    df_val   = load_pickle(f"{base}_trans_val{step_tag}.pkl")
    print(f"Train: {len(df_train)}, Val: {len(df_val)} transitions")

    activity_to_idx, feat_means, feat_stds = build_vocab_and_stats(df_train)
    n_activities = len(activity_to_idx)

    all_prefixes = list(df_train['prefix']) + list(df_train['next_prefix'])
    max_len = max((len(p) for p in all_prefixes), default=1)

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

        # Print per-action stats
        for a in sorted(sub_train['action'].unique()):
            mask = sub_train['action'] == a
            m = sub_train.loc[mask, 'case_outcome'].mean()
            print(f"  Int.{int_idx} action={a}: n={mask.sum()}, outcome_mean={m:.1f}")

        # Encode prefixes for Q-network
        tr_acts, tr_feats, tr_lens = encode(sub_train['prefix'].tolist(), activity_to_idx, feat_means, feat_stds, max_len)
        va_acts, va_feats, va_lens = encode(sub_val['prefix'].tolist(), activity_to_idx, feat_means, feat_stds, max_len)

        # Phase 1: Train GBR S-learner on state vectors
        print(f"\n[GBR S-learner Int.{int_idx}]")
        tr_states = np.stack(sub_train['state'].tolist())
        tr_actions = np.array(sub_train['action'].tolist())
        tr_outcomes = np.array(sub_train['case_outcome'].tolist(), dtype=np.float64)

        va_states = np.stack(sub_val['state'].tolist())
        va_actions = np.array(sub_val['action'].tolist())
        va_outcomes = np.array(sub_val['case_outcome'].tolist(), dtype=np.float64)

        gbr_model, state_scaler, outcome_mean, outcome_std = train_econml_slearner(
            tr_states, tr_actions, tr_outcomes, n_act)

        # Validate: predict for each action on val set
        va_states_norm = state_scaler.transform(va_states)
        val_preds = []
        for a in range(n_act):
            X_val = np.column_stack([va_states_norm, np.full(len(va_states), a)])
            val_preds.append(gbr_model.predict(X_val) * outcome_std + outcome_mean)
        val_preds = np.stack(val_preds, axis=1)
        print(f"  Val pred means per action: {np.mean(val_preds, axis=0)}")

        # Save GBR S-learner artifacts
        import pickle
        save_dict[f'gbr_{int_idx}'] = pickle.dumps(gbr_model)
        save_dict[f'scaler_{int_idx}'] = pickle.dumps(state_scaler)
        save_dict[f'outcome_mean_{int_idx}'] = outcome_mean
        save_dict[f'outcome_std_{int_idx}'] = outcome_std
        save_dict[f'n_actions_{int_idx}'] = n_act

    os.makedirs("models", exist_ok=True)
    model_path = f"models/procause_econml_{suffix}_{args.n_cases}_s{args.seed}{step_tag}.pth"
    torch.save(save_dict, model_path)
    print(f"\n[OK] {model_path}")


if __name__ == "__main__":
    main()
