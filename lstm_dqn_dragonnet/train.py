"""Train LSTM-DQN-DragonNet: LSTM-DragonNet causal S-learner (causal rewards) + LSTM-DQN (backward TD)."""
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
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
os.chdir(project_root)

from shared import (
    load_pickle, FEATURE_COLS, N_ACTIONS, LSTM_DQN,
    build_vocab_and_stats, encode, seed_worker,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Phase 1: LSTM-DragonNet  (prefix, action) -> outcome + propensity
# ---------------------------------------------------------------------------

class LSTM_DragonNet(nn.Module):
    """Shared LSTM encoder Φ + per-action outcome heads + propensity head (DragonNet)."""

    def __init__(self, n_activities, n_features, n_actions,
                 emb_dim=32, hidden=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.n_actions = n_actions
        self.hidden    = hidden
        self.emb  = nn.Embedding(n_activities, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim + n_features, hidden, n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        # Per-action outcome heads — one per possible action for this intervention
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
            for _ in range(n_actions)
        ])
        # Propensity head — predicts which action was taken (behaviour policy)
        self.prop_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_actions),
        )
        # Targeted regularisation scalar (one per model, learned)
        self.eps = nn.Parameter(torch.zeros(1))

    def encode(self, acts, feats, lens):
        x = torch.cat([self.emb(acts), feats], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return h[-1]                                  # (B, hidden)

    def forward(self, acts, feats, lens, actions):
        r = self.encode(acts, feats, lens)            # (B, hidden)
        # Factual outcome predictions (observed action only)
        preds = torch.zeros(r.size(0), device=r.device)
        for a in range(self.n_actions):
            mask = (actions == a)
            if mask.any():
                preds[mask] = self.heads[a](r[mask]).squeeze(-1)
        # Propensity logits over all actions
        prop_logits = self.prop_head(r)               # (B, n_actions)
        return preds, prop_logits, r


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


def train_dragonnet(model, train_loader, val_loader, args):
    """Train DragonNet with factual + propensity + targeted losses; early stop on val factual MSE."""
    opt = optim.Adam(model.parameters(), lr=args.slearner_lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    best_val, best_state = float('inf'), copy.deepcopy(model.state_dict())
    patience_count = 0

    for epoch in range(args.slearner_epochs):
        model.train()
        tl = 0.0
        for b in train_loader:
            actions = b['action'].squeeze(1).to(device)
            target  = b['outcome'].squeeze(1).to(device)

            pred, prop_logits, _ = model(
                b['acts'].to(device), b['feats'].to(device),
                b['lens'].squeeze(1), actions,
            )

            factual_loss = F.mse_loss(pred, target)

            propensity_loss = F.cross_entropy(prop_logits, actions)

            # Targeted regularisation: nudge outcome head toward observed outcome
            # using inverse propensity of the observed action (Shi et al. 2019)
            prop_scores = torch.softmax(prop_logits, dim=-1)                  # (B, n_actions)
            obs_prop    = prop_scores.gather(1, actions.unsqueeze(1)).squeeze(1).clamp(min=1e-6)
            targeted_loss = F.mse_loss(pred + model.eps * (1.0 / obs_prop), target)

            loss = (factual_loss
                    + args.alpha_prop     * propensity_loss
                    + args.alpha_targeted * targeted_loss)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()

        # Early-stopping criterion: validation factual MSE only
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for b in val_loader:
                actions = b['action'].squeeze(1).to(device)
                pred, _, _ = model(
                    b['acts'].to(device), b['feats'].to(device),
                    b['lens'].squeeze(1), actions,
                )
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


def predict_outcomes(dragonnet, acts, feats, lens, actions, batch_size=1024):
    """Predict factual outcomes for given (prefix, action) pairs in batches.
    Only the outcome head predictions are used; propensity outputs are discarded."""
    dragonnet.eval()
    n = len(lens)
    preds = np.zeros(n, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            acts_t   = torch.LongTensor(acts[start:end]).to(device)
            feats_t  = torch.FloatTensor(feats[start:end]).to(device)
            lens_t   = torch.LongTensor(lens[start:end])
            action_t = torch.LongTensor(actions[start:end]).to(device)
            out, _, _ = dragonnet(acts_t, feats_t, lens_t, action_t)
            preds[start:end] = out.cpu().numpy()

    return preds


# ---------------------------------------------------------------------------
# Phase 3: LSTM-DQN with backward TD (same architecture as lstm/train.py)
# ---------------------------------------------------------------------------

class SeqDataset(Dataset):
    def __init__(self, acts, feats, lens, n_acts, n_feats, n_lens, actions, rewards, terminals, next_interventions):
        self.acts, self.feats, self.lens = acts, feats, lens
        self.n_acts, self.n_feats, self.n_lens = n_acts, n_feats, n_lens
        self.actions, self.rewards, self.terminals = actions, rewards, terminals
        self.next_interventions = next_interventions

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, i):
        return {
            'acts': torch.LongTensor(self.acts[i]),
            'feats': torch.FloatTensor(self.feats[i]),
            'lens': torch.LongTensor([self.lens[i]]),
            'n_acts': torch.LongTensor(self.n_acts[i]),
            'n_feats': torch.FloatTensor(self.n_feats[i]),
            'n_lens': torch.LongTensor([self.n_lens[i]]),
            'action': torch.LongTensor([self.actions[i]]),
            'reward': torch.FloatTensor([self.rewards[i]]),
            'terminal': torch.FloatTensor([self.terminals[i]]),
            'next_intervention': torch.LongTensor([self.next_interventions[i]]),
        }


def make_dqn_loader(df, int_idx, activity_to_idx, feat_means, feat_stds, max_len, batch_size,
                    shuffle=True, seed=42):
    sub = df[df['intervention'] == int_idx]
    if sub.empty:
        return None

    acts, feats, lens = encode(sub['prefix'].tolist(), activity_to_idx, feat_means, feat_stds, max_len)
    n_acts, n_feats, n_lens = encode(sub['next_prefix'].tolist(), activity_to_idx, feat_means, feat_stds, max_len)

    ds = SeqDataset(acts, feats, lens, n_acts, n_feats, n_lens,
                    sub['action'].tolist(), sub['reward'].tolist(),
                    [float(t) for t in sub['terminal'].tolist()],
                    [int(ni) for ni in sub['next_intervention'].tolist()])
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, worker_init_fn=seed_worker, generator=g)


def train_q(model, target, opt, tr, va, target_fn, args):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    best_val, best_state = float('inf'), copy.deepcopy(model.state_dict())
    patience_count = 0
    for epoch in range(args.dqn_epochs):
        model.train()
        tl = 0.0
        for b in tr:
            a = b['action'].squeeze(1).to(device)
            q = model(b['acts'].to(device), b['feats'].to(device), b['lens'].squeeze(1))
            q_taken = q.gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                tgt = target_fn(b)
            loss = F.mse_loss(q_taken, tgt)
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
                vl += F.mse_loss(q_taken, target_fn(b)).item()
        vl /= max(len(va), 1)
        scheduler.step(vl)
        if vl < best_val - args.es_delta:
            best_val, best_state = vl, copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
        if (epoch + 1) % 10 == 0:
            print(f"  [{epoch+1:3d}/{args.dqn_epochs}] train={tl/len(tr):.4f}  val={vl:.4f}")
        if patience_count >= args.dqn_patience:
            print(f"  [early stop] epoch {epoch+1}, best_val={best_val:.4f}")
            break

    return best_state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    # DragonNet hyperparams
    parser.add_argument('--slearner_epochs', type=int,  default=150)
    parser.add_argument('--slearner_lr',    type=float, default=1e-3)
    parser.add_argument('--alpha_prop',     type=float, default=1.0)
    parser.add_argument('--alpha_targeted', type=float, default=1.0)
    # DQN hyperparams
    parser.add_argument('--dqn_epochs',     type=int,   default=50)
    parser.add_argument('--dqn_lr',         type=float, default=1e-3)
    parser.add_argument('--dqn_patience',   type=int,   default=10)
    parser.add_argument('--gamma',          type=float, default=0.99)
    parser.add_argument('--tau',            type=float, default=0.005)
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
    base   = f"data/lstm_dqn_dragonnet_{suffix}_{args.n_cases}"
    step_tag = "" if args.steps == 3 else f"_steps{args.steps}"
    print(f"Training LSTM-DQN-DragonNet — {suffix} | steps={args.steps}")

    df_train = load_pickle(f"{base}_trans_train{step_tag}.pkl")
    df_val   = load_pickle(f"{base}_trans_val{step_tag}.pkl")
    print(f"Train: {len(df_train)}, Val: {len(df_val)} transitions")

    activity_to_idx, feat_means, feat_stds = build_vocab_and_stats(df_train)
    n_activities = max(activity_to_idx.values(), default=0) + 1

    all_prefixes = list(df_train['prefix']) + list(df_train['next_prefix'])
    max_len = max((len(p) for p in all_prefixes), default=1)

    # Outcome normalization for S-learner targets
    all_outcomes = df_train['case_outcome'].values.astype(np.float64)
    outcome_mean = float(np.mean(all_outcomes))
    outcome_std  = float(np.std(all_outcomes)) + 1e-8
    print(f"Outcome stats: mean={outcome_mean:.1f}, std={outcome_std:.1f}")

    cfg = {
        'n_activities':  n_activities,
        'n_features':    len(FEATURE_COLS),
        'feature_cols':  FEATURE_COLS,
        'activity_to_idx': activity_to_idx,
        'feat_means':    feat_means,
        'feat_stds':     feat_stds,
        'max_len':       max_len,
        'emb_dim':       args.emb_dim,
        'hidden':        args.hidden,
        'n_layers':      args.n_layers,
        'dropout':       args.dropout,
        'n_actions':     N_ACTIONS,
        'steps':         args.steps,
        'alpha_prop':    args.alpha_prop,
        'alpha_targeted': args.alpha_targeted,
    }

    save_dict = {'config': cfg}
    active_interventions = list(range(args.steps))
    slearners = {}

    # ===================================================================
    # Phase 1: Train DragonNet per intervention
    # ===================================================================
    print(f"\n{'='*50}")
    print("Phase 1: Training DragonNets")
    print('='*50)

    for int_idx in active_interventions:
        n_act = N_ACTIONS[int_idx]
        sub_train = df_train[df_train['intervention'] == int_idx]
        sub_val   = df_val[df_val['intervention'] == int_idx]

        if sub_train.empty:
            print(f"  [skip] No training data for intervention {int_idx}")
            continue

        unique_actions = sub_train['action'].nunique()
        if unique_actions < 2:
            print(f"  [skip] Int.{int_idx}: only action {sub_train['action'].iloc[0]} in data")
            continue

        for a in sorted(sub_train['action'].unique()):
            mask = sub_train['action'] == a
            m = sub_train.loc[mask, 'case_outcome'].mean()
            print(f"  Int.{int_idx} action={a}: n={mask.sum()}, outcome_mean={m:.1f}")

        tr_acts, tr_feats, tr_lens = encode(sub_train['prefix'].tolist(), activity_to_idx, feat_means, feat_stds, max_len)
        va_acts, va_feats, va_lens = encode(sub_val['prefix'].tolist(), activity_to_idx, feat_means, feat_stds, max_len)

        tr_outcomes_norm = [(o - outcome_mean) / outcome_std for o in sub_train['case_outcome'].tolist()]
        va_outcomes_norm = [(o - outcome_mean) / outcome_std for o in sub_val['case_outcome'].tolist()]

        print(f"\n[DragonNet Int.{int_idx}]")
        dragonnet = LSTM_DragonNet(
            n_activities, len(FEATURE_COLS), n_act,
            args.emb_dim, args.hidden, args.n_layers, args.dropout,
        ).to(device)

        tr_sl_ds = SLearnerDataset(tr_acts, tr_feats, tr_lens,
                                   sub_train['action'].tolist(), tr_outcomes_norm)
        va_sl_ds = SLearnerDataset(va_acts, va_feats, va_lens,
                                   sub_val['action'].tolist(), va_outcomes_norm)
        g = torch.Generator()
        g.manual_seed(args.seed + int_idx)
        tr_sl_loader = DataLoader(tr_sl_ds, batch_size=args.batch_size, shuffle=True,
                                  worker_init_fn=seed_worker, generator=g)
        va_sl_loader = DataLoader(va_sl_ds, batch_size=args.batch_size, shuffle=False)

        dragonnet = train_dragonnet(dragonnet, tr_sl_loader, va_sl_loader, args)
        slearners[int_idx] = dragonnet
        save_dict[f'dragonnet_{int_idx+1}'] = dragonnet.state_dict()

    # ===================================================================
    # Phase 2: Compute causal rewards using DragonNet outcome heads
    # Only the 'reward' column at terminal transitions is replaced;
    # non-terminal rewards stay 0.0. Causal rewards then feed Phase 3 DQN.
    # Propensity head and targeted regularizer are not used here.
    # ===================================================================
    print(f"\n{'='*50}")
    print("Phase 2: Computing causal rewards")
    print('='*50)

    df_train_causal = df_train.copy()
    df_val_causal   = df_val.copy()

    for int_idx, slearner in slearners.items():
        for df_c, label in [(df_train_causal, 'train'), (df_val_causal, 'val')]:
            terminal_mask = (df_c['intervention'] == int_idx) & (df_c['terminal'] == True)
            if not terminal_mask.any():
                continue

            sub = df_c[terminal_mask]
            acts, feats, lens = encode(sub['prefix'].tolist(), activity_to_idx, feat_means, feat_stds, max_len)
            actions = np.array(sub['action'].tolist())

            preds_norm = predict_outcomes(slearner, acts, feats, lens, actions)
            causal_rewards = preds_norm * outcome_std + outcome_mean

            df_c.loc[terminal_mask, 'reward'] = causal_rewards
            orig_mean = sub['reward'].mean()
            causal_mean = float(np.mean(causal_rewards))
            print(f"  Int.{int_idx} {label}: {terminal_mask.sum()} terminal transitions, "
                  f"observed_reward={orig_mean:.1f}, causal_reward={causal_mean:.1f}")

    # ===================================================================
    # Phase 3: Train LSTM-DQN with causal rewards (backward TD)
    # ===================================================================
    print(f"\n{'='*50}")
    print("Phase 3: Training LSTM-DQN with causal rewards")
    print('='*50)

    # Normalize causal rewards for DQN training
    term_r = df_train_causal.loc[df_train_causal['terminal'] == True, 'reward'].values
    r_mean = float(term_r.mean())
    r_std  = float(term_r.std()) + 1e-8
    def norm(r): return (r - r_mean) / r_std
    print(f"DQN reward normalization: mean={r_mean:.1f}, std={r_std:.1f}")

    bs = args.batch_size

    def make_model(n_act):
        m  = LSTM_DQN(n_activities, len(FEATURE_COLS), n_act, args.emb_dim, args.hidden, args.n_layers, args.dropout).to(device)
        mt = LSTM_DQN(n_activities, len(FEATURE_COLS), n_act, args.emb_dim, args.hidden, args.n_layers, args.dropout).to(device)
        mt.load_state_dict(m.state_dict())
        return m, mt

    def loader(df, int_idx, shuffle=True):
        return make_dqn_loader(df, int_idx, activity_to_idx, feat_means, feat_stds, max_len, bs, shuffle,
                               seed=args.seed + int_idx)

    if args.steps == 1:
        Q1, Q1t = make_model(N_ACTIONS[0])
        tr0 = loader(df_train_causal, 0); va0 = loader(df_val_causal, 0, False)
        print("\n[Q1]")
        best1 = train_q(Q1, Q1t, optim.Adam(Q1.parameters(), args.dqn_lr, weight_decay=1e-5), tr0, va0,
                        lambda b: norm(b['reward'].squeeze(1).to(device)), args)
        Q1.load_state_dict(best1)
        save_dict['Q1'] = Q1.state_dict()

    elif args.steps == 2:
        Q1, Q1t = make_model(N_ACTIONS[0])
        Q2, Q2t = make_model(N_ACTIONS[1])
        tr0 = loader(df_train_causal, 0); va0 = loader(df_val_causal, 0, False)
        tr1 = loader(df_train_causal, 1); va1 = loader(df_val_causal, 1, False)

        print("\n[Q2]")
        best2 = train_q(Q2, Q2t, optim.Adam(Q2.parameters(), args.dqn_lr, weight_decay=1e-5), tr1, va1,
                        lambda b: norm(b['reward'].squeeze(1).to(device)), args)
        Q2.load_state_dict(best2); Q2t.load_state_dict(best2)

        print("\n[Q1]")
        def tgt1_2step(b):
            r, term = b['reward'].squeeze(1).to(device), b['terminal'].squeeze(1).to(device)
            with torch.no_grad():
                nq2 = Q2t(b['n_acts'].to(device), b['n_feats'].to(device), b['n_lens'].squeeze(1))
            return term * norm(r) + (1 - term) * args.gamma * nq2.max(1)[0]
        best1 = train_q(Q1, Q1t, optim.Adam(Q1.parameters(), args.dqn_lr, weight_decay=1e-5), tr0, va0, tgt1_2step, args)
        Q1.load_state_dict(best1)
        save_dict.update({'Q1': Q1.state_dict(), 'Q2': Q2.state_dict()})

    else:  # steps == 3
        Q1, Q1t = make_model(N_ACTIONS[0])
        Q2, Q2t = make_model(N_ACTIONS[1])
        Q3, Q3t = make_model(N_ACTIONS[2])
        tr0 = loader(df_train_causal, 0); va0 = loader(df_val_causal, 0, False)
        tr1 = loader(df_train_causal, 1); va1 = loader(df_val_causal, 1, False)
        tr2 = loader(df_train_causal, 2); va2 = loader(df_val_causal, 2, False)

        print("\n[Q3]")
        best3 = train_q(Q3, Q3t, optim.Adam(Q3.parameters(), args.dqn_lr, weight_decay=1e-5), tr2, va2,
                        lambda b: norm(b['reward'].squeeze(1).to(device)), args)
        Q3.load_state_dict(best3); Q3t.load_state_dict(best3)

        print("\n[Q2]")
        def tgt2(b):
            r, term = b['reward'].squeeze(1).to(device), b['terminal'].squeeze(1).to(device)
            with torch.no_grad():
                nq = Q3t(b['n_acts'].to(device), b['n_feats'].to(device), b['n_lens'].squeeze(1))
            return term * norm(r) + (1 - term) * args.gamma * nq.max(1)[0]
        Q3t.eval()
        best2 = train_q(Q2, Q2t, optim.Adam(Q2.parameters(), args.dqn_lr, weight_decay=1e-5), tr1, va1, tgt2, args)
        Q2.load_state_dict(best2); Q2t.load_state_dict(best2)

        print("\n[Q1]")
        def tgt1(b):
            r, term = b['reward'].squeeze(1).to(device), b['terminal'].squeeze(1).to(device)
            ni = b['next_intervention'].squeeze(1).to(device)
            t = term * norm(r)
            with torch.no_grad():
                nq2 = Q2t(b['n_acts'].to(device), b['n_feats'].to(device), b['n_lens'].squeeze(1))
                nq3 = Q3t(b['n_acts'].to(device), b['n_feats'].to(device), b['n_lens'].squeeze(1))
            m2 = ((1 - term).bool()) & (ni == 1)
            m3 = ((1 - term).bool()) & (ni == 2)
            if m2.any():
                t[m2] = args.gamma * nq2[m2].max(1)[0]
            if m3.any():
                t[m3] = args.gamma * nq3[m3].max(1)[0]
            m_other = ((1 - term).bool()) & (~(m2 | m3))
            if m_other.any():
                t[m_other] = args.gamma * torch.max(nq2[m_other].max(1)[0], nq3[m_other].max(1)[0])
            return t
        Q2t.eval()
        Q3t.eval()
        best1 = train_q(Q1, Q1t, optim.Adam(Q1.parameters(), args.dqn_lr, weight_decay=1e-5), tr0, va0, tgt1, args)
        Q1.load_state_dict(best1)
        save_dict.update({'Q1': Q1.state_dict(), 'Q2': Q2.state_dict(), 'Q3': Q3.state_dict()})

    os.makedirs("models", exist_ok=True)
    model_path = f"models/lstm_dqn_dragonnet_{suffix}_{args.n_cases}_s{args.seed}{step_tag}.pth"
    torch.save(save_dict, model_path)
    print(f"\n[OK] {model_path}")


if __name__ == "__main__":
    main()
