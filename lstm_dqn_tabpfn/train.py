"""Train LSTM-DQN-TabPFN: TabPFN causal S-learner (causal rewards) + LSTM-DQN (backward TD)."""
import sys
import os
import argparse
import copy
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tabpfn import TabPFNRegressor
from sklearn.preprocessing import StandardScaler

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
# Phase 1: TabPFN S-learner  (state, action) -> outcome
# ---------------------------------------------------------------------------

def train_tabpfn_slearner(states, actions, outcomes, n_actions,
                          seed, max_samples):
    """Train S-learner via TabPFNRegressor. Returns model, scaler, outcome stats."""
    state_scaler = StandardScaler()
    states_norm = state_scaler.fit_transform(states)
    outcome_mean, outcome_std = outcomes.mean(), outcomes.std() + 1e-8
    outcomes_norm = (outcomes - outcome_mean) / outcome_std

    X = np.column_stack([states_norm, actions.reshape(-1, 1)])
    y = outcomes_norm

    if max_samples is not None and X.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        print(f"  [tabpfn] subsampling {X.shape[0]} -> {max_samples} (seed={seed})")
        X, y = X[idx], y[idx]

    model = TabPFNRegressor(device='cuda' if torch.cuda.is_available() else 'cpu',
                            random_state=seed)
    model.fit(X, y)
    return model, state_scaler, outcome_mean, outcome_std


def predict_with_tabpfn(tabpfn_model, state_scaler, states, actions, outcome_mean, outcome_std):
    """Predict denormalized outcomes for given (state, action) pairs."""
    states_norm = state_scaler.transform(states)
    X = np.column_stack([states_norm, actions.reshape(-1, 1)])
    preds_norm = tabpfn_model.predict(X)
    return preds_norm * outcome_std + outcome_mean


# ---------------------------------------------------------------------------
# Phase 3: LSTM-DQN with backward TD
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
    parser.add_argument('--tabpfn_max_samples', type=int, default=10000)
    parser.add_argument('--patience',       type=int,   default=15)
    parser.add_argument('--es_delta',       type=float, default=1e-4)
    parser.add_argument('--steps',          type=int,   default=3, choices=[1, 2, 3])
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
    base   = f"data/lstm_dqn_tabpfn_{suffix}_{args.n_cases}"
    step_tag = "" if args.steps == 3 else f"_steps{args.steps}"
    print(f"Training LSTM-DQN-TabPFN — {suffix} | steps={args.steps}")

    df_train = load_pickle(f"{base}_trans_train{step_tag}.pkl")
    df_val   = load_pickle(f"{base}_trans_val{step_tag}.pkl")
    print(f"Train: {len(df_train)}, Val: {len(df_val)} transitions")

    # Build vocab/stats from prefixes for DQN encoding
    activity_to_idx, feat_means, feat_stds = build_vocab_and_stats(df_train)
    n_activities = max(activity_to_idx.values(), default=0) + 1

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
    tabpfn_models = {}  # {int_idx: (model, scaler, outcome_mean, outcome_std)}

    # ===================================================================
    # Phase 1: Train TabPFN S-learner per intervention
    # ===================================================================
    print(f"\n{'='*50}")
    print("Phase 1: Training TabPFN S-learners")
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

        print(f"\n[TabPFN S-learner Int.{int_idx}]")
        tr_states = np.stack(sub_train['state'].tolist())
        tr_actions = np.array(sub_train['action'].tolist())
        tr_outcomes = np.array(sub_train['case_outcome'].tolist(), dtype=np.float64)

        tabpfn_model, state_scaler, outcome_mean, outcome_std = train_tabpfn_slearner(
            tr_states, tr_actions, tr_outcomes, n_act,
            seed=args.seed + int_idx,
            max_samples=args.tabpfn_max_samples,
        )

        # Validate
        va_states = np.stack(sub_val['state'].tolist())
        va_states_norm = state_scaler.transform(va_states)
        val_preds = []
        for a in range(n_act):
            X_val = np.column_stack([va_states_norm, np.full(len(va_states), a)])
            val_preds.append(tabpfn_model.predict(X_val) * outcome_std + outcome_mean)
        val_preds = np.stack(val_preds, axis=1)
        print(f"  Val pred means per action: {np.mean(val_preds, axis=0)}")

        tabpfn_models[int_idx] = (tabpfn_model, state_scaler, outcome_mean, outcome_std)
        save_dict[f'tabpfn_{int_idx}']       = pickle.dumps(tabpfn_model)
        save_dict[f'scaler_{int_idx}']       = pickle.dumps(state_scaler)
        save_dict[f'outcome_mean_{int_idx}'] = outcome_mean
        save_dict[f'outcome_std_{int_idx}']  = outcome_std
        save_dict[f'n_actions_{int_idx}']    = n_act

    # ===================================================================
    # Phase 2: Counterfactual-augmented causal rewards (TabPFN)
    # -------------------------------------------------------------------
    # COUNTERFACTUAL AUGMENTATION CHANGE (2026-04-21)
    # Previously, this phase only replaced the observed reward at terminal
    # transitions with model.predict([state | action_taken]). Now, in
    # addition to that factual replacement, we synthesize one extra
    # transition per alternative action a in range(N_ACTIONS[int_idx]) —
    # identical to the original row except action=a and
    # reward=model.predict([state | a]).
    #
    # TabPFN takes the concatenated [state | action] as a flat feature
    # vector (see predict_with_tabpfn), so querying a counterfactual
    # action is literally swapping the action column in that input. The
    # Phase-1 fit/subsampling logic is left untouched.
    #
    # Non-terminal transitions are NOT augmented.
    # Expected buffer size per intervention ~= #terminal * N_ACTIONS[int_idx]
    #                                          + #non-terminal (unchanged).
    #
    # To revert to the previous factual-only behaviour, see the commented
    # block tagged "ORIGINAL PHASE 2 (factual replacement)" below.
    # Phase 1 and Phase 3 are intentionally unchanged.
    # ===================================================================
    print(f"\n{'='*50}")
    print("Phase 2: Counterfactual augmentation via TabPFN S-learners")
    print('='*50)

    df_train_causal = df_train.copy()
    df_val_causal   = df_val.copy()
    train_synth_chunks = []
    val_synth_chunks   = []

    for int_idx, (tabpfn_model, state_scaler, outcome_mean, outcome_std) in tabpfn_models.items():
        n_act_i = N_ACTIONS[int_idx]
        for df_c, synth_list, label in [
            (df_train_causal, train_synth_chunks, 'train'),
            (df_val_causal,   val_synth_chunks,   'val'),
        ]:
            terminal_mask = (df_c['intervention'] == int_idx) & (df_c['terminal'] == True)
            if not terminal_mask.any():
                continue

            sub = df_c[terminal_mask]
            states = np.stack(sub['state'].tolist())
            factual_actions = np.array(sub['action'].tolist())

            # (1) Factual leg — same as before.
            causal_rewards = predict_with_tabpfn(tabpfn_model, state_scaler, states, factual_actions,
                                                 outcome_mean, outcome_std)
            df_c.loc[terminal_mask, 'reward'] = causal_rewards

            orig_mean   = sub['reward'].mean()
            causal_mean = float(np.mean(causal_rewards))

            # (2) Counterfactual leg — swap the action column in the query.
            n_synth_total = 0
            for a in range(n_act_i):
                alt_mask = factual_actions != a
                if not alt_mask.any():
                    continue
                a_states  = states[alt_mask]
                a_actions = np.full(int(alt_mask.sum()), a, dtype=factual_actions.dtype)
                a_rewards = predict_with_tabpfn(tabpfn_model, state_scaler, a_states, a_actions,
                                                outcome_mean, outcome_std)
                synth = sub[alt_mask].copy()
                synth['action'] = a
                synth['reward'] = a_rewards
                synth_list.append(synth)
                n_synth_total += len(synth)

            print(f"  Int.{int_idx} {label}: {int(terminal_mask.sum())} terminal, "
                  f"obs_reward={orig_mean:.1f}, factual_causal={causal_mean:.1f}, "
                  f"synthetic_added={n_synth_total}")

    if train_synth_chunks:
        df_train_causal = pd.concat([df_train_causal] + train_synth_chunks, ignore_index=True)
    if val_synth_chunks:
        df_val_causal = pd.concat([df_val_causal] + val_synth_chunks, ignore_index=True)
    print(f"  Post-augmentation sizes: train={len(df_train_causal)}, val={len(df_val_causal)}")

    # -------------------------------------------------------------------
    # ORIGINAL PHASE 2 (factual replacement) — kept verbatim for rollback.
    # Uncomment this block and remove the counterfactual-augmentation block
    # above to restore the previous behaviour.
    # -------------------------------------------------------------------
    # df_train_causal = df_train.copy()
    # df_val_causal   = df_val.copy()
    # for int_idx, (tabpfn_model, state_scaler, outcome_mean, outcome_std) in tabpfn_models.items():
    #     for df_c, label in [(df_train_causal, 'train'), (df_val_causal, 'val')]:
    #         terminal_mask = (df_c['intervention'] == int_idx) & (df_c['terminal'] == True)
    #         if not terminal_mask.any():
    #             continue
    #         sub = df_c[terminal_mask]
    #         states = np.stack(sub['state'].tolist())
    #         actions = np.array(sub['action'].tolist())
    #         causal_rewards = predict_with_tabpfn(tabpfn_model, state_scaler, states, actions,
    #                                              outcome_mean, outcome_std)
    #         orig_mean = sub['reward'].mean()
    #         causal_mean = float(np.mean(causal_rewards))
    #         df_c.loc[terminal_mask, 'reward'] = causal_rewards
    #         print(f"  Int.{int_idx} {label}: {terminal_mask.sum()} terminal transitions, "
    #               f"observed_reward={orig_mean:.1f}, causal_reward={causal_mean:.1f}")

    # ===================================================================
    # Phase 3: Train LSTM-DQN with causal rewards (backward TD)
    # ===================================================================
    print(f"\n{'='*50}")
    print("Phase 3: Training LSTM-DQN with causal rewards")
    print('='*50)

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
    model_path = f"models/lstm_dqn_tabpfn_{suffix}_{args.n_cases}_s{args.seed}{step_tag}.pth"
    torch.save(save_dict, model_path)
    print(f"\n[OK] {model_path}")


if __name__ == "__main__":
    main()
