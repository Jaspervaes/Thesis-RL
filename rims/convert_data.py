"""Train P_T (processing time) and P_C (control flow) simulator models from raw logs."""
import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
os.chdir(project_root)

from shared import load_pickle, save_pickle, split_train_val, FEATURE_COLS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INTERVENTION_ACTIVITIES = {
    'start_standard', 'start_priority',
    'contact_headquarters', 'skip_contact',
    'calculate_offer',
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ProcessingTimeModel(nn.Module):
    """LSTM predicting log(duration_seconds + 1) of next event given prefix."""

    def __init__(self, n_activities, n_features, emb_dim=32, hidden=64, n_layers=1):
        super().__init__()
        self.emb  = nn.Embedding(n_activities, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim + n_features, hidden, n_layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, acts, feats, lens):
        x = torch.cat([self.emb(acts), feats], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.head(h[-1]).squeeze(-1)


class ControlFlowModel(nn.Module):
    """LSTM classifier predicting next activity given prefix."""

    def __init__(self, n_activities, n_features, emb_dim=32, hidden=64, n_layers=1):
        super().__init__()
        self.emb  = nn.Embedding(n_activities, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim + n_features, hidden, n_layers, batch_first=True)
        self.head = nn.Linear(hidden, n_activities)

    def forward(self, acts, feats, lens):
        x = torch.cat([self.emb(acts), feats], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.head(h[-1])


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _safe_float(v):
    """Convert value to float, replacing NaN/None with 0."""
    try:
        f = float(v)
        return f if np.isfinite(f) else 0.0
    except (TypeError, ValueError):
        return 0.0


def build_vocab(df):
    """Build activity vocabulary from raw event log."""
    activities = sorted(df['activity'].unique().tolist())
    activity_to_idx = {a: i + 1 for i, a in enumerate(activities)}
    activity_to_idx[''] = 0
    idx_to_activity = {v: k for k, v in activity_to_idx.items()}
    return activity_to_idx, idx_to_activity


def compute_feat_stats(df):
    """Compute feature normalization stats from raw log."""
    feat_means = {c: float(pd.to_numeric(df[c], errors='coerce').fillna(0).mean()) for c in FEATURE_COLS}
    feat_stds  = {c: max(float(pd.to_numeric(df[c], errors='coerce').fillna(0).std()), 1e-8) for c in FEATURE_COLS}
    return feat_means, feat_stds


def prepare_sim_data(df, activity_to_idx):
    """Extract (prefix_acts, prefix_feats, next_activity_idx, log_duration) samples."""
    prefixes_acts, prefixes_feats, next_acts, durations = [], [], [], []

    for _, group in df.groupby('case_nr'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        events = group.to_dict('records')
        timestamps = group['timestamp'].tolist()

        for i in range(1, len(events)):
            p_acts = [activity_to_idx.get(e.get('activity', ''), 0) for e in events[:i]]
            p_feats = [[_safe_float(e.get(c, 0)) for c in FEATURE_COLS] for e in events[:i]]
            next_act = activity_to_idx.get(events[i].get('activity', ''), 0)
            dur = (timestamps[i] - timestamps[i - 1]).total_seconds()
            if not np.isfinite(dur) or dur < 0:
                continue
            log_dur = np.log(dur + 1)

            prefixes_acts.append(p_acts)
            prefixes_feats.append(p_feats)
            next_acts.append(next_act)
            durations.append(log_dur)

    return prefixes_acts, prefixes_feats, next_acts, durations


def mine_acceptance_model(df):
    """Mine customer acceptance logic from post-calculate_offer transitions.

    Fits a logistic regression on (interest_rate, min_interest_rate, amount,
    elapsed_time) -> accepted (1) vs refused/cancelled (0).
    Also mines the cancellation rule: interest_rate < min_interest_rate -> cancel.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    features, labels = [], []
    for _, group in df.groupby('case_nr'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        activities = group['activity'].tolist()
        for i, act in enumerate(activities):
            if act == 'calculate_offer' and i + 1 < len(activities):
                next_act = activities[i + 1]
                row = group.iloc[i]
                ir = _safe_float(row.get('interest_rate', 0))
                min_ir = _safe_float(row.get('min_interest_rate', 0))
                amount = _safe_float(row.get('amount', 0))
                elapsed = _safe_float(row.get('elapsed_time', 0))

                if next_act == 'receive_acceptance':
                    features.append([ir, min_ir, amount, elapsed])
                    labels.append(1)
                elif next_act in ('receive_refusal', 'cancel_application'):
                    features.append([ir, min_ir, amount, elapsed])
                    labels.append(0)

    if not features or len(set(labels)) < 2:
        print("  [warn] Not enough acceptance/refusal data for logistic regression")
        return None

    X = np.array(features)
    y = np.array(labels)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    acc = model.score(X_scaled, y)
    n_accept = y.sum()
    n_total = len(y)
    print(f"  Acceptance model: {n_accept}/{n_total} accepted ({n_accept/n_total*100:.1f}%), accuracy={acc:.3f}")

    return {
        'coef': model.coef_.tolist(),
        'intercept': model.intercept_.tolist(),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'feature_names': ['interest_rate', 'min_interest_rate', 'amount', 'elapsed_time'],
    }


def mine_transition_matrix(df, activity_to_idx):
    """Mine valid activity transitions from the event log."""
    valid_successors = {}
    for _, group in df.groupby('case_nr'):
        activities = group.sort_values('timestamp')['activity'].tolist()
        for i in range(len(activities) - 1):
            src = activities[i]
            dst = activities[i + 1]
            if src not in valid_successors:
                valid_successors[src] = set()
            valid_successors[src].add(dst)

    # Convert to idx-based mask: {src_idx: [valid_dst_idx, ...]}
    transition_mask = {}
    for src, dsts in valid_successors.items():
        src_idx = activity_to_idx.get(src, 0)
        transition_mask[src_idx] = [activity_to_idx.get(d, 0) for d in dsts]

    return valid_successors, transition_mask


def extract_initial_prefixes(df, steps):
    """Extract prefixes up to the first intervention point for env.reset()."""
    initial_prefixes = []
    for _, group in df.groupby('case_nr'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        events = group.to_dict('records')

        int0_rows = group[group['activity'].isin(['start_standard', 'start_priority'])]
        if int0_rows.empty or int0_rows.index[0] == 0:
            continue

        i0 = int0_rows.index[0]
        initial_prefixes.append(events[:i0])

    return initial_prefixes


class SimDataset(Dataset):
    def __init__(self, acts_list, feats_list, targets, max_len, feat_means, feat_stds,
                 target_dtype='float'):
        self.acts_list = acts_list
        self.feats_list = feats_list
        self.targets = targets
        self.max_len = max_len
        self.feat_means = feat_means
        self.feat_stds = feat_stds
        self.target_dtype = target_dtype

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        seq_len = min(len(self.acts_list[i]), self.max_len)
        seq_len = max(seq_len, 1)

        acts = np.zeros(self.max_len, dtype=np.int64)
        feats = np.zeros((self.max_len, len(FEATURE_COLS)), dtype=np.float32)

        for j in range(seq_len):
            acts[j] = self.acts_list[i][j]
            for k, col in enumerate(FEATURE_COLS):
                val = self.feats_list[i][j][k]
                feats[j, k] = (val - self.feat_means[col]) / self.feat_stds[col]
                if not np.isfinite(feats[j, k]):
                    feats[j, k] = 0.0

        if self.target_dtype == 'long':
            target = torch.LongTensor([self.targets[i]])
        else:
            target = torch.FloatTensor([self.targets[i]])

        return {
            'acts': torch.LongTensor(acts),
            'feats': torch.FloatTensor(feats),
            'lens': torch.LongTensor([seq_len]),
            'target': target,
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, loss_fn, epochs=50, lr=1e-3, patience=10):
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    best_val, best_state = float('inf'), None
    patience_count = 0

    for epoch in range(epochs):
        model.train()
        tl = 0.0
        for b in train_loader:
            pred = model(b['acts'].to(device), b['feats'].to(device), b['lens'].squeeze(1))
            target = b['target'].squeeze(1).to(device)
            loss = loss_fn(pred, target)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for b in val_loader:
                pred = model(b['acts'].to(device), b['feats'].to(device), b['lens'].squeeze(1))
                target = b['target'].squeeze(1).to(device)
                vl += loss_fn(pred, target).item()
        vl /= max(len(val_loader), 1)
        scheduler.step(vl)

        if vl < best_val - 1e-4:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if (epoch + 1) % 10 == 0:
            print(f"  [{epoch+1:3d}/{epochs}] train={tl/len(train_loader):.4f}  val={vl:.4f}")
        if patience_count >= patience:
            print(f"  [early stop] epoch {epoch+1}, best_val={best_val:.4f}")
            break

    return best_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cases',    type=int, default=10000)
    parser.add_argument('--confounded', action='store_true')
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--steps',      type=int, default=3, choices=[1, 2, 3])
    parser.add_argument('--epochs',     type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr',         type=float, default=1e-3)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    suffix = "CONF" if args.confounded else "RCT"
    base = f"data/rims_{suffix}_{args.n_cases}"
    step_tag = "" if args.steps == 3 else f"_steps{args.steps}"

    print(f"Training simulator models — {suffix} | steps={args.steps}")

    df = load_pickle(f"{base}_raw.pkl")
    df_train, df_val = split_train_val(df, val_ratio=0.2, seed=args.seed)

    # Build vocab and stats
    activity_to_idx, idx_to_activity = build_vocab(df)
    feat_means, feat_stds = compute_feat_stats(df)
    n_activities = len(activity_to_idx)

    # Prepare data
    print("Preparing training data...")
    tr_acts, tr_feats, tr_next, tr_dur = prepare_sim_data(df_train, activity_to_idx)
    va_acts, va_feats, va_next, va_dur = prepare_sim_data(df_val, activity_to_idx)

    max_len = max(max(len(a) for a in tr_acts), max(len(a) for a in va_acts))

    # P_T: Processing Time Model
    print(f"\n[P_T] Training processing time model ({len(tr_acts)} samples)")
    tr_ds_pt = SimDataset(tr_acts, tr_feats, tr_dur, max_len, feat_means, feat_stds)
    va_ds_pt = SimDataset(va_acts, va_feats, va_dur, max_len, feat_means, feat_stds)
    tr_loader_pt = DataLoader(tr_ds_pt, batch_size=args.batch_size, shuffle=True)
    va_loader_pt = DataLoader(va_ds_pt, batch_size=args.batch_size, shuffle=False)

    pt_model = ProcessingTimeModel(n_activities, len(FEATURE_COLS)).to(device)
    pt_state = train_model(pt_model, tr_loader_pt, va_loader_pt,
                           nn.MSELoss(), epochs=args.epochs, lr=args.lr)

    # P_C: Control Flow Model (trained on all transitions; only called at non-intervention points)
    print(f"\n[P_C] Training control flow model")
    tr_ds_pc = SimDataset(tr_acts, tr_feats, tr_next, max_len, feat_means, feat_stds, target_dtype='long')
    va_ds_pc = SimDataset(va_acts, va_feats, va_next, max_len, feat_means, feat_stds, target_dtype='long')
    tr_loader_pc = DataLoader(tr_ds_pc, batch_size=args.batch_size, shuffle=True)
    va_loader_pc = DataLoader(va_ds_pc, batch_size=args.batch_size, shuffle=False)

    pc_model = ControlFlowModel(n_activities, len(FEATURE_COLS)).to(device)
    pc_state = train_model(pc_model, tr_loader_pc, va_loader_pc,
                           nn.CrossEntropyLoss(), epochs=args.epochs, lr=args.lr)

    # Mine acceptance model from logs
    print("\nMining acceptance model...")
    acceptance_model = mine_acceptance_model(df)

    # Mine transition matrix from logs
    valid_successors, transition_mask = mine_transition_matrix(df, activity_to_idx)
    print(f"\nMined transition matrix: {len(valid_successors)} source activities")
    for src, dsts in sorted(valid_successors.items()):
        print(f"  {src} -> {sorted(dsts)}")

    # Extract initial prefixes for reset()
    initial_prefixes = extract_initial_prefixes(df_train, args.steps)
    print(f"\nExtracted {len(initial_prefixes)} initial prefixes")

    # Save simulator artifact
    os.makedirs("data", exist_ok=True)
    sim_path = f"{base}_simulator{step_tag}.pkl"
    save_pickle({
        'pt_state_dict': pt_state,
        'pc_state_dict': pc_state,
        'activity_to_idx': activity_to_idx,
        'idx_to_activity': idx_to_activity,
        'feat_means': feat_means,
        'feat_stds': feat_stds,
        'max_len': max_len,
        'n_activities': n_activities,
        'initial_prefixes': initial_prefixes,
        'transition_mask': transition_mask,
        'acceptance_model': acceptance_model,
        'steps': args.steps,
    }, sim_path)
    print(f"[OK] {sim_path}")


if __name__ == "__main__":
    main()
