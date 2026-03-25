"""Train Single-Model CQL."""
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

from shared import load_pickle, STATE_DIM, seed_worker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SingleModelCQL(nn.Module):
    def __init__(self, state_dim=16, hidden_dims=[256, 256]):
        super().__init__()
        self.valid_actions = [2, 2, 3]
        layers = []
        input_dim = state_dim + 3
        for h in hidden_dims:
            layers.extend([nn.Linear(input_dim, h), nn.ReLU(), nn.LayerNorm(h)])
            input_dim = h
        layers.append(nn.Linear(input_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, state, int_id):
        onehot = torch.zeros(state.shape[0], 3, device=state.device)
        onehot.scatter_(1, int_id.view(-1, 1), 1)
        return self.net(torch.cat([state, onehot], dim=1))

    def masked_q(self, state, int_id):
        q = self.forward(state, int_id)
        for i in range(state.shape[0]):
            q[i, self.valid_actions[int_id[i].item()]:] = float('-inf')
        return q


class TransitionDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return {
            'state':        torch.FloatTensor(r['state']),
            'action':       torch.LongTensor([r['action']]),
            'reward':       torch.FloatTensor([r['reward']]),
            'next_state':   torch.FloatTensor(r['next_state']),
            'terminal':     torch.FloatTensor([r['terminal']]),
            'intervention': torch.LongTensor([r['intervention']]),
            'next_int':     torch.LongTensor([r['next_intervention']]),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cases',    type=int,   default=10000)
    parser.add_argument('--confounded', action='store_true')
    parser.add_argument('--steps',      type=int,   default=3, choices=[1, 2, 3])
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=256)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--alpha',      type=float, default=1.0)
    parser.add_argument('--gamma',      type=float, default=0.99)
    parser.add_argument('--tau',        type=float, default=0.005)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--patience',   type=int,   default=10)
    parser.add_argument('--es_delta',   type=float, default=1e-4)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    suffix   = "CONF" if args.confounded else "RCT"
    base     = f"data/single_cql_{suffix}_{args.n_cases}"
    step_tag = "" if args.steps == 3 else f"_steps{args.steps}"
    print(f"Training Single-Model CQL -- {suffix} | lr={args.lr} alpha={args.alpha} steps={args.steps}")

    df_train = load_pickle(f"{base}_trans_train{step_tag}.pkl")
    df_val   = load_pickle(f"{base}_trans_val{step_tag}.pkl")
    print(f"Train: {len(df_train)}, Val: {len(df_val)} transitions")

    scaler = StandardScaler()
    train_states = np.vstack(df_train['state'].values)
    scaler.fit(train_states)

    df_train = df_train.copy()
    df_val   = df_val.copy()
    df_train['state']      = list(scaler.transform(train_states))
    df_train['next_state'] = list(scaler.transform(np.vstack(df_train['next_state'].values)))
    df_val['state']        = list(scaler.transform(np.vstack(df_val['state'].values)))
    df_val['next_state']   = list(scaler.transform(np.vstack(df_val['next_state'].values)))

    model  = SingleModelCQL(STATE_DIM).to(device)
    target = SingleModelCQL(STATE_DIM).to(device)
    target.load_state_dict(model.state_dict())
    opt = optim.Adam(model.parameters(), lr=args.lr)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(TransitionDataset(df_train), batch_size=args.batch_size, shuffle=True,
                              worker_init_fn=seed_worker, generator=g)
    val_loader   = DataLoader(TransitionDataset(df_val),   batch_size=args.batch_size)

    best_val, best_state, patience_count = float('inf'), copy.deepcopy(model.state_dict()), 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            s      = batch['state'].to(device)
            a      = batch['action'].squeeze(1).to(device)
            r      = batch['reward'].squeeze(1).to(device)
            ns     = batch['next_state'].to(device)
            term   = batch['terminal'].squeeze(1).to(device)
            int_id = batch['intervention'].squeeze(1).to(device)
            ni     = batch['next_int'].squeeze(1).to(device)

            q   = model(s, int_id)
            q_a = q.gather(1, a.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                ni_safe = ni.clone()
                ni_safe[ni_safe < 0] = 0
                max_nq  = target.masked_q(ns, ni_safe).max(1)[0]
                targets = r + args.gamma * max_nq * (1 - term)

            loss = F.mse_loss(q_a, targets) + args.alpha * (torch.logsumexp(q, 1).mean() - q_a.mean())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            for p, tp in zip(model.parameters(), target.parameters()):
                tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                s      = batch['state'].to(device)
                a      = batch['action'].squeeze(1).to(device)
                r      = batch['reward'].squeeze(1).to(device)
                ns     = batch['next_state'].to(device)
                term   = batch['terminal'].squeeze(1).to(device)
                int_id = batch['intervention'].squeeze(1).to(device)
                ni     = batch['next_int'].squeeze(1).to(device)

                q   = model(s, int_id)
                q_a = q.gather(1, a.unsqueeze(1)).squeeze(1)
                ni_safe = ni.clone()
                ni_safe[ni_safe < 0] = 0
                max_nq  = target.masked_q(ns, ni_safe).max(1)[0]
                targets = r + args.gamma * max_nq * (1 - term)
                val_loss += F.mse_loss(q_a, targets).item()

        val_loss /= len(val_loader)
        if val_loss < best_val - args.es_delta:
            best_val, best_state = val_loss, copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1

        if (epoch + 1) % 10 == 0:
            print(f"  [{epoch+1:3d}/{args.epochs}] train={train_loss/len(train_loader):.4f}  val={val_loss:.4f}")
        if patience_count >= args.patience:
            print(f"  [early stop] epoch {epoch+1}, best_val={best_val:.4f}")
            break

    model.load_state_dict(best_state)
    os.makedirs("models", exist_ok=True)
    model_path = f"models/single_cql_{suffix}_{args.n_cases}_s{args.seed}{step_tag}.pth"
    torch.save({'model': model.state_dict(), 'scaler': scaler,
                'config': {'state_dim': STATE_DIM, 'steps': args.steps}}, model_path)
    print(f"\n[OK] {model_path}")


if __name__ == "__main__":
    main()
