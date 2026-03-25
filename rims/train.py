"""Train RIMS-DQN: online LSTM-DQN in learned simulator with epsilon-greedy exploration."""
import sys
import os
import argparse
import copy
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
os.chdir(project_root)

from shared import load_pickle, FEATURE_COLS, N_ACTIONS, LSTM_DQN, build_vocab_and_stats, encode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Simple replay buffer storing prefix-based transitions."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, prefix, action, reward, next_prefix, done):
        self.buffer.append((prefix, action, reward, next_prefix, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        prefixes, actions, rewards, next_prefixes, dones = zip(*batch)
        return list(prefixes), list(actions), list(rewards), list(next_prefixes), list(dones)

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def encode_batch(prefixes, activity_to_idx, feat_means, feat_stds, max_len):
    """Encode and move to device, replacing NaN with 0."""
    acts, feats, lens = encode(prefixes, activity_to_idx, feat_means, feat_stds, max_len)
    feats_t = torch.FloatTensor(feats)
    feats_t = torch.nan_to_num(feats_t, nan=0.0, posinf=0.0, neginf=0.0)
    return (torch.LongTensor(acts).to(device),
            feats_t.to(device),
            torch.LongTensor(lens))


def update_q(q_net, q_target, optimizer, replay, batch_size,
             activity_to_idx, feat_means, feat_stds, max_len,
             gamma, tau, next_q_target=None):
    """One gradient step on a Q-network from its replay buffer."""
    if len(replay) < batch_size:
        return

    prefixes, actions, rewards, next_prefixes, dones = replay.sample(batch_size)

    s_acts, s_feats, s_lens = encode_batch(prefixes, activity_to_idx, feat_means, feat_stds, max_len)
    actions_t = torch.LongTensor(actions).to(device)
    rewards_t = torch.FloatTensor(rewards).to(device)
    dones_t = torch.FloatTensor(dones).to(device)

    # Current Q-values
    q_vals = q_net(s_acts, s_feats, s_lens)
    q_taken = q_vals.gather(1, actions_t.unsqueeze(1)).squeeze(1)

    # Target computation (rewards clipped for stability, no normalization)
    with torch.no_grad():
        clipped_r = rewards_t.clamp(-5000, 10000) / 1000.0
        if next_q_target is not None:
            # Non-terminal: bootstrap from next intervention's target network
            n_acts, n_feats, n_lens = encode_batch(next_prefixes, activity_to_idx, feat_means, feat_stds, max_len)
            next_q = next_q_target(n_acts, n_feats, n_lens).max(1)[0]
            target = clipped_r + (1 - dones_t) * gamma * next_q
        else:
            # Terminal intervention: just scaled reward
            target = clipped_r

    loss = F.mse_loss(q_taken, target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
    optimizer.step()

    # Soft target update
    for p, tp in zip(q_net.parameters(), q_target.parameters()):
        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cases',     type=int,   default=10000)
    parser.add_argument('--confounded',  action='store_true')
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--steps',       type=int,   default=3, choices=[1, 2, 3])
    parser.add_argument('--n_episodes',  type=int,   default=20000)
    parser.add_argument('--batch_size',  type=int,   default=128)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--gamma',       type=float, default=0.99)
    parser.add_argument('--tau',         type=float, default=0.005)
    parser.add_argument('--eps_start',   type=float, default=1.0)
    parser.add_argument('--eps_end',     type=float, default=0.05)
    parser.add_argument('--eps_decay',   type=float, default=0.00005)
    parser.add_argument('--buffer_size', type=int,   default=50000)
    parser.add_argument('--emb_dim',     type=int,   default=32)
    parser.add_argument('--hidden',      type=int,   default=128)
    parser.add_argument('--n_layers',    type=int,   default=2)
    parser.add_argument('--dropout',     type=float, default=0.2)
    parser.add_argument('--eval_every',  type=int,   default=500)
    parser.add_argument('--eval_episodes', type=int, default=200)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    suffix = "CONF" if args.confounded else "RCT"
    base = f"data/rims_{suffix}_{args.n_cases}"
    step_tag = "" if args.steps == 3 else f"_steps{args.steps}"

    print(f"Training RIMS-DQN — {suffix} | episodes={args.n_episodes} steps={args.steps}")

    # Load simulator
    sim_artifact = load_pickle(f"{base}_simulator{step_tag}.pkl")

    from rims.simulator import LearnedSimBankEnv
    env = LearnedSimBankEnv(sim_artifact, steps=args.steps)

    activity_to_idx = sim_artifact['activity_to_idx']
    feat_means = sim_artifact['feat_means']
    feat_stds = sim_artifact['feat_stds']
    max_len = sim_artifact['max_len']
    n_activities = sim_artifact['n_activities']
    n_features = len(FEATURE_COLS)

    # Create Q-networks per intervention
    def make_model(n_act):
        m  = LSTM_DQN(n_activities, n_features, n_act, args.emb_dim, args.hidden, args.n_layers, args.dropout).to(device)
        mt = LSTM_DQN(n_activities, n_features, n_act, args.emb_dim, args.hidden, args.n_layers, args.dropout).to(device)
        mt.load_state_dict(m.state_dict())
        return m, mt

    q_nets, q_targets, optimizers, replays = {}, {}, {}, {}
    for i in range(args.steps):
        q_nets[i], q_targets[i] = make_model(N_ACTIONS[i])
        q_targets[i].eval()  # Disable dropout for target computation
        optimizers[i] = optim.Adam(q_nets[i].parameters(), lr=args.lr, weight_decay=1e-5)
        replays[i] = ReplayBuffer(args.buffer_size)

    # Best validation tracking
    best_val_reward = -float('inf')
    best_states = None

    # Training loop
    episode_rewards = []
    for episode in range(args.n_episodes):
        epsilon = max(args.eps_end, args.eps_start - episode * args.eps_decay)
        prefix, info = env.reset()
        done = False
        step_count = 0

        while not done and step_count < 10:
            int_idx = info['int_idx']

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, N_ACTIONS[int_idx] - 1)
            else:
                with torch.no_grad():
                    s_acts, s_feats, s_lens = encode_batch(
                        [prefix], activity_to_idx, feat_means, feat_stds, max_len)
                    q = q_nets[int_idx](s_acts, s_feats, s_lens)
                action = q[0, :N_ACTIONS[int_idx]].argmax().item()

            prev_prefix = copy.deepcopy(prefix)
            next_prefix, reward, done, truncated, info = env.step(action)

            # Store transition
            if int_idx < args.steps:
                replays[int_idx].push(
                    prev_prefix, action, reward,
                    copy.deepcopy(next_prefix),
                    float(done)
                )

            prefix = next_prefix
            step_count += 1

        episode_rewards.append(reward)

        # Train each Q-net
        for i in range(args.steps):
            if len(replays[i]) >= args.batch_size:
                # Determine next Q-target for bootstrapping
                if i == args.steps - 1:
                    next_qt = None  # terminal intervention
                else:
                    next_qt = q_targets[i + 1]

                update_q(q_nets[i], q_targets[i], optimizers[i], replays[i],
                         args.batch_size, activity_to_idx, feat_means, feat_stds,
                         max_len, args.gamma, args.tau, next_qt)

        # Logging
        if (episode + 1) % 100 == 0:
            recent = episode_rewards[-100:]
            print(f"  [{episode+1:5d}/{args.n_episodes}] eps={epsilon:.3f}  "
                  f"avg100={np.mean(recent):.1f}")

        # Validation
        if (episode + 1) % args.eval_every == 0:
            # Put Q-nets in eval mode to disable dropout during validation
            for i in range(args.steps):
                q_nets[i].eval()
            val_rewards = []
            for _ in range(args.eval_episodes):
                p, info = env.reset()
                d = False
                sc = 0
                while not d and sc < 10:
                    idx = info['int_idx']
                    with torch.no_grad():
                        sa, sf, sl = encode_batch(
                            [p], activity_to_idx, feat_means, feat_stds, max_len)
                        q = q_nets[idx](sa, sf, sl)
                    a = q[0, :N_ACTIONS[idx]].argmax().item()
                    p, r, d, _, info = env.step(a)
                    sc += 1
                val_rewards.append(r)
            # Restore Q-nets to train mode for continued training
            for i in range(args.steps):
                q_nets[i].train()

            val_mean = np.mean(val_rewards)
            print(f"  [EVAL] episode {episode+1}: val_mean={val_mean:.1f}")

            if val_mean > best_val_reward:
                best_val_reward = val_mean
                best_states = {i: copy.deepcopy(q_nets[i].state_dict()) for i in range(args.steps)}
                print(f"  [BEST] new best val={val_mean:.1f}")

    # Load best states
    if best_states is not None:
        for i in range(args.steps):
            q_nets[i].load_state_dict(best_states[i])

    # Save model
    cfg = {
        'n_activities':    n_activities,
        'n_features':      n_features,
        'feature_cols':    FEATURE_COLS,
        'activity_to_idx': activity_to_idx,
        'feat_means':      feat_means,
        'feat_stds':       feat_stds,
        'max_len':         max_len,
        'emb_dim':         args.emb_dim,
        'hidden':          args.hidden,
        'n_layers':        args.n_layers,
        'dropout':         args.dropout,
        'n_actions':       N_ACTIONS,
        'steps':           args.steps,
    }

    save_dict = {'config': cfg}
    for i in range(args.steps):
        save_dict[f'Q{i+1}'] = q_nets[i].state_dict()

    os.makedirs("models", exist_ok=True)
    model_path = f"models/rims_{suffix}_{args.n_cases}_s{args.seed}{step_tag}.pth"
    torch.save(save_dict, model_path)
    print(f"\n[OK] {model_path}")


if __name__ == "__main__":
    main()
