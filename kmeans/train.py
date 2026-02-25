"""Train K-means offline RL: cluster states per intervention, fitted Q-iteration."""
import sys
import os
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
os.chdir(project_root)

from shared import load_pickle, save_pickle

N_ACTIONS = [2, 2, 3]


def fit_clusters(df_train, int_idx, k, seed):
    """Fit scaler + K-means on training states for one intervention."""
    sub = df_train[df_train['intervention'] == int_idx]
    states = np.vstack(sub['state'].values)
    scaler = StandardScaler().fit(states)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(scaler.transform(states))
    return km, scaler


def fitted_q(df, int_idx, km, scaler, n_actions, gamma, next_models):
    """
    Fitted Q-iteration (one backward pass).
    Returns Q-table of shape (n_clusters, n_actions) with mean Bellman targets.
    """
    sub = df[df['intervention'] == int_idx]
    if sub.empty:
        return None

    n_clusters = km.n_clusters
    q = np.zeros((n_clusters, n_actions))
    counts = np.zeros((n_clusters, n_actions))

    states = np.vstack(sub['state'].values)
    clusters = km.predict(scaler.transform(states))

    for idx, row in enumerate(sub.itertuples(index=False)):
        c, a = clusters[idx], row.action

        if row.terminal:
            target = row.reward
        else:
            ns = np.array(row.next_state, dtype=np.float32).reshape(1, -1)
            ni = row.next_intervention
            next_km, next_sc, next_q = next_models[ni]
            nc = next_km.predict(next_sc.transform(ns))[0]
            target = gamma * next_q[nc].max()

        q[c, a] += target
        counts[c, a] += 1

    mask = counts > 0
    q[mask] /= counts[mask]
    return q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cases',    type=int,   default=10000)
    parser.add_argument('--confounded', action='store_true')
    parser.add_argument('--k',          type=int,   default=50)
    parser.add_argument('--gamma',      type=float, default=0.99)
    parser.add_argument('--seed',       type=int,   default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    suffix = "CONF" if args.confounded else "RCT"
    base = f"data/kmeans_{suffix}_{args.n_cases}"
    print(f"Training K-means RL — {suffix} | k={args.k} γ={args.gamma}")

    df_train = load_pickle(f"{base}_trans_train.pkl")

    km0, sc0 = fit_clusters(df_train, 0, args.k, args.seed)
    km1, sc1 = fit_clusters(df_train, 1, args.k, args.seed)
    km2, sc2 = fit_clusters(df_train, 2, args.k, args.seed)
    print(f"  Clusters fit: int0={args.k}, int1={args.k}, int2={args.k}")

    print("\n[Q3]")
    q3 = fitted_q(df_train, 2, km2, sc2, N_ACTIONS[2], args.gamma, {})

    print("[Q2]")
    q2 = fitted_q(df_train, 1, km1, sc1, N_ACTIONS[1], args.gamma, {2: (km2, sc2, q3)})

    print("[Q1]")
    q1 = fitted_q(df_train, 0, km0, sc0, N_ACTIONS[0], args.gamma,
                  {1: (km1, sc1, q2), 2: (km2, sc2, q3)})

    os.makedirs("models", exist_ok=True)
    model_path = f"models/kmeans_{suffix}_{args.n_cases}_s{args.seed}.pkl"
    save_pickle({
        'models':   {0: (km0, sc0), 1: (km1, sc1), 2: (km2, sc2)},
        'q_tables': {0: q1, 1: q2, 2: q3},
        'config':   {'k': args.k, 'n_actions': N_ACTIONS, 'gamma': args.gamma},
    }, model_path)
    print(f"\n[OK] {model_path}")


if __name__ == "__main__":
    main()
