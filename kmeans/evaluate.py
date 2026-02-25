"""Evaluate K-means offline RL against bank and random baselines."""
import sys
import os
import argparse
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
os.chdir(project_root)

from shared import (
    load_pickle, bank_policy, random_policy, evaluate_policy,
    print_results, print_action_dist, BASE_FEATURES, TRACKED_ACTIVITIES,
)


class KMeansPolicy:
    """K-means offline RL policy."""

    def __init__(self, models, q_tables):
        self.models   = models    # {int_idx: (kmeans, scaler)}
        self.q_tables = q_tables  # {int_idx: np.array (n_clusters, n_actions)}
        self.counts   = {a: 0 for a in TRACKED_ACTIVITIES}

    def reset(self):
        self.counts = {a: 0 for a in TRACKED_ACTIVITIES}

    def __call__(self, prev_event, int_idx, prefix=None):
        if prefix:
            self.counts = {a: 0 for a in TRACKED_ACTIVITIES}
            for e in prefix:
                act = e.get('activity', '').lower()
                for t in TRACKED_ACTIVITIES:
                    if t in act:
                        self.counts[t] += 1

        state = [float(prev_event.get(f, 0)) for f in BASE_FEATURES]
        state += [float(self.counts.get(a, 0)) for a in TRACKED_ACTIVITIES]
        state = np.array(state, dtype=np.float32).reshape(1, -1)

        km, sc = self.models[int_idx]
        q = self.q_tables[int_idx]
        cluster = km.predict(sc.transform(state))[0]
        return int(q[cluster].argmax())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cases',    type=int, default=10000)
    parser.add_argument('--confounded', action='store_true')
    parser.add_argument('--n_episodes',  type=int, default=1000)
    parser.add_argument('--seed',        type=int, default=1042)
    parser.add_argument('--train_seed',  type=int, default=42)
    parser.add_argument('--results_file', type=str, default=None)
    args = parser.parse_args()

    suffix = "CONF" if args.confounded else "RCT"
    ckpt   = load_pickle(f"models/kmeans_{suffix}_{args.n_cases}_s{args.train_seed}.pkl")
    params = load_pickle(f"data/kmeans_{suffix}_{args.n_cases}_params.pkl")

    policy = KMeansPolicy(ckpt['models'], ckpt['q_tables'])

    print(f"Evaluating K-means RL — {suffix}")
    bank_res   = evaluate_policy(bank_policy,   args.n_episodes, params, args.seed)
    random_res = evaluate_policy(random_policy, args.n_episodes, params, args.seed)
    km_res     = evaluate_policy(policy, args.n_episodes, params, args.seed,
                                 use_prefix=True, reset_fn=policy.reset)

    results = {'Bank': bank_res, 'Random': random_res, f'KMeans {suffix}': km_res}
    print_results(results)
    print_action_dist(results)

    gain = ((km_res['avg'] / bank_res['avg']) - 1) * 100
    print(f"\nK-means {'beats' if gain > 0 else 'underperforms'} Bank by {gain:+.1f}%")

    if args.results_file:
        import json
        os.makedirs(os.path.dirname(os.path.abspath(args.results_file)), exist_ok=True)
        with open(args.results_file, 'w') as f:
            json.dump({'Bank': bank_res['avg'], f'KMeans {suffix}': km_res['avg'], 'Random': random_res['avg']}, f)


if __name__ == "__main__":
    main()
