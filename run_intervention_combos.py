"""
Run all 7 non-empty subsets of {Int0, Int1, Int2} using existing steps=3 models.

Reuses the steps=3-trained Q1, Q2, Q3 networks for every intervention. For each
combination, non-active interventions fall back to bank_policy via SubsetPolicy.
No retraining is performed.

Methods with missing model files are skipped with a warning, so this script can
be run on any machine that has at least some of the steps=3 models trained.

Usage:
    python -u run_intervention_combos.py                              # all methods, RCT+CONF
    python -u run_intervention_combos.py --methods lstm rims          # subset
    python -u run_intervention_combos.py --no_confounded              # RCT only
"""
import sys
import os
import argparse
import json
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from shared import (
    load_pickle, bank_policy, evaluate_policy,
    LSTM_DQN, N_ACTIONS,
)
from shared.subset_policy import SubsetPolicy

SEEDS = [42, 123, 456, 789, 1024]

COMBINATIONS = [
    (frozenset({0}),       "Int0_only"),
    (frozenset({1}),       "Int1_only"),
    (frozenset({2}),       "Int2_only"),
    (frozenset({0, 1}),    "Int01"),
    (frozenset({0, 2}),    "Int02"),
    (frozenset({1, 2}),    "Int12"),
    (frozenset({0, 1, 2}), "Int012"),
]

# Most methods follow the same pattern: torch.load → cfg + Q1/Q2/Q3 state_dicts → wrap in policy class.
# (file_prefix, policy_module, policy_class, pass_activity_enc)
LSTM_DQN_METHODS = {
    'lstm':               ('lstm',                'lstm.evaluate',                     'LSTMPolicy',            True),
    'rims':               ('rims',                'rims.evaluate',                     'RIMSPolicy',            True),
    'multiModelCQL':      ('multi_cql',           'multiModelCQL.evaluate',            'CQLPolicy',             False),
    'lstm_dqn_dragonnet': ('lstm_dqn_dragonnet',  'lstm_dqn_dragonnet.evaluate',       'LSTMDQNCFRNetPolicy',   False),
    'lstm_dqn_tabpfn':    ('lstm_dqn_tabpfn',     'lstm_dqn_tabpfn.evaluate',          'LSTMDQNTabPFNPolicy',   False),
    'procause_econml':    ('procause_econml',     'procause.econml_slearner.evaluate', 'ProCauseEconMLPolicy',  False),
    'procause_lstm':      ('procause_lstm',       'procause.lstm_slearner.evaluate',   'ProCauseLSTMPolicy',    False),
}

ALL_METHODS = ['kmeans'] + list(LSTM_DQN_METHODS.keys()) + ['singleModelCQL']

FILE_PREFIX = {m: LSTM_DQN_METHODS[m][0] for m in LSTM_DQN_METHODS}
FILE_PREFIX['kmeans']         = 'kmeans'
FILE_PREFIX['singleModelCQL'] = 'single_cql'


def model_path(method, suffix, n_cases, seed):
    ext = 'pkl' if method == 'kmeans' else 'pth'
    return f"models/{FILE_PREFIX[method]}_{suffix}_{n_cases}_s{seed}.{ext}"


def models_exist(method, suffix, n_cases):
    return all(os.path.exists(model_path(method, suffix, n_cases, s)) for s in SEEDS)


def _load_lstm_dqn(method_key, suffix, n_cases, seed):
    import torch
    import importlib

    file_prefix, module_path, class_name, pass_activity_enc = LSTM_DQN_METHODS[method_key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_path(method_key, suffix, n_cases, seed),
                      map_location=device, weights_only=False)
    cfg = ckpt['config']

    def load_net(key, n_act):
        kwargs = {}
        if pass_activity_enc:
            kwargs['activity_enc'] = cfg.get('activity_enc', 'integer')
        m = LSTM_DQN(cfg['n_activities'], cfg['n_features'], n_act,
                     cfg['emb_dim'], cfg['hidden'], cfg['n_layers'], cfg['dropout'],
                     **kwargs).to(device)
        m.load_state_dict(ckpt[key])
        m.eval()
        return m

    models = {0: load_net('Q1', N_ACTIONS[0]),
              1: load_net('Q2', N_ACTIONS[1]),
              2: load_net('Q3', N_ACTIONS[2])}

    mod = importlib.import_module(module_path)
    policy_class = getattr(mod, class_name)
    return policy_class(models, cfg, steps=3)


def make_lstm_dqn_loader(method_key):
    return lambda suffix, n_cases, seed: _load_lstm_dqn(method_key, suffix, n_cases, seed)


def load_kmeans_policy(suffix, n_cases, seed):
    from kmeans.evaluate import KMeansPolicy
    ckpt = load_pickle(model_path('kmeans', suffix, n_cases, seed))
    return KMeansPolicy(ckpt['models'], ckpt['q_tables'], steps=3)


def load_single_cql_policy(suffix, n_cases, seed):
    """singleModelCQL has one shared LSTM_DQN with cfg['max_actions'] outputs, masked at eval."""
    import torch
    from singleModelCQL.evaluate import CQLPolicy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_path('singleModelCQL', suffix, n_cases, seed),
                      map_location=device, weights_only=False)
    cfg = ckpt['config']
    model = LSTM_DQN(cfg['n_activities'], cfg['n_features'], cfg['max_actions'],
                     cfg['emb_dim'], cfg['hidden'], cfg['n_layers'], cfg['dropout']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return CQLPolicy(model, cfg, steps=3)


METHOD_LOADERS = {
    'kmeans':         load_kmeans_policy,
    'singleModelCQL': load_single_cql_policy,
}
for _m in LSTM_DQN_METHODS:
    METHOD_LOADERS[_m] = make_lstm_dqn_loader(_m)


def aggregate_combo(per_seed_avgs):
    arr = np.array(per_seed_avgs, dtype=float)
    return {
        'mean':     float(arr.mean()),
        'std':      float(arr.std()),
        'per_seed': {str(s): float(v) for s, v in zip(SEEDS, per_seed_avgs)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods',       nargs='+', default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument('--n_cases',       type=int, default=10000)
    parser.add_argument('--n_episodes',    type=int, default=1000)
    parser.add_argument('--no_confounded', action='store_true')
    parser.add_argument('--results_out',   type=str, default='results/intervention_combos_results.json')
    args = parser.parse_args()

    os.chdir(script_dir)
    os.makedirs(os.path.dirname(args.results_out), exist_ok=True)

    suffixes = ['RCT'] if args.no_confounded else ['RCT', 'CONF']

    all_results = {}
    if os.path.exists(args.results_out):
        with open(args.results_out) as f:
            all_results = json.load(f)

    for suffix in suffixes:
        params = load_pickle(f"data/simbank_{suffix}_{args.n_cases}_params.pkl")

        bank_per_seed = []
        for seed in SEEDS:
            eval_seed = int(f"99{seed}")
            print(f"[{suffix} | seed={seed}] Bank baseline")
            res = evaluate_policy(bank_policy, args.n_episodes, params, eval_seed, verbose=False)
            bank_per_seed.append(res['avg'])
            print(f"  Bank avg = {res['avg']:.2f}")

        all_results[f"Bank_{suffix}"] = aggregate_combo(bank_per_seed)

        for method in args.methods:
            if not models_exist(method, suffix, args.n_cases):
                missing = [s for s in SEEDS if not os.path.exists(model_path(method, suffix, args.n_cases, s))]
                print(f"\n  [SKIP] {method} {suffix}: missing seeds {missing}")
                continue

            print(f"\n{'='*60}\n  {method.upper()} | {suffix}\n{'='*60}")
            method_results = {label: [] for _, label in COMBINATIONS}

            for seed in SEEDS:
                eval_seed = int(f"99{seed}")
                print(f"\n  seed={seed}")
                base_policy = METHOD_LOADERS[method](suffix, args.n_cases, seed)

                for active_ints, label in COMBINATIONS:
                    policy = SubsetPolicy(base_policy, active_ints)
                    res = evaluate_policy(policy, args.n_episodes, params, eval_seed,
                                          use_prefix=True, reset_fn=policy.reset, verbose=False)
                    method_results[label].append(res['avg'])
                    print(f"    {label:<12} avg={res['avg']:.2f}")

            for label, per_seed in method_results.items():
                all_results[f"{method}_{suffix}_{label}"] = aggregate_combo(per_seed)

            with open(args.results_out, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n  [saved] {args.results_out}")

    print(f"\n{'='*78}\n  SUMMARY\n{'='*78}")
    print(f"{'Method':<20} {'Cond':<6} {'Combination':<14} {'Mean':>10} {'Std':>8} {'vs Bank':>10}")
    print('-' * 78)
    for method in args.methods:
        for suffix in suffixes:
            bank_mean = all_results.get(f"Bank_{suffix}", {}).get('mean', 0)
            for _, label in COMBINATIONS:
                key = f"{method}_{suffix}_{label}"
                if key not in all_results:
                    continue
                agg = all_results[key]
                gain = ((agg['mean'] / bank_mean) - 1) * 100 if bank_mean > 0 else float('nan')
                print(f"{method:<20} {suffix:<6} {label:<14} {agg['mean']:>10.2f} {agg['std']:>8.2f} {gain:>9.1f}%")

    print(f"\n[OK] Saved to {args.results_out}")


if __name__ == "__main__":
    main()
