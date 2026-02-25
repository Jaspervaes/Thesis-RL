"""Run train + evaluate for all 5 seeds and report aggregated results (mean ± std)."""
import sys
import os
import argparse
import subprocess
import json
import tempfile
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

SEEDS = [42, 123, 456, 789, 1024]

METHOD_SCRIPTS = {
    'kmeans':         ('kmeans/train.py',        'kmeans/evaluate.py'),
    'lstm':           ('lstm/train.py',           'lstm/evaluate.py'),
    'rims':           ('rims/train.py',           'rims/evaluate.py'),
    'singleModelCQL': ('singleModelCQL/train.py', 'singleModelCQL/evaluate.py'),
    'multiModelCQL':  ('multiModelCQL/train.py',  'multiModelCQL/evaluate.py'),
}


def run_seed(method, seed, n_cases, confounded, n_episodes, results_file, extra_train_args):
    suffix_flag = ['--confounded'] if confounded else []

    # Train
    train_cmd = [
        sys.executable, METHOD_SCRIPTS[method][0],
        '--seed', str(seed),
        '--n_cases', str(n_cases),
    ] + suffix_flag + extra_train_args
    print(f"  [train] seed={seed} ...", flush=True)
    subprocess.run(train_cmd, check=True, cwd=script_dir)

    # Evaluate
    eval_cmd = [
        sys.executable, METHOD_SCRIPTS[method][1],
        '--train_seed', str(seed),
        '--n_cases', str(n_cases),
        '--n_episodes', str(n_episodes),
        '--results_file', results_file,
    ] + suffix_flag
    print(f"  [eval]  seed={seed} ...", flush=True)
    subprocess.run(eval_cmd, check=True, cwd=script_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=list(METHOD_SCRIPTS.keys()))
    parser.add_argument('--n_cases',    type=int, default=10000)
    parser.add_argument('--confounded', action='store_true')
    parser.add_argument('--n_episodes', type=int, default=1000)
    args, extra = parser.parse_known_args()

    suffix = "CONF" if args.confounded else "RCT"
    print(f"\n{'='*60}")
    print(f"Running {args.method} — {suffix} | n_cases={args.n_cases} | seeds={SEEDS}")
    print('='*60)

    tmpdir = tempfile.mkdtemp()
    seed_results = []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        results_file = os.path.join(tmpdir, f"results_s{seed}.json")
        run_seed(args.method, seed, args.n_cases, args.confounded,
                 args.n_episodes, results_file, extra)
        with open(results_file) as f:
            seed_results.append(json.load(f))

    # Aggregate
    policy_key = [k for k in seed_results[0] if k not in ('Bank', 'Random')][0]
    bank_avgs   = [r['Bank']       for r in seed_results]
    method_avgs = [r[policy_key]   for r in seed_results]
    random_avgs = [r['Random']     for r in seed_results]

    print(f"\n{'='*60}")
    print(f"AGGREGATED RESULTS — {args.method} {suffix} n={args.n_cases} ({len(SEEDS)} seeds)")
    print('='*60)
    print(f"{'Policy':<22} {'Mean':>10} {'Std':>8}")
    print('-'*42)
    print(f"{'Bank':<22} {np.mean(bank_avgs):>10.2f} {np.std(bank_avgs):>8.2f}")
    print(f"{'Random':<22} {np.mean(random_avgs):>10.2f} {np.std(random_avgs):>8.2f}")
    print(f"{policy_key:<22} {np.mean(method_avgs):>10.2f} {np.std(method_avgs):>8.2f}")

    gain = ((np.mean(method_avgs) / np.mean(bank_avgs)) - 1) * 100
    print(f"\n{args.method} {'beats' if gain > 0 else 'underperforms'} Bank by {gain:+.1f}% (averaged over {len(SEEDS)} seeds)")

    # Per-seed breakdown
    print(f"\nPer-seed breakdown:")
    print(f"  {'Seed':<8} {'Bank':>10} {policy_key:>14}")
    for seed, r in zip(SEEDS, seed_results):
        print(f"  {seed:<8} {r['Bank']:>10.2f} {r[policy_key]:>14.2f}")


if __name__ == "__main__":
    main()
