"""
Run all methods × steps (1,2,3) × conditions (RCT, CONF) × seeds.
Saves aggregated results to results/all_results.json for plotting.

Usage:
    python run_all_steps.py                              # all methods, all steps, RCT+CONF
    python run_all_steps.py --methods kmeans lstm        # subset of methods
    python run_all_steps.py --steps 1 3                  # only 1-step and 3-step
    python run_all_steps.py --no_confounded              # RCT only
    python run_all_steps.py --n_cases 5000               # smaller dataset
"""
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
ALL_METHODS = ['kmeans', 'lstm', 'rims', 'multiModelCQL', 'singleModelCQL']
ALL_STEPS   = [1, 2, 3]

METHOD_SCRIPTS = {
    'kmeans': {
        'generate': 'kmeans/generate_data.py',
        'convert':  'kmeans/convert_data.py',
        'train':    'kmeans/train.py',
        'evaluate': 'kmeans/evaluate.py',
    },
    'lstm': {
        'generate': 'lstm/generate_data.py',
        'convert':  'lstm/convert_data.py',
        'train':    'lstm/train.py',
        'evaluate': 'lstm/evaluate.py',
    },
    'rims': {
        'generate': 'rims/generate_data.py',
        'convert':  'rims/convert_data.py',
        'train':    'rims/train.py',
        'evaluate': 'rims/evaluate.py',
    },
    'multiModelCQL': {
        'generate': 'multiModelCQL/generate_data.py',
        'convert':  'multiModelCQL/convert_data.py',
        'train':    'multiModelCQL/train.py',
        'evaluate': 'multiModelCQL/evaluate.py',
    },
    'singleModelCQL': {
        'generate': 'singleModelCQL/generate_data.py',
        'convert':  'singleModelCQL/convert_data.py',
        'train':    'singleModelCQL/train.py',
        'evaluate': 'singleModelCQL/evaluate.py',
    },
}

# File prefix used in data/ and models/ directories
FILE_PREFIX = {
    'kmeans':       'kmeans',
    'lstm':         'lstm',
    'rims':         'rims',
    'multiModelCQL':  'multi_cql',
    'singleModelCQL': 'single_cql',
}

# singleModelCQL generates _train.pkl/_val.pkl instead of _raw.pkl
SPLIT_DATA_METHODS = {'singleModelCQL'}


def run(cmd):
    print(f"    $ {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    subprocess.run(cmd, cwd=script_dir, check=True, env=env)


def data_exists(method, suffix, n_cases):
    prefix = FILE_PREFIX[method]
    if method in SPLIT_DATA_METHODS:
        path = os.path.join(script_dir, f"data/{prefix}_{suffix}_{n_cases}_train.pkl")
    else:
        path = os.path.join(script_dir, f"data/{prefix}_{suffix}_{n_cases}_raw.pkl")
    return os.path.exists(path)


def transitions_exist(method, suffix, n_cases, steps):
    prefix   = FILE_PREFIX[method]
    step_tag = "" if steps == 3 else f"_steps{steps}"
    if method == 'rims':
        path = os.path.join(script_dir, f"data/{prefix}_{suffix}_{n_cases}_simulator{step_tag}.pkl")
    else:
        path = os.path.join(script_dir, f"data/{prefix}_{suffix}_{n_cases}_trans_train{step_tag}.pkl")
    return os.path.exists(path)


def model_exists(method, suffix, n_cases, seed, steps):
    prefix   = FILE_PREFIX[method]
    step_tag = "" if steps == 3 else f"_steps{steps}"
    ext      = "pkl" if method == "kmeans" else "pth"
    path = os.path.join(script_dir, f"models/{prefix}_{suffix}_{n_cases}_s{seed}{step_tag}.{ext}")
    return os.path.exists(path)


def generate_data(method, suffix, n_cases, data_seed=42, delta=0.95, force=False):
    if not force and data_exists(method, suffix, n_cases):
        print(f"    [skip] {method} {suffix} data already exists")
        return
    cmd = [sys.executable, METHOD_SCRIPTS[method]['generate'],
           '--n_cases', str(n_cases), '--seed', str(data_seed)]
    if suffix == 'CONF':
        cmd += ['--confounded', '--delta', str(delta)]
    run(cmd)


def convert_data(method, suffix, n_cases, steps, split_seed=42, force=False):
    if not force and transitions_exist(method, suffix, n_cases, steps):
        print(f"    [skip] {method} {suffix} steps={steps} transitions already exist")
        return
    cmd = [sys.executable, METHOD_SCRIPTS[method]['convert'],
           '--n_cases', str(n_cases), '--seed', str(split_seed),
           '--steps', str(steps)]
    if suffix == 'CONF':
        cmd += ['--confounded']
    run(cmd)


def train_model(method, suffix, n_cases, steps, seed, extra_args, force=False):
    if not force and model_exists(method, suffix, n_cases, seed, steps):
        print(f"    [skip] {method} {suffix} steps={steps} seed={seed} model already exists")
        return
    cmd = [sys.executable, METHOD_SCRIPTS[method]['train'],
           '--n_cases', str(n_cases), '--seed', str(seed), '--steps', str(steps)]
    if suffix == 'CONF':
        cmd += ['--confounded']
    cmd += extra_args
    run(cmd)


def evaluate_model(method, suffix, n_cases, steps, train_seed, n_episodes, results_file):
    eval_seed = int(f"99{train_seed}")
    cmd = [sys.executable, METHOD_SCRIPTS[method]['evaluate'],
           '--n_cases', str(n_cases), '--train_seed', str(train_seed),
           '--steps', str(steps), '--n_episodes', str(n_episodes),
           '--seed', str(eval_seed), '--results_file', results_file]
    if suffix == 'CONF':
        cmd += ['--confounded']
    run(cmd)


def run_combination(method, suffix, steps, n_cases, n_episodes, extra_train_args, force):
    """Run all seeds for one (method, suffix, steps) combination. Returns per-seed results."""
    print(f"\n  [{method} | {suffix} | {steps}-step]")

    tmpdir = tempfile.mkdtemp()
    seed_results = {}

    for seed in SEEDS:
        print(f"  seed={seed}")
        results_file = os.path.join(tmpdir, f"r_{seed}.json")
        train_model(method, suffix, n_cases, steps, seed, extra_train_args, force)
        evaluate_model(method, suffix, n_cases, steps, seed, n_episodes, results_file)
        with open(results_file) as f:
            seed_results[str(seed)] = json.load(f)

    return seed_results


def aggregate(seed_results, method_label):
    """Compute mean/std across seeds for Bank, method, and Random."""
    bank_avgs   = [v['Bank']       for v in seed_results.values()]
    random_avgs = [v['Random']     for v in seed_results.values()]
    method_avgs = [v[method_label] for v in seed_results.values()]
    return {
        'Bank':   {'mean': float(np.mean(bank_avgs)),   'std': float(np.std(bank_avgs)),   'per_seed': bank_avgs},
        'Random': {'mean': float(np.mean(random_avgs)), 'std': float(np.std(random_avgs)), 'per_seed': random_avgs},
        method_label: {'mean': float(np.mean(method_avgs)), 'std': float(np.std(method_avgs)), 'per_seed': method_avgs},
    }


def print_summary(all_results, methods):
    print(f"\n{'='*70}")
    print("SUMMARY — all methods × steps × conditions")
    print('='*70)
    print(f"{'Method':<14} {'Cond':<6} {'Steps':<6} {'Bank':>10} {'Policy':>10} {'Random':>10} {'Gain':>8}")
    print('-'*70)

    for method in methods:
        for suffix in ['RCT', 'CONF']:
            for steps in ALL_STEPS:
                key = f"{method}_{suffix}_{steps}"
                if key not in all_results:
                    continue
                agg = all_results[key]['aggregated']
                bank_m = agg['Bank']['mean']
                rand_m = agg['Random']['mean']
                policy_key = [k for k in agg if k not in ('Bank', 'Random')][0]
                pol_m = agg[policy_key]['mean']
                gain  = ((pol_m / bank_m) - 1) * 100 if bank_m > 0 else float('nan')
                print(f"{method:<14} {suffix:<6} {steps:<6} {bank_m:>10.1f} {pol_m:>10.1f} {rand_m:>10.1f} {gain:>7.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods',       nargs='+', default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument('--steps',         nargs='+', type=int, default=ALL_STEPS)
    parser.add_argument('--n_cases',       type=int,  default=10000)
    parser.add_argument('--n_episodes',    type=int,  default=1000)
    parser.add_argument('--no_confounded', action='store_true', help='Skip CONF condition')
    parser.add_argument('--rct_only',      action='store_true', help='Same as --no_confounded')
    parser.add_argument('--force',         action='store_true', help='Re-run even if outputs exist')
    parser.add_argument('--results_out',   type=str,  default='results/all_results.json')
    args, extra = parser.parse_known_args()

    suffixes = ['RCT']
    if not args.no_confounded and not args.rct_only:
        suffixes.append('CONF')

    os.makedirs(os.path.join(script_dir, 'results'), exist_ok=True)

    all_results = {}
    if os.path.exists(os.path.join(script_dir, args.results_out)):
        with open(os.path.join(script_dir, args.results_out)) as f:
            all_results = json.load(f)

    for method in args.methods:
        for suffix in suffixes:
            print(f"\n{'='*60}")
            print(f"  Generating {method} {suffix} data  (n_cases={args.n_cases})")
            print('='*60)
            generate_data(method, suffix, args.n_cases, force=args.force)

            for steps in args.steps:
                print(f"\n  Converting {method} {suffix} steps={steps}")
                convert_data(method, suffix, args.n_cases, steps, force=args.force)

                seed_results = run_combination(
                    method, suffix, steps, args.n_cases, args.n_episodes, extra, args.force
                )

                # Find the method label from results (e.g. 'KMeans RCT (1-step)')
                example = next(iter(seed_results.values()))
                method_label = [k for k in example if k not in ('Bank', 'Random')][0]

                agg = aggregate(seed_results, method_label)
                key = f"{method}_{suffix}_{steps}"
                all_results[key] = {
                    'method': method, 'suffix': suffix, 'steps': steps,
                    'method_label': method_label,
                    'per_seed': seed_results,
                    'aggregated': agg,
                }

                # Save after each combination so progress is preserved
                results_path = os.path.join(script_dir, args.results_out)
                with open(results_path, 'w') as f:
                    json.dump(all_results, f, indent=2)
                print(f"  [saved] {args.results_out}")

    print_summary(all_results, args.methods)
    print(f"\n[OK] All results saved to {args.results_out}")
    print(f"     Run: python plot_results.py  to generate thesis figures")


if __name__ == "__main__":
    main()
