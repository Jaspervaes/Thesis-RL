"""Run Single-Model CQL pipeline over 5 seeds.

Usage:
    python run_experiments.py
    python run_experiments.py --n_cases 50000
    python run_experiments.py --confounded
    python run_experiments.py --n_cases 50000 --confounded
"""
import argparse
import subprocess
import sys

SEEDS = [42, 123, 456, 789, 1024]


def run(cmd):
    print(f"  > {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cases',    type=int,  default=10000)
    parser.add_argument('--confounded', action='store_true')
    parser.add_argument('--method',     choices=['single', 'multi', 'all'], default='all')
    args = parser.parse_args()

    extra = ['--confounded'] if args.confounded else []
    n = str(args.n_cases)
    methods = (['singleModelCQL', 'multiModelCQL'] if args.method == 'all'
               else ['singleModelCQL'] if args.method == 'single'
               else ['multiModelCQL'])

    for method in methods:
        print(f"\n{'='*50}")
        print(f"  {method}  |  n_cases={n}  confounded={args.confounded}")
        print(f"{'='*50}")

        for seed in SEEDS:
            eval_seed = int(f"99{seed}")
            print(f"\n--- Seed {seed} (eval {eval_seed}) ---")

            run([sys.executable, f'{method}/generate_data.py', '--n_cases', n, '--seed', str(seed)] + extra)
            run([sys.executable, f'{method}/convert_data.py',  '--n_cases', n] + extra)
            run([sys.executable, f'{method}/train.py',         '--n_cases', n, '--seed', str(seed)] + extra)
            run([sys.executable, f'{method}/evaluate.py',      '--n_cases', n, '--seed', str(eval_seed)] + extra)

            print(f"  Seed {seed} done.")

    print(f"\n{'='*50}")
    print("  All done.")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
