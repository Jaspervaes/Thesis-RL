"""
Joint vs subset training comparison for LSTM-DQN on CONF or RCT data.

For each subset of active interventions, generates subset-specific data
(inactive interventions always follow bank policy), trains LSTM-DQN Q-networks
with TD routing between active interventions, and evaluates.

Compares against the existing joint 3-step baseline from all_results.json.

Usage:
    python -u run_intervention_combos.py                       # CONF, all subsets, all seeds
    python -u run_intervention_combos.py --suffix RCT          # RCT regime
    python -u run_intervention_combos.py --seeds 42            # single seed (for testing)
    python -u run_intervention_combos.py --subsets 1,2         # specific subsets
"""
import sys
import os
import json
import argparse
import subprocess
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

SUBSETS   = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
SEEDS     = [42, 123, 456, 789, 1024]
N_CASES   = 10000
N_EPISODES = 1000


def active_tag(active):
    """Model/result tag — always unique per subset."""
    return "_active" + "".join(str(i) for i in sorted(active))


def data_tag(active):
    """Data tag — [0,1,2] reuses the standard CONF dataset."""
    return "" if sorted(active) == [0, 1, 2] else "_active" + "".join(str(i) for i in sorted(active))


def ids_str(active):
    return "".join(str(i) for i in sorted(active))


def run(cmd, label):
    print(f"\n  [RUN] {label}")
    print(f"        {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  [ERROR] {label} exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds',   nargs='+', type=int, default=None)
    parser.add_argument('--subsets', type=str, default=None,
                        help='Comma-separated subset ids to run, e.g. "1,2,12,02,012"')
    parser.add_argument('--suffix',  type=str, default='CONF', choices=['CONF', 'RCT'],
                        help='Data regime for both data generation and training (default: CONF).')
    parser.add_argument('--results_out', type=str, default=None,
                        help='Defaults to results/lstm_joint_vs_subset.json for CONF, '
                             'results/lstm_joint_vs_subset_rct.json for RCT.')
    args = parser.parse_args()

    suffix = args.suffix
    confounded_flag = ["--confounded"] if suffix == "CONF" else []
    if args.results_out is None:
        args.results_out = ('results/lstm_joint_vs_subset.json' if suffix == 'CONF'
                            else 'results/lstm_joint_vs_subset_rct.json')

    os.chdir(script_dir)
    os.makedirs("results", exist_ok=True)

    seeds = args.seeds if args.seeds else SEEDS
    subsets = SUBSETS
    if args.subsets:
        requested = set(args.subsets.split(','))
        subsets = [s for s in SUBSETS if ids_str(s) in requested]

    all_results = {}
    if os.path.exists(args.results_out):
        with open(args.results_out) as f:
            all_results = json.load(f)

    for active in subsets:
        tag  = active_tag(active)
        ids  = ids_str(active)
        key  = f"lstm_{suffix}_Int{ids}"
        print(f"\n{'='*64}\n  Subset {{int{ids}}} | active={active} | suffix={suffix}\n{'='*64}")

        per_seed = all_results.get(key, {}).get('per_seed', {})

        for seed in seeds:
            if str(seed) in per_seed:
                print(f"  [skip] seed={seed} already in results")
                continue

            active_str = ",".join(str(i) for i in sorted(active))
            dtag = data_tag(active)  # "" for [0,1,2] — reuses standard regime data

            # 1. Generate data — [0,1,2] reuses the standard regime dataset
            data_raw = f"data/simbank_{suffix}_{N_CASES}{dtag}_raw.pkl"
            if not os.path.exists(data_raw):
                if dtag == "":
                    print(f"  [ERROR] Standard {suffix} raw data not found: {data_raw}", file=sys.stderr)
                    print(f"          Run the main data generation pipeline first.", file=sys.stderr)
                    sys.exit(1)
                run(["python", "methods/shared/generate_data.py",
                     "--n_cases", str(N_CASES),
                     "--seed",    str(seed),
                     "--active",  active_str] + confounded_flag,
                    f"generate_data {suffix} seed={seed} active={active_str}")
            else:
                print(f"  [skip] {data_raw} exists")

            # 2. Convert transitions — always regenerate with this seed's split so that
            # the val set used for early stopping matches the training seed, consistent
            # with how retrain_all_3step.py regenerates per-seed for the joint baseline.
            if dtag == "":
                # active=[0,1,2]: regenerate the shared standard trans file per seed
                run(["python", "methods/LSTM-DQN/convert_data.py",
                     "--n_cases", str(N_CASES),
                     "--seed",    str(seed)] + confounded_flag,
                    f"convert_data {suffix} seed={seed}")
            else:
                run(["python", "methods/LSTM-DQN/convert_data.py",
                     "--n_cases",    str(N_CASES),
                     "--active",     active_str,
                     "--seed",       str(seed)] + confounded_flag,
                    f"convert_data {suffix} active={active_str} seed={seed}")

            # 3. Train — check that checkpoint has Q_int{i} keys (not legacy Q1/Q2/Q3)
            model_file = f"models/lstm_{suffix}_{N_CASES}_s{seed}{tag}.pth"
            needs_train = not os.path.exists(model_file)
            if not needs_train:
                import torch as _torch
                _keys = _torch.load(model_file, map_location='cpu', weights_only=False).keys()
                needs_train = f"Q_int{sorted(active)[0]}" not in _keys
            if needs_train:
                run(["python", "methods/LSTM-DQN/train.py",
                     "--n_cases",    str(N_CASES),
                     "--seed",       str(seed),
                     "--active",     active_str] + confounded_flag,
                    f"train {suffix} seed={seed} active={active_str}")
            else:
                print(f"  [skip] {model_file} exists (Q_int keys present)")

            # 4. Evaluate
            tmp_file = f"results/_tmp_subset_{suffix}_{ids}_seed{seed}.json"
            run(["python", "methods/LSTM-DQN/evaluate.py",
                 "--n_cases",    str(N_CASES),
                 "--train_seed", str(seed),
                 "--n_episodes", str(N_EPISODES),
                 "--active",     active_str,
                 "--results_file", tmp_file] + confounded_flag,
                f"evaluate {suffix} seed={seed} active={active_str}")

            with open(tmp_file) as f:
                tmp = json.load(f)
            per_seed[str(seed)] = tmp[key]
            os.remove(tmp_file)

            # Save partial results after each seed
            vals = [v for v in per_seed.values()]
            all_results[key] = {
                'mean':     float(np.mean(vals)),
                'std':      float(np.std(vals)),
                'per_seed': per_seed,
            }
            with open(args.results_out, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"  [saved] seed={seed} outcome={per_seed[str(seed)]:.2f}")

    # Add joint baseline from all_results.json
    all_json = "results/all_results.json"
    joint_key = f"lstm_{suffix}_joint"
    if os.path.exists(all_json):
        with open(all_json) as f:
            ar = json.load(f)
        src = f"lstm_{suffix}_3"
        if src in ar:
            all_results[joint_key] = ar[src].get("aggregated", {})
            print(f"\n[OK] Loaded {joint_key} from all_results.json")

    with open(args.results_out, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print(f"\n{'='*64}\n  SUMMARY ({suffix})\n{'='*64}")
    print(f"  {'Key':<24} {'Mean':>10} {'Std':>8}")
    print("  " + "-" * 44)
    joint_mean = all_results.get(joint_key, {}).get("mean", 0) or \
                 all_results.get(joint_key, {}).get("avg", 0)
    if joint_mean:
        print(f"  {joint_key:<24} {joint_mean:>10.2f}")
    for k in [f"lstm_{suffix}_Int{''.join(str(i) for i in sorted(s))}" for s in SUBSETS]:
        if k in all_results:
            agg = all_results[k]
            print(f"  {k:<24} {agg['mean']:>10.2f} {agg['std']:>8.2f}")

    print(f"\n[OK] Results saved to {args.results_out}")


if __name__ == "__main__":
    main()
