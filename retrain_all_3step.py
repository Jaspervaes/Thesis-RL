"""
Full retrain of all 9 methods × 5 seeds × {CONF, RCT} for --steps 3.

Skips already-trained models (delete the file first to force retrain).
Writes per-seed evaluation to a fresh aggregate, then overwrites
`results/all_results.json::<method>_{CONF|RCT}_3` with the new aggregated entry.

Usage:
    python -u retrain_all_3step.py                          # both CONF and RCT
    python -u retrain_all_3step.py --suffixes CONF          # CONF only
    python -u retrain_all_3step.py --methods lstm kmeans    # method subset
    python -u retrain_all_3step.py --seeds 42 123           # seed subset
"""
import os
import sys
import json
import argparse
import subprocess
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

# (method_key, train_dir, model_filename_prefix)
METHODS = [
    ("kmeans",             "methods/K-Means-FQI",        "kmeans"),
    ("lstm",               "methods/LSTM-DQN",           "lstm"),
    ("rims",               "methods/RIMS-DQN",           "rims"),
    ("multiModelCQL",      "methods/CQL-MN",             "multi_cql"),
    ("singleModelCQL",     "methods/CQL-SN",             "single_cql"),
    ("lstm_dqn_dragonnet", "methods/LSTM-DQN-DragonNet", "lstm_dqn_dragonnet"),
    ("lstm_dqn_tabpfn",    "methods/LSTM-DQN-TabPFN",    "lstm_dqn_tabpfn"),
    ("procause_econml",    "methods/LSTM-DQN-GBR",       "procause_econml"),
    ("procause_lstm",      "methods/LSTM-DQN-SLearner",  "procause_lstm"),
]
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
N_CASES = 10000
N_EPISODES = 1000


def model_ext(method): return "pkl" if method == "kmeans" else "pth"


def run(cmd, label):
    print(f"\n  > [{label}] {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=False)
    if r.returncode != 0:
        print(f"  [ERROR] {label} exit {r.returncode}")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods',  nargs='+', default=[m[0] for m in METHODS])
    ap.add_argument('--seeds',    nargs='+', type=int, default=DEFAULT_SEEDS)
    ap.add_argument('--suffixes', nargs='+', default=["CONF", "RCT"], choices=["CONF", "RCT"])
    ap.add_argument('--results_file', default="results/all_results.json")
    args = ap.parse_args()

    all_results = {}
    if os.path.exists(args.results_file):
        with open(args.results_file) as f:
            all_results = json.load(f)

    methods_to_run = [m for m in METHODS if m[0] in args.methods]

    for suffix in args.suffixes:
        confounded_flag = ["--confounded"] if suffix == "CONF" else []
        print(f"\n{'#'*70}\n  SUFFIX: {suffix}\n{'#'*70}")

        for method, mdir, prefix in methods_to_run:
            print(f"\n{'='*70}\n  {method} | {suffix}\n{'='*70}")
            per_seed = []
            cat_label = None
            bank_per_seed = []
            random_per_seed = []

            for seed in args.seeds:
                ext = model_ext(method)
                model_path = f"models/{prefix}_{suffix}_{N_CASES}_s{seed}.{ext}"

                # Convert data if needed
                if os.path.exists(f"{mdir}/convert_data.py"):
                    run(["python", f"{mdir}/convert_data.py",
                         "--n_cases", str(N_CASES), "--seed", str(seed)] + confounded_flag,
                        f"convert_data {method} {suffix} s={seed}")

                # Train
                if not os.path.exists(model_path):
                    if not run(["python", f"{mdir}/train.py",
                                "--n_cases", str(N_CASES), "--seed", str(seed)] + confounded_flag,
                               f"train {method} {suffix} s={seed}"):
                        print(f"  skipping {method} {suffix} s={seed}")
                        continue
                else:
                    print(f"  [skip-train] {model_path} exists")

                # Evaluate
                tmp = f"results/_tmp_{prefix}_{suffix}_s{seed}.json"
                if not run(["python", f"{mdir}/evaluate.py",
                            "--n_cases", str(N_CASES),
                            "--train_seed", str(seed), "--n_episodes", str(N_EPISODES),
                            "--results_file", tmp] + confounded_flag,
                           f"evaluate {method} {suffix} s={seed}"):
                    continue

                with open(tmp) as f:
                    er = json.load(f)
                os.remove(tmp)

                for k, v in er.items():
                    if isinstance(v, (int, float)):
                        if k == "Bank":     bank_per_seed.append(v)
                        elif k == "Random": random_per_seed.append(v)
                        else:
                            if cat_label is None: cat_label = k
                            per_seed.append(v)
                            break

                print(f"  [done] {method} {suffix} s={seed}  outcome={per_seed[-1]:.2f}")

            if not per_seed:
                print(f"  no results for {method} {suffix}")
                continue

            agg_method = {"mean": float(np.mean(per_seed)), "std": float(np.std(per_seed)),
                          "per_seed": [float(x) for x in per_seed]}
            agg_bank = {"mean": float(np.mean(bank_per_seed)), "std": float(np.std(bank_per_seed)),
                        "per_seed": [float(x) for x in bank_per_seed]} if bank_per_seed else None
            agg_random = {"mean": float(np.mean(random_per_seed)), "std": float(np.std(random_per_seed)),
                          "per_seed": [float(x) for x in random_per_seed]} if random_per_seed else None

            aggregated = {cat_label or method: agg_method}
            if agg_bank: aggregated["Bank"] = agg_bank
            if agg_random: aggregated["Random"] = agg_random

            all_results[f"{method}_{suffix}_3"] = {
                "method": method, "suffix": suffix, "steps": 3,
                "aggregated": aggregated,
            }

            with open(args.results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"  [saved] {method}_{suffix}_3 mean={agg_method['mean']:.2f}")

    # Summary
    print(f"\n{'='*70}\n  FINAL SUMMARY (--steps 3)\n{'='*70}")
    print(f"  {'Method':<22} {'Suffix':<6} {'Mean':>12} {'Std':>10}")
    print("  " + "-" * 54)
    for suffix in args.suffixes:
        for method, _, _ in methods_to_run:
            entry = all_results.get(f"{method}_{suffix}_3", {}).get("aggregated", {})
            for k, v in entry.items():
                if k in ("Bank", "Random"): continue
                print(f"  {method:<22} {suffix:<6} {v['mean']:>12.2f} {v['std']:>10.2f}")
                break


if __name__ == "__main__":
    main()
