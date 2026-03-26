"""Generate SimBank training data shared across all methods.

Saves:
  data/simbank_{RCT|CONF}_{n_cases}_raw.pkl
  data/simbank_{RCT|CONF}_{n_cases}_params.pkl

All convert_data.py scripts read from these shared files.
"""
import os
import argparse

from shared import generate_rct_data, generate_confounded_data, save_pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cases',    type=int,   default=10000)
    parser.add_argument('--confounded', action='store_true')
    parser.add_argument('--delta',      type=float, default=0.95)
    parser.add_argument('--seed',       type=int,   default=42)
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    suffix = "CONF" if args.confounded else "RCT"
    print(f"Generating {suffix} | n_cases={args.n_cases} seed={args.seed}")

    if args.confounded:
        df, params = generate_confounded_data(args.n_cases, args.seed, args.delta)
    else:
        df, params = generate_rct_data(args.n_cases, args.seed)

    base = f"data/simbank_{suffix}_{args.n_cases}"
    save_pickle(df,     f"{base}_raw.pkl")
    save_pickle(params, f"{base}_params.pkl")
    print(f"[OK] {df['case_nr'].nunique()} cases -> {base}_*.pkl")


if __name__ == "__main__":
    main()
