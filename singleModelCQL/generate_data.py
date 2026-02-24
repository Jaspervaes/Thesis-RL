"""
Generate training data for Single-Model CQL.
"""
import sys
import os
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
os.chdir(project_root)

from shared import generate_rct_data, generate_confounded_data, split_train_val, save_pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cases', type=int, default=10000)
    parser.add_argument('--confounded', action='store_true')
    parser.add_argument('--delta', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_split', type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    suffix = "CONF" if args.confounded else "RCT"

    print(f"\n{'='*50}")
    print(f"Generating {'confounded' if args.confounded else 'RCT'} data")
    print(f"n_cases={args.n_cases}, seed={args.seed}")
    print('='*50)

    # Generate data
    if args.confounded:
        df, params = generate_confounded_data(args.n_cases, args.seed, args.delta)
    else:
        df, params = generate_rct_data(args.n_cases, args.seed)

    # Split train/val
    df_train, df_val = split_train_val(df, args.val_split, args.seed)
    print(f"\nTrain: {df_train['case_nr'].nunique()} cases")
    print(f"Val: {df_val['case_nr'].nunique()} cases")

    # Save
    base = f"data/single_cql_{suffix}_{args.n_cases}"
    save_pickle(df_train.to_dict('records'), f"{base}_train.pkl")
    save_pickle(df_val.to_dict('records'), f"{base}_val.pkl")
    save_pickle(params, f"{base}_params.pkl")

    print(f"\n[OK] Saved to {base}_*.pkl")
    print(f"Next: python singleModelCQL/convert_data.py --n_cases {args.n_cases} {'--confounded' if args.confounded else ''}")


if __name__ == "__main__":
    main()
