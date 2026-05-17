"""Generate SimBank training data shared across all methods.

Saves:
  data/simbank_{RCT|CONF}_{n_cases}_raw.pkl
  data/simbank_{RCT|CONF}_{n_cases}_params.pkl

With --active:
  data/simbank_CONF_{n_cases}_active{IDS}_raw.pkl
  data/simbank_CONF_{n_cases}_active{IDS}_params.pkl
"""
import os
import sys
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
methods_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(methods_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, methods_dir)
os.chdir(project_root)

from shared import (
    generate_rct_data, generate_confounded_data,
    generate_confounded_data_subset, generate_rct_data_subset,
    active_tag, save_pickle,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cases',    type=int,   default=10000)
    parser.add_argument('--confounded', action='store_true')
    parser.add_argument('--delta',      type=float, default=0.95)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--active',     type=str,   default=None,
                        help='Comma-separated active intervention indices, e.g. "0,2". '
                             'Inactive interventions use bank policy. Combine with --confounded '
                             'for the CONF regime (delta-mixed); omit it for the RCT regime '
                             '(pure RCT branch, active interventions randomised).')
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    if args.active is not None:
        active = [int(x) for x in args.active.split(',')]
        tag = active_tag(active)
        if args.confounded:
            print(f"Generating CONF subset active={active} | n_cases={args.n_cases} seed={args.seed}")
            df, params = generate_confounded_data_subset(args.n_cases, args.seed, active, args.delta)
            base = f"data/simbank_CONF_{args.n_cases}{tag}"
        else:
            print(f"Generating RCT subset active={active} | n_cases={args.n_cases} seed={args.seed}")
            df, params = generate_rct_data_subset(args.n_cases, args.seed, active)
            base = f"data/simbank_RCT_{args.n_cases}{tag}"
    elif args.confounded:
        print(f"Generating CONF | n_cases={args.n_cases} seed={args.seed}")
        df, params = generate_confounded_data(args.n_cases, args.seed, args.delta)
        base = f"data/simbank_CONF_{args.n_cases}"
    else:
        print(f"Generating RCT | n_cases={args.n_cases} seed={args.seed}")
        df, params = generate_rct_data(args.n_cases, args.seed)
        base = f"data/simbank_RCT_{args.n_cases}"

    save_pickle(df,     f"{base}_raw.pkl")
    save_pickle(params, f"{base}_params.pkl")
    print(f"[OK] {df['case_nr'].nunique()} cases -> {base}_*.pkl")


if __name__ == "__main__":
    main()
