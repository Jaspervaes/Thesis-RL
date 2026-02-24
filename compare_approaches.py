"""
Compare CQL Approaches - Full Pipeline
=======================================
Runs both newCQLApproach and UnifiedApproach with identical parameters
for a fair 1-to-1 comparison.

Both approaches now support 3 interventions:
1. choose_procedure (binary)
2. time_contact_HQ (binary)
3. set_ir_3_levels (ternary)

Usage:
    python compare_approaches.py --rct --train-cases 100000 --test-cases 10000 --epochs 50
    python compare_approaches.py --confounded --train-cases 100000 --test-cases 10000 --epochs 50
"""
import subprocess
import sys
import os
import argparse
import pickle
from datetime import datetime

def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with return code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='Compare newCQLApproach vs UnifiedApproach')

    # Data type
    parser.add_argument('--rct', dest='rct', action='store_true',
                       help='Use RCT (randomized) data (default)')
    parser.add_argument('--confounded', dest='rct', action='store_false',
                       help='Use confounded (bank policy) data')
    parser.set_defaults(rct=True)

    # Sizes
    parser.add_argument('--train-cases', type=int, default=100000,
                       help='Number of training cases (default: 100000)')
    parser.add_argument('--test-cases', type=int, default=10000,
                       help='Number of test/evaluation cases (default: 10000)')

    # Hyperparameters (shared by both approaches)
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs per network (default: 50)')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='CQL alpha (conservative weight) (default: 1.0)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size (default: 256)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    # Skip flags
    parser.add_argument('--skip-data-gen', action='store_true',
                       help='Skip data generation (use existing data)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training (use existing models)')
    parser.add_argument('--only-unified', action='store_true',
                       help='Only run UnifiedApproach')
    parser.add_argument('--only-new', action='store_true',
                       help='Only run newCQLApproach')

    args = parser.parse_args()

    data_type = "RCT" if args.rct else "CONFOUNDED"

    print("\n" + "="*70)
    print("CQL APPROACH COMPARISON")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Data type: {data_type}")
    print(f"  Training cases: {args.train_cases}")
    print(f"  Evaluation cases: {args.test_cases}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Alpha (CQL): {args.alpha}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Seed: {args.seed}")
    print("="*70)

    python = sys.executable
    rct_flag = "" if args.rct else "--confounded"
    rct_flag_new = "--rct" if args.rct else "--no-rct"

    # =========================================================================
    # STEP 1: GENERATE DATA
    # =========================================================================
    if not args.skip_data_gen:
        print("\n" + "#"*70)
        print("# STEP 1: DATA GENERATION")
        print("#"*70)

        if not args.only_unified:
            # newCQLApproach data generation
            cmd = [python, "newApproachCQL/generate_data.py",
                   rct_flag_new,
                   "--n-cases", str(args.train_cases),
                   "--seed", str(args.seed)]
            if not run_command(cmd, "Generating data for newCQLApproach"):
                print("[ERROR] newCQLApproach data generation failed")
                return

        if not args.only_new:
            # UnifiedApproach data generation
            cmd = [python, "uinifiedApproachCQL/generate_sequential_data.py",
                   "--n_cases", str(args.train_cases),
                   "--seed", str(args.seed)]
            if rct_flag:
                cmd.append(rct_flag)
            if not run_command(cmd, "Generating data for UnifiedApproach"):
                print("[ERROR] UnifiedApproach data generation failed")
                return

            # Convert data for UnifiedApproach
            cmd = [python, "uinifiedApproachCQL/convert_to_cql_unified.py",
                   "--n_cases", str(args.train_cases)]
            if rct_flag:
                cmd.append(rct_flag)
            if not run_command(cmd, "Converting data for UnifiedApproach"):
                print("[ERROR] UnifiedApproach data conversion failed")
                return

    # =========================================================================
    # STEP 2: TRAINING
    # =========================================================================
    if not args.skip_training:
        print("\n" + "#"*70)
        print("# STEP 2: TRAINING")
        print("#"*70)

        if not args.only_unified:
            # Train newCQLApproach
            cmd = [python, "newApproachCQL/train_cql.py",
                   rct_flag_new,
                   "--train-size", str(args.train_cases),
                   "--epochs", str(args.epochs),
                   "--batch-size", str(args.batch_size),
                   "--lr", str(args.lr),
                   "--cql-alpha", str(args.alpha),
                   "--gamma", str(args.gamma),
                   "--seed", str(args.seed)]
            if not run_command(cmd, "Training newCQLApproach"):
                print("[ERROR] newCQLApproach training failed")
                return

        if not args.only_new:
            # Train UnifiedApproach
            cmd = [python, "uinifiedApproachCQL/train_cql_unified_fast.py",
                   "--epochs", str(args.epochs),
                   "--batch_size", str(args.batch_size),
                   "--lr", str(args.lr),
                   "--alpha", str(args.alpha),
                   "--gamma", str(args.gamma),
                   "--n_cases", str(args.train_cases)]
            if rct_flag:
                cmd.append(rct_flag)
            if not run_command(cmd, "Training UnifiedApproach"):
                print("[ERROR] UnifiedApproach training failed")
                return

    # =========================================================================
    # STEP 3: EVALUATION
    # =========================================================================
    print("\n" + "#"*70)
    print("# STEP 3: EVALUATION")
    print("#"*70)

    results = {}

    if not args.only_unified:
        # Evaluate newCQLApproach
        cmd = [python, "newApproachCQL/evaluate_cql.py",
               rct_flag_new,
               "--train-size", str(args.train_cases),
               "--test-size", str(args.test_cases),
               "--seed", str(args.seed)]
        if not run_command(cmd, "Evaluating newCQLApproach"):
            print("[ERROR] newCQLApproach evaluation failed")
        else:
            # Load results
            results_path = f"data/cql_evaluation_{data_type.lower()}_{args.train_cases}.pkl"
            if os.path.exists(results_path):
                with open(results_path, 'rb') as f:
                    results['newCQL'] = pickle.load(f)

    if not args.only_new:
        # Evaluate UnifiedApproach
        cmd = [python, "uinifiedApproachCQL/evaluate_cql_unified.py",
               "--n_episodes", str(args.test_cases),
               "--seed", str(args.seed),
               "--n_cases", str(args.train_cases)]
        if rct_flag:
            cmd.append(rct_flag)
        if not run_command(cmd, "Evaluating UnifiedApproach"):
            print("[ERROR] UnifiedApproach evaluation failed")
        else:
            # Load results
            suffix = "_BANK" if not args.rct else "_RCT"
            results_path = f"data/CQL_unified_3step_evaluation{suffix}.pkl"
            if os.path.exists(results_path):
                with open(results_path, 'rb') as f:
                    results['unified'] = pickle.load(f)

    # =========================================================================
    # STEP 4: COMPARISON SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL COMPARISON SUMMARY")
    print("="*70)

    print(f"\nConfiguration:")
    print(f"  Data type: {data_type}")
    print(f"  Training cases: {args.train_cases}")
    print(f"  Evaluation cases: {args.test_cases}")
    print(f"  Epochs: {args.epochs}")
    print(f"  CQL Alpha: {args.alpha}")

    print(f"\n{'Model':<25} {'Avg Reward':>15} {'Gain vs Bank':>15}")
    print("-"*55)

    if 'newCQL' in results:
        new_avg = results['newCQL']['cql']['average']
        new_gain = results['newCQL']['gains']['cql_vs_bank']
        print(f"{'newCQLApproach':<25} {new_avg:>15.2f} {new_gain:>+14.2f}%")

    if 'unified' in results:
        uni_avg = results['unified']['cql']['average']
        uni_gain = results['unified']['gains']['cql_vs_bank']
        print(f"{'UnifiedApproach':<25} {uni_avg:>15.2f} {uni_gain:>+14.2f}%")

    # Bank baseline (should be same for both)
    if 'newCQL' in results:
        bank_avg = results['newCQL']['bank']['average']
        print(f"{'Bank (baseline)':<25} {bank_avg:>15.2f} {'---':>15}")
    elif 'unified' in results:
        bank_avg = results['unified']['bank']['average']
        print(f"{'Bank (baseline)':<25} {bank_avg:>15.2f} {'---':>15}")

    print("-"*55)

    # Winner
    if 'newCQL' in results and 'unified' in results:
        new_gain = results['newCQL']['gains']['cql_vs_bank']
        uni_gain = results['unified']['gains']['cql_vs_bank']

        if new_gain > uni_gain:
            winner = "newCQLApproach"
            diff = new_gain - uni_gain
        elif uni_gain > new_gain:
            winner = "UnifiedApproach"
            diff = uni_gain - new_gain
        else:
            winner = "TIE"
            diff = 0

        print(f"\nWinner: {winner}")
        if diff > 0:
            print(f"Margin: {diff:.2f}% better gain vs Bank")

    print("\n" + "="*70)

    # Save comparison results
    comparison_results = {
        'config': {
            'data_type': data_type,
            'train_cases': args.train_cases,
            'test_cases': args.test_cases,
            'epochs': args.epochs,
            'alpha': args.alpha,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'gamma': args.gamma,
            'seed': args.seed
        },
        'results': results
    }

    comparison_path = f"data/comparison_{data_type.lower()}_{args.train_cases}.pkl"
    with open(comparison_path, 'wb') as f:
        pickle.dump(comparison_results, f)
    print(f"\nComparison results saved to: {comparison_path}")


if __name__ == "__main__":
    main()
