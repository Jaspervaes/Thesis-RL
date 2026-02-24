"""
Standardized evaluation baselines and utilities.
"""
import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
simbank_parent = os.path.join(project_root, "SimBank-main")
simbank_package = os.path.join(simbank_parent, "SimBank")
sys.path.insert(0, simbank_parent)
sys.path.insert(0, simbank_package)

from SimBank import simulation


def bank_policy(prev_event, int_idx):
    """Exact bank policy from SimBank's extra_flow_conditions.py."""
    if int_idx == 0:  # choose_procedure
        if prev_event.get("amount", 0) > 50000 and prev_event.get("est_quality", 0) >= 5:
            return 1  # priority
        return 0  # standard
    elif int_idx == 1:  # contact_headquarters
        noc = prev_event.get("noc", 0)
        unc = prev_event.get("unc_quality", 1)
        amt = prev_event.get("amount", 0)
        est = prev_event.get("est_quality", 0)
        if noc < 2 and unc == 0 and amt > 10000 and est >= 2:
            return 0  # contact
        return 1  # skip
    else:  # set_ir_3_levels
        amt = prev_event.get("amount", 0)
        if amt > 60000: return 0  # 7%
        if amt > 30000: return 1  # 8%
        return 2  # 9%


def random_policy(prev_event, int_idx):
    """Random baseline policy."""
    return np.random.randint(0, [2, 2, 3][int_idx])


def evaluate_policy(policy_fn, n_episodes, params, seed, use_prefix=False, reset_fn=None, verbose=True):
    """
    Evaluate a policy through SimBank simulator.

    Args:
        policy_fn: Function(prev_event, int_idx, [prefix]) -> action
        n_episodes: Number of test cases
        params: SimBank dataset params
        seed: Random seed for simulation
        use_prefix: If True, pass prefix events to policy
        reset_fn: Optional function to call between episodes
        verbose: Print progress

    Returns:
        Dict with avg, std, outcomes, action_counts
    """
    gen = simulation.PresProcessGenerator(params, seed=seed)
    outcomes = []
    action_counts = {0: defaultdict(int), 1: defaultdict(int), 2: defaultdict(int)}

    for i in range(n_episodes):
        if reset_fn:
            reset_fn()

        prefix_list = gen.start_simulation_inference(seed_to_add=i)
        while gen.int_points_available:
            prefix = prefix_list[0][:-1]
            prev_event = prefix[-1]
            int_idx = gen.current_int_index

            if use_prefix:
                action = policy_fn(prev_event, int_idx, prefix)
            else:
                action = policy_fn(prev_event, int_idx)

            action_counts[int_idx][action] += 1
            prefix_list = gen.continue_simulation_inference(action)

        outcome = float(pd.DataFrame(gen.end_simulation_inference())["outcome"].iloc[-1])
        outcomes.append(outcome)

        if verbose and (i + 1) % 500 == 0:
            print(f"    {i+1}/{n_episodes}, avg: {np.mean(outcomes):.2f}")

    return {
        'avg': np.mean(outcomes),
        'std': np.std(outcomes),
        'outcomes': outcomes,
        'action_counts': dict(action_counts)
    }


def print_results(results_dict):
    """Print comparison table."""
    print(f"\n{'Policy':<20} {'Average':>10} {'Std':>8}")
    print('-' * 40)

    bank_avg = results_dict.get('Bank', {}).get('avg', 0)
    for name, res in results_dict.items():
        if bank_avg > 0 and name != 'Bank':
            vs = f"{((res['avg']/bank_avg)-1)*100:+.1f}%"
        else:
            vs = ""
        print(f"{name:<20} {res['avg']:>10.2f} {res['std']:>8.2f} {vs}")


def print_action_dist(results_dict):
    """Print action distributions."""
    names = {0: ['Std', 'Pri'], 1: ['Contact', 'Skip'], 2: ['7%', '8%', '9%']}
    print("\nAction distributions:")
    for int_idx, int_name in enumerate(['Procedure', 'Contact', 'Interest']):
        print(f"  {int_name}:")
        for policy_name, res in results_dict.items():
            counts = res['action_counts'].get(int_idx, {})
            total = sum(counts.values())
            if total == 0:
                continue
            dist = ", ".join(f"{names[int_idx][a]}={100*counts.get(a,0)/total:.0f}%" for a in range(len(names[int_idx])))
            print(f"    {policy_name}: {dist}")
