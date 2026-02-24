"""
Simple data utilities for standardized experiments.
"""
import sys
import os
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime

# Setup SimBank paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
simbank_parent = os.path.join(project_root, "SimBank-main")
simbank_package = os.path.join(simbank_parent, "SimBank")
sys.path.insert(0, project_root)
sys.path.insert(0, simbank_parent)
sys.path.insert(0, simbank_package)

from SimBank import simulation
from SimBank import confounding_level

from shared.experiment_config import BASE_FEATURES, TRACKED_ACTIVITIES, STATE_DIM


def get_simbank_params(n_cases, seed, rct=True):
    """Get standard SimBank parameters."""
    return {
        "train_size": n_cases,
        "simulation_start": datetime(2024, 3, 20, 8, 0),
        "random_seed_train": seed,
        "intervention_info": {
            "name": ["choose_procedure", "time_contact_HQ", "set_ir_3_levels"],
            "data_impact": ["direct", "direct", "indirect"],
            "actions": [["start_standard", "start_priority"], ["contact_headquarters", "skip_contact"], [0.07, 0.08, 0.09]],
            "action_width": [2, 2, 3], "action_depth": [1, 1, 1], "len": [1, 1, 1],
            "activities": [["start_standard", "start_priority"], ["contact_headquarters", "skip_contact"], ["calculate_offer"]],
            "column": ["activity", "activity", "interest_rate"],
            "start_control_activity": [["initiate_application"], ["collect_documents", "assess_quality"], []],
            "end_control_activity": [["initiate_application"], ["collect_documents", "assess_quality"], []],
            "retain_method": "precise", "RCT": rct, "RCT_timing": [1000, 1000, 1000]
        },
        "policies_info": {
            "general": "real", "choose_procedure": {"amount": 50000, "est_quality": 5},
            "time_contact_HQ": "real", "set_ir_3_levels": "real",
            "min_quality": 2, "max_noc": 3, "max_nor": 1, "min_amount_contact_cust": 50000
        },
        "log_cols": ["case_nr", "activity", "timestamp", "elapsed_time", "cum_cost",
                     "est_quality", "unc_quality", "amount", "interest_rate",
                     "discount_factor", "outcome", "quality", "noc", "nor", "min_interest_rate"],
    }


def generate_rct_data(n_cases, seed):
    """Generate RCT (100% random) data."""
    params = get_simbank_params(n_cases, seed, rct=True)
    gen = simulation.PresProcessGenerator(params, seed)
    return pd.DataFrame(gen.run_simulation_normal(n_cases)), params


def generate_confounded_data(n_cases, seed, delta=0.95):
    """Generate confounded data (delta% bank + (1-delta)% RCT)."""
    # Bank policy data
    params_bank = get_simbank_params(n_cases, seed, rct=False)
    gen_bank = simulation.PresProcessGenerator(params_bank, seed)
    df_bank = pd.DataFrame(gen_bank.run_simulation_normal(n_cases))

    # RCT data
    rct_seed = seed * 10
    params_rct = get_simbank_params(n_cases, rct_seed, rct=True)
    params_rct["simulation_start"] = deepcopy(gen_bank.simulation_end)
    gen_rct = simulation.PresProcessGenerator(params_rct, rct_seed)
    df_rct = pd.DataFrame(gen_rct.run_simulation_normal(n_cases))

    # Mix
    df_combined = confounding_level.set_delta(data=df_bank, data_RCT=df_rct, delta=delta)
    return df_combined, params_bank


def split_train_val(df, val_ratio=0.2, seed=42):
    """Split data by case_nr (not by events!)."""
    np.random.seed(seed)
    cases = df['case_nr'].unique()
    np.random.shuffle(cases)
    n_val = int(len(cases) * val_ratio)
    val_cases = set(cases[:n_val])
    return df[~df['case_nr'].isin(val_cases)].copy(), df[df['case_nr'].isin(val_cases)].copy()


def count_activities(events, up_to_idx):
    """Count activity occurrences up to index."""
    counts = {act: 0 for act in TRACKED_ACTIVITIES}
    for i in range(min(up_to_idx, len(events))):
        activity = events.iloc[i].get('activity', '').lower()
        for tracked in TRACKED_ACTIVITIES:
            if tracked in activity:
                counts[tracked] += 1
    return counts


def extract_state(event, activity_counts):
    """Build state vector from event + activity counts."""
    state = [float(event.get(f, 0)) for f in BASE_FEATURES]
    state += [float(activity_counts.get(a, 0)) for a in TRACKED_ACTIVITIES]
    return np.array(state, dtype=np.float32)


def get_ir_action(ir):
    """Convert interest rate to action index."""
    if abs(ir - 0.07) < 0.001: return 0
    if abs(ir - 0.08) < 0.001: return 1
    return 2


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
