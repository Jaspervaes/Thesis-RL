"""
Standardized constants for fair experimental comparison.
Keep it simple - just the shared settings.
"""

# Standard random seeds (run each experiment 5 times with same seeds across methods)
SEEDS = [42, 123, 456, 789, 1024]

# Standard dataset sizes
SIZES = {'small': 5000, 'medium': 10000, 'large': 50000, 'full': 100000}

# Standard confounding levels (delta = fraction of bank policy data)
DELTAS = {'rct': 0.0, 'low': 0.5, 'medium': 0.8, 'high': 0.95}

# State representation
BASE_FEATURES = ['amount', 'est_quality', 'unc_quality', 'cum_cost', 'elapsed_time']
TRACKED_ACTIVITIES = [
    'initiate_application', 'start_standard', 'start_priority',
    'call_customer', 'email_customer', 'validate_application',
    'contact_headquarters', 'skip_contact', 'calculate_offer',
    'cancel_application', 'receive_acceptance'
]
STATE_DIM = len(BASE_FEATURES) + len(TRACKED_ACTIVITIES)  # 16

# SimBank intervention info (shared across all methods)
INTERVENTION_INFO = {
    "name": ["choose_procedure", "time_contact_HQ", "set_ir_3_levels"],
    "actions": [["start_standard", "start_priority"], ["contact_headquarters", "skip_contact"], [0.07, 0.08, 0.09]],
    "action_width": [2, 2, 3],
    "activities": [["start_standard", "start_priority"], ["contact_headquarters", "skip_contact"], ["calculate_offer"]],
}
