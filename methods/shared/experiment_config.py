"""Standardized constants for fair experimental comparison."""

SEEDS = [42, 123, 456, 789, 1024]

SIZES = {'small': 5000, 'medium': 10000, 'large': 50000, 'full': 100000}

DELTAS = {'rct': 0.0, 'low': 0.5, 'medium': 0.8, 'high': 0.95}

BASE_FEATURES = ['amount', 'est_quality', 'unc_quality', 'cum_cost', 'elapsed_time']
TRACKED_ACTIVITIES = [
    'initiate_application', 'start_standard', 'start_priority',
    'call_customer', 'email_customer', 'validate_application',
    'contact_headquarters', 'skip_contact', 'calculate_offer',
    'cancel_application', 'receive_acceptance', 'receive_refusal'
]
STATE_DIM = len(BASE_FEATURES) + len(TRACKED_ACTIVITIES)  # 16

INTERVENTION_INFO = {
    "name": ["choose_procedure", "time_contact_HQ", "set_ir_3_levels"],
    "actions": [["start_standard", "start_priority"], ["contact_headquarters", "skip_contact"], [0.07, 0.08, 0.09]],
    "action_width": [2, 2, 3],
    "activities": [["start_standard", "start_priority"], ["contact_headquarters", "skip_contact"], ["calculate_offer"]],
}
