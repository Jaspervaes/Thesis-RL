"""SubsetPolicy: wrap any RL policy so it only controls a subset of interventions."""
from shared.evaluation import bank_policy


class SubsetPolicy:
    """Wraps any policy, replacing non-active interventions with bank_policy.

    The wrapped base_policy must be a steps=3-trained policy with Q-networks for
    every intervention (so any subset of {0,1,2} is callable). For interventions
    not in `active`, falls back to bank_policy.
    """

    def __init__(self, base_policy, active_interventions):
        self.base_policy = base_policy
        self.active = set(active_interventions)

    def __call__(self, prev_event, int_idx, prefix=None):
        if int_idx not in self.active:
            return bank_policy(prev_event, int_idx)
        return self.base_policy(prev_event, int_idx, prefix)

    def reset(self):
        if hasattr(self.base_policy, 'reset'):
            self.base_policy.reset()
