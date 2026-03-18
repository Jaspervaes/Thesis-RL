"""Learned SimBank environment wrapping P_T (processing time) and P_C (control flow) models."""
import copy
import math
import numpy as np
import torch

from shared import FEATURE_COLS, bank_policy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# From SimBank/activity_execution.py
COSTS = {
    "initiate_application": 0,
    "start_standard": 10,
    "start_priority": 5000,
    "validate_application": 20,
    "contact_headquarters": 3000,
    "skip_contact": 0,
    "email_customer": 10,
    "call_customer": 20,
    "calculate_offer": 400,
    "cancel_application": 30,
    "receive_acceptance": 10,
    "receive_refusal": 10,
    "stop_application": 0,
}

TERMINAL_ACTIVITIES = {'cancel_application', 'receive_acceptance', 'stop_application'}

INTERVENTION_ACTIONS = {
    0: {0: 'start_standard', 1: 'start_priority'},
    1: {0: 'contact_headquarters', 1: 'skip_contact'},
    2: 'calculate_offer',
}

IR_LEVELS = [0.07, 0.08, 0.09]

FIXED_COST = 100
LOAN_LENGTH = 10


class LearnedSimBankEnv:
    """Gymnasium-style learned environment for SimBank."""

    def __init__(self, sim_artifact, steps=3):
        from rims.convert_data import ProcessingTimeModel, ControlFlowModel

        art = sim_artifact
        self.activity_to_idx = art['activity_to_idx']
        self.idx_to_activity = art['idx_to_activity']
        self.feat_means = art['feat_means']
        self.feat_stds = art['feat_stds']
        self.max_len = art['max_len']
        self.n_activities = art['n_activities']
        self.initial_prefixes = art['initial_prefixes']
        self.transition_mask = art.get('transition_mask', {})
        self.acceptance_model = art.get('acceptance_model')
        self.steps = steps

        n_features = len(FEATURE_COLS)

        self.pt_model = ProcessingTimeModel(self.n_activities, n_features).to(device)
        self.pt_model.load_state_dict(art['pt_state_dict'])
        self.pt_model.eval()

        self.pc_model = ControlFlowModel(self.n_activities, n_features).to(device)
        self.pc_model.load_state_dict(art['pc_state_dict'])
        self.pc_model.eval()

        self.n_actions = [2, 2, 3]
        self.prefix = None
        self.int_idx = 0

    def _encode_prefix(self, prefix):
        """Encode prefix for model input."""
        seq_len = min(len(prefix), self.max_len)
        seq_len = max(seq_len, 1)

        acts = np.zeros((1, self.max_len), dtype=np.int64)
        feats = np.zeros((1, self.max_len, len(FEATURE_COLS)), dtype=np.float32)

        for j, e in enumerate(prefix[:seq_len]):
            acts[0, j] = self.activity_to_idx.get(e.get('activity', ''), 0)
            for k, col in enumerate(FEATURE_COLS):
                val = e.get(col, 0)
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    val = 0.0
                if not np.isfinite(val):
                    val = 0.0
                feats[0, j, k] = (val - self.feat_means[col]) / self.feat_stds[col]

        return (torch.LongTensor(acts).to(device),
                torch.FloatTensor(feats).to(device),
                torch.LongTensor([seq_len]))

    def _predict_duration(self, prefix):
        """Predict duration in seconds using P_T."""
        with torch.no_grad():
            log_dur = self.pt_model(*self._encode_prefix(prefix)).item()
        return max(np.exp(log_dur) - 1, 0)

    def _predict_next_activity(self, prefix):
        """Sample next activity using P_C, masked by mined transition matrix."""
        last_activity = prefix[-1].get('activity', '') if prefix else ''
        last_idx = self.activity_to_idx.get(last_activity, 0)

        with torch.no_grad():
            logits = self.pc_model(*self._encode_prefix(prefix))
            logits = logits.squeeze(0).cpu().numpy()

        # Apply transition mask: only allow observed successors
        valid_idxs = self.transition_mask.get(last_idx)
        if valid_idxs:
            mask = np.full(len(logits), -1e9)
            for vi in valid_idxs:
                if vi < len(mask):
                    mask[vi] = 0.0
            logits = logits + mask

        probs = np.exp(logits - logits.max())
        if not np.all(np.isfinite(probs)) or probs.sum() == 0:
            if valid_idxs:
                probs = np.zeros(len(logits))
                for vi in valid_idxs:
                    if vi < len(probs):
                        probs[vi] = 1.0
            else:
                probs = np.ones(len(logits))
        probs = probs / probs.sum()
        idx = np.random.choice(len(probs), p=probs)
        return self.idx_to_activity.get(idx, '')

    def _predict_post_offer(self, event):
        """Predict accept/refuse/cancel after calculate_offer using mined acceptance model."""
        ir = float(event.get('interest_rate', 0))
        min_ir = float(event.get('min_interest_rate', 0))

        # Mined cancellation rule: if ir < min_ir, cancel
        if np.isfinite(min_ir) and ir < min_ir:
            return 'cancel_application'

        if self.acceptance_model is None:
            return self._predict_next_activity(self.prefix)

        am = self.acceptance_model
        amount = float(event.get('amount', 0))
        elapsed = float(event.get('elapsed_time', 0))
        if not np.isfinite(amount): amount = 0.0
        if not np.isfinite(elapsed): elapsed = 0.0
        x = np.array([[ir, min_ir, amount, elapsed]])
        x_scaled = (x - np.array(am['scaler_mean'])) / np.array(am['scaler_scale'])
        logit = float(np.dot(x_scaled, np.array(am['coef']).T) + np.array(am['intercept']))
        accept_prob = 1.0 / (1.0 + np.exp(-logit))
        if np.random.random() < accept_prob:
            return 'receive_acceptance'
        else:
            return 'receive_refusal'

    def _make_event(self, activity, prev_event, duration_seconds):
        """Create a new event dict with deterministic feature updates."""
        event = {
            'activity': activity,
            'amount': prev_event['amount'],
            'quality': prev_event.get('quality', prev_event.get('est_quality', 5)),
            'est_quality': prev_event['est_quality'],
            'unc_quality': prev_event['unc_quality'],
            'interest_rate': prev_event.get('interest_rate', 0),
            'discount_factor': prev_event.get('discount_factor', float('nan')),
            'noc': prev_event.get('noc', 0),
            'nor': prev_event.get('nor', 0),
            'min_interest_rate': prev_event.get('min_interest_rate', float('nan')),
            'cum_cost': prev_event.get('cum_cost', 0) + COSTS.get(activity, 0),
            'elapsed_time': prev_event.get('elapsed_time', 0) + duration_seconds / 86400,
            'outcome': float('nan'),
        }

        if activity == 'contact_headquarters':
            # HQ cost depends on uncertainty: unc_quality * 1000 + 1000
            hq_cost = prev_event['unc_quality'] * 1000 + 1000
            event['cum_cost'] = prev_event.get('cum_cost', 0) + hq_cost
            # HQ reveals true quality: uncertainty eliminated
            event['unc_quality'] = 0
            event['est_quality'] = prev_event.get('quality', prev_event.get('est_quality', 5))

        elif activity in ('call_customer', 'email_customer'):
            decrease = 3 if activity == 'call_customer' else 2
            event['unc_quality'] = max(prev_event['unc_quality'] - decrease, 0)
            event['noc'] = prev_event.get('noc', 0) + 1
            # Re-estimate quality based on new uncertainty
            quality = prev_event.get('quality', prev_event.get('est_quality', 5))
            unc = event['unc_quality']
            # Simplified: est_quality converges to true quality as unc decreases
            if unc == 0:
                event['est_quality'] = quality
            else:
                event['est_quality'] = int(np.clip(
                    np.random.normal(quality, unc / 3), 0, 10
                ))

        elif activity == 'calculate_offer':
            # interest_rate and discount_factor set by the agent action (handled in step())
            pass

        elif activity == 'receive_acceptance':
            event['outcome'] = self._calc_outcome(event, accepted=True)

        elif activity == 'cancel_application':
            event['outcome'] = self._calc_outcome(event, accepted=False)

        elif activity == 'receive_refusal':
            event['nor'] = prev_event.get('nor', 0) + 1

        return event

    def _calc_outcome(self, event, accepted):
        """Calculate reward (from activity_execution.py:calc_outcome)."""
        if accepted:
            ir = event['interest_rate']
            amount = event['amount']
            quality = event.get('quality', event.get('est_quality', 5))
            risk_factor = (10 - quality) / 200
            df = 0.03 + risk_factor
            future_earnings = amount * (1 + ir) ** LOAN_LENGTH
            discounted = future_earnings / (1 + df) ** LOAN_LENGTH
            return discounted - event['cum_cost'] - amount - FIXED_COST
        else:
            return -event['cum_cost'] - FIXED_COST

    def reset(self):
        """Sample a random initial prefix, return (prefix, info)."""
        idx = np.random.randint(len(self.initial_prefixes))
        self.prefix = copy.deepcopy(self.initial_prefixes[idx])
        self.int_idx = 0
        return self.prefix, {'int_idx': 0}

    def step(self, action):
        """Apply action at current intervention, roll forward to next intervention or terminal.

        Returns: (prefix, reward, done, truncated, info)
        """
        prev_event = self.prefix[-1]
        int_idx = self.int_idx

        # Map action to activity and apply
        if int_idx == 0:
            activity = INTERVENTION_ACTIONS[0][action]
        elif int_idx == 1:
            activity = INTERVENTION_ACTIONS[1][action]
        else:
            activity = 'calculate_offer'

        duration = self._predict_duration(self.prefix)
        event = self._make_event(activity, prev_event, duration)

        # Special handling for calculate_offer: set interest rate from action
        if int_idx == 2:
            ir = IR_LEVELS[action]
            event['interest_rate'] = ir
            # Compute min_interest_rate and discount_factor
            risk_factor = (10 - prev_event['est_quality']) / 200
            df_rate = 0.03 + risk_factor
            best_case_costs = event['cum_cost'] + COSTS['receive_acceptance'] + FIXED_COST
            min_ir = ((best_case_costs / prev_event['amount'] + 1) ** (1 / LOAN_LENGTH)) * (1 + df_rate) - 1
            min_ir = math.ceil(min_ir * 100) / 100
            event['min_interest_rate'] = min_ir
            event['discount_factor'] = df_rate

        self.prefix.append(event)

        # Roll forward with P_C / P_T until next intervention or terminal
        done, reward, truncated = False, 0.0, False
        next_int_idx = int_idx + 1

        for _ in range(20):  # safety limit
            last = self.prefix[-1]
            last_activity = last.get('activity', '')

            # Check if terminal
            if last_activity in TERMINAL_ACTIVITIES:
                done = True
                outcome = last.get('outcome', float('nan'))
                if not np.isnan(outcome):
                    reward = outcome
                else:
                    reward = self._calc_outcome(last, accepted=False)
                break

            # Predict next activity: use acceptance model after calculate_offer
            if last_activity == 'calculate_offer':
                next_act = self._predict_post_offer(last)
            else:
                next_act = self._predict_next_activity(self.prefix)
            if not next_act or next_act == '':
                done = True
                reward = self._calc_outcome(last, accepted=False)
                break

            # Check if next activity is the expected next intervention
            is_intervention = False
            if next_int_idx <= 2:
                if next_int_idx == 0 and next_act in ('start_standard', 'start_priority'):
                    is_intervention = True
                elif next_int_idx == 1 and next_act in ('contact_headquarters', 'skip_contact'):
                    is_intervention = True
                elif next_int_idx == 2 and next_act == 'calculate_offer':
                    is_intervention = True

            if is_intervention:
                break

            # Auto-generate non-intervention event
            dur = self._predict_duration(self.prefix)
            new_event = self._make_event(next_act, last, dur)
            self.prefix.append(new_event)

            # receive_refusal loops back to calculate_offer (intervention 2)
            if next_act == 'receive_refusal':
                next_int_idx = 2
                break

            # Check terminal
            if next_act in TERMINAL_ACTIVITIES:
                done = True
                outcome = new_event.get('outcome', float('nan'))
                if not np.isnan(outcome):
                    reward = outcome
                else:
                    reward = self._calc_outcome(new_event, accepted=False)
                break

        if not done:
            # Check if next intervention is beyond steps limit → use bank policy
            if next_int_idx >= self.steps:
                # Auto-complete remaining interventions with bank policy
                reward, done = self._auto_complete_bank(next_int_idx)
            else:
                self.int_idx = next_int_idx

        info = {'int_idx': self.int_idx}
        return self.prefix, reward, done, truncated, info

    def _auto_complete_bank(self, start_int_idx):
        """Auto-complete remaining interventions using bank policy."""
        int_idx = start_int_idx
        max_offers = 5  # cap refusal loops

        while int_idx <= 2 and max_offers > 0:
            prev_event = self.prefix[-1]
            action = bank_policy(prev_event, int_idx)

            if int_idx == 0:
                activity = INTERVENTION_ACTIONS[0][action]
            elif int_idx == 1:
                activity = INTERVENTION_ACTIONS[1][action]
            else:
                activity = 'calculate_offer'

            dur = self._predict_duration(self.prefix)
            event = self._make_event(activity, prev_event, dur)

            if int_idx == 2:
                ir = IR_LEVELS[action]
                event['interest_rate'] = ir
                risk_factor = (10 - prev_event['est_quality']) / 200
                df_rate = 0.03 + risk_factor
                best_case_costs = event['cum_cost'] + COSTS['receive_acceptance'] + FIXED_COST
                min_ir = ((best_case_costs / prev_event['amount'] + 1) ** (1 / LOAN_LENGTH)) * (1 + df_rate) - 1
                min_ir = math.ceil(min_ir * 100) / 100
                event['min_interest_rate'] = min_ir
                event['discount_factor'] = df_rate

            self.prefix.append(event)
            int_idx += 1

            # Roll forward to next intervention or terminal
            for _ in range(20):
                last = self.prefix[-1]
                if last.get('activity', '') in TERMINAL_ACTIVITIES:
                    outcome = last.get('outcome', float('nan'))
                    if not np.isnan(outcome):
                        return outcome, True
                    return self._calc_outcome(last, accepted=False), True

                last_activity = last.get('activity', '')
                if last_activity == 'calculate_offer':
                    next_act = self._predict_post_offer(last)
                else:
                    next_act = self._predict_next_activity(self.prefix)
                if not next_act:
                    return self._calc_outcome(last, accepted=False), True

                # Check if next activity is an intervention we still need to handle
                is_next_intervention = False
                if int_idx <= 2:
                    if int_idx == 1 and next_act in ('contact_headquarters', 'skip_contact'):
                        is_next_intervention = True
                    elif int_idx == 2 and next_act == 'calculate_offer':
                        is_next_intervention = True

                if is_next_intervention:
                    break

                dur = self._predict_duration(self.prefix)
                new_event = self._make_event(next_act, last, dur)
                self.prefix.append(new_event)

                # receive_refusal loops back to calculate_offer (intervention 2)
                if next_act == 'receive_refusal':
                    int_idx = 2
                    max_offers -= 1
                    break

                if next_act in TERMINAL_ACTIVITIES:
                    outcome = new_event.get('outcome', float('nan'))
                    if not np.isnan(outcome):
                        return outcome, True
                    return self._calc_outcome(new_event, accepted=False), True

        # If we exhausted all interventions without terminal, force terminal
        last = self.prefix[-1]
        return self._calc_outcome(last, accepted=False), True
