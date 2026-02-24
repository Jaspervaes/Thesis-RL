data_folder = "data"
results_folder = "res"
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(parent_dir)
from config.config import path
sys.path.append(path)
sys.path.append(path + "\\SimBank")

import pandas as pd
import numpy as np
from copy import deepcopy
from src.utils.inference import Forward_pass
import random
from SimBank.activity_execution import ActivityExecutioner
from SimBank import simulation
import d3rlpy

val_multiplier = 1000

class CQLModelEvaluator(Forward_pass):
    """
    CQL Model Evaluator for SimBank
    Integrates CQL (Conservative Q-Learning) models with SimBank's evaluation framework
    Supports both single and sequential interventions
    """
    def __init__(self, cql_models, int_dataset_params, full_dataset_params, print_cases=False):
        """
        Initialize CQL evaluator

        Args:
            cql_models: Single d3rlpy model OR list of models for sequential interventions
            int_dataset_params: List of intervention-specific params
            full_dataset_params: Full dataset parameters
            print_cases: Enable detailed debugging output
        """
        super().__init__()

        # Handle single or multiple models
        if isinstance(cql_models, list):
            self.cql_models = cql_models
        else:
            self.cql_models = [cql_models]  # Single model as list for consistency

        self.int_dataset_params = int_dataset_params
        self.full_dataset_params = full_dataset_params
        self.print_cases = print_cases
        random.seed(full_dataset_params["random_seed_test"])
        np.random.seed(full_dataset_params["random_seed_test"])

    def extract_features(self, state_dict):
        """
        Convert state dict to feature vector for CQL model
        Must match the features used during training
        """
        return np.array([
            state_dict.get('amount', 0),
            state_dict.get('est_quality', 0),
            state_dict.get('unc_quality', 0),
            state_dict.get('interest_rate', 0) if not np.isnan(state_dict.get('interest_rate', 0)) else 0,
            state_dict.get('cum_cost', 0),
        ], dtype=np.float32)

    def get_bank_best_action(self, prefix_list, current_int_index):
        """Bank's default policy for interventions"""
        prefix_without_int = prefix_list[0][0:-1]
        prev_event = prefix_without_int[-1]
        action_index = 0

        if self.full_dataset_params["intervention_info"]["name"][current_int_index] == "time_contact_HQ":
            cancel_condition = ((prev_event["unc_quality"] == 0 and prev_event["est_quality"] < self.full_dataset_params["policies_info"]["min_quality"] and prev_event["noc"] >= self.full_dataset_params["policies_info"]["max_noc"]) or (prev_event["noc"] >= self.full_dataset_params["policies_info"]["max_noc"] and prev_event["unc_quality"] > 0))
            contact_condition = (prev_event["noc"] < 2 and prev_event["unc_quality"] == 0 and prev_event["amount"] > 10000 and prev_event["est_quality"] >= self.full_dataset_params["policies_info"]["min_quality"])

            # Per dataset_params: action 0 = contact_headquarters, action 1 = skip_contact
            if cancel_condition:
                action_index = 1  # skip_contact (leads to cancellation)
            elif contact_condition:
                action_index = 0  # contact_headquarters

        elif self.full_dataset_params["intervention_info"]["name"][current_int_index] == "choose_procedure":
            priority_condition = (prev_event["amount"] > self.full_dataset_params["policies_info"]["choose_procedure"]["amount"] and prev_event["est_quality"] >= self.full_dataset_params["policies_info"]["choose_procedure"]["est_quality"])

            if priority_condition:
                action_index = 1
            else:
                action_index = 0

        elif self.full_dataset_params["intervention_info"]["name"][current_int_index] == "set_ir_3_levels":
            activity_executioner = ActivityExecutioner()
            ir, _, _ = activity_executioner.calculate_offer(prev_event=prev_event, intervention_info=self.full_dataset_params["intervention_info"])
            action_index = self.full_dataset_params["intervention_info"]["actions"][current_int_index].index(ir)

        return action_index

    def bank_policy_inference(self, n_cases, validation=False):
        """Run bank policy through simulator and measure performance"""
        bank_performance = 0
        full_dataset_params = deepcopy(self.full_dataset_params)
        if validation:
            full_dataset_params["random_seed_test"] = full_dataset_params["random_seed_test"]*val_multiplier

        case_gen = simulation.PresProcessGenerator(self.full_dataset_params, seed=full_dataset_params["random_seed_test"])

        for case_nr in range(n_cases):
            prefix_list = []
            prefix_list = case_gen.start_simulation_inference(seed_to_add=case_nr)
            while case_gen.int_points_available:
                bank_best_action = self.get_bank_best_action(prefix_list, case_gen.current_int_index)
                prefix_list = case_gen.continue_simulation_inference(bank_best_action)

                if self.print_cases:
                    print("Bank best action", bank_best_action, '\n')

            full_case = case_gen.end_simulation_inference()
            full_case = pd.DataFrame(full_case)
            bank_performance += full_case["outcome"].iloc[-1]

            if self.print_cases:
                print("Bank full case", full_case, "\n")

        return bank_performance

    def random_policy_inference(self, n_cases, iteration=0, validation=False):
        """Run random policy through simulator and measure performance"""
        random_performance = 0
        full_dataset_params = deepcopy(self.full_dataset_params)
        if validation:
            full_dataset_params["random_seed_test"] = full_dataset_params["random_seed_test"]*val_multiplier

        case_gen = simulation.PresProcessGenerator(self.full_dataset_params, seed=full_dataset_params["random_seed_test"])

        # Use different random seed for each iteration (like CI method)
        random_object_for_random_policy = random.Random(self.full_dataset_params["random_seed_test"] + 5*iteration)

        for case_nr in range(n_cases):
            prefix_list = []
            prefix_list = case_gen.start_simulation_inference(seed_to_add=case_nr)
            timings = [0] * len(self.full_dataset_params["intervention_info"]["name"])
            random_best_timings = []
            for int_index in range(len(self.full_dataset_params["intervention_info"]["name"])):
                if self.full_dataset_params["intervention_info"]["name"][int_index] == "time_contact_HQ":
                    random_best_timings.append(random_object_for_random_policy.choice(range(self.full_dataset_params["intervention_info"]["action_depth"][int_index])) * 2)
                else:
                    random_best_timings.append(0)
            random_best_action = 0 # control
            while case_gen.int_points_available:
                if timings[case_gen.current_int_index] == random_best_timings[case_gen.current_int_index]:
                    random_best_action = random_object_for_random_policy.choice(range(self.full_dataset_params["intervention_info"]["action_width"][case_gen.current_int_index]))
                timings[case_gen.current_int_index] += 1
                prefix_list = case_gen.continue_simulation_inference(random_best_action)

                if self.print_cases:
                    print("Random best action", random_best_action, '\n')

            full_case = case_gen.end_simulation_inference()
            full_case = pd.DataFrame(full_case)
            random_performance += full_case["outcome"].iloc[-1]

            if self.print_cases:
                print("Full case", full_case, "\n")

        return random_performance

    def model_policy_inference(self, n_cases, iteration=0, validation=False):
        """Run CQL policy through simulator and measure performance (supports sequential)"""
        cql_performance = 0
        full_dataset_params = deepcopy(self.full_dataset_params)
        if validation:
            full_dataset_params["random_seed_test"] = full_dataset_params["random_seed_test"]*val_multiplier

        case_gen = simulation.PresProcessGenerator(self.full_dataset_params, seed=full_dataset_params["random_seed_test"])

        action_counts = {}
        for int_index in range(len(self.full_dataset_params["intervention_info"]["name"])):
            action_counts[int_index] = {}
            for action_index in range(self.full_dataset_params["intervention_info"]["action_width"][int_index]):
                action_counts[int_index][action_index] = 0

        for case_nr in range(n_cases):
            prefix_list = []
            prefix_list = case_gen.start_simulation_inference(seed_to_add=case_nr)

            while case_gen.int_points_available:
                # Get current intervention index (for sequential interventions)
                current_int_index = case_gen.current_int_index

                # Get current state from last event
                prefix_without_int = prefix_list[0][0:-1]
                prev_event = prefix_without_int[-1]

                # Extract features and predict action with appropriate CQL model
                state_features = self.extract_features(prev_event)
                state_features = state_features.reshape(1, -1)  # Shape for model

                # Get CQL action from model for this intervention
                cql_action = self.cql_models[current_int_index].predict(state_features)[0]

                action_counts[current_int_index][cql_action] += 1

                # Continue simulation with CQL action
                prefix_list = case_gen.continue_simulation_inference(cql_action)

                if self.print_cases:
                    int_name = self.full_dataset_params["intervention_info"]["name"][current_int_index]
                    print(f"Intervention {current_int_index} ({int_name}): CQL action={cql_action}, state={state_features}", '\n')

            full_case = case_gen.end_simulation_inference()
            full_case = pd.DataFrame(full_case)
            cql_performance += full_case["outcome"].iloc[-1]

            if self.print_cases:
                print("CQL full case outcome:", full_case["outcome"].iloc[-1], "\n")

        if self.print_cases:
            print(f"CQL Action distribution: {action_counts}")

        return cql_performance

    def evaluate_all_policies(self, n_cases, validation=False):
        """
        Evaluate all policies (Bank, Random, CQL) and return comparison metrics
        Following BRANCHI/RL methodology: single run for each policy

        Args:
            n_cases: Number of test cases
            validation: Use validation seed multiplier
        """
        print("\n" + "="*70)
        print("EVALUATING ALL POLICIES ON SIMULATOR")
        print("="*70)

        print(f"\nEvaluating {n_cases} cases...")

        # Run bank policy (once - deterministic)
        print("\n[1/3] Running Bank Policy...")
        bank_perf = self.bank_policy_inference(n_cases, validation)
        bank_avg = bank_perf / n_cases
        print(f"  Bank Policy: Total={bank_perf:.2f}, Avg={bank_avg:.2f}")

        # Run random policy (once)
        print("\n[2/3] Running Random Policy...")
        random_perf = self.random_policy_inference(n_cases, iteration=0, validation=validation)
        random_avg = random_perf / n_cases
        print(f"  Random Policy: Total={random_perf:.2f}, Avg={random_avg:.2f}")

        # Run CQL policy (once)
        print("\n[3/3] Running CQL Policy...")
        cql_perf = self.model_policy_inference(n_cases, iteration=0, validation=validation)
        cql_avg = cql_perf / n_cases
        print(f"  CQL Policy: Total={cql_perf:.2f}, Avg={cql_avg:.2f}")

        # Calculate gains
        print("\n" + "="*70)
        print("PERFORMANCE COMPARISON")
        print("="*70)

        print(f"\n{'Policy':<15} {'Total':<15} {'Average':<15} {'Gain vs Bank':<15}")
        print("-" * 70)
        print(f"{'Bank':<15} {bank_perf:<15.2f} {bank_avg:<15.2f} {'-':<15}")
        print(f"{'Random':<15} {random_perf:<15.2f} {random_avg:<15.2f} {((random_avg/bank_avg - 1)*100):<15.2f}%")
        print(f"{'CQL':<15} {cql_perf:<15.2f} {cql_avg:<15.2f} {((cql_avg/bank_avg - 1)*100):<15.2f}%")

        results = {
            'bank': {'total': bank_perf, 'avg': bank_avg},
            'random': {
                'total': random_perf,
                'avg': random_avg,
                'gain': (random_avg/bank_avg - 1)*100
            },
            'cql': {
                'total': cql_perf,
                'avg': cql_avg,
                'gain': (cql_avg/bank_avg - 1)*100
            }
        }

        return results
