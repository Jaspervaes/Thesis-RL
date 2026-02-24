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
import math
from copy import deepcopy
from src.utils.inference import Forward_pass
import random
from SimBank.activity_execution import ActivityExecutioner
from SimBank.extra_flow_conditions import ExtraFlowConditioner
from SimBank import simulation
import src.methods.CI.CI_data_preparation as CI_data_preparation
import src.methods.BOZORGI.BOZORGI_data_preparation as BOZORGI_data_preparation

val_multiplier = 1000

class CIModelEvaluator(Forward_pass):
    def __init__(self, model_params, int_dataset_params, full_dataset_params, prep_utils, print_cases=False, realcause_model_list=None, realcause_prep_utils=None):
        super().__init__()
        self.device = "cpu"
        self.int_dataset_params = int_dataset_params
        self.full_dataset_params = full_dataset_params
        self.prep_utils = prep_utils
        self.print_cases = print_cases
        self.realcause_model_list = realcause_model_list
        self.realcause_prep_utils = realcause_prep_utils
        random.seed(full_dataset_params["random_seed_test"])
        np.random.seed(full_dataset_params["random_seed_test"])
    
    def get_bank_best_action(self, prefix_list, current_int_index):
        prefix_without_int = prefix_list[0][0:-1]
        prev_event = prefix_without_int[-1]
        action_index = 0
        
        if self.full_dataset_params["intervention_info"]["name"][current_int_index] == "time_contact_HQ":
            cancel_condition = ((prev_event["unc_quality"] == 0 and prev_event["est_quality"] < self.full_dataset_params["policies_info"]["min_quality"] and prev_event["noc"] >= self.full_dataset_params["policies_info"]["max_noc"]) or (prev_event["noc"] >= self.full_dataset_params["policies_info"]["max_noc"] and prev_event["unc_quality"] > 0))
            contact_condition = (prev_event["noc"] < 2 and prev_event["unc_quality"] == 0 and prev_event["amount"] > 10000 and prev_event["est_quality"] >= self.full_dataset_params["policies_info"]["min_quality"])

            if cancel_condition:
                action_index = 0
            elif contact_condition:
                action_index = 1
        
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

    def get_model_best_action(self, prefix_list, case_prep_list, current_int_index, opt_th):
        if self.full_dataset_params["intervention_info"]["name"][current_int_index] == "time_contact_HQ":
            prefix_without_int = prefix_list[0][0:-1]
            prev_event = prefix_without_int[-1]
            if prev_event["activity"] == "validate_application": # only execute after contact customer
                return 0
            
        outcome_scaler = self.prep_utils[current_int_index][self.key]["scaler_dict_train"]["outcome"]

        if self.full_dataset_params["intervention_info"]["action_depth"][current_int_index] == 1:
            model_best_y = -100000000
            model_best_action = -1
            for action_index, prefix in enumerate(prefix_list):
                # Preprocess (with correct prep_utils)
                prefix = pd.DataFrame(prefix)
                prep_prefix = case_prep_list[current_int_index].preprocess_sample_CI(prefix, self.prep_utils[current_int_index][self.key], self.device)
                [t, _, x_case, x_event, _, _, _, _] = prep_prefix
                # Predict using model
                y_pred_norm = self.forward_pass(self.model_list[current_int_index].model, x_case, x_event, t, device=self.device)
                y_pred = outcome_scaler.inverse_transform(y_pred_norm.reshape(-1, 1).detach().cpu())
                # Take max predicion as action
                if y_pred > model_best_y:
                    model_best_y = y_pred
                    model_best_action = action_index
        else:
            # Define control
            control_prefix = prefix_list[0]
            control_prefix = pd.DataFrame(control_prefix)
            control_prep_prefix = case_prep_list[current_int_index].preprocess_sample_CI(control_prefix, self.prep_utils[current_int_index][self.key], self.device)
            [control_t, _, control_x_case, control_x_event, _, _, _, _] = control_prep_prefix
            control_y_pred_norm = self.forward_pass(self.model_list[current_int_index].model, control_x_case, control_x_event, control_t, device=self.device)
            control_y_pred = outcome_scaler.inverse_transform(control_y_pred_norm.reshape(-1, 1).detach().cpu())
            
            model_best_y = control_y_pred
            model_best_action = 0
            for action_index, prefix in enumerate(prefix_list):
                if action_index == 0:
                    continue
                # Preprocess (with correct prep_utils)
                prefix = pd.DataFrame(prefix)
                prep_prefix = case_prep_list[current_int_index].preprocess_sample_CI(prefix, self.prep_utils[current_int_index][self.key], self.device)
                [t, _, x_case, x_event, _, _, _, _] = prep_prefix
                # Predict using model
                y_pred_norm = self.forward_pass(self.model_list[current_int_index].model, x_case, x_event, t, device=self.device)
                y_pred = outcome_scaler.inverse_transform(y_pred_norm.reshape(-1, 1).detach().cpu())
                y_diff = y_pred - control_y_pred
                self.current_c_y_preds.append(control_y_pred[0][0])
                self.current_t_y_preds.append(y_pred[0][0])
                self.current_y_diffs.append(y_diff[0][0])
                if y_diff > opt_th:
                    model_best_y = y_pred
                    model_best_action = action_index
                if y_diff > self.max_y_diff:
                    self.max_y_diff = y_diff

        return model_best_action

    def bank_policy_inference(self, n_cases):
        self.ExtraFlowConditioner = ExtraFlowConditioner()

        # GENERATING BANK POLICY ONLINE TO MAINTAIN SAME TEST SET AS MODEL

        #Init performance metrics
        bank_performance = 0

        #Init data generator
        case_gen = simulation.PresProcessGenerator(self.full_dataset_params, seed=self.full_dataset_params["random_seed_test"])

        #Run
        for case_nr in range(n_cases):
            prefix_list = []
            prefix_list = case_gen.start_simulation_inference(seed_to_add=case_nr)
            while case_gen.int_points_available:
                bank_best_action = self.get_bank_best_action(prefix_list, case_gen.current_int_index)
                # Break if intervention done or in last timing
                prefix_list = case_gen.continue_simulation_inference(bank_best_action)

                if self.print_cases:
                    print("Bank best action", bank_best_action, '\n')

            full_case = case_gen.end_simulation_inference()
            full_case = pd.DataFrame(full_case)
            bank_performance += full_case["outcome"].iloc[-1]

            if self.print_cases:
                print("Full case", full_case, "\n")
        
        return bank_performance
    
    def random_policy_inference(self, n_cases, iteration=0):
        #Init performance metrics
        random_performance = 0

        #Init data generator
        case_gen = simulation.PresProcessGenerator(self.full_dataset_params, seed=self.full_dataset_params["random_seed_test"])

        random_object_for_random_policy = random.Random(self.full_dataset_params["random_seed_test"] + 5*iteration)

        #Run
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
                # Break if intervention done or in last timing
                prefix_list = case_gen.continue_simulation_inference(random_best_action)

                if self.print_cases:
                    print("Random best action", random_best_action, '\n')

            full_case = case_gen.end_simulation_inference()
            full_case = pd.DataFrame(full_case)
            random_performance += full_case["outcome"].iloc[-1]

            if self.print_cases:
                print("Full case", full_case, "\n")
        
        return random_performance

    def model_policy_inference(self, n_cases, model_list, key, opt_th, iteration, tuning=False, calculate_realcause_performance=False):
        #Init performance metrics
        self.max_y_diff = 0
        self.model_list = model_list
        self.key = key
        self.iteration = iteration
        full_dataset_params = deepcopy(self.full_dataset_params)
        self.model_action_timings = {0: 0, 2: 0, 4: 0, 6: 0, 8: 0, 100: 0}
        self.model_actions = {}
        self.current_c_y_preds = ["x"]
        self.current_t_y_preds = ["x"]
        self.current_y_diffs = ["x"]
        if tuning:
            full_dataset_params["random_seed_test"] = full_dataset_params["random_seed_test"]*val_multiplier
            self.device = "cuda"
        else:
            self.device = "cpu"
            self.test_set_df = pd.DataFrame([])
        model_performance = 0
        model_performance_realcause = 0

        #Init data generator and preprocessors
        case_gen = simulation.PresProcessGenerator(full_dataset_params, seed=full_dataset_params["random_seed_test"])
        case_prep_list = []
        case_prep_list_realcause = []
        for int in range(len(self.full_dataset_params["intervention_info"]["name"])):
            case_prep_list.append(CI_data_preparation.LoanProcessPreprocessor(dataset_params=self.int_dataset_params[int], time_wise=self.full_dataset_params["time_wise"]))
            case_prep_list_realcause.append(BOZORGI_data_preparation.LoanProcessPreprocessor(dataset_params=self.int_dataset_params[int]))
        
        #Run
        for case_nr in range(n_cases):
            if case_nr % 500 == 0:
                print("Case nr: ", case_nr)
            int_executed = False
            int_point_available = False
            realcause_outcome_list = []
            self.current_timing = 0
            prefix_list = []
            prefix_list = case_gen.start_simulation_inference(seed_to_add=case_nr)
            while case_gen.int_points_available:
                int_point_available = True
                if self.full_dataset_params["intervention_info"]["name"] == ["time_contact_HQ"] and self.current_timing % 2 != 0:
                    model_best_action = 0
                    if self.print_cases:
                        print("not intervention point for time_contact_HQ (validate_application) (model_policy_inference)")
                else:
                    model_best_action = self.get_model_best_action(prefix_list, case_prep_list, case_gen.current_int_index, opt_th=opt_th)

                if model_best_action == 1:
                    self.model_action_timings[self.current_timing] += 1
                    int_executed = True
                if model_best_action not in self.model_actions.keys():
                    self.model_actions[model_best_action] = 0
                self.model_actions[model_best_action] += 1

                if not tuning and calculate_realcause_performance:
                    if (model_best_action == 1 or self.current_timing == 6) and self.full_dataset_params["intervention_info"]["name"][case_gen.current_int_index] == "time_contact_HQ":
                        realcause_outcome_list = self.get_realcause_outcome(prefix_list[model_best_action], case_prep_list_realcause[case_gen.current_int_index], self.realcause_prep_utils[case_gen.current_int_index], model_best_action, iteration)

                prefix_list = case_gen.continue_simulation_inference(model_best_action)
                if self.full_dataset_params["intervention_info"]["name"] == ["time_contact_HQ"]:
                    self.current_timing += 1

                if self.print_cases:
                    print("Model best action", model_best_action, '\n')

            if (not int_executed) and int_point_available:
                self.model_action_timings[8] += 1
            if not int_point_available:
                self.model_action_timings[100] += 1

            full_case = case_gen.end_simulation_inference()
            full_case = pd.DataFrame(full_case)
            model_performance += full_case["outcome"].iloc[-1]
            if not tuning and calculate_realcause_performance:
                if len(realcause_outcome_list) > 0:
                    model_performance_realcause += realcause_outcome_list[0]
                else:
                    model_performance_realcause += full_case["outcome"].iloc[-1]

            if not tuning:
                if self.full_dataset_params["intervention_info"]["name"] == ["time_contact_HQ"]:
                    # Make sure they are the same length, add "x" if not, and add to full case
                    if len(self.current_c_y_preds) < len(full_case):
                        self.current_c_y_preds += ["x"] * (len(full_case) - len(self.current_c_y_preds))
                        self.current_t_y_preds += ["x"] * (len(full_case) - len(self.current_t_y_preds))
                        self.current_y_diffs += ["x"] * (len(full_case) - len(self.current_y_diffs))

                    full_case["c_y_pred"] = deepcopy(self.current_c_y_preds)
                    full_case["t_y_pred"] = deepcopy(self.current_t_y_preds)
                    full_case["y_diff"] = deepcopy(self.current_y_diffs)

                    self.current_c_y_preds = ["x"]
                    self.current_t_y_preds = ["x"]
                    self.current_y_diffs = ["x"]
                  
                self.test_set_df = pd.concat([self.test_set_df, full_case], ignore_index=True)

            if self.print_cases:
                print("Full case", full_case, "\n")
      
        return model_performance, model_performance_realcause
    


    def get_realcause_outcome(self, prefix, case_prep, prep_utils, chosen_action, iteration=None):
        prefix = pd.DataFrame(prefix)
        prep_prefix = case_prep.preprocess_sample_bozorgi_retaining(prefix, prep_utils, self.print_cases)
        prep_prefix = prep_prefix.drop(columns=["treatment"])
        realcause_outcome_list = []
        outcome_scaler = prep_utils["scaler_dict"]["outcome"]
        if iteration is None:
            realcause_model_list = self.realcause_model_list
        else:
            realcause_model_list = [self.realcause_model_list[iteration]]
        for model in realcause_model_list:
            # w is equal to prep_prefix without the last column (treatment)
            w = prep_prefix.to_numpy()
            t = np.array([chosen_action] * len(w))
            t = t.reshape(1, 1)
            y0, y1 = model.sample_y(
                t, w, ret_counterfactuals=True, seed=self.full_dataset_params["random_seed_test"]
            )
            if chosen_action == 0:
                y_pred_norm = y0
            else:
                y_pred_norm = y1
            y_pred = outcome_scaler.inverse_transform(y_pred_norm.reshape(-1, 1))
            realcause_outcome_list.append(y_pred)
            if self.print_cases:
                print('t', t)
                print('y0', outcome_scaler.inverse_transform(y0.reshape(-1, 1)), 'y1', outcome_scaler.inverse_transform(y1.reshape(-1, 1)), 'y_pred', y_pred)
        if self.print_cases:
            print("prefix", prefix)
            print("prep_prefix", prep_prefix)
            print("realcause_outcome_list", realcause_outcome_list)
        # flatten list
        realcause_outcome_list = [item for sublist in realcause_outcome_list for item in sublist]
        # flatten one more time
        realcause_outcome_list = [item for sublist in realcause_outcome_list for item in sublist]
        return realcause_outcome_list