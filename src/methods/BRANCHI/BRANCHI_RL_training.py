data_folder = "data"
results_folder = "res"
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(parent_dir)
from config.config import path
sys.path.append(path)
sys.path.append(path + "\\SimBank")

import math
import random
import matplotlib.pyplot as plt
from itertools import count
import pandas as pd
import numpy as np
#import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
from copy import deepcopy, copy
from SimBank import simulation
from src.methods.BRANCHI import BRANCHI_data_preparation
from src.methods.BRANCHI.BRANCHI_utils import EarlyStopping


class RLModel():
    def __init__(self, model_params, full_dataset_params, int_dataset_params, kmeans, best_k, prep_utils, iteration, baseline_performances_test_val_dict, validator, print_transitions = False):
        # Dataset parameters
        self.full_dataset_params = full_dataset_params
        self.int_dataset_params = int_dataset_params
        self.baseline_performances_test_val_dict = baseline_performances_test_val_dict
        self.validator = validator
        self.kmeans = kmeans
        self.best_k = best_k
        # Model parameters
        self.device = model_params["device"]
        self.print_transitions = print_transitions
        self.print_model_params = model_params["print_model_params"]
        # Intervention parameters
        self.intervention_info = full_dataset_params["intervention_info"]
        self.actions = [action for actions in self.intervention_info["actions"] for action in actions]
        if self.full_dataset_params["extra_do_nothing_action"]:
            # put it at the beginning
            self.actions = ["do_nothing"] + self.actions
        print(self.actions)
        self.intervention_total_len = sum(self.intervention_info["len"])
        # Data gen parameters
        self.prep_utils = prep_utils
        self.iteration = iteration
        # Seeds
        self.overall_seed = copy(model_params["random_seed"]) + self.iteration*5
        torch.manual_seed(self.overall_seed)
        torch.cuda.manual_seed_all(self.overall_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.overall_seed)
        random.seed(self.overall_seed)

    
    def start_training(self, training_params, iteration=0):
        self.init_train_params(training_params, iteration)
        self.init_q_table()
        self.train_model()
        # self.send_model_to_cpu()
        self.load_best_model()


    def init_train_params(self, training_params, iteration=0):
        self.filename = training_params["filename"]
        self.trainsize = training_params["train_size"]
        self.val_share = training_params["val_share"]
        self.calc_val = training_params["calc_val"]
        self.gamma = training_params["gamma"]
        self.alpha = training_params["alpha"]
        self.eps_strategy = training_params["eps_strategy"]
        self.eps_start = training_params["eps_start"]
        self.eps_end = training_params["eps_end"]
        self.eps_decay = training_params["eps_decay"]
        self.time_adjustable_exploration = training_params["time_adjustable_exploration"]
        self.no_wrong_int = training_params["no_wrong_int"]
        self.penalty = training_params["penalty"]
        self.earlystop = training_params["early_stop"]
        self.es_patience = training_params["es_patience"]
        self.es_delta = training_params["es_delta"]
        self.num_episodes = training_params["num_episodes"]
        self.early_stop_path = training_params["early_stop_path"]
        self.iteration = iteration
        self.earlystopfile = self.early_stop_path + f"BRANCHI_earlystops_{self.filename}_{iteration}_{str(np.random.randint(0, 10000))}.pt"
        self.rewards = []
        self.eps = 0
        self.eps_terminal_list = []
        self.timing_list_train = []
        self.timing_list_exploit = []
        self.val_performance = 0
        self.val_performance_list = []
        self.steps_done = 0  # used for exploration
        self.int_points_done = 0
        if self.earlystop:
            self.early_stopping = EarlyStopping(patience=self.es_patience, verbose=False, delta=self.es_delta,
                                                path=self.earlystopfile)
        self.best_val_performance, self.best_step = 0, 0

    
    def init_q_table(self):
        self.q_table = {}
        for unique_activity in self.prep_utils["unique_activities"]:
            for k in range(self.best_k):
                state_preproc = (k, unique_activity)
                self.q_table[state_preproc] = {}
                for action in self.actions:
                    self.q_table[state_preproc][action] = 0


    def live_plot(self):
        plt.figure(figsize=(10, 10))
        # First subplot: Losses and Profit Gain
        plt.title('Training: Rewards and Profit Gain for Iteration ' + str(self.iteration))
        plt.xlabel('Steps')
        plt.ylabel('Terminal Rewards')
        plt.plot(self.terminal_reward_list, label="rewards", c="tab:red")

        # Take 100 episode averages and plot them too
        if len(self.terminal_reward_list) >= 100:
            means = self.terminal_reward_list.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy(), label="moving average rewards", c="tab:orange")

        # Show policy results on second y-axis
        plt_twin = plt.twinx()
        plt_twin.plot((self.val_performance_list - self.baseline_performances_test_val_dict["bank"]) / abs(self.baseline_performances_test_val_dict["bank"]), label=f"Model", c="tab:green")
        plt_twin.set_ylabel(f"Profit gain over bank policy on val set (%)")
        plt_twin.axhline(y=0, linestyle='--', color='red', label=f"Bank")
        plt_twin.axhline(y=(self.baseline_performances_test_val_dict["random"] - self.baseline_performances_test_val_dict["bank"]) / abs(self.baseline_performances_test_val_dict["bank"]), linestyle='--', color='black', label=f"Random")

        plt.yscale("log")
        plt.grid(True, which='major')
        plt.show()


    def select_action(self, state_preproc, current_int_index, int_point, exploit=False, current_timing=None, q_table=None):
        sample = random.random()
        current_int_action_index = None
        
        # Set epsilon threshold
        if exploit:
            eps_threshold = 0
        else:
            if self.eps_strategy == "exponential":
                eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                                math.exp(-1. * self.steps_done / self.eps_decay)
            elif self.eps_strategy == "linear":            
            # Linear decay
                eps_threshold = self.eps_start + self.steps_done * (self.eps_end - self.eps_start) / self.eps_decay
            
            eps_threshold = min(eps_threshold, self.eps_start)
            eps_threshold = max(eps_threshold, self.eps_end)
        
        self.eps = eps_threshold
        no_wrong_int = self.no_wrong_int

        if sample > eps_threshold:
            if self.print_transitions:
                print("Exploiting")
            max_val = q_table[state_preproc].values()
            action_taken = max(q_table[state_preproc], key=q_table[state_preproc].get)
        else:
            if self.print_transitions:
                print("Exploring")
                
            if self.intervention_info["action_depth"][current_int_index] == 1 or not self.time_adjustable_exploration or not int_point:
                action_taken = random.choice(self.actions)

            if self.intervention_info["name"] == ["time_contact_HQ"] and self.time_adjustable_exploration:
                if self.print_transitions:
                    print('nr_enabled', current_timing)
                    print('rand_timing', self.rand_timing)

                if current_timing == self.rand_timing:
                    action_taken = "contact_headquarters"
                else:
                    action_taken = "do_nothing"

        if int_point:
            if action_taken not in self.full_dataset_params["intervention_info"]["actions"][current_int_index]:
                if exploit:
                    # take the max value of the actions that are allowed
                    action_taken = max([k for k in q_table[state_preproc].keys() if k in self.full_dataset_params["intervention_info"]["actions"][current_int_index]], key=q_table[state_preproc].get)
                    # get the index of the action taken in the int_index action list
                    current_int_action_index = self.full_dataset_params["intervention_info"]["actions"][current_int_index].index(action_taken)
                elif no_wrong_int:
                    action_taken = random.choice(self.full_dataset_params["intervention_info"]["actions"][current_int_index])
                    current_int_action_index = self.full_dataset_params["intervention_info"]["actions"][current_int_index].index(action_taken)
                else:
                    current_int_action_index = None
            else:
                current_int_action_index = self.full_dataset_params["intervention_info"]["actions"][current_int_index].index(action_taken)
                if action_taken == "contact_headquarters" and current_timing % 2 == 0 and self.intervention_info["name"] == ["time_contact_HQ"]:
                    if exploit:
                        self.timing_list_exploit.append(current_timing)
                    else:
                        self.timing_list_train.append(current_timing)
        else:
            current_int_action_index = None

        if self.print_transitions:
            print('action taken:', action_taken)
            print('current_int_action_index:', current_int_action_index)
        
        return action_taken, current_int_action_index
    

    def observe_reward(self, running_df, terminal, int_point, action_taken, current_int_index, wrong_int):
        reward = 0
        penalty_size = self.penalty
        if self.intervention_info["name"] == ["time_contact_HQ"] or self.intervention_info["name"] == ["set_ir_3_levels"] or self.intervention_info["name"] == ["choose_procedure", "set_ir_3_levels"]:
            do_nothing_action = self.actions[0]
        else:
            do_nothing_action = "do_nothing"

        if (not int_point and action_taken != do_nothing_action) and not (terminal and not wrong_int):
            reward += -penalty_size
        elif (not int_point and action_taken != do_nothing_action) and (terminal and not wrong_int):
            reward += -penalty_size
        elif wrong_int:
            reward += -penalty_size
        
        if terminal and not wrong_int:
            reward_ter = running_df["outcome"].iloc[-1]
            reward += reward_ter
        if reward != 0:
            reward_preproc = self.case_prep.preprocess_reward_kmeans_RL(reward, self.prep_utils, device=self.device)
        else:
            reward_preproc = 0

        return reward_preproc
        

    def update_q_table(self, next_state_preproc, reward_preproc, state_preproc, action_taken):
        if next_state_preproc is not None:
            next_max_val = max(self.q_table[next_state_preproc].values())
            next_action_taken = max(self.q_table[next_state_preproc], key=self.q_table[next_state_preproc].get)
        else:
            next_max_val = 0
            next_action_taken = None
        old_val = self.q_table[state_preproc][action_taken]
        new_val = (1 - self.alpha) * old_val + self.alpha * (reward_preproc + self.gamma * next_max_val)
        self.q_table[state_preproc][action_taken] = new_val
        return next_action_taken, next_max_val
        

    def sample_timing(self, event=None):
        lower_bound = 0
        upper_bound = self.intervention_info["action_depth"][0] - 1
        if "do_nothing" in self.intervention_info["activities"][0]:
            upper_bound += 1
        timing = self.random_obj_for_timing.randint(lower_bound, upper_bound)
        if "time_contact_HQ" in self.intervention_info["name"]:
            timing = timing*2
        return timing


    def train_model(self):
        self.terminal_reward_list = []
        self.case_gen = simulation.PresProcessGenerator(self.full_dataset_params, seed=self.full_dataset_params["random_seed_train"])
        self.case_prep = BRANCHI_data_preparation.LoanProcessPreprocessor(dataset_params=self.int_dataset_params[0]) # Just take the first one, does not matter as we only need the scaler, 
        self.random_obj_for_timing = random.Random()
        self.random_obj_for_timing.seed(self.full_dataset_params["random_seed_train"])
        print('Training started')

        for i_episode in range(self.num_episodes):
            terminal = False
            wrong_int = False
            self.int_point = False
            prefix_list = []
            another_seed_to_add = 0 + self.full_dataset_params["train_size"]*70
            actions_taken_list = []
            while prefix_list == []:
                prefix_list = self.case_gen.start_simulation_inference(seed_to_add=i_episode + another_seed_to_add)
                another_seed_to_add += 1
            
            running_df = pd.DataFrame([prefix_list[0][0]])
            state_preproc = self.case_prep.preprocess_kmeans_RL(data_sample=running_df, reward_preproc = 0, prep_utils=self.prep_utils, kmeans=self.kmeans)

            if self.print_transitions:
                print('\n\n\n')
                print('NEW EPISODE: ', i_episode)
                print('\n')
                print('initial running_df:', running_df)
                print("state_preproc: ", state_preproc)
                print("\n")

            self.current_timing = 0
            self.rand_timing = 0
            if self.intervention_info["name"] == ["time_contact_HQ"]:
                self.rand_timing = self.sample_timing(running_df.iloc[0])
           
            for t in count(1):
                if t == len(prefix_list[0]) - 1 and self.case_gen.int_points_available:
                    self.int_point = True
                elif t >= len(prefix_list[0]) - 1 and not self.case_gen.int_points_available:
                    terminal = True
                
                int_point_before_action = copy(self.int_point)
                current_int_index_before_action = copy(self.case_gen.current_int_index)
                self.timing_before_action = copy(self.current_timing)
                action_taken, current_int_action_index = self.select_action(state_preproc, self.case_gen.current_int_index, int_point = self.int_point, current_timing=self.current_timing, q_table=self.q_table)
                actions_taken_list.append(action_taken)
                self.steps_done += 1
                
                if int_point_before_action and current_int_action_index is None:
                    wrong_int = True
                    terminal = True
                    self.int_point = False

                if self.intervention_info["name"] == ["time_contact_HQ"]:
                    if self.current_timing % 2 != 0:
                        if current_int_action_index == 1:
                            current_int_action_index = 0
                        int_point_before_action = False
                        if self.print_transitions:
                            print('not intervention point for time_contact_HQ (validate_application)')

                if self.print_transitions:
                    print("CURRENT STATE")
                    print('state :', running_df)
                    print('action taken:', action_taken)
                    print('state_preproc: ', state_preproc)

                if self.int_point:
                    prefix_list = self.case_gen.continue_simulation_inference(current_int_action_index)
                    if not self.case_gen.int_points_available:
                        full_case = self.case_gen.end_simulation_inference()
                        prefix_list = [full_case] * self.intervention_info["len"][self.case_gen.current_int_index]
                    index = len(running_df) + 1
                    running_df = pd.DataFrame(prefix_list[0][:index])
                    self.int_point = False
                    self.int_points_done += 1
                    if self.intervention_info["name"] == ["time_contact_HQ"]:
                        self.current_timing += 1
                elif not terminal:
                    index = t + 1
                    running_df = pd.DataFrame(prefix_list[0][:index])
                else:
                    running_df = pd.DataFrame(prefix_list[0])
                    full_case = running_df

                reward_preproc = self.observe_reward(running_df, terminal, int_point_before_action, action_taken, current_int_index_before_action, wrong_int)
                next_state_preproc = self.case_prep.preprocess_kmeans_RL(data_sample=running_df, reward_preproc = reward_preproc, prep_utils=self.prep_utils, kmeans=self.kmeans)

                # Observe reward
                if terminal:
                    next_state_preproc = None
                
                next_action_taken, next_max_val = self.update_q_table(next_state_preproc, reward_preproc, state_preproc, action_taken)
                self.val_performance_list.append(self.val_performance)

                if self.print_transitions:
                    print('NEXT STATE')
                    print('next state:', running_df)
                    print('next state preproc: ', next_state_preproc)
                    print('next action taken:', next_action_taken)
                    print('next max val:', next_max_val)
                    print('reward:', reward_preproc)
                    print("\n")

                # Move to the next state
                state_preproc = deepcopy(next_state_preproc)
                
                if terminal:
                    self.terminal_reward_list.append(reward_preproc)
                    self.eps_terminal_list.append(self.eps)
                    break

            if self.print_transitions:
                print("TERMINAL STATE")
                print("all actions taken:", actions_taken_list)
                print('full case:', full_case)
                print("reward:", reward_preproc)
                print("\n")

            # Validation and possibly early stopping
            if ((i_episode + 1) % self.calc_val == 0):
                self.timing_list_exploit = []
                self.val_performance, _ = self.validator.model_policy_inference(self.full_dataset_params["test_val_size"], model_class=self, q_table=self.q_table, kmeans=self.kmeans, validation=True)
                print('Validation performance:', self.val_performance, "at episode", i_episode, "\n")
                if self.val_performance > self.best_val_performance:
                    self.best_val_performance = self.val_performance
                    self.best_step = self.steps_done
                    self.best_q_table = deepcopy(self.q_table)
                    print('     New best model at episode {}'.format(i_episode))
                    print("     Model: ")
                    print("     Model action timings: ", self.validator.model_action_timings)
                    print('     Model actions: ', self.validator.model_actions)
                    print("\n")
                if i_episode == 9999:
                    print('Saving model at episode 10000')
                    self.best_step_at_10000 = self.best_step
                    self.best_val_performance_at_10000 = self.best_val_performance
                if self.earlystop:
                    self.early_stopping(-self.val_performance, self.q_table, epoch=self.steps_done)
                    if self.early_stopping.early_stop:
                       print("\nEarly stopping at episode {}".format(i_episode))
                       break

        print('Training complete')

    def load_best_model(self):
        self.final_q_table = self.best_q_table