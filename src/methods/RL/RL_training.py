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
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy, copy
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
from src.utils.models import LSTM
from SimBank import simulation
from src.methods.RL import RL_data_preparation
from matplotlib import pyplot as plt
from src.utils.utils import EarlyStopping
import gc


class RLModel():
    def __init__(self, model_params, full_dataset_params, int_dataset_params, prep_utils, iteration, baseline_performances_test_val_dict, validator, print_transitions = False, grid_combination_path='default'):
        # Dataset parameters
        self.full_dataset_params = full_dataset_params
        self.int_dataset_params = int_dataset_params
        self.baseline_performances_test_val_dict = baseline_performances_test_val_dict
        self.validator = validator
        # Model parameters
        self.device = model_params["device"]
        self.nr_lstm_layers = model_params["nr_lstm_layers"]
        self.lstm_size = model_params["lstm_size"]
        self.nr_dense_layers = model_params["nr_dense_layers"]
        self.dense_width = model_params["dense_width"]
        self.p = model_params["p"]
        self.print_transitions = print_transitions
        self.print_model_params = model_params["print_model_params"]
        # Intervention parameters
        self.intervention_info = full_dataset_params["intervention_info"]
        self.nr_of_outputs = len([action for actions in self.intervention_info["actions"] for action in actions])
        self.intervention_total_len = sum(self.intervention_info["len"])
        # Data gen parameters
        self.prep_utils = prep_utils
        self.iteration = iteration
        self.transition = namedtuple('Transition', ('state_case', 'state_proc', 'state_t', 'action',
                                                'next_state_case', 'next_state_proc', 'next_state_t', 'reward', 'next_int_point', 'current_int_index'))
        self.input_size_case = len(prep_utils["case_cols_encoded"])
        self.input_size_event = len(prep_utils["event_cols_encoded"])
        # Seeds
        self.overall_seed = copy(model_params["random_seed"]) + self.iteration*5
        torch.manual_seed(self.overall_seed)
        torch.cuda.manual_seed_all(self.overall_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.overall_seed)
        random.seed(self.overall_seed)
        # Models
        self.policy_net = LSTM(input_size_case=self.input_size_case, input_size_process=self.input_size_event,
                               nr_outputs=self.nr_of_outputs, nr_lstm_layers=self.nr_lstm_layers,
                               lstm_size=self.lstm_size, nr_dense_layers=self.nr_dense_layers,
                               dense_width=self.dense_width, p=self.p, treatment_length=self.intervention_total_len, iteration=self.iteration)
        self.target_net = LSTM(input_size_case=self.input_size_case, input_size_process=self.input_size_event,
                               nr_outputs=self.nr_of_outputs, nr_lstm_layers=self.nr_lstm_layers,
                               lstm_size=self.lstm_size, nr_dense_layers=self.nr_dense_layers,
                               dense_width=self.dense_width, p=self.p, treatment_length=self.intervention_total_len, iteration=self.iteration)
        self.policy_net = self.policy_net.to(self.device)
        self.target_net = self.target_net.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        if self.print_model_params:
            print('Policy net: ')
            for name, param in self.policy_net.state_dict().items():
                print(name)
                print(param.data)
                break
            print('Target net: ')
            for name, param in self.target_net.state_dict().items():
                print(param.data)
                break
        self.grid_combination_path = grid_combination_path

    
    def start_training(self, training_params, iteration=0):
        self.init_train_params(training_params, iteration)
        self.train_model()
        self.send_model_to_cpu()
        self.load_best_model()


    def send_model_to_cpu(self):
        #Move model to cpu
        self.policy_net = self.policy_net.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()


    def init_train_params(self, training_params, iteration=0):
        self.filename = training_params["filename"]
        self.trainsize = training_params["train_size"]
        self.val_share = training_params["val_share"]
        self.batch_size = training_params["batch_size"]
        self.calc_val = training_params["calc_val"]
        self.gamma = training_params["gamma"]
        self.tau = training_params["tau"]
        self.eps_strategy = training_params["eps_strategy"]
        self.eps_start = training_params["eps_start"]
        self.eps_end = training_params["eps_end"]
        self.eps_decay = training_params["eps_decay"]
        self.memory_size = int(training_params["memory_size"])
        self.target_update = training_params["target_update"]
        self.transitions_per_optimization = training_params["transitions_per_optimization"]
        self.time_adjustable_exploration = training_params["time_adjustable_exploration"]
        self.prioritized_replay = training_params["prioritized_replay"]
        self.no_wrong_int = training_params["no_wrong_int"]
        self.no_int_at_no_int_point = training_params["no_int_at_no_int_point"]
        self.penalty = training_params["penalty"]
        self.earlystop = training_params["early_stop"]
        self.es_patience = training_params["es_patience"]
        self.es_delta = training_params["es_delta"]
        self.aleatoric = training_params["aleatoric"]
        self.num_episodes = training_params["num_episodes"]
        self.iteration = iteration
        self.early_stop_path = training_params["early_stop_path"]
        self.earlystopfile = self.early_stop_path + f"RL_earlystops_{self.filename}_{iteration}_{str(np.random.randint(0, 10000))}.pt"
        self.loss_function = training_params["loss_function"]
        if "time_contact_HQ" in self.intervention_info["name"]:
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001, amsgrad=True)
        elif len(self.intervention_info["name"]) > 1:
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001, amsgrad=True)
        elif self.intervention_info["name"] == ["choose_procedure"]:
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001, amsgrad=True)
        else:
            self.optimizer = optim.Adam(self.policy_net.parameters(), amsgrad=True)
        self.memory = self.ReplayMemory(self.memory_size, self.transition, iteration=self.iteration, prioritized_replay=self.prioritized_replay)
        self.losses = []
        self.losses_mae = []
        self.rewards = []
        self.eps = 0
        self.eps_terminal_list = []
        self.timing_list_train = []
        self.timing_list_exploit = []
        self.val_performance = 0
        self.val_performance_list = []
        self.prev_weight_list = []
        self.steps_done = 0  # used for exploration
        self.int_points_done = 0
        self.optimization_steps = 0
        self.best_optimization_step = 0
        if self.earlystop:
            self.early_stopping = EarlyStopping(patience=self.es_patience, verbose=False, delta=self.es_delta,
                                                path=self.earlystopfile)
        self.best_val_performance, self.best_step = -1000000, 0


    def find_full_action_index_from_current_int_action_index(self, current_int_action_index, index_int):
        to_add = 0
        for int in range(len(self.intervention_info["name"])):
            if int == index_int:
                break
            to_add += self.intervention_info["len"][int]
        
        if self.intervention_info["len"][index_int] == 1:
            start_index = 0
        else:
            start_index = current_int_action_index
        return start_index + to_add
    

    def find_current_int_action_index_from_result_model(self, index_model, index_int):
        to_subsract = 0
        for int in range(len(self.intervention_info["name"])):
            if int == index_int:
                break
            to_subsract += self.intervention_info["action_width"][int]
        
        return index_model - to_subsract
    

    def find_full_action_index_from_result_model(self, index_model, index_int):
        to_subsract = 0
        for int in range(len(self.intervention_info["name"])):
            if int == index_int:
                break
            to_subsract += self.intervention_info["len"][int]
        
        return index_model - to_subsract

    
    def find_index_model_range(self, index_int):
        to_add = 0
        for int in range(len(self.intervention_info["name"])):
            if int == index_int:
                break
            to_add += self.intervention_info["action_width"][int]
        
        return to_add, to_add + self.intervention_info["action_width"][index_int]
    
    def find_index_len_range(self, index_int):
        # Exclusive the upper bound
        to_add = 0
        for int in range(len(self.intervention_info["name"])):
            if int == index_int:
                break
            to_add += self.intervention_info["len"][int]
        
        return to_add, to_add + self.intervention_info["len"][index_int]
    
    def map_action_taken_to_full_action_index(self, index_model):
        full_action_index = 0
        if self.intervention_total_len == 1:
            full_action_index = 0
        else:
            if self.intervention_info["name"] == ["set_ir_3_levels"]:
                full_action_index = index_model
            elif self.intervention_info["name"] == ["choose_procedure", "set_ir_3_levels"]:
                if index_model != 0:
                    full_action_index = index_model - 1
        
        return full_action_index

    def plot_learning(self, episode=0):
        if episode == 0:
            episode = self.num_episodes
        if self.steps_done >= self.batch_size:
            print("Plotting")
            fig, ax1 = plt.subplots()
            # First subplot: Losses and Profit Gain
            mae_losses_t = torch.tensor(self.losses_mae, dtype=torch.float)  # + 1e-10
            ax1.set_title('Training: Losses and Profit Gain')
            ax1.set_xlabel('Optimization Steps')
            ax1.set_ylabel('Losses (MAE)')
            ax1.plot(mae_losses_t, label="losses", c="tab:blue")

            # Take 100 episode averages and plot them too
            if len(mae_losses_t) >= 100:
                means = mae_losses_t.unfold(0, 100, 1).mean(1).view(-1)
                means = torch.cat((torch.zeros(99), means))
                ax1.plot(means.numpy(), label="moving average losses", c="tab:orange")

            # Show policy results on second y-axis
            ax2 = ax1.twinx()
            ax2.set_ylabel('Profit gain over bank policy on val set (%)')
            ax2.plot((self.val_performance_list - self.baseline_performances_test_val_dict["bank"]) / abs(self.baseline_performances_test_val_dict["bank"]), label=f"Model", c="tab:green")

            ax2.axhline(y=0, linestyle='--', color='red', label=f"Bank")
            ax2.axhline(y=(self.baseline_performances_test_val_dict["random"] - self.baseline_performances_test_val_dict["bank"]) / abs(self.baseline_performances_test_val_dict["bank"]), linestyle='--', color='black', label=f"Random")

            ax1.grid(True, which='major')
            fig.tight_layout()


    def select_action(self, net, state_preproc_case, state_preproc_event, state_preproc_t, current_int_index, int_point, exploit=False, terminal=False, current_timing=None):
        sample = random.random()
        full_action = [0] * self.intervention_total_len
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
        no_int_at_no_int_point = self.no_int_at_no_int_point

        if sample > eps_threshold:
            if self.print_transitions:
                print("Exploiting")
            with torch.no_grad():
                if exploit:
                    net.eval()
                result = net(state_preproc_case, state_preproc_event, state_preproc_t)
                if self.print_transitions:
                    print('result', result)
                # Make sure that model only selects actions that are possible in the current intervention
                action_taken = result.max(1)[1].view(1, 1).item()
        else:
            if self.print_transitions:
                print("Exploring")
                
            if self.intervention_info["action_depth"][current_int_index] == 1 or not self.time_adjustable_exploration or not int_point:
                action_taken = random.randrange(self.nr_of_outputs)

            if self.intervention_info["name"] == ["time_contact_HQ"] and self.time_adjustable_exploration:
                if self.print_transitions:
                    print('nr_enabled', current_timing)
                    print('rand_timing', self.rand_timing)

                if current_timing == self.rand_timing:
                    action_taken = 1
                else:
                    action_taken = 0
        
        full_action_index = self.map_action_taken_to_full_action_index(action_taken)

        if int_point:
            lower_bound, upper_bound = self.find_index_model_range(current_int_index)
            if action_taken < lower_bound or action_taken >= upper_bound:
                if exploit:
                    result = result[:, lower_bound:upper_bound]
                    current_int_action_index = result.max(1)[1].view(1, 1).item()
                    full_action_index = self.find_full_action_index_from_current_int_action_index(current_int_action_index, current_int_index)
                elif no_wrong_int:
                    # Sample from nr of outputs, but make sure that action is within bounds
                    while action_taken < lower_bound or action_taken >= upper_bound:
                        action_taken = random.randrange(self.nr_of_outputs)
                    current_int_action_index = self.find_current_int_action_index_from_result_model(action_taken, current_int_index)
                    full_action_index = self.find_full_action_index_from_current_int_action_index(current_int_action_index, current_int_index)
                else:
                    current_int_action_index = None
            else:
                current_int_action_index = self.find_current_int_action_index_from_result_model(action_taken, current_int_index)
                if action_taken == 1 and current_timing % 2 == 0 and self.intervention_info["name"] == ["time_contact_HQ"]:
                    if exploit:
                        self.timing_list_exploit.append(current_timing)
                    else:
                        self.timing_list_train.append(current_timing)
        else:
            current_int_action_index = None
            if no_int_at_no_int_point:
                action_taken = 0
        
        # If there is no action taken in the current episode, add timing 8
        index_contact_headquarters = 9
        contact_headquarters_slice = state_preproc_event[:, index_contact_headquarters, :]
        # Check if there is no '1' in this slice
        no_one_in_contact_headquarters = (contact_headquarters_slice == 1).sum().item() == 0
        if no_one_in_contact_headquarters and self.intervention_info["name"] == ["time_contact_HQ"]:
            if exploit:
                if current_timing == 7:
                    self.timing_list_exploit.append(8)
            else:
                if current_timing == 8:
                    self.timing_list_train.append(8)
        

        if action_taken == 0 and self.intervention_total_len < self.nr_of_outputs: # there is a do nothing action
            value_to_fill = 0
        else:
            value_to_fill = 1

        full_action[full_action_index] = value_to_fill

        if self.print_transitions:
            print('full_action', full_action)
            print('current_int_action_index', current_int_action_index)
            print('full_action_index', full_action_index)
        
        return full_action, current_int_action_index
    

    def observe_reward(self, running_df, terminal, int_point, action_list, current_int_index, wrong_int):
        penalty = False
        action_taken = any(x == 1 for x in action_list)
        reward = 0
        penalty_size = self.penalty
        if (not int_point and action_taken) and not (terminal and not wrong_int):
            reward += -penalty_size
            penalty = True
        elif (not int_point and action_taken) and (terminal and not wrong_int):
            reward += -penalty_size
            penalty = True
        elif wrong_int:
            reward += -penalty_size
            penalty = True
        
        if terminal and not wrong_int:
            reward_ter = running_df["outcome"].iloc[-1]
            reward += reward_ter

        reward_preproc = self.case_prep.preprocess_reward_RL(reward, self.prep_utils, device=self.device)
        reward_preproc = torch.tensor([reward_preproc], device=self.device)

        return reward_preproc, penalty
        

    class ReplayMemory(object):
        def __init__(self, capacity, transition, iteration=0, prioritized_replay=False):
            self.memory = deque([], maxlen=capacity)
            self.priorities = deque([], maxlen=capacity)
            self.transition = transition
            self.iteration = iteration
            self.prioritized_replay = prioritized_replay

        def push(self, *args, int_point=False, terminal=False):
            priority = 1
            if terminal or int_point:
                priority += 300
            self.priorities.append(priority)
            self.memory.append(self.transition(*args))

        def sample(self, batch_size, seed=0):
            if self.prioritized_replay:
                total_priority = sum(self.priorities)
                probabilities = [priority / total_priority for priority in self.priorities]
                sampled_indices = random.choices(range(len(self.memory)), k=batch_size, weights=probabilities)
                transitions = [self.memory[i] for i in sampled_indices]
                batch = self.transition(*zip(*transitions))
            else:
                transitions = random.sample(self.memory, batch_size)
                batch = self.transition(*zip(*transitions))
            return batch

        def __len__(self):
            return len(self.memory)


    def optimize_model(self):
        loss = 0
        if len(self.memory) < self.batch_size:
            return loss

        batch = self.memory.sample(self.batch_size)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state_case)), device=self.device, dtype=torch.bool)
        non_final_next_states_case = torch.cat([s for s in batch.next_state_case
                                                if s is not None])
        non_final_next_states_proc = torch.cat([s for s in batch.next_state_proc
                                                if s is not None])
        non_final_next_states_t = torch.cat([s for s in batch.next_state_t
                                             if s is not None])

        state_batch_case = torch.cat(batch.state_case)
        state_batch_proc = torch.cat(batch.state_proc)
        state_batch_t = torch.cat(batch.state_t)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_int_point_batch = batch.next_int_point
        current_int_index_batch = batch.current_int_index
        state_forward_pass = self.policy_net(state_batch_case, state_batch_proc,
                                              state_batch_t)
        if self.intervention_total_len > 1:
            action_batch_index = torch.zeros(action_batch.size(0), 1, device=self.device, dtype=torch.long)
            for i, row in enumerate(action_batch):
                indices = torch.nonzero(row).squeeze()
                if indices.numel() == 0:
                    index = 0
                else:
                    index = indices.item()
                    if self.intervention_info["name"] == ["choose_procedure", "set_ir_3_levels"]:
                        index += 1
                action_batch_index[i] = index
        else:
            action_batch_index = action_batch

        # Without aleatoric
        state_action_values = state_forward_pass.gather(1, action_batch_index)  # = Q(s,a)
        preproc_zero = float(self.case_prep.preprocess_reward_RL(0, self.prep_utils, device=self.device))
        next_state_values = torch.ones(self.batch_size, device=self.device) * preproc_zero
        with torch.no_grad():
            if self.no_wrong_int and self.intervention_info["name"] == ["choose_procedure", "set_ir_3_levels"]:
                all_next_state_values = self.target_net(non_final_next_states_case, non_final_next_states_proc,
                                                    non_final_next_states_t)
                idx_to_add = 0
                for idx, next_int_point in enumerate(next_int_point_batch):
                    if not non_final_mask[idx]:
                        idx_to_add -= 1
                        continue
                    if next_int_point:
                        lower_bound, upper_bound = self.find_index_model_range(current_int_index_batch[idx])
                        max_value = all_next_state_values[idx + idx_to_add, lower_bound:upper_bound].max().detach()
                    else:
                        if self.no_int_at_no_int_point:
                            max_value = all_next_state_values[idx + idx_to_add, 0].detach()
                        else:
                            max_value = all_next_state_values[idx + idx_to_add].max().detach()
                    next_state_values[idx] = max_value
            else:
                next_state_values[non_final_mask] = self.target_net(non_final_next_states_case, non_final_next_states_proc,
                                                                non_final_next_states_t).max(1)[0].detach()
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        expected_state_action_values = expected_state_action_values.unsqueeze(1).float()
        state_action_values = state_action_values.float()

        if self.print_transitions:
            print('rewards', reward_batch)
            print("action_batch", action_batch)
            print("action_batch_index", action_batch_index)
            print('state_forward_pass', state_forward_pass)
            print('state_action_values', state_action_values)
            print('expected_state_action_values', expected_state_action_values)
            print('all target net stuff: ', self.target_net(non_final_next_states_case, non_final_next_states_proc,
                                                            non_final_next_states_t))
            print('next_state_values', next_state_values)

        if self.loss_function == "mse":
            loss = torch.nn.functional.mse_loss(target=expected_state_action_values, input=state_action_values)
        else:
            criterion = nn.SmoothL1Loss(beta=0.075)
            loss = criterion(state_action_values, expected_state_action_values)

        mae_loss = torch.nn.functional.l1_loss(target=expected_state_action_values, input=state_action_values)
        self.losses_mae.append(mae_loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_net.parameters(), clip_value=1)
        self.optimizer.step()
        self.optimization_steps += 1

        return loss
    

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
        self.case_prep = RL_data_preparation.LoanProcessPreprocessor(dataset_params=self.int_dataset_params[0]) # Just take the first one, does not matter as we only need the scaler, 
        self.random_obj_for_timing = random.Random()
        self.random_obj_for_timing.seed(self.full_dataset_params["random_seed_train"])
        print('Training started')

        for i_episode in range(self.num_episodes):
            terminal = False
            wrong_int = False
            self.int_point = False
            self.next_int_point = False
            self.policy_net.train()
            prefix_list = []
            another_seed_to_add = 0 + self.full_dataset_params["train_size"]*70
            while prefix_list == []:
                prefix_list = self.case_gen.start_simulation_inference(seed_to_add=i_episode + another_seed_to_add)
                another_seed_to_add += 1
            
            running_df = pd.DataFrame([prefix_list[0][0]])
            state_t = [[0] * self.intervention_total_len] * len(running_df)
            state_preproc_case, state_preproc_event, state_preproc_t, _ = self.case_prep.preprocess_sample_RL(data_sample=running_df, data_t=state_t, prep_utils=self.prep_utils, device=self.device, treat_len=self.intervention_total_len)

            if self.print_transitions:
                print('\n\n\n')
                print('NEW EPISODE: ', i_episode)
                print('\n')
                print('initial running_df:', running_df)
                print("state_t:", state_t)
                print("\n")

            self.current_timing = 0
            self.rand_timing = 0
            if self.intervention_info["name"] == ["time_contact_HQ"]:
                self.rand_timing = self.sample_timing(running_df.iloc[0])
           
            for t in count(1):
                self.int_point = False
                self.next_int_point = False
                terminal = False
                if t == len(prefix_list[0]) - 1 and self.case_gen.int_points_available:
                    self.int_point = True
                elif t == len(prefix_list[0]) - 2 and self.case_gen.int_points_available:
                    self.next_int_point = True
                elif t >= len(prefix_list[0]) - 1 and not self.case_gen.int_points_available:
                    terminal = True
                
                int_point_before_action = copy(self.int_point)
                current_int_index_before_action = copy(self.case_gen.current_int_index)
                int_index_next_int_point = current_int_index_before_action
                self.timing_before_action = copy(self.current_timing)
                full_action_list, current_int_action_index = self.select_action(self.policy_net, state_preproc_case, state_preproc_event, state_preproc_t, self.case_gen.current_int_index, int_point = self.int_point, terminal=terminal, current_timing=self.current_timing)
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
                    print('state_t:', state_t)
                    print('action taken:', full_action_list)

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
                    if t == len(prefix_list[0]) - 2 and self.case_gen.int_points_available:
                        self.next_int_point = True
                        int_index_next_int_point = current_int_index_before_action + 1
                elif not terminal:
                    index = t + 1
                    running_df = pd.DataFrame(prefix_list[0][:index])
                else:
                    running_df = pd.DataFrame(prefix_list[0])

                state_t.append(full_action_list)
                next_state_preproc_case, next_state_preproc_event, next_state_preproc_t, full_action_list_prep = self.case_prep.preprocess_sample_RL(data_sample=running_df, data_t=state_t, data_full_action=full_action_list, prep_utils=self.prep_utils, device=self.device, treat_len=self.intervention_total_len)

                # Observe reward
                reward_preproc, penalty = self.observe_reward(running_df, terminal, int_point_before_action, full_action_list, current_int_index_before_action, wrong_int)
                if terminal:
                # if terminal or penalty:
                    next_state_preproc_case, next_state_preproc_event, next_state_preproc_t = None, None, None

                # Store the transition in memory
                next_state_running_df = deepcopy(running_df)
                if self.next_int_point:
                    if self.print_transitions:
                        print('next state is an int_point:', next_state_running_df.iloc[-1])
                self.memory.push(state_preproc_case, state_preproc_event, state_preproc_t, full_action_list_prep,
                                        next_state_preproc_case,
                                        next_state_preproc_event, next_state_preproc_t, reward_preproc, self.next_int_point, int_index_next_int_point, int_point=int_point_before_action, terminal=terminal)
                
                # Perform one step of the optimization (on the policy network)
                if self.steps_done % self.transitions_per_optimization == 0:
                    loss = self.optimize_model()
                    self.losses.append(loss)
                    self.val_performance_list.append(self.val_performance)

                if self.print_transitions:
                    print('NEXT STATE')
                    print('next state:', running_df)
                    print('next state_t:', state_t)
                    print('reward:', reward_preproc)
                    if self.steps_done % self.transitions_per_optimization == 0:
                        print("optimized loss:", loss)
                    print("\n")

                # Move to the next state
                state_preproc_t = deepcopy(next_state_preproc_t)
                state_preproc_case = deepcopy(next_state_preproc_case)
                state_preproc_event = deepcopy(next_state_preproc_event)
                
                if terminal:
                # if terminal or penalty:
                    self.terminal_reward_list.append(reward_preproc)
                    self.eps_terminal_list.append(self.eps)
                    break

            if self.print_transitions:
                print("TERMINAL STATE")
                print("all actions taken:", state_t)
                print('full case:', full_case)
                print("reward:", reward_preproc)
                print('optimized loss:', loss)
                print("\n")

            if (i_episode + 1) % self.target_update == 0:
                target_net_state_dict = {}
                for key in self.policy_net.state_dict():
                    target_net_state_dict[key] = self.tau * self.policy_net.state_dict()[key] + (1 - self.tau) * self.target_net.state_dict()[key]
                self.target_net.load_state_dict(target_net_state_dict)
            
            # Validation and possibly early stopping
            if (self.steps_done > self.batch_size) and ((i_episode + 1) % self.calc_val == 0):
                self.timing_list_exploit = []
                self.val_performance, _ = self.validator.model_policy_inference(self.full_dataset_params["test_val_size"], device=self.device, model_class=self, model=self.policy_net, validation=True)
                print('Validation performance:', self.val_performance, "at episode", i_episode, "\n")
                if self.val_performance > self.best_val_performance:
                    self.best_val_performance = self.val_performance
                    self.best_step = len(self.losses) - 1
                    self.best_model = deepcopy(self.policy_net)
                    self.best_optimization_step = deepcopy(self.optimization_steps)
                    print('     New best model at episode {}'.format(i_episode))
                    print("     Model: ")
                    print("     Model action timings: ", self.validator.model_action_timings)
                    print("     Model actions: ", self.validator.model_actions)
                    print('     Model loss: ', self.losses[-1])
                    for name, param in self.best_model.state_dict().items():
                        print(param.data)
                        break
                    print("\n")
                if i_episode == 9999:
                    print('Saving model at episode 10000')
                    self.model_at_10000 = deepcopy(self.policy_net)
                    self.best_model_at_10000 = deepcopy(self.best_model)
                    self.best_step_at_10000 = self.best_step
                    self.best_val_performance_at_10000 = self.best_val_performance
                if ((i_episode + 1) % 25000) == 0:
                    self.plot_learning(episode=i_episode)
                if self.earlystop:
                    self.early_stopping(-self.val_performance, self.policy_net, epoch=len(self.losses) - 1)
                    if self.early_stopping.early_stop:
                       print("\nEarly stopping at episode {}".format(i_episode))
                       break

        print('Training complete')

    def load_best_model(self, model_path=None):
        self.final_net = deepcopy(self.policy_net)
        if os.path.exists(self.earlystopfile) and model_path is None:
            print('Loading best model')
            self.final_net.load_state_dict(torch.load(self.earlystopfile))
        else:
            print('Loading model from path')
            self.final_net.load_state_dict(torch.load(model_path))
            self.best_model = deepcopy(self.final_net)
            # send to cpu
            self.best_model = self.best_model.to("cpu")
            self.final_net = self.final_net.to("cpu")
            self.policy_net = self.policy_net.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
        if self.print_model_params:
            print('Final net: ')
            for name, param in self.final_net.state_dict().items():
                print(param.data)
                break
            print('Best net: ')
            for name, param in self.best_model.state_dict().items():
                print(param.data)
                break
            print('Policy net: ')
            for name, param in self.policy_net.state_dict().items():
                print(param.data)
                break