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
from datetime import datetime
from copy import deepcopy
from src.utils.tools import save_data
from itertools import product
from SimBank import simulation





#DATASET parameters
dataset_params = {}
#general
dataset_params["train_size"] = 100000
dataset_params["test_size"] = 10000
dataset_params["val_share"] = .5
dataset_params["train_val_size"] = 10000
dataset_params["test_val_size"] = min(int(dataset_params["val_share"] * dataset_params["test_size"]), 1000)
dataset_params["nr_cfs"] = {"min": 40, "max": 1000}
dataset_params["cf_conf_width"] = 100
dataset_params["simulation_start"] = datetime(2024, 3, 20, 8, 0)
dataset_params["random_seed_train"] = 82*82
dataset_params["random_seed_test"] = 130*130
#process
dataset_params["log_cols"] = ["case_nr", "activity", "timestamp", "elapsed_time", "cum_cost", "est_quality", "unc_quality", "amount", "interest_rate", "discount_factor", "outcome", "quality", "noc", "nor", "min_interest_rate"]
dataset_params["case_cols"] = ["amount"]
dataset_params["event_cols"] = ["activity", "elapsed_time", "cum_cost", "est_quality", "unc_quality", "interest_rate", "discount_factor"]
dataset_params["cat_cols"] = ["activity"]
dataset_params["scale_cols"] = ["amount", "elapsed_time", "cum_cost", "est_quality", "unc_quality", "interest_rate", "discount_factor", "outcome"]
#intervention
dataset_params["intervention_info"] = {}
dataset_params["intervention_info"]["name"] = ["time_contact_HQ"]
dataset_params["intervention_info"]["data_impact"] = ["direct"]
dataset_params["intervention_info"]["actions"] = [["do_nothing","contact_headquarters"]] #If binary, last action is the 'treatment' action
dataset_params["intervention_info"]["action_width"] = [2]
dataset_params["intervention_info"]["action_depth"] = [4] #Is the max number of times the intervention can be applied
dataset_params["intervention_info"]["activities"] = [["do_nothing", "contact_headquarters"]]
dataset_params["intervention_info"]["column"] = ["activity"]
dataset_params["intervention_info"]["start_control_activity"] = [["start_standard"]]
dataset_params["intervention_info"]["end_control_activity"] = [["start_standard", "email_customer", "call_customer"]]

dataset_params["intervention_info"]["retain_method"] = "precise"
# dataset_params["intervention_info"]["retain_method"] = "non-precise"

# Combinations
dataset_params["intervention_info"]["action_combinations"] = list(product(*dataset_params["intervention_info"]["actions"]))
dataset_params["intervention_info"]["action_width_combinations"] = math.prod(dataset_params["intervention_info"]["action_width"])
dataset_params["intervention_info"]["action_depth_combinations"] = math.prod(dataset_params["intervention_info"]["action_depth"])

dataset_params["intervention_info"]["len"] = [action_width if action_width > 2 else 1 for action_width in dataset_params["intervention_info"]["action_width"]]
dataset_params["intervention_info"]["RCT"] = False
dataset_params["filename"] = "loan_log_" +  str(dataset_params["intervention_info"]["name"])
#policy
dataset_params["policies_info"] = {}
dataset_params["policies_info"]["general"] = "real"
dataset_params["policies_info"]["choose_procedure"] = {"amount": 50000, "est_quality": 5}
dataset_params["policies_info"]["time_contact_HQ"] = "real"
dataset_params["policies_info"]["min_quality"] = 2
dataset_params["policies_info"]["max_noc"] = 3
dataset_params["policies_info"]["max_nor"] = 1
# dataset_params["policies_info"]["max_nor"] = 3
dataset_params["policies_info"]["min_amount_contact_cust"] = 50000








## Data generation (only run if new data needed)
gen_normal = simulation.PresProcessGenerator(dataset_params, dataset_params["random_seed_train"])
train_normal = gen_normal.run_simulation_normal(dataset_params["train_size"])
train_normal_val = gen_normal.run_simulation_normal(dataset_params["train_val_size"], seed_to_add=88)
dataset_params_RCT = deepcopy(dataset_params)
dataset_params_RCT["intervention_info"]["RCT"] = True
dataset_params_RCT["random_seed_train"] = dataset_params["random_seed_train"]*10
dataset_params_RCT["simulation_start"] = deepcopy(gen_normal.simulation_end)
gen_RCT = simulation.PresProcessGenerator(dataset_params_RCT, dataset_params_RCT["random_seed_train"])
train_RCT = gen_RCT.run_simulation_normal(dataset_params_RCT["train_size"])
train_RCT_val = gen_RCT.run_simulation_normal(dataset_params_RCT["train_val_size"], seed_to_add=88)
##### Save data
save_data(dataset_params, data_folder + "\\" + dataset_params["filename"] + "_" + str(dataset_params["train_size"]) + "_dataset_params_BOZORGI_RealCause")
save_data(train_normal, data_folder + "\\" + dataset_params["filename"] + "_" + str(dataset_params["train_size"]) + "_train_normal_BOZORGI_RealCause")
save_data(train_normal_val, data_folder + "\\" + dataset_params["filename"] + "_" + str(dataset_params["train_size"]) + "_train_normal_val_BOZORGI_RealCause")
save_data(train_RCT, data_folder + "\\" + dataset_params["filename"] + "_" + str(dataset_params["train_size"]) + "_train_RCT_BOZORGI_RealCause")
save_data(train_RCT_val, data_folder + "\\" + dataset_params["filename"] + "_" + str(dataset_params["train_size"]) + "_train_RCT_val_BOZORGI_RealCause")