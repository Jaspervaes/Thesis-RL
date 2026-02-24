data_folder = "data"
results_folder = "res"
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from config.config import path
sys.path.append(path)
sys.path.append(path + "\\SimBank")
sys.path.append(path + "\\src\\methods\\BOZORGI")

from copy import deepcopy
from src.methods.BRANCHI.BRANCHI_data_preparation import LoanProcessPreprocessor
from src.methods.BRANCHI.BRANCHI_RL_evaluation import RLModelEvaluator
from src.utils.tools import save_data, load_data
from src.methods.BRANCHI.BRANCHI_RL_training import RLModel





#DATASET parameters
intervention_name = ["time_contact_HQ"]
train_size = 100000
# big_data = False
big_data = True
# already_preprocessed = False
already_preprocessed = True
if not big_data:
    already_preprocessed = False
iterations_to_skip = []
calc_other_policies_test = False
# calc_other_policies_test = True
num_iterations = 5
# already_trained = True
already_trained = False
calc_realcause_performance = True
# calc_realcause_performance = False
train_size_realcause = 10000
# calc_realcause_performance = False
bias_path = "" # Always not biased (online method)


dataset_params = load_data(os.path.join(os.getcwd(), data_folder, "loan_log_" + str(intervention_name) + "_" + str(train_size) + "_dataset_params"))
data_path = os.path.join(os.getcwd(), data_folder, dataset_params["filename"] + "_" + str(dataset_params["train_size"]))
results_path = os.path.join(os.getcwd(), results_folder, "RL_BRANCHI_" + dataset_params["filename"] + "_" + str(dataset_params["train_size"]))
results_path_realcause = os.path.join(os.getcwd(), results_folder, "BOZORGI_" + dataset_params["filename"] + "_" + str(train_size_realcause))
std_policies_path = os.path.join(os.getcwd(), results_folder)

int_dataset_params = []
for int_index, int_name in enumerate(intervention_name):
    params = deepcopy(dataset_params)
    for key, value in params["intervention_info"].items():
        if isinstance(value, list):
            params["intervention_info"][key] = value[int_index]
    int_dataset_params.append(params)

#MODEL parameters
model_params = {}
#general
model_params["device"] = "cpu"
# model_params["device"] = "cpu"
model_params["random_seed"] = 42
model_params["intervention_info"] = dataset_params["intervention_info"]
model_params["print_transitions"] = False
# model_params["print_transitions"] = True
# model_params["print_model_params"] = False
model_params["print_model_params"] = True


#TRAINING parameters
training_params = {}

training_params["eps_start"] = 0.99
training_params["eps_end"] = 0.25
training_params["eps_decay"] = 1000000 # * max_process_length       # in nr of transitions
training_params["penalty"] = 8000
# training_params["k_values"] = [100, 200, 300, 400, 500, 750]
# We know 750 is best k
training_params["k_values"] = [750]
training_params["no_wrong_int"] = False
training_params["alpha"] = 0.01
dataset_params["extra_do_nothing_action"] = False

training_params["filename"] = "RL_BRANCHI_training_" + str(dataset_params["intervention_info"]["name"])
training_params["early_stop_path"] = "C:/Users/u0166838/Music/"
training_params["train_size"] = dataset_params["train_size"]
training_params["val_share"] = dataset_params["val_share"]
training_params["gamma"] = 0.999999
training_params["early_stop"] = True
training_params["es_patience"] = 200
training_params["es_delta"] = .000001
training_params["time_adjustable_exploration"] = False
# training_params["time_adjustable_exploration"] = True
# training_params["eps_strategy"] = "exponential"
training_params["eps_strategy"] = "linear"
training_params["num_episodes"] = 100000
training_params["calc_val"] = 400                                    # in nr of episodes

if not big_data:
    training_params["num_episodes"] = 10
    training_params["calc_val"] = 9                                   # in nr of episodes
    training_params["eps_start"] = 0.99
    training_params["eps_end"] = 0.75
    training_params["eps_decay"] = 1000 # * max_process_length       # in nr of transitions
    training_params["k_values"] = [12]
    training_params["penalty"] = 8000








##### Load data
train_normal = load_data(data_path + "_train_RCT")
train_normal_val = load_data(data_path + "_train_RCT_val")
if not big_data:
    train_normal = train_normal[:300]
    train_normal_val = train_normal_val[:300]
# PREPROCESS
print('Preprocessing Started', '\n')
if not already_preprocessed:
    RL_prep_kmeans = LoanProcessPreprocessor(dataset_params=int_dataset_params[0],
                                             data_train=train_normal,
                                             data_train_val=train_normal_val)
    
    train_prep_k_means, prep_utils_kmeans = RL_prep_kmeans.preprocess_for_kmeans()
    # NOTE: train_prep_k_means will contain 10 000 cases (see function preprocess_for_kmeans)
    best_km_list = []
    best_k_list = []
    best_silhouette_list = []
    if big_data:
        save_data(train_prep_k_means, data_path + "_train_prep_kmeans_RCT" + bias_path)
        save_data(prep_utils_kmeans, data_path + "_prep_utils_kmeans_RCT" + bias_path)
    for iteration in range(num_iterations):
        if iteration in iterations_to_skip:
            continue
        best_km, best_k, best_silhouette = RL_prep_kmeans.kmeans_clustering(train_prep_k_means=train_prep_k_means, k_values=training_params["k_values"], random_seed = model_params["random_seed"] + iteration*5)
        best_km_list.append(best_km)
        best_k_list.append(best_k)
        best_silhouette_list.append(best_silhouette)
    if big_data:
        save_data(best_km_list, data_path + "_best_km_list_RCT" + bias_path)
        save_data(best_k_list, data_path + "_best_k_list_RCT" + bias_path)
        save_data(best_silhouette_list, data_path + "_best_silhouette_list_RCT" + bias_path)
##### Load prep data
if big_data:
    train_prep_kmeans = load_data(data_path + "_train_prep_kmeans_RCT" + bias_path)
    prep_utils_kmeans = load_data(data_path + "_prep_utils_kmeans_RCT" + bias_path)
    best_km_list = load_data(data_path + "_best_km_list_RCT" + bias_path)
    best_k_list = load_data(data_path + "_best_k_list_RCT" + bias_path)
    best_silhouette_list = load_data(data_path + "_best_silhouette_list_RCT" + bias_path)
print('Best K values: ', best_k_list, '\n')
print('Preprocessing Done', '\n')



















realcause_model_list = []
prep_utils_realcause = []
if intervention_name == ['time_contact_HQ'] and calc_realcause_performance:
    for iteration in range(num_iterations):
        if intervention_name == ['time_contact_HQ']:
            realcause_model = load_data(results_path_realcause + "model_" + str(0) + "_" + str(iteration) + "_BOZORGI_RealCause" + bias_path)
            realcause_model_list.append(realcause_model)
    prep_utils_realcause = load_data(os.path.join(os.getcwd(), data_folder, dataset_params["filename"] + "_" + str(train_size_realcause) + "_prep_utils_BOZORGI_RealCause" + bias_path))

##### Empty cache
import torch
torch.cuda.empty_cache()
##### Run
#INITIALIZE EVALUATOR AND VALIDATOR
RL_evaluator = RLModelEvaluator(
    model_params=model_params,
    int_dataset_params=int_dataset_params,
    full_dataset_params=dataset_params,
    prep_utils=prep_utils_kmeans,
    realcause_model_list=realcause_model_list,
    realcause_prep_utils=prep_utils_realcause,
    # print_cases=True,
    # print_transitions=True
    )

#GET BASELINE PERFORMANCE
if calc_other_policies_test:
    bank_performance = RL_evaluator.bank_policy_inference(n_cases=dataset_params["test_size"])
    print("Bank performance: ", bank_performance, "\n")
    random_performance = RL_evaluator.random_policy_inference(n_cases=dataset_params["test_size"])
    print("Random performance: ", random_performance, "\n")
bank_performance_test_val = RL_evaluator.bank_policy_inference(n_cases=dataset_params["test_val_size"], validation=True)
print("Bank performance test_val: ", bank_performance_test_val, "\n")
random_performance_test_val = RL_evaluator.random_policy_inference(n_cases=dataset_params["test_val_size"], validation=True)
print("Random performance test_val: ", random_performance_test_val, "\n")
baseline_performances_test_val_dict = {"bank": bank_performance_test_val, "random": random_performance_test_val}

#INITIALIZE PERFORMANCE DICTIONARIES
model_performance_list = []
model_stdev_performance_list = []
model_list = []
model_best_step_list = []

for iteration in range(num_iterations):
    if iteration in iterations_to_skip:
        continue
    print("Iteration: ", iteration, "\n")
    training_params["filename"] = "RL_BRANCHI_loan_log_" +  str(dataset_params["intervention_info"]["name"]) + "_" + str(dataset_params["train_size"]) + "_iteration_" + str(iteration)

    #INITIALIZE MODEL
    RL_model = RLModel(
        model_params=model_params,
        full_dataset_params=dataset_params,
        int_dataset_params=int_dataset_params,
        kmeans = best_km_list[iteration],
        best_k=best_k_list[iteration],
        prep_utils=prep_utils_kmeans,
        iteration=iteration,
        baseline_performances_test_val_dict=baseline_performances_test_val_dict,
        validator=RL_evaluator,
        # print_transitions=True
        )
    
    #TRAINING
    if not already_trained:
        RL_model.start_training(training_params=training_params, iteration=iteration)
        model_list.append(RL_model.final_q_table)
        model_best_step_list.append(RL_model.best_step)
        if big_data:
            save_data(RL_model.final_q_table, results_path + "_model_iteration_" + str(iteration) + bias_path)
            save_data(RL_model.best_step, results_path + "_best_step_iteration_" + str(iteration) + bias_path)

        torch.cuda.empty_cache()
    else:
        RL_model.init_train_params(training_params=training_params, iteration=iteration)
        RL_model.final_q_table = load_data(results_path + "_model_iteration_" + str(iteration) + bias_path)
        RL_model.best_step = load_data(results_path + "_best_step_iteration_" + str(iteration) + bias_path)
        model_list.append(RL_model.final_q_table)
        model_best_step_list.append(RL_model.best_step)
        print("Model loaded from file", "\n")

    #GET MODEL PERFORMANCE
    model_performance, model_realcause_performance = RL_evaluator.model_policy_inference(n_cases=dataset_params["test_size"], model_class=RL_model, q_table=RL_model.final_q_table, kmeans=best_km_list[iteration], calc_realcause_performance=calc_realcause_performance, iteration=iteration)
    model_performance_list.append(model_performance)
    print("Iteration: ", iteration, ", Full model performance and efficiency: ", model_performance, '; ', RL_model.best_step, ", Realcause performance: ", model_realcause_performance, "\n")
    if big_data:
        save_data(model_performance, results_path + "_model_performance_iteration_" + str(iteration) + bias_path)
        if calc_realcause_performance:
            save_data(model_realcause_performance, results_path + "_model_realcause_performance_iteration_" + str(iteration) + bias_path)

        print("Model action timings: ",RL_evaluator.model_action_timings, "\n")
        save_data(RL_evaluator.model_action_timings, results_path + "_model_action_timings_iteration_" + str(iteration) + bias_path)
        print("Model actions: ", RL_evaluator.model_actions, "\n")
        save_data(RL_evaluator.model_actions, results_path + "_model_actions_iteration_" + str(iteration) + bias_path)
        # save test set df
        save_data(RL_evaluator.test_set_df, results_path + "_test_set_df_iteration_" + str(iteration) + bias_path)