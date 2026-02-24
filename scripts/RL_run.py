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
from src.methods.RL.RL_data_preparation import LoanProcessPreprocessor
from src.utils.tools import save_data, load_data
from src.methods.RL.RL_training import RLModel
from src.methods.RL.RL_evaluation import RLModelEvaluator



# big_data = False
big_data = True
already_preprocessed = False
# already_preprocessed = True
iterations_to_skip = []
calc_other_policies_test = False
# calc_other_policies_test = True
already_trained = False
# already_trained = True
# calc_realcause_performance = False
calc_realcause_performance = True
bias_realcause_path = '' # Always not biased (online method)
train_size_realcause = 10000

#DATASET parameters
# intervention_name = ["choose_procedure"]
# intervention_name = ["set_ir_3_levels"]
intervention_name = ["time_contact_HQ"]
train_size = 100000
dataset_params = load_data(os.path.join(os.getcwd(), data_folder, "loan_log_" + str(intervention_name) + "_" + str(train_size) + "_dataset_params"))
data_path = os.path.join(os.getcwd(), data_folder, dataset_params["filename"] + "_" + str(dataset_params["train_size"]))
results_path = os.path.join(os.getcwd(), results_folder, "RL_" + dataset_params["filename"] + "_" + str(dataset_params["train_size"]))
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
model_params["device"] = "cuda"
# model_params["device"] = "cpu"
model_params["nr_lstm_layers"] = 2
model_params["lstm_size"] = 25
model_params["nr_dense_layers"] = 2
model_params["dense_width"] = 20
model_params["p"] = 0.0
model_params["penalty"] = -100000
model_params["random_seed"] = 42
model_params["intervention_info"] = dataset_params["intervention_info"]
model_params["print_transitions"] = False
# model_params["print_transitions"] = True
# model_params["print_model_params"] = False
model_params["print_model_params"] = True


#TRAINING parameters
training_params = {}

training_params["filename"] = "RL_training_" + str(dataset_params["intervention_info"]["name"])
training_params["early_stop_path"] = "C:/Users/u0166838/Music/"
training_params["train_size"] = dataset_params["train_size"]
training_params["val_share"] = dataset_params["val_share"]
training_params["gamma"] = 0.999999
training_params["aleatoric"] = True
training_params["early_stop"] = True

training_params["es_patience"] = 100000
training_params["es_delta"] = .000001
training_params["transitions_per_optimization"] = 4
training_params["time_adjustable_exploration"] = False
training_params["prioritized_replay"] = False
training_params["loss_function"] = "mse"
training_params["batch_size"] = 256
# training_params["eps_strategy"] = "exponential"
training_params["eps_strategy"] = "linear"
training_params["num_episodes"] = 10000
training_params["calc_val"] = 400                                    # in nr of episodes

if dataset_params["intervention_info"]["name"] == ["choose_procedure"]:
    training_params["calc_val"] = 400                                    # in nr of episodes
    training_params["es_patience"] = 200
    training_params["num_episodes"] = 100000
    training_params["batch_size"] = 256
    training_params["eps_strategy"] = "linear"
    training_params["time_adjustable_exploration"] = False
    training_params["prioritized_replay"] = False
    training_params["transitions_per_optimization"] = 4
    training_params["eps_start"] = 0.99
    training_params["eps_end"] = 0.25
    training_params["eps_decay"] = 1000000 # * max_process_length       # in nr of transitions
    training_params["target_update"] = 25                                # in nr of episodes
    training_params["tau"] = 0.005
    training_params["penalty"] = 8000
    training_params["no_wrong_int"] = False
    training_params["no_int_at_no_int_point"] = False
    training_params["memory_size"] = 3 * training_params["batch_size"] * 13  # in nr of transitions
elif dataset_params["intervention_info"]["name"] == ["set_ir_3_levels"]:
    training_params["calc_val"] = 400                                    # in nr of episodes
    training_params["es_patience"] = 200
    training_params["num_episodes"] = 100000
    training_params["batch_size"] = 256
    training_params["eps_strategy"] = "linear"
    training_params["time_adjustable_exploration"] = False
    training_params["prioritized_replay"] = False
    training_params["transitions_per_optimization"] = 4
    training_params["eps_start"] = 0.99
    training_params["eps_end"] = 0.25
    training_params["eps_decay"] = 1000000 # * max_process_length       # in nr of transitions
    training_params["target_update"] = 25                                # in nr of episodes
    training_params["tau"] = 0.005
    training_params["penalty"] = 8000
    training_params["no_wrong_int"] = False
    training_params["no_int_at_no_int_point"] = False
    training_params["memory_size"] = 3 * training_params["batch_size"] * 13  # in nr of transitions
elif dataset_params["intervention_info"]["name"] == ["time_contact_HQ"]:
    training_params["calc_val"] = 400                                    # in nr of episodes
    training_params["es_patience"] = 200
    training_params["num_episodes"] = 100000
    training_params["batch_size"] = 256
    training_params["eps_strategy"] = "linear"
    training_params["time_adjustable_exploration"] = False
    training_params["prioritized_replay"] = False
    training_params["transitions_per_optimization"] = 4
    training_params["eps_start"] = 0.99
    training_params["eps_end"] = 0.25
    training_params["eps_decay"] = 1000000 # * max_process_length       # in nr of transitions
    training_params["target_update"] = 25                                # in nr of episodes
    training_params["tau"] = 0.005
    training_params["penalty"] = 8000
    training_params["no_wrong_int"] = False
    training_params["no_int_at_no_int_point"] = False
    training_params["memory_size"] = 3 * training_params["batch_size"] * 13  # in nr of transitions

if not big_data:
    training_params["num_episodes"] = 50
    training_params["calc_val"] = 49                                   # in nr of episodes
    training_params["batch_size"] = 8
    training_params["eps_start"] = 0.99
    training_params["eps_end"] = 0.05
    training_params["eps_decay"] = 100000000 # * max_process_length       # in nr of transitions
    training_params["target_update"] = 2                                 # in nr of episodes
    training_params["tau"] = 1








##### Load data
train_normal = load_data(data_path + "_train_RCT")
train_normal_val = load_data(data_path + "_train_RCT_val")
if not big_data:
    train_normal = train_normal[:300]
    train_normal_val = train_normal_val[:300]
## Preprocess
# PREPROCESS
print('Preprocessing Started', '\n')
if not already_preprocessed:
    RL_prep = LoanProcessPreprocessor(dataset_params=int_dataset_params[0], # Just take the first one, does not matter
                                    data_train=train_normal, 
                                    data_train_val=train_normal_val)
    train_prep, train_val_prep, prep_utils = RL_prep.preprocess()
    if big_data:
        save_data(train_prep, data_path + "_train_prep_RCT")
        save_data(train_val_prep, data_path + "_train_val_prep_RCT")
        save_data(prep_utils, data_path + "_prep_utils_RCT")
##### Load prep data
if big_data:
    train_prep = load_data(data_path + "_train_prep_RCT")
    train_val_prep = load_data(data_path + "_train_val_prep_RCT")
    prep_utils = load_data(data_path + "_prep_utils_RCT")
print('Preprocessing Done', '\n')
















num_iterations = 5
realcause_model_list = []
prep_utils_realcause = []
if intervention_name == ['time_contact_HQ'] and calc_realcause_performance:
    for iteration in range(num_iterations):
        if intervention_name == ['time_contact_HQ']:
            realcause_model = load_data(results_path_realcause + "model_" + str(0) + "_" + str(iteration) + "_BOZORGI_RealCause" + bias_realcause_path)
            realcause_model_list.append(realcause_model)
    prep_utils_realcause = load_data(os.path.join(os.getcwd(), data_folder, dataset_params["filename"] + "_" + str(train_size_realcause) + "_prep_utils_BOZORGI_RealCause" + bias_realcause_path))

##### Empty cache
import torch
torch.cuda.empty_cache()
##### Run
#INITIALIZE EVALUATOR AND VALIDATOR
RL_evaluator = RLModelEvaluator(
    model_params=model_params,
    int_dataset_params=int_dataset_params,
    full_dataset_params=dataset_params,
    prep_utils=prep_utils,
    realcause_model_list=realcause_model_list,
    realcause_prep_utils=prep_utils_realcause,
    # print_cases=True,
    # print_transitions=True
    )

#GET BASELINE PERFORMANCE
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
    training_params["filename"] = "loan_log_" +  str(dataset_params["intervention_info"]["name"]) + "_" + str(dataset_params["train_size"]) + "_iteration_" + str(iteration)

    #INITIALIZE MODEL
    RL_model = RLModel(
        model_params=model_params,
        full_dataset_params=dataset_params,
        int_dataset_params=int_dataset_params,
        prep_utils=prep_utils,
        iteration=iteration,
        baseline_performances_test_val_dict=baseline_performances_test_val_dict,
        validator=RL_evaluator,
        # print_transitions=True
        )

    if not already_trained:
        #TRAINING
        RL_model.start_training(training_params=training_params, iteration=iteration)
        RL_model.plot_learning()
        model_list.append(RL_model.final_net)
        model_best_step_list.append(RL_model.best_step)
        if big_data:
            torch.save(RL_model.final_net.state_dict(), results_path + "_model_state_iteration_" + str(iteration), _use_new_zipfile_serialization=False)
            save_data(RL_model.best_step, results_path + "_best_step_iteration_" + str(iteration))
            save_data(RL_model.best_optimization_step, results_path + "_efficiency_iteration_" + str(iteration))

        torch.cuda.empty_cache()
    
    else:
        RL_model.init_train_params(training_params=training_params, iteration=iteration)
        RL_model.load_best_model(model_path=results_path + "_model_state_iteration_" + str(iteration))
        RL_model.best_step = load_data(results_path + "_best_step_iteration_" + str(iteration))
        RL_model.best_optimization_step = load_data(results_path + "_efficiency_iteration_" + str(iteration))
        model_list.append(RL_model.final_net)
        model_best_step_list.append(RL_model.best_step)
        print("Model loaded from file", "\n")

    #GET MODEL PERFORMANCE
    model_performance, model_realcause_performance = RL_evaluator.model_policy_inference(n_cases=dataset_params["test_size"], device="cpu", model_class=RL_model, model=RL_model.final_net, calc_realcause_performance=calc_realcause_performance, iteration=iteration)
    model_performance_list.append(model_performance)
    print("Iteration: ", iteration, ", Full model performance and efficiency: ", model_performance, '; ', RL_model.best_optimization_step, ", Realcause model performance: ", model_realcause_performance, "\n")
    if big_data:
        save_data(model_performance, results_path + "_model_performance_iteration_" + str(iteration))
        if calc_realcause_performance:
            save_data(model_realcause_performance, results_path + "_model_realcause_performance_iteration_" + str(iteration))
        print("Model action timings: ",RL_evaluator.model_action_timings, "\n")
        save_data(RL_evaluator.model_action_timings, results_path + "_model_action_timings_iteration_" + str(iteration))
        # save test set df
        save_data(RL_evaluator.test_set_df, results_path + "_test_set_df_iteration_" + str(iteration))