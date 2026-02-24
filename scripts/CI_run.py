data_folder = "data"
results_folder = "res"
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from config.config import path
sys.path.append(path)
sys.path.append(path + "\\SimBank")

from copy import deepcopy
from src.methods.CI.CI_data_preparation import LoanProcessPreprocessor
from SimBank.confounding_level import set_delta
from src.utils.tools import save_data, load_data
from src.methods.CI.CI_training import CIModel
from src.methods.CI.CI_evaluation import CIModelEvaluator
from src.methods.CI.CI_validation import CIModelValidator













#DATASET parameters
intervention_name = ["choose_procedure"]
# intervention_name = ["set_ir_3_levels"]
# intervention_name = ["time_contact_HQ"]
# intervention_name = ["choose_procedure", "set_ir_3_levels"]
train_size = 100000
train_size_realcause = 10000
# train_size = 10
# big_data = False
big_data = True
already_preprocessed = False
# already_preprocessed = True
calculate_other_policies_test = True
# calculate_other_policies_test = False
iterations_to_skip = []
biased = False
# biased = True
alread_trained = False
# alread_trained = True
# calculate_realcause_performance = False
calculate_realcause_performance = True


if biased:
    delta_level_bias = 0.999
    bias_path = '_biased'
else:
    bias_path = ''
dataset_params = load_data(os.path.join(os.getcwd(), data_folder, "loan_log_" + str(intervention_name) + "_" + str(train_size) + "_dataset_params"))
data_path = os.path.join(os.getcwd(), data_folder, dataset_params["filename"] + "_" + str(dataset_params["train_size"]))
results_path = os.path.join(os.getcwd(), results_folder, "CI_" + dataset_params["filename"] + "_" + str(dataset_params["train_size"]))
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
model_params["random_seed"] = 42
model_params["intervention_info"] = dataset_params["intervention_info"]

dataset_params["time_wise"] = True

#TRAINING parameters
training_params = {}
#general
training_params["train_size"] = dataset_params["train_size"]
training_params["val_share"] = dataset_params["val_share"]
training_params["batch_size"] = 256

training_params["calc_val"] = 200                                    # in nr of episodes
training_params["earlystop"] = True
training_params["early_stop_path"] = "C:/Users/u0166838/Music/"
training_params["es_patience"] = 50
training_params["es_delta"] = .0001
training_params["nb_epochs"] = 100000
training_params["aleatoric"] = True
training_params["verbose"] = True
training_params["nr_future_est"] = None
if dataset_params["intervention_info"]["action_depth_combinations"] > 1:
    training_params["tuning"] = True
else:
    training_params["tuning"] = False

if not big_data:
    training_params["nb_epochs"] = 3
    training_params["calc_val"] = 2
    training_params["tuning"] = False
    dataset_params['test_size'] = 10























## Set up experiment
##### Load data
train_normal = load_data(data_path + "_train_normal")
train_normal_val = load_data(data_path + "_train_normal_val")
train_RCT = load_data(data_path + "_train_RCT")
train_RCT_val = load_data(data_path + "_train_RCT_val")

if not big_data:
    train_normal = train_normal[:100]
    train_normal_val = train_normal_val[:100]
    train_RCT = train_RCT[:100]
    train_RCT_val = train_RCT_val[:100]

exp = "RCT"
delta_levels = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999]

if biased:
    delta_levels = [delta_level_bias]

print(delta_levels)
train_dict = {}
train_val_dict = {}
test_val_nested_list_dict = {}

for delta in delta_levels:
    train_dict[delta] = set_delta(data=train_normal,
                                                data_RCT=train_RCT,
                                                delta=delta)
    train_val_dict[delta] = set_delta(data=train_normal_val,
                                                    data_RCT=train_RCT_val,
                                                    delta=delta)
exp_keys = delta_levels

if biased:
    if big_data:
        save_data(train_dict[delta_level_bias], data_path + "_train_biased")
        save_data(train_val_dict[delta_level_bias], data_path + "_train_biased_val")











## Preprocess
train_prep = []
train_val_prep = []
prep_utils = []
print('Preprocessing started', '\n')
if not already_preprocessed:
    for int_index, intervention in enumerate(dataset_params["intervention_info"]["name"]):
        # PREPROCESS
        train_prep_dict = {}
        train_val_prep_dict = {}
        test_prep_pred_nested_list_dict = {}
        test_val_prep_pred_nested_list_dict = {}
        test_policy_nested_list_dict = {}
        prep_utils_dict = {}
        for key in exp_keys:
            CI_prep = LoanProcessPreprocessor(dataset_params=int_dataset_params[int_index],
                                            data_train=train_dict[key], 
                                            data_train_val=train_val_dict[key], time_wise=dataset_params["time_wise"])
            train_prep_dict[key], train_val_prep_dict[key], prep_utils_dict[key] = CI_prep.preprocess()
        
        train_prep.append(train_prep_dict)
        train_val_prep.append(train_val_prep_dict)
        prep_utils.append(prep_utils_dict)

    ##### Save prep data
    if big_data:
        save_data(train_prep, data_path + "_train_prep" + bias_path)
        save_data(train_val_prep, data_path + "_train_val_prep" + bias_path)
        save_data(prep_utils, data_path + "_prep_utils" + bias_path)

print('Preprocessing done', '\n')
##### Load prep data
if big_data:
    train_prep = load_data(data_path + "_train_prep" + bias_path)
    train_val_prep = load_data(data_path + "_train_val_prep" + bias_path)
    prep_utils = load_data(data_path + "_prep_utils" + bias_path)



















##### Empty cache
import torch
torch.cuda.empty_cache()
##### Run
num_iterations = 5

realcause_model_list = []
prep_utils_realcause = []
if intervention_name == ['time_contact_HQ'] and calculate_realcause_performance:
    for iteration in range(num_iterations):
        if intervention_name == ['time_contact_HQ']:
            realcause_model = load_data(results_path_realcause + "model_" + str(0) + "_" + str(iteration) + "_BOZORGI_RealCause" + bias_path)
            realcause_model_list.append(realcause_model)
    prep_utils_realcause = load_data(os.path.join(os.getcwd(), data_folder, dataset_params["filename"] + "_" + str(train_size_realcause) + "_prep_utils_BOZORGI_RealCause" + bias_path))
    

#INITIALIZE EVALUATOR AND VALIDATOR
CI_evaluator = CIModelEvaluator(
    model_params=model_params,
    int_dataset_params=int_dataset_params,
    full_dataset_params=dataset_params,
    prep_utils=prep_utils,
    realcause_model_list=realcause_model_list,
    realcause_prep_utils=prep_utils_realcause,
    # print_cases=True
    )

CI_validator = CIModelValidator(full_dataset_params=dataset_params)

#GET BASELINE PERFORMANCE
if calculate_other_policies_test:
    bank_performance = CI_evaluator.bank_policy_inference(n_cases=dataset_params["test_size"])
    print("Bank performance: ", bank_performance, "\n")
    save_data(bank_performance, std_policies_path + "\\Bank\\" + + dataset_params["filename"] + "_" + str(dataset_params["train_size"]) + "_bank_performance" + bias_path)

    for iteration in range(num_iterations):
        if iteration in iterations_to_skip:
            continue
        random_performance = CI_evaluator.random_policy_inference(n_cases=dataset_params["test_size"], iteration=iteration)
        print("Random performance: ", random_performance, "\n")
        save_data(random_performance, std_policies_path + "\\Random\\" + dataset_params["filename"] + "_" + str(dataset_params["train_size"]) + "_random_performance_" + str(iteration) + bias_path)

#INITIALIZE PERFORMANCE DICTIONARIES
model_performance_dict = {}
model_avg_performance_dict = {}
model_stdev_performance_dict = {}
opt_th_dict = {}
model_dict = {}
model_per_key_per_iteration = {}

for key in exp_keys:
    model_performance_dict[key] = []
    opt_th_dict[key] = []
    model_per_key_per_iteration[key] = {}
    for iteration in range(num_iterations):
        if iteration in iterations_to_skip:
            continue
        CI_model_list = [None] * len(dataset_params["intervention_info"]["name"])
        if not alread_trained:
            for int_index, intervention in enumerate(dataset_params["intervention_info"]["name"]):
                training_params["filename"] = "loan_log_" +  str(dataset_params["intervention_info"]["name"]) + "_" + str(dataset_params["train_size"]) + "_" + exp + "_iteration_" + str(iteration) + "_intervention_" + str(intervention)
                
                train_prep_dict = train_prep[int_index]
                train_val_prep_dict = train_val_prep[int_index]
                prep_utils_dict = prep_utils[int_index]

                #INITIALIZE MODEL
                CI_model_list[int_index] = CIModel(
                    model_params=model_params,
                    int_dataset_params=int_dataset_params[int_index],
                    full_dataset_params=dataset_params,
                    data_train=train_prep_dict[key],
                    data_train_val=train_val_prep_dict[key],
                    prep_utils=prep_utils_dict[key],
                    iteration=iteration)

                #TRAINING
                CI_model_list[int_index].start_training(training_params=training_params, key=key, iteration=iteration)

                torch.cuda.empty_cache()

            model_per_key_per_iteration[key][iteration] = CI_model_list
            efficiency = 0
            for model in CI_model_list:
                efficiency += model.best_optimization_step
            if big_data:
                CI_actual_model_list = []
                for model in CI_model_list:
                    CI_actual_model_list.append(model.model)
                torch.save([net.state_dict() for net in CI_actual_model_list], results_path + "_model_state_iteration_" + str(iteration) + "_key_" + str(key) + bias_path)
                save_data(efficiency, results_path + "_model_efficiency_iteration_" + str(iteration) + "_key_" + str(key) + bias_path)
                
            #TUNING
            if training_params["tuning"]:
                opt_th, opt_obj = CI_validator.threshold_tuning(CI_evaluator=CI_evaluator, model_list=CI_model_list, key=key, iteration=iteration, model_params=model_params)
            else:
                opt_th = 0
            opt_th_dict[key].append(opt_th)
            if big_data:
                save_data(opt_th, results_path + "_opt_th_iteration_" + str(iteration) + "_key_" + str(key) + bias_path)

            torch.cuda.empty_cache()
        else:
            for int_index, intervention in enumerate(dataset_params["intervention_info"]["name"]):
                training_params["filename"] = "loan_log_" +  str(dataset_params["intervention_info"]["name"]) + "_" + str(dataset_params["train_size"]) + "_" + exp + "_iteration_" + str(iteration) + "_intervention_" + str(intervention)
                train_prep_dict = train_prep[int_index]
                train_val_prep_dict = train_val_prep[int_index]
                prep_utils_dict = prep_utils[int_index]
                print("train prep dict", train_prep_dict)
                #INITIALIZE MODEL
                CI_model_list[int_index] = CIModel(
                    model_params=model_params,
                    int_dataset_params=int_dataset_params[int_index],
                    full_dataset_params=dataset_params,
                    data_train=train_prep_dict[key],
                    data_train_val=train_val_prep_dict[key],
                    prep_utils=prep_utils_dict[key],
                    iteration=iteration)


            CI_actual_model_list = []
            for model in CI_model_list:
                CI_actual_model_list.append(model.model)
            for net, params in zip(CI_actual_model_list, torch.load(results_path + "_model_state_iteration_" + str(iteration) + "_key_" + str(key) + bias_path)):
                net.load_state_dict(params)
            for index, model in enumerate(CI_actual_model_list):
                CI_model_list[index].model = model
            opt_th = load_data(results_path + "_opt_th_iteration_" + str(iteration) + "_key_" + str(key) + bias_path)
            efficiency = load_data(results_path + "_model_efficiency_iteration_" + str(iteration) + "_key_" + str(key) + bias_path)
            model_per_key_per_iteration[key][iteration] = CI_model_list
            print('Model loaded', '\n')
        
        #GET MODEL PERFORMANCE
        model_performance, model_realcause_performance = CI_evaluator.model_policy_inference(n_cases=dataset_params["test_size"], model_list=model_per_key_per_iteration[key][iteration], key=key, iteration=iteration, opt_th=opt_th, calculate_realcause_performance=calculate_realcause_performance)
        # make sure model_performance is a float
        model_performance = float(model_performance)
        model_performance_dict[key].append(model_performance)
        print("Key: ", key, ", Iteration: ", iteration, ", Full model performance and efficiency: ", model_performance, "; ", efficiency, ", Realcause performance: ", model_realcause_performance, "\n")
        if big_data:
            save_data(model_performance, results_path + "_model_performance_iteration_" + str(iteration) + "_key_" + str(key) + bias_path)
            if calculate_realcause_performance:
                save_data(model_realcause_performance, results_path + "_model_realcause_performance_iteration_" + str(iteration) + "_key_" + str(key) + bias_path)
            print("Model action timings: ", CI_evaluator.model_action_timings, "\n")
            save_data(CI_evaluator.model_action_timings, results_path + "_model_action_timings_iteration_" + str(iteration) + "_key_" + str(key) + bias_path)
            print("Model actions: ", CI_evaluator.model_actions, "\n")
            save_data(CI_evaluator.model_actions, results_path + "_model_actions_iteration_" + str(iteration) + "_key_" + str(key) + bias_path)
            #save model test set df
            save_data(CI_evaluator.test_set_df, results_path + "_test_set_df_iteration_" + str(iteration) + "_key_" + str(key) + bias_path)