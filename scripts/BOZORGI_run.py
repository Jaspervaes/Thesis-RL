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

import numpy as np
from src.utils.tools import save_data, load_data
from copy import deepcopy
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import subprocess
from src.methods.BOZORGI.BOZORGI_evaluation import BOZORGIModelEvaluator
from src.methods.BOZORGI.BOZORGI_validation import BOZORGIModelValidator
# Load the data
intervention_name = ["time_contact_HQ"]
train_size = 100000
biased = False
# biased = True
# train_both_true_and_generated = False
train_both_true_and_generated = True
big_data = True
# big_data = False
iterations_to_skip = []
num_iterations = 5
plot_training = False
# plot_training = True
# calculate_other_policies = False
calculate_other_policies = True
# already_trained = True
already_trained = False
s_learner = True
# s_learner = False
realcause_already_trained = False
# realcause_already_trained = True
atoms_already_calculated = False
# atoms_already_calculated = True
data_already_generated = False
# data_already_generated = True



# Generate data
if not data_already_generated:
    subprocess.run(["python", path + "\\src\\methods\\BOZORGI\\BOZORGI_data_generation.py"])
print('Data generated', "\n")

# Calculate atoms
if not atoms_already_calculated:
    subprocess.run(["python", path + "\\src\\methods\\BOZORGI\\BOZORGI_calc_atoms.py", str(biased)])
print('Atoms calculated', "\n")

# Train RealCause
if not realcause_already_trained:
    subprocess.run(["python", path + "\\src\\methods\\BOZORGI\\BOZORGI_realcause.py", str(biased), str(train_size)])
print('RealCause trained', "\n")


dataset_params = load_data(os.path.join(os.getcwd(), data_folder, "loan_log_" + str(intervention_name) + "_" + str(train_size) + "_dataset_params"))
data_path = os.path.join(os.getcwd(), data_folder, "data", dataset_params["filename"] + "_" + str(dataset_params["train_size"]))
results_path = os.path.join(os.getcwd(), results_folder, "BOZORGI_" + dataset_params["filename"] + "_" + str(dataset_params["train_size"]))
train_size_realcause = 10000
results_path_realcause = os.path.join(os.getcwd(), results_folder, "BOZORGI_" + dataset_params["filename"] + "_" + str(train_size_realcause))

if biased:
    bias_path = '_biased'
else:
    bias_path = ''
prep_utils_list = load_data(os.path.join(os.getcwd(), data_folder, dataset_params["filename"] + "_" + str(train_size_realcause) + "_prep_utils_BOZORGI_RealCause" + bias_path))

int_dataset_params = []
for int_index, int_name in enumerate(intervention_name):
    params = deepcopy(dataset_params)
    for key, value in params["intervention_info"].items():
        if isinstance(value, list):
            params["intervention_info"][key] = value[int_index]
    int_dataset_params.append(params)












#TRAINING parameters
training_params = {}
#general
training_params["early_stopping_rounds"] = 1000
if dataset_params["intervention_info"]["action_depth_combinations"] > 1:
    training_params["tuning"] = True
else:
    training_params["tuning"] = False

if not big_data:
    training_params["early_stopping_rounds"] = 50
    training_params["n_estimators"] = 1000


#MODEL parameters
model_params = {}
model_params["n_estimators"] = 10000
model_params["max_depth"] = 15
model_params["learning_rate"] = 0.001
model_params["subsample"] = 0.7
model_params["colsample_bytree"] = 0.7
model_params["n_jobs"] = 4
model_params["random_seed"] = 42





to_generate_prep_list = load_data(os.path.join(os.getcwd(), data_folder, dataset_params["filename"] + "_" + str(train_size_realcause) + "_to_generate_prep_BOZORGI_RealCause" + bias_path))
to_generate_prep_val_list = load_data(os.path.join(os.getcwd(), data_folder, dataset_params["filename"] + "_" + str(train_size_realcause) + "_to_generate_prep_val_BOZORGI_RealCause" + bias_path))
train_prep_list = load_data(os.path.join(os.getcwd(), data_folder, dataset_params["filename"] + "_" + str(train_size_realcause) + "_train_prep_BOZORGI_RealCause" + bias_path))
print('len(train_prep_list[0]):', len(train_prep_list[0]))
print('len(to_generate_prep_list[0]):', len(to_generate_prep_list[0]))
print('len(to_generate_prep_val_list[0]):', len(to_generate_prep_val_list[0]))
print('\n')

realcause_model_list = []
for iteration in range(num_iterations):
    realcause_model = load_data(results_path_realcause + "model_" + str(int_index) + "_" + str(iteration) + "_BOZORGI_RealCause" + bias_path)
    realcause_model_list.append(realcause_model)

#INITIALIZE EVALUATOR AND VALIDATOR
CI_evaluator = BOZORGIModelEvaluator(
    int_dataset_params=int_dataset_params,
    full_dataset_params=dataset_params,
    prep_utils=prep_utils_list,
    realcause_model_list=realcause_model_list,
    s_learner=s_learner,
    # print_cases=True
    )

CI_validator = BOZORGIModelValidator(full_dataset_params=dataset_params, s_learner=s_learner)

if calculate_other_policies:
    for iteration in range(num_iterations):
        if iteration in iterations_to_skip:
            continue
        print("Random policy inference: ")
        random_performance, random_performance_realcause_per_iteration = CI_evaluator.random_policy_inference(n_cases=dataset_params["test_size"], iteration=iteration)
        print("Random performance: ", random_performance)
        print("Random performance realcause per iteration: ", random_performance_realcause_per_iteration, "\n")

        save_data(random_performance, results_path + "_random_performance_" + str(iteration) + bias_path)
        save_data(random_performance_realcause_per_iteration, results_path + "_random_performance_realcause_per_iteration_" + str(iteration) + bias_path)

    print("Bank policy inference: ")
    bank_performance, bank_performance_realcause_per_iteration = CI_evaluator.bank_policy_inference(n_cases=dataset_params["test_size"])
    print("Bank performance: ", bank_performance)
    print("Bank performance realcause per iteration: ", bank_performance_realcause_per_iteration, "\n")

    save_data(bank_performance, results_path + "_bank_performance" + bias_path)
    save_data(bank_performance_realcause_per_iteration, results_path + "_bank_performance_realcause_per_iteration" + bias_path)

#INITIALIZE PERFORMANCE DICTIONARIES
model_performance_dict = {}
model_avg_performance_dict = {}
model_stdev_performance_dict = {}
opt_th_dict = {}
model_dict = {}
model_per_iteration = {}
for iteration in range(num_iterations):
    if iteration in iterations_to_skip:
        continue
    generated_data_list = []
    generated_data_val_list = []
    
    model_list = []
    for int_index, int_name in enumerate(intervention_name):
        generated_data = load_data(results_path_realcause + "generated_data_" + str(int_index) + "_" + str(iteration) + "_BOZORGI_RealCause" + bias_path)
        generated_data_list.append(generated_data)
        generated_data_val = load_data(results_path_realcause + "generated_data_val_" + str(int_index) + "_" + str(iteration) + "_BOZORGI_RealCause" + bias_path)
        generated_data_val_list.append(generated_data_val)
        w_train, t_train, (y_train_0, y_train_1) = generated_data_list[int_index]
        w_val, t_val, (y_val_0, y_val_1) = generated_data_val_list[int_index]
        # convert to float32 to save memory
        w_train = w_train.astype(np.float32)
        t_train = t_train.astype(np.float32)
        y_train_0 = y_train_0.astype(np.float32)
        y_train_1 = y_train_1.astype(np.float32)
        w_val = w_val.astype(np.float32)
        t_val = t_val.astype(np.float32)
        y_val_0 = y_val_0.astype(np.float32)
        y_val_1 = y_val_1.astype(np.float32)
        
        data_true = to_generate_prep_list[int_index]
        t_true = data_true["treatment"]
        y_true = data_true["outcome"]
        w_true = data_true.drop(["treatment", "outcome"], axis=1)
        data_true_val = to_generate_prep_val_list[int_index]
        t_true_val = data_true_val["treatment"]
        y_true_val = data_true_val["outcome"]
        w_true_val = data_true_val.drop(["treatment", "outcome"], axis=1)
        # reshape
        t_true = np.array(t_true).reshape(-1, 1)
        y_true = np.array(y_true).reshape(-1, 1)
        w_true = np.array(w_true)
        t_true_val = np.array(t_true_val).reshape(-1, 1)
        y_true_val = np.array(y_true_val).reshape(-1, 1)
        w_true_val = np.array(w_true_val)
        # convert to float32 to save memory
        t_true = t_true.astype(np.float32)
        y_true = y_true.astype(np.float32)
        w_true = w_true.astype(np.float32)

        del data_true, data_true_val

        if s_learner:
            if train_both_true_and_generated:
                # first make sure the lengths are matching (just get the minimum length and delete the rest)
                min_len = min(len(y_true), len(y_train_0), len(y_train_1))
                y_true = y_true[:min_len]
                t_true = t_true[:min_len]
                w_true = w_true[:min_len]
                
                y_train_0 = y_train_0[:min_len]
                y_train_1 = y_train_1[:min_len]
                t_train = t_train[:min_len]
                w_train = w_train[:min_len]
                
                # Take the true y of generated data that corresponds to the not true t (so if t_true = 0, take y_train_1, and if t_true = 1, take y_train_0)
                y_gen = (1 - t_true) * y_train_1 + t_true * y_train_0
                # t_gen should be the opposite of t_true (so if t_true = 0, t_gen = 1, and if t_true = 1, t_gen = 0)
                t_gen = 1 - t_true
                X_gen = np.hstack((w_train, t_gen))
                X_true = np.hstack((w_train, t_true))
                # now I want to concatenate the true data with the generated data, and also shuffle them (but keep all data in the correct order)
                y_train = np.concatenate((y_gen, y_true))
                X_train = np.concatenate((X_gen, X_true))
                del y_gen, t_gen, X_gen, X_true, y_true, w_true, t_true, y_train_0, y_train_1, t_train, w_train
                # shuffle the data
                np.random.seed(iteration)
                shuffle_idx = np.random.permutation(len(y_train))
                y_train = y_train[shuffle_idx]
                X_train = X_train[shuffle_idx]

                # do the same for the validation data
                min_len = min(len(y_true_val), len(y_val_0), len(y_val_1))
                y_true_val = y_true_val[:min_len]
                t_true_val = t_true_val[:min_len]
                w_true_val = w_true_val[:min_len]

                y_val_0 = y_val_0[:min_len]
                y_val_1 = y_val_1[:min_len]
                t_val = t_val[:min_len]
                w_val = w_val[:min_len]

                y_gen_val = (1 - t_true_val) * y_val_1 + t_true_val * y_val_0
                t_gen_val = 1 - t_true_val
                X_gen_val = np.hstack((w_val, t_gen_val))
                X_true_val = np.hstack((w_val, t_true_val))
                y_val = np.concatenate((y_gen_val, y_true_val))
                X_val = np.concatenate((X_gen_val, X_true_val))
                del y_gen_val, t_gen_val, X_gen_val, X_true_val, y_true_val, w_true_val, t_true_val, y_val_0, y_val_1, t_val, w_val

                np.random.seed(iteration)
                shuffle_idx = np.random.permutation(len(y_val))
                y_val = y_val[shuffle_idx]
                X_val = X_val[shuffle_idx]
            else:
                y_train = (1 - t_train) * y_train_0 + t_train * y_train_1
                X_train = np.hstack((w_train, t_train))
                y_val = (1 - t_val) * y_val_0 + t_val * y_val_1
                X_val = np.hstack((w_val, t_val))
        else:
            y_train = y_train_1 - y_train_0
            X_train = w_train
            y_val = y_val_1 - y_val_0
            X_val = w_val

        print("Iteration:", iteration, ", Intervention:", int_name)
        print("Training data shape:", X_train.shape)
        print("Validation data shape:", X_val.shape)

        if not already_trained:
            # Define the XGBRegressor model with early stopping rounds
            xgb_model = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=model_params["n_estimators"],
                max_depth=model_params["max_depth"],
                learning_rate=model_params["learning_rate"],
                subsample=model_params["subsample"],
                colsample_bytree=model_params["colsample_bytree"],
                n_jobs=model_params["n_jobs"],
                random_state=model_params["random_seed"] + iteration*5
            )

            # Fit the model with early stopping and store evaluation results
            evals = [(X_train, y_train), (X_val, y_val)]
            history = xgb_model.fit(
                X_train, y_train,
                eval_set=evals,
                early_stopping_rounds=training_params["early_stopping_rounds"],
                verbose=False
            )

            rmse_train_list = history.evals_result()['validation_0']['rmse']
            mse_train = [rmse ** 2 for rmse in rmse_train_list]
            rmse_val_list = history.evals_result()['validation_1']['rmse']
            mse_val = [rmse ** 2 for rmse in rmse_val_list]

            print('Best MSE on training data:', min(mse_train))
            print('Best MSE on validation data:', min(mse_val))

            # Plotting the learning curves
            if plot_training:
                plt.figure(figsize=(10, 6))
                plt.plot(mse_train, label='Training MSE', color='green')
                plt.plot(mse_val, label='Validation MSE', color='red')
                plt.title(f'Learning Curve for {int_name}')
                plt.xlabel('Boosting Rounds')
                plt.ylabel('MSE')
                plt.legend()
                plt.grid()
                plt.show()

            model_list.append(xgb_model)
    
    if not already_trained:
        model_per_iteration[iteration] = model_list
        if big_data:
            save_data(model_per_iteration[iteration], results_path + "_model_class_iteration_" + str(iteration) + bias_path)
    else:
        model_per_iteration[iteration] = load_data(results_path + "_model_class_iteration_" + str(iteration) + bias_path)
        model_list = model_per_iteration[iteration]

     #TUNING
    if not already_trained:
        if training_params["tuning"]:
            opt_th, opt_obj = CI_validator.threshold_tuning(CI_evaluator=CI_evaluator, model_list=model_list, iteration=iteration, model_params=model_params)
        else:
            opt_th = 0
        opt_th_dict[iteration] = opt_th
        if big_data:
            save_data(opt_th, results_path + "_opt_th_iteration_" + str(iteration) + bias_path)
    else:
        opt_th = load_data(results_path + "_opt_th_iteration_" + str(iteration) + bias_path)
        opt_th_dict[iteration] = opt_th

    model_performance, model_performance_realcause_per_iteration = CI_evaluator.model_policy_inference(n_cases=dataset_params["test_size"], model_list=model_per_iteration[iteration], iteration=iteration, opt_th=opt_th)
    model_performance = float(model_performance)
    model_performance_dict[iteration] = model_performance

    print("Iteration: ", iteration, ", Full model performance: ", model_performance, ", Realcause model performance per iteration: ", model_performance_realcause_per_iteration, "\n")
    if big_data:
        save_data(model_performance, results_path + "_model_performance_iteration_" + str(iteration) + bias_path)
        save_data(model_performance_realcause_per_iteration, results_path + "_model_performance_realcause_per_iteration_iteration_" + str(iteration) + bias_path)
        print("Model action timings: ", CI_evaluator.model_action_timings, "\n")
        save_data(CI_evaluator.model_action_timings, results_path + "_model_action_timings_iteration_" + str(iteration) + bias_path)
        print("Model actions: ", CI_evaluator.model_actions, "\n")
        save_data(CI_evaluator.model_actions, results_path + "_model_actions_iteration_" + str(iteration) + bias_path)
        #save model test set df
        save_data(CI_evaluator.test_set_df, results_path + "_test_set_df_iteration_" + str(iteration) + bias_path)
