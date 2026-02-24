data_folder = "data"
results_folder = "res"
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(parent_dir)
from config.config import path
sys.path.append(path)
sys.path.append(path + "\\SimBank")

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from src.methods.BOZORGI.BOZORGI_data_preparation import LoanProcessPreprocessor
import torch
from BOZORGI_models import TarNet, TrainingParams, MLPParams
from BOZORGI_models import distributions
from src.utils.tools import save_data, load_data
from copy import deepcopy
from SimBank.confounding_level import set_delta








# Load the data
intervention_name = ["time_contact_HQ"]
train_size = 10000
biased = sys.argv[1] == "True"
train_size_to_generate = int(sys.argv[2])

dataset_params = load_data(os.path.join(os.getcwd(), data_folder, "loan_log_" + str(intervention_name) + "_" + str(train_size) + "_dataset_params_BOZORGI_RealCause"))
data_path = os.path.join(os.getcwd(), data_folder, dataset_params["filename"] + "_" + str(dataset_params["train_size"]))
results_path = os.path.join(os.getcwd(), results_folder, "BOZORGI_" + dataset_params["filename"] + "_" + str(dataset_params["train_size"]))



if biased:
    delta = 0.999
    train_normal = load_data(data_path + "_train_normal" + "_BOZORGI_RealCause")
    RCT = load_data(data_path + "_train_RCT" + "_BOZORGI_RealCause")
    train_RCT = set_delta(data=train_normal, data_RCT=RCT, delta=delta)
    data_path_to_generate = os.path.join(os.getcwd(), data_folder, dataset_params["filename"] + "_" + str(train_size_to_generate))
    to_generate_RCT = load_data(data_path_to_generate + "_train_biased")
    to_generate_RCT_val = load_data(data_path_to_generate + "_train_biased_val")
    bias_path = '_biased'
else:
    train_RCT = load_data(data_path + "_train_RCT" + "_BOZORGI_RealCause")
    data_path_to_generate = os.path.join(os.getcwd(), data_folder, dataset_params["filename"] + "_" + str(train_size_to_generate))
    to_generate_RCT = load_data(data_path_to_generate + "_train_RCT")
    to_generate_RCT_val = load_data(data_path_to_generate + "_train_RCT_val")
    bias_path = ''

int_dataset_params = []
for int_index, int_name in enumerate(intervention_name):
    params = deepcopy(dataset_params)
    for key, value in params["intervention_info"].items():
        if isinstance(value, list):
            params["intervention_info"][key] = value[int_index]
    int_dataset_params.append(params)


data_size = train_size
already_preprocessed = False
# already_preprocessed = True
# only_preprocess_atoms = True
only_preprocess_atoms = False
only_preprocess_realcause_data = False
# only_preprocess_realcause_data = True
already_trained = False
# already_trained = True
# big_data = False
big_data = True
# bozorgi_retaining = False
bozorgi_retaining = True
iterations_to_skip = []
num_iterations = 5









# Preprocess the data
print("Preprocessing started")
train_prep_list = []
prep_utils_list = []
to_generate_prep_list = []
to_generate_prep_val_list = []
if not already_preprocessed:
    atoms = load_data(os.path.join(os.getcwd(), data_folder, "grouped_averages_" + str(intervention_name) + "_" + str(train_size) + "_BOZORGI_RealCause" + bias_path))
    bin_with = load_data(os.path.join(os.getcwd(), data_folder, "bin_width_" + str(intervention_name) + "_" + str(train_size) + "_BOZORGI_RealCause" + bias_path))
    stdev_atoms = load_data(os.path.join(os.getcwd(), data_folder, "grouped_stds_" + str(intervention_name) + "_" + str(train_size) + "_BOZORGI_RealCause" + bias_path))
    split_list = load_data(os.path.join(os.getcwd(), data_folder, "split_list_" + str(intervention_name) + "_" + str(train_size) + "_BOZORGI_RealCause" + bias_path))
    for int_index, int in enumerate(int_dataset_params):
        prep = LoanProcessPreprocessor(dataset_params=int_dataset_params[int_index], data_train=train_RCT, data_to_generate=to_generate_RCT, data_to_generate_val=to_generate_RCT_val, big_data=big_data, atoms=atoms, bin_with=bin_with, stdev_atoms=stdev_atoms, split_list=split_list, data_size=data_size)
        if only_preprocess_atoms:
            train_prep = load_data(os.path.join(os.getcwd(), data_folder, dataset_params["filename"] + "_" + str(dataset_params["train_size"]) + "_train_prep_BOZORGI_RealCause" + bias_path))[int_index]
            prep_utils = load_data(os.path.join(os.getcwd(), data_folder, dataset_params["filename"] + "_" + str(dataset_params["train_size"]) + "_prep_utils_BOZORGI_RealCause" + bias_path))[int_index]
            to_generate_prep = load_data(os.path.join(os.getcwd(), data_folder, dataset_params["filename"] + "_" + str(dataset_params["train_size"]) + "_to_generate_prep_BOZORGI_RealCause" + bias_path))[int_index]
            to_generate_prep_val = load_data(os.path.join(os.getcwd(), data_folder, dataset_params["filename"] + "_" + str(dataset_params["train_size"]) + "_to_generate_prep_val_BOZORGI_RealCause" + bias_path))[int_index]
            _, _, _, _, atoms_scaled, bin_width_scaled, stdev_atoms_scaled, _ = prep.scale(data=train_prep, scaler=prep_utils["scaler_dict"], only_atoms=True)
            prep_utils["atoms_scaled"] = atoms_scaled
            prep_utils["bin_width_scaled"] = bin_width_scaled
            prep_utils["stdev_atoms_scaled"] = stdev_atoms_scaled
        elif bozorgi_retaining:
            train_prep, prep_utils, to_generate_prep, to_generate_prep_val = prep.preprocess_bozorgi_retaining(only_preprocess_realcause_data=only_preprocess_realcause_data)
        else:
            train_prep, prep_utils, to_generate_prep, to_generate_prep_val = prep.preprocess(only_preprocess_realcause_data=only_preprocess_realcause_data)
        train_prep_list.append(train_prep)
        prep_utils_list.append(prep_utils)
        to_generate_prep_list.append(to_generate_prep)
        to_generate_prep_val_list.append(to_generate_prep_val)
    if big_data:
        save_data(train_prep_list, os.path.join(os.getcwd(), data_folder, dataset_params["filename"] + "_" + str(dataset_params["train_size"]) + "_train_prep_BOZORGI_RealCause" + bias_path))
        save_data(prep_utils_list, os.path.join(os.getcwd(), data_folder, dataset_params["filename"] + "_" + str(dataset_params["train_size"]) + "_prep_utils_BOZORGI_RealCause" + bias_path))
        if not only_preprocess_realcause_data:
            save_data(to_generate_prep_list, os.path.join(os.getcwd(), data_folder, dataset_params["filename"] + "_" + str(dataset_params["train_size"]) + "_to_generate_prep_BOZORGI_RealCause" + bias_path))
            save_data(to_generate_prep_val_list, os.path.join(os.getcwd(), data_folder, dataset_params["filename"] + "_" + str(dataset_params["train_size"]) + "_to_generate_prep_val_BOZORGI_RealCause" + bias_path))
if big_data:
    train_prep_list = load_data(os.path.join(os.getcwd(), data_folder, dataset_params["filename"] + "_" + str(dataset_params["train_size"]) + "_train_prep_BOZORGI_RealCause" + bias_path))
    prep_utils_list = load_data(os.path.join(os.getcwd(), data_folder, dataset_params["filename"] + "_" + str(dataset_params["train_size"]) + "_prep_utils_BOZORGI_RealCause" + bias_path))
    to_generate_prep_list = load_data(os.path.join(os.getcwd(), data_folder, dataset_params["filename"] + "_" + str(dataset_params["train_size"]) + "_to_generate_prep_BOZORGI_RealCause" + bias_path))
    to_generate_prep_val_list = load_data(os.path.join(os.getcwd(), data_folder, dataset_params["filename"] + "_" + str(dataset_params["train_size"]) + "_to_generate_prep_val_BOZORGI_RealCause" + bias_path))
train_prep_list[0].head(55)
print(train_prep_list[0].head(55))
print("Preprocessing done")
















for iteration in range(num_iterations):
    random_seed = 42 + iteration*5
    if iteration in iterations_to_skip:
        continue
    # Training arguments
    args = {
        "lr": 0.0001,
        "batch_size": 1024,
        "num_epochs": 100000,
        "model_type": "tarnet",
        "n_hidden_layers": 2,
        "dim_h": 25,
        "activation": "ReLU",
        "train_prop": 0.8,
        "val_prop": 0.1,
        "test_prop": 0.1,
        "seed": random_seed,
        "early_stop": True,
        "patience": 15000,
        "ignore_w": False,
        "grad_norm": 1.0,
        "test_size":None,
        "train": True,
        "eval": False,
        "savepath": "C:/Users/u0166838/Music/cache_best_model" + str(iteration) + str(intervention_name) + ".pt",
        "print_every_iters": 10000,
        "eval_every": 2000,
        "plot_every": 50000,
        "p_every": 10000000000
    }

    if not big_data:
        args["num_epochs"] = 10
        args["patience"] = 10
        args["lr"] = 0.001
        args["print_every_iters"] = 10
        args["eval_every"] = 10
        args["plot_every"] = 50
        args["p_every"] = 10



    for int_index, int in enumerate(int_dataset_params):
        y = train_prep_list[int_index]["outcome"]
        t = train_prep_list[int_index]["treatment"]
        # get without outcome and treatment columns
        w = train_prep_list[int_index].drop(columns=["outcome", "treatment"])
        
        y_to_generate = to_generate_prep_list[int_index]["outcome"]
        t_to_generate = to_generate_prep_list[int_index]["treatment"]
        w_to_generate = to_generate_prep_list[int_index].drop(columns=["outcome", "treatment"])

        y_to_generate_val = to_generate_prep_val_list[int_index]["outcome"]
        t_to_generate_val = to_generate_prep_val_list[int_index]["treatment"]
        w_to_generate_val = to_generate_prep_val_list[int_index].drop(columns=["outcome", "treatment"])
        
        print("size of w: ", w.shape)
        print(w.columns)
        prep_utils = prep_utils_list[int_index]
        n_dimensions = 1
        
        distribution = distributions.MixedDistributionAtoms(atoms=prep_utils["atoms_scaled"], dist=distributions.SigmoidFlow(ndim=n_dimensions, base_distribution='normal'), bin_width=prep_utils["bin_width_scaled"], stdev_atoms = prep_utils["stdev_atoms_scaled"], atom_part="none")

        training_params = TrainingParams(
            lr = args["lr"], batch_size=args["batch_size"], num_epochs=args["num_epochs"], eval_every=args["eval_every"], plot_every=args["plot_every"], p_every=args["p_every"], print_every_iters=args["print_every_iters"]
        )
        additional_args = dict()

        if args["model_type"] == 'tarnet':
            Model = TarNet
            mlp_params = MLPParams(
                n_hidden_layers=args["n_hidden_layers"],
                dim_h=args["dim_h"],
                activation=getattr(torch.nn, args["activation"])(),
            )

            network_params = {"mlp_params_w": mlp_params, "mlp_params_t_w": mlp_params, "mlp_params_y0_w": mlp_params, "mlp_params_y1_w": mlp_params}
        else:
            raise Exception(f'model type {args["model_type"]} not implemented')

        if args["n_hidden_layers"] < 0:
            raise Exception(f'`n_hidden_layers` must be nonnegative, got {args["n_hidden_layers"]}')

        print(args["model_type"])
        model = Model(w, t, y,
                    training_params=training_params,
                    network_params=network_params,
                    binary_treatment=True, outcome_distribution=distribution,
                    outcome_min=prep_utils["min_outcome_scaled"],
                    outcome_max=prep_utils["max_outcome_scaled"],
                    train_prop=args["train_prop"],
                    val_prop=args["val_prop"],
                    test_prop=args["test_prop"],
                    seed=args["seed"],
                    early_stop=args["early_stop"],
                    patience=args["patience"],
                    ignore_w=args["ignore_w"],
                    grad_norm=args["grad_norm"],
                    savepath=args["savepath"],
                    prep_utils=prep_utils,
                    test_size=args["test_size"],
                    additional_args=additional_args
                    )
        
        if args["train"] and not already_trained:
            print("Training started")
            model.train()
            print("Training done")
        else:
            model = load_data(results_path + "model_" + str(int_index) + "_" + str(iteration) + "_BOZORGI_RealCause" + bias_path)
            args = load_data(results_path + "args_" + str(int_index) + "_" + str(iteration) + "_BOZORGI_RealCause" + bias_path)
            training_params = load_data(results_path + "training_params_" + str(int_index) + "_" + str(iteration) + "_BOZORGI_RealCause" + bias_path)

        # convert w_to_generate (a dataframe) to numpy
        w_to_generate = w_to_generate.to_numpy()
        w_to_generate_val = w_to_generate_val.to_numpy()
        generated_data = model.sample(w=w_to_generate, ret_counterfactuals=True, seed=random_seed)
        generated_data_val = model.sample(w=w_to_generate_val, ret_counterfactuals=True, seed=random_seed)

        # print for generated data and generated data val the very last 5 rows
        print(generated_data[-1][-1])
        print(generated_data_val[-1][-1])

        save_data(generated_data, results_path + "generated_data_" + str(int_index) + "_" + str(iteration) + "_BOZORGI_RealCause" + bias_path)
        save_data(generated_data_val, results_path + "generated_data_val_" + str(int_index) + "_" + str(iteration) + "_BOZORGI_RealCause" + bias_path)
        save_data(model, results_path + "model_" + str(int_index) + "_" + str(iteration) + "_BOZORGI_RealCause" + bias_path)
        save_data(args, results_path + "args_" + str(int_index) + "_" + str(iteration) + "_BOZORGI_RealCause" + bias_path)
        save_data(training_params, results_path + "training_params_" + str(int_index) + "_" + str(iteration) + "_BOZORGI_RealCause" + bias_path)

        uni_metrics_test = model.get_univariate_quant_metrics(dataset="test", verbose=False, outcome_distribution=distribution, seed=random_seed)
        multi_metrics_test = model.get_multivariate_quant_metrics(dataset="test", verbose=False, include_w=False, n_permutations=1000, seed=random_seed)

        print('\n')
        print("Univariate metrics test: ", uni_metrics_test)
        print("Multivariate metrics test: ", multi_metrics_test)

        save_data(uni_metrics_test, results_path + "uni_metrics_test_" + str(int_index) + "_" + str(iteration) + "_BOZORGI_RealCause" + bias_path)
        save_data(multi_metrics_test, results_path + "multi_metrics_test_" + str(int_index) + "_" + str(iteration) + "_BOZORGI_RealCause" + bias_path)