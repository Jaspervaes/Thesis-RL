import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from copy import deepcopy, copy
import gc
import torch.utils
import torch.utils.data
from src.utils.inference import Forward_pass
from src.utils.utils import EarlyStopping
import torch
from torch import nn
import matplotlib.pyplot as plt
import os
from src.utils.models import LSTM

class CIModel(Forward_pass):
    def __init__(self, model_params, int_dataset_params, full_dataset_params, data_train, data_train_val, prep_utils, iteration):
        super().__init__()
        self.device = model_params["device"]
        self.nr_lstm_layers = model_params["nr_lstm_layers"]
        self.lstm_size = model_params["lstm_size"]
        self.nr_dense_layers = model_params["nr_dense_layers"]
        self.dense_width = model_params["dense_width"]
        self.p = model_params["p"]

        self.full_dataset_params = full_dataset_params
        self.current_int_info = int_dataset_params["intervention_info"]
        self.scaler_dict = prep_utils["scaler_dict_train"]
        self.iteration = iteration

        [_, _, _, _, _, x_case_cols_encoded, x_event_cols_encoded, _] = data_train
        self.input_size_case = len(x_case_cols_encoded)
        self.input_size_event = len(x_event_cols_encoded)
        self.overall_seed = copy(model_params["random_seed"]) + self.iteration*5
        torch.manual_seed(self.overall_seed)
        torch.cuda.manual_seed_all(self.overall_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.overall_seed)
        self.model = LSTM(input_size_case=self.input_size_case, input_size_process=self.input_size_event,
                               nr_outputs=len(self.current_int_info["actions"]), nr_lstm_layers=self.nr_lstm_layers,
                               lstm_size=self.lstm_size, nr_dense_layers=self.nr_dense_layers,
                               dense_width=self.dense_width, p=self.p, treatment_length=self.current_int_info["len"], iteration=self.iteration)
        self.model = self.model.to(self.device)
        if self.current_int_info["name"] == "time_contact_HQ":
            self.optimizer = optim.Adam(self.model.parameters(), amsgrad=True, lr=0.001)
        elif self.current_int_info["name"] == "choose_procedure":
            self.optimizer = optim.Adam(self.model.parameters(), amsgrad=True, lr=0.001)
        elif self.current_int_info["name"] == "set_ir_3_levels":
            self.optimizer = optim.Adam(self.model.parameters(), amsgrad=True, lr=0.001)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), amsgrad=True)

        self.losses = []
        self.val_result = 0
        self.val_results = []
        self.optimization_steps = 0
        self.best_optimization_step = 0
        self.steps_done = 0  # used for exploration
        self.model_type = LSTM

        self.init_data(data_train, data_train_val)
        

    def init_data(self, data_train, data_train_val):
        #Send to GPU later in batches
        [treatment_train, prefix_len_train, X_case_train, X_event_train, Y_train, _, _, case_nr_train] = data_train
        self.data_train = data_train
        self.T_train = treatment_train
        self.X_train_case = X_case_train
        self.X_train_event = X_event_train
        self.Y_train = Y_train
        self.pref_len_train = prefix_len_train
        self.N = len(self.Y_train)
        self.case_nr_train = case_nr_train

        [treatment_train_val, prefix_len_train_val, X_case_train_val, X_event_train_val, Y_train_val, _, _, case_nr_train_val] = data_train_val
        self.data_train_val = data_train_val
        self.T_train_val = treatment_train_val.to(device=self.device)
        self.X_train_val_case = X_case_train_val.to(device=self.device)
        self.X_train_val_event = X_event_train_val.to(device=self.device)
        self.Y_train_val = Y_train_val.to(device=self.device)
        self.pref_len_train_val = prefix_len_train_val.to(device=self.device)
        self.case_nr_train_val = case_nr_train_val.to(device=self.device)

        self.tensor_val = torch.utils.data.TensorDataset(self.X_train_val_case, self.X_train_val_event, self.T_train_val, self.Y_train_val)
        self.val_loader = torch.utils.data.DataLoader(self.tensor_val, batch_size=10000, shuffle=False)

    def start_training(self, training_params, key, iteration=0):
        self.init_train_params(training_params, key, iteration)
        self.train_model()
        self.send_model_to_cpu()

    def init_train_params(self, training_params, key, iteration=0):
        self.filename = training_params["filename"]
        self.trainsize = training_params["train_size"]
        self.val_share = training_params["val_share"]
        self.batch_size = training_params["batch_size"]
        self.calc_val = training_params["calc_val"]
        self.earlystop = training_params["earlystop"]
        self.es_patience = training_params["es_patience"]
        self.es_delta = training_params["es_delta"]
        self.nb_epochs = training_params["nb_epochs"]
        self.aleatoric = training_params["aleatoric"]
        self.verbose = training_params["verbose"]
        self.nr_future_est = training_params["nr_future_est"]
        self.iteration = iteration
        self.key = key

        self.early_stop_path = training_params["early_stop_path"] + f"CI_early_stop_{self.filename}_{self.key}_{iteration}_{self.current_int_info['name']}.pt"


    def train_model(self):
        self.train_losses = []
        self.val_losses = []
        torch.manual_seed(self.overall_seed)
        torch.cuda.manual_seed_all(self.overall_seed)
        np.random.seed(self.overall_seed)

        print("KEY: ", self.key, ", ITERATION: ", self.iteration, ", INTERVENTION: ", self.current_int_info["name"])
        print("     Training: ")

        if self.earlystop:
            self.early_stopping = EarlyStopping(patience=self.es_patience, verbose=False, delta=self.es_delta, path = f"{self.early_stop_path}")

            print("         - len dataset:", len(self.Y_train))

        for epoch in range(self.nb_epochs):
            self.model.train()  # prep model for training
            loop_losses = []
            nr_batches = int(np.ceil(self.N / self.batch_size))
            for self.batch in range(nr_batches):
                self.model, self.optimizer, loop_losses = self.train_batch(self.model, self.optimizer,
                                                                            self.X_train_case, self.X_train_event,
                                                                            self.T_train, self.Y_train,
                                                                            loop_losses)
            # reshuffle data
            shuffle_ind = np.random.permutation(len(self.Y_train))
            self.X_train_case, self.Y_train = self.X_train_case[shuffle_ind], self.Y_train[shuffle_ind]
            self.X_train_event, self.T_train = self.X_train_event[shuffle_ind], self.T_train[shuffle_ind]
            # calculate training losses for epoch
            loop_losses = np.array(loop_losses)
            epoch_loss = np.average(loop_losses[:, 0], weights=loop_losses[:, 1])
            self.train_losses.append((epoch_loss))
            # calculate losses on validations set
            if self.val_share:
                val_loss_sum = 0.0
                for batch in self.val_loader:
                    X_batch_case, X_batch_event, T_batch, Y_batch = batch
                    y_batch_pred = self.forward_pass(self.model, X_batch_case, X_batch_event, T_batch, self.aleatoric, self.device)
                    batch_loss = torch.nn.functional.mse_loss(Y_batch, y_batch_pred)
                    val_loss_sum += batch_loss.item()
                val_loss = val_loss_sum / len(self.val_loader)
                self.val_losses.append(val_loss)

                if self.earlystop:
                    if self.early_stopping.best_score is not None:
                        if val_loss <= -self.early_stopping.best_score:
                            self.best_optimization_step = deepcopy(self.optimization_steps)
                    self.early_stopping(val_loss, self.model, epoch=epoch)
                    if self.early_stopping.early_stop:
                        print("         - Early stopping at epoch {}  ".format(epoch))
                        self.model.load_state_dict(torch.load(f"{self.early_stop_path}"))
                        break
            else:
                val_loss = "-"
            print("\r         - epoch {}/{}, training loss: {}, best val loss: {}, best epoch: {}".format(epoch, self.nb_epochs, epoch_loss, -self.early_stopping.best_score, self.early_stopping.best_epoch), end='\r')


    def train_batch(self, model, optimizer, X_case, X_event, T, Y, loop_losses):
        torch.manual_seed(self.overall_seed)
        torch.cuda.manual_seed_all(self.overall_seed)
        np.random.seed(self.overall_seed)

        # prepare batch
        x_case = X_case[self.batch_size * self.batch: self.batch_size * (self.batch + 1)]  # .cuda()
        x_event = X_event[self.batch_size * self.batch: self.batch_size * (self.batch + 1)]  # .cuda()
        t = T[self.batch_size * self.batch: self.batch_size * (self.batch + 1)]
        y = Y[self.batch_size * self.batch: self.batch_size * (self.batch + 1)]  # .cuda()

        # Send to GPU
        x_case = x_case.to(device=self.device)
        x_event = x_event.to(device=self.device)
        t = t.to(device=self.device)
        y = y.to(device=self.device)
      
        y_pred = model(x_case, x_event, t)
        loss = torch.nn.functional.mse_loss(target=y, input=y_pred[:, 0])
        loop_losses.append([loss.detach().cpu().numpy(), len(y)])
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
        optimizer.step()

        self.optimization_steps += 1

        #Send to CPU
        x_case = x_case.to("cpu")
        x_event = x_event.to("cpu")
        t = t.to("cpu")
        y = y.to("cpu")

        return model, optimizer, loop_losses


    def plot_learning(self, train_losses, val_losses, key, iteration, title_suffix=""):
        plt.figure()
        plt.plot(train_losses, label="training losses")
        plt.plot(val_losses, label="validation losses")
        title = "CI training & validation loss for key " + str(key) + " and iteration " + str(iteration) + " and intervention " + self.current_int_info["name"]
        plt.title("{}".format(title))
        plt.suptitle("Learning curves " + title_suffix)
        plt.legend()


    def send_model_to_cpu(self):
        #Move model to cpu
        self.model = self.model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()