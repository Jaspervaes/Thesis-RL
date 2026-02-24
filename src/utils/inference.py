import torch
import numpy as np

class Forward_pass:
    def forward_pass(self, model, X_case, X_process, T, aleatoric=True, device="cuda"):
        # if any of the data is not on the device, move it there
        if X_case.device != device:
            X_case = X_case.to(device=device)
        if X_process.device != device:
            X_process = X_process.to(device=device)
        if T.device != device:
            T = T.to(device=device)
        # check if model is on device
        model_device = next(model.parameters()).device
        if model_device != device:
            model = model.to(device=device)
        model.eval()
        with torch.no_grad():
            # y_pred, logvar = model(X_case, X_process, T)
            y_pred = model(X_case, X_process, T)
            # print(y_pred.shape)
            # print(y_pred)
            y_pred = y_pred[:, 0]
            # logvar = logvar[:, 0]
        # if aleatoric:
        #     al_unc = torch.exp(logvar)
        # else:
            # al_unc = torch.zeros(y_pred.shape).to(device=device)
        al_unc = torch.zeros(y_pred.shape).to(device=device)
        return y_pred
        # return y_pred, logvar, al_unc


class Predict(Forward_pass):
    def predict(self):
        T_zeros = torch.zeros_like(self.T_test)
        T_ones = torch.zeros_like(self.T_test)
        T_ones[:, -1] = 1
        self.Y_pred_C, logvar_C, al_unc_C = self.forward_pass(self.model, self.X_test_case, self.X_test_process, T_zeros)
        self.Y_pred_T, logvar_T, al_unc_T = self.forward_pass(self.model, self.X_test_case, self.X_test_process, T_ones) #self.T_test)
        if self.scale_y:
            self.Y_pred_C = torch.tensor(self.DPP.scaler_y.inverse_transform(self.Y_pred_C.detach().cpu())).to(self.device)
            self.Y_pred_T = torch.tensor(self.DPP.scaler_y.inverse_transform(self.Y_pred_T.detach().cpu())).to(self.device)
        # ignore not-measured uncertainties
        if self.aleatoric:
            if self.scale_y:
                self.al_unc_C = torch.square(torch.tensor(self.DPP.scaler_y.inverse_transform(torch.sqrt(al_unc_C).detach().cpu())).to(self.device))
                self.al_unc_T = torch.square(
                    torch.tensor(self.DPP.scaler_y.inverse_transform(torch.sqrt(al_unc_T).detach().cpu())).to(
                        self.device))
        else:
            al_unc_T, al_unc_C = torch.zeros(al_unc_T.shape).to(device=self.device), torch.zeros(al_unc_C.shape).to(device=self.device)
        # compute ITE
        self.Y_pred_T = self.Y_pred_T.detach()
        self.Y_pred_C = self.Y_pred_C.detach()
        self.ITE_pred = (self.Y_pred_T - self.Y_pred_C)
        # compute MSE
        loss = torch.mean((self.ITE_test - self.ITE_pred)**2)
        if self.verbose:
            print("MSE on test set: {}".format(loss))
        # compute uncertainty
        al_unc = .5 * (al_unc_T + al_unc_C)
        self.unc_ITE = al_unc.detach()
        self.unc_T = al_unc_T.detach()
        self.unc_C = al_unc_C.detach()


class Predict_Neighbors(Forward_pass):

    def predict_future(self):
        self.predict_training_data()
        self.find_neighbors()
        self.ITE_pred_fut = {} # collects future forecasts
        self.ITE_unc_fut = {} # collects future uncertainties
        self.predict_neighbors()


    def predict_training_data(self):
        T_zeros = torch.zeros_like(self.T_train)
        T_ones = torch.zeros_like(self.T_train)
        T_ones[:, -1] = 1
        Y_pred_C, _, logvar_C, ep_unc_C, al_unc_C = self.forward_pass(self.model, self.X_train_case,
                                                                      self.X_train_process, T_zeros)
        Y_pred_T, _, logvar_T, ep_unc_T, al_unc_T = self.forward_pass(self.model, self.X_train_case,
                                                                      self.X_train_process, T_ones)
        # ignore not-measured uncertainties
        if not self.bayesian:
            ep_unc_T, ep_unc_C = torch.zeros(ep_unc_T.shape).to(device=self.device), torch.zeros(
                ep_unc_C.shape).to(device=self.device)
        if not self.aleatoric:
            al_unc_T, al_unc_C = torch.zeros(al_unc_T.shape).to(device=self.device), torch.zeros(
                al_unc_C.shape).to(device=self.device)
        # compute ITE
        self.ITE_train_pred = (Y_pred_T - Y_pred_C).detach()
        # compute uncertainty
        al_unc = .5 * (al_unc_T + al_unc_C)
        ep_unc = .5 * (ep_unc_T + ep_unc_C)
        self.ITE_train_unc = (al_unc + ep_unc).detach()
        pass


    def find_neighbors(self):
        self.neighbors = {}
        max_pref_len = int(torch.max(self.test_pref_len).cpu().numpy())
        for sample_nr in range(len(self.T_test)):
            sample_neighbors = {}
            # concatenate case and process features
            sample = torch.concat((torch.unsqueeze(self.X_test_case[sample_nr], dim=1).repeat(1,self.seq_len), self.X_test_process[sample_nr]), dim=0)
            dataset = torch.concat((torch.unsqueeze(self.X_train_case, dim=2).repeat(1, 1, self.seq_len), self.X_train_process), dim=1)
            sample_pref_len = int(self.test_pref_len[sample_nr].cpu().numpy())
            for pref_len in range(sample_pref_len, max_pref_len):
                #pref_len_idx = self.train_pref_len == pref_len
                pref_len_idx = np.argwhere(self.train_pref_len.cpu() == pref_len)[0, :].numpy()
                # compute distances
                eucl_dist = torch.sum(torch.sum((dataset[pref_len_idx] - sample) ** 2, dim=1), dim=1)
                # get nearest
                _, idx = torch.topk(eucl_dist, min(self.nr_future_est, len(eucl_dist)))
                sample_neighbors[pref_len] = pref_len_idx[idx.cpu().numpy()]
            self.neighbors[sample_nr] = sample_neighbors


    def predict_neighbors(self):
        for sample_nr, sample_neigbors in self.neighbors.items():
            ITE_pred_fut, ITE_unc_fut = {}, {}
            for pref_len, pref_len_neighbors in sample_neigbors.items():
                ITE_pred_fut[pref_len] = torch.mean(self.ITE_train_pred[pref_len_neighbors]).cpu().numpy() / self.nr_iters
                ITE_unc_fut[pref_len] = torch.mean(self.ITE_train_unc[pref_len_neighbors]).cpu().numpy() / self.nr_iters
            self.ITE_pred_fut[sample_nr] = ITE_pred_fut
            self.ITE_unc_fut[sample_nr] = ITE_unc_fut

