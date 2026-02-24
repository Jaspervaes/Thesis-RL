from BOZORGI_models.nonlinear import MLP, MLPParams, TrainingParams
from BOZORGI_models import preprocess
from BOZORGI_models import distributions
import torch
import numpy as np


_DEFAULT_TARNET = dict(
    mlp_params_w=MLPParams(),
    mlp_params_t_w=MLPParams(),
    mlp_params_y0_w=MLPParams(),
    mlp_params_y1_w=MLPParams(),
)


class TarNet(MLP):
    # noinspection PyAttributeOutsideInit
    def build_networks(self):
        self.MLP_params_w = self.network_params['mlp_params_w']
        self.MLP_params_t_w = self.network_params['mlp_params_t_w']
        self.MLP_params_y0_w = self.network_params['mlp_params_y0_w']
        self.MLP_params_y1_w = self.network_params['mlp_params_y1_w']

        output_multiplier_t = 1 if self.binary_treatment else 2
        self._mlp_w = self._build_mlp(self.dim_w, self.MLP_params_w.dim_h, self.MLP_params_w, 1)
        self._mlp_t_w = self._build_mlp(self.MLP_params_w.dim_h, self.dim_t, self.MLP_params_t_w, output_multiplier_t)
        self._mlp_y0_w = self._build_mlp(self.MLP_params_w.dim_h, self.dim_y, self.MLP_params_y0_w,
                                         self.outcome_distribution.num_params)
        self._mlp_y1_w = self._build_mlp(self.MLP_params_w.dim_h, self.dim_y, self.MLP_params_y1_w,
                                         self.outcome_distribution.num_params)
        self.networks = [self._mlp_w, self._mlp_t_w, self._mlp_y0_w, self._mlp_y1_w]

    def mlp_w(self, w):
        return self.MLP_params_w.activation(self._mlp_w(w))

    def mlp_t_w(self, w):
        return self._mlp_t_w(self.mlp_w(w))

    def mlp_y_tw(self, wt, ret_counterfactuals=False):
        """
        :param wt: concatenation of w and t
        :return: parameter of the conditional distribution p(y|t,w)
        """
        w, t = wt[:, :-1], wt[:, -1:]
        w = self.mlp_w(w)
        y0 = self._mlp_y0_w(w)
        y1 = self._mlp_y1_w(w)
        if ret_counterfactuals:
            return y0, y1
        else:
            return y0 * (1 - t) + y1 * t


    def _get_loss(self, w, t, y):
        # compute w_ only once
        w_ = self.mlp_w(w)
        t_ = self._mlp_t_w(w_)
        if self.ignore_w:
            w_ = torch.zeros_like(w_)

        y0 = self._mlp_y0_w(w_)
        y1 = self._mlp_y1_w(w_)
        if torch.isnan(y0).any():
            print("y0 has nan values")
        if torch.isnan(y1).any():
            print("y1 has nan values")
        y_ = y0 * (1 - t) + y1 * t
        # check if y_ has nan values
        if torch.isnan(y_).any():
            print("y_ has nan values")
            print(y_, "y_")
        loss_t = self.treatment_distribution.loss(t, t_)
        loss_y = self.outcome_distribution.loss(y, y_)
        loss = loss_t + loss_y
        return loss, loss_t, loss_y