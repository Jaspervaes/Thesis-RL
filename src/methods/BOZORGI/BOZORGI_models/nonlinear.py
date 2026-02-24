import numpy as np
from BOZORGI_models import distributions
from BOZORGI_models.base import BaseGenModel
from BOZORGI_models import preprocess
from BOZORGI_models.preprocess import PlaceHolderTransform
import torch
from torch import nn
from torch.utils import data
from itertools import chain
# from plotting.plotting import fig2img
from tqdm import tqdm
import matplotlib.pyplot as plt
from contextlib import contextmanager


@contextmanager
def eval_ctx(mdl, debug=False, is_train=False):
    for net in mdl.networks: net.eval()
    torch.autograd.set_detect_anomaly(debug)
    with torch.set_grad_enabled(mode=is_train):
        yield
    torch.autograd.set_detect_anomaly(False)
    for net in mdl.networks: net.train()


class MLPParams:
    def __init__(self, n_hidden_layers=1, dim_h=64, activation=nn.ReLU()):
        self.n_hidden_layers = n_hidden_layers
        self.dim_h = dim_h
        self.activation = activation


_DEFAULT_MLP = dict(mlp_params_t_w=MLPParams(), mlp_params_y_tw=MLPParams())


class TrainingParams:
    def __init__(self, batch_size=32, lr=0.001, num_epochs=100, verbose=True, print_every_iters=100,
                 eval_every=100, plot_every=100, p_every=100,
                 optim=torch.optim.Adam, **optim_args):
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.print_every_iters = print_every_iters
        self.optim = optim
        self.eval_every = eval_every
        self.plot_every = plot_every
        self.p_every = p_every
        self.optim_args = optim_args


class CausalDataset(data.Dataset):
    def __init__(self, w, t, y, wtype='float32', ttype='float32', ytype='float32',
                 w_transform: preprocess.Preprocess = preprocess.PlaceHolderTransform(),
                 t_transform: preprocess.Preprocess = preprocess.PlaceHolderTransform(),
                 y_transform: preprocess.Preprocess = preprocess.PlaceHolderTransform()):
                 
        self.w = w.astype(wtype)
        self.t = t.astype(ttype)
        self.y = y.astype(ytype)
        # todo: no need anymore, remove?
        self.w_transform = w_transform
        self.t_transform = t_transform
        self.y_transform = y_transform

    def __len__(self):
        return self.w.shape[0]

    def __getitem__(self, index):
        return (
            self.w_transform.transform(self.w[index]),
            self.t_transform.transform(self.t[index]),
            self.y_transform.transform(self.y[index]),
        )


# TODO: for more complex w, we might need to share parameters (dependent on the problem)
class MLP(BaseGenModel):
    def __init__(self, w, t, y, seed=1,
                 network_params=None,
                 training_params=TrainingParams(),
                 binary_treatment=False,
                 outcome_distribution: distributions.BaseDistribution = distributions.FactorialGaussian(),
                 outcome_min=None,
                 outcome_max=None,
                 train_prop=1,
                 val_prop=0,
                 test_prop=0,
                 shuffle=True,
                 early_stop=True,
                 patience=None,
                 ignore_w=False,
                 grad_norm=float('inf'),
                 prep_utils=PlaceHolderTransform,
                 w_transform=PlaceHolderTransform,
                 t_transform=PlaceHolderTransform,
                 y_transform=PlaceHolderTransform,
                 savepath='.cache_best_model.pt',
                 test_size=None,
                 additional_args=dict()):
        super(MLP, self).__init__(*self._matricize((w, t, y)), seed=seed,
                                  train_prop=train_prop, val_prop=val_prop,
                                  test_prop=test_prop, shuffle=shuffle,
                                  w_transform=w_transform,
                                  t_transform=t_transform,
                                  y_transform=y_transform,
                                  test_size=test_size)

        self.binary_treatment = binary_treatment
        if binary_treatment:  # todo: input?
            self.treatment_distribution = distributions.Bernoulli()
        else:
            self.treatment_distribution = distributions.FactorialGaussian()
        self.outcome_distribution = outcome_distribution
        self.outcome_min = outcome_min
        self.outcome_max = outcome_max
        self.early_stop = early_stop
        self.patience = patience
        self.ignore_w = ignore_w
        self.grad_norm = grad_norm
        self.savepath = savepath
        self.additional_args = additional_args

        self.dim_w = self.w_transformed.shape[1]
        self.dim_t = self.t_transformed.shape[1]
        self.dim_y = self.y_transformed.shape[1]

        if network_params is None:
            network_params = _DEFAULT_MLP
        self.network_params = network_params
        self.build_networks()

        self.training_params = training_params
        self.optim = training_params.optim(
            chain(*[net.parameters() for net in self.networks]),
            training_params.lr,
            **training_params.optim_args
        )

        self.data_loader = data.DataLoader(
            CausalDataset(self.w_transformed, self.t_transformed, self.y_transformed),
            batch_size=training_params.batch_size,
            shuffle=True,
        )

        if len(self.val_idxs) > 0:
            self.data_loader_val = data.DataLoader(
                CausalDataset(
                    self.w_val_transformed,
                    self.t_val_transformed,
                    self.y_val_transformed,
                ),
                batch_size=training_params.batch_size,
                shuffle=True,
            )

        self.best_val_loss = float("inf")

    def _matricize(self, data):
        return [np.reshape(d, [d.shape[0], -1]) for d in data]

    def _build_mlp(self, dim_x, dim_y, MLP_params=MLPParams(), output_multiplier=2):
        dim_h = MLP_params.dim_h
        hidden_layers = [nn.Linear(dim_x, dim_h), MLP_params.activation]
        for _ in range(MLP_params.n_hidden_layers - 1):
            hidden_layers += [nn.Linear(dim_h, dim_h), MLP_params.activation]
        hidden_layers += [nn.Linear(dim_h, dim_y * output_multiplier)]
        return nn.Sequential(*hidden_layers)

    def build_networks(self):
        self.MLP_params_t_w = self.network_params['mlp_params_t_w']
        self.MLP_params_y_tw = self.network_params['mlp_params_y_tw']
        output_multiplier_t = 1 if self.binary_treatment else 2
        self.mlp_t_w = self._build_mlp(self.dim_w, self.dim_t, self.MLP_params_t_w, output_multiplier_t)
        self.mlp_y_tw = self._build_mlp(self.dim_w + self.dim_t, self.dim_y, self.MLP_params_y_tw,
                                        self.outcome_distribution.num_params)
        self.networks = [self.mlp_t_w, self.mlp_y_tw]

    def _get_loss(self, w, t, y):
        t_ = self.mlp_t_w(w)
        if self.ignore_w:
            w = torch.zeros_like(w)
        y_ = self.mlp_y_tw(torch.cat([w, t], dim=1))
        # check whether y_ contains nan values
        if torch.isnan(y_).any():
            print('y_ contains nan values')
        print('y_:', y_)
        loss_t = self.treatment_distribution.loss(t, t_)
        loss_y = self.outcome_distribution.loss(y, y_)
        loss = loss_t + loss_y
        return loss, loss_t, loss_y

    def train(self, early_stop=None, print_=lambda s, print_: print(s), comet_exp=None):
        self.losses = []
        self.val_losses = []
        self.t_losses = []
        self.y_losses = []
        loss_val = float("inf")
        if early_stop is None:
            early_stop = self.early_stop

        c = 0
        self.best_val_loss = float("inf")
        self.best_val_idx = 0
        for _ in tqdm(range(self.training_params.num_epochs)):
            for w, t, y in self.data_loader:
                self.optim.zero_grad()
                loss, loss_t, loss_y = self._get_loss(w, t, y)
                self.losses.append(loss.item())
                self.t_losses.append(loss_t.item())
                self.y_losses.append(loss_y.item())
                self.val_losses.append(loss_val)
                # TODO: learning rate can be separately adjusted by weighting the losses here
                loss.backward()
                torch.nn.utils.clip_grad_norm_(chain(*[net.parameters() for net in self.networks]), self.grad_norm)
                self.optim.step()

                c += 1
                if self.training_params.verbose and c % self.training_params.print_every_iters == 0:
                    print("\n")
                    print("Iteration :", c)
                    print('    Training loss:', loss.item())

                    if comet_exp is not None:
                        comet_exp.log_metric("loss_t", loss_t.item())
                        comet_exp.log_metric("loss_y", loss_y.item())

                if c % self.training_params.eval_every == 0 and len(self.val_idxs) > 0:
                    with eval_ctx(self):
                        loss_val = self.evaluate(self.data_loader_val, only_y_loss=True).item()
                    if comet_exp is not None:
                        comet_exp.log_metric('loss_val', loss_val)
                    print("    Val loss:", loss_val)
                    if loss_val < self.best_val_loss:
                        self.best_val_loss = loss_val
                        self.best_val_idx = c
                        print("    saving best-val-loss model")
                        torch.save([net.state_dict() for net in self.networks], self.savepath)

                if c % self.training_params.plot_every == 0:
                    # use matplotlib
                    plt.figure()
                    plt.plot(self.losses, label='loss')
                    plt.plot(self.t_losses, label='t_loss')
                    plt.plot(self.y_losses, label='y_loss')
                    plt.plot(self.val_losses, label='val_loss')
                    plt.legend()
                    plt.ioff()
                    plt.close()


                    with eval_ctx(self):
                        plots = self.plot_ty_dists(verbose=False, dataset="train")
                        plots_val = self.plot_ty_dists(verbose=False, dataset="val")
                        plots_test = self.plot_ty_dists(verbose=False, dataset="test")
                        
                if c % self.training_params.p_every == 0:
                    with eval_ctx(self):
                        uni_metrics_test = self.get_univariate_quant_metrics(dataset="test", verbose=False, outcome_distribution=self.outcome_distribution)
                        multi_variate_metrics_test = self.get_multivariate_quant_metrics(include_w=False, n_permutations=50, verbose=False, dataset="test", calculate_wssd=False)
                        print("    Test: ", uni_metrics_test)
                        print("    Test Multi: ", multi_variate_metrics_test)

                
            if early_stop and self.patience is not None and c - self.best_val_idx > self.patience:
                print('early stopping criterion reached. Ending experiment.')
                plt.figure()
                plt.plot(self.losses, label='loss')
                plt.plot(self.val_losses, label='val_loss')
                plt.legend()
                plt.show()
                break

        if early_stop and len(self.val_idxs) > 0:
            print("loading best-val-loss model (early stopping checkpoint)")
            for net, params in zip(self.networks, torch.load(self.savepath)):
                net.load_state_dict(params)

    def evaluate(self, data_loader, only_y_loss=False):
        loss = 0
        n = 0
        for w, t, y in data_loader:
            if only_y_loss:
                loss += self._get_loss(w, t, y)[2] * w.size(0)
            else:
                loss += self._get_loss(w, t, y)[0] * w.size(0)
            n += w.size(0)
        return loss / n

    def _sample_t(self, w=None, overlap=1):
        t_ = self.mlp_t_w(torch.from_numpy(w).float())
        return self.treatment_distribution.sample(t_, overlap=overlap)

    def _sample_y(self, t, w=None, ret_counterfactuals=False):
        if self.ignore_w:
            w = np.zeros_like(w)
        wt = np.concatenate([w, t], 1)
        if ret_counterfactuals:
            y0_, y1_ = self.mlp_y_tw(torch.from_numpy(wt).float(), ret_counterfactuals=True)
            y0_samples = self.outcome_distribution.sample(y0_)
            y1_samples = self.outcome_distribution.sample(y1_)
            if self.outcome_min is not None or self.outcome_max is not None:
                y0_samples = np.clip(y0_samples, self.outcome_min, self.outcome_max)
                y1_samples = np.clip(y1_samples, self.outcome_min, self.outcome_max)
            return y0_samples, y1_samples
        else:
            y_ = self.mlp_y_tw(torch.from_numpy(wt).float(), ret_counterfactuals=False)
            y_samples = self.outcome_distribution.sample(y_)
            if self.outcome_min is not None or self.outcome_max is not None:
                y_samples = np.clip(y_samples, self.outcome_min, self.outcome_max)
            return y_samples

    def mean_y(self, t, w):
        if self.ignore_w:
            w = np.zeros_like(w)
        wt = np.concatenate([w, t], 1)
        return self.outcome_distribution.mean(self.mlp_y_tw(torch.from_numpy(wt).float()))