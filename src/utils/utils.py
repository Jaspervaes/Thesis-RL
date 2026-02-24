import numpy as np
import torch

def make_label(settings):
    label = ""
    for key, value in settings.items():
        label += ", {}:{}".format(key, value)
    return label[2:]


class Incremental():
    '''
    incrementally computes mean and variance
    purpose is to avoid memory problems
    '''
    def __init__(self):
        self.mu = 0
        self.previous_mu = 0
        self.n_var = 0
        self.counter = 0

    def update(self, x):
        self.counter += 1
        self.previous_mu = self.mu
        # self.previous_var = self.var
        self.mu = self.previous_mu + (x - self.previous_mu) / self.counter
        self.n_var = self.n_var + (x - self.previous_mu) * (x - self.mu)

        return self.mu, self.n_var / self.counter


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    """From: https://github.com/Bjarten/early-stopping-pytorch"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model, epoch=None):
        if isinstance(model, list):
            self.is_list = True
        else:
            self.is_list = False

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.is_list:
                self.save_checkpoint(val_loss, model[0], suffix="_T")
                self.save_checkpoint(val_loss, model[1], suffix="_C")
            else:
                self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            if self.is_list:
                self.save_checkpoint(val_loss, model[0], suffix="_T")
                self.save_checkpoint(val_loss, model[1], suffix="_C")
            else:
                self.save_checkpoint(val_loss, model)
            self.counter = 0
        #print("best score: {}, counter: {}".format(self.best_score, self.counter))

    def save_checkpoint(self, val_loss, model, suffix=""):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if suffix == "":
            torch.save(model.state_dict(), self.path, _use_new_zipfile_serialization=False)
        else:
            torch.save(model.state_dict(), self.path+suffix, _use_new_zipfile_serialization=False)
        self.val_loss_min = val_loss

    #def get_best_weights(self, model):
    #    return model.load_state_dict(torch.load(self.path))


from torch.nn.modules.module import _addindent
import torch
import numpy as np
def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr