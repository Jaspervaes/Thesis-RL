import torch

def mse_loss(y, y_pred, logvar=None, aleatoric=False, device='cpu'):
    if aleatoric:
        precision = torch.exp(-logvar)
        squared_error = precision * (y - y_pred) ** 2 + logvar
    else:
        squared_error = (y - y_pred) ** 2
    return torch.mean(squared_error, 0)

