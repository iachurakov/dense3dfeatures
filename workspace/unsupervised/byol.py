import torch
import copy
import torch.nn.functional as F
import numpy as np
from ..utils.losses import byol_loss


class MLP(torch.nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

    
def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

        
def update_ema(target_model, online_model, decay):
    with torch.no_grad():
        for tp, op in zip(target_model.parameters(), online_model.parameters()):
            new_params = decay * tp + (1 - decay) * op
            tp.copy_(new_params)
        

def byol_loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class BYOL(torch.nn.Module):
    def __init__(self,
                 base_net: torch.nn.Module,
                 net_output_dim: int,
                 hidden_size: int,
                 projection_size: int,
                 n_iters: int,
                 moving_average_decay: float = 0.996,
                 target_net: torch.nn.Module = None) -> None:
        """
        :param base_net: base model for embeddings extraction
        :param net_output_dim: dim of base model outputs
        :param hidden_size: size of hidden layer in projector and predictor
        :param projection_size: size of projection (embedding)
        :param n_iters: the number of iterations in training procedure, need for tau scheduling
        :param moving_average_decay: param for exponential moving average
        :param target_net: should have the same init weights as base_net
        workaround for models which are not deepcopyable
        """
        super().__init__()
        self.moving_average_decay = moving_average_decay
        self.n_iters = n_iters
        self.online_net = base_net
        self.online_projector = MLP(net_output_dim, projection_size, hidden_size)
        self.online_predictor = MLP(projection_size, projection_size, hidden_size)
        
        if target_net is None:
            self.target_net = copy.deepcopy(self.online_net)
        else:
            self.target_net = target_net

        self.target_projector = copy.deepcopy(self.online_projector)
    
        set_requires_grad(self.target_net, False)
        set_requires_grad(self.target_projector, False)
        
    def update_target(self, iter):
        """
        :param iter: the number of current iteration
        """
        tau = 1 - (1 - self.moving_average_decay) * (np.cos(np.pi * iter / self.n_iters) + 1) / 2 if iter > 0 else self.moving_average_decay
        update_ema(self.target_net, self.online_net, tau)
        update_ema(self.target_projector, self.online_projector, tau)
        
    def forward(self, x1, x2):
        """
        :param x1: first augmented view of batch
        :param x2: second augmented view of batch
        :return: mean byol loss
        """
        target_embeddings_one = self.target_projector(self.target_net(x1))
        target_embeddings_two = self.target_projector(self.target_net(x2))
        online_predictions_one = self.online_predictor(self.online_projector(self.online_net(x1)))
        online_predictions_two = self.online_predictor(self.online_projector(self.online_net(x2)))

        return target_embeddings_one, target_embeddings_two, online_predictions_one, online_predictions_two
        
        loss_one = byol_loss_fn(target_embeddings_one, online_predictions_two)
        loss_two = byol_loss_fn(target_embeddings_two, online_predictions_one)
        loss = loss_one + loss_two
        
        return loss.mean()
    
    def get_embeddings(self, x):
        self.eval()
        with torch.no_grad():
            return self.online_projector(self.online_net(x))
