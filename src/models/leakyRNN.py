from collections.abc import Callable
import torch
from torch import nn, Tensor
from torch.nn import Module

class Offset(Module):
    """learnable initial state of any RNN layer"""
    def __init__(self, n_params) -> None:
        super(Offset, self).__init__()

        # TODO: alternative initializations? -> later
        X = torch.empty((1, n_params))
        nn.init.kaiming_normal_(X)
        self.X = nn.Parameter(X)

    def forward(self, z) -> Tensor:
        return torch.broadcast_to(self.X, z.shape)


class LRNN(Module):
    """leaky RNN layer with trainable offset and time constant"""
    def __init__(self, n_inputs: int, n_hidden: int, act_func: Callable, bias: bool = True, alpha: float = 0.0,
                 individual: bool = True, trainable_alpha: bool = True) -> None:
        super(LRNN, self).__init__()

        self.n_hidden = n_hidden

        self.U = nn.Linear(n_inputs, n_hidden, bias)  # input weights
        self.W = nn.Linear(n_hidden, n_hidden, False)  # recurrent weights

        self.Oh = Offset(n_hidden)  # h at t0

        self.init_alpha(alpha, individual, trainable_alpha)

        self.act_func = act_func

    def get_alpha(self) -> Tensor:
        """returns the value of the effective alpha"""
        # using sigmoid to reliably maintain alpha in [0, 1]
        return torch.sigmoid(self.alpha_param)

    def init_alpha(self, alpha: float = 1., individual: bool = True, trainable: bool = True) -> None:
        """the effective alpha is between 0 and 1, so a sigmoid is applied to an unbounded parameter to obtain this"""
        assert 0. <= alpha <= 1.
        alpha_param_value = torch.log(torch.tensor([alpha / (1. - min([alpha, 1. - 1e-6]))]))  # inverse sigmoid
        self.alpha_param = nn.Parameter(torch.ones(self.n_hidden if individual else 1) * alpha_param_value,
                                        requires_grad=trainable)

    def forward(self, x: Tensor, h: Tensor = None) -> (Tensor, Tensor):

        out = []  # output is a sequence

        # x should be organized as (time steps, batch examples, data)
        for x_ in x:

            z = self.U(x_)  # apply input weights

            if h is None:
                h = self.Oh(z)  # if necessary, init hidden state

            h = self.get_alpha() * h + z + self.W(self.act_func(h))  # update hidden state

            out.append(self.act_func(h))  # compute output

        return torch.stack(out), h