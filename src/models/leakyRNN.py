from collections.abc import Callable
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Module, init
from collections import OrderedDict
from src.utils import reparameterize as rp
from src.extratyping import *


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


class TransitionNetLRNNPB(Module):
    def __init__(self, action_dim: int, state_dim: int, hidden_dim: int, num_layers: int = 1,
                 bias: bool = True, act_func: Callable = F.leaky_relu, repeat_input: int = 1, out_style: str = 'mean', device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransitionNetLRNNPB, self).__init__()

        layers = OrderedDict()
        layers['lrnn1'] = LRNN(state_dim + action_dim, hidden_dim, act_func, bias=bias, **factory_kwargs)
        for i in range(num_layers - 1):
            layers[f'lrnn{2+i}'] = LRNN(hidden_dim, hidden_dim, act_func, bias=bias, **factory_kwargs)
        layers['mu'] = nn.Linear(hidden_dim, state_dim, bias, **factory_kwargs)
        layers['logvar'] = nn.Linear(hidden_dim, state_dim, bias, **factory_kwargs)

        self.basis = nn.ModuleDict(layers)

        self.num_layers = num_layers
        self.act_func = act_func
        self.bias = bias
        init.zeros_(self.basis.fc_mu.bias)
        init.ones_(self.basis.fc_var.bias)

        self.state_initialized = False
        assert repeat_input >= 1
        self.repeat_input = repeat_input
        assert out_style.lower() in ['mean', 'last']
        self.out_style = out_style.lower()

    def reset_state(self) -> None:
        self.state_initialized = False

    def init_state(self, batch_size: int = 1) -> None:
        for name, layer in self.basis.items():
            layer.init_state(batch_size)
        self.state_initialized = True

    def step(self, state: Tensor, action: Tensor) -> Union[Tensor, Tensor]:
        x = torch.cat((state, action), -1)
        for name, layer in self.basis.items():
            if 'lrnn' in name.lower():
                x = layer(x)
        return self.basis.mu(x), self.basis.logvar(x)

    def forward(self, state: Tensor, action: Tensor) -> Union[Tensor, Tensor]:

        if len(state.shape) == 2:
            state.unsqueeze_(0)

        if len(action.shape) == 2:
            action.unsqueeze_(0)

        T = state.shape[0]
        N = state.shape[1]
        D = state.shape[2]

        if not self.state_initialized:
            self.init_state(N)

        mu_outs = torch.empty((T, N, D), device=next(self.basis.mu.parameters()).device)
        logvar_outs = torch.empty((T, N, D), device=next(self.basis.logvar.parameters()).device)
        for t in range(T):
            mus, logvars = [], []
            for i in range(self.repeat_input):
                mu, logvar = self.step(state[t], action[t])
                mus.append(mu)
                logvars.append(logvar)
            if self.out_style == 'last':
                mu_outs[t] = mu
                logvar_outs[t] = logvar
            elif self.out_style == 'mean':
                mus = torch.stack(mus)
                logvars = torch.stack(logvars)
                mu_outs[t] = mus.mean(dim=0)
                logvar_outs[t] = logvars.mean(dim=0)

        return mu_outs, logvar_outs

    def predict(self, state: Tensor, action: Tensor, deterministic: bool = False):

        mu, logvar = self(state, action)

        if deterministic:
            return mu
        else:
            return rp(mu, logvar)
