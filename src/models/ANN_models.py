import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter, init
from src.models.leakyRNN import LRNN
from src.extratyping import *
from src.utils import reparameterize as rp
from collections import OrderedDict

class StatefulModel(Module):
    """Base class for models with a hidden stat """

    def __init__(self) -> None:
        super().__init__()
        self.h = None

    def reset_state(self) -> None:
        self.h = None

    def detach_state(self) -> None:
        self.h.detach_()

    def get_state(self) -> Tensor:
        return self.h

    def update_state(self, h: Tensor) -> None:
        self.h = h

    def count_parameters(self):
        # TODO: check if this counts MaskedTensor parameters correctly
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AdaptiveModel(StatefulModel):
    def __init__(self):
        super().__init__()

    def reset_adaptation_weights(self) -> None:

        for k, l in self.adaptive_layers.items():
            l.weight.data.fill_(0.)
            if l.bias is not None:
                l.bias.data.fill_(0.)

    def get_adaptation_weights(self) -> dict:

        return self.adaptive_layers.state_dict()


class PolicyNetPBAdaptive(AdaptiveModel):
    """probabilistic MLP policy network"""

    def __init__(self, action_dim: int, state_dim: int, target_dim: int, hidden_dim: int, num_rec_layers: int = 0,
                 num_ff_layers: int = 1, bias: bool = True, act_fn: Callable = F.leaky_relu, **kwargs) -> None:
        super().__init__()

        # gather layer parameters
        in_dim = state_dim + target_dim

        # make layers
        layers = OrderedDict()
        if num_rec_layers:
            layers['gru'] = nn.GRU(in_dim, hidden_dim, num_rec_layers, bias)
        for i in range(num_ff_layers):
            dim = in_dim if (i == 0 and not num_rec_layers) else hidden_dim
            layers[f'fc_ff{i + 1}'] = nn.Linear(dim, hidden_dim, bias)
        layers['fc_mu'] = nn.Linear(hidden_dim, action_dim, bias)
        layers['fc_var'] = nn.Linear(hidden_dim, action_dim, bias)
        self.basis = nn.ModuleDict(layers)

        self.adaptive_layers = nn.ModuleDict({
            'fc_mu_adapt': nn.Linear(hidden_dim, action_dim, bias),
            'fc_var_adapt': nn.Linear(hidden_dim, action_dim, bias)
        })
        self.bias = bias
        self.act_fn = act_fn
        self.reset_adaptation_weights()
        init.zeros_(self.basis.fc_mu.bias)
        init.zeros_(self.basis.fc_var.bias)

    def forward(self, state: Tensor, target: Tensor) -> (Tensor, Tensor):

        if len(state.shape) == 3:
            state.squeeze_(0)

        if len(target.shape) == 3:
            target.squeeze_(0)

        x = torch.cat((state, target), -1)
        for name, layer in self.basis.items():
            if 'gru' in name.lower():
                x, self.h = layer(x)
                x = self.act_fn(x)
            elif 'fc_ff' in name.lower():
                x = self.act_fn(layer(x))

        mu = self.basis['fc_mu'](x)  # + self.adaptive_layers['fc_mu_adapt'](x)
        logvar = self.basis['fc_var'](x)  # + self.adaptive_layers['fc_var_adapt'](x)

        return torch.tanh(mu), logvar

    def predict(self, state: Tensor, target: Tensor, deterministic: bool = False, record: bool = False) -> Tensor:

        mu, logvar = self(state, target)

        if deterministic:
            return mu
        else:
            return rp(mu, logvar)


class TransitionNetPBAdaptive(AdaptiveModel):
    """probabilistic RNN transition network"""

    def __init__(self, action_dim: int, state_dim: int, hidden_dim: int, num_rec_layers: int = 1, num_ff_layers: int = 1,
                 bias: bool = True, act_fn: Callable = F.leaky_relu, **kwargs) -> None:
        super().__init__()

        # gather layer parameters
        in_dim = state_dim + action_dim

        # make layers
        layers = OrderedDict()
        if num_rec_layers:
            layers['gru'] = nn.GRU(in_dim, hidden_dim, num_rec_layers, bias)
        for i in range(num_ff_layers):
            dim = in_dim if (i == 0 and not num_rec_layers) else hidden_dim
            layers[f'fc_ff{i + 1}'] = nn.Linear(dim, hidden_dim, bias)
        layers['fc_mu'] = nn.Linear(hidden_dim, state_dim, bias)
        layers['fc_var'] = nn.Linear(hidden_dim, state_dim, bias)
        self.basis = nn.ModuleDict(layers)

        self.adaptive_layers = nn.ModuleDict({
            'fc_mu_adapt': nn.Linear(hidden_dim, state_dim, bias),
            'fc_var_adapt': nn.Linear(hidden_dim, state_dim, bias)
        })
        self.act_fn = act_fn
        self.bias = bias
        self.reset_adaptation_weights()
        init.zeros_(self.basis.fc_mu.bias)
        init.zeros_(self.basis.fc_var.bias)

    def forward(self, state: Tensor, action: Tensor) -> Union[Tensor, Tensor]:

        if len(state.shape) == 2:
            state.unsqueeze_(0)

        if len(action.shape) == 2:
            action.unsqueeze_(0)

        x = torch.cat((state, action), -1)
        for name, layer in self.basis.items():
            if 'gru' in name.lower():
                x, self.h = layer(x)
                x = self.act_fn(x)
            elif 'fc_ff' in name.lower():
                x = self.act_fn(layer(x))

        mu = self.basis['fc_mu'](x)  # + self.adaptive_layers['fc_mu_adapt'](x)
        logvar = self.basis['fc_var'](x)  # + self.adaptive_layers['fc_var_adapt'](x)

        return mu, logvar

    def predict(self, state: Tensor, action: Tensor, deterministic: bool = False, record: bool = False) -> Tensor:

        mu, logvar = self(state, action)

        if deterministic:
            return mu
        else:
            return rp(mu, logvar)