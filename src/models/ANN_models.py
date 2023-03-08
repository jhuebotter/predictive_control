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


class TransitionNet(StatefulModel):
    """deterministic MLP transition network"""

    def __init__(self, action_dim: int, state_dim: int, hidden_dim: int, bias: bool = True,
                 act_func: Callable = F.leaky_relu, **kwargs) -> None:
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim, bias)
        self.fc2 = nn.Linear(hidden_dim, state_dim, bias)
        self.act_func = act_func

    def forward(self, state: Tensor, action: Tensor) -> Tensor:

        x = self.fc1(torch.cat((state, action), -1))
        x = self.act_func(x)

        return self.fc2(x)


class PolicyNet(StatefulModel):
    """deterministic MLP policy network"""

    def __init__(self, action_dim: int, state_dim: int, target_dim: int, hidden_dim: int, bias: bool = True,
                 act_func: Callable = F.leaky_relu, **kwargs) -> None:
        super().__init__()

        self.fc1 = nn.Linear(state_dim + target_dim, hidden_dim, bias)
        self.fc2 = nn.Linear(hidden_dim, action_dim, bias)
        self.act_func = act_func

    def forward(self, state: Tensor, target: Tensor) -> Tensor:

        x = self.fc1(torch.cat((state, target), -1))
        x = self.act_func(x)

        return torch.tanh(self.fc2(x))


class PolicyAdaptationNet(StatefulModel):
    """deterministic MLP policy adaptation network"""

    def __init__(self, action_dim: int, bias: bool = True,
                 act_func: Callable = None, **kwargs) -> None:
        super().__init__()

        self.bias = bias
        self.action_dim = action_dim
        self.fc1 = nn.Linear(self.action_dim, self.action_dim, self.bias)
        self.act_func = act_func
        self.reset_weights()

    def reset_weights(self):

        if self.bias:
            self.fc1.bias.data.fill_(0.0)
        #self.fc1.weight.data.fill_(0.0)
        self.fc1.weight.data = torch.eye(self.action_dim)
        #self.fc1.weight.data = torch.tensor(
        #    [[0.40808206,  0.91294525],
        #     [-0.91294525,  0.40808206]]
        #)

    def get_weights(self):

        if not self.bias:
            return self.fc1.weight.data

        return self.fc1.weight.data, self.fc1.bias.data

    def forward(self, action: Tensor) -> Tensor:

        x = self.fc1(action)
        if self.act_func is not None:
            x = self.act_func(x)

        return x


class TransitionNetGRU(StatefulModel):
    """deterministic RNN transition network"""

    def __init__(self, action_dim: int, state_dim: int, hidden_dim: int, num_layers: int = 1, bias: bool = True,
                 act_func: Callable = F.leaky_relu, **kwargs) -> None:
        super().__init__()

        self.gru1 = nn.GRU(state_dim + action_dim, hidden_dim, num_layers, bias)
        self.fc1 = nn.Linear(hidden_dim, state_dim, bias)
        self.act_func = act_func

    def forward(self, state: Tensor, action: Tensor) -> Tensor:

        if len(state.shape) == 2:
            state.unsqueeze_(0)

        if len(action.shape) == 2:
            action.unsqueeze_(0)

        x, self.h = self.gru1(torch.cat((state, action), -1), self.h)
        x = self.act_func(x)

        return self.fc1(x)


class PolicyNetGRU(StatefulModel):
    """deterministic RNN policy network"""

    def __init__(self, action_dim: int, state_dim: int, target_dim: int, hidden_dim: int, num_layers: int = 1,
                 bias: bool = True, act_func: Callable = F.leaky_relu, **kwargs) -> None:
        super().__init__()

        self.gru1 = nn.GRU(state_dim + target_dim, hidden_dim, num_layers, bias)
        self.fc1 = nn.Linear(hidden_dim, action_dim, bias)
        self.act_func = act_func

    def forward(self, state: Tensor, target: Tensor) -> Tensor:

        if len(state.shape) == 2:
            state.unsqueeze_(0)

        if len(target.shape) == 2:
            target.unsqueeze_(0)

        x, self.h = self.gru1(torch.cat((state, target), -1), self.h)
        x = self.act_func(x)

        return torch.tanh(self.fc1(x))


class TransitionNetLRNN(StatefulModel):
    """deterministic leaky RNN transition network"""

    def __init__(self, action_dim: int, state_dim: int, hidden_dim: int, bias: bool = True,
                 act_func: Callable = F.leaky_relu, **kwargs) -> None:
        super().__init__()

        self.lrnn1 = LRNN(state_dim + action_dim, hidden_dim, act_func, bias)
        self.fc1 = nn.Linear(hidden_dim, state_dim, bias)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:

        if len(state.shape) == 2:
            state.unsqueeze_(0)

        if len(action.shape) == 2:
            action.unsqueeze_(0)

        x, self.h = self.lrnn1(torch.cat((state, action), -1), self.h)

        return self.fc1(x)


class PolicyNetLRNN(StatefulModel):
    """deterministic leaky RNN policy network"""

    def __init__(self, action_dim: int, state_dim: int, target_dim: int, hidden_dim: int,
                 bias: bool = True, act_func: Callable = F.leaky_relu, **kwargs) -> None:
        super().__init__()

        self.lrnn1 = nn.GRU(state_dim + target_dim, hidden_dim, act_func, bias)
        self.fc1 = nn.Linear(hidden_dim, action_dim, bias)

    def forward(self, state: Tensor, target: Tensor) -> Tensor:

        if len(state.shape) == 2:
            state.unsqueeze_(0)

        if len(target.shape) == 2:
            target.unsqueeze_(0)

        x, self.h = self.lrnn1(torch.cat((state, target), -1), self.h)

        return torch.tanh(self.fc1(x))


class TransitionNetPB(StatefulModel):
    """probabilistic MLP transition network"""

    def __init__(self, action_dim: int, state_dim: int, hidden_dim: int, bias: bool = True,
                 act_func: Callable = F.leaky_relu, **kwargs) -> None:
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim, bias)
        self.fc_mu = nn.Linear(hidden_dim, state_dim, bias)
        self.fc_var = nn.Linear(hidden_dim, state_dim, bias)
        self.act_func = act_func

    def forward(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):

        if len(state.shape) == 3:
            state.squeeze_(0)

        if len(action.shape) == 3:
            action.squeeze_(0)

        x = self.fc1(torch.cat((state, action), -1))
        x = self.act_func(x)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar


class PolicyNetPB(StatefulModel):
    """probabilistic MLP policy network"""

    def __init__(self, action_dim: int, state_dim: int, target_dim: int, hidden_dim: int, bias: bool = True,
                 act_func: Callable = F.leaky_relu, **kwargs) -> None:
        super().__init__()

        self.fc1 = nn.Linear(state_dim + target_dim, hidden_dim, bias)
        self.fc_mu = nn.Linear(hidden_dim, action_dim, bias)
        self.fc_var = nn.Linear(hidden_dim, action_dim, bias)
        self.act_func = act_func

    def forward(self, state: Tensor, target: Tensor) -> (Tensor, Tensor):

        if len(state.shape) == 3:
            state.squeeze_(0)

        if len(target.shape) == 3:
            target.squeeze_(0)

        x = self.fc1(torch.cat((state, target), -1))
        x = self.act_func(x)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return torch.tanh(mu), logvar


class TransitionNetGRUPB(StatefulModel):
    """probabilistic RNN transition network"""

    def __init__(self, action_dim: int, state_dim: int, hidden_dim: int, num_layers: int = 1, bias: bool = True,
                 act_func: Callable = F.leaky_relu, **kwargs) -> None:
        super().__init__()

        self.gru1 = nn.GRU(state_dim + action_dim, hidden_dim, num_layers, bias)
        self.fc_mu = nn.Linear(hidden_dim, state_dim, bias)
        self.fc_var = nn.Linear(hidden_dim, state_dim, bias)
        self.act_func = act_func

    def forward(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):

        if len(state.shape) == 2:
            state.unsqueeze_(0)

        if len(action.shape) == 2:
            action.unsqueeze_(0)

        x, self.h = self.gru1(torch.cat((state, action), -1), self.h)
        x = self.act_func(x)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar


class PolicyNetGRUPB(StatefulModel):
    """probabilistic RNN policy network"""

    def __init__(self, action_dim: int, state_dim: int, target_dim: int, hidden_dim: int, num_layers: int = 1,
                 bias: bool = True, act_func: Callable = F.leaky_relu, **kwargs) -> None:
        super().__init__()

        self.gru1 = nn.GRU(state_dim + target_dim, hidden_dim, num_layers, bias)
        self.fc_mu = nn.Linear(hidden_dim, action_dim, bias)
        self.fc_var = nn.Linear(hidden_dim, action_dim, bias)
        self.act_func = act_func

    def forward(self, state: Tensor, target: Tensor) -> (Tensor, Tensor):

        if len(state.shape) == 2:
            state.unsqueeze_(0)

        if len(target.shape) == 2:
            target.unsqueeze_(0)

        x, self.h = self.gru1(torch.cat((state, target), -1), self.h)
        x = self.act_func(x)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return torch.tanh(mu), logvar


class TransitionNetLRNNPB(StatefulModel):
    """probabilistic leaky RNN transition network"""

    def __init__(self, action_dim: int, state_dim: int, hidden_dim: int, bias: bool = True,
                 act_func: Callable = F.leaky_relu, **kwargs) -> None:
        super().__init__()

        self.lrnn1 = LRNN(state_dim + action_dim, hidden_dim, act_func, bias)
        self.fc_mu = nn.Linear(hidden_dim, state_dim, bias)
        self.fc_var = nn.Linear(hidden_dim, state_dim, bias)

    def forward(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):

        if len(state.shape) == 2:
            state.unsqueeze_(0)

        if len(action.shape) == 2:
            action.unsqueeze_(0)

        x, self.h = self.lrnn1(torch.cat((state, action), -1), self.h)

        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar


class PolicyNetLRNNPB(StatefulModel):
    """probabilistic leaky RNN policy network"""

    def __init__(self, action_dim: int, state_dim: int, target_dim: int, hidden_dim: int,
                 bias: bool = True, act_func: Callable = F.leaky_relu, **kwargs) -> None:
        super().__init__()

        self.lrnn1 = nn.GRU(state_dim + target_dim, hidden_dim, act_func, bias)
        self.fc_mu = nn.Linear(hidden_dim, action_dim, bias)
        self.fc_var = nn.Linear(hidden_dim, action_dim, bias)

    def forward(self, state: Tensor, target: Tensor) -> (Tensor, Tensor):

        if len(state.shape) == 2:
            state.unsqueeze_(0)

        if len(target.shape) == 2:
            target.unsqueeze_(0)

        x, self.h = self.lrnn1(torch.cat((state, target), -1), self.h)

        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return torch.tanh(mu), logvar


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

    def __init__(self, action_dim: int, state_dim: int, target_dim: int, hidden_dim: int, num_ff_layers: int = 0,
                 bias: bool = True, act_func: Callable = F.leaky_relu, **kwargs) -> None:
        super().__init__()

        layers = OrderedDict()
        layers['fc_in'] = nn.Linear(state_dim + target_dim, hidden_dim, bias)
        for i in range(num_ff_layers):
            layers[f'fc_h{i}'] = nn.Linear(hidden_dim, hidden_dim, bias)
        layers['fc_mu'] = nn.Linear(hidden_dim, action_dim, bias)
        layers['fc_var'] = nn.Linear(hidden_dim, action_dim, bias)

        self.basis = nn.ModuleDict(layers)

        self.adaptive_layers = nn.ModuleDict({
            'fc_mu_adapt': nn.Linear(hidden_dim, action_dim, bias),
            'fc_var_adapt': nn.Linear(hidden_dim, action_dim, bias)
        })
        self.bias = bias
        self.act_func = act_func
        self.reset_adaptation_weights()
        init.zeros_(self.basis.fc_mu.bias)
        init.zeros_(self.basis.fc_var.bias)


    def forward(self, state: Tensor, target: Tensor) -> (Tensor, Tensor):

        if len(state.shape) == 3:
            state.squeeze_(0)

        if len(target.shape) == 3:
            target.squeeze_(0)

        x = self.basis['fc_in'](torch.cat((state, target), -1))
        x = self.act_func(x)
        for name, layer in self.basis.items():
            if 'fc_h' in name.lower():
                x = self.act_func(layer(x))
        mu = self.basis['fc_mu'](x)  # + self.adaptive_layers['fc_mu_adapt'](x)
        logvar = self.basis['fc_var'](x)  # + self.adaptive_layers['fc_var_adapt'](x)

        return torch.tanh(mu), logvar

    def predict(self, state: Tensor, target: Tensor, deterministic: bool = False) -> Tensor:

        mu, logvar = self(state, target)

        if deterministic:
            return mu
        else:
            return rp(mu, logvar)

class TransitionNetGRUPBAdaptive(AdaptiveModel):
    """probabilistic RNN transition network"""

    def __init__(self, action_dim: int, state_dim: int, hidden_dim: int, num_rec_layers: int = 1, num_ff_layers: int = 0,
                 bias: bool = True, act_func: Callable = F.leaky_relu, **kwargs) -> None:
        super().__init__()

        layers = OrderedDict()
        layers['gru1'] = nn.GRU(state_dim + action_dim, hidden_dim, num_rec_layers, bias)
        for i in range(num_ff_layers):
            layers[f'fc_h{i}'] = nn.Linear(hidden_dim, hidden_dim, bias)
        layers['fc_mu'] = nn.Linear(hidden_dim, state_dim, bias)
        layers['fc_var'] = nn.Linear(hidden_dim, state_dim, bias)

        self.basis = nn.ModuleDict(layers)

        self.adaptive_layers = nn.ModuleDict({
            'fc_mu_adapt': nn.Linear(hidden_dim, state_dim, bias),
            'fc_var_adapt': nn.Linear(hidden_dim, state_dim, bias)
        })
        self.act_func = act_func
        self.bias = bias
        self.reset_adaptation_weights()
        init.zeros_(self.basis.fc_mu.bias)
        init.zeros_(self.basis.fc_var.bias)


    def forward(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):

        if len(state.shape) == 2:
            state.unsqueeze_(0)

        if len(action.shape) == 2:
            action.unsqueeze_(0)

        x, self.h = self.basis['gru1'](torch.cat((state, action), -1), self.h)
        x = self.act_func(x)
        for name, layer in self.basis.items():
            if 'fc_h' in name.lower():
                x = self.act_func(layer(x))
        mu = self.basis['fc_mu'](x)  # + self.adaptive_layers['fc_mu_adapt'](x)
        logvar = self.basis['fc_var'](x)  # + self.adaptive_layers['fc_var_adapt'](x)

        return mu, logvar

    def predict(self, state: Tensor, target: Tensor, deterministic: bool = False) -> Tensor:

        mu, logvar = self(state, target)

        if deterministic:
            return mu
        else:
            return rp(mu, logvar)