import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.leakyRNN import LRNN
from extratyping import *


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

        return mu, logvar


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

        return mu, logvar


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

        return mu, logvar