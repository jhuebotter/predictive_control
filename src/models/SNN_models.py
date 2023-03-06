import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter, init
import math
from src.extratyping import *
from typing import Callable, Iterable, Optional, Union
from src.utils import reparameterize as rp
from abc import ABC
from collections import OrderedDict


def sparsify_matrix(mat: Tensor, p: float = 1.0) -> Tensor:
    """helper function to make weight matrices sparse if desired."""
    new_mat = mat.clone()
    new_mat[torch.rand_like(mat) > p] = 0.

    return new_mat


def simple_spike(mem: Tensor) -> Tensor:

    return (mem > 0.).float()


class SurrGradSpike(torch.autograd.Function):
    """
    This surrogate gradient function class is entirely taken from https://github.com/fzenke/spytorch/blob/main/notebooks/SpyTorchTutorial1.ipynb

    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 100.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad

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


class FLIF_B(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None, dtype=None, dt=1e-3, spike_func: Callable = SurrGradSpike.apply,
                 recurrent: bool = True, train_v0: bool = False, return_V: bool = False,
                 train_threshold: bool = False, threshold_std: float = 0.0, threshold_delta_max: float = 1.,
                 train_threshold_tau: bool = False, threshold_tau_mean: float = 0.99, threshold_tau_std: float = 0.0,  # look up plausible values
                 train_V_tau: bool = False, V_tau_mean: float = 10e-3, V_tau_std: float = 0.0,
                 train_I_tau: bool = False, I_tau_mean: float = 5e-3, I_tau_std: float = 0.0,
                 input_dropout: float = 0.0, input_density: float = 1.0,
                 recurrent_dropout: float = 0.0, recurrent_density: float = 1.0
                 ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(FLIF_B, self).__init__()
        self.in_features = in_features    # inputs
        self.out_features = out_features  # N neurons

        self.dt = dt                      # sim time step size
        self.spike_func = spike_func      # this is where surrogate gradients happen
        self.recurrent = recurrent
        self.return_V = return_V

        self.train_V0 = train_v0                           # initial membrane potential state
        self.train_threshold = train_threshold             # makes neuron threshold jump at spike
        self.threshold_std = threshold_std
        self.threshold_delta_max = threshold_delta_max
        self.train_threshold_tau = train_threshold_tau     # threshold decay time constant
        self.threshold_tau_mean = threshold_tau_mean
        self.threshold_tau_std = threshold_tau_std
        self.train_V_tau = train_V_tau                     # membrane decay time constant
        self.V_tau_mean = V_tau_mean
        self.V_tau_std = V_tau_std
        self.train_I_tau = train_I_tau                     # synapse decay time constant
        self.I_tau_mean = I_tau_mean
        self.I_tau_std = I_tau_std

        self.input_con = nn.Linear(in_features, out_features, False, device, dtype)  # make sparse?
        self.input_dropout = input_dropout
        self.input_density = input_density
        if self.recurrent:
            self.recurrent_con = nn.Linear(out_features, out_features, False, device, dtype)  # make sparse or factorize?
            self.recurrent_dropout = recurrent_dropout
            self.recurrent_density = recurrent_density

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.register_buffer('V_rest', torch.zeros(out_features, **factory_kwargs))
        self.V_0 = Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=self.train_V0)
        self.threshold = Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=self.train_threshold)
        self.register_buffer('threshold_delta', torch.empty(out_features, **factory_kwargs))
        self.threshold_tau = Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=self.train_threshold_tau)            # threshold decay time constant
        self.V_tau = Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=self.train_V_tau)
        self.I_tau = Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=self.train_I_tau)

        self.state_initialized = False

        self.reset_parameters()

    def reset_parameters(self) -> None:

        init.kaiming_normal_(self.input_con.weight)         # adapt ?
        if self.input_density < 1.:
            self.input_con.weight.data = sparsify_matrix(self.input_con.weight.data, self.input_density)   # TODO: make this a hyperparamter
        self.input_con.weight.data *= 0.05
        if self.recurrent:
            init.orthogonal_(self.recurrent_con.weight)     # adapt ?
            if self.recurrent_density < 1.:
                self.recurrent_con.weight.data = sparsify_matrix(self.recurrent_con.weight.data, self.recurrent_density)   # TODO: make this a hyperparamter
        init.zeros_(self.V_0)
        if self.threshold_std > 0.:
            init.trunc_normal_(self.threshold, mean=1., std=self.threshold_std, a=0.5, b=1.5)
        else:
            init.ones_(self.threshold)
        init.uniform_(self.threshold_delta, 0, self.threshold_delta_max)    # replace 0 by - self.threshold_delta_max?
        if self.threshold_tau_std > 0.:
            init.trunc_normal_(self.threshold_tau, mean=self.threshold_tau_mean, std=self.threshold_tau_std, a=0., b=1.)
        else:
            init.constant_(self.threshold_tau, self.threshold_tau_mean)
        if self.V_tau_std > 0.:
            init.trunc_normal_(self.V_tau, mean=self.V_tau_mean, std=self.V_tau_std, a=0., b=1.)
        else:
            init.constant_(self.V_tau, self.V_tau_mean)
        if self.I_tau_std > 0.:
            init.trunc_normal_(self.I_tau, mean=self.I_tau_mean, std=self.I_tau_std, a=0., b=1.)
        else:
            init.constant_(self.I_tau, self.I_tau_mean)
        if self.bias is not None:
            init.constant_(self.bias, 0.)

    def init_state(self, batch_size: int) -> None:

        self.V_t = self.V_0.repeat((batch_size, 1))
        self.I_t = torch.zeros((batch_size, self.out_features), device=self.V_0.device)
        self.threshold_eff = self.threshold.repeat((batch_size, 1))
        self.spk = torch.zeros((batch_size, self.out_features), device=self.V_0.device)   # change dtype ?
        self.state_initialized = True

    def forward(self, inpt: Tensor) -> Tensor:

        if not self.state_initialized:
            self.init_state(inpt.shape[0])

        # calculate decay variables
        alpha = torch.clamp(torch.exp(-self.dt/self.I_tau), 0., 1.)
        beta  = torch.clamp(torch.exp(-self.dt/self.V_tau), 0., 1.)

        # integrate new input
        I_in = F.dropout(self.input_con(inpt), self.input_dropout, True, False)
        self.I_t = alpha * self.I_t + I_in
        if self.recurrent:
            I_rec = F.dropout(self.recurrent_con(self.spk), self.recurrent_dropout, True, False)
            self.I_t = self.I_t + I_rec
        if self.bias is not None:
            self.I_t = self.I_t + self.bias

        # TODO: add option for reset to V_rest instead
        self.V_t = beta * self.V_t + self.I_t

        # make spike
        V_shift = self.V_t - self.threshold_eff
        self.spk = self.spike_func(V_shift)

        self.V_t = self.V_t - self.spk.detach() * self.threshold_eff     # subtractive method

        # adjust threshold if spiked
        self.threshold_eff = self.threshold_tau * (self.threshold_eff - self.threshold) + self.threshold + self.threshold_delta * self.spk

        if self.return_V: return self.V_t
        return self.spk


class Readout(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None, dtype=None, dt=1e-3,
                 train_V_tau: bool = False, V_tau_mean: float = 10e-3, V_tau_std: float = 0.0,
                 train_I_tau: bool = False, I_tau_mean: float = 5e-3, I_tau_std: float = 0.0,
                 input_dropout: float = 0.0, input_density: float = 1.0,
                 ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Readout, self).__init__()
        self.in_features = in_features  # inputs
        self.out_features = out_features  # N neurons

        self.dt = dt  # sim time step size

        self.train_V_tau = train_V_tau  # membrane decay time constant
        self.V_tau_mean = V_tau_mean
        self.V_tau_std = V_tau_std
        self.train_I_tau = train_I_tau  # synapse decay time constant
        self.I_tau_mean = I_tau_mean
        self.I_tau_std = I_tau_std

        self.input_con = nn.Linear(in_features, out_features, False, device, dtype)  # make sparse?
        self.input_dropout = input_dropout
        self.input_density = input_density

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.register_buffer('V_rest', torch.zeros(out_features, **factory_kwargs))
        self.V_0 = Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=False)
        self.V_tau = Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=self.train_V_tau)
        self.I_tau = Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=self.train_I_tau)

        self.state_initialized = False

        self.reset_parameters()

    def reset_parameters(self) -> None:

        init.kaiming_normal_(self.input_con.weight)  # adapt ?
        if self.input_density < 1.:
            self.input_con.weight.data = sparsify_matrix(
                self.input_con.weight.data,
                self.input_density
            )
        self.input_con.weight.data *= 0.05
        init.zeros_(self.V_0)

        if self.V_tau_std > 0.:
            init.trunc_normal_(self.V_tau, mean=self.V_tau_mean, std=self.V_tau_std, a=0., b=1.)
        else:
            init.constant_(self.V_tau, self.V_tau_mean)
        if self.I_tau_std > 0.:
            init.trunc_normal_(self.I_tau, mean=self.I_tau_mean, std=self.I_tau_std, a=0., b=1.)
        else:
            init.constant_(self.I_tau, self.I_tau_mean)
        if self.bias is not None:  # adjust to sensible value
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.recurrent_con.weight)
            bound = 0.1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def init_state(self, batch_size: int) -> None:

        self.V_t = self.V_0.repeat((batch_size, 1))
        self.I_t = torch.zeros((batch_size, self.out_features), device=self.V_0.device)
        self.state_initialized = True

    def forward(self, inpt) -> Tensor:

        if not self.state_initialized:
            self.init_state(inpt.shape[0])

        # integrate new input
        I_in = F.dropout(self.input_con(inpt), self.input_dropout, True, False)

        alpha = torch.exp(-self.dt / self.I_tau)
        beta = torch.exp(-self.dt / self.V_tau)

        self.I_t = alpha * self.I_t + I_in
        if self.bias is not None:
            self.I_t = self.I_t + self.bias
        self.V_t = beta * self.V_t + self.I_t

        return self.V_t


class TransitionNetRSNNPB(Module):
    def __init__(self, action_dim: int, state_dim: int, hidden_dim: int, num_layers: int = 1,
                 bias: bool = True, repeat_input: int = 1, out_style: str = 'mean', dt: float = 1e-3, device=None,
                 dtype=None, flif_kwargs: dict = {}, readout_kwargs: dict = {}) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransitionNetRSNNPB, self).__init__()

        layers = OrderedDict()
        layers['lif1'] = FLIF_B(state_dim + action_dim, hidden_dim, bias=bias, dt=dt, **factory_kwargs, **flif_kwargs)
        for i in range(num_layers - 1):
            layers[f'lif{2+i}'] = FLIF_B(hidden_dim, hidden_dim, bias=bias, dt=dt, **factory_kwargs, **flif_kwargs)
        layers['mu'] = Readout(hidden_dim, state_dim, dt=dt, **factory_kwargs, **readout_kwargs)
        layers['logvar'] = Readout(hidden_dim, state_dim, dt=dt, **factory_kwargs, **readout_kwargs)

        self.basis = nn.ModuleDict(layers)

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

    def step(self, state: Tensor, action: Tensor) -> [Tensor, Tensor]:
        x = torch.cat((state, action), -1)
        for name, layer in self.basis.items():
            if 'lif' in name.lower():
                x = layer(x)
        return self.basis.mu(x), self.basis.logvar(x)

    def forward(self, state: Tensor, action: Tensor) -> [Tensor, Tensor]:

        if len(state.shape) == 2:
            state.unsqueeze_(0)

        if len(action.shape) == 2:
            action.unsqueeze_(0)

        if not self.state_initialized:
            self.init_state(state.shape[1])

        T = state.shape[0]
        mu_outs = []
        logvar_outs = []
        for t in range(T):
            mus, logvars = [], []
            for i in range(self.repeat_input):
                mu, logvar = self.step(state[t], action[t])
                mus.append(mu)
                logvars.append(logvar)
            if self.out_style == 'last':
                mu_outs.append(mu)
                logvar_outs.append(logvar)
            elif self.out_style == 'mean':
                mus = torch.stack(mus)
                logvars = torch.stack(logvars)
                mu_outs.append(mus.mean(dim=0))
                logvar_outs.append(logvars.mean(dim=0))

        return torch.stack(mu_outs), torch.stack(logvar_outs)

    def predict(self, state: Tensor, action: Tensor, deterministic: bool = False):

        mu, logvar = self(state, action)

        if deterministic:
            return mu
        else:
            return rp(mu, logvar)


class PolicyNetFFSNNPB(Module):
    def __init__(self, action_dim: int, state_dim: int, target_dim: int, hidden_dim: int, num_layers: int = 1,
                 bias: bool = True, repeat_input: int = 1, out_style: str = 'mean', dt: float = 1e-3, device=None,
                 dtype=None, flif_kwargs: dict = {'recurrent': False}, readout_kwargs: dict = {}) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PolicyNetFFSNNPB, self).__init__()

        layers = OrderedDict()
        layers['lif1'] = FLIF_B(state_dim + target_dim, hidden_dim, bias=bias, dt=dt, **factory_kwargs, **flif_kwargs)
        for i in range(num_layers - 1):
            layers[f'lif{2+i}'] = FLIF_B(hidden_dim, hidden_dim, bias=bias, dt=dt, **factory_kwargs, **flif_kwargs)
        layers['mu'] = Readout(hidden_dim, action_dim, dt=dt, **factory_kwargs, **readout_kwargs)
        layers['logvar'] = Readout(hidden_dim, action_dim, dt=dt, **factory_kwargs, **readout_kwargs)

        self.basis = nn.ModuleDict(layers)

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

    def step(self, state: Tensor, target: Tensor) -> [Tensor, Tensor]:
        x = torch.cat((state, target), -1)
        for name, layer in self.basis.items():
            if 'lif' in name.lower():
                x = layer(x)
        return self.basis.mu(x), self.basis.logvar(x)

    def forward(self, state: Tensor, target: Tensor) -> [Tensor, Tensor]:

        if len(state.shape) == 2:
            state.unsqueeze_(0)

        if len(target.shape) == 2:
            target.unsqueeze_(0)

        if not self.state_initialized:
            self.init_state(state.shape[1])

        T = state.shape[0]
        mu_outs = []
        logvar_outs = []
        for t in range(T):
            mus, logvars = [], []
            for i in range(self.repeat_input):
                mu, logvar = self.step(state[t], target[t])
                mus.append(mu)
                logvars.append(logvar)
            if self.out_style == 'last':
                mu_outs.append(mu)
                logvar_outs.append(logvar)
            elif self.out_style == 'mean':
                mus = torch.stack(mus)
                logvars = torch.stack(logvars)
                mu_outs.append(mus.mean(dim=0))
                logvar_outs.append(logvars.mean(dim=0))

        return torch.stack(mu_outs), torch.stack(logvar_outs)

    def predict(self, state: Tensor, target: Tensor, deterministic: bool = False) -> Tensor:

        mu, logvar = self(state, target)

        if deterministic:
            return mu
        else:
            return rp(mu, logvar)