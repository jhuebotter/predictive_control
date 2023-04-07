import torch
from torch import Tensor
from control_stork.nodes import LIFGroup, FastLIFGroup, NoisyFastLIFGroup
from control_stork.nodes import InputGroup, ReadoutGroup, FastReadoutGroup, DirectReadoutGroup, TimeAverageReadoutGroup
from control_stork.connections import Connection, BottleneckLinearConnection
from control_stork.models import RecurrentSpikingModel
from control_stork.initializers import FluctuationDrivenCenteredNormalInitializer, KaimingNormalInitializer, DistInitializer, AverageInitializer
#from control_stork.monitors import SpikeMonitor, SpikeCountMonitor, StateMonitor, PopulationFiringRateMonitor, PopulationSpikeCountMonitor
#from control_stork.regularizers import LowerBoundL2, UpperBoundL2
from control_stork.activations import SigmoidSpike
from control_stork.layers import Layer

from src.utils import reparameterize as rp
from src.extratyping import *


class PolicyNetRSNNPB_cstork(torch.nn.Module):
    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        target_dim: int,
        hidden_dim: int,
        num_rec_layers: int = 0,
        num_ff_layers: int = 1,
        repeat_input: int = 1,
        out_style: str = "mean",
        dt: float = 1e-3,
        device=None,
        dtype=None,
        flif_kwargs: dict = {},
        readout_kwargs: dict = {},
        neuron_type=FastLIFGroup,
        act_func = SigmoidSpike,
        connection_dims: Optional[int] = None,
        **kwargs,
    ) -> None:

        factory_kwargs = {"device": device, "dtype": dtype}
        super(PolicyNetRSNNPB_cstork, self).__init__()

        # gather layer parameters
        self.device = device
        self.action_dim = action_dim
        assert repeat_input >= 1
        self.repeat_input = repeat_input
        in_dim = state_dim + target_dim
        self.dt = dt
        assert out_style.lower() in ["mean", "last"]
        self.out_style = out_style.lower()
        # V_tau_lif = torch.tensor(flif_kwargs.get('V_tau_mean', torch.rand(hidden_dim)))
        # I_tau_lif = torch.tensor(flif_kwargs.get('I_tau_mean', torch.rand(hidden_dim)))
        # alpha_lif = torch.exp(-self.dt / I_tau_lif)
        # beta_lif = torch.exp(-self.dt / V_tau_lif)
        # V_tau_out = torch.tensor(readout_kwargs.get('V_tau_mean', torch.rand(action_dim)))
        # I_tau_out = torch.tensor(readout_kwargs.get('I_tau_mean', torch.rand(action_dim)))
        # alpha_out = torch.exp(-self.dt / I_tau_out)
        # beta_out = torch.exp(-self.dt / V_tau_out)
        
        readout_kwargs['n_readouts'] = readout_kwargs.get("n_pop", 1)

        # handle regularization
        regs = []

        # make the initializers
        initializer = FluctuationDrivenCenteredNormalInitializer(
            sigma_u=1.0,
            nu=15,
            time_step=dt,
        )

        neuron_kwargs = dict(
            tau_mem = flif_kwargs.get('V_tau_mean', 5e-3),
            tau_syn = flif_kwargs.get('I_tau_mean', 2e-3),
            activation = act_func,
        )

        if connection_dims is None:
            connection_class = Connection
            connection_kwargs = dict(
                bias = True,
            )
        else:
            connection_class = BottleneckLinearConnection
            connection_kwargs = dict(
                bias = True,
                n_dims = connection_dims,  # TODO! Make this a parameter
            )

        if 'V_tau_mean' not in readout_kwargs:
            readout_kwargs['V_tau_mean'] = 5e-3
        if 'I_tau_mean' not in readout_kwargs:
            readout_kwargs['I_tau_mean'] = 2e-3


        # make the model
        self.basis = RecurrentSpikingModel(device=device, dtype=dtype)
        input_group = prev = self.basis.add_group(InputGroup(in_dim, name='Input Group'))
        first = True
        for i in range(num_rec_layers):
            new = Layer(
                name=f'Recurrent LIF Cell Group {i+1}',
                model=self.basis,
                size=hidden_dim,
                input_group=prev,
                recurrent=True,
                regs = regs,
                connection_class=Connection if first else connection_class,
                recurrent_connection_class=connection_class,
                neuron_class=neuron_type,
                neuron_kwargs=neuron_kwargs,
                connection_kwargs=dict(bias=True) if first else connection_kwargs,
                recurrent_connection_kwargs=connection_kwargs
            )
            first = False
            initializer.initialize(new)
            prev = new.output_group
        for i in range(num_ff_layers):
            new = Layer(
                name=f'FF LIF Cell Group {i+1}',
                model=self.basis,
                size=hidden_dim,
                input_group=prev,
                recurrent=False,
                regs = regs,
                connection_class=Connection if first else connection_class,
                neuron_class=neuron_type,
                neuron_kwargs=neuron_kwargs,
                connection_kwargs=dict(bias=True) if first else connection_kwargs
            )
            first = False
            initializer.initialize(new)
            prev = new.output_group

        # make the readout
        new = Layer(
            name='Readout Pool Layer',
            model=self.basis,
            size=2 * action_dim * readout_kwargs['n_readouts'],
            input_group=prev,
            recurrent=False,
            regs=regs,
            connection_class=connection_class,
            neuron_class=FastReadoutGroup,
            neuron_kwargs=readout_kwargs,
            connection_kwargs=connection_kwargs
        )
        initializer.initialize(new)
        prev = new.output_group

        # make the readout
        if self.out_style == "mean":
            output_group = new = self.basis.add_group(TimeAverageReadoutGroup(
                2 * action_dim,
                steps=self.repeat_input,
                weight_scale=1.,
                name='Time Average Readout Group'))

        elif self.out_style == "last":
            output_group = new = self.basis.add_group(DirectReadoutGroup(
                2 * action_dim,
                weight_scale=1.,
                name='Direct Readout Group'))

        con = self.basis.add_connection(Connection(prev, new, bias=False, requires_grad=False))
        readout_initializer = AverageInitializer()
        con.init_parameters(readout_initializer)

        # configure the model
        # TODO: Rework how optimizers work!
        self.basis.configure(
            input_group,
            output_group,
            time_step=dt,
        )

        self.state_initialized = False

    def reset_state(self) -> None:
        self.state_initialized = False

    def init_state(self, batch_size: int = 1) -> None:
        self.basis.reset_state(batch_size)
        self.state_initialized = True

    def step(self, state: Tensor, target: Tensor, record: bool = False) -> Union[Tensor, Tensor]:
        x = torch.cat((state, target), -1)

        x = self.basis(x, record=record)

        mu, logvar = x.chunk(2, -1)

        return mu, logvar

    def forward(self, state: Tensor, target: Tensor, record: bool = False) -> Union[Tensor, Tensor]:

        if len(state.shape) == 2:
            state.unsqueeze_(0)

        if len(target.shape) == 2:
            target.unsqueeze_(0)

        T = state.shape[0]
        N = state.shape[1]
        D = state.shape[2]

        # control stork networks want (N, T, D)
        state = state.transpose(0, 1)
        target = target.transpose(0, 1)

        if not self.state_initialized:
            self.init_state(N)

        mu_outs = torch.empty((T, N, self.action_dim), device=self.basis.device)
        logvar_outs = torch.empty((T, N, self.action_dim), device=self.basis.device)
        for t in range(T):
            for i in range(self.repeat_input):
                mu, logvar = self.step(
                    state[:, t].view(N, 1, D), 
                    target[:, t].view(N, 1, D), 
                    record=record)
            mu_outs[t] = mu[:, -1]
            logvar_outs[t] = logvar[:, -1]

        return mu_outs, logvar_outs

    def predict(
        self, state: Tensor, target: Tensor, deterministic: bool = False
    ) -> Tensor:

        mu, logvar = self(state, target)

        if deterministic:
            return mu
        else:
            return rp(mu, logvar)

    def to(self, device: Union[str, torch.device]) -> None:
        self.device = device
        self.basis.to(device)
        super().to(device)
        return self