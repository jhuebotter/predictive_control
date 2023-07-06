import gym
import torch
import torch.nn.functional as F
import numpy as np
from .extratyping import *
import collections
from control_stork import activations
import matplotlib.figure
import wandb
from wandb import Image


def convert_figs_to_wandb_images(d: dict):

    for key, value in d.items():
        if isinstance(value, dict):
            # If the value is a dict, we recursively process it
            convert_figs_to_wandb_images(value)
        elif isinstance(value, matplotlib.figure.Figure):
            # If the value is a matplotlib Figure, we convert it to a wandb.Image
            d[key] = wandb.Image(value)
        elif isinstance(value, list):
            # If the value is a list, we check each element in the list
            for i, v in enumerate(value):
                if isinstance(v, matplotlib.figure.Figure):
                    value[i] = wandb.Image(v)
                elif isinstance(v, dict):
                    # If the list element is a dict, we recursively process it
                    convert_figs_to_wandb_images(v)


def save_checkpoint(model: Module, optimizer: Optional[Optimizer] = None, path: str = "model_checkpoint.cpt", **kwargs) -> None:
    """save model parameters to disk"""

    checkpoint = {
        "model_state_dict": model.state_dict()
    }
    if optimizer is not None:
        checkpoint.update({
            "optimizer_state_dict": optimizer.state_dict()
        })

    misc = dict(**kwargs)

    checkpoint.update({
        "misc": misc
    })

    torch.save(checkpoint, path)


def load_checkpoint(path: str, device: str = 'cpu') -> dict:
    """load model parameters from disk"""

    return torch.load(path, map_location=torch.device(device))


def load_weights_from_disk(model: Module, path: str, optim: Optional[Optimizer] = None ,device: str = 'cpu') -> \
        tuple[Module, Optional[Optimizer]]:
    """update (partial) model parameters based on previous checkpoint"""

    cp = load_checkpoint(path, device)
    current_weights = model.state_dict()
    new_weights = {**current_weights}
    new_weights.update(**cp['model_state_dict'])
    model.load_state_dict(new_weights)

    #for k, v in model.state_dict().items():
    #    print(k, torch.all(torch.eq(current_weights[k], v)))

    if optim:
        current_state = optim.state_dict()
        current_state.update(cp['optimizer_state_dict'])
        optim.load_state_dict(current_state)

    return model, optim


class ReplayMemory:
    """buffer object with maximum size to store recent experience"""

    def __init__(self, max_size: int) -> None:
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def append(self, obj: object) -> None:
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size: int) -> list:
        indices = np.random.choice(range(self.size), batch_size)
        
        return [self.buffer[index] for index in indices]


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """perform reparametrization trick for a Gaussian distribution"""
    
    std = logvar.mul(0.5).exp_()   # ! THIS WAS AN INPLACE exp_() BEFORE
    eps = torch.randn_like(std)
    
    return eps * std + mu


def make_env(config: dict, seed: int = 0) -> gym.Env:
    """create an environment object containing the task"""

    task = config['type']
    if task == 'reacher':
        print('loading 2d reacher task')
        from src.envs.reacher import ReacherEnv
        env = ReacherEnv(seed, **config['params'])
    elif task == 'reacher2':
        print('loading 2d reacher V2 task')
        from src.envs.reacher_v2 import Reacherv2Env
        env = Reacherv2Env(seed, **config['params'])
    elif task == 'plane_simple':
        print('loading simple 2d plane task')
        from src.envs.two_d_plane import TwoDPlaneEnvSimple
        env = TwoDPlaneEnvSimple(seed, **config['params'])
    elif task == 'plane':
        print('loading 2d plane task')
        from src.envs.two_d_plane import TwoDPlaneEnv
        env = TwoDPlaneEnv(seed, **config['params'])
    elif task == 'plane2':
        print('loading 2d plane task')
        from src.envs.two_d_plane_v2 import TwoDPlaneEnv
        env = TwoDPlaneEnv(seed, **config['params'])
    else:
        raise NotImplementedError(f'the task {task} is not implemented')

    return env


def make_act_fn(params: dict) -> dict:
    if 'activation_kwargs' in params.keys():
        af = params['activation_kwargs'].pop('act_fn').lower()
        if af == 'relu':
            params['act_fn'] = torch.nn.ReLU(**params['activation_kwargs'])
        elif af == 'lrelu':
            params['act_fn'] = torch.nn.LeakyReLU(**params['activation_kwargs'])
        elif af == 'sigmoid':
            params['act_fn'] = torch.nn.Sigmoid(**params['activation_kwargs'])
        elif af == 'tanh':
            params['act_fn'] = torch.nn.Tanh(**params['activation_kwargs'])
        elif af == 'sigmoidspike':
            fn = activations.SigmoidSpike
            fn.beta = params['activation_kwargs'].pop('beta', fn.beta)
            fn.gamma = params['activation_kwargs'].pop('gamma', fn.gamma)
            params['act_fn'] = fn
        elif af == 'gaussianspike':
            fn = activations.GaussianSpike
            fn.beta = params['activation_kwargs'].pop('beta', fn.beta)
            fn.gamma = params['activation_kwargs'].pop('gamma', fn.gamma)
            fn.scale = params['activation_kwargs'].pop('scale', fn.scale)
            fn.hight = params['activation_kwargs'].pop('hight', fn.hight)
            params['act_fn'] = fn
        elif af == 'superspike':
            fn = activations.SuperSpike
            fn.beta = params['activation_kwargs'].pop('beta', fn.beta)
            fn.gamma = params['activation_kwargs'].pop('gamma', fn.gamma)
            params['act_fn'] = fn
        elif af == 'default':
            params.pop('act_fn', None)
        else:
            raise NotImplementedError(f"the activation function {params['act_fn']} is not implemented")

    return params


def make_transition_model(env: gym.Env, config: dict, verbose: bool = True) -> Module:
    """create a policy network according to the parameters specified by the config file and task"""

    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]

    type_ = config['type'].lower()
    params = dict(**config['params'])
    if type_ == 'rnnpbadapt':
        from src.models.ANN_models import TransitionNetPBAdaptive
        model = TransitionNetPBAdaptive
    elif type_ == 'rsnnpb':
        from src.models.SNN_models import TransitionNetRSNNPB
        model = TransitionNetRSNNPB
    elif type_ == 'rsnncs':
        from src.models.cstork_SNN import TransitionNetRSNN_cstork
        model = TransitionNetRSNN_cstork
    else:
        raise NotImplementedError(f"the transition model {type_} is not implemented")

    params = make_act_fn(params)

    transitionnet = model(
        action_dim=action_dim,
        state_dim=state_dim,
        **params,
    )
    if verbose: print(transitionnet)

    return transitionnet


def make_policy_model(env: gym.Env, config: dict, verbose: bool = True) -> Module:
    """create a policy network according to the parameters specified by the config file and task"""
    
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]

    type_ = config['type'].lower()
    params = dict(**config['params'])
    if type_ == 'rnnpbadapt':
        from src.models.ANN_models import PolicyNetPBAdaptive
        model = PolicyNetPBAdaptive
    elif type_ == 'rsnnpb':
        from src.models.SNN_models import PolicyNetRSNNPB
        model = PolicyNetRSNNPB
    elif type_ == 'rsnnpbn':
        from src.models.SNN_models import PolicyNetRSNNPB_snntorch
        model = PolicyNetRSNNPB_snntorch
    elif type_ == 'rsnnpbcs':
        from src.models.cstork_SNN import PolicyNetRSNNPB_cstork
        model = PolicyNetRSNNPB_cstork
    elif type_ == 'rsnncs':
        from src.models.cstork_SNN import PolicyNetRSNN_cstork
        model = PolicyNetRSNN_cstork
    else:
        raise NotImplementedError(f"the policy model {type_} is not implemented")

    params = make_act_fn(params)

    policynet = model(
        action_dim=action_dim,
        state_dim=state_dim,
        target_dim=state_dim,
        **params
    )
    if verbose: print(policynet)

    return policynet


def make_policy_adaptation_model(env: gym.Env, config: dict, verbose: bool = True) -> Module:
    """create a policy network according to the parameters specified by the config file and task"""

    action_dim = env.action_space.shape[0]

    type_ = config['type'].lower()
    params = dict(**config['params'])
    if type_ == 'mlp':
        from src.models.ANN_models import PolicyAdaptationNet
        model = PolicyAdaptationNet
    else:
        raise NotImplementedError(f"the policy model {type_} is not implemented")

    if params['act_fn'] == 'relu':
        params['act_fn'] = F.relu
    elif params['act_fn'] == 'lrelu':
        params['act_fn'] = F.leaky_relu
    elif params['act_fn'] == 'sigmoid':
        params['act_fn'] = torch.sigmoid
    elif params['act_fn'] == 'none':
        params['act_fn'] = None
    else:
        raise NotImplementedError(f"the activation function {params['act_fn']} is not implemented")

    policynet = model(
        action_dim=action_dim,
        **params
    )
    if verbose: print(policynet)

    return policynet


def make_train_fn(config: dict, model: str) -> Callable:
    """create a training function according to the parameters specified by the config file"""

    assert model.lower() in ['policy', 'transition'], f"the model {model} is not implemented"
    type_ = config.get('model', {}).get('type', '').lower()
    train_fn = None
    
    if model.lower() == 'policy':

        if type_ in ['rnnpbadapt']:
            from src.training_functions import train_policynetPB_sample
            train_fn = train_policynetPB_sample
        elif type_ in ['rsnncs']:
            from src.training_functions import train_policynetSNN
            train_fn = train_policynetSNN
        else:
            raise NotImplementedError(f"the training function for policy model {type_} is not implemented")

    elif model.lower() == 'transition':

        if type_ in ['rnnpbadapt']:
            from src.training_functions import train_transitionnetRNNPBNLL_sample_unroll
            train_fn = train_transitionnetRNNPBNLL_sample_unroll
        elif type_ in ['rsnncs']:
            from src.training_functions import train_transitionnetSNN
            train_fn = train_transitionnetSNN
        else:
            raise NotImplementedError(f"the training function for transition model {type_} is not implemented")

    return train_fn


def make_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """make an optimizer for a model"""

    optim = config['type'].lower()
    if optim == 'adam':
        Opt = torch.optim.Adam
    elif optim == 'sgd':
        Opt = torch.optim.SGD
    elif optim == 'smorms3':
        from control_stork.optimizers import SMORMS3
        Opt = SMORMS3
    else:
        raise NotImplementedError(f'The optimizer {optim} is not implemented')
    if isinstance(model, torch.nn.Module):
        return Opt(model.parameters(), **config['params'])
    elif isinstance(model, list):
        return Opt([l.parameters() for l in model], **config['params'])



def dict_mean(dict_list: list[dict]) -> dict:
    """for a list of dicts with the same keys and numeric values return a dict with the same keys and averaged values"""

    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            mean_dict[key] = np.mean([d[key] for d in dict_list], axis=0)

    return mean_dict


def update_dict(d: dict, u: Optional[collections.abc.Mapping] = None) -> dict:
    """update a (nested) dictionary with the values of another (nested) dictionary."""

    if u is not None:
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = update_dict(d.get(k, {}), v)
            else:
                d[k] = v
    return d


def gradnorm(model: torch.nn.Module) -> float:
    """calculates the total L2 norm of gradients for a model.
    This function was taken 1:1 from the pytoch forum:
    https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961/2"""

    total_norm = 0.
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    return total_norm