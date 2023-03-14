import gym
import torch
import torch.nn.functional as F
import numpy as np
from .extratyping import *
import collections


def save_checkpoint(model: Module, optimizer: Optimizer = None, path: str = "model_checkpoint.cpt", **kwargs) -> None:
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
    
    std = logvar.mul(0.5).exp_()
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


def make_act_func(params: dict) -> dict:
    if 'act_func' in params.keys():
        af = params['act_func']
        if af == 'relu':
            params['act_func'] = F.relu
        elif af == 'lrelu':
            params['act_func'] = F.leaky_relu
        elif af == 'sigmoid':
            params['act_func'] = torch.sigmoid
        elif af == 'tanh':
            params['act_func'] = F.tanh
        else:
            raise NotImplementedError(f"the activation function {params['act_func']} is not implemented")

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
    else:
        raise NotImplementedError(f"the transition model {type_} is not implemented")

    params = make_act_func(params)

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
    else:
        raise NotImplementedError(f"the policy model {type_} is not implemented")

    params = make_act_func(params)

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

    if params['act_func'] == 'relu':
        params['act_func'] = F.relu
    elif params['act_func'] == 'lrelu':
        params['act_func'] = F.leaky_relu
    elif params['act_func'] == 'sigmoid':
        params['act_func'] = torch.sigmoid
    elif params['act_func'] == 'none':
        params['act_func'] = None
    else:
        raise NotImplementedError(f"the activation function {params['act_func']} is not implemented")

    policynet = model(
        action_dim=action_dim,
        **params
    )
    if verbose: print(policynet)

    return policynet


def make_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """make an optimizer for a model"""

    optim = config['type'].lower()
    if optim == 'adam':
        Opt = torch.optim.Adam
    elif optim == 'sgd':
        Opt = torch.optim.SGD
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