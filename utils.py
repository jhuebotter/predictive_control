import gym
import torch
import torch.nn.functional as F
import numpy as np
from types import *


def save_checkpoint(model: Module, optimizer: torch.optim.Optimizer = None, step: int = None,
                    loss: float = None, path: str = "model_checkpoint.cpt") -> None:
    """save model parameters to disk"""

    checkpoint = {
        "model_state_dict": model.state_dict()
    }
    if optimizer is not None:
        checkpoint.update({
            "optimizer_state_dict": optimizer.state_dict()
        })
    if step is not None:
        checkpoint.update({
            "step": step
        })
    if loss is not None:
        checkpoint.update({
            "loss": loss
        })

    torch.save(checkpoint, path)


def load_checkpoint(path: str) -> dict:
    """load model parameters from disk"""

    return torch.load(path)


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
    
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    
    return eps * std + mu


def make_env(config: dict) -> gym.Env:
    """create an environment object containing the task"""

    task = config['task']
    seed = config['seed']
    if task == 'reacher':
        print('loading 2d reacher task')
        from src.envs.reacher import ReacherEnv
        env = ReacherEnv(seed, max_episode_steps=config['env_steps'])
    elif task == 'plane_simple':
        print('loading simple 2d plane task')
        from src.envs.two_d_plane import TwoDPlaneEnvSimple
        env = TwoDPlaneEnvSimple(seed, max_episode_steps=config['env_steps'])
    elif task == 'plane':
        print('loading 2d plane task')
        from src.envs.two_d_plane import TwoDPlaneEnv
        env = TwoDPlaneEnv(seed, max_episode_steps=config['env_steps'])
    else:
        raise NotImplementedError(f'the task {task} is not implemented')

    return env


def make_transition_model(env: gym.Env, config: dict, verbose: bool = True) -> Module:
    """create a policy network according to the parameters specified by the config file and task"""

    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]

    params = config['tra_params']
    if params['model'] == 'MLP':
        from src.models.ANN_models import TransitionNet
        model = TransitionNet
    elif params['model'] == 'GRU':
        from src.models.ANN_models import TransitionNetGRU
        model = TransitionNetGRU
    elif params['model'] == 'LRNN':
        from src.models.ANN_models import TransitionNetLRNN
        model = TransitionNetLRNN
    elif params['model'] == 'MLPPB':
        from src.models.ANN_models import TransitionNetPB
        model = TransitionNetPB
    elif params['model'] == 'GRUPB':
        from src.models.ANN_models import TransitionNetGRUPB
        model = TransitionNetGRUPB
    elif params['model'] == 'LRNNPB':
        from src.models.ANN_models import TransitionNetLRNNPB
        model = TransitionNetLRNNPB
    else:
        raise NotImplementedError(f"the transition model {params['model']} is not implemented")

    if params['act_func'] == 'relu':
        params['act_func'] = F.relu
    elif params['act_func'] == 'lrelu':
        params['act_func'] = F.leaky_relu
    elif params['act_func'] == 'sigmoid':
        params['act_func'] = torch.sigmoid
    else:
        raise NotImplementedError(f"the activation function {params['act_func']} is not implemented")

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

    params = config['pol_params']
    if params['model'] == 'MLP':
        from src.models.ANN_models import PolicyNet
        model = PolicyNet
    elif params['model'] == 'GRU':
        from src.models.ANN_models import PolicyNetGRU
        model = PolicyNetGRU
    elif params['model'] == 'LRNN':
        from src.models.ANN_models import PolicyNetLRNN
        model = PolicyNetLRNN
    elif params['model'] == 'MLPPB':
        from src.models.ANN_models import PolicyNetPB
        model = PolicyNetPB
    elif params['model'] == 'GRUPB':
        from src.models.ANN_models import PolicyNetGRUPB
        model = PolicyNetGRUPB
    elif params['model'] == 'LRNNPB':
        from src.models.ANN_models import PolicyNetLRNNPB
        model = PolicyNetLRNNPB
    else:
        raise NotImplementedError(f"the policy model {params['model']} is not implemented")

    if params['act_func'] == 'relu':
        params['act_func'] = F.relu
    elif params['act_func'] == 'lrelu':
        params['act_func'] = F.leaky_relu
    elif params['act_func'] == 'sigmoid':
        params['act_func'] = torch.sigmoid
    else:
        raise NotImplementedError(f"the activation function {params['act_func']} is not implemented")

    policynet = model(
        action_dim=action_dim,
        state_dim=state_dim,
        target_dim=state_dim,
        **params
    )
    if verbose: print(policynet)

    return policynet


def dict_mean(dict_list: list[dict]) -> dict:
    """for a list of dicts with the same keys and numeric values return a dict with the same keys and averaged values"""
    
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = np.mean([d[key] for d in dict_list], axis=0)
    
    return mean_dict