import torch
import torch.nn.functional as F
import numpy as np
from utils import ReplayMemory, reparameterize as rp
from tqdm import tqdm
from types import *


def train_transitionnetRNNPBNLL(transition_model: Module, memory: ReplayMemory, optimizer: torch.optim.Optimizer,
                           batch_size: int, warmup_steps: int = 20, n_batches: int = 1):
    """function used to update the parameters of a probabilistic transition network"""

    losses = []
    pbar = tqdm(range(n_batches), desc=f"{'updating transition network':30}")
    for i in pbar:

        maxlen = -1
        while maxlen <= warmup_steps:
            # TODO: find better solution for this!
            episode_batch = memory.sample(batch_size)  # [sample, step, (state, target, action, next_state)]
            maxlen = min([len(e) for e in episode_batch])

        state_batch = torch.stack([torch.stack([step[0].squeeze() for step in episode[:maxlen]]) for episode in episode_batch]).transpose(0, 1)
        action_batch = torch.stack([torch.stack([step[2].squeeze() for step in episode[:maxlen]]) for episode in episode_batch]).transpose(0, 1)
        next_state_batch = torch.stack([torch.stack([step[3].squeeze() for step in episode[:maxlen]]) for episode in episode_batch]).transpose(0, 1)

        transition_model.reset_state()
        s_hat_delta_mu, s_hat_delta_logvar = transition_model(state_batch, action_batch)

        # TODO: the code below does not work but need to find a way to avoid extreme logvar values
        #with torch.no_grad():
        #    s_hat_delta_logvar = torch.clamp(s_hat_delta_logvar, -10., 10.)

        s_hat_mu = s_hat_delta_mu + state_batch
        s_hat_delta_var = torch.exp(s_hat_delta_logvar)

        loss = F.gaussian_nll_loss(s_hat_mu, next_state_batch, s_hat_delta_var)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix_str(f'loss: {loss.item()}')

    return {'transition model loss': np.mean(losses)}


def train_policynetPB(policy_model: Module, transition_model: Module, memory: ReplayMemory,
                      optimizer: torch.optim.Optimizer, batch_size: int, loss_gain: array,
                      warmup_steps: int = 20, n_batches: int = 1, unroll_steps: int = 20, beta: float = 0.0) -> dict:
    """function used to update the parameters of a probabilistic policy network"""

    losses = []
    action_losses = []
    reg_losses = []
    loss_gain = torch.tensor(loss_gain)

    # TODO: this should be parameter somewhere
    action_target_std = 0.01

    pbar = tqdm(range(n_batches), desc=f"{'updating policy network':30}")
    for i in pbar:

        optimizer.zero_grad()
        loss = torch.tensor([0.])
        action_loss_ = torch.tensor([0.])
        reg_loss_ = torch.tensor([0.])

        maxlen = -1
        while maxlen <= warmup_steps:
            # TODO: find better solution for this!
            episode_batch = memory.sample(batch_size)  # [sample, step, (state, target, action, next_state)]
            maxlen = min([len(e) for e in episode_batch])

        state_batch = torch.stack([torch.stack([step[0].squeeze() for step in episode[:maxlen]]) for episode in episode_batch]).transpose(0, 1)
        target_batch = torch.stack([torch.stack([step[1].squeeze() for step in episode[:maxlen]]) for episode in episode_batch]).transpose(0, 1)
        action_batch = torch.stack([torch.stack([step[2].squeeze() for step in episode[:maxlen]]) for episode in episode_batch]).transpose(0, 1)

        # TODO: is this really the appropriate way to train an RNN?
        n_rollouts = maxlen // warmup_steps
        for w in range(n_rollouts):

            policy_model.reset_state()
            transition_model.reset_state()

            _ = policy_model(state_batch[warmup_steps*w:warmup_steps*(w+1)], target_batch[warmup_steps*w:warmup_steps*(w+1)])
            _ = transition_model(state_batch[warmup_steps*w:warmup_steps*(w+1)], action_batch[warmup_steps*w:warmup_steps*(w+1)])

            new_state_hat = state_batch[warmup_steps]
            t = target_batch[warmup_steps]

            for j in range(unroll_steps):

                a_mu, a_logvar = policy_model(new_state_hat, t)
                # TODO: the code below does not work but need to find a way to avoid extreme logvar values
                #with torch.no_grad():
                #    a_logvar = torch.clamp(a_logvar, -10., 10.)

                a_std = torch.exp(0.5 * a_logvar)
                action_dist = torch.distributions.normal.Normal(a_mu, a_std)
                a = action_dist.rsample()

                s_hat_delta_mu, s_hat_delta_logvar = transition_model(new_state_hat, a)
                s_hat_mu = s_hat_delta_mu + new_state_hat

                # TODO: the code below is an alternative approach directly based on the KL-divergence and seems to work
                #s_hat_delta_std = torch.exp(0.5 * s_hat_delta_logvar)
                #s_hat_dist = torch.distributions.normal.Normal(s_hat_mu, s_hat_delta_std)
                #target_loss = torch.mean(torch.distributions.kl_divergence(s_hat_dist, target_dist) * loss_gain)
                #action_loss = torch.distributions.kl_divergence(next_state_dist, target_dist).mean() \
                #              + alpha * torch.distributions.kl_divergence(action_dist, action_target_dist).mean()

                action_loss = torch.mean(torch.pow(t - s_hat_mu, 2) * loss_gain)

                action_target_dist = torch.distributions.normal.Normal(0., action_target_std)
                action_reg = torch.distributions.kl_divergence(action_dist, action_target_dist).mean()
                reg_loss = beta * action_reg

                loss += action_loss + reg_loss
                action_loss_ += action_loss
                reg_loss_ += reg_loss

                new_state_hat = s_hat_mu

        loss = loss / n_rollouts
        action_loss_ = action_loss_ / n_rollouts
        reg_loss_ = reg_loss_ / n_rollouts
        losses.append(loss.item())
        action_losses.append(action_loss_.item())
        reg_losses.append(reg_loss_.item())
        loss.backward()
        optimizer.step()

        pbar.set_postfix_str(f'loss: {loss.item()}')

    return {
        'policy model loss': np.mean(losses),
        'policy model action loss': np.mean(action_losses),
        'policy model reg loss': np.mean(reg_losses)
    }


def baseline_prediction(transitionnet: Module, episode: list) -> dict:
    """function used to evaluate transition network predictions against baseline values"""

    transitionnet.reset_state()

    states = torch.stack([step[0] for step in episode])
    actions = torch.stack([step[2] for step in episode])
    next_states = torch.stack([step[3] for step in episode])

    # predict next state based on action with transition net
    delta_states = rp(*transitionnet(states, actions))
    predicted_states = states + delta_states.detach()
    predicted_state_mse = torch.pow(predicted_states - next_states, 2).mean()
    #print("predicted state MSE:", predicted_state_mse.item())

    # use linear extrapolation as a baseline estimate
    previous_deltas = states[1:] - states[:-1]
    extrapolated_states = states[1:] + previous_deltas
    extrapolated_state_mse = torch.pow(extrapolated_states - next_states[1:], 2).mean()
    #print("extrapolated state MSE:", extrapolated_state_mse.item())

    # use current state as a second baseline estimate
    current_state_mse = torch.pow(states - next_states, 2).mean()
    #print("current state MSE:", current_state_mse.item())

    return {
        'predicted_state_MSE': predicted_state_mse.item(),
        'extrapolated_state_MSE': extrapolated_state_mse.item(),
        'current_state_MSE': current_state_mse.item()
    }