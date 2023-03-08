import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
from .utils import ReplayMemory, gradnorm
from tqdm import tqdm
from .extratyping import *


def train_transitionnetRNNPBNLL(transition_model: Module, memory: ReplayMemory, optimizer: torch.optim.Optimizer,
                                batch_size: int, warmup_steps: int = 20, n_batches: int = 1,
                                max_norm: Optional[float] = None, **kwargs):

    """function used to update the parameters of a probabilistic transition network"""

    losses = []
    grad_norms = []
    clipped_grad_norms = []
    pbar = tqdm(range(n_batches), desc=f"{'updating transition network':30}")
    for i in pbar:

        maxlen = -1
        while maxlen <= warmup_steps:
            # TODO: find better solution for this!
            episode_batch = memory.sample(batch_size)  # [sample, step, (state, target, action, reward, next_state)]
            maxlen = min([len(e) for e in episode_batch])

        state_batch = torch.stack([torch.stack([step[0].squeeze() for step in episode[:maxlen]]) for episode in episode_batch]).transpose(0, 1)
        action_batch = torch.stack([torch.stack([step[2].squeeze() for step in episode[:maxlen]]) for episode in episode_batch]).transpose(0, 1)
        next_state_batch = torch.stack([torch.stack([step[4].squeeze() for step in episode[:maxlen]]) for episode in episode_batch]).transpose(0, 1)

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
        grad_norms.append(gradnorm(transition_model))
        if max_norm:
            clip_grad_norm_(transition_model.parameters(), max_norm)
        clipped_grad_norms.append(gradnorm(transition_model))
        optimizer.step()

        pbar.set_postfix_str(f'loss: {loss.item()}')

    return {
        'transition model loss': np.mean(losses),
        'transition model grad norm': np.mean(grad_norms),
        'transition model clipped grad norm': np.mean(clipped_grad_norms),
    }

def train_transitionnetRNNPBNLL_sample(transition_model: Module, memory: ReplayMemory, optimizer: torch.optim.Optimizer,
                                batch_size: int, warmup_steps: int = 20, n_batches: int = 1, unroll_steps: int = 20,
                                max_norm: Optional[float] = None, **kwargs):

    """function used to update the parameters of a probabilistic transition network"""

    device = next(transition_model.parameters()).device

    steps = warmup_steps + unroll_steps

    losses = []
    grad_norms = []
    clipped_grad_norms = []
    logvars_mu = []
    logvars_std = []
    vars_mu = []
    vars_std = []
    pbar = tqdm(range(n_batches), desc=f"{'updating transition network':30}")
    for i in pbar:

        transition_model.zero_grad(set_to_none=True)
        # get a batch of episodes
        episode_batch = []
        while len(episode_batch) < batch_size:
            sample_batch = memory.sample(batch_size)  # [sample, step, (state, target, action, reward, next_state)]
            for episode in sample_batch:
                if len(episode) >= steps:
                    episode_batch.append(episode)
                    if len(episode_batch) == batch_size: break

        # make a random sample from each episode of length = warmup_steps
        state_dim = episode_batch[0][0][0].size(-1)
        action_dim = episode_batch[0][0][2].size(-1)

        state_batch = torch.zeros((steps, batch_size, state_dim), device=device)
        action_batch = torch.zeros((steps, batch_size, action_dim), device=device)
        next_state_batch = torch.zeros((steps, batch_size, state_dim), device=device)

        for j in range(batch_size):
            r = torch.randint(low=0, high=len(episode_batch[j]) - steps, size=(1,))
            state_batch[:, j, :] = torch.stack([step[0].squeeze() for step in episode_batch[j][r:r+steps]])
            action_batch[:, j, :] = torch.stack([step[2].squeeze() for step in episode_batch[j][r:r+steps]])
            next_state_batch[:, j, :] = torch.stack([step[4].squeeze() for step in episode_batch[j][r:r+steps]])

        transition_model.reset_state()
        s_hat_delta_mu, s_hat_delta_logvar = transition_model(state_batch, action_batch)

        # TODO: the code below does not work but need to find a way to avoid extreme logvar values
        #with torch.no_grad():
        #    s_hat_delta_logvar = torch.clamp(s_hat_delta_logvar, -10., 10.)

        s_hat_mu = s_hat_delta_mu + state_batch
        s_hat_delta_var = torch.exp(s_hat_delta_logvar)

        loss = F.gaussian_nll_loss(s_hat_mu[warmup_steps:], next_state_batch[warmup_steps:], s_hat_delta_var[warmup_steps:])

        losses.append(loss.item())
        loss.backward()
        grad_norms.append(gradnorm(transition_model))
        if max_norm:
            clip_grad_norm_(transition_model.parameters(), max_norm)
        clipped_grad_norms.append(gradnorm(transition_model))
        optimizer.step()

        logvars_mu.append(s_hat_delta_logvar[warmup_steps:].mean().item())
        logvars_std.append(s_hat_delta_logvar[warmup_steps:].std().item())
        vars_mu.append(s_hat_delta_var[warmup_steps:].mean().item())
        vars_std.append(s_hat_delta_var[warmup_steps:].std().item())

        pbar.set_postfix_str(f'loss: {loss.item()}')

    return {
        'transition model loss': np.mean(losses),
        'transition model grad norm': np.mean(grad_norms),
        'transition model clipped grad norm': np.mean(clipped_grad_norms),
        'transition model mean logvars mean': np.mean(logvars_mu),
        'transition model mean logvars std': np.mean(logvars_std),
        'transition model mean vars mean': np.mean(vars_mu),
        'transition model mean vars std': np.mean(vars_std)
    }


def train_transitionnetRNNPBNLL_sample_unroll(transition_model: Module, memory: ReplayMemory, optimizer: torch.optim.Optimizer,
                                batch_size: int, warmup_steps: int = 20, n_batches: int = 1, unroll_steps: int = 20,
                                max_norm: Optional[float] = None, **kwargs):

    """function used to update the parameters of a probabilistic transition network"""

    device = next(transition_model.parameters()).device

    steps = warmup_steps + unroll_steps

    losses = []
    grad_norms = []
    clipped_grad_norms = []
    logvars_mu = []
    logvars_std = []
    vars_mu = []
    vars_std = []
    pbar = tqdm(range(n_batches), desc=f"{'updating transition network':30}")
    for i in pbar:

        transition_model.zero_grad(set_to_none=True)
        # get a batch of episodes
        episode_batch = []
        while len(episode_batch) < batch_size:
            sample_batch = memory.sample(batch_size)  # [sample, step, (state, target, action, reward, next_state)]
            for episode in sample_batch:
                if len(episode) >= steps:
                    episode_batch.append(episode)
                    if len(episode_batch) == batch_size: break

        # make a random sample from each episode of length = warmup_steps
        state_dim = episode_batch[0][0][0].size(-1)
        action_dim = episode_batch[0][0][2].size(-1)

        state_batch = torch.zeros((steps, batch_size, state_dim), device=device)
        action_batch = torch.zeros((steps, batch_size, action_dim), device=device)
        next_state_batch = torch.zeros((steps, batch_size, state_dim), device=device)

        for j in range(batch_size):
            r = torch.randint(low=0, high=len(episode_batch[j]) - steps, size=(1,))
            state_batch[:, j, :] = torch.stack([step[0].squeeze() for step in episode_batch[j][r:r+steps]])
            action_batch[:, j, :] = torch.stack([step[2].squeeze() for step in episode_batch[j][r:r+steps]])
            next_state_batch[:, j, :] = torch.stack([step[4].squeeze() for step in episode_batch[j][r:r+steps]])

        # reset and warm up
        transition_model.reset_state()
        s_hat_delta_mu, s_hat_delta_logvar = transition_model(state_batch[:warmup_steps], action_batch[:warmup_steps])
        s_hat_delta_std = torch.exp(0.5 * s_hat_delta_logvar)
        s_hat = state_batch[warmup_steps-1] + s_hat_delta_mu[-1] + s_hat_delta_std[-1] * torch.randn_like(s_hat_delta_std[-1], device=device)

        s_hat_mus = torch.empty((unroll_steps, batch_size, state_dim), device=device)
        s_hat_vars = torch.empty((unroll_steps, batch_size, state_dim), device=device)

        for k in range(unroll_steps):
            s_hat_delta_mu, s_hat_delta_logvar = transition_model(s_hat, action_batch[warmup_steps+k])
            s_hat_mus[k] = s_hat + s_hat_delta_mu
            s_hat_vars[k] = torch.exp(s_hat_delta_logvar)
            s_hat = s_hat + s_hat_delta_mu + torch.exp(0.5 * s_hat_delta_logvar) * torch.randn_like(s_hat, device=device)

        loss = F.gaussian_nll_loss(s_hat_mus, next_state_batch[warmup_steps:], s_hat_vars)

        losses.append(loss.item())
        loss.backward()
        grad_norms.append(gradnorm(transition_model))
        if max_norm:
            clip_grad_norm_(transition_model.parameters(), max_norm)
        clipped_grad_norms.append(gradnorm(transition_model))
        optimizer.step()

        logvars_mu.append(s_hat_vars.log().mean().item())
        logvars_std.append(s_hat_vars.log().std().item())
        vars_mu.append(s_hat_vars.mean().item())
        vars_std.append(s_hat_vars.std().item())

        pbar.set_postfix_str(f'loss: {loss.item()}')

    return {
        'transition model loss': np.mean(losses),
        'transition model grad norm': np.mean(grad_norms),
        'transition model clipped grad norm': np.mean(clipped_grad_norms),
        'transition model mean logvars mean': np.mean(logvars_mu),
        'transition model mean logvars std': np.mean(logvars_std),
        'transition model mean vars mean': np.mean(vars_mu),
        'transition model mean vars std': np.mean(vars_std)
    }


def train_policynetPB(policy_model: Module, transition_model: Module, memory: ReplayMemory,
                      optimizer: torch.optim.Optimizer, batch_size: int, loss_gain: array,
                      warmup_steps: int = 20, n_batches: int = 1, unroll_steps: int = 20, beta: float = 0.0,
                      max_norm: Optional[float] = None, deterministic_transition: bool = True, **kwargs) -> dict:
    """function used to update the parameters of a probabilistic policy network"""

    losses = []
    action_losses = []
    reg_losses = []
    loss_gain = torch.tensor(loss_gain).to(next(policy_model.parameters()).device)

    grad_norms = []
    clipped_grad_norms = []

    # TODO: this should be parameter somewhere
    if beta:
        action_target_std = 0.1
        action_target_dist = torch.distributions.normal.Normal(0., action_target_std)

    pbar = tqdm(range(n_batches), desc=f"{'updating policy network':30}")
    for i in pbar:

        optimizer.zero_grad()
        loss = torch.tensor([0.]).to(next(policy_model.parameters()).device)
        action_loss_ = torch.tensor([0.]).to(next(policy_model.parameters()).device)
        reg_loss_ = torch.tensor([0.]).to(next(policy_model.parameters()).device)

        maxlen = -1
        while maxlen <= warmup_steps:
            # TODO: find better solution for this!
            episode_batch = memory.sample(batch_size)  # [sample, step, (state, target, action, next_state)]
            maxlen = min([len(e) for e in episode_batch])

        state_batch = torch.stack([torch.stack([step[0].squeeze() for step in episode[:maxlen]]) for episode in episode_batch]).transpose(0, 1).to(next(policy_model.parameters()).device)
        target_batch = torch.stack([torch.stack([step[1].squeeze() for step in episode[:maxlen]]) for episode in episode_batch]).transpose(0, 1).to(next(policy_model.parameters()).device)
        action_batch = torch.stack([torch.stack([step[2].squeeze() for step in episode[:maxlen]]) for episode in episode_batch]).transpose(0, 1).to(next(policy_model.parameters()).device)

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
                a = a_mu + a_std * torch.randn_like(a_mu)

                s_hat_delta = transition_model.predict(new_state_hat, a, deterministic=deterministic_transition)
                s_hat = s_hat_delta + new_state_hat

                # TODO: the code below is an alternative approach directly based on the KL-divergence and seems to work
                #s_hat_delta_std = torch.exp(0.5 * s_hat_delta_logvar)
                #s_hat_dist = torch.distributions.normal.Normal(s_hat_mu, s_hat_delta_std)
                #target_loss = torch.mean(torch.distributions.kl_divergence(s_hat_dist, target_dist) * loss_gain)
                #action_loss = torch.distributions.kl_divergence(next_state_dist, target_dist).mean() \
                #              + alpha * torch.distributions.kl_divergence(action_dist, action_target_dist).mean()

                action_loss = torch.mean(torch.pow(t - s_hat, 2) * loss_gain)
                loss += action_loss
                action_loss_ += action_loss

                if beta:
                    action_dist = torch.distributions.normal.Normal(a_mu, a_std)
                    action_reg = torch.distributions.kl_divergence(action_dist, action_target_dist).mean().to(next(policy_model.parameters()).device)
                    reg_loss = beta * action_reg
                    loss += reg_loss
                    reg_loss_ += reg_loss

                new_state_hat = s_hat

        loss = loss / n_rollouts
        action_loss_ = action_loss_ / n_rollouts
        reg_loss_ = reg_loss_ / n_rollouts
        losses.append(loss.item())
        action_losses.append(action_loss_.item())
        reg_losses.append(reg_loss_.item())
        loss.backward()
        grad_norms.append(gradnorm(policy_model))
        if max_norm:
            clip_grad_norm_(policy_model.parameters(), max_norm)
        clipped_grad_norms.append(gradnorm(policy_model))
        optimizer.step()

        pbar.set_postfix_str(f'loss: {loss.item()}')

    return {
        'policy model loss': np.mean(losses),
        'policy model action loss': np.mean(action_losses),
        'policy model reg loss': np.mean(reg_losses),
        'policy model grad norm': np.mean(grad_norms),
        'policy model clipped grad norm': np.mean(clipped_grad_norms),
    }


def train_policynetPB_sample(policy_model: Module, transition_model: Module, memory: ReplayMemory,
                      optimizer: torch.optim.Optimizer, batch_size: int, loss_gain: array,
                      warmup_steps: int = 20, n_batches: int = 1, unroll_steps: int = 20, beta: float = 0.0,
                      max_norm: Optional[float] = None, deterministic_transition: bool = True, **kwargs) -> dict:
    """function used to update the parameters of a probabilistic policy network"""

    device = next(policy_model.parameters()).device

    losses = []
    action_losses = []
    reg_losses = []
    loss_gain = torch.tensor(loss_gain).to(device)

    grad_norms = []
    clipped_grad_norms = []
    logvars_mu = []
    logvars_std = []
    vars_mu = []
    vars_std = []

    # TODO: this should be parameter somewhere
    if beta:
        action_target_std = 0.1
        action_target_dist = torch.distributions.normal.Normal(0., action_target_std)

    pbar = tqdm(range(n_batches), desc=f"{'updating policy network':30}")
    for i in pbar:

        policy_model.zero_grad(set_to_none=True)
        transition_model.zero_grad(set_to_none=True)
        loss = torch.zeros(1, device=device)
        action_loss_ = torch.zeros(1, device=device)
        reg_loss_ = torch.zeros(1, device=device)

        # get a batch of episodes
        episode_batch = []
        while len(episode_batch) < batch_size:
            sample_batch = memory.sample(batch_size)  # [sample, step, (state, target, action, next_state)]
            for episode in sample_batch:
                if len(episode) >= warmup_steps:
                    episode_batch.append(episode)
                    if len(episode_batch) == batch_size: break

        # make a random sample from each episode of length = warmup_steps
        state_dim = episode_batch[0][0][0].size(-1)
        target_dim = episode_batch[0][0][1].size(-1)
        action_dim = episode_batch[0][0][2].size(-1)

        state_batch = torch.zeros((warmup_steps, batch_size, state_dim), device=device)
        target_batch = torch.zeros((warmup_steps, batch_size, target_dim), device=device)
        action_batch = torch.zeros((warmup_steps, batch_size, action_dim), device=device)

        for j in range(batch_size):
            r = torch.randint(low=0, high=len(episode_batch[j]) - warmup_steps, size=(1,))
            state_batch[:, j, :] = torch.stack([step[0].squeeze() for step in episode_batch[j][r:r+warmup_steps]])
            target_batch[:, j, :] = torch.stack([step[1].squeeze() for step in episode_batch[j][r:r+warmup_steps]])
            action_batch[:, j, :] = torch.stack([step[2].squeeze() for step in episode_batch[j][r:r+warmup_steps]])

        # reset the models
        policy_model.reset_state()
        transition_model.reset_state()

        # warm up both models
        _ = policy_model(state_batch[:-1], target_batch[:-1])
        _ = transition_model(state_batch[:-1], action_batch[:-1])

        new_state_hat = state_batch[-1]
        t = target_batch[-1]

        for k in range(unroll_steps):

            # sample an action from policy network
            a_mu, a_logvar = policy_model(new_state_hat, t)
            a_var = torch.exp(a_logvar)    # this is just for logging now and can probably be deleted later
            a_std = torch.exp(0.5 * a_logvar)
            a = a_mu + a_std * torch.randn_like(a_mu)

            logvars_mu.append(a_logvar.mean().item())
            logvars_std.append(a_logvar.std().item())
            vars_mu.append(a_var.mean().item())
            vars_std.append(a_var.std().item())

            s_hat_delta = transition_model.predict(new_state_hat, a, deterministic=deterministic_transition)
            s_hat = s_hat_delta + new_state_hat

            # TODO: the code below is an alternative approach directly based on the KL-divergence and seems to work
            #s_hat_delta_std = torch.exp(0.5 * s_hat_delta_logvar)
            #s_hat_dist = torch.distributions.normal.Normal(s_hat_mu, s_hat_delta_std)
            #target_loss = torch.mean(torch.distributions.kl_divergence(s_hat_dist, target_dist) * loss_gain)
            #action_loss = torch.distributions.kl_divergence(next_state_dist, target_dist).mean() \
            #              + alpha * torch.distributions.kl_divergence(action_dist, action_target_dist).mean()

            action_loss = torch.mean(torch.pow(t - s_hat, 2) * loss_gain)
            loss += action_loss
            action_loss_ += action_loss

            if beta:
                action_dist = torch.distributions.normal.Normal(a_mu, a_std)
                action_reg = torch.distributions.kl_divergence(action_dist, action_target_dist).mean().to(device)
                reg_loss = beta * action_reg
                loss += reg_loss
                reg_loss_ += reg_loss

            new_state_hat = s_hat

        losses.append(loss.item())
        action_losses.append(action_loss_.item())
        reg_losses.append(reg_loss_.item())
        loss.backward()
        grad_norms.append(gradnorm(policy_model))
        if max_norm:
            clip_grad_norm_(policy_model.parameters(), max_norm)
        clipped_grad_norms.append(gradnorm(policy_model))
        optimizer.step()

        pbar.set_postfix_str(f'loss: {loss.item()}')

    return {
        'policy model loss': np.mean(losses),
        'policy model action loss': np.mean(action_losses),
        'policy model reg loss': np.mean(reg_losses),
        'policy model grad norm': np.mean(grad_norms),
        'policy model clipped grad norm': np.mean(clipped_grad_norms),
        'policy model mean logvars mean': np.mean(logvars_mu),
        'policy model mean logvars std': np.mean(logvars_std),
        'policy model mean vars mean': np.mean(vars_mu),
        'policy model mean vars std': np.mean(vars_std)
    }

@torch.no_grad()
def baseline_prediction(transitionnet: Module, episode: list) -> dict:
    """function used to evaluate transition network predictions against baseline values"""

    transitionnet.reset_state()
    T = len(episode)

    states = torch.stack([step[0] for step in episode]).view((T, 1, -1))
    actions = torch.stack([step[2] for step in episode]).view((T, 1, -1))
    next_states = torch.stack([step[4] for step in episode]).view((T, 1, -1))

    # predict next state based on action with transition net
    delta_states = transitionnet.predict(states, actions, deterministic=True)
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
