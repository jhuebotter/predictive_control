import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
from .utils import ReplayMemory, gradnorm, reparameterize as rp
from tqdm import tqdm
from .extratyping import *
import time


def train_transitionnetRNNPBNLL_sample(
    transition_model: Module,
    memory: ReplayMemory,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    warmup_steps: int = 20,
    n_batches: int = 1,
    unroll_steps: int = 20,
    max_norm: Optional[float] = None,
    **kwargs,
):
    """function used to update the parameters of a probabilistic transition network"""

    device = next(transition_model.parameters()).device

    losses = []
    grad_norms = []
    clipped_grad_norms = []
    logvars_mu = []
    logvars_std = []
    vars_mu = []
    vars_std = []
    pbar = tqdm(range(n_batches), desc=f"{'updating transition network':30}")
    for i in pbar:
        
        # sample a batch of episodes from memory
        state_batch, target_batch, action_batch, reward_batch, next_state_batch = sample_batch(
            batch_size, memory, warmup_steps, unroll_steps, device
        )

        # reset the model
        transition_model.zero_grad(set_to_none=True)
        transition_model.reset_state()

        # warm up
        s_hat_delta_mu, s_hat_delta_logvar = transition_model(state_batch, action_batch)

        # TODO: the code below does not work but need to find a way to avoid extreme logvar values
        # with torch.no_grad():
        #    s_hat_delta_logvar = torch.clamp(s_hat_delta_logvar, -10., 10.)

        s_hat_mu = s_hat_delta_mu + state_batch
        s_hat_delta_var = torch.exp(s_hat_delta_logvar)

        loss = F.gaussian_nll_loss(
            s_hat_mu[warmup_steps:],
            next_state_batch[warmup_steps:],
            s_hat_delta_var[warmup_steps:],
        )

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

        pbar.set_postfix_str(f"loss: {loss.item()}")

    return {
        "transition model loss": np.mean(losses),
        "transition model grad norm": np.mean(grad_norms),
        "transition model clipped grad norm": np.mean(clipped_grad_norms),
        "transition model mean logvars mean": np.mean(logvars_mu),
        "transition model mean logvars std": np.mean(logvars_std),
        "transition model mean vars mean": np.mean(vars_mu),
        "transition model mean vars std": np.mean(vars_std),
    }


def train_transitionnetRNNPBNLL_sample_unroll(
    transition_model: Module,
    memory: ReplayMemory,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    warmup_steps: int = 20,
    n_batches: int = 1,
    unroll_steps: int = 20,
    max_norm: Optional[float] = None,
    **kwargs,
):
    """function used to update the parameters of a probabilistic transition network"""

    device = next(transition_model.parameters()).device

    losses = []
    grad_norms = []
    clipped_grad_norms = []
    logvars_mu = []
    logvars_std = []
    vars_mu = []
    vars_std = []
    pbar = tqdm(range(n_batches), desc=f"{'updating transition network':30}")
    for i in pbar:

        # sample a batch of episodes from memory
        state_batch, target_batch, action_batch, reward_batch, next_state_batch = sample_batch(
            batch_size, memory, warmup_steps, unroll_steps, device
        )

        # reset and warm up
        transition_model.zero_grad(set_to_none=True)
        transition_model.reset_state()
        s_hat_delta_mu, s_hat_delta_logvar = transition_model(
            state_batch[:warmup_steps], action_batch[:warmup_steps]
        )
        s_hat = state_batch[warmup_steps]

        state_dim = state_batch.shape[-1]

        # make a container for the predictions to go in
        s_hat_mus = torch.empty((unroll_steps, batch_size, state_dim), device=device)
        s_hat_vars = torch.empty((unroll_steps, batch_size, state_dim), device=device)

        # autoregressive prediction
        for k in range(unroll_steps):
            s_hat_delta_mu, s_hat_delta_logvar = transition_model(
                s_hat, action_batch[warmup_steps + k]
            )
            s_hat_mus[k] = s_hat + s_hat_delta_mu
            s_hat_vars[k] = torch.exp(s_hat_delta_logvar)
            s_hat = s_hat + rp(s_hat_delta_mu, s_hat_delta_logvar)

        # TODO: CHECK IF IT MAKES A DIFFERENCE TO DO THIS STEP BY STEP
        # compare predictions with ground truth
        loss = F.gaussian_nll_loss(
            s_hat_mus, next_state_batch[warmup_steps:], s_hat_vars
        ) # this already computes the mean over steps

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

        pbar.set_postfix_str(f"loss: {loss.item()}")

    return {
        "transition model loss": np.mean(losses),
        "transition model grad norm": np.mean(grad_norms),
        "transition model clipped grad norm": np.mean(clipped_grad_norms),
        "transition model mean logvars mean": np.mean(logvars_mu),
        "transition model mean logvars std": np.mean(logvars_std),
        "transition model mean vars mean": np.mean(vars_mu),
        "transition model mean vars std": np.mean(vars_std),
    }


def train_policynetPB_sample(
    policy_model: Module,
    transition_model: Module,
    memory: ReplayMemory,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    loss_gain: array,
    warmup_steps: int = 20,
    n_batches: int = 1,
    unroll_steps: int = 20,
    beta: float = 0.0,
    max_norm: Optional[float] = None,
    deterministic_transition: bool = True,
    **kwargs,
) -> dict:
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
        action_target_dist = torch.distributions.normal.Normal(0.0, action_target_std)

    pbar = tqdm(range(n_batches), desc=f"{'updating policy network':30}")
    for i in pbar:
        loss = torch.zeros(1, device=device)
        action_loss = torch.zeros(1, device=device)
        reg_loss = torch.zeros(1, device=device)

        # sample a batch of episodes from memory
        state_batch, target_batch, action_batch, reward_batch, next_state_batch = sample_batch(
            batch_size, memory, warmup_steps, 0, device
        )

        # reset the models
        policy_model.zero_grad(set_to_none=True)
        transition_model.zero_grad(set_to_none=True)
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
            a_var = torch.exp(
                a_logvar
            )  # this is just for logging now and can probably be deleted later
            a_std = torch.exp(0.5 * a_logvar)
            a = a_mu + a_std * torch.randn_like(a_mu)

            logvars_mu.append(a_logvar.mean().item())
            logvars_std.append(a_logvar.std().item())
            vars_mu.append(a_var.mean().item())
            vars_std.append(a_var.std().item())

            s_hat_delta = transition_model.predict(
                new_state_hat, a, deterministic=deterministic_transition
            )
            s_hat = s_hat_delta + new_state_hat

            # TODO: the code below is an alternative approach directly based on the KL-divergence and seems to work
            # s_hat_delta_std = torch.exp(0.5 * s_hat_delta_logvar)
            # s_hat_dist = torch.distributions.normal.Normal(s_hat_mu, s_hat_delta_std)
            # target_loss = torch.mean(torch.distributions.kl_divergence(s_hat_dist, target_dist) * loss_gain)
            # action_loss = torch.distributions.kl_divergence(next_state_dist, target_dist).mean() \
            #              + alpha * torch.distributions.kl_divergence(action_dist, action_target_dist).mean()

            action_loss += torch.mean(torch.pow(t - s_hat, 2) * loss_gain)

            if beta:
                action_dist = torch.distributions.normal.Normal(a_mu, a_std)
                action_reg = (
                    torch.distributions.kl_divergence(action_dist, action_target_dist)
                    .mean()
                    .to(device)
                )
                r_loss = beta * action_reg
                reg_loss += r_loss

            new_state_hat = s_hat

        action_loss = action_loss / unroll_steps
        reg_loss = reg_loss / unroll_steps
        loss = action_loss + reg_loss
        action_losses.append(action_loss.item())
        reg_losses.append(reg_loss.item())
        losses.append(loss.item())
        loss.backward()
        grad_norms.append(gradnorm(policy_model))
        if max_norm:
            clip_grad_norm_(policy_model.parameters(), max_norm)
        clipped_grad_norms.append(gradnorm(policy_model))
        optimizer.step()

        pbar.set_postfix_str(f"loss: {loss.item()}")

    return {
        "policy model loss": np.mean(losses),
        "policy model action loss": np.mean(action_losses),
        "policy model reg loss": np.mean(reg_losses),
        "policy model grad norm": np.mean(grad_norms),
        "policy model clipped grad norm": np.mean(clipped_grad_norms),
        "policy model mean logvars mean": np.mean(logvars_mu),
        "policy model mean logvars std": np.mean(logvars_std),
        "policy model mean vars mean": np.mean(vars_mu),
        "policy model mean vars std": np.mean(vars_std),
    }


def train_policynetPB_sample2(
    policy_model: Module,
    transition_model: Module,
    memory: ReplayMemory,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    loss_gain: array,
    warmup_steps: int = 20,
    n_batches: int = 1,
    unroll_steps: int = 20,
    beta: float = 0.0,
    max_norm: Optional[float] = None,
    deterministic_policy: bool = True,
    record_policy: bool = True,
    deterministic_transition: bool = True,
    **kwargs,
) -> dict:
    """function used to update the parameters of a probabilistic policy network"""

    device = next(policy_model.parameters()).device

    losses = []
    loss_gain = torch.tensor(loss_gain).to(device)

    grad_norms = []
    clipped_grad_norms = []

    pbar = tqdm(range(n_batches), desc=f"{'updating policy network':30}")
    for i in pbar:

        loss = torch.zeros(1, device=device)

        # get a batch of episodes
        state_batch, target_batch, action_batch, reward_batch, next_state_batch = sample_batch(
            batch_size, memory, warmup_steps, 0, device
        )

        # reset the models
        policy_model.zero_grad(set_to_none=True)
        transition_model.zero_grad(set_to_none=True)
        policy_model.reset_state()
        transition_model.reset_state()

        # warm up both models
        _ = policy_model(state_batch[:-1], target_batch[:-1])
        _ = transition_model(state_batch[:-1], action_batch[:-1])

        new_state_hat = state_batch[-1]
        t = target_batch[-1]

        for k in range(unroll_steps):
            # sample an action from policy network
            a = policy_model.predict(
                new_state_hat,
                t,
                deterministic=deterministic_policy,
                record=record_policy,
            )

            s_hat_delta = transition_model.predict(
                new_state_hat, a, deterministic=deterministic_transition
            )
            s_hat = s_hat_delta + new_state_hat

            loss = torch.mean(torch.pow(t - s_hat, 2) * loss_gain)

            new_state_hat = s_hat

        loss = loss / unroll_steps
        losses.append(loss.item())
        loss.backward()
        grad_norms.append(gradnorm(policy_model))
        if max_norm:
            clip_grad_norm_(policy_model.parameters(), max_norm)
        clipped_grad_norms.append(gradnorm(policy_model))
        optimizer.step()

        pbar.set_postfix_str(f"loss: {loss.item()}")

    return {
        "policy model loss": np.mean(losses),
        "policy model grad norm": np.mean(grad_norms),
        "policy model clipped grad norm": np.mean(clipped_grad_norms),
    }


def train_transitionnetSNN(
    transition_model: Module,
    memory: ReplayMemory,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    warmup_steps: int = 10,
    n_batches: int = 1,
    unroll_steps: int = 10,
    max_norm: Optional[float] = None,
    exclude_monitors: list = [],
    **kwargs,
) -> dict:
    """
    Trains the transition model using the SNN (Stochastic Neural Network) method.

    Args:
        transition_model (Module): The transition model to be trained.
        memory (ReplayMemory): The replay memory containing the episodes.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        batch_size (int): The number of episodes to sample per batch.
        warmup_steps (int, optional): The number of warmup steps to perform before unrolling. Defaults to 10.
        n_batches (int, optional): The number of batches to train on. Defaults to 1.
        unroll_steps (int, optional): The number of steps to unroll the model for. Defaults to 10.
        max_norm (Optional[float], optional): The maximum norm for gradient clipping. Defaults to None.
        exclude_monitors (list, optional): A list of monitor names to exclude from the results. Defaults to [].

    Returns:
        dict: A dictionary containing the results of the training.
    """
    # set the device
    device = next(transition_model.parameters()).device

    # record some losses
    prediction_losses = []
    reg_losses = []
    losses = []
    grad_norms = []
    clipped_grad_norms = []

    pbar = tqdm(range(n_batches), desc=f"{'updating transition network':30}")
    for i in pbar:
        # initialize losses
        loss = torch.zeros(1, device=device)
        prediction_loss = torch.zeros(1, device=device)
        reg_loss = torch.zeros(1, device=device)

        # get a batch of episodes
        state_batch, target_batch, action_batch, reward_batch, next_state_batch = sample_batch(
            batch_size, memory, warmup_steps, unroll_steps, device
        )

        # reset the model
        transition_model.zero_grad(set_to_none=True)
        transition_model.reset_state()

        # time the forward pass
        #start = time.time()
        
        # warm up the model
        if warmup_steps > 0:
            _ = transition_model(state_batch[:warmup_steps], action_batch[:warmup_steps])

        state = state_batch[warmup_steps]

        for k in range(unroll_steps):
            action = action_batch[warmup_steps + k]
            next_state = next_state_batch[warmup_steps + k]

            # predict the next state
            next_state_delta_hat = transition_model.predict(state, action)
            next_state_hat = next_state_delta_hat + state

            # compute the prediction loss
            prediction_loss += torch.mean(torch.pow(next_state - next_state_hat, 2))

            # update the state
            state = next_state_hat

        #forward_time = time.time() - start
        #print(f"forward time: {forward_time}")

        # time the backward pass
        #start = time.time()        

        # compute the total loss, record, and backprop
        prediction_loss = prediction_loss / unroll_steps
        reg_loss = transition_model.get_reg_loss()
        loss = prediction_loss + reg_loss
        prediction_losses.append(prediction_loss.item())
        reg_losses.append(reg_loss.item())
        losses.append(loss.item())
        loss.backward()
        grad_norms.append(gradnorm(transition_model))
        if max_norm:
            clip_grad_norm_(transition_model.parameters(), max_norm)
        clipped_grad_norms.append(gradnorm(transition_model))
        optimizer.step()

        #backward_time = time.time() - start
        #print(f"backward time: {backward_time}")

        pbar.set_postfix_str(f"loss: {loss.item()}")

    results = {
        "transition model action loss": np.mean(prediction_losses),
        "transition model reg loss": np.mean(reg_losses),
        "transition model loss": np.mean(losses),
        "transition model grad norm": np.mean(grad_norms),
        "transition model clipped grad norm": np.mean(clipped_grad_norms),
    }

    results.update(transition_model.get_monitor_data(exclude=exclude_monitors))

    return results


def train_policynetSNN(
    policy_model: Module,
    transition_model: Module,
    memory: ReplayMemory,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    loss_gain: array,
    warmup_steps: int = 10,
    n_batches: int = 1,
    unroll_steps: int = 10,
    max_norm: Optional[float] = None,
    record_policy: bool = True,
    deterministic_transition: bool = True,
    exclude_monitors: list = [],
    **kwargs,
) -> dict:
    """
    Trains a policy network using the stochastic neural network (SNN) algorithm.

    Args:
        policy_model (Module): The policy network to train.
        transition_model (Module): The transition network to use for predicting future states.
        memory (ReplayMemory): The replay memory to sample episodes from.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        batch_size (int): The number of episodes to sample per batch.
        loss_gain (array): The weightings to apply to each dimension of the loss function.
        warmup_steps (int, optional): The number of steps to warm up the models before training. Defaults to 10.
        n_batches (int, optional): The number of batches to train on. Defaults to 1.
        unroll_steps (int, optional): The number of steps to unroll the transition model for. Defaults to 10.
        max_norm (Optional[float], optional): The maximum gradient norm to clip to. Defaults to None.
        record_policy (bool, optional): Whether to record the policy network's actions during training. Defaults to True.
        deterministic_transition (bool, optional): Whether to use deterministic predictions from the transition model. Defaults to True.
        exclude_monitors (list, optional): A list of monitor names to exclude from the results. Defaults to [].

    Returns:
        dict: A dictionary containing the results of the training.
    """

    device = next(policy_model.parameters()).device

    action_losses = []
    reg_losses = []
    losses = []
    loss_gain = torch.tensor(loss_gain).to(device)

    grad_norms = []
    clipped_grad_norms = []

    pbar = tqdm(range(n_batches), desc=f"{'updating policy network':30}")
    for i in pbar:
        # initialize losses
        loss = torch.zeros(1, device=device)
        action_loss = torch.zeros(1, device=device)

        # get a batch of episodes
        state_batch, target_batch, action_batch, reward_batch, next_state_batch = sample_batch(
            batch_size, memory, warmup_steps, 0, device
        )

        # reset the models
        policy_model.zero_grad(set_to_none=True)
        transition_model.zero_grad(set_to_none=True)
        policy_model.reset_state()
        transition_model.reset_state()

        # time the forward pass
        #start = time.time()

        # warm up both models
        _ = policy_model(state_batch[:-1], target_batch[:-1], record=record_policy)
        _ = transition_model(state_batch[:-1], action_batch[:-1])

        new_state_hat = state_batch[-1]
        t = target_batch[-1]

        for k in range(unroll_steps):
            # sample an action from policy network
            a = policy_model.predict(new_state_hat, t, record=record_policy)

            s_hat_delta = transition_model.predict(
                new_state_hat, a, deterministic=deterministic_transition
            )
            s_hat = s_hat_delta + new_state_hat

            action_loss += torch.mean(torch.pow(t - s_hat, 2) * loss_gain)

            new_state_hat = s_hat

        #forward_time = time.time() - start
        #print(f"forward time: {forward_time}")

        # time backward pass
        #start = time.time()

        action_loss = action_loss / unroll_steps
        reg_loss = policy_model.get_reg_loss()
        loss = action_loss + reg_loss
        action_losses.append(action_loss.item())
        reg_losses.append(reg_loss.item())
        losses.append(loss.item())
        loss.backward()
        grad_norms.append(gradnorm(policy_model))
        if max_norm:
            clip_grad_norm_(policy_model.parameters(), max_norm)
        clipped_grad_norms.append(gradnorm(policy_model))
        optimizer.step()

        #backward_time = time.time() - start
        #print(f"backward time: {backward_time}")

        pbar.set_postfix_str(f"loss: {loss.item()}")

    results = {
        "policy model action loss": np.mean(action_losses),
        "policy model reg loss": np.mean(reg_losses),
        "policy model loss": np.mean(losses),
        "policy model grad norm": np.mean(grad_norms),
        "policy model clipped grad norm": np.mean(clipped_grad_norms),
    }

    results.update(policy_model.get_monitor_data(exclude=exclude_monitors))

    return results


@torch.no_grad()
def baseline_prediction(transitionnet: Module, episode: list, warmup: int = 0) -> dict:
    """function used to evaluate transition network predictions against baseline values"""

    transitionnet.reset_state()
    T = len(episode)

    states = torch.stack([step[0] for step in episode]).view((T, 1, -1))
    actions = torch.stack([step[2] for step in episode]).view((T, 1, -1))
    next_states = torch.stack([step[4] for step in episode]).view((T, 1, -1))

    # predict next state based on action with transition net
    delta_states = transitionnet.predict(states, actions, deterministic=True)
    predicted_states = states + delta_states.detach()
    predicted_state_mse = torch.pow(
        predicted_states[warmup:] - next_states[warmup:], 2
    ).mean()
    # print("predicted state MSE:", predicted_state_mse.item())

    # use linear extrapolation as a baseline estimate
    previous_deltas = states[1 + warmup :] - states[warmup:-1]
    extrapolated_states = states[1 + warmup :] + previous_deltas
    extrapolated_state_mse = torch.pow(
        extrapolated_states - next_states[1 + warmup :], 2
    ).mean()
    # print("extrapolated state MSE:", extrapolated_state_mse.item())

    # use current state as a second baseline estimate
    current_state_mse = torch.pow(states[warmup:] - next_states[warmup:], 2).mean()
    # print("current state MSE:", current_state_mse.item())

    return {
        "predicted_state_MSE": predicted_state_mse.item(),
        "extrapolated_state_MSE": extrapolated_state_mse.item(),
        "current_state_MSE": current_state_mse.item(),
    }


def sample_batch(
    batch_size: int,
    memory: ReplayMemory,
    warmup_steps: int,
    unroll_steps: int,
    device: torch.device,
) -> list[Tensor]:
    """sample a batch of episodes from memory and return a batch of state, target, action, reward, next_state tensors
        of shape [warmup_steps + unroll_steps, batch_size, dim]

    Args:
        batch_size (int): number of episodes to sample
        memory (ReplayMemory): replay memory to sample from
        warmup_steps (int): number of steps to warm up the models
        unroll_steps (int): number of steps to unroll the models
        device (torch.device): device to store tensors on

    Returns:
        list[Tensor]: list of tensors of shape [warmup_steps + unroll_steps, batch_size, dim]
    """
    steps = warmup_steps + unroll_steps
    assert 0 < steps
    assert 0 < memory.size
    episodes = []

    # TODO: need to make sure that episodes are not too short

    # sample episodes until batch is full
    while len(episodes) < batch_size:
        sample_batch = memory.sample(
            batch_size - len(episodes)
        )  # [sample, step, (state, target, action, reward, next_state)]
        for episode in sample_batch:
            if len(episode) >= steps:
                episodes.append(episode)
                if len(episodes) == batch_size:
                    break

    # make a random sample from each episode of length = warmup_steps
    state_dim, target_dim, action_dim, reward_dim, next_state_dim = (
        episodes[0][0][0].size(-1),
        episodes[0][0][1].size(-1),
        episodes[0][0][2].size(-1),
        1,
        episodes[0][0][4].size(-1),
    )

    # initialize tensors
    state_batch = torch.zeros((steps, batch_size, state_dim), device=device)
    target_batch = torch.zeros((steps, batch_size, target_dim), device=device)
    action_batch = torch.zeros((steps, batch_size, action_dim), device=device)
    reward_batch = torch.zeros((steps, batch_size, reward_dim), device=device)
    next_state_batch = torch.zeros((steps, batch_size, next_state_dim), device=device)

    # fill tensors with random samples from episodes
    for j, episode in enumerate(episodes):
        r = torch.randint(low=0, high=len(episode) - steps, size=(1,))
        state_batch[:, j, :] = torch.stack(
            [step[0].squeeze() for step in episode[r : r + steps]]
        )
        target_batch[:, j, :] = torch.stack(
            [step[1].squeeze() for step in episode[r : r + steps]]
        )
        action_batch[:, j, :] = torch.stack(
            [step[2].squeeze() for step in episode[r : r + steps]]
        )
        reward_batch[:, j, :] = torch.stack(
            [torch.tensor(step[3]).unsqueeze_(-1) for step in episode[r : r + steps]]
        )
        next_state_batch[:, j, :] = torch.stack(
            [step[4].squeeze() for step in episode[r : r + steps]]
        )

    return [state_batch, target_batch, action_batch, reward_batch, next_state_batch]
