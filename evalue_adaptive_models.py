from tqdm import tqdm
import torch
from src.utils import make_env, dict_mean
from src.training_functions import baseline_prediction
from src.plotting import animate_predictions, render_video, plot_trajectories
from src.extratyping import *
import wandb
import numpy as np


def rotate(vec, deg):
    rad = deg * np.pi / 180
    r = np.array([[np.cos(rad), -np.sin(rad)],
                  [np.sin(rad), np.cos(rad)]]).T
    return vec @ r


def make_eval_tasks(task_config: dict) -> (list, list, object):

    states = []
    targets = []
    env = None

    if task_config['type'] == 'plane':
        env = make_env(task_config)
        env.random_target = False
        env.target_angle = 180.0

        states = np.zeros((8, 4))
        targets = states.copy()

        p = np.array([0.7, 0.0])

        for i in range(len(targets)):
            targets[i, :2] = rotate(p, 45*i)

        if not task_config['params']['moving_target'] == 0.0:
            r = np.array([0., 0.5])
            for i in range(len(targets)):
                targets[i, 2:4] = rotate(r, 45*i)

    elif task_config['type'] == 'plane2':
        env = make_env(task_config)
        env.random_target = False
        env.target_angle = 180.0

        states = np.zeros((8, 6))
        targets = states.copy()

        p = np.array([0.7, 0.0])

        for i in range(len(targets)):
            targets[i, :2] = rotate(p, 45 * i)

        if not task_config['params']['moving_target'] == 0.0:
            r = np.array([0., 0.5])
            for i in range(len(targets)):
                targets[i, 2:4] = rotate(r, 45 * i)

    elif task_config['type'] == 'reacher2':
        env = make_env(task_config)
        env.random_target = False

        states = np.zeros((8, 4))
        targets = states.copy()

        for i in range(len(targets)):
            targets[i, :2] = 2 * np.pi * i / 8
            if i % 2 == 1:
                pass

        if not task_config['params']['moving_target'] == 0.0:
            for i in range(len(targets)):
                targets[i, 2:4] = np.array([0.0, 0.5]) * env.max_vel

    else:
        raise NotImplementedError('eval task for environment type not specified')
    # TODO: add more tasks here

    return states, targets, env


def evalue_adaptive_models(policynet: Module, transitionnet: Module, task_config: dict, record: bool = True,
                           device: str = 'cpu', step: int = 0, run_dir: str = 'results') -> dict:

    frames = []
    baseline_predictions = []
    rewards = []
    episodes = []

    states, targets, env = make_eval_tasks(task_config)

    for e in tqdm(range(1, len(states) + 1), desc=f"{'evaluating models':30}"):
        if record:
            render_mode = 'rgb_array'
        else:
            render_mode = None

        # reset the environment
        observation, target = env.reset(state=states[e-1], target=targets[e-1])
        observation = torch.tensor(observation, device=device, dtype=torch.float32)
        target = torch.tensor(target, device=device, dtype=torch.float32)

        # reset the network states
        policynet.reset_state()
        transitionnet.reset_state()

        episode = []
        total_reward = 0.
        done = False
        while not done:

            # chose action and advance simulation
            action = policynet.predict(observation.view((1, 1, -1)), target.view((1, 1, -1)), deterministic=True)
            a = action.flatten().detach().cpu().numpy().clip(env.action_space.low, env.action_space.high)
            next_observation, next_target, reward, done, info = env.step(a)
            next_observation = torch.tensor(next_observation, device=device, dtype=torch.float32)
            next_target = torch.tensor(next_target, device=device, dtype=torch.float32)

            # save transition for later
            transition = (observation.clone(), target.clone(), action.detach().clone(), reward, next_observation.clone())
            episode.append(transition)

            if render_mode:
                pixels = env.render(mode=render_mode)
                frames.append(pixels)

            #env.render('human')

            # environment step complete
            observation = next_observation
            target = next_target
            total_reward += reward

        rewards.append({'mean episode reward eval': total_reward})
        # compute prediction performance against baseline
        # TODO: NEEDS WARMUP WINDOW PASSED!
        baseline_predictions.append(baseline_prediction(transitionnet, episode))
        episodes.append(episode)

    baseline_results = dict_mean(baseline_predictions)
    rewards = dict_mean(rewards)

    results = {**rewards}
    for k, v in baseline_results.items():
        results[f'{k} eval'] = v

    # render video and save to disk
    if len(frames) and record:
        plot_trajectories(episodes, save=Path(run_dir, f'episode_trajectories_eval_{step}.png'))
        render_video(frames, env.metadata['render_fps'], save=Path(run_dir, f'episode_animation_eval_{step}.mp4'))
        animate_predictions(episode, transitionnet, env.state_labels,
                            save=Path(run_dir, f'prediction_animation_eval_{step}_{e}.mp4'))
        results.update({
            f'episode trajectories eval': wandb.Image(str(Path(run_dir, f'episode_trajectories_eval_{step}.png'))),
            f'episode animation eval': wandb.Video(str(Path(run_dir, f'episode_animation_eval_{step}.mp4'))),
            f'prediction animation eval': wandb.Video(str(Path(run_dir, f'prediction_animation_eval_{step}_{e}.mp4')))
        })

    return results






