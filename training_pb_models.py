import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from utils import ReplayMemory, make_env, make_transition_model, make_policy_model, make_optimizer, save_checkpoint, dict_mean, \
    reparameterize as rp
from training_functions import train_policynetPB, train_transitionnetRNNPBNLL, baseline_prediction
from plotting import render_video, animate_predictions
from config import get_config, save_config
from tqdm import tqdm
import wandb

torch.autograd.set_detect_anomaly(True)

# read some parameters from a config file
config = get_config()
seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)

# chose device to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

# pick the task to be learned
env = make_env(config)
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
loss_gain = env.loss_gain

# initialize a replay memory
memory = ReplayMemory(config['memory_size'])

# initialize the models
transitionnet = make_transition_model(env, config['transition']['model']).to(device)
policynet = make_policy_model(env, config['policy']['model']).to(device)

# initialize the optimizers
opt_trans = make_optimizer(transitionnet, config['transition']['optim'])
opt_policy = make_optimizer(policynet, config['policy']['optim'])

# make directory to save model and plots
run_id = datetime.now().strftime('%Y%m%d%H%M%S')
run_dir = Path('results', config['experiment'], config['task'],
               f"tra{config['transition']['model']['type']}_pol{config['policy']['model']['type']}", run_id)
run_dir.mkdir(parents=True)
wandb.init(config=config, project=config['project'], entity=config['entity'], dir='./results')
print(wandb.run.dir)
wandb.watch([transitionnet, policynet], log='all')
# TODO: only need to make one directory - should probably use the one made by wandb
# TODO: make wandb optional with local logging via pandas and plotting via matplotlib - later

# save the run configuration in the result dir
config_path = Path(run_dir, 'config.yaml')
save_config(config, config_path)

step = 1
episode_count = 1
transitionnet_updates = 0
policynet_updates = 0
iteration = 1

best_transition_loss = np.inf
best_policy_loss = np.inf

have_done_update = False

while step <= config['total_env_steps']:
    # record a bunch of episodes to memory

    # TODO: THIS IS ONLY A TEMPORARY HARDCODED TEST BECAUSE I AM LAZY!
    if step > config['total_env_steps'] / 2 and not have_done_update:
        for g in opt_trans.param_groups:
            g['lr'] = config['transition']['optim']['lr'] / np.sqrt(config['batch_size'])
        for g in opt_policy.param_groups:
            g['lr'] = config['policy']['optim']['lr'] / np.sqrt(config['batch_size'])
        config['batch_size'] = 1
        have_done_update = True

    print()
    print(f'beginning iteration {iteration}')

    frames = []
    iteration_results = {}
    baseline_predictions = []

    for e in tqdm(range(1, config['episodes_per_iteration'] + 1), desc=f"{'obtaining experience':30}"):
        if e < config['record_first_n_episodes'] + 1 and (iteration % config['record_every_n_iterations'] == 0 or iteration == 1):
            render_mode = 'rgb_array'
        else:
            render_mode = None

        # reset the environment
        observation, target = env.reset()
        observation = torch.tensor(observation, device=device, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(target, device=device, dtype=torch.float32).unsqueeze(0)

        # reset the network states
        policynet.reset_state()
        transitionnet.reset_state()

        episode = []
        done = False
        while not done:

            # chose action and advance simulation
            action = rp(*policynet(observation, target))
            if len(action.shape) == 3:
                action.squeeze_(0)
            a = action[0].detach().cpu().numpy().clip(env.action_space.low, env.action_space.high)
            next_observation, target, done, info = env.step(a)
            next_observation = torch.tensor(next_observation, device=device, dtype=torch.float32).unsqueeze(0)
            target = torch.tensor(target, device=device, dtype=torch.float32).unsqueeze(0)

            # save transition for later
            transition = (observation, target, action.detach(), next_observation)
            episode.append(transition)

            if render_mode:
                pixels = env.render(mode=render_mode)
                frames.append(pixels)

            # environment step complete
            observation = next_observation
            step += 1

        # save episode to memory
        memory.append(episode)
        episode_count += 1

        # compute prediction performance against baseline
        baseline_predictions.append(baseline_prediction(transitionnet, episode))

    baseline_results = dict_mean(baseline_predictions)

    # render video and save to disk
    if len(frames):
        render_video(frames, env.metadata['render_fps'], save=Path(run_dir, f'episode_animation_{iteration}.mp4'))
        animate_predictions(episode, transitionnet, env.state_labels, save=Path(run_dir, f'prediction_animation_{iteration}.mp4'))
        wandb.log({f'episode animation': wandb.Video(str(Path(run_dir, f'episode_animation_{iteration}.mp4'))),
                   f'prediction animation': wandb.Video(str(Path(run_dir, f'prediction_animation_{iteration}.mp4')))},
                  step=iteration)

    # update transition and policy models based on data in memory
    transition_results = train_transitionnetRNNPBNLL(transitionnet, memory, opt_trans, config['batch_size'],
                                                     config['warmup'], config['updates_per_iteration'])
    transitionnet_updates += config['updates_per_iteration']

    policy_results = train_policynetPB(policynet, transitionnet, memory, opt_policy, config['batch_size'],
                                       env.loss_gain, config['warmup'], config['updates_per_iteration'],
                                       config['unroll_steps'], config['beta'])
    policynet_updates += config['updates_per_iteration']

    # save the model parameters
    save_checkpoint(transitionnet, opt_trans, path=Path(run_dir, 'transitionnet_latest.cpt'))
    if transition_results['transition model loss'] < best_transition_loss:
        best_transition_loss = transition_results['transition model loss']
        save_checkpoint(transitionnet, opt_trans, path=Path(run_dir, 'transitionnet_best.cpt'))
    save_checkpoint(policynet, opt_policy, path=Path(run_dir, 'policynet_latest.cpt'))
    if policy_results['policy model loss'] < best_policy_loss:
        best_policy_loss = policy_results['policy model loss']
        save_checkpoint(policynet, opt_trans, path=Path(run_dir, 'policynet_best.cpt'))

    # log the iteration results
    data = {'environment step': step, 'episode': episode_count, 'iteration': iteration,
            'policynet updates': policynet_updates, 'transitionnet updates': transitionnet_updates}
    iteration_results = dict(**transition_results, **policy_results, **baseline_results, **data)
    print()
    for k, v in iteration_results.items():
        print(f'{k:30}: {v:.3e}')
    wandb.log(iteration_results, step=iteration)

    # iteration complete
    iteration += 1
