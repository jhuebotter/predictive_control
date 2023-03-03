import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from src.utils import ReplayMemory, make_env, make_transition_model, make_policy_model, make_optimizer, save_checkpoint, \
    dict_mean, load_weights_from_disk
from src.training_functions import train_policynetPB, train_policynetPB_sample, train_transitionnetRNNPBNLL, baseline_prediction
from src.plotting import render_video, animate_predictions
from src.config import get_config, save_config
from src.logger import PandasLogger
from tqdm import tqdm
import argparse
import wandb
from evalue_adaptive_models import evalue_adaptive_models

torch.autograd.set_detect_anomaly(True)

# read the directory to be loaded
parser = argparse.ArgumentParser()
parser.add_argument('--load_dir', help='directory with config file and model parameters', type=str, default='')
args, left_argv = parser.parse_known_args()

# read some parameters from a config file
if args.load_dir:
    # config = get_config(Path(args.load_dir, 'config.yaml'))
    config = get_config(Path(args.load_dir, 'config_snn.yaml'))
else:
    #config = get_config()
    config = get_config('config_snn.yaml')

seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)

# chose device to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

# pick the task to be learned
env = make_env(config['task'], seed=seed)
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
loss_gain = env.loss_gain

# initialize a replay memory
memory = ReplayMemory(config['memory_size'])

# initialize the models
transitionnet = make_transition_model(env, config['transition']['model']).to(device)
policynet = make_policy_model(env, config['policy']['model']).to(device)

# initialize the optimizers
opt_trans = make_optimizer(transitionnet.basis, config['transition']['optim'])
opt_policy = make_optimizer(policynet.basis, config['policy']['optim'])

# load model and other things from checkpoint
if args.load_dir:
    print("loading weights from", args.load_dir)
    transitionnet, opt_trans = load_weights_from_disk(transitionnet, Path(args.load_dir, 'transitionnet_latest.cpt'), opt_trans, device)
    for pg in opt_trans.param_groups:
        pg.update(config['transition']['optim']['params'])
    policynet, opt_policy = load_weights_from_disk(policynet, Path(args.load_dir, 'policynet_latest.cpt'), opt_policy, device)
    for pg in opt_policy.param_groups:
        pg.update(config['policy']['optim']['params'])

# make directory to save model and plots
run_id = datetime.now().strftime('%Y%m%d%H%M%S')
run_dir = Path('results', config['experiment'], config['task']['type'],
               f"tra{config['transition']['model']['type']}_pol{config['policy']['model']['type']}", run_id)
run_dir.mkdir(parents=True)
wandb.init(config=config, project=config['project'], entity=config['entity'], dir='./results')
print(wandb.run.dir)
wandb.watch([transitionnet, policynet], log='all')
# TODO: only need to make one directory - should probably use the one made by wandb
# TODO: make wandb optional with local logging via pandas and plotting via matplotlib - later

# create a logger that will save results to .csv
logger = PandasLogger(name=wandb.run.id, dir=Path('results', config['experiment']))

# save the run configuration in the result dir
#config_path = Path(run_dir, 'config.yaml')
config_path = Path(run_dir, 'config_snn.yaml')
save_config(config, config_path)

step = 1
episode_count = 1
transitionnet_updates = 0
policynet_updates = 0
iteration = 1

best_transition_loss = np.inf
best_policy_loss = np.inf

while step <= config['total_env_steps']:
    # record a bunch of episodes to memory

    print()
    print(f'beginning iteration {iteration}')

    frames = []
    baseline_predictions = []
    rewards = []

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
        total_reward = 0.
        done = False
        while not done:

            # chose action and advance simulation
            action = policynet.predict(observation, target)
            #if len(action.shape) == 3:
            #    action.squeeze_(0)
            a = action[0, 0].detach().cpu().numpy().clip(env.action_space.low, env.action_space.high)
            next_observation, next_target, reward, done, info = env.step(a)
            next_observation = torch.tensor(next_observation, device=device, dtype=torch.float32).unsqueeze_(0)
            next_target = torch.tensor(next_target, device=device, dtype=torch.float32).unsqueeze_(0)

            # save transition for later
            transition = (observation.clone(), target.clone(), action.detach().clone(), reward, next_observation.clone())
            episode.append(transition)

            if render_mode:
                pixels = env.render(mode=render_mode)
                frames.append(pixels)

            # environment step complete
            observation = next_observation
            target = next_target
            total_reward += reward
            step += 1

        # save episode to memory
        memory.append(episode)
        rewards.append({'mean episode reward training': total_reward})
        episode_count += 1

        # compute prediction performance against baseline
        baseline_predictions.append(baseline_prediction(transitionnet, episode))

    baseline_results = dict_mean(baseline_predictions)
    rewards = dict_mean(rewards)

    # render video and save to disk
    if len(frames) and config['animate']:
        render_video(frames, env.metadata['render_fps'], save=Path(run_dir, f'episode_animation_{iteration}.mp4'))
        animate_predictions(episode, transitionnet, env.state_labels, save=Path(run_dir, f'prediction_animation_{iteration}_{e}.mp4'))
        wandb.log({f'episode animation': wandb.Video(str(Path(run_dir, f'episode_animation_{iteration}.mp4'))),
                   f'prediction animation': wandb.Video(str(Path(run_dir, f'prediction_animation_{iteration}_{e}.mp4')))},
                  step=iteration)

    # update transition and policy models based on data in memory
    transition_results = train_transitionnetRNNPBNLL(transitionnet, memory, opt_trans, **config['transition']['learning']['params'])
    transitionnet_updates += config['transition']['learning']['params']['n_batches']

    policy_results = train_policynetPB_sample(policynet, transitionnet, memory, opt_policy,
                                       loss_gain=env.loss_gain, **config['policy']['learning']['params'])
    policynet_updates += config['policy']['learning']['params']['n_batches']

    # log the iteration results
    data = {'environment step': step, 'episode': episode_count, 'iteration': iteration,
            'policynet updates': policynet_updates, 'transitionnet updates': transitionnet_updates}
    iteration_results = dict(**transition_results, **policy_results, **baseline_results, **data, **rewards)

    # evaluate if necessary
    if config['evaluate']:
        record = False
        if (iteration % config['record_every_n_iterations'] == 0 or iteration == 1) and config['animate']:
            record = True
        eval_results = evalue_adaptive_models(policynet, transitionnet, config['task'], record=record,
                                              device=device, step=iteration, run_dir=run_dir)
        iteration_results.update(eval_results)

    # save stuff to .csv
    summary = dict(**iteration_results, **config)
    logger.save_summary(summary)

    print()
    for k, v in iteration_results.items():
        try:
            print(f'{k:30}: {v:.3e}')
        except:
            continue
    wandb.log(iteration_results, step=iteration)

    # save the model parameters
    save_checkpoint(transitionnet, opt_trans, path=Path(run_dir, 'transitionnet_latest.cpt'), **iteration_results)
    if transition_results['transition model loss'] < best_transition_loss:
        best_transition_loss = transition_results['transition model loss']
        save_checkpoint(transitionnet, opt_trans, path=Path(run_dir, 'transitionnet_best.cpt'), **iteration_results)
    save_checkpoint(policynet, opt_policy, path=Path(run_dir, 'policynet_latest.cpt'), **iteration_results)
    if policy_results['policy model loss'] < best_policy_loss:
        best_policy_loss = policy_results['policy model loss']
        save_checkpoint(policynet, opt_trans, path=Path(run_dir, 'policynet_best.cpt'), **iteration_results)

    # iteration complete
    iteration += 1

env.close()