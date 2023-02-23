from typing import Union
import torch
import numpy as np
import matplotlib as mlp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from celluloid import Camera
from .utils import reparameterize as rp
from .extratyping import *

plt.rcParams['font.size'] = '12'


def plot_trajectories(episodes: list, save: Optional[Union[Path, str]] = '', show: bool = False, skip: int = 2):

    fig = plt.figure(figsize=(3, 3))
    plt.xlim(-1.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.axis('off')

    for e in episodes:
        states = [s[0].squeeze() for s in e]
        xs = [x[0] for x in states[::skip]]
        ys = [x[1] for x in states[::skip]]

        if isinstance(xs[0], torch.Tensor):
            xs = [x.cpu().numpy() for x in xs]
            ys = [y.cpu().numpy() for y in ys]

        c = np.linspace(0.0, 1.0, len(xs))
        plt.scatter(xs, ys, s=0.5, c=c, cmap=mlp.cm.get_cmap('viridis'))

    plt.tight_layout()

    if save:
        plt.savefig(save)
    if show:
        plt.show()
    plt.close()

    return


def plot_curves(data=dict[str: list[array]], logy: bool = True, show: bool = False,
                save: Optional[Union[Path, str]] = 'curve.png') -> None:
    """create a plot from a dictionary of lists and save to disk"""

    fig, ax = plt.subplots()
    for k, v in data.items():
        x = np.arange(1, len(v)+1)
        ax.plot(x, v, label=k)
    if logy:
        plt.yscale('log')
    plt.xlabel("episode")
    plt.ylabel("value")
    plt.legend()
    if save:
        plt.savefig(save)
    if show:
        plt.show()
    plt.close()


def render_video(frames: list[array], framerate: int = 30, dpi: int = 70,
                 save: Optional[Union[Path, str]] = './animation.mp4') -> object:
    """render a video from a list of numpy RGB arrays and save to disk"""

    assert len(frames) > 0
    height, width, _ = frames[0].shape
    orig_backend = mlp.get_backend()
    mlp.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    mlp.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
        im.set_data(frame)
        return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    if save:
        anim.save(save)

    return anim.to_html5_video()


def create_episode(env, transitionnet, policynet, steps: int = 100):

    observation, target = env.reset()
    device = transitionnet.device
    observation = torch.tensor(observation, device=device, dtype=torch.float32).unsqueeze(0)
    target = torch.tensor(target, device=device, dtype=torch.float32).unsqueeze(0)

    transitionnet.reset_state()
    policynet.reset_state()
    episode = []

    for i in range(steps):
        # get action from policy
        action = rp(*policynet(observation, target))

        # take action and get next observation
        if len(action.shape) == 3:
            action.squeeze_(0)
        a = action[0].detach().cpu().numpy().clip(env.action_space.low, env.action_space.high)
        next_observation, target, done, info = env.step(a)
        next_observation = torch.tensor(next_observation, device=device, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(target, device=device, dtype=torch.float32).unsqueeze(0)

        # save transition for later
        transition = (observation[0], target, action.detach()[0], next_observation[0])
        episode.append(transition)

        # advance to next step
        observation = next_observation

    env.close()

    return episode


def make_predictions(episode: list, transitionnet: Module, h: int = 100, deterministic: bool = True) -> Tensor:\

    n = len(episode)
    observations = torch.stack([step[0] for step in episode]).unsqueeze(0).transpose(0, 1)
    actions = torch.stack([step[2] for step in episode]).unsqueeze(0).transpose(0, 1)
    d = observations.shape[-1]
    predictions = torch.zeros((n, h, d))

    try:
        transitionnet.state_initialized = False
    except:
        pass
    #hidden_state = None
    for i in range(n):
        current_observation = observations[i]
        #transitionnet.update_state(hidden_state)
        mu_pred, logvar_pred = transitionnet(observations[i], actions[i])
        #hidden_state = transitionnet.get_state()
        if deterministic:
            delta_pred = mu_pred
        else:
            delta_pred = rp(mu_pred, logvar_pred)
        state_pred = current_observation + delta_pred

        predictions[i, 0] = state_pred
        for j in range(1, h):
            if i+j >= n:
                break
            mu_pred, logvar_pred = transitionnet(state_pred, actions[i+j])
            if deterministic:
                delta_pred = mu_pred
            else:
                delta_pred = rp(mu_pred, logvar_pred)
            state_pred = state_pred + delta_pred

            predictions[i, j] = state_pred

    return predictions


def animate_predictions(episode: list, transitionnet: Module, labels: list, h: int = 100, fps: float = 20.,
                        save: Optional[Union[Path, str]] = './animation.mp4', deterministic: bool = True, dpi: int = 50,
                        font_size: int = 12) -> object:
    plt.rcParams['font.size'] = f'{font_size}'

    predictions = make_predictions(episode, transitionnet, h, deterministic=deterministic)
    predictions = predictions.detach().cpu().numpy()
    next_observations = [step[4].squeeze().cpu().numpy() for step in episode]

    n, h, d = predictions.shape

    fig, ax = plt.subplots(d, figsize=(5, d), sharex=True, sharey=True, dpi=dpi)
    plt.ylim(-1.1, 1.1)

    cmap = plt.get_cmap('plasma')

    camera = Camera(fig)

    # make an initial snapshot without prediction
    for i in range(d):
        ax[i].plot([o[i] for o in next_observations], c='g', alpha=0.5)
        ax[i].set_ylabel(labels[i])
    ax[-1].set_xlabel('step')
    plt.tight_layout()

    camera.snap()

    # animate the prediction
    for t in np.arange(0, n):
        for i in range(d):
            max_ = np.min([h, n - t])
            idx = np.arange(0., 1., 1. / (h))
            ax[i].scatter(np.arange(t, np.min([t + h, n])), predictions[t, :max_, i], c=cmap(idx[:max_]), s=4)
            ax[i].plot([o[i] for o in next_observations], c='g', alpha=0.5)

        plt.tight_layout()
        camera.snap()

    animation = camera.animate(interval=1000. / fps, blit=True)
    plt.close()

    if save:
        animation.save(save) #, bitrate=-1)

    return animation


# deprecated:
def make_arrowplot(model, target=None):
    """"""
    # the use case of this plot is very limited and can likely be removed soon
    model.eval()

    device = next(model.parameters()).device

    X, Y = np.meshgrid(np.arange(-0.9, 0.91, .1), np.arange(-0.9, 0.91, .1))
    U, V = np.zeros(X.shape), np.zeros(X.shape)

    if target is None:
        target = np.zeros(4)
        target = torch.tensor(target, device=device, dtype=torch.float32).unsqueeze(0)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            state = np.ones(4)
            state[:2] = X[i, j], Y[i, j]
            state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)

            action = model(state, target)
            U[i, j], V[i, j] = action[0].detach().numpy()

    plt.figure(figsize=(4, 4))
    plt.scatter(target[0, 0], target[0, 1])
    plt.quiver(X, Y, U, V, units='width')
    plt.axis("off")
    plt.tight_layout()
    plt.show()