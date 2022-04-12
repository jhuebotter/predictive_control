from typing import Union

import torch
import numpy as np
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from extratyping import *


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
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
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