import numpy as np
from dm_control import mujoco
from dm_control import suite
import gym
from gym import spaces
import matplotlib.pyplot as plt

class ReacherEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 50
    }

    def __init__(self, seed: int = None, max_episode_steps: int = 200, rl_mode: bool = False, **kwargs):

        random_state = np.random.RandomState(seed)
        self.env = suite.load('reacher', 'hard', task_kwargs={'random': random_state})
        self.max_episode_steps = max_episode_steps
        self.observation_noise_std = np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])

        self.dt = 0.02
        self.rl_mode = rl_mode

        self.action_space = spaces.Box(
            low=self.env.action_spec().minimum,
            high=self.env.action_spec().maximum,
            shape=self.env.action_spec().shape
        )

        self.max_pos = 1.0
        self.min_pos = -self.max_pos

        self.max_vel = 1.0
        self.min_vel = -self.max_vel

        self.max_angle = 1.0
        self.min_angle = - self.max_angle

        self.observation_space = spaces.Box(
            low=np.array([self.min_pos, self.min_pos, self.min_angle, self.min_angle, self.min_angle, self.min_angle,
                          self.min_vel, self.min_vel]),
            high=np.array([self.max_pos, self.max_pos, self.max_angle, self.max_angle, self.max_angle, self.max_angle,
                           self.max_vel, self.max_vel])
        )

        self.loss_gain = np.array([1., 1., 0., 0., 0., 0., 0., 0.])

        self.img = None
        self.ctx = None

        self.state_labels = ['hand x', 'hand y', 'cos a1', 'cos a2', 'sin a1', 'sin a2', 'vel a1', 'vel a2']

        self.done_on_target = False
        self.epsilon = 0.01


    def step(self, action):

        self.episode_step_count += 1
        timestep = self.env.step(action)

        targetxy = self.get_target_pos()
        handxy = self.get_hand_pos()
        jointsradpos = timestep.observation['position']
        jointsradvel = timestep.observation['velocity']

        self.state = np.hstack([handxy, np.sin(jointsradpos), np.cos(jointsradpos), jointsradvel / 6.])

        self.target = np.zeros(self.state.shape)
        self.target[:2] = targetxy
        self.target[2:] = self.state[2:]  # TODO: THIS IS SUPER WEIRD...

        done = False
        on_target = False
        # if np.allclose(self.state[~np.isnan(self.target)], self.target[~np.isnan(self.target)], atol=self.epsilon):
        if np.allclose(self.state[:4], self.target[:4], atol=self.epsilon):
            on_target = True
            if self.done_on_target:
                # self.reset(state=self.state)
                done = True

        max_steps = False
        if self.max_episode_steps and self.episode_step_count == self.max_episode_steps:
            done = True
            max_steps = True

        observation = np.random.normal(self.state, self.observation_noise_std)

        reward = - np.linalg.norm(self.target[:2] - self.state[:2]) * self.dt

        info = {'on_target': on_target, 'max_steps': max_steps}

        if not self.rl_mode:
            return observation, self.target, reward, done, info

        else:
            s = np.hstack(observation, self.target)
            return s, reward, done, info

    def reset(self, state=None, target=None):

        self.episode_step_count = 0

        timestep = self.env.reset()

        targetxy = self.get_target_pos()
        handxy = self.get_hand_pos()
        jointsradpos = timestep.observation['position']
        jointsradvel = timestep.observation['velocity']

        self.state = np.hstack([handxy, np.sin(jointsradpos), np.cos(jointsradpos), jointsradvel / 6.])

        self.target = np.zeros(self.state.shape)
        self.target[:2] = targetxy
        self.target[2:] = self.state[2:]

        observation = np.random.normal(self.state, self.observation_noise_std)

        if not self.rl_mode:
            return observation, self.target
        else:
            s = np.hstack(observation, self.target)
            return s

    def render(self, mode="human"):

        pixels = self.env.physics.render(height=400, width=400, camera_id=0)

        if mode == "human":
            if self.img is None:
                self.img = plt.imshow(pixels)
                plt.axis('off')
            else:
                if self.ctx is None:
                    self.ctx = mujoco.GLContext(400, 400)
                    self.ctx.make_current()
                self.img.set_data(pixels)
            plt.draw()
            plt.pause(0.000001)

            return self.img
        if mode == "rgb_array":
            return pixels

    def get_target_pos(self):
        x = self.env.physics.named.data.geom_xpos['target', 'x']
        y = self.env.physics.named.data.geom_xpos['target', 'y']
        return np.array([x, y])

    def get_hand_pos(self):
        x = self.env.physics.named.data.geom_xpos['finger', 'x']
        y = self.env.physics.named.data.geom_xpos['finger', 'y']
        return np.array([x, y])


if __name__ == '__main__':

    env = ReacherEnv(42)
    for j in range(5):
        obs, tar = env.reset()
        env.render()
        while True:
            a = np.array([-1., -0.])
            obs, tar, done, _ = env.step(a)
            #print(obs[:2], tar, done)
            env.render()
            if done:
                break
