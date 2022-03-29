import gym
from gym import spaces
from gym.utils import seeding
import pygame
from pygame import gfxdraw
import numpy as np
import math

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
BLACK = (0, 0, 0)


class TwoDPlaneEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 50
    }

    def __init__(self, seed: int = None, max_episode_steps: int = 200):

        self.dt = 0.02  # seconds between state updates
        self.max_episode_steps = max_episode_steps
        self.min_action = -1.0
        self.max_action = 1.0
        self.force_mag = 5.0
        self.drag = 0.0

        self.max_pos = 1.0
        self.min_pos = -self.max_pos

        self.max_vel = 1.0
        self.min_vel = -self.max_vel

        self.stop_on_edge = False
        self.done_on_edge = True

        self.random_target = True
        self.done_on_target = False
        self.epsilon = 0.05

        self.process_noise_std = np.array([0., 0., 0., 0.])
        self.observation_noise_std = np.array([0.001, 0.001, 0.001, 0.001])

        self.gravity = np.array([0., 0.])

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(2,)
        )

        self.observation_space = spaces.Box(
            low=np.array([self.min_pos, self.min_pos, self.min_vel, self.min_vel]),
            high=np.array([self.max_pos, self.max_pos, self.max_vel, self.max_vel])
        )

        self.target_space = spaces.Box(
            low=np.array([self.min_pos, self.min_pos, self.min_vel, self.min_vel]),
            high=np.array([self.max_pos, self.max_pos, self.max_vel, self.max_vel])
        )

        self.loss_gain = np.array([1., 1., 0., 0.])

        self.seed(seed)
        self.screen = None
        self.clock = None
        self.isopen = True

        self.state = None
        self.target = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def check_pos_limit(self, x, dx):

        if x < self.min_pos:
            x = self.min_pos
            dx = 0.0
        if x > self.max_pos:
            x = self.max_pos
            dx = 0.0

        return x, dx

    def stepPhysics(self, action):

        pos = self.state[:2]  # get last position
        vel = self.state[2:]  # get last velocity

        # get change in state since last update
        dpdt = vel
        dvdt = action * self.force_mag + self.gravity - self.drag * vel**2
        #vel = 0.5**self.dt * vel + dvdt * self.dt

        # update state
        pos += dpdt * self.dt
        vel += dvdt * self.dt

        # clip velocity in allowed limits
        vel = np.clip(vel, self.min_vel, self.max_vel)
        mag_V = np.sqrt(np.sum(np.square(vel)))
        if mag_V > self.max_vel:
            vel = vel / mag_V

        return np.hstack([pos, vel])

    def step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))
        # Cast action to float to strip np trappings

        self.state = np.array(self.stepPhysics(action))
        self.state = np.random.normal(self.state, self.process_noise_std)

        # stop when position reaches edge of state space
        if self.stop_on_edge:
            self.state[0], self.state[2] = self.check_pos_limit(self.state[0], self.state[2])
            self.state[1], self.state[3] = self.check_pos_limit(self.state[1], self.state[3])

        self.episode_step_count += 1

        done = False
        if self.done_on_target:
            if np.allclose(self.state[~np.isnan(self.target)], self.target[~np.isnan(self.target)], atol=self.epsilon):
                #self.reset(state=self.state)
                done = True

        if self.done_on_edge:
            if np.min(self.state) <= self.min_pos or np.max(self.state) >= self.max_pos:
                done = True

        if self.max_episode_steps and self.episode_step_count == self.max_episode_steps:
            done = True

        observation = np.random.normal(self.state, self.observation_noise_std)

        return observation, self.target, done, {}


    def reset(self, state=None, target=None):
        self.episode_step_count = 0
        if state is None:
            self.state = np.zeros(4)
            self.state[:2] = self.np_random.uniform(low=0.8*self.min_pos, high=0.8*self.max_pos, size=(2,))
        else:
            self.state = state

        if target is None:
            self.target = np.zeros(4)
            if self.random_target:
                self.target[:2] = self.np_random.uniform(low=0.8*self.min_pos, high=0.8*self.max_pos, size=(2,))
        else:
            self.target = target

        observation = np.random.normal(self.state, self.observation_noise_std)

        return observation, self.target

    def render(self, mode="human"):
        screen_width = 400
        screen_height = screen_width

        center_x = screen_width / 2
        center_y = screen_height / 2

        pos_x = self.state[0] * center_x + center_x
        pos_y = self.state[1] * center_y + center_y

        vel_x = self.state[2] * screen_width / 10
        vel_y = self.state[3] * screen_height / 10

        tar_x = self.target[0] * center_x + center_x
        tar_y = self.target[1] * center_y + center_y

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill(WHITE)

        rad = int(screen_width / 50)

        gfxdraw.filled_circle(
            self.surf,
            int(np.rint(tar_x)),
            int(np.rint(tar_y)),
            rad,
            BLUE
        )

        gfxdraw.filled_circle(
            self.surf,
            int(np.rint(pos_x)),
            int(np.rint(pos_y)),
            rad,
            RED
        )

        def draw_arrow(screen, colour, start, end, trirad=4, lwidth=3):
            pygame.draw.line(screen, colour, start, end, lwidth)
            rotation = math.degrees(math.atan2(start[1] - end[1], end[0] - start[0])) + 90
            pygame.draw.polygon(screen, colour, (
            (end[0] + trirad * math.sin(math.radians(rotation)), end[1] + trirad * math.cos(math.radians(rotation))), (
            end[0] + trirad * math.sin(math.radians(rotation - 120)), end[1] + trirad * math.cos(math.radians(rotation - 120))),
            (end[0] + trirad * math.sin(math.radians(rotation + 120)),
             end[1] + trirad * math.cos(math.radians(rotation + 120)))))

        draw_arrow(self.surf, BLACK, (pos_x, pos_y), (pos_x + vel_x, pos_y + vel_y), screen_width//100, screen_width//200)

        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class TwoDPlaneEnvSimple(TwoDPlaneEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, seed: int = None, max_episode_steps: int = 200):
        super(TwoDPlaneEnvSimple, self).__init__(seed, max_episode_steps)
        self.force_mag = 1.0


    def stepPhysics(self, action):

        pos = self.state[:2]  # get last position
        vel = self.state[2:]  # get last velocity

        dpdt = vel  # get change in position since last update

        pos += dpdt * self.dt  # update position
        vel = action * self.force_mag  # update velocity

        # clip velocity in allowed limits
        vel = np.clip(vel, self.min_vel, self.max_vel)
        mag_V = np.sqrt(np.sum(np.square(vel)))
        if mag_V > self.max_vel:
            vel = vel / mag_V

        return np.hstack([pos, vel])