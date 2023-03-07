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
GRAY = (200, 200, 200)


class Reacherv2Env(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 50
    }

    def __init__(self, seed: int = None, max_episode_steps: int = 200, rl_mode: bool = False,
                 moving_target: float = 0.0, **kwargs):

        self.dt = 0.02  # seconds between state updates
        self.max_episode_steps = max_episode_steps
        self.min_action = -1.0
        self.max_action = 1.0
        self.force_mag = 8.0
        self.damp = 5.0

        self.rl_mode = rl_mode

        self.max_vel = np.pi
        self.min_vel = -self.max_vel

        self.random_target = True
        self.moving_target = moving_target
        self.done_on_target = False
        self.epsilon = 0.05

        self.process_noise_std = np.array([0., 0., 0., 0.])
        self.observation_noise_std = np.ones(8) * 0.01
        self.l1 = 0.5
        self.l2 = 0.4
        self.max_reach = self.l1 + self.l2

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(2,)
        )

        self.observation_space = spaces.Box(
            low=np.array([-self.max_reach, -self.max_reach, -1.0, -1.0, -1.0, -1.0, self.min_vel, self.min_vel]),
            high=np.array([self.max_reach, self.max_reach, 1.0, 1.0, 1.0, 1.0, self.max_vel, self.max_vel])
        )

        self.target_space = spaces.Box(
            low=np.array([-self.max_reach, -self.max_reach, -1.0, -1.0, -1.0, -1.0, self.min_vel, self.min_vel]),
            high=np.array([self.max_reach, self.max_reach, 1.0, 1.0, 1.0, 1.0, self.max_vel, self.max_vel])
        )

        self.loss_gain = np.array([1., 1., 0., 0., 0., 0., 0., 0.])

        self.state_labels = ['hand x', 'hand y', 'sin alpha', 'cos alpha', 'sin beta', 'cos beta', 'vel alpha', 'vel beta']

        self.seed(seed)
        self.screen = None
        self.clock = None
        self.isopen = True

        self.state = None
        self.target = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepPhysics(self, action):

        angles = self.state[:2]  # get last joint angles
        vel = self.state[2:]  # get last joint velocity

        # get change in state since last update
        dadt = vel
        dvdt = action * self.force_mag - self.damp * vel
        # update state
        angles += dadt * self.dt
        vel += dvdt * self.dt

        # clip velocity in allowed limits
        vel = np.clip(vel, self.min_vel, self.max_vel)

        # avoid very small velocity residues that bounce due to dampening
        vel[np.abs(vel) < 0.01 * self.max_vel] = 0.0

        return np.hstack([angles, vel])

    def stepTarget(self):

        angles = self.target[:2]  # get last joint angles
        vel = self.target[2:]  # get last joint velocity

        # get change in state since last update
        dadt = vel
        dvdt = 0.0
        # update target state
        angles += dadt * self.dt
        vel += dvdt * self.dt

        # clip velocity in allowed limits
        vel = np.clip(vel, self.min_vel, self.max_vel)

        # avoid very small velocity residues that bounce due to dampening
        vel[np.abs(vel) < 0.01 * self.max_vel] = 0.0

        self.target = np.hstack([angles, vel])

    def step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))
        # Cast action to float to strip np trappings

        self.state = np.array(self.stepPhysics(action))
        self.state = self.np_random.normal(self.state, self.process_noise_std)
        self.stepTarget()

        self.episode_step_count += 1

        done = False
        on_target = False
        if np.allclose(self.state[:2], self.target[:2], atol=self.epsilon):
            on_target = True
            if self.done_on_target:
                #self.reset(state=self.state)
                done = True

        max_steps = False
        if self.max_episode_steps and self.episode_step_count == self.max_episode_steps:
            done = True
            max_steps = True

        observation = self.make_observation(self.state)
        ob_rew = self.make_observation(self.state, noise=False)
        target_observation = self.make_observation(self.target, noise=False)

        reward = - np.linalg.norm(ob_rew[:2] - target_observation[:2]) * self.dt

        info = {'on_target': on_target, 'max_steps': max_steps}

        if not self.rl_mode:
            return observation, target_observation, reward, done, info
        else:
            s = np.hstack(observation, target_observation)
            return s, reward, done, info

    def make_observation(self, state, noise=True):

        a1 = state[0]
        a2 = state[1]

        p1_x = self.l1 * np.cos(a1)
        p1_y = self.l1 * np.sin(a1)

        hand_x = p1_x + self.l2 * np.cos(a1 + a2)
        hand_y = p1_y + self.l2 * np.sin(a1 + a2)

        norm_vel1 = state[2] / self.max_vel
        norm_vel2 = state[3] / self.max_vel

        observation = np.array([
            hand_x, hand_y, np.sin(a1), np.cos(a1), np.sin(a2), np.cos(a2), norm_vel1, norm_vel2
        ])

        if noise:
            observation = self.np_random.normal(observation, self.observation_noise_std)

        return observation

    def reset(self, state=None, target=None):
        self.episode_step_count = 0

        # state is [theta_1, theta_2, \dot{theta_1}, \dot{theta_2}]
        if state is None:
            self.state = np.zeros(4)
            self.state[:2] = self.np_random.uniform(low=0.0, high=2*np.pi, size=(2,))
        else:
            self.state = state

        if target is None:
            self.target = np.zeros(4)
            if self.random_target:
                self.target[:2] = self.np_random.uniform(low=0.0, high=2*np.pi, size=(2,))
                if self.np_random.uniform() < self.moving_target:
                    self.target[2:] = self.np_random.uniform(low=0.3*self.min_vel, high=0.3*self.max_vel, size=(2,))
            else:
                self.target[:2] = np.pi, np.pi
            #angle = self.np_random.uniform(low=0.0, high=2*np.pi)
            #radius = self.np_random.uniform(low=0.2, high=self.max_reach)
            #self.target[:2] = radius * np.array([np.cos(angle), np.sin(angle)])

            """
            if self.random_target:
                self.target[:2] = self.np_random.uniform(low=0.8*self.min_pos, high=0.8*self.max_pos, size=(2,))
                if self.np_random.rand() < self.moving_target:
                    self.target[2:4] = self.np_random.uniform(low=-0.5, high=0.5, size=(2,))
                    self.target_angle = self.np_random.uniform(low=30, high=180)
                    if self.np_random.rand() < 0.5:
                        self.target_angle *= -1
            """
        else:
            self.target = target

        observation = self.make_observation(self.state)
        target_observation = self.make_observation(self.target, noise=False)

        return observation, target_observation

    def render(self, mode="human", show_target_arm=False):
        screen_width = 400
        screen_height = screen_width

        center_x = screen_width / 2
        center_y = screen_height / 2

        a1 = self.state[0]
        a2 = self.state[1]

        p1_x = self.l1 * np.cos(a1) * center_x + center_x
        p1_y = self.l1 * np.sin(a1) * center_y + center_y

        p2_x = p1_x + self.l2 * np.cos(a1 + a2) * center_x
        p2_y = p1_y + self.l2 * np.sin(a1 + a2) * center_y

        target_observation = self.make_observation(self.target, noise=False)

        tar_x = target_observation[0] * center_x + center_x
        tar_y = target_observation[1] * center_y + center_y

        at1 = self.target[0]
        at2 = self.target[1]

        t1_x = self.l1 * np.cos(at1) * center_x + center_x
        t1_y = self.l1 * np.sin(at1) * center_y + center_y

        t2_x = t1_x + self.l2 * np.cos(at1 + at2) * center_x
        t2_y = t1_y + self.l2 * np.sin(at1 + at2) * center_y

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill(WHITE)

        rad = int(screen_width / 50)

        # draw the target

        if show_target_arm:
            pygame.draw.line(self.surf, GRAY, (center_x, center_y), (t1_x, t1_y), 10)
            pygame.draw.line(self.surf, GRAY, (t1_x, t1_y), (t2_x, t2_y), 8)

            gfxdraw.filled_circle(
                self.surf,
                int(np.rint(t1_x)),
                int(np.rint(t1_y)),
                rad,
                GRAY
            )

        gfxdraw.filled_circle(
            self.surf,
            int(np.rint(tar_x)),
            int(np.rint(tar_y)),
            int(np.rint(rad)),
            BLUE
        )

        # draw the arm

        pygame.draw.line(self.surf, BLACK, (center_x, center_y), (p1_x, p1_y), 10)
        pygame.draw.line(self.surf, BLACK, (p1_x, p1_y), (p2_x, p2_y), 8)

        gfxdraw.filled_circle(
            self.surf,
            int(np.rint(center_x)),
            int(np.rint(center_y)),
            int(np.rint(1.2*rad)),
            BLACK
        )

        gfxdraw.filled_circle(
            self.surf,
            int(np.rint(p1_x)),
            int(np.rint(p1_y)),
            rad,
            BLACK
        )

        gfxdraw.filled_circle(
            self.surf,
            int(np.rint(p2_x)),
            int(np.rint(p2_y)),
            rad,
            RED
        )

        """
        def draw_arrow(screen, colour, start, end, trirad=4, lwidth=3):
            pygame.draw.line(screen, colour, start, end, lwidth)
            rotation = math.degrees(math.atan2(start[1] - end[1], end[0] - start[0])) + 90
            pygame.draw.polygon(screen, colour, (
            (end[0] + trirad * math.sin(math.radians(rotation)), end[1] + trirad * math.cos(math.radians(rotation))), (
            end[0] + trirad * math.sin(math.radians(rotation - 120)), end[1] + trirad * math.cos(math.radians(rotation - 120))),
            (end[0] + trirad * math.sin(math.radians(rotation + 120)),
             end[1] + trirad * math.cos(math.radians(rotation + 120)))))

        if np.any(self.state[2:]):
            draw_arrow(self.surf, BLACK, (pos_x, pos_y), (pos_x + vel_x, pos_y + vel_y), screen_width//100, screen_width//200)
        if np.any(self.target[2:]):
            draw_arrow(self.surf, BLACK, (tar_x, tar_y), (tar_x + tar_vel_x, tar_y + tar_vel_y), screen_width//100, screen_width//200)
        """

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


if __name__ == '__main__':
    env = Reacherv2Env(seed=4)
    target = np.array([0.75, 0.75, 0.2, 0.2])
    target[:2] *= 2 * np.pi
    target[2:] *= env.max_vel
    o = env.reset() #target=target)
    #print(o)
    env.render(show_target_arm=True)
    for i in range(200):
        a = env.action_space.sample()
        if i < 100:
            a = np.array([1.0, 1.0], dtype=np.float32)
        else:
            a = np.array([.0, .0], dtype=np.float32)
        o = env.step(a)
        print(o[0][-2:])
        env.render(show_target_arm=True)
    env.close()