from src.envs.two_d_plane import TwoDPlaneEnv
import numpy as np

"""
The relevant arrays have the following shape:

state:  [x_1, x_2, \dot{x}_1, \dot{x}_1]
target: [y_1, y_2, \dot{y}_1, \dot{y}_1]
action: [a_1, a_2] where a_i is proportional to \ddot{x}_i.
"""

# setup some parameters
N_steps = 1000      # total number of steps
moving_target = 0.  # the probability of the target to be moving
dt = 0.02           # dt = 0.02 s

# Pablo's parameters
pi_pos = 1.
pi_vel = 1.
pi_t = 1.
k_a = 1.

# to keep things simple, we can set the initial state and target (or comment out)
initial_state = np.array([1., 0., 0., 0.])
initial_target = np.array([0., 0., 0., 0.])

# initialize the environment
env = TwoDPlaneEnv(
    moving_target=moving_target,
    # force_mag=XX,
    dt=dt
)

state, target = env.reset(initial_state, initial_target)
# or uncomment for random state and target:
# state, target = env.reset()

env.render()

# Pablo's math:
x_dot = np.zeros_like(state)  # not sure what this is for?
x_pos = state[:2]
x_vel = state[2:]

# pick an initial action to do nothing
a = np.array([0., 0.])

# simulate the system
for i in range(N_steps):

    # the legal action space is limited between -1 and 1
    # in the env, 'a' is multiplied by a scalar which is 5.0 by default
    # this can be changed by passing the 'force_mag' to the env init
    a = np.clip(a, -1., 1.)

    # the gym.env class wants float32
    state, target = env.step(a.astype(np.float32))[:2]

    print()
    print("performed action:", a)
    print("new state:", state)
    print("new target:", target)

    env.render()

    # Pablo's math:
    #x_dot = (state - x) * pi_x + (target - x) * pi_t
    #a_dot = -k_a * (pi_vel * (vel - x_vel) + pi_pos * (pos - x_pos))
    #x += dt * x_dot
    #a += dt * a_dot

