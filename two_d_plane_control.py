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
moving_target = 1.  # the probability of the target to be moving
dt = 0.02           # time step size
drag = 1.           # friction parameter
seed = 0            # random seed

# AIF parameters
# Precision parameters
pi_x_0 = 1.
pi_x_1 = 1.
pi_s_0 = 1.
pi_s_1 = 1.
# Gains
k_x = 0.5   # Perception gain
k_a = 2.    # Action gain
k_att = 1.  # Attractor (target)

# to keep things simple, we can set the initial state and target (or comment out)
initial_state = np.array([0.5, 0.5, 0., 0.])
initial_target = np.array([0., 0., 0., 0.])

# initialize the environment
env = TwoDPlaneEnv(
    moving_target=moving_target,
    # force_mag=XX,
    drag=drag,
    dt=dt,
    seed=seed
)

# state, target = env.reset(initial_state, initial_target)
# or uncomment for random state and target:
state, target = env.reset()

env.render()

# Auxiliary variables for clarity
state_pos = state[:2]
state_vel = state[2:]
target_pos = target[:2]
target_vel = target[2:]

# Generalized coordinates internal belief
x_0 = np.zeros_like(state_pos)
x_1 = np.zeros_like(state_vel)
x_2 = np.zeros_like(state_vel)

# Initial action (0 does nothing)
a = np.array([0., 0.])

# simulate the system
for i in range(N_steps):

    # Active inference (Laplace approx. MLE)
    # Perception
    x_dot_0 = x_1 - k_x * ( -pi_s_0 * (state_pos - x_0) + pi_x_0 * (x_1 + k_att * (x_0 - target_pos) ) )
    x_dot_1 = x_2 - k_x * ( -pi_s_1 * (state_vel - x_1) + pi_x_0 * (x_1 + k_att * (x_0 - target_pos) ) + pi_x_1 * (x_2 + x_1) )
    x_dot_2 = - k_x * ( pi_x_0 * (x_2 - x_1) )

    # Action
    a_dot = -k_a * ( pi_s_0 * (state_pos - x_0) + pi_s_1 * (state_vel - x_1) )
    # Euler integration
    x_0 += dt * x_dot_0
    x_1 += dt * x_dot_1
    x_2 += dt * x_dot_2
    a += dt * a_dot

    # the legal action space is limited between -1 and 1
    # in the env, 'a' is multiplied by a scalar which is 5.0 by default
    # this can be changed by passing the 'force_mag' to the env init
    a = np.clip(a, -1., 1.)

    # Execute action; the gym.env class wants float32
    state, target = env.step(a.astype(np.float32))[:2]

    # Update state
    state_pos = state[:2]
    state_vel = state[2:]
    target_pos = target[:2]
    target_vel = target[2:]

    # Display variables
    print()
    print("elapsed time [s]:", (i+1) * dt)
    print("performed action:", a)
    print("state:", state, "belief pos", x_0, "belief vel", x_1)
    print("new target:", target)

    # Visualize environment
    env.render()



