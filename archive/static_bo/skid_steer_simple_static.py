"""Model a skid steer robot on simple 2D grid."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
import time as timer

# for reproducing the risk map
SEED = 2
np.random.seed(SEED)
RNG = np.random.RandomState(SEED)

### PARAMETERS
# grid
# 10 m x 10 m
GRID_CELL_SIZE = 0.1  # m
GRID_L = 10  # m
RISK_MAP_DIM = int(GRID_L / GRID_CELL_SIZE)
GRID_RANGE = [0, GRID_L]

# skid steer robot
L = 0.5  # wheel base (m)
r = 0.1  # wheel radius (m)
INITIAL_STATE = np.array([9.5, 5.8, np.pi])
GOAL_STATE    = np.array([5.9, 0.3, 0])

# control constraints
V_MAX = 1.0  # m/s
W_MAX = V_MAX / r  # rad/s

# simulation params
SIM_UNTIL_GOAL = True  # ignore T_sim
T     = 2  # horizon (s) keeps H ~ 200
T_sim = 100  # s
dt    = 0.01  # s


### functions

def dynamics(state, control, r=r, L=L):
    """Simulate skid steer dynamics for one time step"""
    x, y, theta = state
    u_l, u_r = control

    dx     = r * (u_l + u_r) * np.cos(theta) / 2
    dy     = r * (u_l + u_r) * np.sin(theta) / 2
    dtheta = r * (u_r - u_l) / L

    return np.array([dx, dy, dtheta])


def evaluate_cost(x, u, goal_state, risk_map):
    """Cost function used by the CEM planner"""
    cost = 0
    W = [1, 0, 5.0 / 600, 25]

    for ind, state in enumerate(x):
        i = int(np.clip(np.floor(state[0] / GRID_CELL_SIZE), 0, RISK_MAP_DIM - 1))
        j = int(np.clip(np.floor(state[1] / GRID_CELL_SIZE), 0, RISK_MAP_DIM - 1))

        cost_risk     = risk_map[i, j]
        cost_dist     = np.linalg.norm(goal_state[:2] - state[:2])
        cost_controls = np.sum(np.square(u[ind, :]))

        cost += cost_risk + W[1] * cost_dist + W[2] * cost_controls

    total_cost = cost + W[3] * np.linalg.norm(goal_state[:2] - x[-1, :2])
    return total_cost


def simulate(x0, u, dt=dt, T=T):
    """Simulate one trajectory of length T from x0 under control sequence u."""
    x = np.zeros((int(np.ceil(T / dt)), 3))
    x[0, :] = x0

    for t_idx in range(1, int(np.ceil(T / dt))):
        dx_t = dynamics(x[t_idx - 1, :], u[t_idx - 1, :])
        x[t_idx, :] = x[t_idx - 1, :] + dt * dx_t
        x[t_idx, 0] = np.clip(x[t_idx, 0], GRID_RANGE[0], GRID_RANGE[1])
        x[t_idx, 1] = np.clip(x[t_idx, 1], GRID_RANGE[0], GRID_RANGE[1])

    return x


def generate_cost_map(x_dim, y_dim):
    mu, sigma = 0.7, 1.0
    cost_map = np.random.normal(loc=mu, scale=sigma, size=(x_dim, y_dim))

    # # custom safe zone
    # for i in range(60, 86):
    #     for j in range(54, 68):
    #         cost_map[i, j] = 0.4

    cost_map = gaussian_filter(cost_map, sigma=5.0)
    cost_map = (cost_map - np.min(cost_map)) / (np.max(cost_map) - np.min(cost_map))
    return cost_map


def cvar_risk_map(x_dim, y_dim, alpha, cost_map):
    """Compute a CVaR risk map at the given alpha level"""
    grid = np.linspace(0.001, 0.999, 1001)
    risk_map_cvar = np.zeros_like(cost_map)

    for i in range(x_dim):
        for j in range(y_dim):
            cell_mean  = cost_map[i, j]
            cell_sigma = 0.05 + cost_map[i, j] * 2

            dist       = norm(cell_mean, cell_sigma)
            percentiles = dist.ppf(grid)

            # upper-tail CVaR
            risk_map_cvar[i, j] = percentiles[grid > (1 - alpha)].mean()

    return risk_map_cvar


def cem_step(x0, goal_state, risk_map, mu_init, sigma_init):
    """One MPC step using the Cross-Entropy Method"""
    N_SAMPLES   = 100
    N_ELITES    = 20
    N_ITERATIONS = 2
    ALPHA_CEM   = 0.3
    SIGMA_MIN   = 0.5

    mu    = mu_init
    sigma = sigma_init

    for _ in range(N_ITERATIONS):
        a_i = (-W_MAX - mu) / sigma
        b_i = ( W_MAX - mu) / sigma

        U_samples = truncnorm.rvs(
            a=a_i, b=b_i, loc=mu, scale=sigma,
            size=(N_SAMPLES, int(np.ceil(T / dt)), 2)
        )

        costs = np.zeros(N_SAMPLES)
        for i in range(N_SAMPLES):
            X_i     = simulate(x0, U_samples[i])
            costs[i] = evaluate_cost(X_i, U_samples[i], goal_state, risk_map)

        elite_indices = np.argsort(costs)[:N_ELITES]
        elite_samples = U_samples[elite_indices]

        mu_new    = np.mean(elite_samples, axis=0)
        sigma_new = np.std(elite_samples,  axis=0)

        mu    = ALPHA_CEM * mu    + (1 - ALPHA_CEM) * mu_new
        sigma = ALPHA_CEM * sigma + (1 - ALPHA_CEM) * sigma_new
        sigma = np.maximum(sigma, SIGMA_MIN)

    return mu, sigma


def plot_trajectory(x, goal_state, risk_map):
    fig, ax = plt.subplots(figsize=(10, 10))

    im = ax.imshow(
        risk_map.T,
        origin='lower',
        extent=[GRID_RANGE[0], GRID_RANGE[1], GRID_RANGE[0], GRID_RANGE[1]],
        cmap='Reds',
        alpha=0.6
    )
    plt.colorbar(im, ax=ax, label='CVaR Cost')

    ax.plot(x[:, 0], x[:, 1], 'b-o', linewidth=2, markersize=4)
    ax.plot(x[0,  0], x[0,  1], 'go', markersize=8, label='Start')
    ax.plot(x[-1, 0], x[-1, 1], 'ro', markersize=8, label='End')
    ax.plot(goal_state[0], goal_state[1], 'mo', markersize=8, label='Goal')

    num_arrows  = 10
    marker_size = 100
    for i in range(0, len(x), max(1, len(x) // num_arrows)):
        theta = x[i, 2]
        ax.scatter(x[i, 0], x[i, 1],
                   marker=(3, 0, np.degrees(theta) - 90),
                   s=marker_size, c='red', alpha=0.8, zorder=5)

    ax.grid(which='minor', color='#AAAAAA', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.grid(which='major', color='#AAAAAA', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.set_xlim(GRID_RANGE[0], GRID_RANGE[1])
    ax.set_ylim(GRID_RANGE[0], GRID_RANGE[1])
    ax.set_xticks(np.arange(GRID_RANGE[0], GRID_RANGE[1] + GRID_CELL_SIZE, GRID_CELL_SIZE), minor=True)
    ax.set_yticks(np.arange(GRID_RANGE[0], GRID_RANGE[1] + GRID_CELL_SIZE, GRID_CELL_SIZE), minor=True)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Skid-steer robot traversal')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    plt.show()


def main():
    # static alpha (1 = risk-neutral, 0 = risk-averse)
    alpha_CVaR = 0.99

    cost_map = generate_cost_map(RISK_MAP_DIM, RISK_MAP_DIM)
    risk_map = cvar_risk_map(RISK_MAP_DIM, RISK_MAP_DIM, alpha_CVaR, cost_map)

    current_state = INITIAL_STATE.copy()
    executed_path = [current_state.copy()]

    mu_control    = np.zeros((int(np.ceil(T / dt)), 2))
    sigma_control = np.ones((int(np.ceil(T / dt)), 2)) * W_MAX * 0.5

    max_steps = int(np.ceil(T_sim / dt))

    # stuck window - 10 s of sim time
    check_stuck_iters = int(10.0 / dt)

    # goal tol - 1 cell
    eps = GRID_CELL_SIZE

    if SIM_UNTIL_GOAL:
        print(f'Sim until robot reaches goal (eps={eps:.2f} m).')
    else:
        print(f'Sim time is {T_sim} s  ({max_steps} steps).')
    print(f'Starting MPC sim  dt={dt}  Max U={W_MAX:.1f} rad/s  alpha={alpha_CVaR}')

    start_time = timer.time()
    curr_step  = 0

    while True:
        if not SIM_UNTIL_GOAL and curr_step >= max_steps:
            break

        # stuck -> just stop (alpha is static, nothing to adapt)
        if curr_step % check_stuck_iters == 0 and curr_step >= check_stuck_iters:
            prev_state = executed_path[curr_step - check_stuck_iters]
            if (np.abs(current_state[0] - prev_state[0]) <= eps and
                    np.abs(current_state[1] - prev_state[1]) <= eps):
                print(f'Robot stuck at step {curr_step} with static alpha={alpha_CVaR:.2f} — stopping.')
                break

        best_u_plan, new_sigma = cem_step(
            x0=current_state,
            goal_state=GOAL_STATE,
            risk_map=risk_map,
            mu_init=mu_control,
            sigma_init=sigma_control
        )

        optimal_control = best_u_plan[0, :]
        dx_applied      = dynamics(current_state, optimal_control)
        current_state   = current_state + dt * dx_applied
        current_state[0] = np.clip(current_state[0], GRID_RANGE[0], GRID_RANGE[1])
        current_state[1] = np.clip(current_state[1], GRID_RANGE[0], GRID_RANGE[1])
        executed_path.append(current_state.copy())

        mu_control        = np.roll(best_u_plan, -1, axis=0)
        mu_control[-1, :] = mu_control[-2, :]
        sigma_control        = np.roll(new_sigma, -1, axis=0)
        sigma_control[-1, :] = new_sigma[-2, :]

        distance_to_goal = np.linalg.norm(current_state[:2] - GOAL_STATE[:2])
        map_i = int(np.floor(current_state[0] / GRID_CELL_SIZE))
        map_j = int(np.floor(current_state[1] / GRID_CELL_SIZE))

        if curr_step % 100 == 0:
            print(f"Step {curr_step:5d}: Pos=({current_state[0]:.2f}, {current_state[1]:.2f})"
                  f"  Map[{map_i}][{map_j}]"
                  f"  GoalDist={distance_to_goal:.2f} m"
                  f"  U=({optimal_control[0]:.2f}, {optimal_control[1]:.2f})")

        if distance_to_goal < eps:
            print(f"Step {curr_step:5d}: Pos=({current_state[0]:.2f}, {current_state[1]:.2f})"
                  f"  GoalDist={distance_to_goal:.2f} m — Goal reached.")
            break

        curr_step += 1

    elapsed = timer.time() - start_time
    print(f'Elapsed: {elapsed:.2f} s  ({curr_step} steps)')

    path_arr = np.array(executed_path)
    plot_trajectory(path_arr, GOAL_STATE, risk_map)


if __name__ == "__main__":
    main()
