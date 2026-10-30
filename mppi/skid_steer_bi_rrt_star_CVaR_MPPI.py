"""Skid-steer robot on a 2D grid: bi-RRT* global plan + CVaR-MPPI tracking."""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
import time as timer

### CFG
plt.rcParams.update({'font.size': 11})

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

# control constraints
V_MAX = 1.0  # m/s
W_MAX = V_MAX/r  # rad/s

# simulation flag
SIM_UNTIL_GOAL = False

# MPPI tracks a local reference traj from the global planner
T = 1.5  # horizon (s)
T_sim = 50  # sim length (s)
dt = 0.05  # s
N_STEPS_HORIZON = int(np.ceil(T/dt))
MAX_STEPS_MPC = int(np.ceil(T_sim/dt))

# W = [terrain CVaR, tracking, control, terminal]
W_COST = [1.0, 10.0, 0.01, 0.5]
W_HEADING = 0.05
W_DU = 0.05

# MPPI params
N_SAMPLES_MPPI = 2000

# temperature
LAMBDA_MPPI = 0.03
SIGMA_MPPI = W_MAX * 0.15

# lookahead along the bi-RRT* path for the local reference (m)
PATH_LOOKAHEAD = 0.3
REF_SPEED = 0.2 * V_MAX

# monte carlo cost evaluation
# inner - resample on one fixed path outer - fresh (map, plan, run) scenarios
N_RESAMPLES = 25

# CVaR-MPPI terrain-belief sampling
N_TERRAIN_SAMPLES = 32
SIGMA_SCALE = 1.25

# terrain CVaR in the MPPI cost
#   "sampled" monte carlo over terrain draws (noisy)
#   "deterministic" closed-form per-cell CVaR summed along the path
# not quite the same objective (CVaR of the sum vs sum of CVaRs) expect a
# small offset override with CVAR_MODE=sampled python
CVAR_MODE = os.environ.get("CVAR_MODE", "deterministic")

# evaluation should have same tail across every controller beta
EVAL_CVAR_BETA = 0.5

## SEEDS
N_OUTER_SEEDS = 5
SEED_START = 6000

# bi-RRT*
# bi-RRT* minimises the CVaR risk integral
RRT_STEP_SIZE = 0.4  # max edge length (m)
RRT_REWIRE_RADIUS = 0.8  # rewire radius (m)
RRT_MAX_ITERS = 4000
RRT_EXTRA_ITERS = 500  # keep improving after first connect

### functions
def dynamics(state, control, r=r, L=L):
    """One step of skid-steer dynamics. state = (x, y, theta), control = (u_l, u_r)."""
    x, y, theta = state
    u_l, u_r = control
    
    dx = r * (u_l + u_r) * np.cos(theta) / 2
    dy = r * (u_l + u_r) * np.sin(theta) / 2
    dtheta = r * (u_r - u_l) / L
        
    return np.array([dx, dy, dtheta])

def dynamics_batch(states, controls, r=r, L=L):
    """Batched skid-steer step."""
    x, y, theta = states[:, 0], states[:, 1], states[:, 2]
    u_l, u_r = controls[:, 0], controls[:, 1]

    dx     = r * (u_l + u_r) * np.cos(theta) / 2
    dy     = r * (u_l + u_r) * np.sin(theta) / 2
    dtheta = r * (u_r - u_l) / L

    return np.stack([dx, dy, dtheta], axis=-1)

def upper_tail_cvar(sample_costs, beta):
    if beta <= 0.0 or beta > 1.0:
        raise ValueError('beta must be in (0, 1].')

    n_samples = sample_costs.shape[-1]
    n_tail = max(1, int(np.ceil(beta * n_samples)))

    sorted_costs = np.sort(sample_costs, axis=-1)
    tail_start = n_samples - n_tail
    tail_costs = sorted_costs[:, tail_start:]
    return np.mean(tail_costs, axis=1)

def cvar_normal_upper(mu, sigma, beta):
    """Upper-tail CVaR of Normal(mu, sigma)."""
    if beta <= 0.0 or beta > 1.0:
        raise ValueError('beta must be in (0, 1].')
    z = norm.ppf(1.0 - beta)
    return mu + sigma * norm.pdf(z) / beta

def integrate_piecewise_linear(y, x):
    y = np.asarray(y)
    x = np.asarray(x)

    left_y = y[:-1]
    right_y = y[1:]
    widths = np.diff(x)
    trapezoids = 0.5 * (left_y + right_y) * widths

    return np.sum(trapezoids)

# batched cost
def evaluate_cost_batch(X_batch, U_batch, ref_traj_xy, cost_map, noise_map, W,
                        beta_cvar, mu_nominal):

    # metres -> grid indices (clipped)
    I = np.clip(np.floor(X_batch[:, :, 0] / GRID_CELL_SIZE).astype(int), 0, RISK_MAP_DIM-1)
    J = np.clip(np.floor(X_batch[:, :, 1] / GRID_CELL_SIZE).astype(int), 0, RISK_MAP_DIM-1)
    mu_terrain = cost_map[I, J]
    sigma_terrain = SIGMA_SCALE * np.maximum(noise_map[I, J], 1e-3)

    # risk term see CVAR_MODE
    step_lengths = np.linalg.norm(np.diff(X_batch[:, :, :2], axis=1), axis=-1)

    if CVAR_MODE == "deterministic":
        # closed-form per-cell CVaR along the path
        per_cell_cvar = np.maximum(
            cvar_normal_upper(mu_terrain, sigma_terrain, beta_cvar), 0.0
        )
        average_step_cost = 0.5 * (per_cell_cvar[:, :-1] + per_cell_cvar[:, 1:])
        cvar_terrain_cost = np.sum(average_step_cost * step_lengths, axis=-1)
    else:
        # monte carlo - sample terrain integrate take tail mean
        sample_shape = (X_batch.shape[0], N_TERRAIN_SAMPLES, X_batch.shape[1])
        terrain_samples = np.random.normal(
            loc=mu_terrain[:, np.newaxis, :],
            scale=sigma_terrain[:, np.newaxis, :],
            size=sample_shape,
        )
        terrain_samples = np.maximum(terrain_samples, 0.0)
        average_step_cost = 0.5 * (terrain_samples[:, :, :-1] + terrain_samples[:, :, 1:])
        weighted_step_cost = average_step_cost * step_lengths[:, np.newaxis, :]
        terrain_integrals = np.sum(weighted_step_cost, axis=-1)
        cvar_terrain_cost = upper_tail_cvar(terrain_integrals, beta_cvar)

    cost_terrain = W[0] * cvar_terrain_cost

    # track the planned path over the whole horizon
    ref_positions = ref_traj_xy[np.newaxis, :, :2]
    cost_dist = W[1] * np.linalg.norm(ref_positions - X_batch[:, :, :2], axis=-1)

    ref_headings = ref_traj_xy[np.newaxis, :, 2]
    raw_heading_err = ref_headings - X_batch[:, :, 2]
    heading_err = np.arctan2(np.sin(raw_heading_err), np.cos(raw_heading_err))
    cost_heading = W_HEADING * np.square(heading_err)

    # normalised wheel effort
    cost_controls = W[2] * np.sum(np.square(U_batch / W_MAX), axis=-1)
    dU = np.diff(U_batch / W_MAX, axis=1)
    cost_smooth = np.zeros_like(cost_controls)
    cost_smooth[:, 1:] = W_DU * np.sum(np.square(dU), axis=-1)

    first_du = (U_batch[:, 0, :] - mu_nominal[0, :]) / W_MAX
    cost_smooth[:, 0] = W_DU * np.sum(np.square(first_du), axis=-1)

    # normalise by horizon length
    running_cost = cost_dist + cost_heading + cost_controls + cost_smooth
    deterministic_cost = np.sum(running_cost, axis=-1) / N_STEPS_HORIZON

    # add final state penalty
    terminal_dist = np.linalg.norm(ref_traj_xy[-1, :2] - X_batch[:, -1, :2], axis=-1)
    return deterministic_cost + cost_terrain + W[3] * terminal_dist

# U_batch: (N_SAMPLES, T/dt, 2)
def simulate_batch(x0, U_batch, dt=dt):
    N_SAMPLES = U_batch.shape[0]

    X = np.zeros((N_SAMPLES, N_STEPS_HORIZON, 3))
    X[:, 0, :] = x0
    for t_idx in range(1, N_STEPS_HORIZON):
        dx_t = dynamics_batch(X[:, t_idx-1, :], U_batch[:, t_idx-1, :])
        X[:, t_idx, :] = X[:, t_idx-1, :] + dt*dx_t

    return X

def generate_cost_map(x_dim, y_dim):
    # smooth nominal terrain encourage risk to come from uncertainty
    cost_map = np.random.normal(loc=0.55, scale=0.7, size=(x_dim, y_dim))
    cost_map = gaussian_filter(cost_map, sigma=7.0)

    # push down extremes threshold
    cost_map = (cost_map - np.min(cost_map)) / (np.max(cost_map) - np.min(cost_map))
    cost_map = 0.12 + 0.55 * cost_map
    cost_map[cost_map <= 0.35] = 0.35 * (cost_map[cost_map <= 0.35] / 0.35) ** 1.5

    return np.clip(cost_map, 0.0, 1.0)

def draw_blob(noise, center_m, radius_m, intensity=1.0):
    ii, jj = np.meshgrid(np.arange(noise.shape[0]), np.arange(noise.shape[1]), indexing='ij')
    cx, cy = center_m[0] / GRID_CELL_SIZE, center_m[1] / GRID_CELL_SIZE
    rx, ry = radius_m[0] / GRID_CELL_SIZE, radius_m[1] / GRID_CELL_SIZE
    x_distance = (ii - cx) ** 2 / rx ** 2
    y_distance = (jj - cy) ** 2 / ry ** 2
    mask = x_distance + y_distance <= 1.0
    noise[mask] = np.maximum(noise[mask], intensity)

def generate_noise_map(x_dim, y_dim, rng, start_xy, goal_xy):
    noise = 0.02 * np.ones((x_dim, y_dim))
    xs = (np.arange(x_dim) + 0.5) * GRID_CELL_SIZE
    ys = (np.arange(y_dim) + 0.5) * GRID_CELL_SIZE
    xx, yy = np.meshgrid(xs, ys, indexing='ij')

    start_xy = np.asarray(start_xy, dtype=float)
    goal_xy = np.asarray(goal_xy, dtype=float)

    dist_from_start = np.linalg.norm(
        np.stack([xx - start_xy[0], yy - start_xy[1]], axis=-1),
        axis=-1,
    )
    dist_prior = dist_from_start / (np.sqrt(2.0) * GRID_L)
    noise += 0.05 * dist_prior

    n_blobs = rng.randint(3, 6)
    for _ in range(n_blobs):
        center_m = np.array([
            rng.uniform(1.5, GRID_L - 1.5),
            rng.uniform(1.5, GRID_L - 1.5),
        ])
        radius_m = np.array([
            rng.uniform(0.8, 2.5),
            rng.uniform(0.8, 2.5),
        ])
        draw_blob(noise, center_m, radius_m, intensity=rng.uniform(0.08, 0.30))

    # uncertainty blobs near the start-goal corridor decoupled from cost
    seg = goal_xy - start_xy
    seg_len = np.linalg.norm(seg)
    tangent = seg / seg_len
    normal = np.array([-tangent[1], tangent[0]])

    for frac in np.linspace(0.18, 0.82, rng.randint(4, 7)):
        center_m = start_xy + (frac + rng.uniform(-0.08, 0.08)) * seg
        center_m += rng.uniform(-1.1, 1.1) * normal
        center_m = np.clip(center_m, 1.0, GRID_L - 1.0)
        radius_m = np.array([
            rng.uniform(0.55, 1.35),
            rng.uniform(0.55, 1.35),
        ])
        intensity = rng.uniform(0.35, 0.85)
        draw_blob(noise, center_m, radius_m, intensity=intensity)

    # smooth boundaries
    noise = gaussian_filter(noise, sigma=3.0)
    return np.clip(noise, 0.0, 1.0)

def cvar_risk_map(beta, cost_map, noise_map):
    ## CVaR (conditional value at risk)
    # ref - https://gdmarmerola.github.io/risk-aware-bandits/
    if beta <= 0.0 or beta > 1.0:
        raise ValueError('beta must be in (0, 1].')

    cell_sigma = SIGMA_SCALE * np.maximum(noise_map, 1e-3)

    # beta = tail probability lower beta -> more conservative
    tail_start = norm.ppf(1.0 - beta)
    return cost_map + cell_sigma * norm.pdf(tail_start) / beta

def belief_maps(cost_map, noise_map):
    return cost_map, SIGMA_SCALE * np.maximum(noise_map, 1e-3)

def generate_safety_map(risk_map_cvar):
    safety_map = 1.0 / (1.0 + risk_map_cvar)
    return safety_map

### bi-RRT*
# refs
#  - RRT* cost-to-come rewiring: Karaman & Frazzoli, "Sampling-based Algorithms for
#    optimal motion planning" IJRR 2011
#  - Bidirectional tree connection: Kuffner & LaValle, "RRT-Connect:
#    an efficient approach to single-query path planning" ICRA 2000

def bilinear_sample(grid_map, x, y):
    # bilinear lookup at a continuous (x, y)
    gx = np.clip(x / GRID_CELL_SIZE - 0.5, 0, RISK_MAP_DIM - 1)
    gy = np.clip(y / GRID_CELL_SIZE - 0.5, 0, RISK_MAP_DIM - 1)
    i0, j0 = int(np.floor(gx)), int(np.floor(gy))
    i1, j1 = min(i0+1, RISK_MAP_DIM-1), min(j0+1, RISK_MAP_DIM-1)
    fx, fy = gx - i0, gy - j0
    v00, v10 = grid_map[i0, j0], grid_map[i1, j0]
    v01, v11 = grid_map[i0, j1], grid_map[i1, j1]
    top_left = v00 * (1 - fx) * (1 - fy)
    bottom_left = v10 * fx * (1 - fy)
    top_right = v01 * (1 - fx) * fy
    bottom_right = v11 * fx * fy
    return top_left + bottom_left + top_right + bottom_right

def check_edge(risk_map, p_from, p_to):
    # supersample the edge at ~cell resolution
    edge_len = np.linalg.norm(p_to - p_from)
    n_samples = max(2, int(np.ceil(edge_len / GRID_CELL_SIZE)))
    ts = np.linspace(0.0, 1.0, n_samples)
    start = p_from[np.newaxis, :]
    direction = (p_to - p_from)[np.newaxis, :]
    pts = start + ts[:, np.newaxis] * direction

    # cost-to-come = CVaR risk integral along the edge
    risks = []
    for px, py in pts:
        risks.append(bilinear_sample(risk_map, px, py))
    risks = np.array(risks)
    sample_distances = np.linspace(0.0, edge_len, n_samples)
    edge_cost = integrate_piecewise_linear(risks, sample_distances)
    return edge_cost

class RRTNode:
    __slots__ = ('pos', 'parent', 'cost')
    def __init__(self, pos, parent, cost):
        self.pos = pos  # m
        self.parent = parent  # parent index none for root
        self.cost = cost  # cost-to-come

def nearest(tree, point):
    dists = [np.linalg.norm(n.pos - point) for n in tree]
    idx = int(np.argmin(dists))
    return idx, dists[idx]

def near_indices(tree, point, radius):
    nearby = []
    for i, node in enumerate(tree):
        if np.linalg.norm(node.pos - point) <= radius:
            nearby.append(i)
    return nearby

def steer(p_from, p_to, step_size):
    vec = p_to - p_from
    dist = np.linalg.norm(vec)
    if dist <= step_size:
        return p_to
    return p_from + vec * (step_size / dist)

def extend_tree(tree, sample, risk_map):
    # nearest -> steer -> cost -> rewire
    idx_near, _ = nearest(tree, sample)
    p_new = steer(tree[idx_near].pos, sample, RRT_STEP_SIZE)
    p_new = np.clip(p_new, GRID_RANGE[0], GRID_RANGE[1])

    edge_cost = check_edge(risk_map, tree[idx_near].pos, p_new)

    # best parent among neighbours
    near = near_indices(tree, p_new, RRT_REWIRE_RADIUS)
    best_parent, best_cost = idx_near, tree[idx_near].cost + edge_cost
    for j in near:
        cost_j = check_edge(risk_map, tree[j].pos, p_new)
        if tree[j].cost + cost_j < best_cost:
            best_parent, best_cost = j, tree[j].cost + cost_j

    new_idx = len(tree)
    tree.append(RRTNode(p_new, best_parent, best_cost))

    # rewire neighbours through the new node if cheaper
    for j in near:
        cost_j = check_edge(risk_map, p_new, tree[j].pos)
        if best_cost + cost_j < tree[j].cost:
            tree[j].parent = new_idx
            tree[j].cost = best_cost + cost_j

    return new_idx


def extract_chain(tree, idx):
    chain = []
    while idx is not None:
        chain.append(tree[idx].pos)
        idx = tree[idx].parent
    return chain  # leaf-to-root order

def bi_rrt_star(risk_map, start_xy, goal_xy):
    tree_a = [RRTNode(np.array(start_xy, dtype=float), None, 0.0)]
    tree_b = [RRTNode(np.array(goal_xy, dtype=float), None, 0.0)]

    best_connection = None  # (idx_a, idx_b, cost)
    iters_left = None

    for it in range(RRT_MAX_ITERS):
        sample = np.random.uniform(GRID_RANGE[0], GRID_RANGE[1], size=2)

        new_idx = extend_tree(tree_a, sample, risk_map)
        # try to connect the other tree to the new node
        idx_b_near, _ = nearest(tree_b, tree_a[new_idx].pos)
        edge_cost = check_edge(risk_map, tree_b[idx_b_near].pos, tree_a[new_idx].pos)
        total = tree_a[new_idx].cost + tree_b[idx_b_near].cost + edge_cost
        if best_connection is None or total < best_connection[2]:
            best_connection = (new_idx, idx_b_near, total)
            if iters_left is None:
                iters_left = RRT_EXTRA_ITERS

        # swap trees
        tree_a, tree_b = tree_b, tree_a
        if best_connection is not None:
            best_connection = (best_connection[1], best_connection[0], best_connection[2])

        if iters_left is not None:
            iters_left -= 1
            if iters_left <= 0:
                break

    if best_connection is None:
        print("bi-RRT* failed to connect start and goal trees")
        return None

    idx_a, idx_b, _ = best_connection
    chain_a = extract_chain(tree_a, idx_a)
    chain_b = extract_chain(tree_b, idx_b)

    # trees get swapped so check which chain starts at start_xy
    if np.linalg.norm(chain_a[-1] - np.array(start_xy)) < np.linalg.norm(chain_b[-1] - np.array(start_xy)):
        start_chain, goal_chain = chain_a, chain_b
    else:
        start_chain, goal_chain = chain_b, chain_a

    path = list(reversed(start_chain)) + goal_chain
    return np.array(path)

### MPPI refs
# williams aldrich theodorou "model predictive path integral control - from
# theory to parallel computation" J guidance control and dynamics 2017
# CVaR risk-aware MPPI - yin zhang tsiotras "risk-aware model predictive path
# integral control using conditional value-at-risk" RA-L 2023

def path_arclength(path_xy):
    segment_lengths = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))

def nearest_progress(pos, path_xy, path_s):
    seg_start = path_xy[:-1]
    seg_vec = path_xy[1:] - path_xy[:-1]
    seg_len_sq = np.sum(seg_vec * seg_vec, axis=1)
    seg_len_sq = np.maximum(seg_len_sq, 1e-12)
    pos_row = pos[np.newaxis, :]
    tau = np.sum((pos_row - seg_start) * seg_vec, axis=1) / seg_len_sq
    tau = np.clip(tau, 0.0, 1.0)
    proj = seg_start + tau[:, np.newaxis] * seg_vec
    dist_sq = np.sum((proj - pos_row) ** 2, axis=1)
    best_seg = int(np.argmin(dist_sq))
    seg_len = path_s[best_seg + 1] - path_s[best_seg]
    return path_s[best_seg] + tau[best_seg] * seg_len

def resample_path(path_xy, path_s, sample_s):
    sample_s = np.clip(sample_s, path_s[0], path_s[-1])
    ref_x = np.interp(sample_s, path_s, path_xy[:, 0])
    ref_y = np.interp(sample_s, path_s, path_xy[:, 1])
    ref_xy = np.stack([ref_x, ref_y], axis=-1)
    segment_idx = np.searchsorted(path_s[1:], sample_s, side='right')
    segment_idx = np.clip(segment_idx, 0, len(path_xy) - 2)
    segment_vecs = np.diff(path_xy, axis=0)
    segment_theta = np.arctan2(segment_vecs[:, 1], segment_vecs[:, 0])
    ref_theta = segment_theta[segment_idx]
    return np.column_stack([ref_xy, ref_theta])

def ref_from_path(path_xy, progress_s, path_s):
    # anchor reference at nearest path progress + small offset (stops stalling)
    start_s = progress_s + PATH_LOOKAHEAD
    horizon_s = start_s + REF_SPEED * np.arange(N_STEPS_HORIZON) * dt
    return resample_path(path_xy, path_s, horizon_s)

def mppi_step(x0, ref_traj_xy, cost_map, noise_map, beta_cvar, mu_nominal):
    # sample around nominal clip to wheel-speed limit
    eps = np.random.normal(loc=0.0, scale=SIGMA_MPPI, size=(N_SAMPLES_MPPI, N_STEPS_HORIZON, 2))
    U_samples = np.clip(mu_nominal[np.newaxis, :, :] + eps, -W_MAX, W_MAX)

    X = simulate_batch(x0, U_samples)
    costs = evaluate_cost_batch(
        X, U_samples, ref_traj_xy, cost_map, noise_map, W_COST,
        beta_cvar, mu_nominal,
    )

    # softmax weights over samples
    costs_shifted = costs - np.min(costs)
    weights = np.exp(-costs_shifted / LAMBDA_MPPI)
    weights /= np.sum(weights)

    mu_new = np.sum(weights[:, np.newaxis, np.newaxis] * U_samples, axis=0)
    return mu_new

### RUN
def run_simulation(beta_cvar, cost_map, noise_map, initial_state, goal_state):
    risk_map = cvar_risk_map(beta_cvar, cost_map, noise_map)

    print(f"bi-RRT* for beta={beta_cvar}")
    t0 = timer.time()
    path_xy = bi_rrt_star(risk_map, initial_state[:2], goal_state[:2])
    if path_xy is None:
        raise RuntimeError(f"bi-RRT* failed for beta={beta_cvar}")
    path_s = path_arclength(path_xy)
    print(f"Done in {timer.time()-t0:.2f}s, {len(path_xy)} waypoints")

    current_state = initial_state.copy()
    executed_path = [current_state.copy()]
    u_exec = []
    progress_s = 0.0
    eps = GRID_CELL_SIZE/2

    mu_control = np.zeros((N_STEPS_HORIZON, 2))

    curr_step = 0
    while True:
        if not SIM_UNTIL_GOAL and curr_step >= MAX_STEPS_MPC:
            break

        nearest_s = nearest_progress(current_state[:2], path_xy, path_s)
        progress_s = max(progress_s, nearest_s)
        ref_traj_xy = ref_from_path(path_xy, progress_s, path_s)

        new_mu = mppi_step(
            current_state, ref_traj_xy, cost_map, noise_map, beta_cvar, mu_control)

        optimal_control = new_mu[0, :]
        dx_applied = dynamics(current_state, optimal_control)
        current_state = current_state + dt * dx_applied

        executed_path.append(current_state.copy())
        u_exec.append(optimal_control)

        # warm start - shift nominal forward
        mu_control = np.roll(new_mu, -1, axis=0)
        mu_control[-1, :] = mu_control[-2, :]

        distance_to_goal = np.linalg.norm(current_state[:2] - goal_state[:2])

        if curr_step % 10 == 0:
            print(f"Step {curr_step}: ({current_state[0]:.2f},{current_state[1]:.2f}) "
                  f"dist={distance_to_goal:.2f}m")

        if distance_to_goal < eps:
            print(f"Goal reached at step {curr_step}")
            break

        curr_step += 1

    return np.array(executed_path), np.array(u_exec), path_xy, risk_map

### EXPORT
def save_results(output_dir, beta, executed_path, u_exec,
                            cost_map, noise_map, risk_map, safety_map,
                            initial_state, goal_state, path_xy):
    beta_str = f'{beta:.4f}'.replace('.', 'p')
    filename = os.path.join(output_dir, f"data_beta_{beta_str}.npz")
    terrain_mu_map, terrain_sigma_map = belief_maps(cost_map, noise_map)
    data = {
        'initial_state':    initial_state,
        'goal_state':       goal_state,
        'cost_map':         cost_map,
        'noise_map':        noise_map,
        'terrain_mu_map':   terrain_mu_map,
        'terrain_sigma_map': terrain_sigma_map,
        'risk_map':         risk_map,
        'safety_map':       safety_map,
        'path':             executed_path,
        'controls':         u_exec,
        'beta':             beta,
        'rrt_path_m':       path_xy,
    }
    np.savez(filename, **data)

### monte carlo cost evaluation
# inner - resample per-cell cost along one fixed trajectory
def resample_path_costs(executed_path, cost_map, noise_map, n_resamples=N_RESAMPLES):
    path_xy = executed_path[:, :2]
    segment_lengths = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    path_s = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    distance = path_s[-1]

    if distance <= 1e-12:
        return 0.0, np.zeros(n_resamples), np.zeros(n_resamples)

    unique_s, unique_idx = np.unique(path_s, return_index=True)
    unique_xy = path_xy[unique_idx]
    n_eval = max(2, int(np.ceil(distance / GRID_CELL_SIZE)) + 1)
    eval_s = np.linspace(0.0, distance, n_eval)
    eval_x = np.interp(eval_s, unique_s, unique_xy[:, 0])
    eval_y = np.interp(eval_s, unique_s, unique_xy[:, 1])

    I = np.clip(np.floor(eval_x / GRID_CELL_SIZE).astype(int), 0, RISK_MAP_DIM-1)
    J = np.clip(np.floor(eval_y / GRID_CELL_SIZE).astype(int), 0, RISK_MAP_DIM-1)

    mu = cost_map[I, J]
    sigma = SIGMA_SCALE * np.maximum(noise_map[I, J], 1e-3)

    max_cost = np.zeros(n_resamples)
    accum_cost = np.zeros(n_resamples)
    for k in range(n_resamples):
        samples = np.maximum(np.random.normal(mu, sigma), 0.0)
        max_cost[k] = np.max(samples)
        accum_cost[k] = integrate_piecewise_linear(samples, x=eval_s)

    return distance, max_cost, accum_cost

def plot_boxplots(export_dir, betas, distance_lists, max_cost_lists,
                                cost_lists, cvar_cost_lists):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    labels = [str(b) for b in betas]

    fig.suptitle(
        f'Monte Carlo Evaluation over {N_OUTER_SEEDS} Random Seeds / Environments',
        fontsize=14
    )

    axes[0].boxplot(distance_lists, labels=labels)
    axes[0].set_ylabel('Executed Distance [m]')
    axes[0].set_xlabel(r'Risk Level $\beta$ [-]')
    axes[0].set_title('Start-Goal Path Distance')

    axes[1].boxplot(max_cost_lists, labels=labels)
    axes[1].set_ylabel('Max Cost [-]')
    axes[1].set_xlabel(r'Risk Level $\beta$ [-]')
    axes[1].set_title('Max Cost Along Path')

    axes[2].boxplot(cost_lists, labels=labels)
    axes[2].set_ylabel('Accumulated Cost [- m]')
    axes[2].set_xlabel(r'Risk Level $\beta$ [-]')
    axes[2].set_title('Accumulated Cost Along Path')

    axes[3].boxplot(cvar_cost_lists, labels=labels)
    axes[3].set_ylabel('Accumulated Cost [- m]')
    axes[3].set_xlabel(r'Risk Level $\beta$ [-]')
    axes[3].set_title(rf'CVaR Accumulated Cost ($\beta_{{eval}}={EVAL_CVAR_BETA}$)')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(os.path.join(export_dir, 'distance_risk_tradeoff_boxplots.png'), dpi=300)
    plt.close(fig)

def pick_start_goal(cost_map, rng, min_distance_m=6.0):
    # prefer low-cost endpoints relax threshold if none are far enough apart
    for threshold in [0.15, 0.2, 0.3, 0.4, 0.55, 0.7]:
        candidate_cells = np.argwhere(cost_map < threshold)
        if len(candidate_cells) < 2:
            continue

        for _ in range(10000):
            start_cell = candidate_cells[rng.choice(len(candidate_cells))]
            goal_cell = candidate_cells[rng.choice(len(candidate_cells))]
            dist = np.linalg.norm(start_cell - goal_cell) * GRID_CELL_SIZE
            if dist > min_distance_m:
                return start_cell, goal_cell, threshold

    candidate_cells = np.argwhere(cost_map <= np.quantile(cost_map, 0.35))
    if len(candidate_cells) < 2:
        candidate_cells = np.argwhere(np.ones_like(cost_map, dtype=bool))

    start_cell = candidate_cells[rng.choice(len(candidate_cells))]
    dists = np.linalg.norm(candidate_cells - start_cell[np.newaxis, :], axis=1)
    goal_cell = candidate_cells[int(np.argmax(dists))]
    return start_cell, goal_cell, float(np.quantile(cost_map, 0.35))

def main():
    betas = [0.1, 0.3, 0.5, 0.7, 0.9]
    colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']

    # outer monte carlo - collect one representative value per (seed, beta) here
    outer_distance = {b: [] for b in betas}
    outer_max_cost = {b: [] for b in betas}
    outer_cost = {b: [] for b in betas}
    outer_cvar_cost = {b: [] for b in betas}

    for seed in range(SEED_START, SEED_START + N_OUTER_SEEDS):
        np.random.seed(seed)
        rng = np.random.RandomState(seed)

        export_dir = os.path.join('load-data', 'bi-rrt-star-MPPI-test', f'seed-{seed}')
        os.makedirs(export_dir, exist_ok=True)

        cost_map = generate_cost_map(RISK_MAP_DIM, RISK_MAP_DIM)

        fig, ax = plt.subplots(figsize=(10, 10))

        start_cell, goal_cell, endpoint_threshold = pick_start_goal(cost_map, rng)
        print(f'seed={seed}: endpoint cost threshold={endpoint_threshold:.2f}')

        start_xy = (start_cell + 0.5) * GRID_CELL_SIZE
        goal_xy = (goal_cell + 0.5) * GRID_CELL_SIZE
        initial_state = np.array([start_xy[0], start_xy[1], np.pi])
        goal_state = np.array([goal_xy[0], goal_xy[1], 0.0])

        noise_map = generate_noise_map(
            RISK_MAP_DIM, RISK_MAP_DIM, rng,
            start_xy=initial_state[:2],
            goal_xy=goal_state[:2],
        )

        # noise map
        fig_n, ax_n = plt.subplots(figsize=(8, 8))
        im_n = ax_n.imshow(noise_map.T, origin='lower', extent=[0, GRID_L, 0, GRID_L], cmap='Reds')
        plt.colorbar(im_n, ax=ax_n, label='Epistemic Uncertainty')
        ax_n.set_title('Noise Map (Decoupled Uncertainty)')
        plt.savefig(os.path.join(export_dir, 'noise_map.png'), dpi=300)
        plt.close(fig_n)

        im = ax.imshow(
            cost_map.T,
            origin='lower',
            extent=[GRID_RANGE[0], GRID_RANGE[1], GRID_RANGE[0], GRID_RANGE[1]],
            cmap='Greys',
            alpha=0.6
        )
        plt.colorbar(im, ax=ax, label='Underlying Cost Map')

        beta_safety_map = 0.5
        risk_map_display = cvar_risk_map(beta_safety_map, cost_map, noise_map)
        safety_map = generate_safety_map(risk_map_display)

        for beta_cvar, color in zip(betas, colors):
            print(f"\n--- seed={seed} beta={beta_cvar} ---")
            t0 = timer.time()

            executed_path, u_exec, path_xy, risk_map = run_simulation(
                beta_cvar, cost_map, noise_map, initial_state, goal_state)

            print(f'  Elapsed: {timer.time()-t0:.2f}s')

            # CVaR map for this beta
            fig_r, ax_r = plt.subplots(figsize=(8, 8))
            im_r = ax_r.imshow(risk_map.T, origin='lower', extent=[0, GRID_L, 0, GRID_L], cmap='viridis')
            plt.colorbar(im_r, ax=ax_r, label='CVaR Risk Cost')
            ax_r.plot(path_xy[:, 0], path_xy[:, 1], 'r--', linewidth=2, label='bi-RRT* Path')
            ax_r.set_title(f'CVaR Risk Map (beta={beta_cvar})')
            ax_r.legend()
            beta_filename = str(beta_cvar).replace(".", "p")
            plt.savefig(os.path.join(export_dir, f'cvar_map_beta_{beta_filename}.png'), dpi=300)
            plt.close(fig_r)

            ax.plot(path_xy[:, 0], path_xy[:, 1],
                    linestyle='--', color=color, linewidth=1.0, alpha=0.5,
                    label=f'beta={beta_cvar} bi-RRT*')

            ax.plot(executed_path[:, 0], executed_path[:, 1],
                    linestyle='-', color=color, linewidth=2.5,
                    label=f'beta={beta_cvar} robot')

            save_results(export_dir, beta_cvar, executed_path, u_exec,
                                    cost_map, noise_map, risk_map, safety_map,
                                    initial_state, goal_state, path_xy)

            # inner resampling for this (seed, beta)
            distance, max_cost_samples, cost_samples = (
                resample_path_costs(executed_path, cost_map, noise_map)
            )
            accum_cost_row = cost_samples.reshape(1, -1)
            cvar_eval = upper_tail_cvar(accum_cost_row, EVAL_CVAR_BETA)[0]
            print(f'  inner resample: distance={distance:.2f} m, '
                  f'mean max cost={np.mean(max_cost_samples):.3f}, '
                  f'mean accum cost={np.mean(cost_samples):.3f}, '
                  f'eval cvar accum={cvar_eval:.3f}')

            # one value per (seed, beta) for the boxplots
            outer_distance[beta_cvar].append(distance)
            outer_max_cost[beta_cvar].append(np.mean(max_cost_samples))
            outer_cost[beta_cvar].append(np.mean(cost_samples))
            outer_cvar_cost[beta_cvar].append(cvar_eval)

        ax.plot(initial_state[0], initial_state[1], 'k^', markersize=10, zorder=10, label='Start')
        ax.plot(goal_state[0], goal_state[1], 'k*', markersize=12, zorder=10, label='Goal')

        ax.grid(which='minor', color='#AAAAAA', linestyle='-', linewidth=0.5, alpha=0.7)
        ax.grid(which='major', color='#AAAAAA', linestyle='-', linewidth=0.5, alpha=0.7)
        ax.set_xlim(GRID_RANGE[0], GRID_RANGE[1])
        ax.set_ylim(GRID_RANGE[0], GRID_RANGE[1])
        ax.set_xticks(np.arange(GRID_RANGE[0], GRID_RANGE[1]+GRID_CELL_SIZE, GRID_CELL_SIZE), minor=True)
        ax.set_yticks(np.arange(GRID_RANGE[0], GRID_RANGE[1]+GRID_CELL_SIZE, GRID_CELL_SIZE), minor=True)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(f'Skid-steer robot traversal for varying CVaR confidence level beta (Map seed={seed})')
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(os.path.join(export_dir, 'multi_beta_trajectories.png'))
        plt.close(fig)

    # boxplots generated once after all outer seeds
    top_level_dir = os.path.join('load-data', 'bi-rrt-star-MPPI-test')
    plot_boxplots(
        top_level_dir, betas,
        [outer_distance[b] for b in betas],
        [outer_max_cost[b] for b in betas],
        [outer_cost[b] for b in betas],
        [outer_cvar_cost[b] for b in betas],
    )

if __name__ == "__main__":
    main()
