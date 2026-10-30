import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import truncnorm

def _sample_controls(mu, sigma, size):
    """Fast truncated normal sampler using rejection sampling"""
    out = np.random.normal(loc=mu, scale=sigma, size=size)
    mask = (out < -grid.W_MAX) | (out > grid.W_MAX)
    while mask.any():
        out[mask] = np.random.normal(loc=mu, scale=sigma, size=size)[mask]
        mask = (out < -grid.W_MAX) | (out > grid.W_MAX)
    return out
from bayes_opt import BayesianOptimization

import skid_steer_simple_static as grid


# config

# dt=0.01 is ~7 h per BO run dt=0.05 ~20 min
# grid.dt = 0.05
# grid.T = 10.0
# grid.INITIAL_STATE = np.array([7.8, 6.2, np.pi])
# grid.GOAL_STATE    = np.array([5.9, 0.3, 0])

# horizons
T_EXPERT  = grid.T  # s
T_BO_EVAL = grid.T  # s

# sim budget (s)
SIM_TIME_EXPERT = 100.0
SIM_TIME_BO     = 100.0

# CEM
CEM_EXPERT   = dict(N_SAMPLES=100, N_ELITES=20, N_ITERATIONS=2)
CEM_BO_EVAL  = dict(N_SAMPLES=100, N_ELITES=20, N_ITERATIONS=2)

# objective
W_CTRL_LOSS  = 0   # weight on control effort difference term
# no goal penalty no time decay

# search space
LW2_BOUNDS   = (-8.0, 2.0)  # ~[3e-4, 7.4]
LW3_BOUNDS   = (-2.0, 5.1)  # ~[0.14, 148]
ALPHA_BOUNDS = (0.01, 0.99)


# risk map cache

class RiskMapCache:
    def __init__(self, cost_map, alpha_round=2):
        self.cost_map = cost_map
        self.alpha_round = alpha_round
        self._cache = {}

    def get(self, alpha):
        a = float(np.round(alpha, self.alpha_round))
        if a not in self._cache:
            self._cache[a] = grid.cvar_risk_map(
                grid.RISK_MAP_DIM,
                grid.RISK_MAP_DIM,
                a,
                self.cost_map
            )
        return self._cache[a]


# CEM optimizer

def cem_step_param(
    x0,
    goal_state,
    risk_map,
    mu_init,
    sigma_init,
    W_cost,
    *,
    N_SAMPLES=40,
    N_ELITES=10,
    N_ITERATIONS=1,
    ALPHA_CEM=0.3,
    SIGMA_MIN=0.5,
    debug=False,
):

    mu = mu_init.copy()
    sigma = sigma_init.copy()

    H = int(np.ceil(grid.T / grid.dt))

    best_u_overall = None
    best_cost = np.inf

    for cem_iter in range(N_ITERATIONS):

        U_samples = _sample_controls(mu, sigma, size=(N_SAMPLES, H, 2))

        X_batch = simulate_batch(x0, U_samples)  # (N, H, 3)
        costs   = evaluate_batch(X_batch, U_samples, goal_state,  # (N,)
                                 risk_map, W_cost)

        k_best = int(np.argmin(costs))

        if costs[k_best] < best_cost:
            best_cost = costs[k_best]
            best_u_overall = U_samples[k_best].copy()

        elite_idx = np.argsort(costs)[:N_ELITES]
        elites = U_samples[elite_idx]

        mu_new = np.mean(elites, axis=0)
        sigma_new = np.std(elites, axis=0)

        mu = ALPHA_CEM * mu + (1 - ALPHA_CEM) * mu_new
        sigma = ALPHA_CEM * sigma + (1 - ALPHA_CEM) * sigma_new
        sigma = np.maximum(sigma, SIGMA_MIN)

    return best_u_overall, sigma



def state_to_ij(xy):
    i = int(np.clip(np.floor(xy[0] / grid.GRID_CELL_SIZE), 0, grid.RISK_MAP_DIM - 1))
    j = int(np.clip(np.floor(xy[1] / grid.GRID_CELL_SIZE), 0, grid.RISK_MAP_DIM - 1))
    return i, j


# vectorised simulate + evaluate

def simulate_batch(x0, U_batch):
    """Simulate N skid-steer trajectories simultaneously"""
    N, H, _ = U_batch.shape
    r, L, dt = grid.r, grid.L, grid.dt

    X = np.zeros((N, H, 3))
    X[:, 0, :] = x0

    ul = U_batch[:, :, 0]
    ur = U_batch[:, :, 1]

    for t in range(1, H):
        theta = X[:, t-1, 2]
        sum_u = ul[:, t-1] + ur[:, t-1]
        diff_u = ur[:, t-1] - ul[:, t-1]

        dx     = r / 2 * sum_u * np.cos(theta)
        dy     = r / 2 * sum_u * np.sin(theta)
        dtheta = r / L * diff_u

        X[:, t, 0] = np.clip(X[:, t-1, 0] + dt * dx,
                             grid.GRID_RANGE[0], grid.GRID_RANGE[1])
        X[:, t, 1] = np.clip(X[:, t-1, 1] + dt * dy,
                             grid.GRID_RANGE[0], grid.GRID_RANGE[1])
        X[:, t, 2] = X[:, t-1, 2] + dt * dtheta

    return X


def evaluate_batch(X_batch, U_batch, goal_state, risk_map, W):
    """Evaluate costs for a batch of trajectories — fully vectorised."""
    N, H, _ = X_batch.shape

    # (N, H) grid indices
    i_idx = np.clip(
        np.floor(X_batch[:, :, 0] / grid.GRID_CELL_SIZE).astype(int),
        0, grid.RISK_MAP_DIM - 1
    )
    j_idx = np.clip(
        np.floor(X_batch[:, :, 1] / grid.GRID_CELL_SIZE).astype(int),
        0, grid.RISK_MAP_DIM - 1
    )

    risk_vals = risk_map[i_idx, j_idx]

    dist_vals = np.linalg.norm(
        X_batch[:, :, :2] - goal_state[:2], axis=2
    )

    ctrl_vals = np.sum(U_batch ** 2, axis=2)

    costs = np.sum(
        W[0] * risk_vals + W[1] * dist_vals + W[2] * ctrl_vals,
        axis=1
    )

    costs += W[3] * np.linalg.norm(
        X_batch[:, -1, :2] - goal_state[:2], axis=1
    )

    return costs


# features

def compute_features(path, controls, goal_state, risk_map):

    total_risk = 0.0
    total_dist = 0.0
    total_u2 = 0.0

    for t in range(len(path)):

        i, j = state_to_ij(path[t, :2])

        total_risk += risk_map[i, j]
        total_dist += np.linalg.norm(goal_state[:2] - path[t, :2])

        if t < len(controls):
            total_u2 += np.sum(controls[t] ** 2)

    terminal = np.linalg.norm(path[-1, :2] - goal_state[:2])

    return np.array([total_risk, total_dist, total_u2, terminal])


# MPC rollout with custom W

def run_mpc(
    cost_map,
    W,
    *,
    alpha_init=0.1,
    max_steps=2000,
    cem_cfg=None,
    verbose=True,
    risk_cache=None,
    seed_base=1000,
):

    W = np.asarray(W).reshape(-1)

    if cem_cfg is None:
        cem_cfg = CEM_EXPERT

    print(f'[rollout] computing risk map for alpha={alpha_init}...')
    risk_map = risk_cache.get(alpha_init)
    print(f'[rollout] risk map done, starting loop  max_steps={max_steps}')

    current_state = grid.INITIAL_STATE.copy()

    path = [current_state.copy()]
    controls = []

    H = int(np.ceil(grid.T / grid.dt))

    mu = np.zeros((H, 2))
    sigma = np.ones_like(mu) * grid.W_MAX * 0.5

    # 3 cells (0.3 m) so the robot can't step over it
    eps = grid.GRID_CELL_SIZE

    for step in range(max_steps):

        # check before stepping
        if np.linalg.norm(current_state[:2] - grid.GOAL_STATE[:2]) < eps:
            if verbose:
                print("Goal reached")
            break

        np.random.seed(seed_base + step)

        best_u, new_sigma = cem_step_param(
            current_state,
            grid.GOAL_STATE,
            risk_map,
            mu,
            sigma,
            W,
            **cem_cfg
        )

        u0 = best_u[0]

        dx = grid.dynamics(current_state, u0)
        next_state = current_state + grid.dt * dx
        current_state = next_state

        current_state[0] = np.clip(current_state[0], grid.GRID_RANGE[0], grid.GRID_RANGE[1])
        current_state[1] = np.clip(current_state[1], grid.GRID_RANGE[0], grid.GRID_RANGE[1])

        path.append(current_state.copy())
        controls.append(u0.copy())

        mu = np.roll(best_u, -1, axis=0)
        mu[-1] = mu[-2]

        sigma = np.roll(new_sigma, -1, axis=0)
        sigma[-1] = sigma[-2]

    return np.array(path), np.array(controls), alpha_init, risk_map



def align_paths(path_a, path_b):
    """Pad the shorter of two (N x 3) paths so both have the same length."""
    n_a, n_b = len(path_a), len(path_b)
    max_len = max(n_a, n_b)

    def pad(path, target_len):
        if len(path) >= target_len:
            return path
        pad_rows = np.tile(path[-1], (target_len - len(path), 1))
        return np.vstack([path, pad_rows])

    return pad(path_a, max_len), pad(path_b, max_len)  # TODO: plot traj distance during BO


def trajectory_distance(path_a, path_b):
    """Time-aligned mean per-timestep Euclidean distance (x,y only)."""
    a, b = align_paths(path_a, path_b)
    dists = np.linalg.norm(a[:, :2] - b[:, :2], axis=1)
    return float(np.mean(dists))


# BO over (lw2, lw3, alpha) W[0]=1 W[1]=0 fixed
# loss = padded per-step trajectory distance

def fit_W_to_expert(cost_map, expert_path, expert_u, W2_init, W3_init, alpha_init=0.99):
    """Recover W = [1, 0, W2, W3] from an expert trajectory"""
 
    risk_cache = RiskMapCache(cost_map)
 
    eval_counter = {"n": 0}
 
    # same CEM as the expert so true weights reproduce it
    cem_fast = CEM_BO_EVAL
 
    W_FIXED = np.array([1.0, 0.0, 0.0, 0.0])
 
    def bo_objective(lw2, lw3, alpha):
 
        eval_counter["n"] += 1
        k = eval_counter["n"]
 
        W = W_FIXED.copy()
        W[2] = np.exp(lw2)
        W[3] = np.exp(lw3)
 
        _orig_T = grid.T
        grid.T  = T_BO_EVAL
        path_hat, _, _, _ = run_mpc(
            cost_map,
            W,
            alpha_init=alpha,
            max_steps=int(SIM_TIME_BO / grid.dt),
            verbose=False,
            cem_cfg=cem_fast,
            risk_cache=risk_cache,
            seed_base=1000,  # expert's seed
        )
        grid.T = _orig_T
 
        # trajectory distance
        traj_loss = trajectory_distance(expert_path, path_hat)
 
        # control diff - recover (ul, ur) from consecutive states via skid-steer IK
        #   ur = v/r + L*dth/(2r) ul = v/r - L*dth/(2r)
 
        def reconstruct_controls(path):
            """Reconstruct (ul, ur) from a skid-steer path using inv"""
            r_wheel = grid.r
            L_axle  = grid.L
            dt      = grid.dt
 
            n = len(path) - 1
            if n < 1:
                return np.zeros((0, 2))
 
            dx    = np.diff(path[:, 0]) / dt
            dy    = np.diff(path[:, 1]) / dt
            dth   = np.diff(np.unwrap(path[:, 2])) / dt  # unwrap to avoid 2pi jumps
 
            # signed speed along heading
            theta = path[:-1, 2]
            v = dx * np.cos(theta) + dy * np.sin(theta)
 
            ul = v / r_wheel - (L_axle * dth) / (2 * r_wheel)
            ur = v / r_wheel + (L_axle * dth) / (2 * r_wheel)
 
            return np.stack([ul, ur], axis=1)  # (n, 2)
 
        u_hat_ik = reconstruct_controls(path_hat)
 
        # pad the shorter one with zeros (stopped)
        n_exp = len(expert_u)
        n_hat = len(u_hat_ik)
        n_max = max(n_exp, n_hat)
 
        def pad_controls(u, target):
            if len(u) >= target:
                return u[:target]
            return np.vstack([u, np.zeros((target - len(u), 2))])
 
        u_exp_padded = pad_controls(expert_u,          n_max)
        u_hat_padded = pad_controls(u_hat_ik, n_max)
 
        # mean sq control diff / W_MAX^2
        effort_diff = np.mean(np.sum((u_exp_padded - u_hat_padded) ** 2, axis=1))
        ctrl_loss = effort_diff / (2 * grid.W_MAX ** 2 + 1e-8)
 
        val = traj_loss + W_CTRL_LOSS * ctrl_loss
 
        if k % 5 == 0:
            print(f"eval {k}  W=[1, 0, {np.exp(lw2):.5f}, {np.exp(lw3):.3f}]"
                  f"  alpha={alpha:.3f}  traj_loss={traj_loss:.4f}"
                  f"  ctrl_loss={ctrl_loss:.4f}"
                  f"  obj={val:.4f}")
 
        return -val  # BO maximises
 
    pbounds = {
        "lw2":  LW2_BOUNDS,
        "lw3":  LW3_BOUNDS,
        "alpha": ALPHA_BOUNDS,
    }
 
    optimizer = BayesianOptimization(
        f=bo_objective,
        pbounds=pbounds,
        random_state=42,
        verbose=2,
    )
 
    # seed the GP across alpha first otherwise random init sticks to the extremes
 
    lw2_init = np.log(W2_init)
    lw3_init = np.log(W3_init)
 
    # 1. initial guess
    optimizer.probe(
        params={"lw2": lw2_init, "lw3": lw3_init, "alpha": alpha_init},
        lazy=False,
    )
 
    # 2. alpha sweep at the init weights
    for alpha_seed in [0.1, 0.3, 0.5, 0.7, 0.9]:
        if abs(alpha_seed - alpha_init) > 0.05:
            optimizer.probe(
                params={"lw2": lw2_init, "lw3": lw3_init, "alpha": alpha_seed},
                lazy=False,
            )
 
    # 3. vary W2/W3 at a few alphas
    for alpha_seed, lw2_offset, lw3_offset in [
        (0.3, -2.0,  1.0),
        (0.7,  1.0, -1.0),
        (0.5,  0.0,  0.0),
    ]:
        optimizer.probe(
            params={
                "lw2": np.clip(lw2_init + lw2_offset, -8.0, 2.0),
                "lw3": np.clip(lw3_init + lw3_offset, -2.0, 5.0),
                "alpha": alpha_seed,
            },
            lazy=False,
        )
 
    # 4. BO all budget on guided steps
    optimizer.maximize(init_points=0, n_iter=74)
 
    best_params = optimizer.max["params"]
    W_hat = W_FIXED.copy()
    W_hat[2] = np.exp(best_params["lw2"])
    W_hat[3] = np.exp(best_params["lw3"])
    alpha_hat = best_params["alpha"]
 
    print(f"\nRecovered W:     {W_hat}")
    print(f"Recovered alpha: {alpha_hat:.4f}")
    print(f"Best BO objective (negated loss): {optimizer.max['target']:.4f}")
 
    _orig_T = grid.T
    grid.T  = T_EXPERT
    path_final, _, _, risk_final = run_mpc(
        cost_map,
        W_hat,
        alpha_init=alpha_hat,
        max_steps=int(SIM_TIME_EXPERT / grid.dt),
        risk_cache=risk_cache,
    )
    grid.T = _orig_T
 
    return W_hat, alpha_hat, optimizer, path_final, risk_final
 

def main():

    # start/goal come from skid_steer_simple_static

    # ground truth
    alpha_true = 0.99
    W_true = np.array([1.0, 0.0, 5.0 / 600, 150.0])

    cost_map = grid.generate_cost_map(grid.RISK_MAP_DIM, grid.RISK_MAP_DIM)
    print(f'[main] cost map generated')

    print("=== Running expert rollout ===")
    _orig_T = grid.T
    grid.T  = T_EXPERT
    expert_path, expert_u, _, risk_map_expert = run_mpc(
        cost_map,
        W_true,
        alpha_init=alpha_true,
        max_steps=int(SIM_TIME_EXPERT / grid.dt),
        risk_cache=RiskMapCache(cost_map),
        cem_cfg=CEM_EXPERT,
    )
    grid.T = _orig_T

    # no point recovering if the expert didn't reach the goal
    expert_goal_dist = np.linalg.norm(expert_path[-1, :2] - grid.GOAL_STATE[:2])
    if expert_goal_dist < grid.GRID_CELL_SIZE * 3:
        print(f"Expert reached goal (dist={expert_goal_dist:.3f}m) — proceeding to inverse learning.")
    else:
        print(f"Expert did NOT reach goal (dist={expert_goal_dist:.3f}m). "
              f"Increase SIM_TIME_EXPERT or check weights/alpha before continuing.")
        return

    print("\n=== Inverse learning ===")
    # returns W_hat = [1, 0, w2, w3]
    W2_init   = 5.0 / 600
    W3_init   = 25.0
    alpha_init = 0.99

    W_hat, alpha_hat, bo_optimizer, recovered_path, risk_final = fit_W_to_expert(
        cost_map,
        expert_path,
        expert_u,
        W2_init,
        W3_init,
        alpha_init=alpha_init,
    )

    print(f"\nTrue      W={W_true},  alpha={alpha_true}")
    print(f"Recovered W={W_hat},  alpha={alpha_hat:.4f}")

    plt.figure(figsize=(8, 8))
    plt.imshow(
        risk_final.T,
        origin="lower",
        extent=[grid.GRID_RANGE[0], grid.GRID_RANGE[1], grid.GRID_RANGE[0], grid.GRID_RANGE[1]],
        cmap="Reds",
        alpha=0.6,
    )
    plt.plot(expert_path[:, 0], expert_path[:, 1], "b", label="Expert", linewidth=3)
    plt.plot(recovered_path[:, 0], recovered_path[:, 1], "r--", label="Recovered", linewidth=3)
    plt.scatter(grid.GOAL_STATE[0], grid.GOAL_STATE[1], c="magenta", s=120, label="Goal")
    plt.scatter(expert_path[0, 0], expert_path[0, 1], c="green", s=120, label="Start")
    plt.legend()
    plt.title(f"Expert vs Recovered\nW={W_hat.round(5)}  α={alpha_hat:.4f}")
    plt.gca().set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    main()
