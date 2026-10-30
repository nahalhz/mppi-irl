import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from scipy.stats import truncnorm

import skid_steer_simple_grid as grid


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

        a_i = (-grid.W_MAX - mu) / sigma
        b_i = (grid.W_MAX - mu) / sigma

        U_samples = truncnorm.rvs(
            a=a_i,
            b=b_i,
            loc=mu,
            scale=sigma,
            size=(N_SAMPLES, H, 2)
        )

        costs = np.zeros(N_SAMPLES)

        for k in range(N_SAMPLES):
            X_k = grid.simulate(x0, U_samples[k])
            costs[k] = grid.evaluate_cost(X_k, U_samples[k], goal_state, risk_map)

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


# helper

def state_to_ij(xy):
    i = int(np.clip(np.floor(xy[0] / grid.GRID_CELL_SIZE), 0, grid.RISK_MAP_DIM - 1))
    j = int(np.clip(np.floor(xy[1] / grid.GRID_CELL_SIZE), 0, grid.RISK_MAP_DIM - 1))
    return i, j


# feature computation (aligned with planner cost)

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

    original_cost = grid.evaluate_cost

    def eval_cost_with_W(x, u, goal_state, risk_map):

        total = 0.0

        for t, state in enumerate(x):

            i, j = state_to_ij(state[:2])

            risk = risk_map[i, j]
            dist = np.linalg.norm(goal_state[:2] - state[:2])
            ctrl = np.sum(u[t] ** 2) if t < len(u) else 0.0

            total += W[0] * risk + W[1] * dist + W[2] * ctrl

        total += W[3] * np.linalg.norm(goal_state[:2] - x[-1, :2])

        return total

    grid.evaluate_cost = eval_cost_with_W

    try:

        if cem_cfg is None:
            cem_cfg = dict(N_SAMPLES=100, N_ELITES=20, N_ITERATIONS=2)

        risk_map = risk_cache.get(alpha_init)

        current_state = grid.INITIAL_STATE.copy()

        path = [current_state.copy()]
        controls = []

        H = int(np.ceil(grid.T / grid.dt))

        mu = np.zeros((H, 2))
        sigma = np.ones_like(mu) * grid.W_MAX * 0.5

        eps = grid.GRID_CELL_SIZE / 2

        for step in range(max_steps):

            np.random.seed(seed_base + step)

            best_u, new_sigma = cem_step_param(
                current_state,
                grid.GOAL_STATE,
                risk_map,
                mu,
                sigma,
                **cem_cfg
            )

            u0 = best_u[0]

            dx = grid.dynamics(current_state, u0)

            current_state = current_state + grid.dt * dx

            current_state[0] = np.clip(current_state[0], grid.GRID_RANGE[0], grid.GRID_RANGE[1])
            current_state[1] = np.clip(current_state[1], grid.GRID_RANGE[0], grid.GRID_RANGE[1])

            path.append(current_state.copy())
            controls.append(u0.copy())

            mu = np.roll(best_u, -1, axis=0)
            mu[-1] = mu[-2]

            sigma = np.roll(new_sigma, -1, axis=0)
            sigma[-1] = sigma[-2]

            if np.linalg.norm(current_state[:2] - grid.GOAL_STATE[:2]) < eps:
                if verbose:
                    print("Goal reached")
                break

        return np.array(path), np.array(controls), alpha_init, risk_map

    finally:
        grid.evaluate_cost = original_cost


# inverse learning

def fit_W_to_expert(cost_map, expert_path, expert_u, W_init, alpha_init=0.1):

    risk_cache = RiskMapCache(cost_map)

    base_risk_map = risk_cache.get(alpha_init)

    phi_expert = compute_features(expert_path, expert_u, grid.GOAL_STATE, base_risk_map)

    scale = np.maximum(np.abs(phi_expert), 1e-6)

    phi_expert_n = phi_expert / scale

    x0 = np.log(W_init)

    eval_counter = {"n": 0}

    cem_fast = dict(N_SAMPLES=30, N_ELITES=8, N_ITERATIONS=1)

    def objective(logW):

        eval_counter["n"] += 1
        k = eval_counter["n"]

        W = np.exp(logW)

        path_hat, u_hat, _, _ = run_mpc(
            cost_map,
            W,
            alpha_init=alpha_init,
            max_steps=400,
            verbose=False,
            cem_cfg=cem_fast,
            risk_cache=risk_cache,
            seed_base=5000 + k,
        )

        phi_hat = compute_features(path_hat, u_hat, grid.GOAL_STATE, base_risk_map)

        phi_hat_n = phi_hat / scale

        feat_err = np.linalg.norm(phi_hat_n - phi_expert_n)

        goal_dist = np.linalg.norm(path_hat[-1, :2] - grid.GOAL_STATE[:2])

        val = feat_err + 2000 * goal_dist

        if k % 5 == 0:
            print("eval", k, "W", W, "obj", val)

        return val

    res = minimize(
        objective,
        x0,
        method="Powell",
        options={"maxiter": 80, "disp": True},
    )

    W_hat = np.exp(res.x)

    path_final, u_final, alpha_final, risk_final = run_mpc(
        cost_map,
        W_hat,
        alpha_init=alpha_init,
        max_steps=2000,
        risk_cache=risk_cache,
    )

    return W_hat, res, path_final, alpha_final, risk_final


def main():
    

    alpha0 = 0.99

    W_true = np.array([1,0,5.0/600,30])

    cost_map = grid.generate_cost_map(grid.RISK_MAP_DIM, grid.RISK_MAP_DIM)

    print("Running expert")

    expert_path, expert_u, _, risk_map = run_mpc(
        cost_map,
        W_true,
        alpha_init=alpha0,
        max_steps=2000,
        risk_cache=RiskMapCache(cost_map),
        cem_cfg=dict(N_SAMPLES=100, N_ELITES=20, N_ITERATIONS=4),
    )

    print("Inverse learning")

    W_init = np.array([1,0.01,5.0/600,25])

    W_hat, res, recovered_path, _, risk_final = fit_W_to_expert(
        cost_map,
        expert_path,
        expert_u,
        W_init,
        alpha_init=alpha0,
    )

    print("Recovered W:", W_hat)

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

    plt.title("Expert vs Recovered")

    plt.gca().set_aspect("equal")

    plt.show()


if __name__ == "__main__":
    main()
