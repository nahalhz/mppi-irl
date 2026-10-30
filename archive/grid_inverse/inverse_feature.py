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


# CEM - returns the best sequence not mu

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
    """One CEM step"""
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
        if float(costs[k_best]) < best_cost:
            best_cost = float(costs[k_best])
            best_u_overall = U_samples[k_best].copy()

        if debug:
            best = float(np.min(costs))
            median = float(np.median(costs))
            spread = float(np.max(costs) - np.min(costs))
            print(f"    [CEM iter {cem_iter}] best={best:.3f} median={median:.3f} spread={spread:.3f}")

        elite_idx = np.argsort(costs)[:N_ELITES]
        elites = U_samples[elite_idx]

        mu_new = np.mean(elites, axis=0)
        sigma_new = np.std(elites, axis=0)

        mu = ALPHA_CEM * mu + (1 - ALPHA_CEM) * mu_new
        sigma = ALPHA_CEM * sigma + (1 - ALPHA_CEM) * sigma_new
        sigma = np.maximum(sigma, SIGMA_MIN)

    return best_u_overall, sigma


# features

def state_to_ij(xy):
    i = int(np.clip(np.floor(xy[0] / grid.GRID_CELL_SIZE), 0, grid.RISK_MAP_DIM - 1))
    j = int(np.clip(np.floor(xy[1] / grid.GRID_CELL_SIZE), 0, grid.RISK_MAP_DIM - 1))
    return i, j


def compute_features(path, controls, goal_state, risk_map):
    total_risk = 0.0
    total_u2 = 0.0
    max_risk = -np.inf

    for t in range(len(path)):
        i, j = state_to_ij(path[t, :2])
        r = float(risk_map[i, j])
        total_risk += r
        max_risk = max(max_risk, r)

        if t < len(controls):
            total_u2 += float(np.sum(controls[t] ** 2))

    terminal = float(np.linalg.norm(path[-1, :2] - goal_state[:2]))

    if len(path) >= 2:
        diffs = path[1:, :2] - path[:-1, :2]
        path_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
    else:
        path_len = 0.0

    return np.array([total_risk, total_u2, terminal, path_len, max_risk], dtype=float)


# MPC rollout with custom W (+ alpha unstuck schedule)

def run_mpc(
    cost_map,
    W,
    *,
    alpha_init=0.1,
    max_steps=2000,
    check_stuck_iters=100,
    alpha_step=0.1,
    alpha_max=1.0,
    cem_cfg=None,
    verbose=True,
    risk_cache=None,
    cem_debug=False,
    seed_base=1000,
):
    """Closed-loop MPC rollout using W and alpha schedule like your forward script."""
    W = np.asarray(W, dtype=float).reshape(-1)
    if W.shape[0] != 4:
        raise ValueError("Expected W shape (4,), i.e. [w_risk, w_dist, w_u, w_terminal].")

    original_cost = grid.evaluate_cost

    def eval_cost_with_W(x, u, goal_state, risk_map):
        total = 0.0
        for t, state in enumerate(x):
            i, j = state_to_ij(state[:2])
            risk = float(risk_map[i, j])
            dist = float(np.linalg.norm(goal_state[:2] - state[:2]))
            ctrl = float(np.sum(u[t] ** 2)) if t < len(u) else 0.0
            total += W[0] * risk + W[1] * dist + W[2] * ctrl
        total += W[3] * float(np.linalg.norm(goal_state[:2] - x[-1, :2]))
        return float(total)

    grid.evaluate_cost = eval_cost_with_W

    try:
        if cem_cfg is None:
            cem_cfg = dict(N_SAMPLES=100, N_ELITES=20, N_ITERATIONS=2)

        alpha = float(alpha_init)
        if risk_cache is None:
            risk_map = grid.cvar_risk_map(grid.RISK_MAP_DIM, grid.RISK_MAP_DIM, alpha, cost_map)
        else:
            risk_map = risk_cache.get(alpha)

        current_state = grid.INITIAL_STATE.copy()
        path = [current_state.copy()]
        controls = []

        H = int(np.ceil(grid.T / grid.dt))
        mu = np.zeros((H, 2))
        sigma = np.ones_like(mu) * grid.W_MAX * 0.5

        eps = grid.GRID_CELL_SIZE / 2

        for step in range(max_steps):
            np.random.seed(seed_base + step)

            # stuck detection + alpha schedule
            if step % check_stuck_iters == 0 and step >= check_stuck_iters:
                prev = path[step - check_stuck_iters]
                if (abs(current_state[0] - prev[0]) <= eps * 2) and (abs(current_state[1] - prev[1]) <= eps * 2):
                    old_alpha = alpha
                    alpha = min(alpha + alpha_step, alpha_max)
                    if alpha > old_alpha:
                        if verbose:
                            print(f"Alpha updated to {alpha:.1f}")
                        risk_map = risk_cache.get(alpha) if risk_cache else grid.cvar_risk_map(
                            grid.RISK_MAP_DIM, grid.RISK_MAP_DIM, alpha, cost_map
                        )
                    else:
                        if verbose:
                            print(f"Robot stuck even at max alpha={alpha:.1f}")
                        break

            best_u, new_sigma = cem_step_param(
                current_state,
                grid.GOAL_STATE,
                risk_map,
                mu,
                sigma,
                debug=cem_debug,
                **cem_cfg
            )

            u0 = best_u[0]
            dx = grid.dynamics(current_state, u0)
            current_state = current_state + grid.dt * dx
            current_state[0] = np.clip(current_state[0], grid.GRID_RANGE[0], grid.GRID_RANGE[1])
            current_state[1] = np.clip(current_state[1], grid.GRID_RANGE[0], grid.GRID_RANGE[1])

            path.append(current_state.copy())
            controls.append(u0.copy())

            # warm start shift
            mu = np.roll(best_u, -1, axis=0)
            mu[-1] = mu[-2]
            sigma = np.roll(new_sigma, -1, axis=0)
            sigma[-1] = sigma[-2]

            if np.linalg.norm(current_state[:2] - grid.GOAL_STATE[:2]) < eps:
                if verbose:
                    print(f"Goal reached at step {step}")
                break

        return np.array(path), np.array(controls), alpha, risk_map

    finally:
        grid.evaluate_cost = original_cost


# inverse fit by feature matching

def fit_W_to_expert(cost_map, expert_path, expert_u, W_init, alpha_init=0.1):
    """Black-box optimize W (positive) so that recovered rollout's FEATURES match expert."""
    risk_cache = RiskMapCache(cost_map)

    # normalize features with expert features under expert-final risk map
    phi_expert = None
    scale = None
    phi_expert_n = None

    # fix the feature scale with the alpha_init risk map
    base_risk_map = risk_cache.get(alpha_init)
    phi_expert = compute_features(expert_path, expert_u, grid.GOAL_STATE, base_risk_map)
    scale = np.maximum(np.abs(phi_expert), 1e-6)
    phi_expert_n = phi_expert / scale

    x0 = np.log(np.maximum(np.asarray(W_init, dtype=float), 1e-6))
    eval_counter = {"n": 0}

    cem_fast = dict(N_SAMPLES=30, N_ELITES=8, N_ITERATIONS=1)
    max_steps_fit = 400

    def objective(logW):
        eval_counter["n"] += 1
        k = eval_counter["n"]
        W = np.exp(logW)

        t0 = time.time()
        path_hat, u_hat, alpha_hat, risk_hat = run_mpc(
            cost_map,
            W,
            alpha_init=alpha_init,
            max_steps=max_steps_fit,
            verbose=False,
            cem_cfg=cem_fast,
            risk_cache=risk_cache,
            cem_debug=False,
            seed_base=5000 + 10 * k,  # deterministic per eval
        )
        t_roll = time.time() - t0

        if len(path_hat) < 5:
            if k % 5 == 0:
                print(f"[eval {k:03d}] degenerate rollout len={len(path_hat)} time={t_roll:.2f}s W={W}")
            return 1e6

        # features on the risk map used in that rollout
        phi_hat = compute_features(path_hat, u_hat, grid.GOAL_STATE, risk_hat)
        phi_hat_n = phi_hat / scale

        feat_err = float(np.linalg.norm(phi_hat_n - phi_expert_n))
        goal_dist = float(np.linalg.norm(path_hat[-1, :2] - grid.GOAL_STATE[:2]))

        # strong goal penalty
        if goal_dist > 1.0:
            val = 1e4 + 1e4 * goal_dist + 10.0 * feat_err
        else:
            val = feat_err + 50.0 * goal_dist

        if (k <= 5) or (k % 5 == 0):
            print(
                f"[eval {k:03d}] obj={val:9.4f} feat_err={feat_err:7.4f} "
                f"goal={goal_dist:6.3f} alpha={alpha_hat:4.1f} time={t_roll:5.2f}s W={W}"
            )

        return val

    res = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={"maxiter": 60, "xatol": 1e-2, "fatol": 1e-2, "disp": True},
    )

    W_hat = np.exp(res.x)

    # final rollout (stronger CEM)
    cem_full = dict(N_SAMPLES=100, N_ELITES=20, N_ITERATIONS=2)
    path_final, u_final, alpha_final, risk_final = run_mpc(
        cost_map,
        W_hat,
        alpha_init=alpha_init,
        max_steps=2000,
        verbose=True,
        cem_cfg=cem_full,
        risk_cache=risk_cache,
        cem_debug=False,
        seed_base=9999,
    )

    return W_hat, res, path_final, alpha_final, risk_final


# main

def main():
    alpha0 = 0.1
    W_true = np.array([1.0, 0.0, 5.0 / 600.0, 25.0])

    cost_map = grid.generate_cost_map(grid.RISK_MAP_DIM, grid.RISK_MAP_DIM)

    print("Running expert...")
    expert_path, expert_u, alpha_final, risk_map_final = run_mpc(
        cost_map,
        W_true,
        alpha_init=alpha0,
        max_steps=2000,
        verbose=True,
        risk_cache=RiskMapCache(cost_map),
        cem_cfg=dict(N_SAMPLES=100, N_ELITES=20, N_ITERATIONS=2),
        seed_base=1000,
    )

    expert_goal_dist = np.linalg.norm(expert_path[-1, :2] - grid.GOAL_STATE[:2])
    print(f"Expert alpha_final: {alpha_final}")
    print(f"Expert final dist:  {expert_goal_dist:.4f}")

    if expert_goal_dist > grid.GRID_CELL_SIZE:
        print("\nWARNING: Expert did not reliably reach goal. Increase expert CEM or terminal weight.")
        print("Try cem_cfg: N_SAMPLES=200, N_ELITES=30, N_ITERATIONS=3 (for expert only).")

    print("\nFitting inverse model...")
    W_init = np.array([0.8, 0.1, 0.01, 10.0])

    W_hat, res, recovered_path, alpha_hat, risk_final = fit_W_to_expert(
        cost_map, expert_path, expert_u, W_init, alpha_init=alpha0
    )

    print("\nRecovered W:", W_hat)
    print("Recovered alpha_final:", alpha_hat)

    # plot
    plt.figure(figsize=(8, 8))
    plt.imshow(
        risk_final.T,
        origin="lower",
        extent=[grid.GRID_RANGE[0], grid.GRID_RANGE[1], grid.GRID_RANGE[0], grid.GRID_RANGE[1]],
        cmap="Reds",
        alpha=0.6,
    )
    plt.plot(expert_path[:, 0], expert_path[:, 1], "b", linewidth=3, label="Expert")
    plt.plot(recovered_path[:, 0], recovered_path[:, 1], "r--", linewidth=3, label="Recovered")
    plt.scatter(grid.GOAL_STATE[0], grid.GOAL_STATE[1], c="magenta", s=120, label="Goal")
    plt.scatter(expert_path[0, 0], expert_path[0, 1], c="green", s=120, label="Start")
    plt.legend()
    plt.title("Expert vs Recovered")
    plt.xlim(grid.GRID_RANGE)
    plt.ylim(grid.GRID_RANGE)
    plt.gca().set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    main()
