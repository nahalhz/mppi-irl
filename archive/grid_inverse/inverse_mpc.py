"""MaxEnt IRL demo on the simple grid."""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import truncnorm
from scipy.special import logsumexp

import skid_steer_simple_grid as grid


# risk map cache

class RiskMapCache:
    def __init__(self, cost_map, alpha_round=2):
        self.cost_map = cost_map
        self.alpha_round = alpha_round
        self._cache = {}

    def get(self, alpha: float) -> np.ndarray:
        a = float(np.round(alpha, self.alpha_round))
        if a not in self._cache:
            self._cache[a] = grid.cvar_risk_map(
                grid.RISK_MAP_DIM,
                grid.RISK_MAP_DIM,
                a,
                self.cost_map
            )
        return self._cache[a]


# CEM - returns the best sample not mu

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
    """Features (5): 0) total_risk      sum_t risk_map[x_t] 1) total_u2        sum_t ||u_t||^2 2) terminal_dist   ||x_T - goal|| (xy) 3) path_length     sum_t ||x_{t+1}-x_t|| (xy) 4) max_risk        max_t risk_map[x_t]"""
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
    """Closed-loop MPC rollout using W and alpha schedule (unstuck)."""
    W = np.asarray(W, dtype=float).reshape(-1)
    if W.shape[0] != 5:
        raise ValueError(
            "In this MaxEnt script, W must be length-5 (matching compute_features). "
            "Use W=[w_risk,w_u2,w_terminal,w_pathlen,w_maxrisk]."
        )

    # monkey patch - evaluate_cost(x,u,goal,risk_map) := W · phi(x,u)
    original_cost = grid.evaluate_cost

    def eval_cost_with_W(x, u, goal_state, risk_map):
        phi = compute_features(x, u, goal_state, risk_map)
        return float(np.dot(W, phi))

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


# MaxEnt IRL

def fit_W_maxent(
    cost_map,
    expert_path,
    expert_u,
    W_init,
    *,
    alpha_init=0.1,
    n_iters=20,
    lr=0.1,
    K=20,  # samples per iter
    max_steps_sample=350,
    cem_fast=None,
    verbose=True,
):
    """Maximum Entropy IRL on the SAME map:"""
    risk_cache = RiskMapCache(cost_map)

    # features normalised on a base map
    base_risk_map = risk_cache.get(alpha_init)

    phi_expert_raw = compute_features(expert_path, expert_u, grid.GOAL_STATE, base_risk_map)
    scale = np.maximum(np.abs(phi_expert_raw), 1e-6)
    phi_expert = phi_expert_raw / scale

    W = np.asarray(W_init, dtype=float).reshape(-1)
    if W.shape[0] != 5:
        raise ValueError("W_init must be length-5 to match compute_features().")

    W = np.maximum(W, 1e-6)

    if cem_fast is None:
        cem_fast = dict(N_SAMPLES=25, N_ELITES=6, N_ITERATIONS=1)

    for it in range(n_iters):
        phis = []
        goals = []
        alphas = []
        t0 = time.time()

        # sample trajectories under current W
        for k in range(K):
            seed_base = 20000 + it * 1000 + k * 50

            path_hat, u_hat, alpha_hat, risk_hat = run_mpc(
                cost_map,
                W,
                alpha_init=alpha_init,
                max_steps=max_steps_sample,
                verbose=False,
                cem_cfg=cem_fast,
                risk_cache=risk_cache,
                cem_debug=False,
                seed_base=seed_base,
            )

            if len(path_hat) < 5:
                continue

            # features on the risk map at the end of that rollout (alpha schedule)
            phi_hat_raw = compute_features(path_hat, u_hat, grid.GOAL_STATE, risk_hat)
            phi_hat = phi_hat_raw / scale

            phis.append(phi_hat)
            goals.append(float(np.linalg.norm(path_hat[-1, :2] - grid.GOAL_STATE[:2])))
            alphas.append(float(alpha_hat))

        if len(phis) < max(3, K // 4):
            print(f"[iter {it:02d}] too few valid samples ({len(phis)}/{K}) -> skipping update")
            continue

        phis = np.asarray(phis, dtype=float)
        goals = np.asarray(goals, dtype=float)

        # MaxEnt weights
        # cost = W^T phi
        costs = phis @ W

        # soft goal constraint - penalty grows once goal_dist > 1
        goal_pen = np.where(goals <= 1.0, 0.0, 50.0 * (goals - 1.0))
        energies = costs + goal_pen

        logZ = logsumexp(-energies)
        probs = np.exp(-energies - logZ)  # (Nsamples,)

        phi_model = np.sum(probs[:, None] * phis, axis=0)
        grad = phi_model - phi_expert

        # update
        W = W - lr * grad
        W = np.maximum(W, 1e-6)

        dt_it = time.time() - t0

        if verbose:
            print(
                f"[iter {it:02d}] |grad|={np.linalg.norm(grad):7.4f}  "
                f"avg_goal={goals.mean():6.3f}  "
                f"best_goal={goals.min():6.3f}  "
                f"time={dt_it:5.2f}s  W={W}"
            )

    return W


# plot

def plot_compare(expert_path, recovered_path, risk_map_to_show):
    plt.figure(figsize=(8, 8))
    plt.imshow(
        risk_map_to_show.T,
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
    plt.title("Expert vs Recovered (MaxEnt IRL)")
    plt.xlim(grid.GRID_RANGE)
    plt.ylim(grid.GRID_RANGE)
    plt.gca().set_aspect("equal")
    plt.show()


# main

def main():
    alpha0 = 0.1

    # W = [risk, u^2, terminal, path_len, max_risk]
    W_true = np.array([1.0, 0.01, 25.0, 0.2, 0.5], dtype=float)

    cost_map = grid.generate_cost_map(grid.RISK_MAP_DIM, grid.RISK_MAP_DIM)
    risk_cache = RiskMapCache(cost_map)

    # expert rollout
    print("Running expert...")
    expert_path, expert_u, alpha_final, risk_map_final = run_mpc(
        cost_map,
        W_true,
        alpha_init=alpha0,
        max_steps=2000,
        verbose=True,
        risk_cache=risk_cache,
        cem_cfg=dict(N_SAMPLES=120, N_ELITES=25, N_ITERATIONS=2),
        seed_base=1000,
    )

    expert_goal_dist = float(np.linalg.norm(expert_path[-1, :2] - grid.GOAL_STATE[:2]))
    print(f"Expert alpha_final: {alpha_final}")
    print(f"Expert final dist:  {expert_goal_dist:.4f}")

    if expert_goal_dist > 0.5:
        print("\nWARNING: Expert didn’t reach the goal reliably.")
        print("Try increasing expert terminal weight or stronger expert CEM (more samples/iterations).")
        # carry on but IRL won't do well

    # MaxEnt IRL (recover W)
    print("\nFitting inverse model (MaxEnt IRL)...")
    W_init = np.array([0.8, 0.02, 10.0, 0.2, 0.2], dtype=float)

    W_hat = fit_W_maxent(
        cost_map,
        expert_path,
        expert_u,
        W_init,
        alpha_init=alpha0,
        n_iters=20,
        lr=0.2,
        K=20,
        max_steps_sample=350,
        cem_fast=dict(N_SAMPLES=25, N_ELITES=6, N_ITERATIONS=1),
        verbose=True,
    )

    print("\nRecovered W:", W_hat)

    # rollout recovered policy
    print("\nRolling out recovered policy...")
    recovered_path, recovered_u, alpha_hat_final, risk_hat_final = run_mpc(
        cost_map,
        W_hat,
        alpha_init=alpha0,
        max_steps=2000,
        verbose=True,
        risk_cache=risk_cache,
        cem_cfg=dict(N_SAMPLES=120, N_ELITES=25, N_ITERATIONS=2),
        seed_base=5000,
    )

    rec_goal_dist = float(np.linalg.norm(recovered_path[-1, :2] - grid.GOAL_STATE[:2]))
    print(f"Recovered alpha_final: {alpha_hat_final}")
    print(f"Recovered final dist:  {rec_goal_dist:.4f}")

    plot_compare(expert_path, recovered_path, risk_hat_final)


if __name__ == "__main__":
    main()
