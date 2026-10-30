import numpy as np
import matplotlib.pyplot as plt
from pandas.core.indexing import T
from scipy.optimize import minimize
import time

from scipy.stats import truncnorm

import skid_steer_simple_grid as grid

class RiskMapCache:
    """Cache risk_map(alpha) so we don't recompute CVaR grids repeatedly."""
    def __init__(self, cost_map, x_dim, y_dim, make_risk_map_fn, alpha_round=2):
        self.cost_map = cost_map
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.make_risk_map_fn = make_risk_map_fn
        self.alpha_round = alpha_round
        self._cache = {}

    def get(self, alpha):
        a = float(np.round(alpha, self.alpha_round))
        if a not in self._cache:
            self._cache[a] = self.make_risk_map_fn(self.x_dim, self.y_dim, a, self.cost_map)
        return self._cache[a]


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
    """One CEM step with optional debug printing"""

    mu = mu_init
    sigma = sigma_init

    for cem_iter in range(N_ITERATIONS):
        a_i = (-grid.W_MAX - mu) / sigma
        b_i = (grid.W_MAX - mu) / sigma

        U_samples = truncnorm.rvs(
            a=a_i,
            b=b_i,
            loc=mu,
            scale=sigma,
            size=(N_SAMPLES, int(np.ceil(grid.T / grid.dt)), 2)
        )

        costs = np.zeros(N_SAMPLES)

        for k in range(N_SAMPLES):
            X_k = grid.simulate(x0, U_samples[k])
            costs[k] = grid.evaluate_cost(X_k, U_samples[k], goal_state, risk_map)

        # CEM debug print
        if debug:
            best = float(np.min(costs))
            median = float(np.median(costs))
            worst = float(np.max(costs))
            spread = worst - best
            print(
                f"    [CEM iter {cem_iter}] "
                f"best={best:.3f}  median={median:.3f}  "
                f"spread={spread:.3f}"
            )

        elite_idx = np.argsort(costs)[:N_ELITES]
        elites = U_samples[elite_idx]

        mu_new = np.mean(elites, axis=0)
        sigma_new = np.std(elites, axis=0)

        mu = ALPHA_CEM * mu + (1 - ALPHA_CEM) * mu_new
        sigma = ALPHA_CEM * sigma + (1 - ALPHA_CEM) * sigma_new
        sigma = np.maximum(sigma, SIGMA_MIN)

    return mu, sigma


# MPC rollout with W and alpha schedule
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
):
    """Closed-loop MPC rollout using a weight vector W = [w_risk, w_dist, w_u, w_terminal] and alpha that increases if stuck (like your forward script)."""

    W = np.asarray(W, dtype=float).reshape(-1)
    assert W.shape[0] == 4, "Expected W = [w_risk, w_dist, w_u, w_terminal]"

    # patch evaluate_cost locally
    original_cost = grid.evaluate_cost

    def eval_cost_with_W(x, u, goal_state, risk_map):
        total = 0.0
        for t, state in enumerate(x):
            i = int(np.clip(np.floor(state[0] / grid.GRID_CELL_SIZE), 0, grid.RISK_MAP_DIM - 1))
            j = int(np.clip(np.floor(state[1] / grid.GRID_CELL_SIZE), 0, grid.RISK_MAP_DIM - 1))

            risk = risk_map[i, j]
            dist = np.linalg.norm(goal_state[:2] - state[:2])
            ctrl = np.sum(u[t, :] ** 2) if t < len(u) else 0.0

            total += W[0] * risk + W[1] * dist + W[2] * ctrl

        total += W[3] * np.linalg.norm(goal_state[:2] - x[-1, :2])
        return float(total)

    grid.evaluate_cost = eval_cost_with_W

    try:
        alpha = float(alpha_init)
        if risk_cache is None:
            risk_map = grid.cvar_risk_map(grid.RISK_MAP_DIM, grid.RISK_MAP_DIM, alpha, cost_map)
        else:
            risk_map = risk_cache.get(alpha)

        current_state = grid.INITIAL_STATE.copy()
        path = [current_state.copy()]
        controls = []

        mu_control = np.zeros((int(np.ceil(grid.T / grid.dt)), 2))
        sigma_control = np.ones_like(mu_control) * grid.W_MAX * 0.5

        eps = grid.GRID_CELL_SIZE / 2

        if cem_cfg is None:
            cem_cfg = dict(N_SAMPLES=100, N_ELITES=20, N_ITERATIONS=2)

        for step in range(max_steps):
            # stuck detection
            if step % check_stuck_iters == 0 and step >= check_stuck_iters:
                prev = path[step - check_stuck_iters]
                if (abs(current_state[0] - prev[0]) <= eps * 2) and (abs(current_state[1] - prev[1]) <= eps * 2):
                    old_alpha = alpha
                    alpha = min(alpha + alpha_step, alpha_max)
                    if alpha > old_alpha:
                        if verbose:
                            print(f"Alpha updated to {alpha:.1f}")
                        risk_map = risk_cache.get(alpha) if risk_cache is not None else grid.cvar_risk_map(
                            grid.RISK_MAP_DIM, grid.RISK_MAP_DIM, alpha, cost_map
                        )
                    else:
                        if verbose:
                            print(f"Robot stuck even at max alpha={alpha}")
                        break

            # CEM step
            best_u_plan, new_sigma = cem_step_param(
                x0=current_state,
                goal_state=grid.GOAL_STATE,
                risk_map=risk_map,
                mu_init=mu_control,
                sigma_init=sigma_control,
                debug=True,
                **cem_cfg
            )

            u0 = best_u_plan[0, :]
            dx = grid.dynamics(current_state, u0)
            current_state = current_state + grid.dt * dx

            current_state[0] = np.clip(current_state[0], grid.GRID_RANGE[0], grid.GRID_RANGE[1])
            current_state[1] = np.clip(current_state[1], grid.GRID_RANGE[0], grid.GRID_RANGE[1])

            path.append(current_state.copy())
            controls.append(u0.copy())

            # shift warm start
            mu_control = np.roll(best_u_plan, -1, axis=0)
            mu_control[-1, :] = mu_control[-2, :]

            sigma_control = np.roll(new_sigma, -1, axis=0)
            sigma_control[-1, :] = new_sigma[-2, :]

            if np.linalg.norm(current_state[:2] - grid.GOAL_STATE[:2]) < eps:
                if verbose:
                    print(f"Goal reached at step {step}")
                break

        return np.array(path), np.array(controls), alpha, risk_map

    finally:
        grid.evaluate_cost = original_cost


# inverse fit W by black-box optimization
from scipy.optimize import minimize

def fit_W_to_expert_optimized(cost_map, expert_path, W_init, alpha_init=0.1):
    # cache risk maps once
    risk_cache = RiskMapCache(
        cost_map=cost_map,
        x_dim=grid.RISK_MAP_DIM,
        y_dim=grid.RISK_MAP_DIM,
        make_risk_map_fn=grid.cvar_risk_map,
        alpha_round=2
    )

    # subsample the expert (drops 80% of points)
    stride = 5
    expert_xy = expert_path[::stride, :2].copy()

    # log-space keeps W positive
    x0 = np.log(np.maximum(np.asarray(W_init, dtype=float), 1e-6))

    cem_fast = dict(N_SAMPLES=25, N_ELITES=5, N_ITERATIONS=1)

    # shorter rollouts while fitting
    max_steps_fit = int(min(2000, max(400, len(expert_path))))

    eval_counter = {"n": 0}

    def objective(logW):
        eval_counter["n"] += 1
        k = eval_counter["n"]

        W = np.exp(logW)

        np.random.seed(1234 + k)

        t0 = time.time()
        path_hat, _, alpha_final, _ = run_mpc(
            cost_map,
            W,
            alpha_init=alpha_init,
            max_steps=max_steps_fit,
            verbose=False,
            cem_cfg=cem_fast,
            risk_cache=risk_cache,
        )
        t_roll = time.time() - t0

        if len(path_hat) < 5:
            if k % 2 == 0:
                print(f"[eval {k:03d}] degenerate rollout (len={len(path_hat)})  time={t_roll:.2f}s  W={W}")
            return 1e6

        # compare subsampled geometry
        path_xy = path_hat[::stride, :2]
        n = min(len(expert_xy), len(path_xy))
        traj_err = float(np.mean(np.linalg.norm(expert_xy[:n] - path_xy[:n], axis=1)))

        goal_dist = float(np.linalg.norm(path_hat[-1, :2] - grid.GOAL_STATE[:2]))

        if goal_dist > 2.0:
            val = 1e5 + 1e4 * goal_dist
        else:
            penalty = 50.0 * goal_dist if goal_dist > 0.5 else 0.0
            alpha_penalty = 1.0 if alpha_final < 0.2 else 0.0
            val = traj_err + penalty + alpha_penalty

        if (k <= 5) or (k % 5 == 0):
            print(
                f"[eval {k:03d}] obj={val:8.3f}  traj_err={traj_err:6.3f}  "
                f"goal_dist={goal_dist:6.3f}  alpha_final={alpha_final:4.1f}  "
                f"time={t_roll:5.2f}s  W={W}"
            )

        return val


    res = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={
            "maxiter": 50,  # each iter is expensive
            "xatol": 1e-2,
            "fatol": 1e-2,
            "disp": True
        },
    )

    W_hat = np.exp(res.x)

    # final rollout (stronger CEM, longer)
    cem_full = dict(N_SAMPLES=100, N_ELITES=20, N_ITERATIONS=2)
    path_final, u_final, alpha_final, risk_final = run_mpc(
        cost_map,
        W_hat,
        alpha_init=alpha_init,
        max_steps=2000,
        verbose=True,
        cem_cfg=cem_full,
        risk_cache=risk_cache,
    )

    return W_hat, res, path_final, alpha_final, risk_final


# plot
def plot_compare(expert_x, recovered_x, risk_map):
    fig, ax = plt.subplots(figsize=(9, 9))

    im = ax.imshow(
        risk_map.T,
        origin="lower",
        extent=[grid.GRID_RANGE[0], grid.GRID_RANGE[1], grid.GRID_RANGE[0], grid.GRID_RANGE[1]],
        cmap="Reds",
        alpha=0.6,
    )
    plt.colorbar(im, ax=ax, label="CVaR Risk")

    ax.plot(expert_x[:, 0], expert_x[:, 1], "b", linewidth=3, label="Expert")
    ax.plot(recovered_x[:, 0], recovered_x[:, 1], "r--", linewidth=3, label="Recovered")

    ax.scatter(expert_x[0, 0], expert_x[0, 1], c="green", s=120, label="Start")
    ax.scatter(grid.GOAL_STATE[0], grid.GOAL_STATE[1], c="magenta", s=120, label="Goal")

    ax.set_title("Expert vs Recovered Trajectory (same map)")
    ax.set_xlim(grid.GRID_RANGE)
    ax.set_ylim(grid.GRID_RANGE)
    ax.set_aspect("equal")
    ax.legend()
    plt.show()


# main
def main():
    alpha0 = 0.1
    W_true = np.array([1.0, 0.0, 5.0/600.0, 25.0])

    cost_map = grid.generate_cost_map(grid.RISK_MAP_DIM, grid.RISK_MAP_DIM)

    print("Running expert...")
    expert_path, expert_u, alpha_final, risk_map_final = run_mpc(
        cost_map, W_true, alpha_init=alpha0, max_steps=2000, verbose=True
    )

    print("Expert alpha_final:", alpha_final)
    print("Expert final dist:", np.linalg.norm(expert_path[-1,:2] - grid.GOAL_STATE[:2]))

    print("\nFitting W to expert...")
    W_init = np.array([0.8, 0.1, 0.01, 10.0])

    W_hat, res, recovered_path, alpha_hat_final, risk_final = fit_W_to_expert_optimized(
        cost_map, expert_path, W_init, alpha_init=alpha0
    )

    print("\nRecovered W:", W_hat)
    print("Recovered alpha_final:", alpha_hat_final)

    # both on the recovered run's final risk map
    plot_compare(expert_path, recovered_path, risk_final)


if __name__ == "__main__":
    main()
