"""SAASBO variant of inverse_feature_BO_optimized.py."""

import numpy as np
import matplotlib.pyplot as plt
import torch

from botorch.models import SaasFullyBayesianSingleTaskGP
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

# reuse the rollout / loss / bounds so runs are comparable
from inverse_feature_BO_optimized import (
    run_mpc,
    trajectory_distance,
    RiskMapCache,
    T_EXPERT,
    T_BO_EVAL,
    SIM_TIME_EXPERT,
    SIM_TIME_BO,
    CEM_EXPERT,
    CEM_BO_EVAL,
    W_CTRL_LOSS,
    LW2_BOUNDS,
    LW3_BOUNDS,
    ALPHA_BOUNDS,
)
import skid_steer_simple_static as grid


# ~80 evals to match vanilla BO (83) NUTS refit is the slow part (30 s - 2 min/iter)
N_INIT     = 12
N_BO_ITERS = 35

# NUTS (eriksson 2021 defaults)
NUTS_WARMUP   = 256
NUTS_SAMPLES  = 128
NUTS_THINNING = 16

# acquisition
ACQ_NUM_RESTARTS = 10
ACQ_RAW_SAMPLES  = 512

DTYPE  = torch.double
DEVICE = torch.device("cpu")


# same loss as bo_objective in fit_W_to_expert without bayes_opt
def _make_bo_objective(cost_map, expert_path, expert_u, risk_cache, eval_log):
    """Build the closure that SAASBO will minimize"""
    W_FIXED = np.array([1.0, 0.0, 0.0, 0.0])
    eval_counter = {"n": 0}

    def reconstruct_controls(path):
        r_wheel, L_axle, dt = grid.r, grid.L, grid.dt
        if len(path) < 2:
            return np.zeros((0, 2))
        dx  = np.diff(path[:, 0]) / dt
        dy  = np.diff(path[:, 1]) / dt
        dth = np.diff(np.unwrap(path[:, 2])) / dt
        theta = path[:-1, 2]
        v = dx * np.cos(theta) + dy * np.sin(theta)
        ul = v / r_wheel - (L_axle * dth) / (2 * r_wheel)
        ur = v / r_wheel + (L_axle * dth) / (2 * r_wheel)
        return np.stack([ul, ur], axis=1)

    def pad_controls(u, target):
        if len(u) >= target:
            return u[:target]
        return np.vstack([u, np.zeros((target - len(u), 2))])

    def bo_objective(lw2, lw3, alpha):
        eval_counter["n"] += 1
        k = eval_counter["n"]

        W = W_FIXED.copy()
        W[2] = np.exp(lw2)
        W[3] = np.exp(lw3)

        _orig_T = grid.T
        grid.T  = T_BO_EVAL
        path_hat, _, _, _ = run_mpc(
            cost_map, W,
            alpha_init=alpha,
            max_steps=int(SIM_TIME_BO / grid.dt),
            verbose=False,
            cem_cfg=CEM_BO_EVAL,
            risk_cache=risk_cache,
            seed_base=1000,
        )
        grid.T = _orig_T

        traj_loss = trajectory_distance(expert_path, path_hat)

        u_hat = reconstruct_controls(path_hat)
        n_max = max(len(expert_u), len(u_hat))
        u_exp_p = pad_controls(expert_u, n_max)
        u_hat_p = pad_controls(u_hat,    n_max)
        effort_diff = np.mean(np.sum((u_exp_p - u_hat_p) ** 2, axis=1))
        ctrl_loss = effort_diff / (2 * grid.W_MAX ** 2 + 1e-8)

        val = traj_loss + W_CTRL_LOSS * ctrl_loss

        eval_log.append({
            "iter":  k - 1,
            "loss":  float(val),
            "W2":    float(W[2]),
            "W3":    float(W[3]),
            "alpha": float(alpha),
        })

        if k % 5 == 0:
            print(f"  [saasbo eval {k}] W=[1, 0, {W[2]:.5f}, {W[3]:.3f}] "
                  f"alpha={alpha:.3f}  traj_loss={traj_loss:.4f}  obj={val:.4f}")

        return float(val)

    return bo_objective


# drop-in replacement for fit_W_to_expert
def fit_W_to_expert_saasbo(
    cost_map,
    expert_path,
    expert_u,
    W2_init=5.0 / 600,  # unused kept for the same signature
    W3_init=25.0,
    alpha_init=0.99,
    n_init=N_INIT,
    n_iters=N_BO_ITERS,
    verbose=True,
):
    """Recover W = [1, 0, W2, W3] and alpha from an expert trajectory using SAASBO."""
    risk_cache = RiskMapCache(cost_map)
    eval_log   = []
    bo_objective = _make_bo_objective(
        cost_map, expert_path, expert_u, risk_cache, eval_log
    )

    # bounds: [lw2, lw3, alpha]
    bounds = torch.tensor([
        [LW2_BOUNDS[0],   LW3_BOUNDS[0],   ALPHA_BOUNDS[0]],
        [LW2_BOUNDS[1],   LW3_BOUNDS[1],   ALPHA_BOUNDS[1]],
    ], dtype=DTYPE, device=DEVICE)

    # sobol initial design
    if verbose:
        print(f"\n[saasbo] Sobol init: {n_init} points")
    X_init = draw_sobol_samples(bounds=bounds, n=n_init, q=1).squeeze(1)

    Y_init = []
    for i in range(n_init):
        x = X_init[i]
        y = bo_objective(x[0].item(), x[1].item(), x[2].item())
        Y_init.append(y)
    Y_init = torch.tensor(Y_init, dtype=DTYPE, device=DEVICE).unsqueeze(-1)

    # BoTorch maximises
    train_X = X_init.clone()
    train_Y = -Y_init

    # SAASBO loop
    for it in range(n_iters):
        if verbose:
            best_so_far = (-train_Y.max()).item()
            print(f"\n[saasbo] iter {it+1}/{n_iters}  best-so-far loss={best_so_far:.5f}")

        # SAAS GP via NUTS (inputs in [0,1]^3, outputs standardised)
        gp = SaasFullyBayesianSingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=Normalize(d=3, bounds=bounds),
            outcome_transform=Standardize(m=1),
        )
        fit_fully_bayesian_model_nuts(
            gp,
            warmup_steps=NUTS_WARMUP,
            num_samples=NUTS_SAMPLES,
            thinning=NUTS_THINNING,
            disable_progbar=True,
        )

        # qLogEI
        acq = qLogExpectedImprovement(model=gp, best_f=train_Y.max())
        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=ACQ_NUM_RESTARTS,
            raw_samples=ACQ_RAW_SAMPLES,
        )

        y_new = bo_objective(
            candidate[0, 0].item(),
            candidate[0, 1].item(),
            candidate[0, 2].item(),
        )
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([
            train_Y,
            torch.tensor([[-y_new]], dtype=DTYPE, device=DEVICE),
        ], dim=0)

    # best point
    best_idx = train_Y.argmax().item()
    best_x   = train_X[best_idx]

    W_hat = np.array([1.0, 0.0,
                      float(np.exp(best_x[0].item())),
                      float(np.exp(best_x[1].item()))])
    alpha_hat = float(best_x[2].item())
    best_loss = float(-train_Y[best_idx].item())

    print(f"\n[saasbo] FINAL  W={W_hat}  alpha={alpha_hat:.4f}  loss={best_loss:.5f}")

    # final rollout
    _orig_T = grid.T
    grid.T  = T_EXPERT
    path_final, _, _, risk_final = run_mpc(
        cost_map, W_hat,
        alpha_init=alpha_hat,
        max_steps=int(SIM_TIME_EXPERT / grid.dt),
        risk_cache=risk_cache,
    )
    grid.T = _orig_T

    return W_hat, alpha_hat, eval_log, path_final, risk_final


# convergence plot
def plot_convergence(eval_log, W_true, alpha_true, out_path):
    iters      = np.array([h["iter"]  for h in eval_log])
    loss       = np.array([h["loss"]  for h in eval_log])
    W2_vals    = np.array([h["W2"]    for h in eval_log])
    W3_vals    = np.array([h["W3"]    for h in eval_log])
    alpha_vals = np.array([h["alpha"] for h in eval_log])

    best_loss = np.minimum.accumulate(loss)
    best_idx  = np.zeros(len(loss), dtype=int)
    cur = 0
    for i in range(len(loss)):
        if loss[i] < loss[cur]: cur = i
        best_idx[i] = cur

    W2_true, W3_true = W_true[2], W_true[3]
    saasbo_start = N_INIT

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    ax.plot(iters, loss, "o", alpha=0.3, label="per-eval")
    ax.plot(iters, best_loss, "r-", lw=2, label="best-so-far")
    ax.axvline(saasbo_start, color="gray", ls=":", alpha=0.6, label="SAASBO-guided start")
    ax.set_xlabel("Evaluation index"); ax.set_ylabel("loss"); ax.set_yscale("log")
    ax.set_title("Objective convergence"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(iters, np.abs(W2_vals - W2_true), "o", alpha=0.3, label="per-eval")
    ax.plot(iters, np.abs(W2_vals[best_idx] - W2_true), "r-", lw=2, label="best-so-far")
    ax.axvline(saasbo_start, color="gray", ls=":", alpha=0.6)
    ax.set_xlabel("Evaluation index"); ax.set_ylabel("|W[2] - true|"); ax.set_yscale("log")
    ax.set_title(f"W[2] error (true={W2_true:.5f})"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(iters, np.abs(W3_vals - W3_true), "o", alpha=0.3, label="per-eval")
    ax.plot(iters, np.abs(W3_vals[best_idx] - W3_true), "r-", lw=2, label="best-so-far")
    ax.axvline(saasbo_start, color="gray", ls=":", alpha=0.6)
    ax.set_xlabel("Evaluation index"); ax.set_ylabel("|W[3] - true|")
    ax.set_title(f"W[3] error (true={W3_true:.1f})"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(iters, np.abs(alpha_vals - alpha_true), "o", alpha=0.3, label="per-eval")
    ax.plot(iters, np.abs(alpha_vals[best_idx] - alpha_true), "r-", lw=2, label="best-so-far")
    ax.axvline(saasbo_start, color="gray", ls=":", alpha=0.6)
    ax.set_xlabel("Evaluation index"); ax.set_ylabel("|alpha - true|")
    ax.set_title(f"alpha error (true={alpha_true})"); ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(f"SAASBO convergence  (N_INIT={N_INIT}, N_ITERS={N_BO_ITERS})", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# main (same scenario as inverse_feature_BO_optimized)
def main():
    # ground truth
    alpha_true = 0.99
    W_true = np.array([1.0, 0.0, 5.0 / 600, 150.0])

    cost_map = grid.generate_cost_map(grid.RISK_MAP_DIM, grid.RISK_MAP_DIM)
    print(f"[main] cost map generated")

    print("\n=== Running expert rollout ===")
    _orig_T = grid.T
    grid.T  = T_EXPERT
    expert_path, expert_u, _, _ = run_mpc(
        cost_map, W_true,
        alpha_init=alpha_true,
        max_steps=int(SIM_TIME_EXPERT / grid.dt),
        risk_cache=RiskMapCache(cost_map),
        cem_cfg=CEM_EXPERT,
    )
    grid.T = _orig_T

    expert_goal_dist = np.linalg.norm(expert_path[-1, :2] - grid.GOAL_STATE[:2])
    if expert_goal_dist < grid.GRID_CELL_SIZE * 3:
        print(f"Expert reached goal (dist={expert_goal_dist:.3f}m) — proceeding.")
    else:
        print(f"Expert did NOT reach goal (dist={expert_goal_dist:.3f}m). "
              f"Increase SIM_TIME_EXPERT or check weights/alpha before continuing.")
        return

    print("\n=== SAASBO inverse learning ===")
    W2_init    = 5.0 / 600
    W3_init    = 25.0
    alpha_init = 0.99

    W_hat, alpha_hat, eval_log, recovered_path, risk_final = fit_W_to_expert_saasbo(
        cost_map,
        expert_path,
        expert_u,
        W2_init=W2_init,
        W3_init=W3_init,
        alpha_init=alpha_init,
    )

    print(f"\nTrue      W={W_true},  alpha={alpha_true}")
    print(f"Recovered W={W_hat},  alpha={alpha_hat:.4f}")

    # trajectory plot
    plt.figure(figsize=(8, 8))
    plt.imshow(
        risk_final.T,
        origin="lower",
        extent=[grid.GRID_RANGE[0], grid.GRID_RANGE[1],
                grid.GRID_RANGE[0], grid.GRID_RANGE[1]],
        cmap="Reds",
        alpha=0.6,
    )
    plt.plot(expert_path[:, 0],    expert_path[:, 1],    "b",   lw=3, label="Expert")
    plt.plot(recovered_path[:, 0], recovered_path[:, 1], "r--", lw=3, label="Recovered (SAASBO)")
    plt.scatter(grid.GOAL_STATE[0], grid.GOAL_STATE[1], c="magenta", s=120, label="Goal")
    plt.scatter(expert_path[0, 0], expert_path[0, 1],   c="green",   s=120, label="Start")
    plt.legend()
    plt.title(
        f"SAASBO: Expert vs Recovered\n"
        f"Recovered W={W_hat.round(5)}  α={alpha_hat:.4f}\n"
        f"True      W={W_true}  α={alpha_true}"
    )
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig("saasbo_trajectory.png", dpi=120, bbox_inches="tight")
    print("[main] trajectory plot saved → saasbo_trajectory.png")

    # convergence plot
    plot_convergence(eval_log, W_true, alpha_true, "saasbo_convergence.png")
    print("[main] convergence plot saved → saasbo_convergence.png")


if __name__ == "__main__":
    main()
