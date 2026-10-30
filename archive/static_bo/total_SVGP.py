"""Variational-GP BO variant of BO_SAAS.py."""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from multiprocessing import get_context
from gpytorch.mlls import VariationalELBO

from botorch.models import SingleTaskVariationalGP
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples

from inverse_feature_BO_optimized import (
    run_mpc,
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


# variational-gp BO config
OUTPUT_DIR = "total_svgp_results"

N_INIT        = 32
N_BO_ITERS    = 218  # 250 evals total
N_INDUCING    = 64  # clamped to n_data
N_EPOCHS      = 150  # per iteration
LEARNING_RATE = 0.01

UCB_BETA         = 0.5
ACQ_NUM_RESTARTS = 10
ACQ_RAW_SAMPLES  = 512

def _make_run_tag():
    from inverse_feature_BO_optimized import W_CTRL_LOSS
    b_str  = str(UCB_BETA).replace(".", "p")
    wc_str = str(W_CTRL_LOSS).replace(".", "p")
    return (f"ucb_b{b_str}_n{N_INDUCING}_ep{N_EPOCHS}"
            f"_{N_INIT + N_BO_ITERS}evals_wctrl{wc_str}")
RUN_TAG = _make_run_tag()

N_PARALLEL = 3  # 1 = serial

DTYPE  = torch.double
DEVICE = torch.device("cpu")


# trajectory distance (sum)
def _align_paths(path_a, path_b, max_len):
    def pad(p, n):
        if len(p) >= n:
            return p[:n]
        return np.vstack([p, np.tile(p[-1], (n - len(p), 1))])
    return pad(path_a, max_len), pad(path_b, max_len)


def trajectory_distance(path_a, path_b):
    """Sum of per-step Euclidean distances (x, y only) after padding both paths to equal length by repeating the final position."""
    max_len = max(len(path_a), len(path_b))
    a, b = _align_paths(path_a, path_b, max_len)
    dists = np.linalg.norm(a[:, :2] - b[:, :2], axis=1)
    return float(np.sum(dists))


# objective (same as BO_SAAS)
def _make_bo_objective(cost_map, expert_path, expert_u, risk_cache, eval_log, w_ctrl):
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
        v  = dx * np.cos(theta) + dy * np.sin(theta)
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

        val = traj_loss + w_ctrl * ctrl_loss

        eval_log.append({
            "iter":      k - 1,
            "loss":      float(val),
            "traj_loss": float(traj_loss),
            "ctrl_loss": float(ctrl_loss),
            "W2":        float(W[2]),
            "W3":        float(W[3]),
            "alpha":     float(alpha),
        })

        if k % 5 == 0:
            print(f"  [var-gp-bo eval {k}] W=[1, 0, {W[2]:.5f}, {W[3]:.3f}] "
                  f"alpha={alpha:.3f}  traj_loss={traj_loss:.4f}  obj={val:.4f}")

        return float(val)

    return bo_objective


# lengthscales (works across BoTorch versions)
def _lengthscales(gp):
    """Walk common attribute paths to find the ARD RBF lengthscale tensor."""
    candidates = [
        lambda m: m.model.covar_module.base_kernel.lengthscale,
        lambda m: m.model.model.covar_module.base_kernel.lengthscale,
        lambda m: m.model.covar_module.lengthscale,
        lambda m: m.covar_module.base_kernel.lengthscale,
    ]
    for fn in candidates:
        try:
            ls = fn(gp).detach().squeeze()
            if ls.numel() == 3:
                return ls.flatten()
        except Exception:
            continue
    print("  [ls] WARNING: could not extract length scales — "
          "check model attribute path for this BoTorch version")
    return None


# SVGP fit
def _fit_svgp(train_X_norm, train_Y_std, n_inducing, n_epochs, lr):
    """Train SingleTaskVariationalGP with Adam + VariationalELBO"""
    n_data = train_X_norm.shape[0]
    n_ind  = min(n_inducing, n_data)

    model = SingleTaskVariationalGP(
        train_X=train_X_norm,
        train_Y=train_Y_std,
        num_outputs=1,
        inducing_points=n_ind,
        learn_inducing_points=True,
    )
    mll       = VariationalELBO(model.likelihood, model.model, num_data=n_data)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    model.likelihood.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        output = model.model(train_X_norm)
        loss   = -mll(output, train_Y_std.squeeze(-1))
        loss.backward()
        optimizer.step()

    model.eval()
    model.likelihood.eval()

    ls = _lengthscales(model)
    if ls is not None:
        print(f"  [svgp] length scales — lw2={ls[0]:.4f}  lw3={ls[1]:.4f}  alpha={ls[2]:.4f}")

    return model


# BO recovery
def fit_W_to_expert_svgp(
    cost_map,
    expert_path,
    expert_u,
    w_ctrl=W_CTRL_LOSS,
    beta=UCB_BETA,
    W2_init=5.0 / 600,
    W3_init=25.0,
    alpha_init=0.99,
    n_init=N_INIT,
    n_iters=N_BO_ITERS,
    verbose=True,
):
    """Recover W = [1, 0, W2, W3] and alpha using variational-GP BO."""
    risk_cache   = RiskMapCache(cost_map)
    eval_log     = []
    bo_objective = _make_bo_objective(
        cost_map, expert_path, expert_u, risk_cache, eval_log, w_ctrl
    )

    bounds = torch.tensor([
        [LW2_BOUNDS[0],  LW3_BOUNDS[0],  ALPHA_BOUNDS[0]],
        [LW2_BOUNDS[1],  LW3_BOUNDS[1],  ALPHA_BOUNDS[1]],
    ], dtype=DTYPE, device=DEVICE)

    bounds_low  = bounds[0]
    bounds_high = bounds[1]
    norm_bounds = torch.stack([
        torch.zeros(3, dtype=DTYPE, device=DEVICE),
        torch.ones(3,  dtype=DTYPE, device=DEVICE),
    ])

    # sobol initial design
    if verbose:
        print(f"\n[var-gp-bo] Sobol init: {n_init} points")
    X_init = draw_sobol_samples(bounds=bounds, n=n_init, q=1).squeeze(1)

    Y_init = []
    for i in range(n_init):
        x = X_init[i]
        y = bo_objective(x[0].item(), x[1].item(), x[2].item())
        Y_init.append(y)
    Y_init = torch.tensor(Y_init, dtype=DTYPE, device=DEVICE).unsqueeze(-1)

    train_X = X_init.clone()
    train_Y = -Y_init  # BoTorch maximizes

    # variational-gp BO loop
    for it in range(n_iters):
        if verbose:
            best_so_far = (-train_Y.max()).item()
            print(f"\n[var-gp-bo] iter {it+1}/{n_iters}  "
                  f"best-so-far loss={best_so_far:.5f}")

        train_X_norm = (train_X - bounds_low) / (bounds_high - bounds_low)
        Y_mean = train_Y.mean()
        Y_std  = train_Y.std().clamp_min(1e-6)
        train_Y_std = (train_Y - Y_mean) / Y_std

        gp = _fit_svgp(
            train_X_norm, train_Y_std,
            N_INDUCING, N_EPOCHS, LEARNING_RATE,
        )

        ls = _lengthscales(gp)
        if ls is not None:
            eval_log.append({
                "__ls__": True,
                "iter": n_init + it,
                "ls_lw2": float(ls[0]),
                "ls_lw3": float(ls[1]),
                "ls_alpha": float(ls[2]),
            })

        acq = UpperConfidenceBound(model=gp, beta=beta)
        candidate_norm, _ = optimize_acqf(
            acq_function=acq,
            bounds=norm_bounds,
            q=1,
            num_restarts=ACQ_NUM_RESTARTS,
            raw_samples=ACQ_RAW_SAMPLES,
        )

        candidate = candidate_norm * (bounds_high - bounds_low) + bounds_low

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
    best_idx  = train_Y.argmax().item()
    best_x    = train_X[best_idx]

    W_hat = np.array([1.0, 0.0,
                      float(np.exp(best_x[0].item())),
                      float(np.exp(best_x[1].item()))])
    alpha_hat = float(best_x[2].item())
    best_loss = float(-train_Y[best_idx].item())

    print(f"\n[var-gp-bo] FINAL  W={W_hat}  alpha={alpha_hat:.4f}  loss={best_loss:.5f}")

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
def plot_convergence(eval_log, W_true, alpha_true, out_path, w_ctrl=W_CTRL_LOSS, beta=UCB_BETA, n_iters=N_BO_ITERS):
    obj_log = [h for h in eval_log if "__ls__" not in h]
    ls_log  = [h for h in eval_log if "__ls__" in h]

    iters      = np.array([h["iter"]      for h in obj_log])
    loss       = np.array([h["loss"]      for h in obj_log])
    traj_loss  = np.array([h["traj_loss"] for h in obj_log])
    W2_vals    = np.array([h["W2"]        for h in obj_log])
    W3_vals    = np.array([h["W3"]        for h in obj_log])
    alpha_vals = np.array([h["alpha"]     for h in obj_log])

    # best by traj_loss so the parameter panels match the loss panel
    best_idx = np.zeros(len(traj_loss), dtype=int)
    cur = 0
    for i in range(len(traj_loss)):
        if traj_loss[i] < traj_loss[cur]: cur = i
        best_idx[i] = cur

    best_traj = np.minimum.accumulate(traj_loss)

    W2_true, W3_true = W_true[2], W_true[3]
    guided_start = N_INIT

    fig, axes = plt.subplots(3, 2, figsize=(13, 15))

    ax = axes[0, 0]
    ax.plot(iters, traj_loss, "o", alpha=0.3, label="per-eval")
    ax.plot(iters, best_traj, "r-", lw=2, label="best-so-far")
    ax.axvline(guided_start, color="gray", ls=":", alpha=0.6, label="var-GP-BO guided start")
    ax.set_xlabel("Evaluation index"); ax.set_ylabel("traj_loss"); ax.set_yscale("log")
    ax.set_title("Trajectory loss convergence (ctrl term removed)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(iters, np.abs(W2_vals - W2_true), "o", alpha=0.3, label="per-eval")
    ax.plot(iters, np.abs(W2_vals[best_idx] - W2_true), "r-", lw=2, label="best-so-far")
    ax.axvline(guided_start, color="gray", ls=":", alpha=0.6)
    ax.set_xlabel("Evaluation index"); ax.set_ylabel("|W[2] - true|"); ax.set_yscale("log")
    ax.set_title(f"W[2] error (true={W2_true:.5f})"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(iters, np.abs(W3_vals - W3_true), "o", alpha=0.3, label="per-eval")
    ax.plot(iters, np.abs(W3_vals[best_idx] - W3_true), "r-", lw=2, label="best-so-far")
    ax.axvline(guided_start, color="gray", ls=":", alpha=0.6)
    ax.set_xlabel("Evaluation index"); ax.set_ylabel("|W[3] - true|")
    ax.set_title(f"W[3] error (true={W3_true:.1f})"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(iters, np.abs(alpha_vals - alpha_true), "o", alpha=0.3, label="per-eval")
    ax.plot(iters, np.abs(alpha_vals[best_idx] - alpha_true), "r-", lw=2, label="best-so-far")
    ax.axvline(guided_start, color="gray", ls=":", alpha=0.6)
    ax.set_xlabel("Evaluation index"); ax.set_ylabel("|alpha - true|")
    ax.set_title(f"alpha error (true={alpha_true})"); ax.legend(); ax.grid(alpha=0.3)

    # lengthscale panel
    ax = axes[2, 0]
    if ls_log:
        ls_iters = np.array([h["iter"]     for h in ls_log])
        ls_lw2   = np.array([h["ls_lw2"]   for h in ls_log])
        ls_lw3   = np.array([h["ls_lw3"]   for h in ls_log])
        ls_alpha = np.array([h["ls_alpha"] for h in ls_log])
        ax.plot(ls_iters, ls_lw2,   "b-o", ms=3, label="lw2 (risk weight)")
        ax.plot(ls_iters, ls_lw3,   "g-o", ms=3, label="lw3 (goal weight)")
        ax.plot(ls_iters, ls_alpha, "r-o", ms=3, label="alpha (CVaR level)")
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5, label="=1 boundary")
        ax.set_xlabel("BO iteration"); ax.set_ylabel("Length scale (normalised input space)")
        ax.set_title("Kernel length scales\n(large = dimension uninformative to surrogate)")
        ax.legend(); ax.grid(alpha=0.3)
    else:
        ax.set_visible(False)

    axes[2, 1].set_visible(False)

    fig.suptitle(
        f"Variational-GP BO convergence  "
        f"(N_INIT={N_INIT}, N_ITERS={n_iters}, "
        f"N_INDUCING={N_INDUCING}, UCB β={beta}, W_CTRL={w_ctrl})",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# scenarios
SCENARIOS = [
    {
        "tag":        "sc6",
        "name":       "Scenario 6 — high alpha, top-left to bottom-right",
        "seed":       21,
        "start":      np.array([1.0, 8.5, -np.pi/4]),
        "goal":       np.array([8.5, 1.5, 0.0]),
        "W_true":     np.array([1.0, 0.0, 5.0/600, 145.0]),
        "alpha_true": 0.95,
    },
    {
        "tag":        "sc7",
        "name":       "Scenario 7 — high alpha, vertical traverse",
        "seed":       34,
        "start":      np.array([4.5, 0.5, np.pi/2]),
        "goal":       np.array([4.5, 9.0, np.pi/2]),
        "W_true":     np.array([1.0, 0.0, 8.0/600, 120.0]),
        "alpha_true": 0.85,
    },
    {
        "tag":        "sc8",
        "name":       "Scenario 8 — high alpha, off-axis short hop",
        "seed":       56,
        "start":      np.array([2.5, 3.0, 0.0]),
        "goal":       np.array([7.5, 6.5, np.pi/4]),
        "W_true":     np.array([1.0, 0.0, 3.0/600, 160.0]),
        "alpha_true": 0.9,
    },
    {
        "tag":        "sc9",
        "name":       "Scenario 9 — high alpha, edge hugging",
        "seed":       71,
        "start":      np.array([0.5, 5.0, 0.0]),
        "goal":       np.array([9.5, 5.0, 0.0]),
        "W_true":     np.array([1.0, 0.0, 6.0/600, 110.0]),
        "alpha_true": 0.88,
    },
    {
        "tag":        "sc10",
        "name":       "Scenario 10 — high alpha, bottom-left to top-right",
        "seed":       88,
        "start":      np.array([1.5, 1.5, np.pi/4]),
        "goal":       np.array([8.0, 8.0, np.pi/4]),
        "W_true":     np.array([1.0, 0.0, 4.0/600, 155.0]),
        "alpha_true": 0.92,
    },
    {
        "tag":        "sc11",
        "name":       "Scenario 11 — high alpha, low W[3]",
        "seed":       104,
        "start":      np.array([3.0, 9.0, -np.pi/2]),
        "goal":       np.array([7.0, 1.0, -np.pi/2]),
        "W_true":     np.array([1.0, 0.0, 5.0/600, 100.0]),
        "alpha_true": 0.78,
    },
    {
        "tag":        "sc12",
        "name":       "Scenario 12 — high alpha, high W[3]",
        "seed":       119,
        "start":      np.array([8.5, 2.0, np.pi]),
        "goal":       np.array([1.5, 7.5, np.pi/2]),
        "W_true":     np.array([1.0, 0.0, 7.0/600, 164.0]),
        "alpha_true": 0.97,
    },
    {
        "tag":        "sc13",
        "name":       "Scenario 13 — mid-high alpha, mid map",
        "seed":       137,
        "start":      np.array([2.0, 5.5, 0.0]),
        "goal":       np.array([8.0, 4.5, 0.0]),
        "W_true":     np.array([1.0, 0.0, 5.0/600, 130.0]),
        "alpha_true": 0.7,
    },
    {
        "tag":        "sc14",
        "name":       "Scenario 14 — low alpha, risk averse diagonal",
        "seed":       152,
        "start":      np.array([1.0, 1.0, np.pi/4]),
        "goal":       np.array([8.5, 8.5, 0.0]),
        "W_true":     np.array([1.0, 0.0, 5.0/600, 140.0]),
        "alpha_true": 0.25,
    },
    {
        "tag":        "sc15",
        "name":       "Scenario 15 — low alpha, risk averse off-axis",
        "seed":       168,
        "start":      np.array([7.0, 1.5, np.pi]),
        "goal":       np.array([2.0, 8.0, np.pi/2]),
        "W_true":     np.array([1.0, 0.0, 4.0/600, 125.0]),
        "alpha_true": 0.35,
    },
]


# don't overwrite - append _2 _3
def _safe_path(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 2
    while os.path.exists(f"{base}_{n}{ext}"):
        n += 1
    return f"{base}_{n}{ext}"


# one scenario
def run_scenario(sc, cost_map, w_ctrl=W_CTRL_LOSS, beta=UCB_BETA, n_iters=N_BO_ITERS):
    tag        = sc["tag"]
    W_true     = sc["W_true"]
    alpha_true = sc["alpha_true"]
    ctrl_tag   = f"ctrl{str(w_ctrl).replace('.', 'p')}"
    beta_tag   = f"b{str(beta).replace('.', 'p')}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    _orig_init = grid.INITIAL_STATE.copy()
    _orig_goal = grid.GOAL_STATE.copy()
    grid.INITIAL_STATE = sc["start"]
    grid.GOAL_STATE    = sc["goal"]

    print(f"\n{'='*60}")
    print(f"[{tag}] {sc['name']}")
    print(f"{'='*60}")

    print(f"[{tag}] running expert rollout...")
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

    expert_goal_dist = np.linalg.norm(expert_path[-1, :2] - sc["goal"][:2])
    if expert_goal_dist < grid.GRID_CELL_SIZE * 3:
        print(f"[{tag}] expert reached goal (dist={expert_goal_dist:.3f}m)")
    else:
        print(f"[{tag}] WARNING: expert stopped short (dist={expert_goal_dist:.3f}m) — continuing anyway")

    print(f"[{tag}] running variational-GP BO  (w_ctrl={w_ctrl}, beta={beta}, n_iters={n_iters})...")
    W_hat, alpha_hat, eval_log, recovered_path, risk_final = fit_W_to_expert_svgp(
        cost_map, expert_path, expert_u,
        w_ctrl=w_ctrl,
        beta=beta,
        n_iters=n_iters,
        alpha_init=alpha_true,
    )

    print(f"\n[{tag}] True      W={W_true},  alpha={alpha_true}")
    print(f"[{tag}] Recovered W={W_hat},  alpha={alpha_hat:.4f}")

    # trajectory plot
    traj_file = _safe_path(os.path.join(OUTPUT_DIR, f"{tag}_{ctrl_tag}_{beta_tag}_traj.png"))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(
        risk_final.T,
        origin="lower",
        extent=[grid.GRID_RANGE[0], grid.GRID_RANGE[1],
                grid.GRID_RANGE[0], grid.GRID_RANGE[1]],
        cmap="Reds", alpha=0.6,
    )
    ax.plot(expert_path[:, 0],    expert_path[:, 1],    "b",   lw=3, label="Expert")
    ax.plot(recovered_path[:, 0], recovered_path[:, 1], "r--", lw=3, label="Recovered (var-GP-BO)")
    ax.scatter(sc["goal"][0],     sc["goal"][1],     c="magenta", s=120, label="Goal")
    ax.scatter(expert_path[0, 0], expert_path[0, 1], c="green",   s=120, label="Start")
    ax.legend()
    ax.set_aspect("equal")
    ax.set_title(
        f"Var-GP-BO: {sc['name']}  (W_CTRL={w_ctrl}, β={beta})\n"
        f"Recovered W={W_hat.round(5)}  α={alpha_hat:.4f}\n"
        f"True      W={W_true}  α={alpha_true}"
    )
    plt.tight_layout()
    plt.savefig(traj_file, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{tag}] trajectory plot saved → {traj_file}")

    # convergence plot
    conv_file = _safe_path(os.path.join(OUTPUT_DIR, f"{tag}_{ctrl_tag}_{beta_tag}_conv.png"))
    plot_convergence(eval_log, W_true, alpha_true, conv_file, w_ctrl=w_ctrl, beta=beta, n_iters=n_iters)
    print(f"[{tag}] convergence plot saved → {conv_file}")

    grid.INITIAL_STATE = _orig_init
    grid.GOAL_STATE    = _orig_goal

    return W_hat, alpha_hat, eval_log


# main
def _scenario_job(args):
    sc, cost_map, w_ctrl, beta, n_iters = args
    return run_scenario(sc, cost_map, w_ctrl=w_ctrl, beta=beta, n_iters=n_iters)


def main():
    cost_map = grid.generate_cost_map(grid.RISK_MAP_DIM, grid.RISK_MAP_DIM)
    print(f"[main] cost map generated")
    print(f"[main] {len(SCENARIOS)} scenarios, N_PARALLEL={N_PARALLEL}")

    jobs = [(sc, cost_map, W_CTRL_LOSS, UCB_BETA, N_BO_ITERS) for sc in SCENARIOS]

    if N_PARALLEL > 1:
        ctx = get_context("spawn")
        with ctx.Pool(processes=N_PARALLEL) as pool:
            pool.map(_scenario_job, jobs)
    else:
        for job in jobs:
            _scenario_job(job)

    print("\n[main] all scenarios done.")


if __name__ == "__main__":
    main()
