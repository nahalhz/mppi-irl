"""Per-scenario BO inverse RL for the bi-RRT* + CVaR-MPPI planner.

Recovers W[1:4] and beta_cvar from a single expert rollout (SVGP + UCB).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from gpytorch.mlls import VariationalELBO

from botorch.models import SingleTaskVariationalGP
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples

import skid_steer_bi_rrt_star_CVaR_MPPI as mppi

# outputs
OUTPUT_DIR = "mppi_svgp_results"

# BO config
N_INIT        = 50  # sobol init
N_BO_ITERS    = 250  # 300 evals total
N_INDUCING    = 64
N_EPOCHS      = 300  # per GP refit
LEARNING_RATE = 0.01
MIN_LR        = 1e-4  # cosine LR floor

UCB_BETA         = 2
ACQ_NUM_RESTARTS = 10
ACQ_RAW_SAMPLES  = 512

# proposal std per dim = ARD lengthscale (unit cube) only works because
# samples are reflected into bounds not clamped (see _reflect_unit)
LS_SIGMA_MIN = 0.02
LS_SIGMA_MAX = 1.00

W_CTRL_LOSS = 0.5  # 0 disables

DTYPE  = torch.double
DEVICE = torch.device("cpu")

# expert gets the full horizon BO evals a shorter one
EXPERT_STEPS  = mppi.MAX_STEPS_MPC  # 50 s

EVAL_STEPS = int(np.ceil(25.0 / mppi.dt))  # 25 s
EVAL_SAMPLES = 2000

# expert and evals must use the same N_SAMPLES - a different N is a different
# controller (sc01 at truth - 2000 vs 500 -> loss 61, matched -> 0)
EXPERT_SAMPLES  = EVAL_SAMPLES

# appended to output filenames _s#### = sample count
RUN_TAG = (
    f"v4nosamp_s{EVAL_SAMPLES}"
    f"_b{str(UCB_BETA).replace('.', 'p')}"
    f"_ctrl{str(W_CTRL_LOSS).replace('.', 'p')}"
    f"_n{N_INIT + N_BO_ITERS}"
)

# compare against the expert's first EVAL_STEPS only otherwise the
# padded tail dominates fixed window so stalling isn't rewarded
COMPARE_STEPS = EVAL_STEPS

# bounds (log for weights, linear for beta)
LW1_BOUNDS  = (-2.0,  4.0)  # ~[0.14, 54.6]
LW2_BOUNDS  = (-7.0,  0.0)  # ~[0.001, 1]
LW3_BOUNDS  = (-4.0,  3.0)  # ~[0.018, 20]
BETA_BOUNDS = ( 0.1,  0.99)

# shared cost map seed
COST_MAP_SEED = 6000

# scenarios
SCENARIOS = [
    {
        "tag":            "sc01_default",
        "name":           "Default — SW to NE diagonal, beta_cvar=0.5",
        "seed":           6000,
        "start":          np.array([1.0, 1.0, np.pi / 4]),
        "goal":           np.array([8.5, 8.5, 0.0]),
        "W_true":         np.array([1.0, 10.0, 0.01, 0.5]),
        "beta_cvar_true": 0.5,
    },
    {
        "tag":            "sc02_highrisk",
        "name":           "High risk aversion — NW to SE, beta_cvar=0.2",
        "seed":           6001,
        "start":          np.array([1.0, 8.5, -np.pi / 4]),
        "goal":           np.array([8.5, 1.5, 0.0]),
        "W_true":         np.array([1.0, 12.0, 0.005, 0.3]),
        "beta_cvar_true": 0.2,
    },
    {
        "tag":            "sc03_lowrisk",
        "name":           "Low risk aversion — vertical traverse, beta_cvar=0.8",
        "seed":           6002,
        "start":          np.array([5.0, 1.0, np.pi / 2]),
        "goal":           np.array([5.0, 9.0, np.pi / 2]),
        "W_true":         np.array([1.0, 8.0, 0.02, 0.8]),
        "beta_cvar_true": 0.8,
    },
    {
        "tag":            "sc04_strongtrack",
        "name":           "Strong path tracking — W[1]=15, beta_cvar=0.5",
        "seed":           6003,
        "start":          np.array([2.0, 2.0, np.pi / 4]),
        "goal":           np.array([8.0, 7.0, 0.0]),
        "W_true":         np.array([1.0, 15.0, 0.008, 0.6]),
        "beta_cvar_true": 0.5,
    },
    {
        "tag":            "sc05_horizontal",
        "name":           "Horizontal traverse — W to E, beta_cvar=0.3",
        "seed":           6004,
        "start":          np.array([1.0, 5.0, 0.0]),
        "goal":           np.array([9.0, 5.0, 0.0]),
        "W_true":         np.array([1.0, 9.0, 0.015, 0.4]),
        "beta_cvar_true": 0.3,
    },
    {
        "tag":            "sc06_weaktrack",
        "name":           "Weak path tracking — W[1]=4, beta_cvar=0.5",
        "seed":           6005,
        "start":          np.array([2.0, 8.0, -np.pi / 4]),
        "goal":           np.array([8.0, 2.0, -np.pi / 2]),
        "W_true":         np.array([1.0, 4.0, 0.03, 0.5]),
        "beta_cvar_true": 0.5,
    },
    {
        "tag":            "sc07_highterm",
        "name":           "High terminal weight — W[3]=1.5, beta_cvar=0.6",
        "seed":           6006,
        "start":          np.array([1.5, 3.0, np.pi / 6]),
        "goal":           np.array([8.5, 6.5, 0.0]),
        "W_true":         np.array([1.0, 10.0, 0.01, 1.5]),
        "beta_cvar_true": 0.6,
    },
    {
        "tag":            "sc08_lowctrl",
        "name":           "Low control effort — W[2]=0.002, beta_cvar=0.7",
        "seed":           6007,
        "start":          np.array([8.5, 8.5, -3 * np.pi / 4]),
        "goal":           np.array([1.5, 1.5, np.pi],),
        "W_true":         np.array([1.0, 11.0, 0.002, 0.4]),
        "beta_cvar_true": 0.7,
    },
    {
        "tag":            "sc09_shorthop",
        "name":           "High-risk short hop — beta_cvar=0.15",
        "seed":           6008,
        "start":          np.array([3.0, 3.0, np.pi / 4]),
        "goal":           np.array([7.0, 7.0, np.pi / 4]),
        "W_true":         np.array([1.0, 13.0, 0.007, 0.35]),
        "beta_cvar_true": 0.15,
    },
    {
        "tag":            "sc10_balanced",
        "name":           "Balanced mid-map — beta_cvar=0.9",
        "seed":           6009,
        "start":          np.array([2.0, 7.0, -np.pi / 3]),
        "goal":           np.array([8.0, 3.0, 0.0]),
        "W_true":         np.array([1.0, 7.0, 0.012, 0.7]),
        "beta_cvar_true": 0.9,
    },
]


def _align_paths(path_a, path_b, max_len):
    def pad(p, n):
        if len(p) >= n:
            return p[:n]
        return np.vstack([p, np.tile(p[-1], (n - len(p), 1))])
    return pad(path_a, max_len), pad(path_b, max_len)


def trajectory_distance(path_a, path_b):
    """Sum of per-step Euclidean (x, y) distances after padding both paths to equal length."""
    max_len = max(len(path_a), len(path_b))
    a, b = _align_paths(path_a, path_b, max_len)
    return float(np.sum(np.linalg.norm(a[:, :2] - b[:, :2], axis=1)))


# temporarily patches mppi.W_COST / MAX_STEPS_MPC / N_SAMPLES_MPPI
def run_mppi(W, beta_cvar, cost_map, noise_map,
                        initial_state, goal_state,
                        max_steps=None, n_samples=None, seed=None):
    """Run mppi.run_simulation with a custom W and beta_cvar"""
    orig_W       = list(mppi.W_COST)
    orig_steps   = mppi.MAX_STEPS_MPC
    orig_samples = mppi.N_SAMPLES_MPPI

    mppi.W_COST = list(W)
    if max_steps  is not None:
        mppi.MAX_STEPS_MPC  = max_steps
    if n_samples  is not None:
        mppi.N_SAMPLES_MPPI = n_samples

    if seed is not None:
        np.random.seed(seed)
    try:
        path, controls, rrt_path, risk_map = mppi.run_simulation(
            beta_cvar, cost_map, noise_map, initial_state, goal_state
        )
    finally:
        mppi.W_COST         = orig_W
        mppi.MAX_STEPS_MPC  = orig_steps
        mppi.N_SAMPLES_MPPI = orig_samples

    return np.array(path), np.array(controls), rrt_path, risk_map


# mirror instead of clamp - clamping piles samples on the bounds and UCB chases them
def _reflect_unit(x):
    """Fold values into [0,1] by mirroring at each boundary"""
    x = torch.abs(x)
    x = 1.0 - torch.abs(1.0 - x)
    return x.clamp(0.0, 1.0)  # |x| > 2


# works across BoTorch versions expects a 4-D ARD kernel
def _lengthscales(gp):
    candidates = [
        lambda m: m.model.covar_module.base_kernel.lengthscale,
        lambda m: m.model.model.covar_module.base_kernel.lengthscale,
        lambda m: m.model.covar_module.lengthscale,
        lambda m: m.covar_module.base_kernel.lengthscale,
    ]
    for fn in candidates:
        try:
            ls = fn(gp).detach().squeeze()
            if ls.numel() == 4:
                return ls.flatten()
        except Exception:
            continue
    print("  [ls] WARNING: could not extract length scales — "
          "check model attribute path for this BoTorch version")
    return None


def _fit_svgp(train_X_norm, train_Y_std, n_inducing, n_epochs, lr):
    """Train SingleTaskVariationalGP (4-D inputs) via Adam + VariationalELBO."""
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=MIN_LR
    )

    model.train()
    model.likelihood.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        output = model.model(train_X_norm)
        loss   = -mll(output, train_Y_std.squeeze(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    model.likelihood.eval()

    ls = _lengthscales(model)
    if ls is not None:
        print(f"  [svgp] length scales — "
              f"lw1={ls[0]:.4f}  lw2={ls[1]:.4f}  "
              f"lw3={ls[2]:.4f}  beta_cvar={ls[3]:.4f}")
    return model


# objective over (lw1, lw2, lw3, beta_cvar)
def _make_bo_objective(cost_map, noise_map, expert_path, expert_u,
                       initial_state, goal_state, eval_log, w_ctrl,
                       eval_seed):
    eval_counter = {"n": 0}

    # score over the window the evals actually simulate
    expert_cmp   = expert_path[:COMPARE_STEPS]
    expert_u_cmp = expert_u[:COMPARE_STEPS]
    goal_xy      = np.asarray(goal_state)[:2]
    reach_tol    = mppi.GRID_CELL_SIZE * 3

    def pad_controls(u, target):
        if len(u) >= target:
            return u[:target]
        return np.vstack([u, np.zeros((target - len(u), 2))])

    def bo_objective(lw1, lw2, lw3, beta_cvar):
        eval_counter["n"] += 1
        k = eval_counter["n"]

        W = np.array([1.0, np.exp(lw1), np.exp(lw2), np.exp(lw3)])

        # expert's seed every eval -> same bi-RRT* plan a per-eval seed added
        # ~150 units of noise at the true params
        path_hat, u_hat, _, _ = run_mppi(
            W, beta_cvar, cost_map, noise_map, initial_state, goal_state,
            max_steps=EVAL_STEPS,
            n_samples=EVAL_SAMPLES,
            seed=eval_seed,
        )

        traj_loss = trajectory_distance(expert_cmp, path_hat)

        n_max    = max(len(expert_u_cmp), len(u_hat))
        u_exp_p  = pad_controls(expert_u_cmp, n_max)
        u_hat_p  = pad_controls(u_hat,        n_max)
        effort_diff = np.mean(np.sum((u_exp_p - u_hat_p) ** 2, axis=1))
        ctrl_loss   = effort_diff / (2 * mppi.W_MAX ** 2 + 1e-8)

        val = traj_loss + w_ctrl * ctrl_loss

        # did it reach the goal or stall?
        goal_dist = float(np.linalg.norm(path_hat[-1, :2] - goal_xy))

        eval_log.append({
            "iter":      k - 1,
            "loss":      float(val),
            "traj_loss": float(traj_loss),
            "ctrl_loss": float(ctrl_loss),
            "W1":        float(W[1]),
            "W2":        float(W[2]),
            "W3":        float(W[3]),
            "beta_cvar": float(beta_cvar),
            "goal_dist": goal_dist,
            "reached":   bool(goal_dist < reach_tol),
            "n_steps":   int(len(path_hat)),
        })

        if k % 5 == 0:
            flag = "goal" if goal_dist < reach_tol else f"d={goal_dist:.2f}"
            print(f"  [mppi-bo eval {k}]  "
                  f"W=[1, {W[1]:.3f}, {W[2]:.5f}, {W[3]:.3f}]  "
                  f"beta_cvar={beta_cvar:.3f}  traj_loss={traj_loss:.4f}  "
                  f"obj={val:.4f}  [{flag}]")

        return float(val)

    return bo_objective


def fit_W(
    cost_map,
    noise_map,
    expert_path,
    expert_u,
    initial_state,
    goal_state,
    eval_seed,
    w_ctrl=W_CTRL_LOSS,
    ucb_beta=UCB_BETA,
    n_init=N_INIT,
    n_iters=N_BO_ITERS,
    verbose=True,
):
    """Recover W = [1, W1, W2, W3] and beta_cvar via SVGP BO"""
    eval_log     = []
    bo_objective = _make_bo_objective(
        cost_map, noise_map, expert_path, expert_u,
        initial_state, goal_state, eval_log, w_ctrl, eval_seed,
    )

    bounds = torch.tensor([
        [LW1_BOUNDS[0], LW2_BOUNDS[0], LW3_BOUNDS[0], BETA_BOUNDS[0]],
        [LW1_BOUNDS[1], LW2_BOUNDS[1], LW3_BOUNDS[1], BETA_BOUNDS[1]],
    ], dtype=DTYPE, device=DEVICE)

    bounds_low  = bounds[0]
    bounds_high = bounds[1]
    norm_bounds = torch.stack([
        torch.zeros(4, dtype=DTYPE, device=DEVICE),
        torch.ones(4,  dtype=DTYPE, device=DEVICE),
    ])

    # sobol initial design
    if verbose:
        print(f"\n[mppi-bo] Sobol init: {n_init} points")
    X_init = draw_sobol_samples(bounds=bounds, n=n_init, q=1).squeeze(1)

    Y_init = []
    for i in range(n_init):
        x = X_init[i]
        y = bo_objective(x[0].item(), x[1].item(), x[2].item(), x[3].item())
        Y_init.append(y)
    Y_init  = torch.tensor(Y_init, dtype=DTYPE, device=DEVICE).unsqueeze(-1)

    train_X = X_init.clone()
    train_Y = -Y_init   # BoTorch maximizes → negate the loss

    # variational-gp BO loop
    for it in range(n_iters):
        if verbose:
            best_so_far = (-train_Y.max()).item()
            print(f"\n[mppi-bo] iter {it+1}/{n_iters}  best-so-far={best_so_far:.5f}")

        train_X_norm = (train_X - bounds_low) / (bounds_high - bounds_low)
        Y_mean = train_Y.mean()
        Y_std  = train_Y.std().clamp_min(1e-6)
        train_Y_std = (train_Y - Y_mean) / Y_std

        gp = _fit_svgp(train_X_norm, train_Y_std, N_INDUCING, N_EPOCHS, LEARNING_RATE)

        ls = _lengthscales(gp)
        if ls is not None:
            eval_log.append({
                "__ls__":       True,
                "iter":         n_init + it,
                "ls_lw1":       float(ls[0]),
                "ls_lw2":       float(ls[1]),
                "ls_lw3":       float(ls[2]),
                "ls_beta_cvar": float(ls[3]),
            })

        # plain UCB over the unit box (lengthscales are only logged)
        acq = UpperConfidenceBound(model=gp, beta=ucb_beta)
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
            candidate[0, 3].item(),
        )
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([
            train_Y,
            torch.tensor([[-y_new]], dtype=DTYPE, device=DEVICE),
        ], dim=0)

    # best point by traj_loss
    obj_records    = [h for h in eval_log if "__ls__" not in h]
    traj_losses    = [h["traj_loss"] for h in obj_records]
    best_eval      = int(np.argmin(traj_losses))
    best_rec       = obj_records[best_eval]

    W_hat         = np.array([1.0, best_rec["W1"], best_rec["W2"], best_rec["W3"]])
    beta_cvar_hat = best_rec["beta_cvar"]

    print(f"\n[mppi-bo] FINAL  W={W_hat}  beta_cvar={beta_cvar_hat:.4f}  "
          f"traj_loss={best_rec['traj_loss']:.5f}")

    # final rollout at full budget same seed as the evals
    path_final, _, rrt_final, risk_final = run_mppi(
        W_hat, beta_cvar_hat, cost_map, noise_map, initial_state, goal_state,
        max_steps=EXPERT_STEPS,
        n_samples=EXPERT_SAMPLES,
        seed=eval_seed,
    )

    return W_hat, beta_cvar_hat, eval_log, path_final, rrt_final, risk_final


# convergence plot (3x2)
def plot_convergence(eval_log, W_true, beta_cvar_true, out_path,
                     w_ctrl=W_CTRL_LOSS, ucb_beta=UCB_BETA, n_iters=N_BO_ITERS):
    obj_log = [h for h in eval_log if "__ls__" not in h]
    ls_log  = [h for h in eval_log if "__ls__" in h]

    iters          = np.array([h["iter"]      for h in obj_log])
    traj_loss      = np.array([h["traj_loss"] for h in obj_log])
    W1_vals        = np.array([h["W1"]        for h in obj_log])
    W2_vals        = np.array([h["W2"]        for h in obj_log])
    W3_vals        = np.array([h["W3"]        for h in obj_log])
    beta_cvar_vals = np.array([h["beta_cvar"] for h in obj_log])

    best_idx = np.zeros(len(traj_loss), dtype=int)
    cur = 0
    for i in range(len(traj_loss)):
        if traj_loss[i] < traj_loss[cur]:
            cur = i
        best_idx[i] = cur
    best_traj = np.minimum.accumulate(traj_loss)

    W1_true, W2_true, W3_true = W_true[1], W_true[2], W_true[3]
    guided_start = N_INIT

    fig, axes = plt.subplots(3, 2, figsize=(13, 15))

    ax = axes[0, 0]
    ax.plot(iters, traj_loss, "o", alpha=0.3, label="per-eval")
    ax.plot(iters, best_traj, "r-", lw=2, label="best-so-far")
    ax.axvline(guided_start, color="gray", ls=":", alpha=0.6, label="GP-BO start")
    ax.set_xlabel("Evaluation"); ax.set_ylabel("traj_loss"); ax.set_yscale("log")
    ax.set_title("Trajectory loss convergence"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(iters, np.abs(W1_vals - W1_true), "o", alpha=0.3, label="per-eval")
    ax.plot(iters, np.abs(W1_vals[best_idx] - W1_true), "r-", lw=2, label="best-so-far")
    ax.axvline(guided_start, color="gray", ls=":", alpha=0.6)
    ax.set_xlabel("Evaluation"); ax.set_ylabel("|W[1] − true|"); ax.set_yscale("log")
    ax.set_title(f"W[1] error (true={W1_true:.3f}, path-tracking)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(iters, np.abs(W2_vals - W2_true), "o", alpha=0.3, label="per-eval")
    ax.plot(iters, np.abs(W2_vals[best_idx] - W2_true), "r-", lw=2, label="best-so-far")
    ax.axvline(guided_start, color="gray", ls=":", alpha=0.6)
    ax.set_xlabel("Evaluation"); ax.set_ylabel("|W[2] − true|")
    ax.set_title(f"W[2] error (true={W2_true:.5f}, ctrl effort)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(iters, np.abs(W3_vals - W3_true), "o", alpha=0.3, label="per-eval")
    ax.plot(iters, np.abs(W3_vals[best_idx] - W3_true), "r-", lw=2, label="best-so-far")
    ax.axvline(guided_start, color="gray", ls=":", alpha=0.6)
    ax.set_xlabel("Evaluation"); ax.set_ylabel("|W[3] − true|")
    ax.set_title(f"W[3] error (true={W3_true:.3f}, terminal)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2, 0]
    ax.plot(iters, np.abs(beta_cvar_vals - beta_cvar_true), "o", alpha=0.3, label="per-eval")
    ax.plot(iters, np.abs(beta_cvar_vals[best_idx] - beta_cvar_true), "r-", lw=2, label="best-so-far")
    ax.axvline(guided_start, color="gray", ls=":", alpha=0.6)
    ax.set_xlabel("Evaluation"); ax.set_ylabel("|beta_cvar − true|")
    ax.set_title(f"beta_cvar error (true={beta_cvar_true})"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2, 1]
    if ls_log:
        ls_iters    = np.array([h["iter"]         for h in ls_log])
        ls_lw1      = np.array([h["ls_lw1"]       for h in ls_log])
        ls_lw2      = np.array([h["ls_lw2"]       for h in ls_log])
        ls_lw3      = np.array([h["ls_lw3"]       for h in ls_log])
        ls_beta_cvar = np.array([h["ls_beta_cvar"] for h in ls_log])
        ax.plot(ls_iters, ls_lw1,       "b-o", ms=3, label="lw1 (path-tracking)")
        ax.plot(ls_iters, ls_lw2,       "g-o", ms=3, label="lw2 (ctrl effort)")
        ax.plot(ls_iters, ls_lw3,       "m-o", ms=3, label="lw3 (terminal)")
        ax.plot(ls_iters, ls_beta_cvar, "r-o", ms=3, label="beta_cvar (CVaR)")
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5, label="=1 boundary")
        ax.set_xlabel("BO iteration"); ax.set_ylabel("Length scale (normalised space)")
        ax.set_title("Kernel length scales\n(large = dimension uninformative to surrogate)")
        ax.legend(); ax.grid(alpha=0.3)
    else:
        ax.set_visible(False)

    fig.suptitle(
        f"MPPI Var-GP-BO convergence  "
        f"(N_INIT={N_INIT}, N_ITERS={n_iters}, N_INDUCING={N_INDUCING}, "
        f"UCB β={ucb_beta}, W_CTRL={w_ctrl})",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  convergence plot saved → {out_path}")


# one scenario - noise map per scenario cost map shared
def run_scenario(sc, cost_map,
                 w_ctrl=W_CTRL_LOSS, ucb_beta=UCB_BETA, n_iters=N_BO_ITERS):
    tag            = sc["tag"]
    W_true         = sc["W_true"]
    beta_cvar_true = sc["beta_cvar_true"]
    start          = sc["start"]
    goal           = sc["goal"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[{tag}] {sc['name']}")
    print(f"{'='*60}")

    rng       = np.random.RandomState(sc["seed"])
    noise_map = mppi.generate_noise_map(
        mppi.RISK_MAP_DIM, mppi.RISK_MAP_DIM, rng,
        start_xy=start[:2],
        goal_xy=goal[:2],
    )

    # expert rollout
    print(f"[{tag}] expert rollout (seed={sc['seed']})...")
    expert_path, expert_u, expert_rrt, _ = run_mppi(
        W_true, beta_cvar_true, cost_map, noise_map, start, goal,
        max_steps=EXPERT_STEPS,
        n_samples=EXPERT_SAMPLES,
        seed=sc["seed"],
    )
    goal_dist = np.linalg.norm(expert_path[-1, :2] - goal[:2])
    print(f"[{tag}] expert: {len(expert_path)} steps, goal dist={goal_dist:.3f}m")

    # BO
    print(f"[{tag}] MPPI var-GP-BO  "
          f"(w_ctrl={w_ctrl}, ucb_beta={ucb_beta}, n_iters={n_iters})...")
    W_hat, beta_cvar_hat, eval_log, recovered_path, recovered_rrt, risk_final = (
        fit_W(
            cost_map, noise_map, expert_path, expert_u, start, goal,
            eval_seed=sc["seed"],
            w_ctrl=w_ctrl, ucb_beta=ucb_beta, n_iters=n_iters,
        )
    )

    print(f"\n[{tag}] True      W={W_true}   beta_cvar={beta_cvar_true}")
    print(f"[{tag}] Recovered W={W_hat}  beta_cvar={beta_cvar_hat:.4f}")

    # trajectory plot
    traj_file = os.path.join(OUTPUT_DIR, f"{tag}_{RUN_TAG}_traj.png")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, path, rrt_path, title, color in [
        (axes[0], expert_path,    expert_rrt,    "Expert (bi-RRT* + MPPI)", "blue"),
        (axes[1], recovered_path, recovered_rrt, "Recovered (MPPI-BO)",     "red"),
    ]:
        ax.imshow(
            risk_final.T,
            origin="lower",
            extent=[0, mppi.GRID_L, 0, mppi.GRID_L],
            cmap="Reds", alpha=0.45,
        )
        if rrt_path is not None and len(rrt_path) > 1:
            ax.plot(rrt_path[:, 0], rrt_path[:, 1],
                    "k--", lw=1.5, alpha=0.5, label="bi-RRT* path")
        ax.plot(path[:, 0], path[:, 1], color=color, lw=2.5, label="MPPI executed")
        ax.scatter(goal[0],  goal[1],  c="magenta", s=150, zorder=5, label="Goal")
        ax.scatter(start[0], start[1], c="green",   s=150, zorder=5, label="Start")
        ax.set_xlim(0, mppi.GRID_L); ax.set_ylim(0, mppi.GRID_L)
        ax.set_aspect("equal"); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(alpha=0.2)

    axes[1].plot(expert_path[:, 0], expert_path[:, 1],
                 "b", lw=1.5, alpha=0.4, ls="--", label="Expert (ref)")
    axes[1].legend(fontsize=8)

    fig.suptitle(
        f"MPPI-BO: {sc['name']}\n"
        f"Recovered  W={np.round(W_hat, 4)}   β_cvar={beta_cvar_hat:.4f}\n"
        f"True       W={W_true}   β_cvar={beta_cvar_true}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(traj_file, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{tag}] trajectory saved → {traj_file}")

    # convergence plot
    conv_file = os.path.join(OUTPUT_DIR, f"{tag}_{RUN_TAG}_conv.png")
    plot_convergence(eval_log, W_true, beta_cvar_true, conv_file,
                     w_ctrl=w_ctrl, ucb_beta=ucb_beta, n_iters=n_iters)

    return W_hat, beta_cvar_hat, eval_log


# main
def main():
    np.random.seed(COST_MAP_SEED)
    cost_map = mppi.generate_cost_map(mppi.RISK_MAP_DIM, mppi.RISK_MAP_DIM)
    print(f"[main] cost map generated (seed={COST_MAP_SEED})")
    print(f"[main] {len(SCENARIOS)} scenarios, UCB_BETA={UCB_BETA}")
    print(f"[main] BO budget: {N_INIT} Sobol + {N_BO_ITERS} BO = {N_INIT+N_BO_ITERS} total evals")
    print(f"[main] BO eval rollout: {EVAL_STEPS} steps, {EVAL_SAMPLES} MPPI samples\n")

    for sc in SCENARIOS:
        run_scenario(sc, cost_map,
                     w_ctrl=W_CTRL_LOSS, ucb_beta=UCB_BETA, n_iters=N_BO_ITERS)

    print("\n[main] all scenarios done.")


if __name__ == "__main__":
    main()
