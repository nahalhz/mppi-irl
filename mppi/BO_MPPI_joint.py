"""Joint BO recovery: one shared (W, beta_cvar) fit to every scenario at once.

Loss = sum over scenarios of trajectory_distance + w_ctrl * ctrl_loss.
Helpers are shared with BO_MPPI_SVGP.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.utils.sampling import draw_sobol_samples

import skid_steer_bi_rrt_star_CVaR_MPPI as mppi
import BO_MPPI_SVGP as base
from BO_MPPI_SVGP import (
    trajectory_distance,
    run_mppi,
    _reflect_unit,
    _lengthscales,
    _fit_svgp,
    LW1_BOUNDS, LW2_BOUNDS, LW3_BOUNDS, BETA_BOUNDS,
    DTYPE, DEVICE,
)

# outputs
OUTPUT_DIR = "mppi_joint_results"

# BO config
N_INIT        = 50  # sobol init
N_BO_ITERS    = 250  # 300 evals total
N_INDUCING    = 64
N_EPOCHS      = 300
LEARNING_RATE = 0.01

UCB_BETA        = 2
ACQ_RAW_SAMPLES = 512
LS_SIGMA_MIN    = 0.02
LS_SIGMA_MAX    = 1.00

W_CTRL_LOSS = 0.5

# rollout budget (expert and eval sample counts must match)
EXPERT_STEPS   = mppi.MAX_STEPS_MPC
EVAL_STEPS     = int(np.ceil(25.0 / mppi.dt))  # 25 s
EVAL_SAMPLES   = 500
EXPERT_SAMPLES = EVAL_SAMPLES
COMPARE_STEPS  = EVAL_STEPS

COST_MAP_SEED = 6000

# true params shared by every scenario
W_TRUE         = np.array([1.0, 10.0, 0.01, 0.5])
BETA_CVAR_TRUE = 0.5

# first 10 come from BO_MPPI_SVGP.SCENARIOS (their weights are ignored)
# the extra 5 add new directions / lengths
EXTRA_SCENARIOS = [
    {"tag": "sc11_serev",  "name": "SE→NW reverse diagonal",
     "seed": 6010, "start": np.array([9.0, 1.0, 3*np.pi/4]), "goal": np.array([1.0, 9.0, 3*np.pi/4])},
    {"tag": "sc12_vertdn", "name": "Vertical downward traverse",
     "seed": 6011, "start": np.array([5.0, 9.0, -np.pi/2]),  "goal": np.array([5.0, 1.0, -np.pi/2])},
    {"tag": "sc13_horizw", "name": "Horizontal E→W traverse",
     "seed": 6012, "start": np.array([9.0, 5.0, np.pi]),     "goal": np.array([1.0, 5.0, np.pi])},
    {"tag": "sc14_shallow","name": "Shallow diagonal, long run",
     "seed": 6013, "start": np.array([1.0, 1.5, 0.0]),       "goal": np.array([9.0, 3.5, 0.0])},
    {"tag": "sc15_offaxis","name": "Off-axis NW hop",
     "seed": 6014, "start": np.array([7.5, 2.0, 3*np.pi/4]), "goal": np.array([2.5, 8.0, np.pi/2])},
]

N_SCENARIOS = 15
_base_scen = [
    {"tag": sc["tag"], "name": sc["name"],
     "seed": sc["seed"], "start": sc["start"], "goal": sc["goal"]}
    for sc in base.SCENARIOS
]
SCENARIOS = (_base_scen + EXTRA_SCENARIOS)[:N_SCENARIOS]

# joint fit is run at each of these counts (first k scenarios)
SCENARIO_COUNTS = [10, 15]

# tag has N and CVaR mode so runs don't overwrite each other
_CVAR_MARK = "cvardet" if mppi.CVAR_MODE == "deterministic" else "cvarsamp"
RUN_TAG = (
    f"joint{len(SCENARIOS)}_{_CVAR_MARK}_s{EVAL_SAMPLES}"
    f"_b{str(UCB_BETA).replace('.', 'p')}"
    f"_ctrl{str(W_CTRL_LOSS).replace('.', 'p')}"
    f"_n{N_INIT + N_BO_ITERS}"
)


# one noise field for every scenario -> identical risk map
# only start/goal change
SHARED_NOISE_SEED = 6000
NOISE_REF_START   = np.array([1.0, 1.0])
NOISE_REF_GOAL    = np.array([9.0, 9.0])


def make_shared_noise():
    rng = np.random.RandomState(SHARED_NOISE_SEED)
    return mppi.generate_noise_map(
        mppi.RISK_MAP_DIM, mppi.RISK_MAP_DIM, rng,
        start_xy=NOISE_REF_START, goal_xy=NOISE_REF_GOAL,
    )


# expert rollout per scenario on the shared terrain
def prepare_experts(cost_map, noise_map=None):
    if noise_map is None:
        noise_map = make_shared_noise()
    experts = []
    for sc in SCENARIOS:
        print(f"[expert] {sc['tag']} rollout (seed={sc['seed']})...")
        e_path, e_u, e_rrt, _ = run_mppi(
            W_TRUE, BETA_CVAR_TRUE, cost_map, noise_map,
            sc["start"], sc["goal"],
            max_steps=EXPERT_STEPS, n_samples=EXPERT_SAMPLES,
            seed=sc["seed"],
        )
        gdist = np.linalg.norm(e_path[-1, :2] - sc["goal"][:2])
        print(f"[expert] {sc['tag']}: {len(e_path)} steps, goal dist={gdist:.3f}m")
        experts.append({
            **sc,
            "noise_map":    noise_map,
            "expert_path":  e_path,
            "expert_rrt":   e_rrt,
            "expert_cmp":   e_path[:COMPARE_STEPS],
            "expert_u_cmp": e_u[:COMPARE_STEPS],
            "goal_xy":      np.asarray(sc["goal"])[:2],
        })
    return experts


# joint objective = sum of per-scenario losses
def _make_joint_objective(cost_map, experts, eval_log, w_ctrl):
    eval_counter = {"n": 0}
    reach_tol = mppi.GRID_CELL_SIZE * 3

    def pad_controls(u, target):
        if len(u) >= target:
            return u[:target]
        return np.vstack([u, np.zeros((target - len(u), 2))])

    def joint_objective(lw1, lw2, lw3, beta_cvar):
        eval_counter["n"] += 1
        k = eval_counter["n"]
        W = np.array([1.0, np.exp(lw1), np.exp(lw2), np.exp(lw3)])

        total_traj = 0.0
        total_ctrl = 0.0
        n_reached  = 0

        for e in experts:
            # expert's seed -> same bi-RRT* plan
            path_hat, u_hat, _, _ = run_mppi(
                W, beta_cvar, cost_map, e["noise_map"],
                e["start"], e["goal"],
                max_steps=EVAL_STEPS, n_samples=EVAL_SAMPLES,
                seed=e["seed"],
            )
            total_traj += trajectory_distance(e["expert_cmp"], path_hat)

            n_max   = max(len(e["expert_u_cmp"]), len(u_hat))
            u_exp_p = pad_controls(e["expert_u_cmp"], n_max)
            u_hat_p = pad_controls(u_hat,             n_max)
            effort_diff = np.mean(np.sum((u_exp_p - u_hat_p) ** 2, axis=1))
            total_ctrl += effort_diff / (2 * mppi.W_MAX ** 2 + 1e-8)

            if np.linalg.norm(path_hat[-1, :2] - e["goal_xy"]) < reach_tol:
                n_reached += 1

        val = total_traj + w_ctrl * total_ctrl

        eval_log.append({
            "iter":      k - 1,
            "loss":      float(val),
            "traj_loss": float(total_traj),
            "ctrl_loss": float(total_ctrl),
            "W1":        float(W[1]),
            "W2":        float(W[2]),
            "W3":        float(W[3]),
            "beta_cvar": float(beta_cvar),
            "n_reached": int(n_reached),
        })

        if k % 5 == 0:
            print(f"  [joint eval {k}]  W=[1, {W[1]:.3f}, {W[2]:.5f}, {W[3]:.3f}]  "
                  f"beta_cvar={beta_cvar:.3f}  Σtraj={total_traj:.3f}  "
                  f"obj={val:.3f}  reached={n_reached}/{len(experts)}")

        return float(val)

    return joint_objective


# same loop as BO_MPPI_SVGP.fit_W
def fit_joint(cost_map, experts, w_ctrl=W_CTRL_LOSS,
              ucb_beta=UCB_BETA, n_init=N_INIT, n_iters=N_BO_ITERS,
              verbose=True):
    eval_log = []
    obj = _make_joint_objective(cost_map, experts, eval_log, w_ctrl)

    bounds = torch.tensor([
        [LW1_BOUNDS[0], LW2_BOUNDS[0], LW3_BOUNDS[0], BETA_BOUNDS[0]],
        [LW1_BOUNDS[1], LW2_BOUNDS[1], LW3_BOUNDS[1], BETA_BOUNDS[1]],
    ], dtype=DTYPE, device=DEVICE)
    bounds_low, bounds_high = bounds[0], bounds[1]
    norm_bounds = torch.stack([
        torch.zeros(4, dtype=DTYPE, device=DEVICE),
        torch.ones(4,  dtype=DTYPE, device=DEVICE),
    ])

    # sobol init
    if verbose:
        print(f"\n[joint-bo] Sobol init: {n_init} points "
              f"({n_init * len(experts)} rollouts)")
    X_init = draw_sobol_samples(bounds=bounds, n=n_init, q=1).squeeze(1)
    Y_init = [obj(*X_init[i].tolist()) for i in range(n_init)]
    Y_init = torch.tensor(Y_init, dtype=DTYPE, device=DEVICE).unsqueeze(-1)

    train_X = X_init.clone()
    train_Y = -Y_init  # BoTorch maximises

    # BO loop
    for it in range(n_iters):
        if verbose:
            print(f"\n[joint-bo] iter {it+1}/{n_iters}  "
                  f"best-so-far={(-train_Y.max()).item():.4f}")

        train_X_norm = (train_X - bounds_low) / (bounds_high - bounds_low)
        Y_mean = train_Y.mean()
        Y_std  = train_Y.std().clamp_min(1e-6)
        train_Y_std = (train_Y - Y_mean) / Y_std

        gp = _fit_svgp(train_X_norm, train_Y_std, N_INDUCING, N_EPOCHS, LEARNING_RATE)

        ls = _lengthscales(gp)
        if ls is not None:
            eval_log.append({
                "__ls__": True, "iter": n_init + it,
                "ls_lw1": float(ls[0]), "ls_lw2": float(ls[1]),
                "ls_lw3": float(ls[2]), "ls_beta_cvar": float(ls[3]),
            })

        acq = UpperConfidenceBound(model=gp, beta=ucb_beta)
        if ls is not None:
            best_norm = ((train_X[train_Y.argmax()] - bounds_low)
                         / (bounds_high - bounds_low)).clamp(0.0, 1.0)
            sigma_dims = ls.to(dtype=DTYPE, device=DEVICE).clamp(LS_SIGMA_MIN, LS_SIGMA_MAX)
            if verbose and it % 20 == 0:
                print(f"  [ls-sample] mean={best_norm.numpy().round(3)}  "
                      f"σ=ls={sigma_dims.numpy().round(3)}")
            cand_batch = _reflect_unit(
                best_norm.unsqueeze(0)
                + torch.randn(ACQ_RAW_SAMPLES, 4, dtype=DTYPE, device=DEVICE)
                * sigma_dims.unsqueeze(0)
            )
        else:
            cand_batch = torch.rand(ACQ_RAW_SAMPLES, 4, dtype=DTYPE, device=DEVICE)

        with torch.no_grad():
            acq_vals = acq(cand_batch.unsqueeze(1))
        candidate_norm = cand_batch[int(acq_vals.argmax())].unsqueeze(0)

        candidate = candidate_norm * (bounds_high - bounds_low) + bounds_low
        y_new = obj(*candidate[0].tolist())
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y,
                             torch.tensor([[-y_new]], dtype=DTYPE, device=DEVICE)], dim=0)

    # best by summed traj_loss
    obj_records = [h for h in eval_log if "__ls__" not in h]
    traj_losses = [h["traj_loss"] for h in obj_records]
    best_rec    = obj_records[int(np.argmin(traj_losses))]

    W_hat         = np.array([1.0, best_rec["W1"], best_rec["W2"], best_rec["W3"]])
    beta_cvar_hat = best_rec["beta_cvar"]
    print(f"\n[joint-bo] FINAL  W={W_hat}  beta_cvar={beta_cvar_hat:.4f}  "
          f"Σtraj={best_rec['traj_loss']:.4f}  reached={best_rec['n_reached']}/{len(experts)}")
    return W_hat, beta_cvar_hat, eval_log


# convergence plot
def plot_convergence(eval_log, out_path, n_scen):
    obj_log = [h for h in eval_log if "__ls__" not in h]
    ls_log  = [h for h in eval_log if "__ls__" in h]

    iters = np.array([h["iter"]      for h in obj_log])
    traj  = np.array([h["traj_loss"] for h in obj_log])
    W1    = np.array([h["W1"]        for h in obj_log])
    W2    = np.array([h["W2"]        for h in obj_log])
    W3    = np.array([h["W3"]        for h in obj_log])
    bc    = np.array([h["beta_cvar"] for h in obj_log])

    best_idx = np.zeros(len(traj), dtype=int); cur = 0
    for i in range(len(traj)):
        if traj[i] < traj[cur]: cur = i
        best_idx[i] = cur
    best_traj = np.minimum.accumulate(traj)

    W1t, W2t, W3t = W_TRUE[1], W_TRUE[2], W_TRUE[3]
    gs = N_INIT
    fig, axes = plt.subplots(3, 2, figsize=(13, 15))

    ax = axes[0, 0]
    ax.plot(iters, traj, "o", alpha=0.3, label="per-eval")
    ax.plot(iters, best_traj, "r-", lw=2, label="best-so-far")
    ax.axvline(gs, color="gray", ls=":", alpha=0.6, label="GP-BO start")
    ax.set_xlabel("Evaluation"); ax.set_ylabel(f"Σ traj_loss ({n_scen} scenarios)")
    ax.set_yscale("log"); ax.set_title("Joint trajectory loss convergence")
    ax.legend(); ax.grid(alpha=0.3)

    for ax, v, vt, lbl, log in [
        (axes[0, 1], W1, W1t, f"W[1] error (true={W1t:.3f}, path-tracking)", True),
        (axes[1, 0], W2, W2t, f"W[2] error (true={W2t:.5f}, ctrl effort)",  False),
        (axes[1, 1], W3, W3t, f"W[3] error (true={W3t:.3f}, terminal)",     False),
        (axes[2, 0], bc, BETA_CVAR_TRUE, f"beta_cvar error (true={BETA_CVAR_TRUE})", False),
    ]:
        ax.plot(iters, np.abs(v - vt), "o", alpha=0.3, label="per-eval")
        ax.plot(iters, np.abs(v[best_idx] - vt), "r-", lw=2, label="best-so-far")
        ax.axvline(gs, color="gray", ls=":", alpha=0.6)
        ax.set_xlabel("Evaluation"); ax.set_ylabel("|value − true|")
        if log: ax.set_yscale("log")
        ax.set_title(lbl); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2, 1]
    if ls_log:
        li = np.array([h["iter"] for h in ls_log])
        for key, c, lab in [("ls_lw1", "b-o", "lw1 (path-tracking)"),
                            ("ls_lw2", "g-o", "lw2 (ctrl effort)"),
                            ("ls_lw3", "m-o", "lw3 (terminal)"),
                            ("ls_beta_cvar", "r-o", "beta_cvar (CVaR)")]:
            ax.plot(li, [h[key] for h in ls_log], c, ms=3, label=lab)
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5, label="=1 boundary")
        ax.set_xlabel("BO iteration"); ax.set_ylabel("Length scale (normalised)")
        ax.set_title("Kernel length scales\n(large = dimension uninformative)")
        ax.legend(); ax.grid(alpha=0.3)
    else:
        ax.set_visible(False)

    fig.suptitle(
        f"Joint MPPI Var-GP-BO  ({n_scen} scenarios, shared W/beta, "
        f"CVaR={mppi.CVAR_MODE})  "
        f"N_INIT={N_INIT}, N_ITERS={N_BO_ITERS}, UCB β={UCB_BETA}, "
        f"W_CTRL={W_CTRL_LOSS}, samples={EVAL_SAMPLES}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  convergence plot saved → {out_path}")


# trajectory grid at the recovered params
def plot_trajectory_grid(cost_map, experts, W_hat, beta_cvar_hat, out_path):
    n = len(experts)
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 5.0 * nrows),
                             squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    for ax, e in zip(flat, experts):
        path_hat, _, _, risk = run_mppi(
            W_hat, beta_cvar_hat, cost_map, e["noise_map"],
            e["start"], e["goal"],
            max_steps=EXPERT_STEPS, n_samples=EXPERT_SAMPLES,
            seed=e["seed"],
        )
        ax.imshow(risk.T, origin="lower", extent=[0, mppi.GRID_L, 0, mppi.GRID_L],
                  cmap="Reds", alpha=0.45)
        ax.plot(e["expert_path"][:, 0], e["expert_path"][:, 1], "b", lw=2.5, label="Expert")
        ax.plot(path_hat[:, 0], path_hat[:, 1], "r--", lw=2.5, label="Recovered")
        ax.scatter(*e["goal"][:2], c="magenta", s=90, zorder=5)
        ax.scatter(*e["start"][:2], c="green", s=90, zorder=5)
        ax.set_xlim(0, mppi.GRID_L); ax.set_ylim(0, mppi.GRID_L)
        ax.set_aspect("equal"); ax.set_title(e["tag"], fontsize=10)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(
        f"Joint recovery applied to all {len(experts)} scenarios  "
        f"(CVaR={mppi.CVAR_MODE})\n"
        f"Recovered W={np.round(W_hat, 4)}  β_cvar={beta_cvar_hat:.4f}   |   "
        f"True W={W_TRUE}  β_cvar={BETA_CVAR_TRUE}",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  trajectory grid saved → {out_path}")


# main
def _tag_for(n_scen):
    """Run tag for a given scenario count under the current CVaR mode."""
    mark = "cvardet" if mppi.CVAR_MODE == "deterministic" else "cvarsamp"
    return (f"joint{n_scen}_{mark}_s{EVAL_SAMPLES}"
            f"_b{str(UCB_BETA).replace('.', 'p')}"
            f"_ctrl{str(W_CTRL_LOSS).replace('.', 'p')}"
            f"_n{N_INIT + N_BO_ITERS}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(COST_MAP_SEED)
    cost_map = mppi.generate_cost_map(mppi.RISK_MAP_DIM, mppi.RISK_MAP_DIM)
    print(f"[main] cost map generated (seed={COST_MAP_SEED})")
    print(f"[main] CVaR mode = {mppi.CVAR_MODE}")
    print(f"[main] scenario counts = {SCENARIO_COUNTS}")
    print(f"[main] shared truth: W={W_TRUE}  beta_cvar={BETA_CVAR_TRUE}")
    print(f"[main] budget: {N_INIT} Sobol + {N_BO_ITERS} BO = {N_INIT+N_BO_ITERS} evals")
    print(f"[main] samples={EVAL_SAMPLES} (expert & eval matched)\n")

    # build experts once the first k are shared across counts
    all_experts = prepare_experts(cost_map)

    for n_scen in SCENARIO_COUNTS:
        experts = all_experts[:n_scen]
        tag = _tag_for(n_scen)
        print(f"\n{'#'*62}")
        print(f"[main] JOINT recovery — {n_scen} scenarios  (CVaR={mppi.CVAR_MODE})")
        print(f"[main] tag = {tag}")
        print(f"[main] each eval = {n_scen} rollouts → "
              f"{(N_INIT+N_BO_ITERS)*n_scen} eval-rollouts")
        print(f"{'#'*62}")

        W_hat, beta_cvar_hat, eval_log = fit_joint(cost_map, experts)

        print(f"\n[main] {n_scen} scen — True      W={W_TRUE}   beta_cvar={BETA_CVAR_TRUE}")
        print(f"[main] {n_scen} scen — Recovered W={W_hat}  beta_cvar={beta_cvar_hat:.4f}")

        conv = os.path.join(OUTPUT_DIR, f"joint_{tag}_conv.png")
        grid = os.path.join(OUTPUT_DIR, f"joint_{tag}_grid.png")
        plot_convergence(eval_log, conv, n_scen)
        plot_trajectory_grid(cost_map, experts, W_hat, beta_cvar_hat, grid)

    print("\n[main] all scenario counts done.")


if __name__ == "__main__":
    main()
