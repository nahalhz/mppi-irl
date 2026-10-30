"""Recovery loss vs each parameter (one at a time, others at true)."""
import os
import numpy as np
import matplotlib.pyplot as plt

import skid_steer_bi_rrt_star_CVaR_MPPI as mppi
import BO_MPPI_joint as J
from BO_MPPI_SVGP import (trajectory_distance, run_mppi,
                          LW1_BOUNDS, LW2_BOUNDS, LW3_BOUNDS, BETA_BOUNDS)

OUT_DIR = "value_function_results"
W_TRUE  = J.W_TRUE.copy()
BETA    = J.BETA_CVAR_TRUE
W_CTRL  = J.W_CTRL_LOSS
N_PTS   = 20  # linear grid size

SCEN  = J.SCENARIOS[0]
START, GOAL, SEED = SCEN["start"], SCEN["goal"], SCEN["seed"]
MAX_STEPS, N_SAMPLES = J.EXPERT_STEPS, J.EXPERT_SAMPLES

# same bounds as BO
RANGES = {
    "W1":   (np.exp(LW1_BOUNDS[0]),  np.exp(LW1_BOUNDS[1])),
    "W2":   (np.exp(LW2_BOUNDS[0]),  np.exp(LW2_BOUNDS[1])),
    "W3":   (np.exp(LW3_BOUNDS[0]),  np.exp(LW3_BOUNDS[1])),
    "beta": (BETA_BOUNDS[0],    BETA_BOUNDS[1]),
}
TRUE = {"W1": W_TRUE[1], "W2": W_TRUE[2], "W3": W_TRUE[3], "beta": BETA}
LABEL = {"W1": "W[1] path-tracking (true 10)", "W2": "W[2] control effort (true 0.01)",
         "W3": "W[3] terminal (true 0.5)", "beta": "beta_cvar risk (true 0.5)"}


def build_map():
    np.random.seed(J.COST_MAP_SEED)
    cost  = mppi.generate_cost_map(mppi.RISK_MAP_DIM, mppi.RISK_MAP_DIM)
    noise = J.make_shared_noise()
    return cost, noise


def rollout(W, beta, cost, noise):
    path, u, _, _ = run_mppi(
        W, beta, cost, noise, START, GOAL,
        max_steps=MAX_STEPS, n_samples=N_SAMPLES, seed=SEED)
    return path, u


def pad(u, n):
    return u[:n] if len(u) >= n else np.vstack([u, np.zeros((n - len(u), 2))])


def params_for(key, v):
    W, beta = W_TRUE.copy(), BETA
    if   key == "W1": W[1] = v
    elif key == "W2": W[2] = v
    elif key == "W3": W[3] = v
    else:             beta = v
    return W, beta


def main():
    assert mppi.CVAR_MODE == "deterministic"
    os.makedirs(OUT_DIR, exist_ok=True)
    cost, noise = build_map()

    exp_path, exp_u = rollout(W_TRUE, BETA, cost, noise)
    print(f"[vf] expert: {len(exp_path)} steps")

    data = {}
    for key in ["W1", "W2", "W3", "beta"]:
        lo, hi = RANGES[key]
        # linear grid + extra points near the true value
        near = TRUE[key] * np.array([0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 3.0, 5.0])
        grid = np.concatenate([np.linspace(lo, hi, N_PTS), near, [TRUE[key]]])
        grid = np.sort(np.unique(np.clip(grid, lo, hi)))
        traj, ctrl, tot = [], [], []
        print(f"\n[vf] sweeping {key} over [{lo:.4g}, {hi:.4g}] ({len(grid)} pts)")
        for v in grid:
            W, beta = params_for(key, v)
            path, u = rollout(W, beta, cost, noise)
            dt = trajectory_distance(exp_path, path)
            n  = max(len(exp_u), len(u))
            eff = np.mean(np.sum((pad(exp_u, n) - pad(u, n)) ** 2, axis=1))
            dc = W_CTRL * eff / (2 * mppi.W_MAX ** 2 + 1e-8)
            traj.append(dt); ctrl.append(dc); tot.append(dt + dc)
            print(f"   {key}={v:<10.4g} traj={dt:8.2f}  ctrl={dc:7.3f}  L={dt+dc:8.2f}")
        data[key] = (grid, np.array(traj), np.array(ctrl), np.array(tot))

    np.savez(os.path.join(OUT_DIR, "value_function_data.npz"),
             **{f"{k}_grid": data[k][0] for k in data},
             **{f"{k}_traj": data[k][1] for k in data},
             **{f"{k}_ctrl": data[k][2] for k in data},
             **{f"{k}_total": data[k][3] for k in data})

    # plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, key in zip(axes.ravel(), ["W1", "W2", "W3", "beta"]):
        grid, tj, cl, tt = data[key]
        ax.plot(grid, tt, "-o", lw=2, ms=4, color="#1A1F2B", label="total value L(θ)")
        ax.plot(grid, tj, "--", lw=1.6, color="#377eb8", label="position-match term")
        ax.plot(grid, cl, ":", lw=1.6, color="#3F7D5C", label="control term")
        ax.axvline(TRUE[key], color="#E07A5F", lw=2, ls="-", alpha=0.9, label="true value")
        ax.set_yscale("symlog", linthresh=1.0)  # L=0 at true
        imin = int(np.argmin(tt))
        ax.annotate(f"min at {grid[imin]:.3g}", xy=(grid[imin], max(tt[imin], 0.3)),
                    xytext=(0.5, 0.82), textcoords="axes fraction", fontsize=9,
                    color="#E07A5F",
                    arrowprops=dict(arrowstyle="->", color="#E07A5F", lw=1.2))
        ax.set_title(LABEL[key], fontsize=12)
        ax.set_xlabel("parameter value (linear, full range)")
        ax.set_ylabel("value  (summed path distance, m; symlog)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Recovery-loss 'value function' vs each parameter — the surface BO optimizes\n"
                 "one fixed scenario; other three parameters held at true; deterministic CVaR",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT_DIR, "value_function_vs_param.png")
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"\n[vf] saved {out}")


if __name__ == "__main__":
    main()
