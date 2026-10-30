"""True planner cost of each produced trajectory vs each parameter, split by term."""
import os
import numpy as np
import matplotlib.pyplot as plt

import skid_steer_bi_rrt_star_CVaR_MPPI as mppi
import BO_MPPI_joint as J
from BO_MPPI_SVGP import (run_mppi, trajectory_distance,
                          LW1_BOUNDS, LW2_BOUNDS, LW3_BOUNDS, BETA_BOUNDS)

OUT_DIR = "value_function_results"
W_TRUE  = J.W_TRUE.copy()
BETA    = J.BETA_CVAR_TRUE
N_PTS   = 20

SCEN  = J.SCENARIOS[0]
START, GOAL, SEED = SCEN["start"], SCEN["goal"], SCEN["seed"]
MAX_STEPS, N_SAMPLES = J.EXPERT_STEPS, J.EXPERT_SAMPLES

RANGES = {"W1": (np.exp(LW1_BOUNDS[0]), np.exp(LW1_BOUNDS[1])),
          "W2": (np.exp(LW2_BOUNDS[0]), np.exp(LW2_BOUNDS[1])),
          "W3": (np.exp(LW3_BOUNDS[0]), np.exp(LW3_BOUNDS[1])),
          "beta": (BETA_BOUNDS[0], BETA_BOUNDS[1])}
TRUE  = {"W1": W_TRUE[1], "W2": W_TRUE[2], "W3": W_TRUE[3], "beta": BETA}
LABEL = {"W1": "W[1] path-tracking (true 10)", "W2": "W[2] control effort (true 0.01)",
         "W3": "W[3] terminal (true 0.5)", "beta": "beta_cvar risk (true 0.5)"}


def build_map():
    np.random.seed(J.COST_MAP_SEED)
    cost  = mppi.generate_cost_map(mppi.RISK_MAP_DIM, mppi.RISK_MAP_DIM)
    noise = J.make_shared_noise()
    return cost, noise


def rollout(W, beta, cost, noise):
    path, u, rrt, _ = run_mppi(
        W, beta, cost, noise, START, GOAL,
        max_steps=MAX_STEPS, n_samples=N_SAMPLES, seed=SEED)
    return path, u, rrt


def terrain_cvar_cost(path, cost, noise, beta):
    """Deterministic per-cell CVaR summed along a path."""
    pts = path[:, :2]
    I = np.clip(np.floor(pts[:, 0] / mppi.GRID_CELL_SIZE).astype(int), 0, mppi.RISK_MAP_DIM - 1)
    Jj = np.clip(np.floor(pts[:, 1] / mppi.GRID_CELL_SIZE).astype(int), 0, mppi.RISK_MAP_DIM - 1)
    mu = cost[I, Jj]
    sigma = mppi.SIGMA_SCALE * np.maximum(noise[I, Jj], 1e-3)
    per_cell = np.maximum(mppi.cvar_normal_upper(mu, sigma, beta), 0.0)
    steplen = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    avg_step = 0.5 * (per_cell[:-1] + per_cell[1:])
    return float(np.sum(avg_step * steplen))


def components(path, u, ref_true, cost, noise):
    """The four cost terms under the true W / beta."""
    risk = W_TRUE[0] * terrain_cvar_cost(path, cost, noise, BETA)
    track = W_TRUE[1] * trajectory_distance(ref_true, path)
    ctrl = W_TRUE[2] * float(np.sum((u / mppi.W_MAX) ** 2))
    term = W_TRUE[3] * float(np.linalg.norm(path[-1, :2] - GOAL[:2]))
    return risk, track, ctrl, term


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

    exp_path, _, ref_true = rollout(W_TRUE, BETA, cost, noise)
    print(f"[pc] expert {len(exp_path)} steps; ref {len(ref_true)} pts")

    data = {}
    for key in ["W1", "W2", "W3", "beta"]:
        lo, hi = RANGES[key]
        near = TRUE[key] * np.array([0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 3.0, 5.0])
        grid = np.concatenate([np.linspace(lo, hi, N_PTS), near, [TRUE[key]]])
        grid = np.sort(np.unique(np.clip(grid, lo, hi)))
        rows = []
        print(f"\n[pc] sweeping {key} ({len(grid)} pts)")
        for v in grid:
            W, beta = params_for(key, v)
            path, u, _ = rollout(W, beta, cost, noise)
            risk, track, ctrl, term = components(path, u, ref_true, cost, noise)
            rows.append((risk, track, ctrl, term, risk + track + ctrl + term))
            print(f"   {key}={v:<10.4g} risk={risk:8.2f} track={track:8.2f} "
                  f"ctrl={ctrl:7.3f} term={term:7.3f} TOTAL={rows[-1][4]:8.2f}")
        data[key] = (grid, np.array(rows))

    np.savez(os.path.join(OUT_DIR, "planner_cost_data.npz"),
             **{f"{k}_grid": data[k][0] for k in data},
             **{f"{k}_rows": data[k][1] for k in data},
             component_order=np.array(["risk", "track", "ctrl", "term", "total"]))

    comp = [("total", "#1A1F2B", "-", 2.2),
            ("terrain risk (W0)", "#E07A5F", "--", 1.5),
            ("path-tracking (W1)", "#377eb8", "--", 1.5),
            ("control (W2)", "#3F7D5C", ":", 1.5),
            ("terminal (W3)", "#9467bd", ":", 1.5)]
    order = [4, 0, 1, 2, 3]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, key in zip(axes.ravel(), ["W1", "W2", "W3", "beta"]):
        grid, rows = data[key]
        for (name, col, ls, lw), ci in zip(comp, order):
            ax.plot(grid, rows[:, ci], ls, lw=lw, color=col, label=name,
                    marker="o" if name == "total" else None, ms=3)
        ax.axvline(TRUE[key], color="#E07A5F", lw=2, alpha=0.85)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_title(LABEL[key], fontsize=12)
        ax.set_xlabel("parameter value (linear, full range)")
        ax.set_ylabel("planner cost under TRUE objective (symlog)")
        ax.grid(alpha=0.3); ax.legend(fontsize=7.5)
    fig.suptitle("Raw planner cost of the produced trajectory vs each parameter\n"
                 "scored under the TRUE objective; one fixed scenario; others held at true",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT_DIR, "planner_cost_vs_param.png")
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"\n[pc] saved {out}")


if __name__ == "__main__":
    main()
