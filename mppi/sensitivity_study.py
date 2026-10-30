"""Trajectory sensitivity: scale one parameter by n x true and measure path drift."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import skid_steer_bi_rrt_star_CVaR_MPPI as mppi
from BO_MPPI_SVGP import (
    run_mppi, trajectory_distance,
    LW1_BOUNDS, LW2_BOUNDS, LW3_BOUNDS, BETA_BOUNDS,
)
import BO_MPPI_joint as J

# separate folder sensitivity_results/ is the older x5 run
OUTPUT_DIR = "sensitivity_results_fullrange"

# fixed scenario
SCEN      = J.SCENARIOS[0]  # SW -> NE
START     = SCEN["start"]
GOAL      = SCEN["goal"]
SEED      = SCEN["seed"]
W_TRUE    = J.W_TRUE.copy()
BETA_TRUE = J.BETA_CVAR_TRUE

# rollout budget (same seed everywhere)
MAX_STEPS = J.EXPERT_STEPS
N_SAMPLES = J.EXPERT_SAMPLES

# bounds (same as BO)
W1_BOUNDS = (np.exp(LW1_BOUNDS[0]), np.exp(LW1_BOUNDS[1]))
W2_BOUNDS = (np.exp(LW2_BOUNDS[0]), np.exp(LW2_BOUNDS[1]))
W3_BOUNDS = (np.exp(LW3_BOUNDS[0]), np.exp(LW3_BOUNDS[1]))

# factors n per parameter each spanning its full range n=1 is the baseline
FACTORS = {
    "W1":   [0.02, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
    "W2":   [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0],
    "W3":   [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0],
    "beta": [0.2, 0.5, 0.75, 1.0, 1.25, 1.5, 1.98],
}

PARAMS = [
    ("W[1] path-tracking",  "W1"),
    ("W[2] control effort", "W2"),
    ("W[3] terminal",       "W3"),
    ("beta_cvar (risk)",    "beta"),
]


def perturbed_params(key, n):
    """Scale one parameter by n, keep the rest at true."""
    W, beta, clamped = W_TRUE.copy(), BETA_TRUE, False

    def clip(val, lo, hi):
        nonlocal clamped
        c = float(np.clip(val, lo, hi))
        clamped = clamped or (c != val)
        return c

    if   key == "W1":   W[1] = clip(W_TRUE[1] * n, *W1_BOUNDS)
    elif key == "W2":   W[2] = clip(W_TRUE[2] * n, *W2_BOUNDS)
    elif key == "W3":   W[3] = clip(W_TRUE[3] * n, *W3_BOUNDS)
    elif key == "beta": beta = clip(BETA_TRUE * n, *BETA_BOUNDS)
    return W, beta, clamped


def build_map():
    """Same cost map + shared noise field as the joint recovery."""
    np.random.seed(J.COST_MAP_SEED)
    cost  = mppi.generate_cost_map(mppi.RISK_MAP_DIM, mppi.RISK_MAP_DIM)
    noise = J.make_shared_noise()
    return cost, noise


def rollout(W, beta, cost, noise):
    path, _, _, _ = run_mppi(
        W, beta, cost, noise, START, GOAL,
        max_steps=MAX_STEPS, n_samples=N_SAMPLES, seed=SEED,
    )
    return path


def main():
    assert mppi.CVAR_MODE == "deterministic", \
        "set CVAR_MODE=deterministic so drift is a pure parameter effect"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cost, noise = build_map()
    risk_base = mppi.cvar_risk_map(BETA_TRUE, cost, noise)
    base = rollout(W_TRUE, BETA_TRUE, cost, noise)
    print(f"[sens] baseline at true params: {len(base)} steps, "
          f"goal dist={np.linalg.norm(base[-1, :2] - GOAL[:2]):.3f} m")

    drift = {key: [] for _, key in PARAMS}
    paths = {key: {} for _, key in PARAMS}

    for label, key in PARAMS:
        print(f"\n[sens] sweeping {label}  (factors {FACTORS[key]})")
        for n in FACTORS[key]:
            W, beta, clamped = perturbed_params(key, n)
            p = rollout(W, beta, cost, noise)
            d = trajectory_distance(base, p)
            drift[key].append(d)
            paths[key][n] = p
            val = f"beta={beta:.3f}" if key == "beta" else f"W={np.round(W, 4)}"
            flag = "  [clamped]" if clamped else ""
            print(f"   n={n:<6} {val:<34}  drift={d:8.2f}{flag}")

    # save trajectories
    store = {
        "baseline":  base, "W_TRUE": np.asarray(W_TRUE),
        "BETA_TRUE": np.asarray(BETA_TRUE),
        "start": np.asarray(START), "goal": np.asarray(GOAL),
        "cost_map": cost, "noise_map": noise, "risk_base": risk_base,
        "param_keys": np.asarray([k for _, k in PARAMS]),
    }
    for _, key in PARAMS:
        store[f"factors_{key}"] = np.asarray(FACTORS[key])
        store[f"drift_{key}"]   = np.asarray(drift[key])
        for i, n in enumerate(FACTORS[key]):
            store[f"path_{key}_{i}"] = paths[key][n]
    npz = os.path.join(OUTPUT_DIR, "sensitivity_data.npz")
    np.savez(npz, **store)
    n_paths = sum(len(FACTORS[k]) for _, k in PARAMS)
    print(f"\n[sens] saved trajectory data -> {npz} "
          f"(baseline + {n_paths} perturbed paths)")

    # drift vs n (log-log n=1 floored to 1)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for label, key in PARAMS:
        y = np.maximum(drift[key], 1.0)
        ax.plot(FACTORS[key], y, "-o", lw=2, ms=5, label=label)
    ax.axvline(1.0, color="gray", ls=":", alpha=0.7, label="true (n=1, drift~0)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("multiplicative factor n   (parameter = n x true value)")
    ax.set_ylabel("trajectory drift from baseline   (log; floored at 1)")
    ax.set_title("Trajectory sensitivity: how much each parameter moves the path\n"
                 "per-parameter full-range sweep, deterministic CVaR, shared terrain\n"
                 "(steep = identifiable; flat = dominated/non-identifiable)")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    f1 = os.path.join(OUTPUT_DIR, "sensitivity_drift_vs_n.png")
    plt.tight_layout(); plt.savefig(f1, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[sens] saved {f1}")

    # trajectory overlays on the true-beta risk map
    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    for ax, (label, key) in zip(axes.ravel(), PARAMS):
        ax.imshow(risk_base.T, origin="lower",
                  extent=[0, mppi.GRID_L, 0, mppi.GRID_L], cmap="Reds", alpha=0.45)
        facs = FACTORS[key]
        for i, n in enumerate(facs):
            p = paths[key][n]
            ax.plot(p[:, 0], p[:, 1], color=cm.viridis(i / (len(facs) - 1)),
                    lw=1.5, alpha=0.85, label=f"x{n}")
        ax.plot(base[:, 0], base[:, 1], "r--", lw=2.5, label="baseline")
        ax.scatter(*GOAL[:2],  c="magenta", s=110, zorder=5)
        ax.scatter(*START[:2], c="green",   s=110, zorder=5)
        ax.set_xlim(0, mppi.GRID_L); ax.set_ylim(0, mppi.GRID_L)
        ax.set_aspect("equal"); ax.set_title(label)
        ax.legend(fontsize=6, ncol=2, loc="best")
        if key == "beta":
            ax.set_xlabel("note: the risk map itself shifts with beta "
                          "(background = beta_true reference)", fontsize=8)
    fig.suptitle("Trajectory perturbations: one parameter swept over its full "
                 "in-bounds range (others fixed)\n"
                 f"deterministic CVaR, shared terrain, true W={W_TRUE} beta={BETA_TRUE}",
                 fontsize=13)
    f2 = os.path.join(OUTPUT_DIR, "sensitivity_trajectories.png")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f2, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[sens] saved {f2}")

    # ranking
    print("\n[sens] sensitivity ranking (max drift across each sweep):")
    for label, key in sorted(PARAMS, key=lambda lk: -max(drift[lk[1]])):
        imax = int(np.argmax(drift[key]))
        print(f"   {label:<22} max drift = {max(drift[key]):8.1f} "
              f"(at n={FACTORS[key][imax]})")


if __name__ == "__main__":
    main()
