"""Identifiability checks on the recovery loss: Hessian at the truth + pairwise heatmaps."""
import os
import numpy as np
import matplotlib.pyplot as plt

import skid_steer_bi_rrt_star_CVaR_MPPI as mppi
import BO_MPPI_joint as J
from BO_MPPI_SVGP import (run_mppi, trajectory_distance,
                          LW1_BOUNDS, LW2_BOUNDS, LW3_BOUNDS, BETA_BOUNDS)

OUT_DIR = "identifiability_results"
W_CTRL  = J.W_CTRL_LOSS
SCEN  = J.SCENARIOS[0]
START, GOAL, SEED = SCEN["start"], SCEN["goal"], SCEN["seed"]
MAX_STEPS, N_SAMPLES = J.EXPERT_STEPS, J.EXPERT_SAMPLES

# search-space coords - theta = (lw1, lw2, lw3, beta)
BOUNDS     = np.array([LW1_BOUNDS, LW2_BOUNDS, LW3_BOUNDS, BETA_BOUNDS], float)
SPAN       = BOUNDS[:, 1] - BOUNDS[:, 0]
THETA_TRUE = np.array([np.log(J.W_TRUE[1]), np.log(J.W_TRUE[2]),
                       np.log(J.W_TRUE[3]), J.BETA_CVAR_TRUE])
NAMES = ["lw1 (path-track)", "lw2 (control)", "lw3 (terminal)", "beta (risk)"]
HEAT_GRID = 13  # heatmap points per axis
H_STEP    = 0.06  # hessian step fraction of axis range

_cost = _noise = _exp_path = _exp_u = None


def setup():
    global _cost, _noise, _exp_path, _exp_u
    np.random.seed(J.COST_MAP_SEED)
    _cost = mppi.generate_cost_map(mppi.RISK_MAP_DIM, mppi.RISK_MAP_DIM)
    _noise = J.make_shared_noise()
    _exp_path, _exp_u, _, _ = run_mppi(
        np.array([1.0, J.W_TRUE[1], J.W_TRUE[2], J.W_TRUE[3]]), J.BETA_CVAR_TRUE,
        _cost, _noise, START, GOAL, max_steps=MAX_STEPS, n_samples=N_SAMPLES, seed=SEED)
    print(f"[id] expert {len(_exp_path)} steps")


def pad(u, n):
    return u[:n] if len(u) >= n else np.vstack([u, np.zeros((n - len(u), 2))])


def loss(theta):
    """Search-space theta -> clipped params."""
    th = np.clip(theta, BOUNDS[:, 0], BOUNDS[:, 1])
    W = np.array([1.0, np.exp(th[0]), np.exp(th[1]), np.exp(th[2])])
    path, u, _, _ = run_mppi(W, th[3], _cost, _noise, START, GOAL,
                                        max_steps=MAX_STEPS, n_samples=N_SAMPLES, seed=SEED)
    dt = trajectory_distance(_exp_path, path)
    n = max(len(_exp_u), len(u))
    eff = np.mean(np.sum((pad(_exp_u, n) - pad(u, n)) ** 2, axis=1))
    return dt + W_CTRL * eff / (2 * mppi.W_MAX ** 2 + 1e-8)


# hessian
def hessian():
    h = H_STEP * SPAN
    f0 = loss(THETA_TRUE)
    print(f"[id] loss at true = {f0:.4f}")
    H = np.zeros((4, 4))
    # diagonal
    fp, fm = {}, {}
    for i in range(4):
        ei = np.zeros(4); ei[i] = h[i]
        fp[i] = loss(THETA_TRUE + ei); fm[i] = loss(THETA_TRUE - ei)
        H[i, i] = (fp[i] - 2 * f0 + fm[i]) / (h[i] ** 2)
    # off-diagonal
    for i in range(4):
        for j in range(i + 1, 4):
            e = np.zeros(4); e[i] = h[i]; f = np.zeros(4); f[j] = h[j]
            fpp = loss(THETA_TRUE + e + f); fpm = loss(THETA_TRUE + e - f)
            fmp = loss(THETA_TRUE - e + f); fmm = loss(THETA_TRUE - e - f)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * h[i] * h[j])
    # h is already in range units
    w, V = np.linalg.eigh(H)
    order = np.argsort(w)[::-1]  # stiff -> sloppy
    w, V = w[order], V[:, order]
    np.savez(os.path.join(OUT_DIR, "hessian.npz"), H=H, eigvals=w, eigvecs=V,
             theta_true=THETA_TRUE, names=np.array(NAMES))
    print("\n[id] Hessian eigenvalues (stiff -> sloppy):")
    for k in range(4):
        comp = "  ".join(f"{NAMES[m].split()[0]}:{V[m,k]:+.2f}" for m in range(4))
        print(f"   lambda_{k+1} = {w[k]:.4g}   dir: {comp}")
    cond = w[0] / max(w[-1], 1e-12)
    print(f"[id] condition number (stiff/sloppy) = {cond:.4g}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].bar(range(1, 5), np.maximum(w, 1e-6), color="#1A1F2B")
    ax[0].set_yscale("log"); ax[0].set_xticks(range(1, 5))
    ax[0].set_xlabel("eigen-direction (stiff → sloppy)")
    ax[0].set_ylabel("curvature (eigenvalue, log)")
    ax[0].set_title(f"Loss curvature at the true params\ncondition number ≈ {cond:.0f} "
                    f"(stiff/sloppy)", fontsize=12)
    ax[0].grid(alpha=0.3, axis="y")
    im = ax[1].imshow(np.abs(V.T), cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels([n.split()[0] for n in NAMES])
    ax[1].set_yticks(range(4))
    ax[1].set_yticklabels([f"dir {k+1}\nλ={w[k]:.2g}" for k in range(4)])
    for a in range(4):
        for b in range(4):
            ax[1].text(b, a, f"{V[b,a]:+.2f}", ha="center", va="center", fontsize=9,
                       color="white" if abs(V[b, a]) > 0.6 else "black")
    ax[1].set_title("Eigen-directions: which weight combo each is\n"
                    "(bottom row = sloppiest = least identifiable)", fontsize=12)
    fig.colorbar(im, ax=ax[1], label="|component|")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "hessian_identifiability.png")
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[id] saved {out}")


# heatmaps
def heatmaps():
    pairs = [(0, 1), (0, 2), (1, 3)]
    # (lw2, lw3) too both weak
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    store = {}
    for ax, (i, j) in zip(axes, pairs):
        gi = np.linspace(BOUNDS[i, 0], BOUNDS[i, 1], HEAT_GRID)
        gj = np.linspace(BOUNDS[j, 0], BOUNDS[j, 1], HEAT_GRID)
        Lg = np.zeros((HEAT_GRID, HEAT_GRID))
        print(f"\n[id] heatmap {NAMES[i]} x {NAMES[j]} ({HEAT_GRID}x{HEAT_GRID})")
        for a, vi in enumerate(gi):
            for b, vj in enumerate(gj):
                th = THETA_TRUE.copy(); th[i] = vi; th[j] = vj
                Lg[a, b] = loss(th)
            print(f"   row {a+1}/{HEAT_GRID} done")
        store[f"{i}_{j}_gi"] = gi; store[f"{i}_{j}_gj"] = gj; store[f"{i}_{j}_L"] = Lg
        cs = ax.contourf(gj, gi, np.log10(Lg + 1.0), levels=18, cmap="viridis")
        ax.plot(THETA_TRUE[j], THETA_TRUE[i], "*", color="#E07A5F", ms=20,
                markeredgecolor="white", markeredgewidth=1.3, label="true")
        ax.set_xlabel(NAMES[j]); ax.set_ylabel(NAMES[i])
        ax.set_title(f"{NAMES[i].split()[0]} vs {NAMES[j].split()[0]}", fontsize=12)
        fig.colorbar(cs, ax=ax, label="log10(loss+1)")
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("Pairwise recovery-loss landscape (other params at true) — "
                 "round basin = identifiable, tilted valley = confounded pair", fontsize=13)
    np.savez(os.path.join(OUT_DIR, "heatmaps.npz"), **store, names=np.array(NAMES))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_DIR, "interaction_heatmaps.png")
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[id] saved {out}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    assert mppi.CVAR_MODE == "deterministic"
    setup()
    print("\n===== #2 HESSIAN =====")
    hessian()
    print("\n===== #1 HEATMAPS =====")
    heatmaps()
    print("\n[id] ALL DONE")


if __name__ == "__main__":
    main()
