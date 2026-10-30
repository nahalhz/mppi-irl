"""
Tier-2 "fix" experiments for weight identifiability.
====================================================
Motivated by the identifiability diagnostics: beta is clean, control W2 is
the sloppiest, and W1 (path-track) <-> W3 (terminal) are confounded.

A. FIX W1, recover {W2, W3, beta}.  Pin lw1 to a tight band around log(10)
   and run the normal joint BO.  If W3 now lands near 0.5, removing the
   dominant axis frees the confounded pair.

B. JOINT lw1 x lw3 heatmap (10 scenarios).  The single-scenario landscape
   had a tilted W1-W3 valley.  Does summing over 10 scenarios round it out
   (un-tilt), i.e. do diverse demos break the confounding?

C. ADD A HEADING SIGNAL.  Recompute the local Hessian with position-only vs
   position+heading matching (one shared stencil).  If adding heading drops
   the condition number and separates the lw1/lw3 eigen-directions, the
   extra signal breaks the tie.

Outputs -> tier2_results/ (summary.txt + two figures).
Run from inside mppi/ :  python3 tier2_experiments.py
"""
import os, math
import numpy as np
import matplotlib.pyplot as plt

import skid_steer_bi_rrt_star_CVaR_MPPI as mppi
import BO_MPPI_joint as J
from BO_MPPI_SVGP import (run_mppi, trajectory_distance,
                          LW1_BOUNDS, LW2_BOUNDS, LW3_BOUNDS, BETA_BOUNDS)

OUT = "tier2_results"
N_SCEN_JOINT = 10
W_CTRL = J.W_CTRL_LOSS
BOUNDS = np.array([LW1_BOUNDS, LW2_BOUNDS, LW3_BOUNDS, BETA_BOUNDS], float)
SPAN   = BOUNDS[:, 1] - BOUNDS[:, 0]
THETA_TRUE = np.array([math.log(J.W_TRUE[1]), math.log(J.W_TRUE[2]),
                       math.log(J.W_TRUE[3]), J.BETA_CVAR_TRUE])
NAMES = ["lw1", "lw2", "lw3", "beta"]
summary = []


def log(s):
    print(s); summary.append(s)


def build_cost():
    np.random.seed(J.COST_MAP_SEED)
    return mppi.generate_cost_map(mppi.RISK_MAP_DIM, mppi.RISK_MAP_DIM)


# ───────── single-scenario loss machinery (for Exp C) ─────────
def _angdiff(a, b):
    d = a - b
    return np.arctan2(np.sin(d), np.cos(d))


def pad_rows(arr, n):
    if len(arr) >= n:
        return arr[:n]
    return np.vstack([arr, np.tile(arr[-1], (n - len(arr), 1))])


def single_components(theta, cost, noise, exp_path, exp_u, start, goal, seed):
    th = np.clip(theta, BOUNDS[:, 0], BOUNDS[:, 1])
    W = np.array([1.0, math.exp(th[0]), math.exp(th[1]), math.exp(th[2])])
    path, u, _, _ = run_mppi(W, th[3], cost, noise, start, goal,
                                        max_steps=J.EXPERT_STEPS,
                                        n_samples=J.EXPERT_SAMPLES, seed=seed)
    pos = trajectory_distance(exp_path, path)
    n = max(len(exp_path), len(path))
    ha = pad_rows(exp_path[:, 2:3], n)[:, 0]; hb = pad_rows(path[:, 2:3], n)[:, 0]
    head = float(np.sum(np.abs(_angdiff(ha, hb))))
    m = max(len(exp_u), len(u))
    eff = np.mean(np.sum((pad_rows(exp_u, m) - pad_rows(u, m)) ** 2, axis=1))
    ctrl = W_CTRL * eff / (2 * mppi.W_MAX ** 2 + 1e-8)
    return pos, ctrl, head


def hessian_from(vals, f0, h):
    """Build 4x4 Hessian from a dict of stencil loss values."""
    H = np.zeros((4, 4))
    for i in range(4):
        H[i, i] = (vals[("+", i)] - 2 * f0 + vals[("-", i)]) / (h[i] ** 2)
    for i in range(4):
        for j in range(i + 1, 4):
            H[i, j] = H[j, i] = (vals[("++", i, j)] - vals[("+-", i, j)]
                                 - vals[("-+", i, j)] + vals[("--", i, j)]) / (4 * h[i] * h[j])
    return H


def cond_and_dirs(H):
    w, V = np.linalg.eigh(H)
    o = np.argsort(w)[::-1]
    return w[o], V[:, o]


# ===================== EXPERIMENT A =====================
def exp_A(cost):
    log("\n===== EXP A: fix W1, recover {W2,W3,beta} (10 scenarios) =====")
    J.SCENARIOS = J.SCENARIOS[:N_SCEN_JOINT]
    experts = J.prepare_experts(cost)
    lw1_true = math.log(J.W_TRUE[1])
    orig = J.LW1_BOUNDS
    J.LW1_BOUNDS = (lw1_true - 0.05, lw1_true + 0.05)   # W1 pinned to ~[9.5,10.5]
    try:
        W_hat, beta_hat, _ = J.fit_joint(cost, experts, n_init=30, n_iters=80, verbose=False)
    finally:
        J.LW1_BOUNDS = orig
    log(f"  TRUE   W=[1, {J.W_TRUE[1]}, {J.W_TRUE[2]}, {J.W_TRUE[3]}]  beta={J.BETA_CVAR_TRUE}")
    log(f"  FIXED-W1 recovered: W1={W_hat[1]:.3f} (pinned)  W2={W_hat[2]:.5f}  "
        f"W3={W_hat[3]:.4f}  beta={beta_hat:.4f}")
    log(f"  -> W3 err vs true 0.5: {abs(W_hat[3]-0.5):.4f}   W2 err vs 0.01: {abs(W_hat[2]-0.01):.4f}")
    return experts


# ===================== EXPERIMENT B =====================
def exp_B(cost, experts):
    log("\n===== EXP B: JOINT lw1 x lw3 heatmap (10 scenarios) =====")
    elog = []
    obj = J._make_joint_objective(cost, experts, elog, W_CTRL)
    G = 11
    lw1g = np.linspace(*LW1_BOUNDS, G); lw3g = np.linspace(*LW3_BOUNDS, G)
    L = np.zeros((G, G))
    lw2t, bt = math.log(J.W_TRUE[2]), J.BETA_CVAR_TRUE
    for a, v1 in enumerate(lw1g):
        for b, v3 in enumerate(lw3g):
            L[a, b] = obj(v1, lw2t, v3, bt)
        print(f"  [B] row {a+1}/{G}")
    np.savez(os.path.join(OUT, "joint_heatmap.npz"), lw1g=lw1g, lw3g=lw3g, L=L)

    # side-by-side with the single-scenario one (if present)
    single = None
    sp = "identifiability_results/heatmaps.npz"
    if os.path.exists(sp):
        d = np.load(sp, allow_pickle=True)
        if "0_2_L" in d:
            single = (d["0_2_gi"], d["0_2_gj"], d["0_2_L"])
    n = 2 if single else 1
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5), squeeze=False)
    cs = axes[0, 0].contourf(lw3g, lw1g, np.log10(L + 1), levels=18, cmap="viridis")
    axes[0, 0].plot(math.log(J.W_TRUE[3]), math.log(J.W_TRUE[1]), "*", color="#E07A5F",
                    ms=20, markeredgecolor="white", markeredgewidth=1.3)
    axes[0, 0].set_xlabel("lw3 (terminal)"); axes[0, 0].set_ylabel("lw1 (path-track)")
    axes[0, 0].set_title(f"JOINT ({N_SCEN_JOINT} scenarios)"); fig.colorbar(cs, ax=axes[0, 0])
    if single:
        gi, gj, Ls = single
        cs2 = axes[0, 1].contourf(gj, gi, np.log10(Ls + 1), levels=18, cmap="viridis")
        axes[0, 1].plot(math.log(J.W_TRUE[3]), math.log(J.W_TRUE[1]), "*", color="#E07A5F",
                        ms=20, markeredgecolor="white", markeredgewidth=1.3)
        axes[0, 1].set_xlabel("lw3 (terminal)"); axes[0, 1].set_ylabel("lw1 (path-track)")
        axes[0, 1].set_title("SINGLE scenario (for comparison)"); fig.colorbar(cs2, ax=axes[0, 1])
    fig.suptitle("W1 x W3 recovery-loss valley: does averaging scenarios un-tilt it?", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUT, "joint_vs_single_W1W3.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    # quantify tilt: direction of the min-loss ridge via the Hessian of a quadratic fit
    log(f"  joint heatmap saved; loss range {L.min():.1f}..{L.max():.1f}")


# ===================== EXPERIMENT C =====================
def exp_C(cost):
    log("\n===== EXP C: does a heading signal break the W1-W3 tie? =====")
    scen = J.SCENARIOS_FULL[0] if hasattr(J, "SCENARIOS_FULL") else J.SCENARIOS[0]
    start, goal, seed = scen["start"], scen["goal"], scen["seed"]
    noise = J.make_shared_noise()
    exp_path, exp_u, _, _ = run_mppi(
        np.array([1.0, J.W_TRUE[1], J.W_TRUE[2], J.W_TRUE[3]]), J.BETA_CVAR_TRUE,
        cost, noise, start, goal, max_steps=J.EXPERT_STEPS,
        n_samples=J.EXPERT_SAMPLES, seed=seed)

    h = 0.06 * SPAN
    def comp(theta):
        return single_components(theta, cost, noise, exp_path, exp_u, start, goal, seed)

    pts = {}
    pts["0"] = comp(THETA_TRUE)
    for i in range(4):
        e = np.zeros(4); e[i] = h[i]
        pts[("+", i)] = comp(THETA_TRUE + e); pts[("-", i)] = comp(THETA_TRUE - e)
    for i in range(4):
        for j in range(i + 1, 4):
            e = np.zeros(4); e[i] = h[i]; f = np.zeros(4); f[j] = h[j]
            pts[("++", i, j)] = comp(THETA_TRUE + e + f)
            pts[("+-", i, j)] = comp(THETA_TRUE + e - f)
            pts[("-+", i, j)] = comp(THETA_TRUE - e + f)
            pts[("--", i, j)] = comp(THETA_TRUE - e - f)

    # auto-scale heading weight so its average stencil variation matches position's
    pos0, ctrl0, head0 = pts["0"]
    dpos = np.mean([abs(v[0] - pos0) for k, v in pts.items() if k != "0"])
    dhead = np.mean([abs(v[2] - head0) for k, v in pts.items() if k != "0"]) + 1e-9
    w_head = dpos / dhead
    log(f"  auto heading weight w_head = {w_head:.3f} (equalises avg stencil variation)")

    def build(weighted_head):
        f0 = pos0 + ctrl0 + (w_head * head0 if weighted_head else 0.0)
        vals = {}
        for k, v in pts.items():
            if k == "0":
                continue
            vals[k] = v[0] + v[1] + (w_head * v[2] if weighted_head else 0.0)
        return hessian_from(vals, f0, h)

    for tag, wh in [("position-only", False), ("position+heading", True)]:
        H = build(wh)
        w, V = cond_and_dirs(H)
        cond = w[0] / max(w[-1], 1e-12)
        log(f"  [{tag}] eigenvalues={np.round(w,1).tolist()}  cond={cond:.1f}")
        for k in range(4):
            comps = "  ".join(f"{NAMES[m]}:{V[m,k]:+.2f}" for m in range(4))
            log(f"       dir{k+1} lam={w[k]:8.1f}  {comps}")
        # measure W1-W3 mixing in the two middle directions (how coupled lw1,lw3 are)
        mix = np.mean([abs(V[0, 1] * V[2, 1]), abs(V[0, 2] * V[2, 2])])
        log(f"       W1-W3 mixing index (mid dirs) = {mix:.3f}  (lower = more separated)")


def main():
    os.makedirs(OUT, exist_ok=True)
    assert mppi.CVAR_MODE == "deterministic"
    cost = build_cost()
    experts = exp_A(cost)
    exp_B(cost, experts)
    exp_C(cost)
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")
    print("\n[tier2] ALL DONE -> tier2_results/summary.txt")


if __name__ == "__main__":
    main()
