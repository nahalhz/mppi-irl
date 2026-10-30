"""Reproducibility study for the joint MPPI inverse-RL recovery."""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

import skid_steer_bi_rrt_star_CVaR_MPPI as mppi
import BO_MPPI_joint as j

# config
K        = 5  # torch seeds 0..K-1
N_SCEN   = 5
N_ITERS  = 150
OUT_DIR  = "repro_results"
TAG      = f"repro_K{K}_scen{N_SCEN}_it{N_ITERS}"


def main():
    assert mppi.CVAR_MODE == "deterministic", \
        "run with CVAR_MODE='deterministic' so the objective is fixed"
    os.makedirs(OUT_DIR, exist_ok=True)

    np.random.seed(j.COST_MAP_SEED)
    cost_map = mppi.generate_cost_map(mppi.RISK_MAP_DIM, mppi.RISK_MAP_DIM)

    j.SCENARIOS = j.SCENARIOS[:N_SCEN]
    experts = j.prepare_experts(cost_map)

    print(f"\n[repro] {K} repeats · {N_SCEN} scenarios · {N_ITERS} BO iters")
    print(f"[repro] true W={j.W_TRUE}  beta_cvar={j.BETA_CVAR_TRUE}\n")

    rows = []
    for k in range(K):
        torch.manual_seed(k)  # only source of variation
        print(f"\n{'='*55}\n[repro] repeat {k+1}/{K}  (torch seed {k})\n{'='*55}")
        W_hat, beta_hat, eval_log = j.fit_joint(
            cost_map, experts, n_iters=N_ITERS, verbose=False,
        )
        best = min(h["traj_loss"] for h in eval_log if "__ls__" not in h)
        rows.append({
            "seed": k, "W1": W_hat[1], "W2": W_hat[2], "W3": W_hat[3],
            "beta_cvar": beta_hat, "loss": best,
        })
        print(f"[repro] repeat {k}: W={np.round(W_hat,4)}  "
              f"beta={beta_hat:.4f}  Σloss={best:.3f}")

    # summary
    true = {"W1": j.W_TRUE[1], "W2": j.W_TRUE[2], "W3": j.W_TRUE[3],
            "beta_cvar": j.BETA_CVAR_TRUE}
    keys = ["W1", "W2", "W3", "beta_cvar", "loss"]
    print(f"\n{'='*60}\n[repro] SUMMARY over {K} repeats\n{'='*60}")
    print(f"{'param':10} {'true':>10} {'mean':>10} {'std':>10} {'cv%':>7}")
    stats = {}
    for key in keys:
        vals = np.array([r[key] for r in rows])
        m, s = vals.mean(), vals.std()
        cv = 100 * s / abs(m) if m != 0 else float("nan")
        stats[key] = (m, s, cv, vals)
        t = true.get(key, float("nan"))
        print(f"{key:10} {t:>10.4f} {m:>10.4f} {s:>10.4f} {cv:>6.1f}%")

    # recovered vs true per repeat
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, key in zip(axes, ["W1", "W2", "W3", "beta_cvar"]):
        m, s, cv, vals = stats[key]
        ax.scatter(range(K), vals, s=70, color="#377eb8", zorder=3, label="per repeat")
        ax.axhline(true[key], color="green", ls="--", lw=2, label="true")
        ax.axhline(m, color="#e07a5f", ls="-", lw=1.5, label=f"mean (cv={cv:.0f}%)")
        ax.fill_between([-0.5, K-0.5], m-s, m+s, color="#e07a5f", alpha=0.15)
        ax.set_xlim(-0.5, K-0.5); ax.set_xticks(range(K))
        ax.set_xlabel("repeat (torch seed)"); ax.set_title(key)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(
        f"Reproducibility of joint recovery — {K} repeats, deterministic objective\n"
        f"spread = BO search variance only (same (W,β) always gives the same loss)",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    out = os.path.join(OUT_DIR, f"{TAG}.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[repro] plot saved → {out}")


if __name__ == "__main__":
    main()
