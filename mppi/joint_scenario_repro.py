"""Scenario-count sweep (10 / 15 / 20) repeated K times for error bars."""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

import skid_steer_bi_rrt_star_CVaR_MPPI as mppi
import BO_MPPI_joint as J

OUTPUT_DIR      = "joint_repro_results"
SCENARIO_COUNTS = [10, 15, 20]
K               = 3  # torch seeds 0..K-1
N_ITERS         = 80  # ~4.4 h J.N_BO_ITERS (250) for the full budget

# same 5 extras as joint_scenario_experiment.py
EXTRA5 = [
    {"tag": "sc16_c2sw",   "name": "centre to SW",
     "seed": 6015, "start": np.array([5.0, 5.0,  np.pi/4]),  "goal": np.array([1.0, 1.0, -3*np.pi/4])},
    {"tag": "sc17_upright","name": "up and slightly right",
     "seed": 6016, "start": np.array([4.0, 1.0,  np.pi/2]),  "goal": np.array([6.0, 9.0,  np.pi/2])},
    {"tag": "sc18_ne2sw",  "name": "NE to SW long",
     "seed": 6017, "start": np.array([9.0, 8.0,  np.pi]),    "goal": np.array([2.0, 3.0, -np.pi/2])},
    {"tag": "sc19_shallow_ur","name": "shallow up-right",
     "seed": 6018, "start": np.array([1.0, 4.0,  0.0]),      "goal": np.array([7.0, 9.0,  np.pi/4])},
    {"tag": "sc20_downleft","name": "down and left",
     "seed": 6019, "start": np.array([8.0, 6.0, -3*np.pi/4]),"goal": np.array([3.0, 2.0, -np.pi/2])},
]

KEYS = [("W1", J.W_TRUE[1], "W[1]  (true 10)"),
        ("W2", J.W_TRUE[2], "W[2]  (true 0.01)"),
        ("W3", J.W_TRUE[3], "W[3]  (true 0.5)"),
        ("beta", J.BETA_CVAR_TRUE, "beta  (true 0.5)")]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[repro] CVAR_MODE={mppi.CVAR_MODE}  K={K} repeats  N={SCENARIO_COUNTS}  "
          f"N_ITERS={N_ITERS}")

    # shared map + 20 experts
    np.random.seed(J.COST_MAP_SEED)
    cost_map = mppi.generate_cost_map(mppi.RISK_MAP_DIM, mppi.RISK_MAP_DIM)
    J.SCENARIOS = J.SCENARIOS[:15] + EXTRA5
    all_experts = J.prepare_experts(cost_map)

    # samples[N][key] -> K recovered values
    samples = {N: {k: [] for k, _, _ in KEYS} for N in SCENARIO_COUNTS}
    for N in SCENARIO_COUNTS:
        experts = all_experts[:N]
        for k in range(K):
            torch.manual_seed(k)  # only source of variation
            print(f"\n[repro] N={N}  repeat {k+1}/{K} (torch seed {k})")
            W_hat, beta_hat, _ = J.fit_joint(
                cost_map, experts, n_iters=N_ITERS, verbose=False)
            vals = {"W1": W_hat[1], "W2": W_hat[2], "W3": W_hat[3], "beta": beta_hat}
            for key, _, _ in KEYS:
                samples[N][key].append(vals[key])
            print(f"[repro] N={N} seed{k}: W={np.round(W_hat,4)} beta={beta_hat:.4f}")

    # mean +/- std per N
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (key, true, label) in zip(axes.ravel(), KEYS):
        means = [np.mean(samples[N][key]) for N in SCENARIO_COUNTS]
        stds  = [np.std(samples[N][key])  for N in SCENARIO_COUNTS]
        ax.errorbar(SCENARIO_COUNTS, means, yerr=stds, fmt="-o", lw=2, ms=7,
                    capsize=6, color="#377eb8", label=f"mean ± std (K={K})")
        for N in SCENARIO_COUNTS:
            ax.scatter([N]*K, samples[N][key], s=22, color="#377eb8", alpha=0.4, zorder=3)
        ax.axhline(true, color="green", ls="--", lw=2, label="true")
        ax.set_xticks(SCENARIO_COUNTS); ax.set_xlabel("number of scenarios N")
        ax.set_ylabel("recovered value"); ax.set_title(label)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle("Reproducible scenario sweep: recovered value vs N (mean ± std)\n"
                 f"{K} repeats per N, only the search seed changes — "
                 "an N trend is real only if the bars separate", fontsize=13)
    out = os.path.join(OUTPUT_DIR, "joint_repro_vs_N.png")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"\n[repro] saved {out}")

    # summary
    print(f"\n[repro] SUMMARY (mean ± std over K={K} repeats)")
    for key, true, label in KEYS:
        print(f"  {label}")
        for N in SCENARIO_COUNTS:
            v = np.array(samples[N][key])
            cv = 100 * v.std() / abs(v.mean()) if v.mean() else float("nan")
            print(f"     N={N:>2}:  {v.mean():.4f} ± {v.std():.4f}   (cv {cv:.0f}%)")


if __name__ == "__main__":
    main()
