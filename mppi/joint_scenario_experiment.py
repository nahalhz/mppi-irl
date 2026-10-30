"""Joint recovery vs number of scenarios (10 / 15 / 20)."""
import os
import numpy as np
import matplotlib.pyplot as plt

import skid_steer_bi_rrt_star_CVaR_MPPI as mppi
import BO_MPPI_joint as J

# new folder so joint_sweep_results/ is kept
OUTPUT_DIR      = "joint_sweep_results_v2"
SCENARIO_COUNTS = [10, 15, 20]
N_ITERS         = J.N_BO_ITERS  # + J.N_INIT sobol lower for a smoke test

# 5 more scenarios (15 -> 20)
EXTRA5 = [
    {"tag": "sc16_c2sw",  "name": "centre to SW",
     "seed": 6015, "start": np.array([5.0, 5.0,  np.pi/4]),  "goal": np.array([1.0, 1.0, -3*np.pi/4])},
    {"tag": "sc17_upright","name": "up and slightly right",
     "seed": 6016, "start": np.array([4.0, 1.0,  np.pi/2]),  "goal": np.array([6.0, 9.0,  np.pi/2])},
    {"tag": "sc18_ne2sw", "name": "NE to SW long",
     "seed": 6017, "start": np.array([9.0, 8.0,  np.pi]),    "goal": np.array([2.0, 3.0, -np.pi/2])},
    {"tag": "sc19_shallow_ur","name": "shallow up-right",
     "seed": 6018, "start": np.array([1.0, 4.0,  0.0]),      "goal": np.array([7.0, 9.0,  np.pi/4])},
    {"tag": "sc20_downleft","name": "down and left",
     "seed": 6019, "start": np.array([8.0, 6.0, -3*np.pi/4]),"goal": np.array([3.0, 2.0, -np.pi/2])},
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[sweep] CVAR_MODE={mppi.CVAR_MODE}  true W={J.W_TRUE}  beta={J.BETA_CVAR_TRUE}")

    # one cost map for every run
    np.random.seed(J.COST_MAP_SEED)
    cost_map = mppi.generate_cost_map(mppi.RISK_MAP_DIM, mppi.RISK_MAP_DIM)

    # 20 experts built once
    J.SCENARIOS = J.SCENARIOS[:15] + EXTRA5
    assert len(J.SCENARIOS) == 20, f"expected 20 scenarios, got {len(J.SCENARIOS)}"
    all_experts = J.prepare_experts(cost_map)

    # joint fit for each N
    results = []
    for N in SCENARIO_COUNTS:
        experts = all_experts[:N]
        print(f"\n{'='*60}\n[sweep] JOINT over {N} scenarios "
              f"({(J.N_INIT + N_ITERS) * N} eval-rollouts)\n{'='*60}")
        W_hat, beta_hat, eval_log = J.fit_joint(
            cost_map, experts, n_iters=N_ITERS, verbose=False,
        )
        best = min(h["traj_loss"] for h in eval_log if "__ls__" not in h)
        results.append({"N": N, "W": W_hat, "beta": beta_hat, "loss": best})
        grid = os.path.join(OUTPUT_DIR, f"joint_N{N}_grid.png")
        J.plot_trajectory_grid(cost_map, experts, W_hat, beta_hat, grid)
        conv = os.path.join(OUTPUT_DIR, f"joint_N{N}_conv.png")
        J.plot_convergence(eval_log, conv, N)
        print(f"[sweep] N={N}: W={np.round(W_hat, 4)}  "
              f"beta={beta_hat:.4f}  loss={best:.2f}")

    # recovered value vs N
    Ns     = [r["N"]        for r in results]
    series = {
        "W[1] (true 10)":      ([r["W"][1] for r in results], J.W_TRUE[1]),
        "W[2] (true 0.01)":    ([r["W"][2] for r in results], J.W_TRUE[2]),
        "W[3] (true 0.5)":     ([r["W"][3] for r in results], J.W_TRUE[3]),
        "beta_cvar (true 0.5)":([r["beta"] for r in results], J.BETA_CVAR_TRUE),
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (label, (vals, true)) in zip(axes.ravel(), series.items()):
        ax.plot(Ns, vals, "-o", lw=2, ms=7, color="#377eb8", label="recovered")
        ax.axhline(true, color="green", ls="--", lw=2, label="true")
        ax.set_xticks(Ns); ax.set_xlabel("number of scenarios N")
        ax.set_ylabel("recovered value"); ax.set_title(label)
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("Joint recovery vs scenario count (fixed map, same first-N demos)\n"
                 f"CVAR_MODE={mppi.CVAR_MODE}", fontsize=13)
    out = os.path.join(OUTPUT_DIR, "joint_sweep_recovery_vs_N.png")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"\n[sweep] saved {out}")

    # summary
    print(f"\n{'N':>4}  {'W1':>8} {'W2':>9} {'W3':>8} {'beta':>7}  {'loss':>9}")
    print(f"{'true':>4}  {J.W_TRUE[1]:>8.3f} {J.W_TRUE[2]:>9.4f} "
          f"{J.W_TRUE[3]:>8.3f} {J.BETA_CVAR_TRUE:>7.3f}")
    for r in results:
        print(f"{r['N']:>4}  {r['W'][1]:>8.3f} {r['W'][2]:>9.4f} "
              f"{r['W'][3]:>8.3f} {r['beta']:>7.3f}  {r['loss']:>9.2f}")


if __name__ == "__main__":
    main()
