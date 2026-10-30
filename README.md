# MPPI inverse-RL

Recovering risk preferences (cost weights `W` and CVaR level `β`) from planner
demonstrations.

Run scripts from inside their folder, e.g. `cd mppi && python BO_MPPI_joint.py`.
Outputs are written relative to the current directory.

| Folder | What |
|---|---|
| `mppi/` | current: bi-RRT* + CVaR-MPPI planner, BO recovery (per-scenario, joint), studies |
| `archive/` | earlier versions with different core logic, plus the [timeline](archive/README.md) |

## mppi

| To get | Run | Output |
|---|---|---|
| per-scenario recovery | `python BO_MPPI_SVGP.py` | `mppi_svgp_results/` |
| joint recovery | `python BO_MPPI_joint.py` | `mppi_joint_results/` |
| joint sweep over N | `python joint_scenario_experiment.py` | `joint_sweep_results_v2/` |
| joint repro over N | `python joint_scenario_repro.py` | `joint_repro_results/` |
| reproducibility | `python repro_study.py` | `repro_results/` |
| sensitivity | `python sensitivity_study.py` | `sensitivity_results_fullrange/` |
| identifiability | `python identifiability.py` | `identifiability_results/` |
| value function / planner cost | `python value_function.py`, `python value_planner_cost.py` | `value_function_results/` |
| tier 2 experiments | `python tier2_experiments.py` | `tier2_results/` |

Knobs: `CVAR_MODE` (`"deterministic"` | `"sampled"`) in the planner, `N_SCENARIOS`,
`W_CTRL_LOSS`, `UCB_BETA` in the BO scripts.

## Dependencies

numpy, scipy, matplotlib, torch, botorch, gpytorch, bayes_opt
