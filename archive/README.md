# Archive + timeline

Each script here is a version with different core logic: a new optimizer, loss,
acquisition, surrogate, or sim. Sweep drivers, diagnostics and near-duplicates
aren't included. Run from inside the folder, same as before.

Dates come from the NRC update decks.

## 1. Simple grid, first inverse tries (Feb – early Mar) - `grid_inverse/`

| Script | What changed |
|---|---|
| `inverse_feature_static.py` | feature matching (risk, dist, u², terminal), scipy Powell, fixed alpha |
| `inverse_feature.py` | 5 features (+ path length, max risk), Nelder-Mead, alpha schedule like the forward script |
| `inverse_fit_demo.py` | features → trajectory error + goal penalties, Nelder-Mead on log W |
| `inverse_mpc.py` | MaxEnt IRL try |
| `inverse_risk_fit.py` | grid search over alpha, fit weights per alpha (theta) |

## 2. Bayesian optimization on the grid (Mar 9 – Apr) - `grid_inverse/`

| Script | What changed | Deck |
|---|---|---|
| `inverse_feature_BO.py` | Powell → `bayes_opt` over (lw2, lw3, alpha), W fixed to [1, 0, W2, W3], per-timestep trajectory loss with padding, inverse-kinematics control reconstruction | Mar 9–13, Mar 16–27, Apr 13 |
| `inverse_OPTIMIZED2.py` | candidate W actually goes into the CEM cost, batched rollouts + fast truncated-normal sampler (BO run was ~2 h at dt = 0.01) | Apr (late) |

Also in this stretch: dt 0.1 → 0.05 → 0.01, and W_CTRL 0.5 → 0.1 → 0.

## 3. Static map + BoTorch (May) - `static_bo/`

| Script | What changed | Deck |
|---|---|---|
| `skid_steer_simple_static.py`, `inverse_feature_BO_optimized.py` | static cost map locked to seed 2, same BO as `inverse_OPTIMIZED2` | May 15 |
| `BO_SAAS.py` | SAASBO (fully Bayesian GP, NUTS) + qLogEI. Didn't fix the W2/W3 identifiability | May 15 |
| `BO_SVGP.py` | exact GP + qLogEI | May 26 |
| `total_SVGP.py` | **variational GP (SVGP) + UCB**. EI → UCB with β for exploration, 32 → 64 inducing points, 150 → 300 epochs, multi-scenario runs | May 26 |
| `BO_chamfer.py` | chamfer trajectory distance instead of padding | May 26 |
| `generate_dataset.py` | side try: dataset of (W, alpha) → rollouts for amortized IRL | – |

## 4. MPPI planner (May 26 – Jul 20) - `../mppi/`

These all live in `BO_MPPI_SVGP.py`, which was edited in place, so only the
final version exists (scheduler + length-scale sampling + UCB). The stages
below are from the July 20 deck.

1. First MPPI integration: bi-RRT* + CVaR-MPPI planner, one scenario, β sweep.
2. **Cosine LR scheduler** on the GP fit (0.01 → 1e-4 per refit), 300 epochs, 4 scenarios.
3. **Length-scale sampling**: candidates drawn from N(best, ls²), reflected into bounds. 10 scenarios, 400 evals.
4. Objective bugs found. Every eval reseeded (bi-RRT* path changed each time), and the expert ran 2000 MPPI samples vs 500 for evals.
5. Fixed seed + matched sample counts, so the objective is deterministic.
6. 2000 samples with length-scale sampling.
7. Plain UCB vs length-scale sampling A/B: mixed, no clear winner.
8. Deterministic closed-form per-cell CVaR (`CVAR_MODE`), ~5× faster.
9. Joint recovery across scenarios (`BO_MPPI_joint.py`), 10 and 15 scenarios, plus `repro_study.py`.

## 5. Shared map + scaling scenarios (Oct 15) - `../mppi/`

- One shared risk map for every scenario; only start/goal change.
- 10 / 15 / 20 scenarios × 3 runs (`joint_scenario_experiment.py`, `joint_scenario_repro.py`).
- `sensitivity_study.py` checks what actually moves the path.

## 6. Since Oct 15 - `../mppi/`

`identifiability.py`, `value_function.py`, `value_planner_cost.py`,
`tier2_experiments.py` (+ `make_expC_fig.py`). Not in a deck yet.
