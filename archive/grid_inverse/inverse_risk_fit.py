"""Fit (w_r, w_u, w_T, alpha) to an expert trajectory by feature matching."""

from __future__ import annotations

import numpy as np

import skid_steer_simple_grid
from skid_steer_simple_grid import (
    simulate,
    cem_step,
    cvar_risk_map,
    GRID_CELL_SIZE,
    RISK_MAP_DIM,
    dt,
    T,
    W_MAX,
)

# features

def _state_to_ij(state_xy: np.ndarray) -> tuple[int, int]:
    """Convert (x,y) in metres to risk-map indices (i,j) with clipping."""
    i = int(np.clip(np.floor(state_xy[0] / GRID_CELL_SIZE), 0, RISK_MAP_DIM - 1))
    j = int(np.clip(np.floor(state_xy[1] / GRID_CELL_SIZE), 0, RISK_MAP_DIM - 1))
    return i, j


def compute_features(
    x: np.ndarray,
    u: np.ndarray,
    goal_state: np.ndarray,
    risk_map: np.ndarray,
    include_path_length: bool = True,
    include_max_risk: bool = True,
) -> np.ndarray:
    """Compute feature vector for a trajectory"""
    total_risk = 0.0
    total_control = 0.0
    max_risk = -np.inf

    # risk + control accumulation
    for t in range(len(x)):
        i, j = _state_to_ij(x[t, :2])
        r_ij = float(risk_map[i, j])
        total_risk += r_ij
        max_risk = max(max_risk, r_ij)

        # u and x share the horizon index
        if t < len(u):
            total_control += float(np.sum(np.square(u[t, :])))

    terminal_dist = float(np.linalg.norm(goal_state[:2] - x[-1, :2]))

    feats = [total_risk, total_control, terminal_dist]

    if include_path_length:
        diffs = x[1:, :2] - x[:-1, :2]
        path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
        feats.append(path_length)

    if include_max_risk:
        feats.append(float(max_risk if np.isfinite(max_risk) else 0.0))

    return np.array(feats, dtype=float)


def _normalize_features(phi: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Elementwise normalize with a positive scale vector"""
    return phi / np.maximum(scale, 1e-8)


# forward planning

def plan_with_theta(
    x0: np.ndarray,
    goal_state: np.ndarray,
    risk_map: np.ndarray,
    theta: np.ndarray,
    *,
    include_path_length: bool = True,
    include_max_risk: bool = True,
    mu_init: np.ndarray | None = None,
    sigma_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Runs one MPC planning rollout with given theta weights"""

    theta = np.asarray(theta, dtype=float).reshape(-1)

    # local cost with current theta
    def custom_cost(x, u, goal_state_inner, risk_map_inner):
        phi = compute_features(
            x, u, goal_state_inner, risk_map_inner,
            include_path_length=include_path_length,
            include_max_risk=include_max_risk,
        )
        scale = np.maximum(np.abs(phi), 1e-6)
        phi_norm = _normalize_features(phi, scale)
        return float(np.dot(theta, phi_norm))

    # defaults for CEM sampling distribution
    if mu_init is None:
        mu_init = np.zeros((int(np.ceil(T / dt)), 2))
    if sigma_init is None:
        sigma_init = np.ones((int(np.ceil(T / dt)), 2)) * W_MAX * 0.5

    # patch restore after
    original_cost = skid_steer_simple_grid.evaluate_cost
    skid_steer_simple_grid.evaluate_cost = custom_cost
    try:
        best_u, new_sigma = cem_step(
            x0=x0,
            goal_state=goal_state,
            risk_map=risk_map,
            mu_init=mu_init,
            sigma_init=sigma_init,
        )
        x_traj = simulate(x0, best_u)
    finally:
        skid_steer_simple_grid.evaluate_cost = original_cost

    return x_traj, best_u


# inverse fit

def fit_weights(
    x0: np.ndarray,
    goal_state: np.ndarray,
    cost_map: np.ndarray,
    expert_x: np.ndarray,
    expert_u: np.ndarray,
    alpha: float,
    *,
    n_iters: int = 40,
    lr: float = 0.002,
    include_path_length: bool = True,
    include_max_risk: bool = True,
    theta_init: np.ndarray | None = None,
    normalize_on: str = "expert",  # "expert" or "both"
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Apprenticeship-learning style update (Ratliff-lite): theta <- ProjectSimplex(theta - lr * (phi_robot - phi_expert))"""
    # risk map fixed by alpha
    risk_map = cvar_risk_map(RISK_MAP_DIM, RISK_MAP_DIM, alpha, cost_map)

    # theta on the simplex
    if theta_init is None:
        d = 3 + int(include_path_length) + int(include_max_risk)
        theta = np.ones(d, dtype=float)
        theta[0] = 1.0
        theta[1] = 0.2
        theta[2] = 1.5
        if include_path_length:
            theta[3] = 0.5
        if include_max_risk:
            theta[-1] = 0.8
    else:
        theta = np.asarray(theta_init, dtype=float).reshape(-1)

    theta = np.maximum(theta, 1e-6)
    theta[2] = max(theta[2], 0.2)  # min terminal weight
    theta = theta / np.sum(theta)

    # expert features (raw)
    phi_expert_raw = compute_features(
        expert_x, expert_u, goal_state, risk_map,
        include_path_length=include_path_length,
        include_max_risk=include_max_risk,
    )

    # scale by expert magnitudes
    scale = np.abs(phi_expert_raw)
    scale = np.maximum(scale, 1e-6)

    last_norm = None

    for it in range(n_iters):
        # forward plan with current theta
        x_robot, u_robot = plan_with_theta(
            x0, goal_state, risk_map, theta,
            include_path_length=include_path_length,
            include_max_risk=include_max_risk,
        )

        phi_robot_raw = compute_features(
            x_robot, u_robot, goal_state, risk_map,
            include_path_length=include_path_length,
            include_max_risk=include_max_risk,
        )

        # optionally rescale on both
        if normalize_on == "both":
            scale = np.maximum(np.abs(phi_expert_raw) + np.abs(phi_robot_raw), 1e-6)

        phi_expert = _normalize_features(phi_expert_raw, scale)
        phi_robot = _normalize_features(phi_robot_raw, scale)

        grad = phi_robot - phi_expert

        # gradient step
        theta = theta - lr * grad

        # project to positive simplex
        theta = np.maximum(theta, 1e-8)
        theta = theta / np.sum(theta)

        if verbose:
            diff_norm = float(np.linalg.norm(grad))
            if last_norm is None or it % 5 == 0:
                print(f"[iter {it:02d}] ||grad||={diff_norm:.4f} theta={theta}")
            last_norm = diff_norm

    # final trajectory
    x_final, _ = plan_with_theta(
        x0, goal_state, risk_map, theta,
        include_path_length=include_path_length,
        include_max_risk=include_max_risk,
    )

    return theta, x_final


# outer loop over alpha

def fit_theta_to_expert(
    x0: np.ndarray,
    goal_state: np.ndarray,
    cost_map: np.ndarray,
    expert_x: np.ndarray,
    expert_u: np.ndarray,
    *,
    alpha_candidates: np.ndarray | None = None,
    n_iters: int = 40,
    lr: float = 0.002,
    include_path_length: bool = True,
    include_max_risk: bool = True,
    selection_metric: str = "feature",  # "feature" or "geom"
    verbose: bool = False,
):
    """Full inverse loop: For each alpha: - build risk map - fit weights theta - score match Return best (theta, alpha, traj)"""
    if alpha_candidates is None:
        alpha_candidates = np.linspace(0.05, 0.9, 8)

    best_score = np.inf
    best_theta = None
    best_alpha = None
    best_traj = None

    for alpha in alpha_candidates:
        theta_hat, x_robot = fit_weights(
            x0, goal_state, cost_map,
            expert_x, expert_u,
            alpha,
            n_iters=n_iters,
            lr=lr,
            include_path_length=include_path_length,
            include_max_risk=include_max_risk,
            normalize_on="expert",
            verbose=False,
        )

        # build risk map for scoring (same alpha)
        risk_map = cvar_risk_map(RISK_MAP_DIM, RISK_MAP_DIM, alpha, cost_map)

        if selection_metric == "geom":
            # geometric (time-alignment sensitive)
            n = min(len(x_robot), len(expert_x))
            score = float(np.mean(np.linalg.norm(x_robot[:n, :2] - expert_x[:n, :2], axis=1)))
        else:
            # feature similarity
            phi_e = compute_features(
                expert_x, expert_u, goal_state, risk_map,
                include_path_length=include_path_length,
                include_max_risk=include_max_risk,
            )
            # replan once to get u for scoring
            x_tmp, u_tmp = plan_with_theta(
                x0, goal_state, risk_map, theta_hat,
                include_path_length=include_path_length,
                include_max_risk=include_max_risk,
            )
            phi_r = compute_features(
                x_tmp, u_tmp, goal_state, risk_map,
                include_path_length=include_path_length,
                include_max_risk=include_max_risk,
            )
            scale = np.maximum(np.abs(phi_e), 1e-6)
            score = float(np.linalg.norm(_normalize_features(phi_r, scale) - _normalize_features(phi_e, scale)))

        if verbose:
            print(f"[alpha={alpha:.3f}] score={score:.4f} theta={theta_hat}")

        if score < best_score:
            best_score = score
            best_theta = theta_hat
            best_alpha = float(alpha)
            best_traj = x_robot

    print("Best alpha:", best_alpha)
    print("Recovered theta:", best_theta)
    print("Best score:", best_score)

    return best_theta, best_alpha, best_traj
