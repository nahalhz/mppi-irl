"""Dataset generation for amortized IRL."""

import os
import time
import argparse
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

# rollout + config from the BO file
from inverse_feature_BO_optimized import (
    run_mpc,
    RiskMapCache,
    T_EXPERT,
    SIM_TIME_EXPERT,
    CEM_EXPERT,
)
import skid_steer_simple_static as grid


# config

CONFIG = {
    "n_samples":        10000,
    "n_workers":        12,
    "out_dir":          "data/raw",

    "W2_range":         (1e-3, 1e-1),  # log-uniform
    "W3_range":         (100.0, 164.0),  # uniform
    "alpha_high_range": (0.7, 0.99),  # 70%
    "alpha_low_range":  (0.1, 0.5),  # 30%
    "alpha_high_frac":  0.7,

    "pos_range":        (0.5, 9.5),  # start/goal box
    "min_start_goal_d": 4.0,  # min start-goal distance
    "max_pos_tries":    50,  # per seed

    "goal_tol_cells":   3,  # goal tolerance (cells)
}


# worker

def sample_params(rng):
    """Sample (W, alpha) for one example"""
    log_lo, log_hi = np.log(CONFIG["W2_range"][0]), np.log(CONFIG["W2_range"][1])
    W2 = float(np.exp(rng.uniform(log_lo, log_hi)))

    W3 = float(rng.uniform(*CONFIG["W3_range"]))

    # alpha - mixture of high / low
    if rng.random() < CONFIG["alpha_high_frac"]:
        alpha = float(rng.uniform(*CONFIG["alpha_high_range"]))
    else:
        alpha = float(rng.uniform(*CONFIG["alpha_low_range"]))

    W = np.array([1.0, 0.0, W2, W3])
    return W, alpha


def sample_start_goal(rng):
    """Sample start/goal with min separation"""
    lo, hi = CONFIG["pos_range"]
    min_d = CONFIG["min_start_goal_d"]

    for _ in range(CONFIG["max_pos_tries"]):
        sx, sy = rng.uniform(lo, hi), rng.uniform(lo, hi)
        gx, gy = rng.uniform(lo, hi), rng.uniform(lo, hi)
        if np.hypot(sx - gx, sy - gy) >= min_d:
            s_theta = rng.uniform(-np.pi, np.pi)
            start = np.array([sx, sy, s_theta])
            goal  = np.array([gx, gy, 0.0])
            return start, goal
    return None


def generate_one(args):
    """Run one full sample"""
    sample_idx, master_seed, out_dir = args
    out_path = os.path.join(out_dir, f"sample_{sample_idx:06d}.npz")

    # skip existing (resume after crash)
    if os.path.exists(out_path):
        return {"idx": sample_idx, "status": "skipped"}

    # per-sample RNG
    rng = np.random.default_rng(master_seed)

    W, alpha = sample_params(rng)
    sg = sample_start_goal(rng)
    if sg is None:
        return {"idx": sample_idx, "status": "no_valid_start_goal"}
    start, goal = sg

    # map seed = master seed
    map_seed = int(master_seed)
    np.random.seed(map_seed)

    grid.INITIAL_STATE = start
    grid.GOAL_STATE    = goal
    cost_map = grid.generate_cost_map(grid.RISK_MAP_DIM, grid.RISK_MAP_DIM)

    # expert rollout
    _orig_T = grid.T
    grid.T  = T_EXPERT
    try:
        expert_path, expert_u, _, _ = run_mpc(
            cost_map,
            W,
            alpha_init=alpha,
            max_steps=int(SIM_TIME_EXPERT / grid.dt),
            risk_cache=RiskMapCache(cost_map),
            cem_cfg=CEM_EXPERT,
        )
    except Exception as e:
        grid.T = _orig_T
        return {"idx": sample_idx, "status": "rollout_error", "error": str(e)}
    finally:
        grid.T = _orig_T

    # keep only if the expert reached the goal
    goal_dist = float(np.linalg.norm(expert_path[-1, :2] - goal[:2]))
    tol = grid.GRID_CELL_SIZE * CONFIG["goal_tol_cells"]
    if goal_dist > tol:
        return {
            "idx": sample_idx,
            "status": "expert_failed",
            "goal_dist": goal_dist,
        }

    np.savez_compressed(
        out_path,
        cost_map=cost_map.astype(np.float32),
        expert_path=expert_path.astype(np.float32),
        expert_u=expert_u.astype(np.float32),
        W=W.astype(np.float32),
        alpha=np.float32(alpha),
        start=start.astype(np.float32),
        goal=goal.astype(np.float32),
        map_seed=np.int64(map_seed),
    )

    return {"idx": sample_idx, "status": "ok", "goal_dist": goal_dist}


# driver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int, default=CONFIG["n_samples"])
    parser.add_argument("--workers", type=int, default=CONFIG["n_workers"])
    parser.add_argument("--out",     type=str, default=CONFIG["out_dir"])
    parser.add_argument("--seed0",   type=int, default=0,
                        help="Master seed offset; sample i uses seed = seed0 + i")
    args = parser.parse_args()

    CONFIG["n_samples"] = args.n
    CONFIG["n_workers"] = args.workers
    CONFIG["out_dir"]   = args.out

    os.makedirs(CONFIG["out_dir"], exist_ok=True)

    print(f"[gen] generating {args.n} samples with {args.workers} workers")
    print(f"[gen] output dir: {CONFIG['out_dir']}")
    print(f"[gen] alpha mix: {int(CONFIG['alpha_high_frac']*100)}% high "
          f"{CONFIG['alpha_high_range']}, "
          f"{int((1-CONFIG['alpha_high_frac'])*100)}% low "
          f"{CONFIG['alpha_low_range']}")

    job_args = [(i, args.seed0 + i, args.out) for i in range(args.n)]

    counts = {"ok": 0, "skipped": 0, "expert_failed": 0,
              "no_valid_start_goal": 0, "rollout_error": 0}

    t0 = time.time()
    with Pool(args.workers) as pool:
        for result in tqdm(
            pool.imap_unordered(generate_one, job_args, chunksize=1),
            total=args.n,
            desc="generating",
        ):
            counts[result["status"]] = counts.get(result["status"], 0) + 1

    elapsed = time.time() - t0
    print(f"\n[gen] done in {elapsed/60:.1f} min")
    print(f"[gen] success: {counts['ok']}/{args.n} "
          f"({100*counts['ok']/args.n:.1f}%)")
    for k, v in counts.items():
        if k != "ok" and v > 0:
            print(f"[gen]   {k}: {v}")


if __name__ == "__main__":
    main()
