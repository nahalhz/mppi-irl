"""Re-plot planner_cost_data.npz as raw (unweighted) costs. No re-run needed."""
import os
import numpy as np
import matplotlib.pyplot as plt

OUT = "value_function_results"
d = np.load(os.path.join(OUT, "planner_cost_data.npz"), allow_pickle=True)

# columns of *_rows - [risk(W0), track(W1), ctrl(W2), term(W3), total]
WT = {"risk": 1.0, "track": 10.0, "ctrl": 0.01, "term": 0.5}
TRUE = {"W1": 10.0, "W2": 0.01, "W3": 0.5, "beta": 0.5}
LABEL = {"W1": "W[1] path-tracking (true 10)", "W2": "W[2] control effort (true 0.01)",
         "W3": "W[3] terminal (true 0.5)", "beta": "beta_cvar risk (true 0.5)"}
curves = [("terrain-risk integral", 0, 1.0,  "#E07A5F", "--"),
          ("tracking distance (m)", 1, 10.0, "#377eb8", "-"),
          ("control effort Σ(u/Wmax)²", 2, 0.01, "#3F7D5C", ":"),
          ("terminal distance (m)", 3, 0.5,  "#9467bd", "-.")]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, key in zip(axes.ravel(), ["W1", "W2", "W3", "beta"]):
    grid = d[f"{key}_grid"]; rows = d[f"{key}_rows"]
    for name, ci, w, col, ls in curves:
        ax.plot(grid, rows[:, ci] / w, ls, lw=1.8, color=col, label=name, marker="o", ms=2.5)
    ax.axvline(TRUE[key], color="#E07A5F", lw=2, alpha=0.85)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_title(LABEL[key], fontsize=12)
    ax.set_xlabel("parameter value (linear, full range)")
    ax.set_ylabel("raw physical cost of the produced path (symlog)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("Raw (unweighted) physical costs of the produced trajectory vs each parameter\n"
             "what the path actually incurs; one fixed scenario, others held at true",
             fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(OUT, "planner_cost_raw_vs_param.png")
plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
print("saved", out)
