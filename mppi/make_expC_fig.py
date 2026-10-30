"""Exp C before/after figure (no rollouts; numbers from tier2_results/summary.txt)."""
import os, numpy as np, matplotlib.pyplot as plt
OUT = "tier2_results"
eig_pos = [4103.6, 624.6, 553.4, 119.3]
eig_head = [6038.6, 1362.4, 937.7, 195.8]
mix_pos, mix_head = 0.457, 0.012
DARK, ACCENT, GREEN = "#1A1F2B", "#E07A5F", "#3F7D5C"

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2, 1]})
x = np.arange(4); w = 0.38
a1.bar(x - w/2, eig_pos, w, color=DARK, label="position only")
a1.bar(x + w/2, eig_head, w, color=ACCENT, label="position + heading")
a1.set_yscale("log"); a1.set_xticks(x)
a1.set_xticklabels(["dir1\n(β)", "dir2", "dir3", "dir4\n(control)"])
a1.set_ylabel("curvature (eigenvalue, log)")
a1.set_title("Adding heading stiffens every direction (~1.5–2×)", fontsize=12)
a1.legend(fontsize=9); a1.grid(alpha=0.3, axis="y")

a2.bar([0, 1], [mix_pos, mix_head], color=[DARK, ACCENT], width=0.6)
a2.set_xticks([0, 1]); a2.set_xticklabels(["position\nonly", "position\n+ heading"])
a2.set_ylabel("W1–W3 mixing index")
a2.set_title("…and decouples path-track ↔ terminal", fontsize=12)
for i, v in enumerate([mix_pos, mix_head]):
    a2.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=11, weight="bold")
a2.set_ylim(0, 0.52); a2.grid(alpha=0.3, axis="y")
fig.suptitle("Exp C — a heading signal breaks the W1↔W3 confounding", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(OUT, "expC_heading_fix.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
print("saved", out)
