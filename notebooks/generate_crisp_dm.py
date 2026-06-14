#!/usr/bin/env python3
"""
Generate a clean CRISP-DM lifecycle diagram for the TFG methodology chapter.

Run from the project root:
    python notebooks/generate_crisp_dm.py

Output:
    reports/figures/crisp_dm.pdf   (vector — use this in LaTeX)
    reports/figures/crisp_dm.png   (raster preview, 200 dpi)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# ── palette ─────────────────────────────────────────────────────────────────
BOX_FC   = "#4472C4"   # box fill  (standard CRISP-DM blue)
BOX_EC   = "#2D5BA8"   # box edge
BOX_TXT  = "white"
ARR_COL  = "#8C8C8C"   # arrows and outer ring
CYL_FC   = "#D8D8D8"   # data cylinder fill
CYL_EC   = "#888888"   # data cylinder edge
BG       = "white"

# ── geometry ─────────────────────────────────────────────────────────────────
R    = 3.4   # radius to box centres
ROUT = R + 1.55  # outer ring radius
BW   = 2.40  # box width
BH   = 0.96  # box height
PAD  = 0.10  # FancyBboxPatch corner padding

# ── phase labels and angular positions (degrees, CCW from east) ──────────────
# Hexagonal, starting at 120° (upper-left) going clockwise
PHASES = [
    ("Business\nUnderstanding",  120),
    ("Data\nUnderstanding",       60),
    ("Data\nPreparation",          0),
    ("Modeling",                 -60),
    ("Evaluation",              -120),
    ("Deployment",               180),
]


def phase_centre(deg):
    rad = np.radians(deg)
    return np.array([R * np.cos(rad), R * np.sin(rad)])


def shrink(p1, p2, amount=0.72):
    """Move endpoints inward so arrows don't overlap boxes."""
    d = p2 - p1
    u = d / np.linalg.norm(d)
    return p1 + u * amount, p2 - u * amount


def draw_arrow(ax, p1, p2, style="->", rad=0.0, lw=1.8):
    ax.annotate(
        "", xy=p2, xytext=p1,
        arrowprops=dict(
            arrowstyle=style,
            color=ARR_COL,
            lw=lw,
            connectionstyle=f"arc3,rad={rad}",
            mutation_scale=14,
        ),
        zorder=3,
    )


# ── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 6.4))
ax.set_aspect("equal")
span = ROUT + 0.7
ax.set_xlim(-span, span)
ax.set_ylim(-span, span)
ax.axis("off")
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# ── outer ring (nearly full circle with arrowhead) ───────────────────────────
theta = np.linspace(np.radians(94), np.radians(-262), 500)
ax.plot(ROUT * np.cos(theta), ROUT * np.sin(theta),
        color=ARR_COL, lw=2.4, zorder=1, solid_capstyle="round")

# arrowhead at the end of the outer ring
t1, t2 = np.radians(-261), np.radians(-262.5)
ax.annotate(
    "", xy=(ROUT * np.cos(t2), ROUT * np.sin(t2)),
    xytext=(ROUT * np.cos(t1), ROUT * np.sin(t1)),
    arrowprops=dict(arrowstyle="->", color=ARR_COL, lw=2.4, mutation_scale=16),
    zorder=2,
)

# ── inter-phase arrows ────────────────────────────────────────────────────────
centres = {label: phase_centre(deg) for label, deg in PHASES}

labels = [p[0] for p in PHASES]   # ordered list

# Sequential arrows: BU<->DU, DU->DP, DP<->M, M->E, E->D
SEQUENTIAL = [
    (0, 1, "<->"),  # Business Understanding <-> Data Understanding
    (1, 2, "->"),   # Data Understanding -> Data Preparation
    (2, 3, "<->"),  # Data Preparation <-> Modeling
    (3, 4, "->"),   # Modeling -> Evaluation
    (4, 5, "->"),   # Evaluation -> Deployment
]
for i, j, style in SEQUENTIAL:
    s, e = shrink(centres[labels[i]], centres[labels[j]])
    draw_arrow(ax, s, e, style=style)

# Feedback: Evaluation -> Business Understanding (curved)
p_e  = centres["Evaluation"]
p_bu = centres["Business\nUnderstanding"]
s, e = shrink(p_e, p_bu, amount=0.72)
draw_arrow(ax, s, e, style="->", rad=-0.38)

# ── central "Data" cylinder ───────────────────────────────────────────────────
cw, ch, ctop = 1.60, 1.00, 0.30

body = FancyBboxPatch(
    (-cw / 2, -ch / 2), cw, ch,
    boxstyle=f"round,pad=0.05",
    fc=CYL_FC, ec=CYL_EC, lw=1.5, zorder=4,
)
ax.add_patch(body)

# top ellipse lid
lid = mpatches.Ellipse((0, ch / 2), cw, ctop,
                        fc="#EBEBEB", ec=CYL_EC, lw=1.5, zorder=5)
ax.add_patch(lid)
# bottom ellipse base (partially hidden)
base = mpatches.Ellipse((0, -ch / 2), cw, ctop,
                         fc=CYL_FC, ec=CYL_EC, lw=1.5, zorder=3)
ax.add_patch(base)

ax.text(0, 0, "Data", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#333333", zorder=6)

# ── phase boxes ───────────────────────────────────────────────────────────────
for label, deg in PHASES:
    x, y = centres[label]
    box = FancyBboxPatch(
        (x - BW / 2, y - BH / 2), BW, BH,
        boxstyle=f"round,pad={PAD}",
        fc=BOX_FC, ec=BOX_EC, lw=1.6, zorder=5,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center",
            color=BOX_TXT, fontsize=9, fontweight="bold",
            linespacing=1.30, zorder=6)

# ── save ──────────────────────────────────────────────────────────────────────
out_dir = Path("reports/figures")
out_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(out_dir / "crisp_dm.pdf", bbox_inches="tight", facecolor=BG)
fig.savefig(out_dir / "crisp_dm.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved  reports/figures/crisp_dm.pdf  and  crisp_dm.png")
