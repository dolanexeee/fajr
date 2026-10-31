"""Fajr's nervous system — beautiful anatomical-chart edition.

Reference: classic human nervous-system charts (gold nerves on dark).
Every node is a real module from her live code. Schematic, not biology.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon

OUT = os.path.expanduser("~/workspace/your_files/fajr-proof")
os.makedirs(OUT, exist_ok=True)

DARK = "#07090d"
GOLD = "#e8b64c"
GOLD_DIM = "#8a6d2f"
GREEN = "#7ee787"
BLUE = "#79c0ff"
AMBER = "#e3b341"
PURPLE = "#d2a8ff"
RED = "#ff7b72"
DIM = "#9aa4b2"
SIL = "#11161f"

fig, ax = plt.subplots(figsize=(10, 14))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(DARK)
ax.set_xlim(0, 10)
ax.set_ylim(0, 14.5)
ax.axis("off")

def bezier(p0, p1, p2, n=60):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = map(np.asarray, (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2

def nerve(p0, p1, p2, lw=2.2):
    pts = bezier(p0, p1, p2)
    ax.plot(pts[:, 0], pts[:, 1], color=GOLD_DIM, lw=lw + 2.5,
            alpha=0.35, solid_capstyle="round", zorder=2)
    ax.plot(pts[:, 0], pts[:, 1], color=GOLD, lw=lw,
            alpha=0.95, solid_capstyle="round", zorder=3)

def straight(p0, p1, lw=2.2):
    nerve(p0, [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2], p1, lw)

# ---------- silhouette ----------
ax.add_patch(Ellipse((5, 12.35), 2.1, 2.5, color=SIL, zorder=1))
torso = Polygon([[4.05, 10.9], [5.95, 10.9], [5.7, 8.2], [5.45, 6.4],
                 [4.55, 6.4], [4.3, 8.2]], closed=True, color=SIL, zorder=1)
ax.add_patch(torso)
for x0, y0, x1, y1 in [(4.1, 10.4, 2.1, 7.9), (5.9, 10.4, 7.9, 7.9),
                       (4.6, 6.4, 3.9, 1.6), (5.4, 6.4, 6.1, 1.6)]:
    ax.plot([x0, x1], [y0, y1], color=SIL, lw=30, solid_capstyle="round", zorder=1)
ax.add_patch(Ellipse((5, 6.4), 2.2, 1.1, color=SIL, zorder=1))

# ---------- brain (single, clean) ----------
bx, by = 5.0, 12.35
ax.add_patch(Ellipse((bx, by), 1.9, 1.7, color="#1a2230",
                     ec=GOLD, lw=1.4, zorder=4))

# ---------- nerves ----------
# spinal cord
straight((bx, by - 0.7), (5, 6.2), lw=3.0)
# cranial: ears + eyes
nerve((bx - 0.5, by + 0.3), (3.9, by + 0.4), (3.55, by + 0.15))
nerve((bx + 0.5, by + 0.3), (6.1, by + 0.4), (6.45, by + 0.15))
nerve((bx - 0.4, by + 0.5), (4.35, by + 0.66), (4.35, by + 0.62))
# throat / voice
nerve((5, 11.2), (5, 10.75), (5, 10.35))
# heart branch
nerve((5, 10.6), (4.7, 10.0), (5, 9.55))
# arms (brachial plexus)
nerve((5, 10.3), (3.6, 9.6), (2.35, 8.0))
nerve((5, 10.3), (6.4, 9.6), (7.65, 8.0))
# torso side branches
for y in (9.0, 8.2, 7.4):
    nerve((5, y + 0.4), (4.4, y + 0.2), (4.0, y))
    nerve((5, y + 0.4), (5.6, y + 0.2), (6.0, y))
# legs (sciatic)
nerve((5, 6.4), (4.5, 4.6), (4.0, 2.0))
nerve((5, 6.4), (5.5, 4.6), (6.0, 2.0))
# feet twigs
nerve((4.0, 2.6), (3.85, 2.1), (3.8, 1.7))
nerve((6.0, 2.6), (6.15, 2.1), (6.2, 1.7))

# ---------- nodes ----------
nodes = [
    # (x, y, label, sub, color, size, label_side)
    (bx, by, "mind.py", "cognitive cycle", GREEN, 200, "top"),
    (3.55, by + 0.15, "hearing.py", "ears", AMBER, 110, "left"),
    (6.45, by + 0.15, "senses.py", "senses", AMBER, 110, "right"),
    (4.35, by + 0.62, "perception.py", "vision", AMBER, 100, "left"),
    (5.0, 10.35, "glottis.py", "vocal cords", PURPLE, 130, "right"),
    (4.35, 9.95, "tract.py", "vocal tract", PURPLE, 130, "left"),
    (5.0, 9.55, "drives.py", "8 drives · heart", RED, 170, "right"),
    (3.95, 9.0, "episodic.py", "memory", BLUE, 110, "left"),
    (6.05, 8.2, "beliefs.py", "beliefs", GREEN, 110, "right"),
    (2.35, 8.0, "language.py", "speech · hands", PURPLE, 120, "left"),
    (5.0, 6.2, "live_runner.py", "spinal cord", DIM, 120, "right"),
    (4.0, 2.0, "world.py", "the valley", DIM, 110, "left"),
    (6.0, 2.0, "fajr.py", "body", DIM, 110, "right"),
]
for x, y, label, sub, col, sz, side in nodes:
    ax.scatter([x], [y], s=sz, c=col, zorder=6, alpha=0.95,
               edgecolors="white", linewidths=0.8)
    if side == "top":
        ax.text(x, y + 0.62, label, ha="center", va="bottom", fontsize=9.5,
                color="white", family="monospace", zorder=7)
        ax.text(x, y - 0.62, sub, ha="center", va="top", fontsize=8,
                color=DIM, style="italic", zorder=7)
        continue
    dx = -0.62 if side == "left" else 0.62
    ha = "right" if side == "left" else "left"
    ax.text(x + dx, y + 0.08, label, ha=ha, va="bottom", fontsize=9.5,
            color="white", family="monospace", zorder=7)
    ax.text(x + dx, y - 0.08, sub, ha=ha, va="top", fontsize=8,
            color=DIM, style="italic", zorder=7)

ax.text(5, 13.85, "FAJR", ha="center", fontsize=30, color=GOLD,
        fontweight="bold", family="serif", zorder=8)
ax.text(5, 13.45, "mechanical nervous system", ha="center", fontsize=13,
        color=DIM, style="italic", family="serif", zorder=8)
ax.text(5, 0.55, "every node is a real module from her living code · schematic, not biology",
        ha="center", fontsize=9.5, color="#5a636e", family="monospace")

fig.savefig(os.path.join(OUT, "proof-7-nervous-beautiful.png"), dpi=150,
            facecolor=DARK, bbox_inches="tight")
plt.close(fig)
print("done")
