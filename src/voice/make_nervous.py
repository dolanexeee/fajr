"""Fajr's mechanical nervous system — anatomical map of her REAL code modules.

Like those human nervous-system charts: every "neuron" here is an actual
module from her running system, placed where its function lives.
Schematic, not biology. Zero generated imagery.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch

OUT = os.path.expanduser("~/workspace/your_files/fajr-proof")
os.makedirs(OUT, exist_ok=True)

DARK = "#0b0e14"
GREEN = "#7ee787"   # cognition
BLUE = "#79c0ff"    # memory / models
AMBER = "#e3b341"   # senses / drives
PURPLE = "#d2a8ff"  # voice / language
RED = "#ff7b72"     # drives core
DIM = "#8b949e"

fig, ax = plt.subplots(figsize=(11, 13))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(DARK)
ax.set_xlim(0, 10)
ax.set_ylim(0, 13)
ax.axis("off")
for s in ax.spines.values():
    s.set_color(DARK)

# ---- silhouette (very dark, just a guide) ----
sil = "#151a23"
head = Ellipse((5, 11), 2.6, 2.8, color=sil, zorder=1)
ax.add_patch(head)
torso = FancyBboxPatch((3.7, 6.2), 2.6, 3.2, boxstyle="round,pad=0.3",
                       color=sil, zorder=1)
ax.add_patch(torso)
for (x0, y0, x1, y1) in [(3.7, 8.6, 1.8, 6.6), (6.3, 8.6, 8.2, 6.6),
                         (4.2, 6.2, 3.6, 2.5), (5.8, 6.2, 6.4, 2.5)]:
    ax.plot([x0, x1], [y0, y1], color=sil, lw=26, solid_capstyle="round", zorder=1)

# ---- nodes: (x, y, label, sublabel, color, size) ----
nodes = [
    # brain — cognition
    (5.0, 11.6, "mind.py", "cognitive cycle", GREEN, 260),
    (4.1, 11.1, "workspace.py", "global workspace", GREEN, 150),
    (5.9, 11.1, "beliefs.py", "beliefs", GREEN, 150),
    (4.4, 10.55, "episodic.py", "memory", BLUE, 150),
    (5.6, 10.55, "selfmodel.py", "self model", BLUE, 150),
    (5.0, 10.1, "fwdmodel.py", "forward model", BLUE, 130),
    (3.55, 10.75, "monitor.py", "reliability", DIM, 90),
    # senses
    (3.35, 11.35, "perception.py", "vision", AMBER, 130),
    (6.65, 11.35, "senses.py", "senses", AMBER, 130),
    (6.85, 10.7, "hearing.py", "ears", AMBER, 110),
    # voice — mouth & throat
    (5.0, 9.55, "language.py", "tongue", PURPLE, 150),
    (4.25, 9.25, "lexicon.py", "words", PURPLE, 100),
    (5.75, 9.25, "glottis.py", "vocal cords", PURPLE, 140),
    (5.0, 8.85, "tract.py", "vocal tract", PURPLE, 140),
    (4.3, 8.6, "diva.py", "imitation", PURPLE, 100),
    (5.7, 8.6, "consonants.py", "plosives", PURPLE, 100),
    # heart — drives
    (5.0, 7.75, "drives.py", "8 drives", RED, 220),
    # spine — the tick loop
    (5.0, 6.6, "live_runner.py", "spinal cord · 2 ticks/s", DIM, 130),
    # hands — interaction
    (2.35, 7.3, "conversation.py", "speech", BLUE, 120),
    # body & world
    (5.0, 5.3, "fajr.py", "body", DIM, 140),
    (3.9, 4.3, "world.py", "the valley", DIM, 120),
    (6.1, 4.3, "sim.py", "physics", DIM, 120),
    (5.0, 3.3, "render.py", "vision out", DIM, 100),
]
pos = {}
for x, y, label, sub, col, sz in nodes:
    pos[label] = (x, y)
    ax.scatter([x], [y], s=sz, c=col, zorder=3, alpha=0.92,
               edgecolors="white", linewidths=0.6)
    ax.text(x, y + 0.32, label, ha="center", fontsize=8.5, color="white",
            family="monospace", zorder=4)
    ax.text(x, y - 0.34, sub, ha="center", fontsize=7, color=DIM,
            style="italic", zorder=4)

# ---- nerves (data flow) ----
nerves = [
    ("perception.py", "mind.py"), ("senses.py", "mind.py"),
    ("hearing.py", "mind.py"), ("mind.py", "workspace.py"),
    ("workspace.py", "beliefs.py"), ("beliefs.py", "episodic.py"),
    ("episodic.py", "selfmodel.py"), ("mind.py", "fwdmodel.py"),
    ("mind.py", "language.py"), ("language.py", "glottis.py"),
    ("glottis.py", "tract.py"), ("diva.py", "tract.py"),
    ("mind.py", "drives.py"), ("drives.py", "fajr.py"),
    ("live_runner.py", "mind.py"), ("live_runner.py", "drives.py"),
    ("live_runner.py", "fajr.py"), ("fajr.py", "world.py"),
    ("world.py", "sim.py"), ("mind.py", "conversation.py"),
    ("monitor.py", "mind.py"), ("consonants.py", "tract.py"),
]
for a, b in nerves:
    x0, y0 = pos[a]
    x1, y1 = pos[b]
    ax.plot([x0, x1], [y0, y1], color="#2d333b", lw=1.1, zorder=2, alpha=0.8)

ax.text(5, 12.55, "FAJR — MECHANICAL NERVOUS SYSTEM", ha="center",
        fontsize=17, color=GREEN, fontweight="bold", family="monospace")
ax.text(5, 12.2, "every node is a real module from her running code · placed by function",
        ha="center", fontsize=10, color=DIM, style="italic")
ax.text(5, 0.7, "schematic map — not biology, not generated imagery · src/voice + src/*.py live",
        ha="center", fontsize=9, color="#484f58", family="monospace")
# legend
leg = [("cognition", GREEN), ("memory/models", BLUE), ("senses", AMBER),
       ("voice/language", PURPLE), ("drives", RED)]
for i, (lbl, col) in enumerate(leg):
    x = 1.4 + i * 1.85
    ax.scatter([x], [1.45], s=90, c=col, zorder=3, edgecolors="white", linewidths=0.5)
    ax.text(x + 0.28, 1.45, lbl, fontsize=8.5, color=DIM, va="center")

fig.savefig(os.path.join(OUT, "proof-6-nervous-system.png"), dpi=140,
            facecolor=DARK, bbox_inches="tight")
plt.close(fig)
print("done")
