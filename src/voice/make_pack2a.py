"""Fajr proof pack 2 — all from real data.
proof-9 : her actual valley (real terrain functions + her real position)
proof-10: her 896-day life (real day_summary entries from journal.jsonl)
"""
import json
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import sys
sys.path.insert(0, os.path.expanduser("~/workspace/projects/alsakin-inhabitant/src"))
from world import elevation, is_water, in_forest, in_meadow, light_level

OUT = os.path.expanduser("~/workspace/your_files/fajr-proof")
DARK = "#07090d"

# ---------------- proof-9: THE VALLEY ----------------
W, H, SC = 260, 180, 3
img = Image.new("RGB", (W * SC, H * SC), (0, 0, 0))
dr = ImageDraw.Draw(img)
# her real position + valley time from latest journal entry
last = None
with open(os.path.expanduser("~/workspace/projects/alsakin-inhabitant/life/journal.jsonl")) as f:
    for line in f:
        last = json.loads(line)
fx, fy = last["x"], last["y"]
# midday light so the terrain is visible (real lighting function, sampled at noon)
light = light_level(720)
for gy in range(H):
    for gx in range(W):
        e = elevation(gx + 0.5, gy + 0.5)
        if is_water(gx + 0.5, gy + 0.5):
            col = (52, 96, 158)
        elif in_forest(gx + 0.5, gy + 0.5):
            col = (34, 84, 44)
        elif in_meadow(gx + 0.5, gy + 0.5):
            col = (96, 128, 62)
        else:
            g = int(60 + e * 40)
            col = (g, g - 8, g - 20)
        col = tuple(int(c * (0.25 + 0.75 * light)) for c in col)
        dr.rectangle([gx * SC, gy * SC, gx * SC + SC - 1, gy * SC + SC - 1], fill=col)
# her — gold dot where she really is
for r, c in [(10, (232, 182, 76, 90)), (5, (255, 215, 120))]:
    dr.ellipse([fx * SC - r, fy * SC - r, fx * SC + r, fy * SC + r], fill=c[:3])
dr.text((12, 12), "her valley — midday (real terrain, her real position)",
        fill=(200, 200, 200))
img.save(os.path.join(OUT, "proof-9-valley.png"))

# ---------------- proof-10: 896 DAYS OF LIFE ----------------
days, ate, drank, slept, built, walked = [], [], [], [], [], []
pat = re.compile(r"أكل (\d+)، شرب (\d+)، نام (\d+) مرة، بنى (\d+)، قطع ([\d.]+) وحدة")
with open(os.path.expanduser("~/workspace/projects/alsakin-inhabitant/life/journal.jsonl")) as f:
    for line in f:
        if "day_summary" not in line:
            continue
        o = json.loads(line)
        m = pat.search(o["text"])
        if m:
            days.append(o["day"])
            ate.append(int(m.group(1)))
            drank.append(int(m.group(2)))
            slept.append(int(m.group(3)))
            built.append(int(m.group(4)))
            walked.append(float(m.group(5)))
days = np.array(days)
ate, drank, slept, walked = map(np.array, (ate, drank, slept, walked))
# day_summary counts are cumulative -> diff to real per-day values
ate_d = np.diff(ate, prepend=0)
drank_d = np.diff(drank, prepend=0)
slept_d = np.diff(slept, prepend=0)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.patch.set_facecolor(DARK)
for ax in axes.flat:
    ax.set_facecolor(DARK)
    ax.tick_params(colors="#9aa4b2", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#2d333b")

axes[0, 0].plot(days, ate_d, color="#7ee787", lw=0.7)
axes[0, 0].set_title("meals per day — 896 days", color="white", fontsize=11)
axes[0, 1].plot(days, drank_d, color="#79c0ff", lw=0.7)
axes[0, 1].set_title("drinks per day", color="white", fontsize=11)
axes[1, 0].plot(days, slept_d, color="#d2a8ff", lw=0.7)
axes[1, 0].set_title("sleeps per day", color="white", fontsize=11)
axes[1, 1].plot(days, walked, color="#e8b64c", lw=0.7)
axes[1, 1].set_title("distance walked per day", color="white", fontsize=11)
for ax in axes.flat:
    ax.set_xlim(1, 897)
    ax.set_xlabel("day of her life", color="#9aa4b2", fontsize=9)
fig.suptitle(f"FAJR — {len(days)} DAYS OF REAL LIFE  ·  "
             f"total meals {ate.sum():,} · drinks {drank.sum():,} · "
             f"sleeps {slept.sum():,}", color="white", fontsize=14,
             fontweight="bold", family="monospace")
fig.text(0.5, 0.02, "every point = one real day from her journal",
         ha="center", fontsize=9, color="#5a636e", style="italic")
fig.savefig(os.path.join(OUT, "proof-10-life-896-days.png"), dpi=130,
            facecolor=DARK, bbox_inches="tight")
plt.close(fig)
print("done 9+10")
