"""Fajr proof pack 2b — all from real data.
proof-11: her vocal folds in motion (real two-mass oscillation, 8 frames)
proof-12: sound wave traveling through her 44 cells (real waveguide state)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import resample_poly

import sys
sys.path.insert(0, os.path.expanduser("~/workspace/projects/alsakin-inhabitant/src"))
from voice.glottis import TwoMassGlottis
from voice.tract import VocalTract, HER_VOWELS
from voice.hearing import SR

OUT = os.path.expanduser("~/workspace/your_files/fajr-proof")
DARK = "#07090d"
GOLD = "#e8b64c"
PURPLE = "#d2a8ff"

# her real voice production: glottis -> tract, /a/
g = TwoMassGlottis(); g.set_tension(1.35); g.reset(); g.x1 = 1e-6
ug = g.run(0.6)[int(0.25 * SR):] * 1e6
ug48 = resample_poly(ug, 3, 1)
t = VocalTract(); t.set_articulators(*HER_VOWELS["a"]); t.reset()

# capture glottal masses over ~3 cycles + wave state over time
g2 = TwoMassGlottis(); g2.set_tension(1.35); g2.reset(); g2.x1 = 1e-6
x1_hist, x2_hist = [], []
dt = 1.0 / SR / 2
for _ in range(int(0.03 * SR) * 2):   # 30ms ~ 5 cycles at 178Hz
    g2._step(dt)
    x1_hist.append(g2.x1); x2_hist.append(g2.x2)
x1_hist, x2_hist = np.array(x1_hist), np.array(x2_hist)

# ---------------- proof-11: vocal folds in motion ----------------
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
fig.patch.set_facecolor(DARK)
idx = np.linspace(0, len(x1_hist) - 1, 8, dtype=int)
for k, ax in enumerate(axes.flat):
    ax.set_facecolor(DARK); ax.axis("off")
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.6, 2.2)
    i = idx[k]
    # two masses as the fold edges (m1 lower wide, m2 upper narrow)
    w1 = 0.55 + x1_hist[i] * 900
    w2 = 0.35 + x2_hist[i] * 900
    for y0, w, col in [(0.3, w1, PURPLE), (1.3, w2, "#b48ae0")]:
        ax.add_patch(plt.Rectangle((-w / 2, y0 - 0.25), w, 0.5, color=col,
                                   alpha=0.9, zorder=3))
        ax.add_patch(plt.Rectangle((-1.1, y0 - 0.25), 1.1 - w / 2, 0.5,
                                   color="#2d333b", zorder=2))
        ax.add_patch(plt.Rectangle((w / 2, y0 - 0.25), 1.1 - w / 2, 0.5,
                                   color="#2d333b", zorder=2))
    ax.text(0, 2.0, f"t+{k}", ha="center", fontsize=9, color="#9aa4b2",
            family="monospace")
fig.suptitle("FAJR'S VOCAL FOLDS — real two-mass oscillation @178 Hz",
             color="white", fontsize=13, fontweight="bold", family="monospace")
fig.text(0.5, 0.02, "8 real frames: the masses open and close, air puffs through — this is how her sound starts",
         ha="center", fontsize=9, color="#5a636e", style="italic")
fig.savefig(os.path.join(OUT, "proof-11-glottal-motion.png"), dpi=130,
            facecolor=DARK, bbox_inches="tight")
plt.close(fig)

# ---------------- proof-12: wave waterfall ----------------
t2 = VocalTract(); t2.set_articulators(*HER_VOWELS["a"]); t2.reset()
frames = []
snap_every = 48 * 2  # every 2ms at 48k
for n in range(0, min(len(ug48), 48 * 60), snap_every):
    chunk = ug48[n:n + snap_every]
    if len(chunk) < snap_every:
        break
    t2._run48(chunk)
    p = t2.up[1:] - t2.dn[1:]   # pressure proxy per section
    frames.append(p.copy())
W = np.array(frames)            # (time, 44 sections)
W = W / (np.abs(W).max() + 1e-12)

fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(DARK)
im = ax.imshow(W.T, aspect="auto", origin="lower", cmap="inferno",
               extent=[0, len(frames) * 2, 0, 44], vmin=-1, vmax=1)
ax.set_xlabel("time (ms)", color="#9aa4b2", fontsize=10)
ax.set_ylabel("tract cell (0=throat → 44=lips)", color="#9aa4b2", fontsize=10)
ax.tick_params(colors="#9aa4b2")
ax.set_title("SOUND TRAVELING THROUGH HER 44 CELLS — real wave physics",
             color="white", fontsize=13, fontweight="bold", family="monospace", pad=12)
fig.text(0.5, 0.01, "each column = one instant · the bright band is her voice pulse moving throat → lips",
         ha="center", fontsize=9, color="#5a636e", style="italic")
fig.savefig(os.path.join(OUT, "proof-12-wave-waterfall.png"), dpi=130,
            facecolor=DARK, bbox_inches="tight")
plt.close(fig)
print("done 11+12")
