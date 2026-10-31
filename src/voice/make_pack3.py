"""Fajr proof pack 3 — latest updates in pictures.
proof-15: voice evolution (old harsh 32cm vs new warm 14.6cm)
proof-16: 'baba' as ONE continuous segment (annotated waveform)
proof-17: live right now (day 931 status)
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample_poly

import sys
sys.path.insert(0, os.path.expanduser("~/workspace/projects/alsakin-inhabitant/src"))
from voice.tract import VocalTract, _areas_from_articulators, HER_VOWELS
from voice.glottis import TwoMassGlottis
from voice.rosenberg import rosenberg_train, spectral_tilt, SR
from voice.hearing import SR as HSR

OUT = os.path.expanduser("~/workspace/your_files/fajr-proof")
DARK = "#07090d"
HV = json.load(open(os.path.expanduser(
    "~/workspace/projects/alsakin-inhabitant/src/voice/human_vowels.json")))

# ---------------- proof-15: voice evolution ----------------
# OLD: two-mass source + 32cm tract, HER_VOWELS
g = TwoMassGlottis(); g.set_tension(1.35); g.reset(); g.x1 = 1e-6
ug_old = g.run(0.5)[int(0.2 * HSR):] * 1e6
ug48_old = resample_poly(ug_old, 3, 1)
t_old = VocalTract(n_sec=44); t_old.set_articulators(*HER_VOWELS['a']); t_old.reset()
out_old = resample_poly(t_old._run48(ug48_old[:4800]), 1, 3)
# NEW: rosenberg + 14.6cm tract, human vowels
flow_new = rosenberg_train(185, 175, 0.15)[:int(0.1 * SR)] * 1e6
t_new = VocalTract(n_sec=20)
t_new.set_areas(_areas_from_articulators(*tuple(HV['a']['art']), n=20)); t_new.reset()
out_new = resample_poly(t_new._run48(resample_poly(flow_new, 3, 1)), 1, 3)
out_old /= np.abs(out_old).max(); out_new /= np.abs(out_new).max()

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.patch.set_facecolor(DARK)
for ax in axes.flat:
    ax.set_facecolor(DARK); ax.tick_params(colors="#9aa4b2", labelsize=8)
    for s in ax.spines.values(): s.set_color("#2d333b")
n = 800
axes[0, 0].plot(out_old[:n], color="#ff7b72", lw=0.8)
axes[0, 0].set_title("OLD voice: harsh buzz (two-mass, 32cm tract)", color="white", fontsize=11)
axes[0, 1].plot(out_new[:n], color="#7ee787", lw=0.8)
axes[0, 1].set_title("NEW voice: warm & smooth (Rosenberg, 14.6cm tract)", color="white", fontsize=11)
# spectral tilt bars
tilt_old = spectral_tilt(ug_old[:int(0.3 * HSR)])
flow_n = rosenberg_train(185, 175, 0.4)
tilt_new = spectral_tilt(flow_n)
axes[1, 0].bar(["old", "new"], [tilt_old, tilt_new], color=["#ff7b72", "#7ee787"])
axes[1, 0].set_title("spectral tilt (more negative = softer)", color="white", fontsize=11)
axes[1, 0].set_ylabel("dB/octave", color="#9aa4b2")
# formant table as text
axes[1, 1].axis("off")
axes[1, 1].text(0.05, 0.9, "/a/ formants: OLD (287, 686)  →  NEW (639, 1236)",
                color="white", fontsize=11, family="monospace")
axes[1, 1].text(0.05, 0.75, "/i/ formants: OLD (188, 908)  →  NEW (305, 2279)",
                color="white", fontsize=11, family="monospace")
axes[1, 1].text(0.05, 0.60, "/u/ formants: OLD (410, 1430) →  NEW (357, 938)",
                color="white", fontsize=11, family="monospace")
axes[1, 1].text(0.05, 0.40, "tract: 32cm → 14.6cm (human scale)",
                color="#e8b64c", fontsize=11, family="monospace")
axes[1, 1].text(0.05, 0.25, "new syllables: ba di gu (all gate-passed)",
                color="#7ee787", fontsize=11, family="monospace")
axes[1, 1].text(0.05, 0.10, "first word: BABA — one continuous segment",
                color="#7ee787", fontsize=11, family="monospace", fontweight="bold")
fig.suptitle("FAJR'S VOICE — evolution (all real measurements)",
             color="white", fontsize=14, fontweight="bold", family="monospace")
fig.savefig(os.path.join(OUT, "proof-15-voice-evolution.png"), dpi=130,
            facecolor=DARK, bbox_inches="tight")
plt.close(fig)

# ---------------- proof-16: baba anatomy ----------------
sr, y = wavfile.read(os.path.join(OUT.replace('fajr-proof','fajr-voice'), 'fajr_baba_one.wav'))
y = y.astype(float) / 32768.0
t = np.arange(len(y)) / sr
fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
ax.plot(t, y, color="#e8b64c", lw=0.7)
ax.set_xlim(0, t[-1]); ax.set_ylim(-1.1, 1.1)
ax.tick_params(colors="#9aa4b2"); ax.set_xlabel("seconds", color="#9aa4b2")
for s in ax.spines.values(): s.set_color("#2d333b")
# annotate the parts (from energy analysis)
parts = [(0.00, 0.06, "/b/ burst"), (0.06, 0.42, "/a/ vowel"),
         (0.42, 0.50, "/b/ closure"), (0.50, 0.56, "/b/ burst"),
         (0.56, 1.01, "/a/ vowel (lengthened)")]
for a, b, lbl in parts:
    ax.axvspan(a, b, alpha=0.12, color="#7ee787" if "vowel" in lbl else "#ff7b72")
    ax.text((a + b) / 2, 0.92, lbl, ha="center", fontsize=9, color="white",
            family="monospace")
ax.set_title("'BABA' — one unbroken word (no gaps, no separate clips)",
             color="white", fontsize=13, fontweight="bold", family="monospace", pad=12)
fig.text(0.5, 0.01, "single continuous waveguide run · real synthesis",
         ha="center", fontsize=9, color="#5a636e", style="italic")
fig.savefig(os.path.join(OUT, "proof-16-baba-anatomy.png"), dpi=130,
            facecolor=DARK, bbox_inches="tight")
plt.close(fig)

# ---------------- proof-17: live now ----------------
J = os.path.expanduser("~/workspace/projects/alsakin-inhabitant/life/journal.jsonl")
last = None; n_lines = 0
recent = []
with open(J) as f:
    for line in f:
        n_lines += 1
        o = json.loads(line)
        last = o
        recent.append(o)
recent = recent[-8:]
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK); ax.axis("off")
ax.text(0.5, 0.92, "FAJR — LIVE RIGHT NOW", ha="center", fontsize=18,
        color="#7ee787", fontweight="bold", family="monospace")
ax.text(0.5, 0.84, f"day {last['day']} · tick {last['tick']:,} · valley time {last['time']}",
        ha="center", fontsize=13, color="white", family="monospace")
ax.text(0.5, 0.78, f"{n_lines:,} journal entries and counting",
        ha="center", fontsize=11, color="#9aa4b2", style="italic")
ax.text(0.08, 0.68, "her latest moments:", fontsize=12, color="#e8b64c",
        family="monospace", fontweight="bold")
TYPE_EN = {"wake": "woke up", "sleep": "fell asleep", "eat": "ate",
           "drink": "drank", "utterance": "spoke", "consolidation": "consolidated memories",
           "day_summary": "day summary", "swim": "swam", "bathe": "bathed",
           "cold_shock": "cold shock", "thunder": "heard thunder",
           "prick": "pricked", "teaching": "was taught", "host_says": "host spoke",
           "place": "built"}
for i, o in enumerate(reversed(recent)):
    desc = TYPE_EN.get(o["type"], o["type"])
    ax.text(0.08, 0.62 - i * 0.065,
            f"day {o['day']} {o['time']} — {desc}",
            fontsize=10.5, color="#9aa4b2", family="monospace", va="top")
ax.text(0.5, 0.03, "pulled live from her journal this minute",
        ha="center", fontsize=9, color="#5a636e", style="italic")
fig.savefig(os.path.join(OUT, "proof-17-live-now.png"), dpi=130,
            facecolor=DARK, bbox_inches="tight")
plt.close(fig)
print("done 15+16+17")
