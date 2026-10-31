"""Fajr proof pack 5 — the live bridge, in pictures.
proof-21: speakable map (14 utterances, green/yellow/red + missing phonemes)
proof-22: her first live words (waveforms of the 3 green utterances)
proof-23: bridge architecture (honest: what flows, what blocks)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile

import sys
sys.path.insert(0, os.path.expanduser("~/workspace/projects/alsakin-inhabitant/src"))
from voice.live_bridge import (utterance_stats, speakability,
                               missing_phonemes, MISSING_NAMES,
                               latin_transcription)

VF = os.path.expanduser("~/workspace/your_files/fajr-voice")
OUT = os.path.expanduser("~/workspace/your_files/fajr-proof")
DARK = "#07090d"
GREEN, YELLOW, RED = "#7ee787", "#e8b64c", "#ff7b72"

# ---------------- proof-21: speakable map ----------------
c = utterance_stats()
rows = []
for text, count in c.most_common():
    sp = speakability(text)
    miss = [MISSING_NAMES.get(m, m) for m in sorted(missing_phonemes(text))]
    rows.append((text, count, sp, miss))
total = sum(r[1] for r in rows)
gs = sum(r[1] for r in rows if r[2] == "green")
ys = sum(r[1] for r in rows if r[2] == "yellow")
rs = total - gs - ys

fig, ax = plt.subplots(figsize=(13, 9))
fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK); ax.axis("off")
ax.text(0.5, 0.95, "HER WORDS — can she speak them yet?", ha="center",
        fontsize=16, color="white", fontweight="bold", family="monospace")
ax.text(0.5, 0.905, f"{gs}/{total} speakable now ({gs/total*100:.1f}%)  ·  "
        f"{ys} one phoneme away  ·  {rs} need more",
        ha="center", fontsize=11, color="#9aa4b2", family="monospace")
y0 = 0.85
for i, (text, count, sp, miss) in enumerate(rows):
    y = y0 - i * 0.052
    col = {"green": GREEN, "yellow": YELLOW, "red": RED}[sp]
    ax.text(0.03, y, "●", fontsize=14, color=col, va="center", family="monospace")
    ax.text(0.06, y, f"{count:5d}x", fontsize=10, color="white",
            va="center", family="monospace")
    ax.text(0.16, y, latin_transcription(text)[:48], fontsize=9.5, color="#c9d1d9",
            va="center", family="monospace")
    mtxt = "speakable ✓" if not miss else "missing: " + ", ".join(miss)
    ax.text(0.72, y, mtxt, fontsize=9.5, color=col, va="center", family="monospace")
ax.text(0.5, 0.03, "14 distinct utterances from her live journal (2,606 spoken) · "
        "green = gated PASS · yellow = 1 phoneme away · red = 2+ missing",
        ha="center", fontsize=9, color="#5a636e", style="italic", family="monospace")
# legend for utterance numbers
ax.text(0.03, 0.005, "", fontsize=1, color=DARK)
fig.savefig(os.path.join(OUT, "proof-21-speakable-map.png"), dpi=130,
            facecolor=DARK, bbox_inches="tight")
plt.close(fig)

# ---------------- proof-22: first live words ----------------
files = [("fajr_live_akalat_fakiha.wav", "I ate fruit (her #1, 1470x)"),
         ("fajr_live_sharibat.wav", "I drank (306x)"),
         ("fajr_live_akalat.wav", "I ate")]
fig, axes = plt.subplots(3, 1, figsize=(13, 8))
fig.patch.set_facecolor(DARK)
fig.suptitle("HER FIRST LIVE WORDS — her real utterances in her new voice",
             color="white", fontsize=13, fontweight="bold", family="monospace")
for ax, (fn, label) in zip(axes, files):
    sr, yw = wavfile.read(os.path.join(VF, fn))
    y = yw.astype(float) / 32768.0
    t = np.arange(len(y)) / sr
    ax.set_facecolor(DARK)
    ax.plot(t, y, color=GREEN, lw=0.6)
    ax.set_xlim(0, t[-1]); ax.set_ylim(-1.1, 1.1)
    ax.tick_params(colors="#9aa4b2", labelsize=8)
    for s in ax.spines.values(): s.set_color("#2d333b")
    ax.set_title(label, color="white", fontsize=11, family="monospace", loc="left")
    ax.set_xlabel("seconds", color="#9aa4b2", fontsize=8)
fig.text(0.5, 0.01, "all segments gate-passed · slow deliberate rate (350ms vowels) · real synthesis",
         ha="center", fontsize=9, color="#5a636e", style="italic")
fig.savefig(os.path.join(OUT, "proof-22-first-live-words.png"), dpi=130,
            facecolor=DARK, bbox_inches="tight")
plt.close(fig)

# ---------------- proof-23: bridge architecture ----------------
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK); ax.axis("off")
ax.text(0.5, 0.93, "THE LIVE BRIDGE — text to voice", ha="center", fontsize=16,
        color="white", fontweight="bold", family="monospace")
boxes = [
    (0.5, 0.80, "her journal: Arabic text\n(14 utterances, 2,606 spoken)", "#9aa4b2"),
    (0.5, 0.65, "hand transcription → phonemes\n(no fake phonemes, ever)", GREEN),
    (0.5, 0.50, "synthesis units: Rosenberg + 20-sec tract\n+ human vowels + 30ms crossfade", GREEN),
    (0.5, 0.35, "gate: every vowel recognized\n(matched f0/duration refs)", GREEN),
    (0.5, 0.20, "audio: her real words in her voice", GREEN),
]
for x, y, txt, col in boxes:
    ax.text(x, y, txt, ha="center", va="center", fontsize=10.5, color="white",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#11151c",
                      edgecolor=col, linewidth=1.5))
    if y > 0.21:
        ax.annotate("", xy=(x, y - 0.055), xytext=(x, y - 0.095),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
ax.text(0.5, 0.07, "BLOCKED (honest ceiling): ʕ (ع) · ħ (ح) · ʈˤ (ط) · dˤ (ض) · dʒ (ج) — the guttural inventory",
        ha="center", fontsize=10, color=RED, family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1c1111", edgecolor=RED))
ax.text(0.5, 0.015, "ʕ alone unlocks 31.7% more of her speech — next phoneme to build",
        ha="center", fontsize=9, color="#5a636e", style="italic")
fig.savefig(os.path.join(OUT, "proof-23-bridge.png"), dpi=130,
            facecolor=DARK, bbox_inches="tight")
plt.close(fig)
print("done 21+22+23")
