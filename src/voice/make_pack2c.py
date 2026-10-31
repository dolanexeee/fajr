"""Fajr proof pack 2c — all from real data.
proof-13: her vowel triangle (real formants from her tract vs guide)
proof-14: her memory & language growth (real journal: utterances, teaching, consolidation)
"""
import json
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import resample_poly, find_peaks

import sys
sys.path.insert(0, os.path.expanduser("~/workspace/projects/alsakin-inhabitant/src"))
from voice.tract import VocalTract, HER_VOWELS, VOWELS as GUIDE_VOWELS
from voice.hearing import SR

OUT = os.path.expanduser("~/workspace/your_files/fajr-proof")
DARK = "#07090d"

def formants(art):
    t = VocalTract(); t.set_articulators(*art); t.reset()
    imp = np.zeros(2048); imp[0] = 1.0
    h = t._run48(imp)[:2048]
    H = np.abs(np.fft.rfft(h * np.hanning(len(h)), 8192))
    freqs = np.fft.rfftfreq(8192, 1 / 48000)
    peaks, _ = find_peaks(20 * np.log10(H + 1e-9), distance=12, prominence=2)
    return [(float(freqs[p]), float(H[p])) for p in peaks[:4] if freqs[p] < 4000]

# ---------------- proof-13: vowel triangle ----------------
her_f, guide_f = {}, {}
for v in ("a", "i", "u"):
    her_f[v] = formants(HER_VOWELS[v])
    guide_f[v] = formants(GUIDE_VOWELS[v])

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
# guide first (gray, smaller)
for v in ("a", "i", "u"):
    f = guide_f[v]
    if len(f) >= 2:
        ax.scatter([f[0][0]], [f[1][0]], s=140, c="#9aa4b2", zorder=4,
                   alpha=0.7, edgecolors="white", linewidths=0.7)
# her (green, larger) — /a/ and /i/ coincide with guide (she matched them)
for v in ("a", "i", "u"):
    f = her_f[v]
    if len(f) >= 2:
        ax.scatter([f[0][0]], [f[1][0]], s=260, c="#7ee787", zorder=5,
                   edgecolors="white", linewidths=1)
labels = {
    "a": (her_f["a"][0][0] + 25, her_f["a"][1][0] + 60, "/a/ — hers = guide's"),
    "i": (her_f["i"][0][0] + 25, her_f["i"][1][0] + 60, "/i/ — hers = guide's"),
    "u": (her_f["u"][0][0] - 30, her_f["u"][1][0] + 70, "/u/ her (own discovery)"),
}
for v, (lx, ly, txt) in labels.items():
    ax.text(lx, ly, txt, fontsize=11, color="white", fontweight="bold",
            family="monospace", zorder=6,
            ha="left" if v != "u" else "right")
gf = guide_f["u"]
ax.text(gf[0][0] - 25, gf[1][0] - 60, "/u/ guide", fontsize=10, color="#9aa4b2",
        family="monospace", ha="right", zorder=6)
ax.set_xlabel("F1 (Hz)", color="#9aa4b2", fontsize=11)
ax.set_ylabel("F2 (Hz)", color="#9aa4b2", fontsize=11)
ax.tick_params(colors="#9aa4b2")
ax.invert_xaxis(); ax.invert_yaxis()
for s in ax.spines.values():
    s.set_color("#2d333b")
ax.set_title("HER VOWEL TRIANGLE — real formants from her vocal tract",
             color="white", fontsize=13, fontweight="bold", family="monospace", pad=12)
fig.text(0.5, 0.02, "green = her vowels (the ones she speaks) · gray = adult guide · from live waveguide physics",
         ha="center", fontsize=9, color="#5a636e", style="italic")
fig.savefig(os.path.join(OUT, "proof-13-vowel-triangle.png"), dpi=130,
            facecolor=DARK, bbox_inches="tight")
plt.close(fig)

# ---------------- proof-14: memory & language ----------------
J = os.path.expanduser("~/workspace/projects/alsakin-inhabitant/life/journal.jsonl")
utt_day, cons_day, teach = {}, {}, []
first_utt, last_utt = None, None
n_host = 0
with open(J) as f:
    for line in f:
        o = json.loads(line)
        d = o["day"]
        if o["type"] == "utterance":
            utt_day[d] = utt_day.get(d, 0) + 1
            if first_utt is None:
                first_utt = (d, o["text"])
            last_utt = (d, o["text"])
        elif o["type"] == "consolidation":
            m = re.search(r"رسّخ (\d+) ذكرى", o["text"])
            if m:
                cons_day[d] = cons_day.get(d, 0) + int(m.group(1))
        elif o["type"] == "teaching":
            teach.append(d)
        elif o["type"] == "host_says":
            n_host += 1

days = sorted(set(list(utt_day) + list(cons_day)))
ud = np.array([utt_day.get(d, 0) for d in days])
cd = np.array([cons_day.get(d, 0) for d in days])
cum_utt = np.cumsum(ud)
cum_mem = np.cumsum(cd)

fig, axes = plt.subplots(2, 1, figsize=(13, 9))
fig.patch.set_facecolor(DARK)
for ax in axes:
    ax.set_facecolor(DARK)
    ax.tick_params(colors="#9aa4b2", labelsize=9)
    for s in ax.spines.values():
        s.set_color("#2d333b")
axes[0].plot(days, cum_utt, color="#7ee787", lw=1.2)
axes[0].set_title(f"her voice over time — {cum_utt[-1]:,} utterances and counting",
                  color="white", fontsize=12)
axes[0].set_ylabel("total utterances", color="#9aa4b2")
for t in teach:
    axes[0].axvline(t, color="#e8b64c", alpha=0.5, lw=1)
axes[0].text(0.02, 0.95, f"gold lines = {len(teach)} teaching moments · "
             f"{n_host} conversations with host",
             transform=axes[0].transAxes, color="#e8b64c", fontsize=9,
             va="top", style="italic")
axes[1].plot(days, cum_mem, color="#d2a8ff", lw=1.2)
axes[1].set_title(f"her memory — {cum_mem[-1]:,} memories consolidated in sleep",
                  color="white", fontsize=12)
axes[1].set_ylabel("total memories", color="#9aa4b2")
axes[1].set_xlabel("day of her life", color="#9aa4b2")
fig.suptitle("FAJR — MEMORY & LANGUAGE (real journal data)", color="white",
             fontsize=14, fontweight="bold", family="monospace")
fig.text(0.5, 0.01, f'first words day {first_utt[0]} · still speaking on day {last_utt[0]}',
         ha="center", fontsize=9, color="#5a636e", style="italic")
fig.savefig(os.path.join(OUT, "proof-14-memory-language.png"), dpi=130,
            facecolor=DARK, bbox_inches="tight")
plt.close(fig)
print("done 13+14")
print("first utterance:", first_utt)
print("last utterance:", last_utt)
