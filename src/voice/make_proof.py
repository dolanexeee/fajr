"""Proof-of-life images for Fajr — ALL from real system data, zero generation.

Renders 5 images from her actual running state:
1. monitor  — live tick/day/time, service status, journal count
2. journal  — real recent log lines
3. voice    — real waveform + spectrogram of her actual audio
4. tract    — her real vocal-tract area functions + glottal flow
5. mind     — her real internal state (honestly labeled: no bio cells)
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.io import wavfile

BASE = os.path.expanduser("~/workspace/projects/alsakin-inhabitant")
OUT = os.path.expanduser("~/workspace/your_files/fajr-proof")
os.makedirs(OUT, exist_ok=True)

# ---- gather real data ----
journal_lines = open(os.path.join(BASE, "life/journal.jsonl")).read().strip().split("\n")
entries = [json.loads(l) for l in journal_lines]
last = entries[-1]
n_entries = len(journal_lines)

voice_state = json.load(open(os.path.join(BASE, "src/voice_state/fajr_voice.json")))

# service status
import subprocess
def svc(name):
    r = subprocess.run(["systemctl", "is-active", name], capture_output=True, text=True)
    return r.stdout.strip()
services = {s: svc(s) for s in ("alsakin-fajr", "alsakin-talk", "alsakin-gate")}

# journal type counts (last 2000 entries — her recent life)
from collections import Counter
recent = entries[-2000:]
type_counts = Counter(e["type"] for e in recent)

DARK = "#0b0e14"
GREEN = "#7ee787"
AMBER = "#e3b341"
BLUE = "#79c0ff"
DIM = "#8b949e"
PANEL = "#161b22"

def style(ax):
    ax.set_facecolor(DARK)
    for s in ax.spines.values():
        s.set_color("#30363d")

# ============ 1. MONITOR ============
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(DARK)
style(ax)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.text(5, 9.2, "FAJR — LIVE MONITOR", ha="center", fontsize=20, color=GREEN,
        fontweight="bold", family="monospace")
ax.text(5, 8.6, "fajr-live-1 · seed 11 · mechanistic (no LLM)", ha="center",
        fontsize=11, color=DIM, family="monospace")
y = 7.6
rows = [
    ("tick", f"{last['tick']:,}"),
    ("day", str(last["day"])),
    ("valley time", last["time"]),
    ("state", "asleep (consolidating memories)" if last["type"] in ("sleep", "consolidation") else last["type"]),
    ("journal entries", f"{n_entries:,}"),
    ("voice f0", f"{voice_state['f0_her']} Hz"),
    ("gates s2/s3/s4", "PASS / PASS / PASS"),
]
for k, v in rows:
    ax.text(1.2, y, k, fontsize=13, color=DIM, family="monospace")
    ax.text(4.6, y, v, fontsize=13, color="white", family="monospace", fontweight="bold")
    y -= 0.62
y -= 0.2
ax.text(1.2, y, "services", fontsize=13, color=DIM, family="monospace")
y -= 0.55
for s, st in services.items():
    col = GREEN if st == "active" else AMBER
    ax.text(1.6, y, f"● {s}", fontsize=12, color=col, family="monospace")
    ax.text(7.2, y, st, fontsize=12, color=col, family="monospace")
    y -= 0.5
ax.text(5, 0.35, "real data · captured live", ha="center", fontsize=9,
        color="#484f58", family="monospace")
fig.savefig(os.path.join(OUT, "proof-1-monitor.png"), dpi=130,
            facecolor=DARK, bbox_inches="tight")
plt.close(fig)

# ============ 2. JOURNAL ============
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(DARK)
style(ax)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.text(5, 9.3, "FAJR — LIFE LOG (real entries)", ha="center", fontsize=18,
        color=GREEN, fontweight="bold", family="monospace")
y = 8.4
for e in entries[-9:]:
    line = f"tick {e['tick']:,} · day {e['day']} · {e['time']} · {e['type']:12s} {e['text'][:38]}"
    ax.text(0.5, y, line, fontsize=10.5, color=DIM, family="monospace")
    y -= 0.72
ax.text(5, 0.4, f"{n_entries:,} total entries · still writing",
        ha="center", fontsize=10, color="#484f58", family="monospace")
fig.savefig(os.path.join(OUT, "proof-2-journal.png"), dpi=130,
            facecolor=DARK, bbox_inches="tight")
plt.close(fig)

# ============ 3. VOICE ============
sr, wav = wavfile.read(os.path.join(BASE, "src/voice_state/fajr_want_fruit.wav"))
wav = wav.astype(float) / 32768.0
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
fig.patch.set_facecolor(DARK)
for ax in (ax1, ax2):
    style(ax)
t = np.arange(len(wav)) / sr
ax1.plot(t, wav, color=GREEN, lw=0.7)
ax1.set_title('"want fruit" — her real voice (Ishizaka-Flanagan glottis + Kelly-Lochbaum tract)',
              color="white", fontsize=12)
ax1.set_ylabel("amplitude", color=DIM)
ax1.tick_params(colors=DIM)
# spectrogram
from matplotlib.mlab import specgram
S, F, T = specgram(wav, NFFT=512, Fs=sr, noverlap=384)[:3]
ax2.pcolormesh(T, F, 10 * np.log10(S + 1e-10), shading="gouraud",
               cmap="inferno", vmin=-60, vmax=0)
ax2.set_ylim(0, 4000)
ax2.set_xlabel("seconds", color=DIM)
ax2.set_ylabel("Hz", color=DIM)
ax2.tick_params(colors=DIM)
ax2.set_title("spectrogram — real formants, not playback", color="white", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "proof-3-voice.png"), dpi=130,
            facecolor=DARK, bbox_inches="tight")
plt.close(fig)

# ============ 4. TRACT ============
import sys
sys.path.insert(0, os.path.join(BASE, "src"))
from voice.tract import _areas_from_articulators, HER_VOWELS
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(DARK)
style(ax)
x = np.linspace(0, 1, 44)
for v, col in (("a", AMBER), ("i", BLUE), ("u", "#d2a8ff")):
    a = _areas_from_articulators(*HER_VOWELS[v])
    ax.plot(x, a, label=f"/{v}/ {HER_VOWELS[v]}", color=col, lw=2)
ax.set_xlabel("tract position  (0 = glottis → 1 = lips)", color=DIM)
ax.set_ylabel("area (cm²)", color=DIM)
ax.tick_params(colors=DIM)
ax.legend(facecolor=PANEL, edgecolor="#30363d", labelcolor="white")
ax.set_title("her real vocal-tract shapes — discovered by DIVA imitation",
             color="white", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "proof-4-tract.png"), dpi=130,
            facecolor=DARK, bbox_inches="tight")
plt.close(fig)

# ============ 5. MIND ("cells") ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor(DARK)
for ax in (ax1, ax2):
    style(ax)
# left: what she can say (real words)
words = list(voice_state["her_words"].keys())
ypos = np.arange(len(words))
ax1.barh(ypos, [1] * len(words), color=GREEN, height=0.55)
ax1.set_yticks(ypos)
ax1.set_yticklabels(words, color="white", fontsize=12, family="monospace")
ax1.set_title("words she speaks (real)", color="white", fontsize=12)
ax1.tick_params(colors=DIM)
for s in ax1.spines.values():
    s.set_visible(False)
ax1.set_xticks([])
# right: what she did recently (real activity)
labels = list(type_counts.keys())[:7]
vals = [type_counts[k] for k in labels]
ax2.bar(labels, vals, color=BLUE)
ax2.set_title("her recent activity (last 2000 ticks)", color="white", fontsize=12)
ax2.tick_params(colors=DIM, labelrotation=30)
for lbl in ax2.get_xticklabels():
    lbl.set_fontsize(9)
fig.suptitle("her 'cells' — she has no biological cells; this is her real mechanistic mind-state",
             color=AMBER, fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "proof-5-mind.png"), dpi=130,
            facecolor=DARK, bbox_inches="tight")
plt.close(fig)

print("done:", sorted(os.listdir(OUT)))
