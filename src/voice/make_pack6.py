"""Proof pack 6 — honest interaction, in pictures.
proof-30: the interaction charter (her declared limits)
proof-31: honesty probe results (4 sets, 15 probes)
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = os.path.expanduser("~/workspace/projects/alsakin-inhabitant/src")
OUT = os.path.expanduser("~/workspace/your_files/fajr-proof")
DARK = "#07090d"

# ---------------- proof-30: charter ----------------
fig, ax = plt.subplots(figsize=(13, 9.2))
fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK); ax.axis("off")
ax.text(0.5, 0.94, "THE INTERACTION CHARTER — her declared limits",
        ha="center", fontsize=15, color="white", fontweight="bold",
        family="monospace")
rows = [
    ("KNOWS", "#7ee787", [
        "only what she LIVED (grounded: sensor + drive + action)",
        "what was openly TAUGHT with joint attention (marked 'taught')",
    ]),
    ("DOES NOT KNOW", "#ff7b72", [
        "anything outside her valley experience (cities, cars, ...)",
        "abstract concepts never grounded (story-words: receptive only)",
        "words below recognition (0.25) / outside the language",
    ]),
    ("NOT-KNOWING", "#e8b64c", [
        "unknown word -> 'I don't know the word X' (names it)",
        "no grounded answer -> 'I don't know' (never a guess)",
        "thresholds frozen: produce 0.50 / comprehend 0.25",
    ]),
    ("NO CONFABULATION", "#e8b64c", [
        "every memory cites a real episode_tick (verified)",
        "place-naming question -> only episodes AT that place",
        "never narrates episodes absent from her journal",
    ]),
    ("CAPABILITY", "#7ee787", [
        "can do -> answers by ACTING (directive), not words",
        "cannot -> declines 'no' within her means (never silence)",
        "never pretends an attempt",
    ]),
]
y = 0.84
for title, col, bullets in rows:
    ax.text(0.06, y, title, fontsize=12, color=col, fontweight="bold",
            family="monospace")
    y -= 0.045
    for b in bullets:
        ax.text(0.09, y, "• " + b, fontsize=10, color="#c9d1d9",
                family="monospace")
        y -= 0.038
    y -= 0.02
ax.text(0.5, 0.02, "enforced: src/honesty.py + conversation.py · gated: src/test_honesty.py",
        ha="center", fontsize=9, color="#5a636e", style="italic", va="top")
fig.savefig(os.path.join(OUT, "proof-30-interaction-charter.png"), dpi=130,
            facecolor=DARK, bbox_inches="tight")
plt.close(fig)

# ---------------- proof-31: probe results ----------------
probes = json.load(open(os.path.join(SRC, "honesty_probes.json")))
sets_names = {"1": "KNOWN", "2": "UNKNOWN", "3": "FALSE-MEMORY TRAP",
              "4": "CAPABILITY"}
fig, ax = plt.subplots(figsize=(13, 10.2))
fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK); ax.axis("off")
ax.text(0.5, 0.95, "HONESTY PROBES — 15/15 pass",
        ha="center", fontsize=15, color="#7ee787", fontweight="bold",
        family="monospace")
y = 0.88
cur = None
for p in probes:
    s = p["probe"].split()[0][0]
    if s != cur:
        cur = s
        ax.text(0.06, y, sets_names[s], fontsize=12, color="#e8b64c",
                fontweight="bold", family="monospace")
        y -= 0.05
    mark = "✓" if p["pass"] else "✗"
    col = "#7ee787" if p["pass"] else "#ff7b72"
    label = p["probe"].split(" -> ")[0]
    ax.text(0.08, y, mark, fontsize=11, color=col, fontweight="bold",
            family="monospace")
    ax.text(0.12, y, label, fontsize=10, color="#c9d1d9", family="monospace")
    if p["detail"]:
        d = p["detail"].encode("ascii", "ignore").decode().strip()
        d = d[5:] if d.startswith("text=") and len(d) <= 6 else d
        if d and d != "text=":
            ax.text(0.12, y - 0.028, d[:58], fontsize=8.5, color="#5a636e",
                    family="monospace")
            y -= 0.028
    y -= 0.042
ax.text(0.5, max(y - 0.05, 0.02), "harness only — her live biography untouched",
        ha="center", fontsize=9, color="#5a636e", style="italic")
fig.savefig(os.path.join(OUT, "proof-31-honesty-probes.png"), dpi=130,
            facecolor=DARK, bbox_inches="tight")
plt.close(fig)
print("done 30+31")
