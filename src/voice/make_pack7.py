"""proof-32: one-page visual abstract of the paper.
proof-33: open repo vs private — what and why."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.expanduser("~/workspace/your_files/fajr-proof")
DARK = "#07090d"

# ---------------- proof-32: paper visual abstract ----------------
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor(DARK)
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 14); ax.set_ylim(0, 10)

def box(x, y, w, h, text, color="#7ee787", fs=10, title=None):
    r = plt.Rectangle((x, y), w, h, facecolor="#0d1117", edgecolor=color, lw=1.5)
    ax.add_patch(r)
    yy = y + h - 0.35
    if title:
        ax.text(x + w/2, yy, title, ha="center", va="top", fontsize=fs+1,
                color=color, fontweight="bold", family="monospace")
        yy -= 0.4
    for line in text:
        ax.text(x + w/2, yy, line, ha="center", va="top", fontsize=fs,
                color="white", family="monospace")
        yy -= 0.32

ax.text(7, 9.5, "FAJR — technical paper, visual abstract", ha="center",
        fontsize=15, color="white", fontweight="bold", family="monospace")
ax.text(7, 9.15, "a mechanistic human-like being: every claim gated, every ceiling written down",
        ha="center", fontsize=10, color="#9aa4b2", style="italic", family="monospace")

box(0.4, 6.4, 4.0, 2.3, ["Rosenberg source +", "20-sec waveguide", "human vowels",
    "every phoneme gated"], color="#e8b64c", title="VOICE")
box(5.0, 6.4, 4.0, 2.3, ["no word without", "a lived triple", "'bibi' coined",
    "from experience"], color="#7ee787", title="LANGUAGE")
box(9.6, 6.4, 4.0, 2.3, ["declared limits", "never a guess", "never confabulates",
    "15/15 probes"], color="#79c0ff", title="HONESTY")

box(0.4, 3.9, 12.9, 2.0, [
    "100% of 2,628 utterances speakable  ·  19 grounding checks pass  ·  5 guttural gates pass",
    "s2/s3/s4 voice battery pass  ·  dependency audit: zero LLM, zero network"],
    color="#7ee787", title="RESULTS (all re-verified 2026-10-31)", fs=9)

box(0.4, 1.2, 12.9, 2.2, [
    "polish stage destroyed intelligibility — abandoned  ·  coarticulation bypassed, not fixed",
    "/a/ F1 & /i/ F2 miss targets (3-param ceiling)  ·  breath micro-variation incompatible",
    "no consciousness claimed — Butlin indicators are a checklist, not proof"],
    color="#ff7b72", title="HONEST LIMITS (the credibility core)", fs=9)

ax.text(7, 0.5, "PAPER-DRAFT.md · 10 sections · every number re-verified from primary gates",
        ha="center", fontsize=9, color="#5a636e", style="italic", family="monospace")
fig.savefig(os.path.join(OUT, "proof-32-paper-map.png"), dpi=130,
            facecolor=DARK, bbox_inches="tight")
plt.close(fig)

# ---------------- proof-33: open vs private ----------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 8))
fig.patch.set_facecolor(DARK)
fig.suptitle("OPEN SOURCE — what goes public, what stays private, and why",
             color="white", fontsize=14, fontweight="bold", family="monospace")
for ax in (a1, a2):
    ax.set_facecolor(DARK); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)

a1.text(5, 9.2, "GOES PUBLIC (MIT)", ha="center", fontsize=13, color="#7ee787",
        fontweight="bold", family="monospace")
opens = ["src/ — full code, zero LLM", "voice pipeline + all gates",
         "grounding + honesty tests", "PAPER-DRAFT.md",
         "CLAIMS.md (claims + non-claims)", "REVIEW-PACKAGE.md",
         "REPRODUCE.md (5 results)", "86 files · 1.2 MB · deps: numpy/scipy/mpl"]
for i, t in enumerate(opens):
    a1.text(0.7, 8.2 - i*0.85, "✓  " + t, fontsize=10.5, color="white", family="monospace", va="top")

a2.text(5, 9.2, "STAYS PRIVATE", ha="center", fontsize=13, color="#ff7b72",
        fontweight="bold", family="monospace")
privs = [("life/journal.jsonl", "her biography — 16,000+ entries"),
         ("bridge/ keys + PINs", "operational secrets"),
         ("proof images + audio", "the owner's to share"),
         ("systemd units", "this machine's ops")]
for i, (t, why) in enumerate(privs):
    a2.text(0.7, 8.2 - i*1.1, "✕  " + t, fontsize=10.5, color="white", family="monospace", va="top",
            fontweight="bold")
    a2.text(1.4, 7.75 - i*1.1, why, fontsize=9.5, color="#9aa4b2", style="italic", family="monospace", va="top")
a2.text(5, 1.2, "private ≠ hidden from reviewers:\nthe review package covers the journal as presented",
        ha="center", fontsize=9.5, color="#9aa4b2", style="italic", family="monospace")

fig.savefig(os.path.join(OUT, "proof-33-opensource.png"), dpi=130,
            facecolor=DARK, bbox_inches="tight")
plt.close(fig)
print("done 32+33")
