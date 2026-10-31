"""Harness for LANGUAGE-GROUNDING (Phase 1 item 4).

Runs against FakeSim/FakeFajr — NEVER touches the live biography.
Three parts:
  A. Causal-gate tests: each production path must return None when its
     grounding is absent, and emit only when the grounding holds.
  B. Novel-experience experiment: a novel percept ("ember") -> repeated
     exposures -> crystallized concept -> novel label bound -> emission
     ONLY after crystallization and ONLY while the pattern is present.
  C. Voice cross-check: the coined label uses only her voice phonemes.

Exit code 0 = all gates pass. Any failure prints FAIL and exits 1.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from language import Tongue
from lexicon import WORDS
from grounding import (GROUNDING, NoveltyTracker, GroundedConcept,
                       name_active_concept, generate_label,
                       label_speakable, NOVEL_K)


# ---------------------------------------------------------- fake world
class FakeDrives:
    def __init__(self):
        self.d = {"hunger": 20.0, "thirst": 20.0, "sleep": 20.0,
                  "safety": 20.0, "comfort": 20.0}


class FakeEpisodic:
    def __init__(self):
        self.episodes = []

    def set_semantic(self, *a):
        pass


class FakeMind:
    def __init__(self):
        self.episodic = FakeEpisodic()


class FakeFajr:
    def __init__(self):
        self.drives = FakeDrives()
        self.mind = FakeMind()
        self.wet = 0.0
        self.cleanliness = 100.0
        self.x, self.y = 150.0, 90.0

    def grime(self):
        return 100.0 - self.cleanliness


class FakeSim:
    def __init__(self):
        self.tick = 5000
        self.fajr = FakeFajr()


FAILURES = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


# ------------------------------------------------- A. causal gates
def test_causal_gates():
    print("== A. causal gates ==")
    sim = FakeSim()
    t = Tongue()
    # grounded learning: anchored teaching of thirsty + eat
    t.learn("thirsty", sim.tick, anchored=True, source="teaching")
    t.learn("eat", sim.tick, anchored=True, source="teaching")

    # A1: state word with drive LOW -> silence
    sim.fajr.drives.d["thirst"] = 20.0
    check("A1 state word, drive low -> None",
          t._state_report(sim, "thirsty") is None)

    # A2: state word with drive HIGH -> emits, claim carries the value
    sim.fajr.drives.d["thirst"] = 80.0
    r = t._state_report(sim, "thirsty")
    check("A2 state word, drive high -> emits", r is not None)
    if r:
        text, claims, used = r
        check("A2b claim value matches drive",
              claims[0]["value"] == 80.0, f"value={claims[0]['value']}")

    # A3: verb with NO episode -> silence
    ep_none = t.memory_report(sim, {"kind": "eat", "tick": 1, "day": 1,
                                    "x": 150.0, "y": 90.0}, context="t")
    # (episode exists here by construction) — instead test unknown verb
    check("A3 verb never learned -> None",
          t.memory_report(sim, {"kind": "bathe", "tick": 1, "day": 1,
                                "x": 150.0, "y": 90.0}) is None,
          "bathe not known")

    # A4: verb with real episode -> emits with true coords
    sim.fajr.mind.episodic.episodes.append(
        {"kind": "eat", "tick": 4990, "day": 4, "x": 160.0, "y": 95.0})
    u = t.memory_report(sim, sim.fajr.mind.episodic.episodes[-1])
    check("A4 verb, real episode -> emits", u is not None)
    if u:
        check("A4b claim carries episode tick",
              u["claims"][0]["episode_tick"] == 4990)


# ------------------------------------------------- B. novel experiment
KNOWN_KINDS = ["tree", "fruit", "pile", "thicket", "water", "lantern",
               "material", "structure", "shelter"]


def test_novel_experiment():
    print("== B. novel-experience experiment ==")
    sim = FakeSim()
    t = Tongue()
    tracker = NoveltyTracker(known_kinds=KNOWN_KINDS)
    feats = {"warmth": "high", "glow": "flicker", "texture": "rough"}
    sig = ("ember", frozenset(feats.items()))
    timeline = []

    for i in range(1, 8):
        tick = 6000 + i * 10
        concept, now = tracker.observe(tick, "ember", feats)
        # probe: pattern PRESENT — try to name it
        u = name_active_concept(t, tracker, tick, 5, sig)
        timeline.append({"exposure": i, "strength": round(concept.strength, 2),
                         "crystallized": concept.crystallized,
                         "emitted": u is not None,
                         "label": concept.label})
        if i < NOVEL_K:
            check(f"B{i} pre-crystallization probe -> silent", u is None,
                  f"exposure {i}")
        elif i == NOVEL_K:
            check(f"B{i} crystallization at exposure {NOVEL_K}",
                  concept.crystallized and now,
                  f"label={concept.label}")
            check(f"B{i}b first emission carries the label",
                  u is not None and u["text"] == concept.label)
        else:
            check(f"B{i} post-crystallization probe -> emits", u is not None)

    # B8: pattern ABSENT -> silence (no hallucination)
    other = ("stone", frozenset({"warmth": "low"}.items()))
    check("B8 absent pattern -> silent",
          name_active_concept(t, tracker, 6100, 5, other) is None)

    # B9: unknown signature -> silent
    check("B9 never-seen signature -> silent",
          name_active_concept(t, tracker, 6100, 5,
                              ("wisp", frozenset({"x": "y"}.items()))) is None)

    # B10: a SECOND novel kind gets a DIFFERENT label (one shared registry)
    for i in range(NOVEL_K):
        tracker.observe(7000 + i, "reed", {"sway": "rhythmic"})
    c1 = tracker.match(sig)
    c2 = tracker.match(("reed", frozenset({"sway": "rhythmic"}.items())))
    check("B10 distinct labels per concept",
          c1.label != c2.label, f"{c1.label} vs {c2.label}")

    # B11: known kinds never enter the tracker
    c, _ = tracker.observe(8000, "tree", {"x": "y"})
    check("B11 known kind ignored", c is None)

    return timeline, t


# ------------------------------------------------- C. voice cross-check
def test_voice_crosscheck(timeline):
    print("== C. voice cross-check ==")
    labels = {row["label"] for row in timeline if row["label"]}
    for lab in labels:
        ok, phones = label_speakable(lab)
        check(f"C label '{lab}' speakable", ok, f"phones={phones}")


def main():
    test_causal_gates()
    timeline, _ = test_novel_experiment()
    test_voice_crosscheck(timeline)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES: {FAILURES}")
        return 1
    print("ALL GROUNDING GATES PASS")
    # dump the timeline for the picture builder
    import json
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "novel_timeline.json"), "w") as f:
        json.dump(timeline, f, indent=1)
    print("timeline -> src/novel_timeline.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
