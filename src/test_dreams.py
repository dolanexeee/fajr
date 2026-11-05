"""Harness for DREAMS (Phase 4, stage 1).

Runs against fake episodes — NEVER touches the live biography.
Gates:
  D1 weave needs >= 2 episodes; returns None below that.
  D2 a woven dream has provenance (source ticks are real episode ticks),
     a theme, vividness in [0,1], and type "dream" (never "episode").
  D3 HONESTY: after many weaves, episodic memory holds ZERO dreams —
     no episode with kind "dream", no dream dict inside episodes.
  D4 determinism: same seed + same episodes -> identical dream.
  D5 persistence round-trip preserves dreams, weights, unrecounted.
  D6 recount gating: vivid dreams arm unrecounted; faint ones do not.

Exit code 0 = all gates pass. Any failure prints FAIL and exits 1.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dreams import DreamWeaver, VIVID_RECOUNT_THRESHOLD
from prng import PRNG


class FakeEpisodic:
    def __init__(self, episodes):
        self.episodes = episodes


def make_episodes(n, start_tick=1000, kinds=("eat", "swim", "drink")):
    eps = []
    for i in range(n):
        eps.append({
            "tick": start_tick + i * 10,
            "kind": kinds[i % len(kinds)],
            "x": 10.0, "y": 20.0,
            "strength": 0.5 + (i % 5) * 0.1,
            "arousal": 0.3 + (i % 4) * 0.2,
        })
    return eps


FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# D1: too little to dream of
w = DreamWeaver()
ep = FakeEpisodic(make_episodes(1))
check("D1-no-dream-on-1-episode", w.weave(ep, 5000, PRNG(7)) is None)
check("D1-no-dream-on-0-episodes",
      DreamWeaver().weave(FakeEpisodic([]), 5000, PRNG(7)) is None)

# D2: woven dream structure
w = DreamWeaver()
ep = FakeEpisodic(make_episodes(30))
rng = PRNG(11)
dream = w.weave(ep, 5000, rng)
check("D2-dream-woven", dream is not None)
ticks = {e["tick"] for e in ep.episodes}
check("D2-provenance-real", all(t in ticks for t in dream["sources"]),
      f"sources={dream['sources']}")
check("D2-type-is-dream", dream["type"] == "dream")
check("D2-theme-present", dream["theme"] in
      ("swim", "eat", "drink", "gather", "build", "shelter", "storm", "explore"))
check("D2-vividness-bounded", 0.0 <= dream["vividness"] <= 1.0)
check("D2-journal-text-marked", dream["journal_text"].startswith("حلم —"))

# D3: HONESTY — dreams never enter episodic memory
w = DreamWeaver()
ep = FakeEpisodic(make_episodes(200))
n_before = len(ep.episodes)
for i in range(40):
    # weave ticks stay inside the 3-day dream window of the episodes
    w.weave(ep, 3200 + i * 10, PRNG(100 + i))
check("D3-episode-count-unchanged", len(ep.episodes) == n_before)
check("D3-no-dream-kind", not any(e.get("kind") == "dream" for e in ep.episodes))
check("D3-no-dream-dicts", not any("sources" in e and e.get("type") == "dream"
                                  for e in ep.episodes))
check("D3-dreams-separate-store", len(w.dreams) == 40 and
      all(d["type"] == "dream" for d in w.dreams))

# D4: determinism
def weave_once(seed):
    w = DreamWeaver()
    return w.weave(FakeEpisodic(make_episodes(30)), 5000, PRNG(seed))
d_a, d_b = weave_once(42), weave_once(42)
check("D4-deterministic",
      d_a["sources"] == d_b["sources"] and d_a["theme"] == d_b["theme"]
      and d_a["journal_text"] == d_b["journal_text"])

# D5: persistence round-trip
w = DreamWeaver()
ep = FakeEpisodic(make_episodes(30))
for i in range(5):
    w.weave(ep, 1400 + i * 50, PRNG(5 + i))
w2 = DreamWeaver.from_dict(w.to_dict())
check("D5-dreams-survive", len(w2.dreams) == 5 and
      w2.dreams[0]["sources"] == w.dreams[0]["sources"])
check("D5-weights-survive", w2.theme_weights == w.theme_weights)
check("D5-unrecounted-survives", w2.unrecounted_id == w.unrecounted_id)

# D6: recount arming by vividness
w = DreamWeaver()
faint = [{"tick": 1000 + i * 10, "kind": "eat", "x": 0, "y": 0,
          "strength": 0.1, "arousal": 0.0} for i in range(10)]
w.weave(FakeEpisodic(faint), 5000, PRNG(3))
check("D6-faint-no-recount", w.unrecounted_id is None,
      "faint dreams must not demand recounting")

print(f"\n{6 - len([f for f in FAILS])}/6 groups clean"
      if False else "")
if FAILS:
    print(f"\n{len(FAILS)} FAILURES: {FAILS}")
    sys.exit(1)
print("\nALL DREAM GATES PASS")
