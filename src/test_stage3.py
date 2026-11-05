"""Harness for Phase 4 batch 3 — integration (the final honest depth).

Fake mind/fajr/journal — never the live biography.
  R1 dawn ritual: fires on the wake crossing, once per day.
  R2 evening ritual: fires at sleep onset, mood-colored text.
  O1 obstacle memory: 2nd failure in 3 days -> lesson + frustration
     journaled, mood dipped, _decide bias 0.5 on that kind.
  O2 obstacle decay: after the 3-day window the bias returns to 1.0.
  E1 contentment: ate + high valence -> journaled once/day + mood bump.
  V1 reverie: idle + quiet drives -> revisits a strong episode (a REAL
     memory), once/day, and the episode grows stronger.

Exit 0 = all pass.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mind import Mind
from prng import PRNG

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


class FakeJournal:
    def __init__(self):
        self.entries = []

    def log(self, tick, type_, text, x=None, y=None):
        self.entries.append({"tick": tick, "type": type_, "text": text})


class FakeDrives:
    def __init__(self, **kw):
        base = {"hunger": 10.0, "thirst": 10.0, "sleep": 10.0,
                "safety": 10.0, "curiosity": 10.0, "comfort": 10.0,
                "social": 10.0, "creation": 10.0}
        base.update(kw)
        self.d = base


class FakeFajr:
    def __init__(self, asleep=False):
        self.x, self.y = 50.0, 60.0
        self.asleep = asleep
        self.drives = FakeDrives()


class FakeScene:
    def __init__(self):
        self.objects = {}
        self.foreground = []
        self.pred_error_self_discounted = 0.0


class FakeRng:
    def __init__(self, u=0.05):
        self._u = u

    def uniform(self):
        return self._u


def types(j):
    return [e["type"] for e in j.entries]


# R1: dawn ritual
m = Mind(PRNG(11))
j = FakeJournal()
f = FakeFajr(asleep=False)
m._was_asleep = True                      # the wake crossing
m._dawn_ritual(f, 2 * 1440, j)            # day 3
dawns = [e for e in j.entries if e["type"] == "ritual_dawn"]
check("R1-dawn-fires", len(dawns) == 1, f"found {len(dawns)}")
check("R1-dawn-honest", dawns and "الفجر" in dawns[0]["text"],
      dawns[0]["text"][:40] if dawns else "")
m._dawn_ritual(f, 2 * 1440 + 50, j)        # same day -> silent
check("R1-dawn-daily", len([e for e in j.entries if e["type"] == "ritual_dawn"]) == 1)

# R2: evening ritual, mood-colored
m2 = Mind(PRNG(12))
j2 = FakeJournal()
f2 = FakeFajr(asleep=True)
m2.mood = 0.5
m2._evening_ritual(f2, 3 * 1440, j2)      # day 4
evs = [e for e in j2.entries if e["type"] == "ritual_evening"]
check("R2-evening-fires", len(evs) == 1)
check("R2-evening-mood", evs and "راضية" in evs[0]["text"],
      evs[0]["text"] if evs else "")
m2b = Mind(PRNG(13))
j2b = FakeJournal()
m2b.mood = -0.5
m2b._evening_ritual(f2, 3 * 1440, j2b)
evsb = [e for e in j2b.entries if e["type"] == "ritual_evening"]
check("R2-evening-low-mood", evsb and "راضية" not in evsb[0]["text"])

# O1: obstacle memory — two failures teach
m3 = Mind(PRNG(14))
j3 = FakeJournal()
m3.note_obstacle("eat", 1000, 5.0, 5.0, j3)
check("O1-first-silent", "obstacle_learn" not in types(j3))
check("O1-bias-one", m3._obstacle_bias("eat", 1100) == 1.0)
m3.note_obstacle("eat", 2000, 6.0, 6.0, j3)   # second within 3 days
t3 = types(j3)
check("O1-lesson", "obstacle_learn" in t3, str(t3))
check("O1-frustration", "frustration" in t3, str(t3))
check("O1-frustration-honest",
      any("محبطة" in e["text"] for e in j3.entries if e["type"] == "frustration"))
check("O1-mood-dip", m3.mood < 0.0, f"mood={m3.mood}")
check("O1-bias-half", m3._obstacle_bias("eat", 2100) == 0.5)
check("O1-bias-other-kind", m3._obstacle_bias("drink", 2100) == 1.0)

# O2: decay after the window
check("O2-decayed", m3._obstacle_bias("eat", 2000 + 4320 + 10) == 1.0)

# E1: contentment — ate + high valence, once a day
m4 = Mind(PRNG(15))
j4 = FakeJournal()
f4 = FakeFajr()
m4.affect.valence = 0.8
m4._record_events(f4, FakeScene(), 4 * 1440, {"ate": True}, "clear", j4)
cons = [e for e in j4.entries if e["type"] == "contentment"]
check("E1-content-fires", len(cons) == 1, str(types(j4)))
check("E1-content-honest", cons and "راضية" in cons[0]["text"])
m4._record_events(f4, FakeScene(), 4 * 1440 + 200, {"ate": True}, "clear", j4)
check("E1-content-daily",
      len([e for e in j4.entries if e["type"] == "contentment"]) == 1)
m4b = Mind(PRNG(16))
j4b = FakeJournal()
m4b.affect.valence = 0.3   # low mood -> no contentment
m4b._record_events(f4, FakeScene(), 4 * 1440, {"ate": True}, "clear", j4b)
check("E1-content-gated", "contentment" not in types(j4b))

# V1: reverie — idle, quiet drives, revisits a REAL episode
m5 = Mind(PRNG(17))
j5 = FakeJournal()
f5 = FakeFajr(asleep=False)
for i, kind in enumerate(["eat", "swim", "place", "prick"]):
    m5.episodic.record(5000 + i * 100, kind, 10.0, 20.0, ["x"],
                       {"hunger": 10.0}, 0.5, 0.6, outcome=0.9)
# force headroom so reconsolidation is observable
for e in m5.episodic.episodes:
    e["strength"] = 0.5
strong_before = max(e["strength"] for e in m5.episodic.episodes)
m5._reverie(f5, 6 * 1440, j5, FakeRng(0.05))
revs = [e for e in j5.entries if e["type"] == "reverie"]
check("V1-reverie-fires", len(revs) == 1, str(types(j5)))
# the revisited text must come from a real episode's kind (honesty)
kinds = {e["kind"] for e in m5.episodic.episodes}
check("V1-reverie-real",
      revs and any(m5.STORY_FRAGMENTS.get(k, k) in revs[0]["text"] for k in kinds),
      revs[0]["text"][:50] if revs else "")
strong_after = max(e["strength"] for e in m5.episodic.episodes)
check("V1-reconsolidation", strong_after > strong_before,
      f"{strong_before} -> {strong_after}")
m5._reverie(f5, 6 * 1440 + 300, j5, FakeRng(0.05))  # same day -> silent
check("V1-reverie-daily",
      len([e for e in j5.entries if e["type"] == "reverie"]) == 1)
# busy drives -> no reverie
m5b = Mind(PRNG(18))
j5b = FakeJournal()
f5b = FakeFajr(asleep=False)
f5b.drives = FakeDrives(hunger=80.0)
for i in range(3):
    m5b.episodic.record(5000 + i * 100, "eat", 10.0, 20.0, ["x"],
                        {"hunger": 10.0}, 0.5, 0.6, outcome=0.9)
m5b._reverie(f5b, 6 * 1440, j5b, FakeRng(0.05))
check("V1-reverie-quiet-only", "reverie" not in types(j5b))

print()
if FAILS:
    print(f"{len(FAILS)} FAILURES: {FAILS}")
    sys.exit(1)
print("ALL STAGE-3 GATES PASS")
