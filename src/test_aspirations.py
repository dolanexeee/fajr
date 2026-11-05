"""Harness for ASPIRATIONS (Phase 4, stage 2).

Fake dreamer/fajr/journal — never the live biography.
  A1 birth: theme weight >= BIRTH_W births one aspiration, journaled,
     weight spent, max 3 live.
  A2 bias: bonus() > 1 for themed intents, == 1 for others.
  A3 fulfill: themed stat growth retires it, journaled.
  A4 decay: unfulfilled aspirations fade over days.
  A5 persistence round-trip.

Exit 0 = all pass.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aspirations import AspirationEngine, BIRTH_W

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


class FakeDreamer:
    def __init__(self, weights):
        self.theme_weights = dict(weights)


class FakeFajr:
    def __init__(self):
        self.x, self.y = 1.0, 2.0
        self.stats = {"eats": 10, "swims": 5, "places": 3, "distance": 100.0,
                      "drinks": 0, "gathers": 0, "sleeps": 0}


class FakeJournal:
    def __init__(self):
        self.entries = []

    def log(self, tick, type_, text, x=None, y=None):
        self.entries.append({"type": type_, "text": text})


# A1: birth
e = AspirationEngine()
j = FakeJournal()
f = FakeFajr()
e.update(FakeDreamer({"eat": BIRTH_W + 0.5}), f, 1440, j)
live = [a for a in e.aspirations if a["status"] == "live"]
check("A1-born", len(live) == 1 and live[0]["theme"] == "eat")
check("A1-journaled", any(x["type"] == "aspiration" and "طموح" in x["text"] for x in j.entries))
check("A1-no-double-birth",
      (e.update(FakeDreamer({"eat": 99.0}), f, 1500, j),
       len([a for a in e.aspirations if a["status"] == "live"]) == 1)[1])

# A2: bias
check("A2-bonus-themed", e.bonus("eat") > 1.0, f"bonus={e.bonus('eat')}")
check("A2-bonus-seek_food", e.bonus("seek_food") > 1.0)
check("A2-no-bonus-other", e.bonus("sleep") == 1.0)

# A3: fulfill
f.stats["eats"] = 11  # she ate
e.update(FakeDreamer({}), f, 1600, j)
a = e.aspirations[0]
check("A3-fulfilled", a["status"] == "fulfilled")
check("A3-fulfill-journaled",
      any("حققت طموحها" in x["text"] for x in j.entries))
check("A3-bonus-gone", e.bonus("eat") == 1.0)

# A4: decay over days
e2 = AspirationEngine()
j2 = FakeJournal()
f2 = FakeFajr()
e2.update(FakeDreamer({"build": BIRTH_W + 1.0}), f2, 1440, j2)
a2 = e2.aspirations[0]
s0 = a2["strength"]
for day in range(2, 8):  # 6 more days, no building
    e2.update(FakeDreamer({}), f2, day * 1440, j2)
check("A4-decayed", a2["strength"] < s0, f"{s0} -> {a2['strength']}")
check("A4-faded", a2["status"] == "faded", a2["status"])

# A5: persistence
e3 = AspirationEngine()
e3.update(FakeDreamer({"explore": BIRTH_W + 0.2}), FakeFajr(), 2000, FakeJournal())
r = AspirationEngine.from_dict(e3.to_dict())
check("A5-roundtrip", len(r.aspirations) == 1 and
      r.aspirations[0]["theme"] == "explore" and r.metrics["born"] == 1)

if FAILS:
    print(f"\n{len(FAILS)} FAILURES: {FAILS}")
    sys.exit(1)
print("\nALL ASPIRATION GATES PASS")
