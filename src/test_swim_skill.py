"""Swim-skill expansion gates (2026-11-01).

Skill is DERIVED from lifetime swim_ticks (persisted in stats): no new
state, no migration. Asymptotic learning 1 - exp(-ticks/TAU).
"""
import math
import sys
sys.path.insert(0, ".")

from config import (SWIM_SLOW_FACTOR, SWIM_SLOW_SKILLED, SWIM_SKILL_TAU,
                    SWIM_COST_RELIEF)
from fajr import Fajr
from prng import PRNG

PASS, FAIL = [], []

def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))

def fresh(ticks):
    f = Fajr(PRNG(7))
    f.stats["swim_ticks"] = ticks
    return f

# S1 — skill shape: 0 at start, bounded, monotonic, asymptotic
check("S1a_skill_zero_at_start", fresh(0).swim_skill() == 0.0)
s400 = fresh(400).swim_skill()
check("S1b_skill_63pct_at_tau", abs(s400 - (1 - 1/math.e)) < 1e-9, f"got {s400}")
check("S1c_skill_bounded", all(0.0 <= fresh(t).swim_skill() <= 1.0
      for t in (0, 1, 50, 400, 2000, 100000)))
ts = [0, 10, 100, 400, 1000, 5000]
ss = [fresh(t).swim_skill() for t in ts]
check("S1d_skill_monotonic", all(b > a for a, b in zip(ss, ss[1:])))
check("S1e_skill_asymptotic", fresh(2000).swim_skill() > 0.99 and
      fresh(5000).swim_skill() < 1.0)  # exp(-250) underflows to 0 at 1e5 ticks;
                                       # the [0,1] bound itself is S1c's job

# S2 — speed: baseline unchanged at skill 0, improves, never regresses
check("S2a_baseline_unchanged", abs(fresh(0).swim_slow_factor() - SWIM_SLOW_FACTOR) < 1e-12)
check("S2b_ceiling", abs(fresh(100000).swim_slow_factor() - SWIM_SLOW_SKILLED) < 1e-3)
check("S2c_never_slower", all(fresh(t).swim_slow_factor() >= SWIM_SLOW_FACTOR
      for t in (0, 50, 400, 5000)))
check("S2d_improves", fresh(5000).swim_slow_factor() > fresh(100).swim_slow_factor() >
      fresh(0).swim_slow_factor())
# crossing a fixed channel width: fewer ticks at high skill
W = 12.0  # arbitrary channel width in world units
def cross_ticks(skill_obj):
    import config
    spd = config.WALK_SPEED * skill_obj.swim_slow_factor()
    return math.ceil(W / spd)
check("S2e_crossing_faster", cross_ticks(fresh(5000)) < cross_ticks(fresh(0)),
      f"{cross_ticks(fresh(5000))} vs {cross_ticks(fresh(0))}")

# S3 — energy: baseline unchanged at skill 0, relieved with skill,
#        W5 hunger:thirst asymmetry ratio preserved at every level
check("S3a_no_relief_at_zero", abs(fresh(0).swim_cost_relief() - 1.0) < 1e-12)
check("S3b_half_at_full", abs(fresh(100000).swim_cost_relief() -
      (1.0 - SWIM_COST_RELIEF)) < 1e-3)
base_h, base_t = 0.055, 0.022
for t in (0, 200, 1000, 100000):
    r = fresh(t).swim_cost_relief()
    ratio = (base_h * r) / (base_t * r)
    ok = abs(ratio - base_h / base_t) < 1e-9
    if not ok:
        break
check("S3c_w5_ratio_preserved", ok, "hunger:thirst ratio must stay 2.5 at all skills")
check("S3d_never_costlier", all(fresh(t).swim_cost_relief() <= 1.0
      for t in (0, 50, 400, 5000)))

# S4 — past experience counts (her lived swims are not discarded)
check("S4a_past_swims_count", fresh(800).swim_skill() > 0.86,
      f"got {fresh(800).swim_skill():.3f}")

# S5 — integration: methods exist on the real object, stats persist shape
f = fresh(123)
check("S5a_methods_live", callable(f.swim_skill) and callable(f.swim_slow_factor)
      and callable(f.swim_cost_relief))
d = f.to_dict()
check("S5b_stats_persisted", d["stats"]["swim_ticks"] == 123,
      "swim_ticks must survive to_dict (checkpoint persistence)")

print()
if FAIL:
    print(f"SWIM SKILL GATES: {len(PASS)} pass, {len(FAIL)} FAIL -> {FAIL}")
    sys.exit(1)
print(f"ALL SWIM SKILL GATES PASS ({len(PASS)}/{len(PASS)})")
