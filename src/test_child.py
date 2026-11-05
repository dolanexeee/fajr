# -*- coding: utf-8 -*-
"""Tests for ADAM & EVE P4 — offspring by their free choice. Gates F1-F6.

Binding rule under test: conception follows THEIR sustained mutual union
choice (bond.conceive_ticks) — NO external trigger like "bond > 0.8".
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prng import PRNG
from sim import Sim
from child import Child, INFANT_DAYS, CHILD_DAYS
from bond import CONCEIVE_NEED, GESTATION_TICKS


def _sim_with_pair():
    sim = Sim(seed=11)
    sim.spawn_adam()
    return sim


def _force_mutual(sim, ticks):
    """Drive mutual-union ticks directly through the pair mechanic."""
    f, a = sim.fajr, sim.adam
    f.x, f.y, a.x, a.y = 100.0, 100.0, 101.0, 100.0
    f.partner = True
    for _ in range(ticks):
        sim.tick += 1
        f.drives.d["family"] = 90.0
        f.mind.chosen_intent = ("union", a.x, a.y)
        f._acting = None
        a.drives.d["family"] = 90.0
        a.intent = ("union", f.x, f.y)
        a.acting = None
        sim._pair_tick(sim.tick)


def test_f1_no_conception_without_mutual_choice():
    sim = _sim_with_pair()
    # 1000 ticks of plain life: no mutual union forced
    for _ in range(1000):
        sim.step()
    assert not sim.bond.pregnant, "pregnancy must NOT trigger on its own"
    assert sim.child is None
    print("F1 PASS: no external trigger — 1000 ticks, no pregnancy")


def test_f2_conception_then_birth():
    sim = _sim_with_pair()
    _force_mutual(sim, CONCEIVE_NEED)
    assert sim.bond.pregnant, "sustained mutual choice -> conception"
    assert sim.bond.gestation_left == GESTATION_TICKS
    # fast-forward gestation (physics, not a decision shortcut)
    sim.bond.gestation_left = 3
    sim._pregnancy_tick(sim.tick + 1)
    sim._pregnancy_tick(sim.tick + 2)
    sim._pregnancy_tick(sim.tick + 3)
    assert not sim.bond.pregnant
    assert sim.child is not None, "birth spawns the infant"
    assert sim.child.stage() == "infant"
    c = sim.child
    f = sim.fajr
    import math
    assert math.hypot(c.x - f.x, c.y - f.y) < 20.0, "born near the parents"
    print("F2 PASS: conception (their choice) -> gestation -> birth")


def test_f3_infant_cannot_gather_but_is_fed():
    sim = _sim_with_pair()
    rng = PRNG(5)
    sim.child = Child(rng, 100.0, 100.0, "female")
    c = sim.child
    # infant decisions: follow/sleep/drink only — never seek_food
    f = sim.fajr
    f.x, f.y = 100.0, 100.0
    seen = set()
    for _ in range(300):
        c._decide(sim.world, f, rng, sim.tick)
        seen.add(c.intent[0])
    assert "seek_food" not in seen, f"infant must not seek food, saw {seen}"
    # feeding: adult eats nearby -> infant hunger drops
    c.drives.d["hunger"] = 80.0
    c.x, c.y = 100.0, 100.0
    f.x, f.y = 102.0, 100.0
    sim.fajr.mind_events = {"ate": True}
    sim._feed_tick(sim.tick + 1)
    assert c.drives.d["hunger"] < 80.0, "shared meal must feed the infant"
    print("F3 PASS: infant can't gather; adult meals feed it")


def test_f4_staged_growth():
    rng = PRNG(6)
    c = Child(rng, 50.0, 50.0, "male")
    assert c.stage() == "infant" and c.maturity() == 0.0
    assert c._speed_factor() < 1.0, "infant moves slowly"
    c.age_ticks = int((INFANT_DAYS + 1) * 1440)
    assert c.stage() == "child" and c.maturity() == 0.5
    assert c._speed_factor() == 1.0
    c.age_ticks = int((CHILD_DAYS + 1) * 1440)
    assert c.stage() == "adult" and c.maturity() == 1.0
    print("F4 PASS: infant -> child -> adult at 30/120 days, staged params")


def test_f5_child_learns_double():
    rng = PRNG(7)
    c = Child(rng, 50.0, 50.0, "female")
    c.age_ticks = int((INFANT_DAYS + 1) * 1440)  # child stage
    c.practice_skill("forage", 10)
    assert c.practice["forage"] == 20, "child learns at 2x"
    c.age_ticks = int((CHILD_DAYS + 1) * 1440)   # adult stage
    c.practice_skill("forage", 10)
    assert c.practice["forage"] == 30, "adult learns at 1x"
    print("F5 PASS: child 2x learning (shoulders of parents)")


def test_f6_checkpoint_roundtrip_with_child():
    sim = _sim_with_pair()
    _force_mutual(sim, CONCEIVE_NEED)
    sim.bond.gestation_left = 5
    rng = PRNG(9)
    sim.child = Child(rng, 60.0, 60.0, "male")
    sim.child.age_ticks = 5000
    d = sim.state_dict()
    assert d["child"] is not None and d["child"]["sex"] == "male"
    assert d["bond"]["pregnant"] is True
    # round-trip through save/load
    import tempfile, os
    path = tempfile.mktemp(suffix=".json")
    sim.save(path)
    sim3 = Sim.load(path)
    os.unlink(path)
    assert sim3.child is not None and sim3.child.sex == "male"
    assert sim3.child.stage() == "infant"
    assert sim3.bond.pregnant is True
    assert sim3.bond.gestation_left == 5
    print("F6 PASS: checkpoint round-trip with pregnancy + child")


if __name__ == "__main__":
    test_f1_no_conception_without_mutual_choice()
    test_f2_conception_then_birth()
    test_f3_infant_cannot_gather_but_is_fed()
    test_f4_staged_growth()
    test_f5_child_learns_double()
    test_f6_checkpoint_roundtrip_with_child()
    print("ALL F1-F6 PASS")
