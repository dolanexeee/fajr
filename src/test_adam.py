# -*- coding: utf-8 -*-
"""Tests for ADAM & EVE P2 — Adam enters zero-knowledge. Gates A1-A5."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prng import PRNG
from adam import Adam, SPAWN_POS, SKILLS
from sim import Sim
from world import is_passable


def _fruit_tree(world):
    for i, t in enumerate(world.trees):
        if t.fruit > 0:
            return i, t
    return None, None


def test_a1_blank_slate():
    rng = PRNG(777)
    a = Adam(rng)
    assert all(a.skill(s) == 0.0 for s in SKILLS), "skills must start at 0"
    assert a.food_memory == {}, "no food memory at birth"
    assert a.practice == {s: 0 for s in SKILLS}
    assert a.intent is None and a.acting is None and not a.asleep
    assert is_passable(*SPAWN_POS), "spawn point must be passable"
    print("A1 PASS: Adam spawns blank — skills 0, no memory")


def test_a2_hunger_seeks_food():
    sim = Sim(seed=11)
    sim.spawn_adam()
    a = sim.adam
    a.drives.d["hunger"] = 95.0
    a.drives.d["thirst"] = 10.0
    a.drives.d["sleep"] = 10.0
    a.drives.d["social"] = 10.0
    a.drives.d["curiosity"] = 10.0
    a._decide(sim.world, sim.fajr, sim.rng, sim.tick)
    assert a.intent[0] == "seek_food", f"hungry adam should seek food, got {a.intent}"
    print("A2 PASS: hunger drives food-seeking via normal arbitration")


def test_a3_eat_by_living():
    sim = Sim(seed=11)
    sim.spawn_adam()
    a = sim.adam
    i, t = _fruit_tree(sim.world)
    assert t is not None, "need a fruit tree for the test"
    a.x, a.y = t.x + 1.0, t.y  # stand next to it
    a.drives.d["hunger"] = 90.0
    h0 = a.drives.d["hunger"]
    for _ in range(200):
        a.step(sim.world, sim.fajr, sim.tick, 1.0, sim.rng, sim.journal)
        sim.tick += 1
        if a.stats["eats"] >= 1:
            break
    assert a.stats["eats"] >= 1, "adam should have eaten"
    assert a.drives.d["hunger"] < h0, "hunger must drop after eating"
    assert a.practice["forage"] > 0, "practice accrues by living only"
    assert a.skill("forage") > 0.0, "skill rises from practice"
    print("A3 PASS: he eats, hunger drops, skill rises by practice")


def test_a4_checkpoint_roundtrip(tmp_path=None):
    import json
    sim = Sim(seed=11)
    sim.fajr.mind._kinship_ready = True
    sim.step()  # spawn fires
    assert sim.adam is not None
    sim.adam.drives.d["hunger"] = 61.0
    path = "/tmp/adam_ckpt_test.json"
    sim.save(path)
    sim2 = Sim.load(path)
    assert sim2.adam is not None, "adam must survive checkpoint"
    assert abs(sim2.adam.x - sim.adam.x) < 1e-9
    assert abs(sim2.adam.drives.d["hunger"] - 61.0) < 1e-9
    assert sim2.adam.practice == sim.adam.practice
    print("A4 PASS: Adam persists through save/load")


def test_a5_spawn_gating():
    sim = Sim(seed=11)
    assert sim.fajr.mind._kinship_ready is False
    sim.step()
    assert sim.adam is None, "no spawn before she is ready (P1 gate)"
    sim.fajr.mind._kinship_ready = True
    sim.step()
    assert sim.adam is not None, "spawns once she is ready"
    entries = [e for e in sim.journal.entries if e["type"] == "adam_entered"]
    assert len(entries) == 1
    assert "لا يعرف شيئًا" in entries[0]["text"]
    sim.step()
    entries = [e for e in sim.journal.entries if e["type"] == "adam_entered"]
    assert len(entries) == 1, "spawn fires exactly once"
    print("A5 PASS: spawn gated on P1 readiness, fires once, journaled")


if __name__ == "__main__":
    test_a1_blank_slate()
    test_a2_hunger_seeks_food()
    test_a3_eat_by_living()
    test_a4_checkpoint_roundtrip()
    test_a5_spawn_gating()
    print("ALL P2 ADAM TESTS PASS (A1-A5)")
