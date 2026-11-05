# -*- coding: utf-8 -*-
"""Tests for ADAM & EVE P5 — perpetual learning (التعلم المطلق). Gates L1-L6.

(a) skill = pure function of lived practice only
(b) teaching preserves knowledge across generations (small loss)
(c) child learns parent-known skills at double speed
(d) no skill decay — knowledge ratchets upward
(e) tribe era milestones on collective-capability thresholds
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prng import PRNG
from culture import Culture, skill_level, SKILL_TAU
from sim import Sim
from child import Child


def test_l1_skill_is_pure_function_of_practice():
    c = Culture()
    assert skill_level(0) == 0.0
    assert abs(skill_level(SKILL_TAU) - (1 - 1 / 2.718281828)) < 0.01
    assert skill_level(4000) < 1.0, "asymptotic below 1 for real ranges"
    assert skill_level(4000) > skill_level(400) > skill_level(40), \
        "strictly increasing — early practice teaches most"
    # no way to raise skill except practice: transmit with teacher 0 grants 0
    assert c.transmit(0, 0.0) == 0
    print("L1 PASS: skill = 1-exp(-practice/TAU), practice-only rise")


def test_l2_transmission_with_small_loss():
    c = Culture()
    teacher, learner = 10000, 0
    seen_levels = []
    for _ in range(300):
        g = c.transmit(learner, skill_level(teacher))
        assert g >= 0
        learner += g
        seen_levels.append(skill_level(learner))
    target = skill_level(teacher) * 0.95
    assert seen_levels[-1] <= target + 1e-9, \
        "teaching alone must not cross teacher*(1-loss)"
    assert seen_levels[-1] > target * 0.9, \
        "learner must approach the teacher's level"
    assert all(b >= a for a, b in zip(seen_levels, seen_levels[1:])), \
        "monotone: knowledge preserved, never lost in transmission"
    print("L2 PASS: learner approaches teacher*0.95, monotone, small loss")


def test_l3_child_double_speed_on_parent_known():
    rng = PRNG(21)
    ch = Child(rng, 50.0, 50.0, "female")
    ch.age_ticks = int(31 * 1440)  # child stage
    ch.practice_skill("forage", 10)
    assert ch.practice["forage"] == 20, "child stage: 2x on lived practice"
    # infant: no gathering acts anyway; adult: 1x
    ch.age_ticks = int(121 * 1440)
    ch.practice_skill("forage", 10)
    assert ch.practice["forage"] == 30
    print("L3 PASS: child 2x learning (shoulders of parents)")


def test_l4_no_decay():
    sim = Sim(seed=11)
    sim.spawn_adam()
    a = sim.adam
    a.practice["forage"] = 500
    before = dict(a.practice)
    for _ in range(2000):
        sim.step()
    for s in before:
        assert a.practice[s] >= before[s], f"practice must never decay: {s}"
    print("L4 PASS: no skill decay over 2000 ticks")


def test_l5_era_fires_once_and_persists():
    sim = Sim(seed=11)
    sim.spawn_adam()
    sim.culture.teach_events = 49
    sim.culture.check_skill_eras(
        [{"forage": 0.1, "swim": 0.1, "build": 0.1}],
        sim.tick + 1, sim.journal, 100.0, 100.0)
    assert "era_teach" not in sim.culture.eras
    sim.culture.teach_events = 50
    sim.culture.check_skill_eras(
        [{"forage": 0.1, "swim": 0.1, "build": 0.1}],
        sim.tick + 2, sim.journal, 100.0, 100.0)
    assert "era_teach" in sim.culture.eras
    n = len(sim.culture.eras)
    sim.culture.check_skill_eras(
        [{"forage": 0.99, "swim": 0.99, "build": 0.99}],
        sim.tick + 3, sim.journal, 100.0, 100.0)
    # skill eras fire for the new thresholds, teach era not duplicated
    assert sim.culture.eras.count("era_teach") == 1, "eras journal once"
    assert n < len(sim.culture.eras)
    # persistence
    d = sim.culture.to_dict()
    c2 = Culture.from_dict(d)
    assert c2.eras == sim.culture.eras and c2.teach_events == 50
    print("L5 PASS: eras fire once, journaled, persisted")


def test_l6_fajr_practice_and_checkpoint():
    sim = Sim(seed=11)
    sim.spawn_adam()
    f = sim.fajr
    assert set(f.practice) == {"forage", "swim", "build"}
    # her historical stats seed her practice on old checkpoints
    assert f.practice["swim"] >= 0
    lv = f.skill_level("swim")
    assert 0.0 <= lv < 1.0
    # lived acts increment
    p0 = f.practice["forage"]
    f.practice["forage"] += 1
    assert f.skill_level("forage") >= skill_level(p0)
    import tempfile, os
    path = tempfile.mktemp(suffix=".json")
    sim.save(path)
    sim2 = Sim.load(path)
    os.unlink(path)
    assert sim2.fajr.practice == f.practice
    assert sim2.culture.to_dict() == sim.culture.to_dict()
    print("L6 PASS: her practice counters live + checkpoint round-trip")


if __name__ == "__main__":
    test_l1_skill_is_pure_function_of_practice()
    test_l2_transmission_with_small_loss()
    test_l3_child_double_speed_on_parent_known()
    test_l4_no_decay()
    test_l5_era_fires_once_and_persists()
    test_l6_fajr_practice_and_checkpoint()
    print("ALL L1-L6 PASS")
