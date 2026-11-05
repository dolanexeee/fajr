# -*- coding: utf-8 -*-
"""Tests for ADAM & EVE P3 — teaching + bond + family drive. Gates T1-T6."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drives import DRIVES, WEIGHTS
from bond import PairBond, BOND_MILESTONES
from sim import Sim


def test_t1_family_drive():
    assert "family" in DRIVES, "family must be a drive"
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "weights must sum to 1"
    sim = Sim(seed=11)
    d0 = sim.fajr.drives.d["family"]
    # bond feeds the drive: higher bond -> faster rise (binding design)
    sim.fajr.drives.update({"asleep": False, "light": 1.0, "in_shelter": False,
                            "in_rain_exposed": False, "near_lantern": False,
                            "near_hazard": False, "cold_water": False,
                            "bond": 0.9, "mature": 1.0})
    rise_bonded = sim.fajr.drives.d["family"] - d0
    sim2 = Sim(seed=11)
    d0b = sim2.fajr.drives.d["family"]
    sim2.fajr.drives.update({"asleep": False, "light": 1.0, "in_shelter": False,
                             "in_rain_exposed": False, "near_lantern": False,
                             "near_hazard": False, "cold_water": False,
                             "bond": 0.0, "mature": 1.0})
    rise_lonely = sim2.fajr.drives.d["family"] - d0b
    assert rise_bonded > rise_lonely > 0, "bond must feed family-drive growth"
    print("T1 PASS: family drive exists; bond feeds its growth")


def test_t2_bond_grows_mutual():
    sim = Sim(seed=11)
    sim.fajr.mind._kinship_ready = True
    sim.step()  # spawn fires (partner not yet linked this tick)
    sim.step()  # now f.partner is linked before her cycle
    f, a = sim.fajr, sim.adam
    # put them together, both choosing each other, both wanting
    a.x, a.y = f.x + 2.0, f.y
    f.mind.chosen_intent = ("union", a.x, a.y, -1)
    a.intent = ("union", f.x, f.y)
    f.drives.d["family"] = 80.0
    a.drives.d["family"] = 80.0
    b0 = sim.bond.value
    for _ in range(50):
        # hold the wants (no full step, so no physiology drift)
        f.drives.d["family"] = 80.0
        a.drives.d["family"] = 80.0
        f.mind.chosen_intent = ("union", a.x, a.y, -1)
        sim._pair_tick(sim.tick)
        sim.tick += 1
    assert sim.bond.value > b0, "mutual union must grow the bond"
    assert sim.bond.union_ticks > 0
    assert sim.bond.conceive_ticks > 0, "P4 counter accrues"
    # milestones journaled behaviorally
    sim.bond.value = 0.9
    sim._pair_tick(sim.tick)
    texts = [e["text"] for e in sim.journal.entries if e["type"] == "bond"]
    assert any("زوج الوادي" in t for t in texts), texts
    print("T2 PASS: mutual closeness grows bond; milestones journaled")


def test_t3_imitation_teaches():
    sim = Sim(seed=11)
    sim.fajr.mind._kinship_ready = True
    sim.step()
    f, a = sim.fajr, sim.adam
    a.x, a.y = f.x + 5.0, f.y  # within observation radius
    p0 = a.practice["forage"]
    f.mind_events = {"ate": True}
    # P5: transmission needs a teacher who knows the skill — she is a
    # master forager by lived history (live: seeded from her stats)
    f.practice["forage"] = 500
    sim._imitation_tick(sim.tick)
    assert a.practice["forage"] > p0, "watching her eat must grant practice"
    # Guide never teaches him: no tongue, no learn path on Adam
    assert not hasattr(a, "tongue"), "Adam must have no language to be taught"
    print("T3 PASS: imitation grants practice; Guide has no path to him")


def test_t4_union_decision_hers():
    sim = Sim(seed=11)
    sim.fajr.mind._kinship_ready = True
    sim.step()  # spawn fires (partner not yet linked this tick)
    sim.step()  # now f.partner is linked before her cycle
    f, a = sim.fajr, sim.adam
    # pin her family drive high, everything else low
    for k in f.drives.d:
        f.drives.d[k] = 5.0
    f.drives.d["family"] = 95.0
    a.x, a.y = f.x + 30.0, f.y  # partner in range
    f.asleep = False
    f.mind.cycle(f, sim.world, f._raw, sim.tick, 1.0, sim.rng, sim.journal)
    assert f.mind.chosen_intent is not None
    assert f.mind.chosen_intent[0] == "union", \
        f"family drive must choose union via normal scoring, got {f.mind.chosen_intent}"
    print("T4 PASS: union chosen through her normal intent machinery")


def test_t5_bond_persists():
    sim = Sim(seed=11)
    sim.fajr.mind._kinship_ready = True
    sim.step()
    sim.bond.value = 0.42
    sim.bond.milestones.append("close")
    path = "/tmp/bond_ckpt_test.json"
    sim.save(path)
    sim2 = Sim.load(path)
    assert abs(sim2.bond.value - 0.42) < 1e-9
    assert sim2.bond.milestones == ["close"]
    # old checkpoint without bond key loads fine
    import json
    payload = json.load(open(path))
    del payload["state"]["bond"]
    payload["hash"] = Sim.canonical_hash(payload["state"])
    json.dump(payload, open(path, "w"))
    sim3 = Sim.load(path)
    assert sim3.bond.value == 0.0
    print("T5 PASS: bond persists; old checkpoints default cleanly")


def test_t6_weights_sane():    # valence must stay in [0,1] with the new drive present
    sim = Sim(seed=11)
    v = sim.fajr.drives.valence()
    assert 0.0 <= v <= 1.0, v
    print("T6 PASS: valence bounded with family drive")


def test_t7_observation_radius():
    # imitation works at observation radius (22), NOT only at closeness (8)
    sim = Sim(seed=11)
    sim.fajr.mind._kinship_ready = True
    sim.step()
    sim.step()
    f, a = sim.fajr, sim.adam
    a.x, a.y = f.x + 15.0, f.y  # within 22, outside 8
    p0 = a.practice["build"]
    f.mind_events = {"placed": True}
    f.practice["build"] = 500  # P5: she must know the skill to transmit it
    sim._imitation_tick(sim.tick)
    assert a.practice["build"] > p0, "must learn at observation radius"
    a.x, a.y = f.x + 40.0, f.y  # beyond observation
    p1 = a.practice["build"]
    sim._imitation_tick(sim.tick)
    assert a.practice["build"] == p1, "no learning beyond observation radius"
    print("T7 PASS: imitation at 22u observation radius, silent beyond")


def test_t8_family_voice_and_arrived_mutual():
    # REGRESSION (2026-11-04): at family weight 0.05 the drive was
    # voiceless — 25 live days, union_ticks = 0. The family drive must
    # outrank MODERATE vitals, and arrival must not erase his choice.
    from drives import WEIGHTS
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "weights sum to 1"
    assert WEIGHTS["family"] * 100.0 > 10.0, \
        "family at 100 must shout at vital volume"
    sim = Sim(seed=11)
    sim.fajr.mind._kinship_ready = True
    sim.step()
    f, a = sim.fajr, sim.adam
    # moderate vitals, desperate family -> he must choose union/follow
    for k in ("hunger", "thirst", "sleep", "safety", "curiosity"):
        a.drives.d[k] = 50.0
    a.drives.d["social"] = 10.0
    a.drives.d["family"] = 95.0
    a.intent = None
    a._decide(sim.world, f, sim.rng, sim.tick)
    assert a.intent is not None and a.intent[0] in ("union", "follow"), \
        f"family 95 must outrank vitals 50, chose {a.intent}"
    assert a.seeking_her, "seeking_her set on union/follow choice"
    # arrival clears intent but NOT the choice -> mutual still registers
    a.x, a.y = f.x + 1.0, f.y
    a.intent = None
    f.mind.chosen_intent = ("union", a.x, a.y, -1)
    f.drives.d["family"] = 95.0  # she wants too (broadened mutual, 2026-11-04)
    u0 = sim.bond.union_ticks
    sim._pair_tick(sim.tick)
    assert sim.bond.union_ticks > u0, \
        "arrived partner still counts as mutual (choice fulfilled)"
    print("T8 PASS: family has a voice; arrival keeps the choice")


if __name__ == "__main__":
    test_t1_family_drive()
    test_t2_bond_grows_mutual()
    test_t3_imitation_teaches()
    test_t4_union_decision_hers()
    test_t5_bond_persists()
    test_t6_weights_sane()
    test_t7_observation_radius()
    test_t8_family_voice_and_arrived_mutual()
    print("ALL P3 BOND TESTS PASS (T1-T8)")
