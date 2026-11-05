# -*- coding: utf-8 -*-
"""Tests for structured building (Stage 4). Gates B1-B5."""
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from building import BuildProject, SCHEMAS, SCHEMA_AR


def test_b1_hut_ring():
    p = BuildProject("hut", 100.0, 50.0, 0, 1)
    assert len(p.slots) == 8, f"hut needs 8 slots, got {len(p.slots)}"
    for x, y in p.slots:
        r = math.hypot(x - 100.0, y - 50.0)
        assert abs(r - 3.0) < 1e-9, f"slot off ring: r={r}"
    print("B1 PASS: hut = 8 slots on r=3 ring")


def test_b2_wall_line():
    p = BuildProject("wall", 0.0, 0.0, 0, 1)
    assert len(p.slots) == 5
    xs = [s[0] for s in p.slots]
    assert xs == sorted(xs) and all(s[1] == 0.0 for s in p.slots)
    print("B2 PASS: wall = 5 slots in a line")


def test_b3_progression():
    p = BuildProject("hut", 0.0, 0.0, 0, 1)
    marks = [p.record_placement() for _ in range(8)]
    assert marks[3] == "half", f"expected half at 4th, got {marks[3]}"
    assert marks[7] == "done", f"expected done at 8th, got {marks[7]}"
    assert not p.live
    assert p.next_slot() is None
    assert p.record_placement() is None
    print("B3 PASS: half at 4, done at 8, dead after")


def test_b4_persistence():
    p = BuildProject("hut", 10.0, 20.0, 999, 7)
    p.record_placement()
    p.record_placement()
    d = p.to_dict()
    q = BuildProject.from_dict(d)
    assert q.idx == 2 and q.live and q.schema == "hut"
    assert q.slots == p.slots and q.cx == 10.0 and q.start_day == 7
    print("B4 PASS: to_dict/from_dict round-trip")


def test_b5_ar_labels():
    assert SCHEMA_AR["hut"] == "كوخ" and SCHEMA_AR["wall"] == "جدار"
    assert set(SCHEMAS) == set(SCHEMA_AR)
    print("B5 PASS: schema labels complete")


if __name__ == "__main__":
    test_b1_hut_ring()
    test_b2_wall_line()
    test_b3_progression()
    test_b4_persistence()
    test_b5_ar_labels()
    print("ALL BUILDING TESTS PASS (B1-B5)")
