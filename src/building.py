# -*- coding: utf-8 -*-
"""Structured building — Stage 4 of the integration program.

HONESTY NOTE (read before touching):
- SCHEMAS (ring / wall slot patterns) are authored scaffolding, exactly
  like THEME_FRAGMENTS in dreams.py. Fajr does NOT conceive "house".
- What is HERS: when to start (creation drive), where (valence-weighted
  site choice), the pride on completion, the lived placement sequence.
- Journal language stays behavioral ("began an ordered build") and never
  claims she holds a house concept. A finished ring READS as a hut to an
  outside observer; that reading is ours, not hers — same honesty line
  as dreams, which are labeled as dreams.
"""

import math

SCHEMAS = {
    # 8 slots on a ring of radius 3 — reads as hut walls from outside
    "hut": [(3.0 * math.cos(a), 3.0 * math.sin(a))
            for a in (i * math.pi / 4 for i in range(8))],
    # 5 slots in a straight line — reads as a wall
    "wall": [(i * 2.0 - 4.0, 0.0) for i in range(5)],
}

SCHEMA_AR = {"hut": "كوخ", "wall": "جدار"}


class BuildProject:
    """One ordered build: a fixed slot sequence filled placement by placement."""

    def __init__(self, schema, cx, cy, tick, day):
        self.schema = schema
        self.cx = float(cx)
        self.cy = float(cy)
        self.slots = [(cx + dx, cy + dy) for dx, dy in SCHEMAS[schema]]
        self.idx = 0
        self.start_tick = tick
        self.start_day = day
        self.live = True

    def next_slot(self):
        if not self.live or self.idx >= len(self.slots):
            return None
        return self.slots[self.idx]

    def record_placement(self):
        """Advance one slot. Returns 'half', 'done', or None."""
        if not self.live:
            return None
        self.idx += 1
        n = len(self.slots)
        if self.idx >= n:
            self.live = False
            return "done"
        if self.idx == n // 2:
            return "half"
        return None

    def to_dict(self):
        return {"schema": self.schema, "cx": self.cx, "cy": self.cy,
                "slots": [list(s) for s in self.slots], "idx": self.idx,
                "start_tick": self.start_tick, "start_day": self.start_day,
                "live": self.live}

    @classmethod
    def from_dict(cls, d):
        p = cls(d["schema"], d["cx"], d["cy"], d["start_tick"], d["start_day"])
        p.slots = [tuple(s) for s in d["slots"]]
        p.idx = d["idx"]
        p.live = d["live"]
        return p
