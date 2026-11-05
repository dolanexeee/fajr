# -*- coding: utf-8 -*-
"""PairBond — ADAM & EVE P3: the bond between the two creatures.

HONESTY NOTE (read before touching):
- The bond is a property of the PAIR, not of either individual. It is
  mechanistic bookkeeping of LIVED closeness: proximity ticks, mutual
  union choices, shared acts. Nothing here claims love as a human
  feels it.
- Milestone names («زوج الوادي») are OUR reading of a measured state,
  journaled behaviorally — same honesty line as the hut reading.
- The bond FEEDS the family drive (binding design, his correction):
  bond strength raises the family-drive growth rate in drives.py.
  Physics from us; the decision to seek union is theirs.
- P4 reads union_ticks / conceive_ticks for the conception logic.
  This module never spawns anything.

RNG: none used here (all increments are fixed rates).
"""

UNION_R = 8.0          # mutual closeness radius
OBSERVE_R = 22.0       # imitation observation radius
BOND_UP_MUTUAL = 0.0012   # per mutual-union tick
BOND_UP_CLOSE = 0.0004    # per merely-close tick
BOND_DECAY = 0.00006      # per apart tick (slow)
# family-drive relief: mild for mere closeness, strong for MUTUAL union
# choice (the choice is the center of the design — relief must not wipe
# the drive before it can ever reach the union threshold)
FAMILY_RELIEF_CLOSE = 0.05
FAMILY_RELIEF_MUTUAL = 0.8

# P4: conception follows THEIR sustained mutual choice — never an external
# trigger. The counter only grows on mutual ticks (see sim._pair_tick).
# 300 ≈ 4-8 meetings of chosen togetherness: clearly "repeated", reachable.
CONCEIVE_NEED = 300          # mutual ticks of their choosing
GESTATION_TICKS = 15 * 1440  # fixed world physics (ours) ~ 3 real hours

BOND_MILESTONES = (
    (0.30, "close", "صارا قريبين"),
    (0.60, "inseparable", "لا يفترقان"),
    (0.85, "pair", "زوج الوادي"),
)


class PairBond:
    def __init__(self):
        self.value = 0.0
        self.close_ticks = 0
        self.union_ticks = 0       # both chose union, together
        self.conceive_ticks = 0    # P4: accumulates toward conception
        self.milestones = []       # journaled milestone keys
        self.last_imitate_tick = -10 ** 9
        # P4: pregnancy is pair-state (physics ours, choice theirs)
        self.pregnant = False
        self.gestation_left = 0
        self.last_feed_tick = -10 ** 9

    def to_dict(self):
        return {"value": self.value, "close_ticks": self.close_ticks,
                "union_ticks": self.union_ticks,
                "conceive_ticks": self.conceive_ticks,
                "milestones": list(self.milestones),
                "last_imitate_tick": self.last_imitate_tick,
                "pregnant": self.pregnant,
                "gestation_left": self.gestation_left,
                "last_feed_tick": self.last_feed_tick}

    @classmethod
    def from_dict(cls, d):
        b = cls.__new__(cls)
        b.value = float(d.get("value", 0.0))
        b.close_ticks = int(d.get("close_ticks", 0))
        b.union_ticks = int(d.get("union_ticks", 0))
        b.conceive_ticks = int(d.get("conceive_ticks", 0))
        b.milestones = list(d.get("milestones", []))
        b.last_imitate_tick = int(d.get("last_imitate_tick", -10 ** 9))
        b.pregnant = bool(d.get("pregnant", False))
        b.gestation_left = int(d.get("gestation_left", 0))
        b.last_feed_tick = int(d.get("last_feed_tick", -10 ** 9))
        return b
