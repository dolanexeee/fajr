# -*- coding: utf-8 -*-
"""Child — ADAM & EVE P4: the offspring, born in-world, growing in stages.

HONESTY NOTE (read before touching):
- The child is born from THEIR choice, not our trigger: conception only
  follows sustained MUTUAL union choice (bond.conceive_ticks, accrued
  tick-by-tick when BOTH chose union). There is no external
  "bond > 0.8 -> spawn" anywhere — that was explicitly refused.
- Gestation length and growth STAGES are world physics (ours): infant
  (0-30d) -> child (30-120d) -> adult (120d+). Staged parameter sets,
  NEVER continuous growth.
- Infant: cannot gather (no seek_food intent), moves slowly, needs
  feeding — an adult's completed eat nearby shares food (physics).
- Child: full intents, learns at 2x practice rate (shoulders of parents).
- Adult: a full simplified adult, like Adam.
- Sex is mechanistic (PRNG coin flip), journaled neutrally.
- Like Adam: simplified mind, no episodic/dreams/aspirations. The Guide
  never teaches the child either.

RNG: the shared SplitMix64 PRNG only.
"""
import math

from adam import Adam, SKILL_TAU
from config import WALK_SPEED
from world import is_passable

INFANT_DAYS = 30
CHILD_DAYS = 120
FEED_R = 10.0            # adult's shared meal reaches the infant within this
INFANT_SPEED = 0.35      # infant movement factor
CHILD_LEARN_MULT = 2     # children learn parent-known skills faster


class Child(Adam):
    def __init__(self, rng, x, y, sex="female"):
        super().__init__(rng, x, y)
        self.sex = sex
        self._stage = "infant"
        self.other_parent = None   # transient, set by Sim each tick
        self.kind_name = "child"   # P6: crops record the true planter

    # ---------------- growth stages (physics) ----------------
    def stage(self):
        days = self.age_ticks / 1440.0
        if days < INFANT_DAYS:
            return "infant"
        if days < CHILD_DAYS:
            return "child"
        return "adult"

    def maturity(self):
        return {"infant": 0.0, "child": 0.5, "adult": 1.0}[self.stage()]

    def _speed_factor(self):
        return INFANT_SPEED if self.stage() == "infant" else 1.0

    def practice_skill(self, name, n=1):
        mult = CHILD_LEARN_MULT if self.stage() == "child" else 1
        super().practice_skill(name, n * mult)

    # ---------------- decisions by stage ----------------
    def _nearest_adult(self, fajr):
        cands = [("fajr", fajr.x, fajr.y)]
        if self.other_parent is not None:
            cands.append(("adam", self.other_parent.x, self.other_parent.y))
        cands.sort(key=lambda c: math.hypot(c[1] - self.x, c[2] - self.y))
        return cands[0]

    def _decide(self, world, fajr, rng, tick):
        st = self.stage()
        if st == "infant":
            # cannot gather; stays near adults, rests, drinks
            d = self.drives
            urg = {k: d.urgency(k) for k in ("sleep", "thirst", "social", "curiosity")}
            top = max(urg, key=urg.get)
            if top == "sleep":
                from config import SHELTER_POS
                self.intent = ("sleep", SHELTER_POS[0] + 4.0, SHELTER_POS[1] - 2.0)
            elif top == "thirst":
                self.intent = ("drink", *self._river_target(rng))
            else:
                _, ax, ay = self._nearest_adult(fajr)
                self.intent = ("follow", ax, ay)
            return
        # child and adult: the full simplified repertoire (hers to learn into)
        super()._decide(world, fajr, rng, tick)

    # ---------------- persistence ----------------
    def to_dict(self):
        d = super().to_dict()
        d["sex"] = self.sex
        d["stage"] = self._stage
        return d

    @classmethod
    def from_dict(cls, d, rng):
        c = super().from_dict(d, rng)
        c.sex = d.get("sex", "female")
        c._stage = d.get("stage", "infant")
        c.other_parent = None
        return c
