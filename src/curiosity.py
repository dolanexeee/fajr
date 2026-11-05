# -*- coding: utf-8 -*-
"""Curiosity — ADAM & EVE P6: the open-evolution engine's intrinsic drive.

HONESTY NOTE (read before touching):
- This engine does NOT know "farming" or any discovery. It knows VERBS
  (plant, water, harvest) and TARGETS (seed, crop) and tries untried
  (verb x target) combinations out of novelty. WHAT the combinations DO
  is world physics (world.py); WHETHER they matter is outcome (drive
  reduction). The discovery is theirs; the physics is ours.
- The combination space is bounded by the verbs/objects WE define.
  True unbounded invention needs generative cognition — red zone, out.
  What we guarantee: the engine never stops trying, and knowledge never
  regresses (cultural ratchet). WHAT they discover is open.
- Slow cadence by design: proposals fire at most every CURIOSITY_TICKS
  or when curiosity tops the drives — never every tick (CPU).

RNG: the shared SplitMix64 PRNG only.
"""
# verbs the curiosity engine may try (body must implement each)
VERBS = ("plant", "water", "harvest")

CURIOSITY_TICKS = 720    # forced novelty check cadence (~half a valley day)


class Curiosity:
    """Per-agent novelty tracker. Lives on the simplified minds."""

    def __init__(self):
        self.tried = set()      # "verb:target" already attempted
        self.known = set()      # discovery keys this agent holds

    def mark_tried(self, verb, target):
        self.tried.add(f"{verb}:{target}")

    def has_tried(self, verb, target):
        return f"{verb}:{target}" in self.tried

    def propose(self, agent, world, rng, tick):
        """An untried (verb, target) available RIGHT NOW, or None.
        Never proposes what physics forbids (checked by the caller verbs)."""
        cands = []
        # plant: carrying a seed, standing on soil
        if getattr(agent, "seeds", 0) > 0 and not self.has_tried("plant", "seed"):
            cands.append(("plant", agent.x, agent.y))
        # water: a growing crop nearby, hands wet (drank recently)
        if not self.has_tried("water", "crop"):
            for c in world.crops:
                if c.stage != "mature" and _dist(agent, c) < 12.0:
                    if tick - agent.last_drink_tick < 400:
                        cands.append(("water", c.x, c.y))
                        break
        # harvest: a mature crop nearby
        if not self.has_tried("harvest", "crop"):
            for c in world.crops:
                if c.stage == "mature" and c.fruit > 0 and _dist(agent, c) < 12.0:
                    cands.append(("harvest", c.x, c.y))
                    break
        if not cands:
            return None
        verb, x, y = cands[rng.randint(0, len(cands) - 1)]
        return (verb, x, y)

    def to_dict(self):
        return {"tried": sorted(self.tried), "known": sorted(self.known)}

    @classmethod
    def from_dict(cls, d):
        c = cls()
        c.tried = set(d.get("tried", []))
        c.known = set(d.get("known", []))
        return c


def _dist(agent, crop):
    import math
    return math.hypot(agent.x - crop.x, agent.y - crop.y)
