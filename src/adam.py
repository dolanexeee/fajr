# -*- coding: utf-8 -*-
"""Adam — ADAM & EVE P2: the second creature, zero knowledge.

HONESTY NOTE (read before touching):
- Adam is a SIMPLIFIED learner: drives + drive-arbitrated intents, NO
  episodic memory, NO dreams, NO aspirations, NO workspace. He is
  deliberately less than Fajr — a blank slate that must learn everything
  by living (P3) or not at all.
- He spawns BLANK: skills 0.0, no food memory, no place values. Every
  skill rises strictly by lived practice (same asymptotic pattern as
  her swim skill: 1 - exp(-practice/TAU)) — never granted.
- The Guide NEVER teaches Adam directly (his order). Fajr teaches him
  in P3 via imitation; this module only gives him the body and the
  drive loop to be taught INTO.
- CPU: decides every 3 ticks; movement/physiology every tick. Tick-cheap.
- RNG: the shared SplitMix64 PRNG only. No stdlib random.

Name: آدم (Adam) — the male of the valley, per his words.
"""
import math

from config import (WALK_SPEED, EAT_TICKS_PER_FRUIT, FRUIT_SATIATION,
                    DRINK_RELIEF, SHELTER_POS, SOCIAL_RADIUS)
from drives import Drives
from curiosity import Curiosity, CURIOSITY_TICKS
from world import is_passable, dist_to_river, is_deep_water

# valley-edge spawn: west side, south of the river bend
SPAWN_POS = (18.0, 150.0)

DECIDE_EVERY = 3          # ticks between decisions (CPU-light)
ARRIVE_R = 0.8            # arrival radius for movement targets
EAT_R = 3.5               # must be this close to a tree to eat
DRINK_R = 6.5             # dist_to_river below this counts as "at water"
                        # (same physics as Fajr's drink arrival, fajr.py)
FOLLOW_R = 4.0            # close enough to Fajr: stop following
SKILL_TAU = 400.0         # practice ticks to ~63% skill (same shape as swim)

SKILLS = ("forage", "swim", "build")


class Adam:
    def __init__(self, rng, x=SPAWN_POS[0], y=SPAWN_POS[1]):
        self.x = float(x)
        self.y = float(y)
        self.drives = Drives()
        # zero knowledge: blank slate
        self.practice = {s: 0 for s in SKILLS}  # lived practice counts only
        self.food_memory = {}                   # tree_id -> [x, y] (empty)
        self.asleep = False
        self.intent = None                      # (kind, tx, ty)
        # P4-fix: remembers he chose to seek her — arrival fulfils the
        # choice, it doesn't erase it (else _pair_tick never sees mutual
        # union once they're close). Set in _decide, never serialised.
        self.seeking_her = False
        # P6: seeds (from eaten fruit), wet hands (drink clock), curiosity
        self.seeds = 0
        self.last_drink_tick = -10 ** 9
        self.curiosity = Curiosity()
        self.kind_name = "adam"   # Child overrides to "child"
        self._last_harvest = None  # transient (tick, planter) for SIM
        self.acting = None                      # (kind, ticks_left, target)
        self.stats = {"eats": 0, "drinks": 0, "sleeps": 0,
                      "distance": 0.0}
        self.age_ticks = 0

    def _speed_factor(self):
        return 1.0

    # ---------------- skills: lived practice only ----------------
    def skill(self, name):
        return 1.0 - math.exp(-self.practice[name] / SKILL_TAU)

    def practice_skill(self, name, n=1):
        self.practice[name] += n

    # ---------------- movement ----------------
    def _step_toward(self, tx, ty, rng, arrive_r=ARRIVE_R):
        dx, dy = tx - self.x, ty - self.y
        dist = math.hypot(dx, dy)
        if dist < arrive_r:
            return True
        step = min(WALK_SPEED * self._speed_factor(), dist)
        nx = self.x + dx / dist * step
        ny = self.y + dy / dist * step
        if is_passable(nx, ny):
            self.x, self.y = nx, ny
        else:
            ang = rng.uniform() * 2.0 * math.pi
            nx2 = self.x + math.cos(ang) * step
            ny2 = self.y + math.sin(ang) * step
            if is_passable(nx2, ny2):
                self.x, self.y = nx2, ny2
        self.stats["distance"] += step
        if is_deep_water(self.x, self.y):
            self.practice_skill("swim")
        return False

    # ---------------- decision (every 3 ticks) ----------------
    def _decide(self, world, fajr, rng, tick):
        # P6: novelty first — untried verb×target combos at slow cadence.
        # The engine doesn't know "farming"; it knows novelty.
        if tick % CURIOSITY_TICKS == 0:
            prop = self.curiosity.propose(self, world, rng, tick)
            if prop is not None:
                self.intent = prop
                self.seeking_her = False
                return
        d = self.drives
        urg = {k: d.urgency(k) for k in
               ("hunger", "thirst", "sleep", "safety", "social", "curiosity",
                # ADAM & EVE P3: the family drive — his union-seeking is
                # scored like every other need. Decision his.
                "family")}
        top = max(urg, key=urg.get)
        if top == "hunger":
            tgt = self._nearest_fruit_tree(world)
            if tgt is not None:
                self.intent = ("seek_food", tgt[0], tgt[1])
            else:
                self.intent = ("wander", *self._wander(rng))
        elif top == "thirst":
            self.intent = ("drink", *self._river_target(rng))
        elif top == "sleep":
            self.intent = ("sleep", SHELTER_POS[0] + 4.0, SHELTER_POS[1] - 2.0)
        elif top == "safety":
            self.intent = ("sleep", SHELTER_POS[0] + 4.0, SHELTER_POS[1] - 2.0)
        elif top == "social":
            # the seed of P3: he is drawn to her, knows nothing else
            self.intent = ("follow", fajr.x, fajr.y)
        elif top == "family":
            # union: he seeks HER specifically (not the lantern, not anyone)
            self.intent = ("union", fajr.x, fajr.y)
        elif top == "curiosity":
            # P6: novelty — try an untried verb×target combo, if any exists
            prop = self.curiosity.propose(self, world, rng, tick)
            if prop is not None:
                self.intent = prop
            else:
                self.intent = ("wander", *self._wander(rng))
        else:
            self.intent = ("wander", *self._wander(rng))
        # P6: farming habits — AFTER the discovery is tribal knowledge,
        # the verbs become drive-scored habits (repeated because they work).
        # Never overrides union/follow: the pair bond is not tradable.
        if "farming" in self.curiosity.known and self.intent is not None \
                and self.intent[0] in ("wander", "seek_food"):
            self._farming_habit(world, tick)
        # P4-fix: the choice persists past arrival (see seeking_her)
        self.seeking_her = self.intent is not None and self.intent[0] in ("union", "follow")

    def _farming_habit(self, world, tick):
        """P6: once farming is KNOWN (discovered, not scripted), its verbs
        compete as ordinary drive-scored habits. Still his decision."""
        import math
        d = self.drives.d
        # harvest ripe crops when hungry (like seek_food)
        if d["hunger"] > 50.0:
            best = None
            for c in world.crops:
                if c.stage == "mature" and c.fruit > 0:
                    dd = math.hypot(c.x - self.x, c.y - self.y)
                    if best is None or dd < best[0]:
                        best = (dd, c)
            if best is not None:
                self.intent = ("harvest", best[1].x, best[1].y)
                return
        # water thirsty crops when hands are wet
        if tick - self.last_drink_tick < 400:
            best = None
            for c in world.crops:
                if c.stage != "mature" and c.water < 50.0:
                    dd = math.hypot(c.x - self.x, c.y - self.y)
                    if best is None or dd < best[0]:
                        best = (dd, c)
            if best is not None and best[0] < 30.0:
                self.intent = ("water", best[1].x, best[1].y)
                return
        # plant seeds when hungry and no ripe crop around
        if d["hunger"] > 50.0 and self.seeds > 0:
            ripe = any(c.stage == "mature" and c.fruit > 0 for c in world.crops)
            if not ripe:
                from world import is_passable
                if is_passable(self.x, self.y):
                    self.intent = ("plant", self.x, self.y)

    def _nearest_fruit_tree(self, world):
        best = None
        for t in world.trees:
            if t.fruit > 0:
                dist = math.hypot(t.x - self.x, t.y - self.y)
                if best is None or dist < best[0]:
                    best = (dist, t.x, t.y)
        return None if best is None else (best[1], best[2])

    def _river_target(self, rng):
        # already AT water: stay — the opportunistic drink handles it.
        # (2026-11-04 fix: the old DRINK_R*3.0 "stay" zone trapped him at
        # 6.5–19.5u — told to stay but too far to drink, forever.)
        if dist_to_river(self.x, self.y) < DRINK_R:
            return (self.x, self.y)
        best = None
        for _ in range(8):
            ang = rng.uniform() * 2.0 * math.pi
            r = 6.0 + rng.uniform() * 10.0
            cx, cy = self.x + math.cos(ang) * r, self.y + math.sin(ang) * r
            if is_passable(cx, cy):
                dd = dist_to_river(cx, cy)
                if best is None or dd < best[0]:
                    best = (dd, cx, cy)
        return (best[1], best[2]) if best else (self.x, self.y)

    def _wander(self, rng):
        for _ in range(8):
            ang = rng.uniform() * 2.0 * math.pi
            r = 8.0 + rng.uniform() * 22.0
            cx, cy = self.x + math.cos(ang) * r, self.y + math.sin(ang) * r
            if is_passable(cx, cy):
                return (cx, cy)
        return (self.x, self.y)

    # ---------------- one tick ----------------
    def step(self, world, fajr, tick, light, rng, journal,
             bond=0.0, mature=1.0):
        self.age_ticks += 1
        # acting (eating) runs to completion
        if self.acting is not None:
            kind, left, target = self.acting
            if kind == "eat":
                left -= 1
                if left <= 0:
                    self.acting = None
                    tree = world.trees[target] if 0 <= target < len(world.trees) else None
                    if tree is not None and tree.fruit > 0:
                        tree.fruit -= 1
                        self.drives.d["hunger"] = max(
                            0.0, self.drives.d["hunger"] - FRUIT_SATIATION)
                        self.stats["eats"] += 1
                        self.practice_skill("forage")
                        self.food_memory[target] = [tree.x, tree.y]
                        # P6: every fruit carries a seed — the physics of
                        # future crops, whether or not he ever plants it
                        self.seeds = min(6, self.seeds + 1)
                else:
                    self.acting = (kind, left, target)
            self._physiology(light, world, bond, mature)
            return
        # sleep
        if self.asleep:
            self.seeking_her = False    # sleep suspends the seeking
            if self.drives.d["sleep"] < 25.0:
                self.asleep = False
                self.stats["sleeps"] += 1
            self._physiology(light, world, bond, mature)
            return
        # opportunistic drinking: water at hand is drunk — not only on
        # arrival at a sampled target (bug: he once stood 0.19 short of
        # drinkable water, thirst-pinned, walking to a worse target)
        if dist_to_river(self.x, self.y) < DRINK_R \
                and self.drives.d["thirst"] > 25.0:
            self.drives.d["thirst"] = max(
                0.0, self.drives.d["thirst"] - DRINK_RELIEF)
            self.stats["drinks"] += 1
            self.last_drink_tick = tick   # P6: hands are wet — can water
            self.intent = None
        # decide every few ticks or when idle —
        # P6: curiosity commitments (plant/water/harvest) aren't dropped
        # mid-walk; the discovery needs the act to complete
        committed = self.intent is not None and self.intent[0] in ("plant", "water", "harvest")
        if not committed and (self.intent is None or tick % DECIDE_EVERY == 0):
            self._decide(world, fajr, rng, tick)
        kind, tx, ty = self.intent
        if kind in ("follow", "union"):
            # re-aim at her live position each decision
            tx, ty = fajr.x, fajr.y
            self.intent = (kind, tx, ty)
        arrive_r = FOLLOW_R if kind in ("follow", "union") else ARRIVE_R
        arrived = self._step_toward(tx, ty, rng, arrive_r)
        if arrived:
            if kind == "seek_food":
                for i, t in enumerate(world.trees):
                    if t.fruit > 0 and math.hypot(t.x - self.x, t.y - self.y) < EAT_R:
                        self.acting = ("eat", EAT_TICKS_PER_FRUIT, i)
                        break
                else:
                    self.intent = None
            elif kind == "drink":
                if dist_to_river(self.x, self.y) < DRINK_R:
                    self.drives.d["thirst"] = max(
                        0.0, self.drives.d["thirst"] - DRINK_RELIEF)
                    self.stats["drinks"] += 1
                    self.last_drink_tick = tick   # P6: hands are wet
                self.intent = None
            elif kind == "sleep":
                self.asleep = True
                self.intent = None
            elif kind == "plant":
                # P6: bury a seed — the world does the rest (physics)
                from world import is_passable, Crop
                if self.seeds > 0 and is_passable(self.x, self.y):
                    world.crops.append(Crop(len(world.crops), self.x, self.y,
                                            planter=self.kind_name))
                    self.seeds -= 1
                    self.curiosity.mark_tried("plant", "seed")
                self.intent = None
            elif kind == "water":
                # P6: wet hands on a growing crop — physics does the rest
                for c in world.crops:
                    if math.hypot(c.x - self.x, c.y - self.y) < 5.0 \
                            and c.stage != "mature":
                        c.water = min(100.0, c.water + 50.0)
                        self.curiosity.mark_tried("water", "crop")
                        break
                self.intent = None
            elif kind == "harvest":
                # P6: take ripe fruit from a crop he (or kin) grew
                for c in world.crops:
                    if math.hypot(c.x - self.x, c.y - self.y) < EAT_R \
                            and c.stage == "mature" and c.fruit > 0:
                        n = c.fruit
                        c.fruit = 0
                        c.regrow = 0
                        self.drives.d["hunger"] = max(
                            0.0, self.drives.d["hunger"] - FRUIT_SATIATION * n)
                        self.stats["eats"] += n
                        self.seeds = min(6, self.seeds + n)  # fruit->seeds
                        self.curiosity.mark_tried("harvest", "crop")
                        # the SIM watches this for the discovery (P6)
                        self._last_harvest = (tick, c.planter)
                        break
                self.intent = None
            else:
                self.intent = None
        self._physiology(light, world, bond, mature)

    def _physiology(self, light, world, bond=0.0, mature=1.0):
        from config import LANTERN_POS
        ctx = {
            "asleep": self.asleep, "light": light,
            "in_shelter": math.hypot(self.x - SHELTER_POS[0],
                                     self.y - SHELTER_POS[1]) < 12.0,
            "in_rain_exposed": world.weather in ("rain", "storm") and
            math.hypot(self.x - SHELTER_POS[0], self.y - SHELTER_POS[1]) >= 12.0,
            "near_lantern": math.hypot(self.x - LANTERN_POS[0],
                                       self.y - LANTERN_POS[1]) < SOCIAL_RADIUS,
            "near_hazard": False,
            "cold_water": is_deep_water(self.x, self.y),
            # ADAM & EVE P3: bond feeds the family drive (binding design)
            "bond": bond, "mature": mature,
        }
        self.drives.update(ctx)

    # ---------------- persistence ----------------
    def to_dict(self):
        return {"x": self.x, "y": self.y,
                "drives": self.drives.to_dict(),
                "practice": dict(self.practice),
                "food_memory": {str(k): v for k, v in self.food_memory.items()},
                "asleep": self.asleep,
                "intent": list(self.intent) if self.intent else None,
                "acting": list(self.acting) if self.acting else None,
                "stats": dict(self.stats),
                "age_ticks": self.age_ticks,
                "seeds": self.seeds,
                "last_drink_tick": self.last_drink_tick,
                "curiosity": self.curiosity.to_dict(),
                "kind_name": self.kind_name}

    @classmethod
    def from_dict(cls, d, rng):
        a = cls.__new__(cls)
        a.x = d["x"]
        a.y = d["y"]
        a.drives = Drives.from_dict(d["drives"])
        a.practice = {s: int(d["practice"].get(s, 0)) for s in SKILLS}
        a.food_memory = {int(k): v for k, v in d.get("food_memory", {}).items()}
        a.asleep = d["asleep"]
        a.intent = tuple(d["intent"]) if d["intent"] else None
        a.acting = tuple(d["acting"]) if d["acting"] else None
        a.stats = dict(d["stats"])
        a.age_ticks = d.get("age_ticks", 0)
        # seeking_her is transient (never serialised); recomputed on decide
        a.seeking_her = False
        # P6: seeds, drink clock, curiosity persist; harvest flag transient
        a.seeds = int(d.get("seeds", 0))
        a.last_drink_tick = int(d.get("last_drink_tick", -10 ** 9))
        a.curiosity = Curiosity.from_dict(d.get("curiosity", {}))
        a.kind_name = d.get("kind_name", "adam")
        a._last_harvest = None
        return a
