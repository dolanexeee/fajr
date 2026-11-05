"""Fajr's body — Phase 1 (SPEC §4, §5), with the Phase-2 mind riding on it.

What lives here: drives-driven action selection (homeostatic arbitration +
learned values), associative learning (food memory, place values, one-trial
fear), and action execution. Phase 2: a Mind (mind.py) is attached; when it
steers, intentions come from its workspace cycle and the food memory that
planning reads is belief-written, not raw. The reflexes (flee, prick,
sleep collapse) stay in the body either way. No LLM is called anywhere;
this is mechanism.
"""
import math

from config import (GRID_W, GRID_H, CELL, WALK_SPEED, WATER_SLOW_FACTOR,
                    SWIM_SLOW_FACTOR, BATHE_TICKS,
                    SWIM_SKILL_TAU, SWIM_SLOW_SKILLED, SWIM_COST_RELIEF,
                    EAT_TICKS_PER_FRUIT, FRUIT_SATIATION, DRINK_RELIEF,
                    SHELTER_POS, LANTERN_POS, BUILD_SITE, SOCIAL_RADIUS)
from drives import Drives
from mind import Mind
from world import (dist_to_river, is_water, is_deep_water, is_passable,
                   depth_zone, bath_spot, in_forest)

SLEEP_PRESSURE_GO = 78.0
SLEEP_PRESSURE_ANYWHERE = 96.0


def pile_by_id(world, pid):
    """Piles are looked up BY ID, never by list position: ids are
    arbitrary (harness-placed piles carry large ids) and piles are
    removed when eaten — an index lookup crashes or, worse, silently
    eats the wrong pile. (Bug caught by the Phase-2 P-C probe.)"""
    for p in world.piles:
        if p.id == pid:
            return p
    return None


def cell_of(x, y):
    cx = min(GRID_W - 1, max(0, int(x // CELL)))
    cy = min(GRID_H - 1, max(0, int(y // CELL)))
    return cx, cy


class Fajr:
    def __init__(self, rng):
        self.x, self.y = SHELTER_POS[0] - 6.0, SHELTER_POS[1] + 3.0  # awakens near the shelter (SPEC §6)
        self.drives = Drives()
        self.asleep = False
        self.carry = []                        # material kinds, max 3
        self.food_memory = {}                  # tree_id -> [x, y, fruit_seen, tick_seen]
        self.fear = {}                         # "thicket:<id>" | "deep_water" -> 0..1
        self.place_value = [[0.0] * GRID_W for _ in range(GRID_H)]
        self.novelty = [[1.0] * GRID_W for _ in range(GRID_H)]
        self.visits = [[0] * GRID_W for _ in range(GRID_H)]
        self.intent = None                     # (kind, tx, ty, payload)
        self.intent_age = 0
        self.host_directive = None              # Phase 3 (PHASE3-GATE §0): a Host
                                               # instruction SHE understood, as
                                               # (kind, tx, ty, payload, expiry).
                                               # Transient; never serialised.
        self.acting = None                     # (kind, ticks_left, target_id)
        self.in_deep = False
        # --- water & hygiene state (Stage 2, WATER-GATE) ---
        self.cleanliness = 100.0   # 100 = clean; grime = 100 - cleanliness.
                                   # A bodily state, not a ninth drive: it is
                                   # FELT through the comfort drive (below).
        self.wet = 0.0             # 0..1 surface wetness; soaks in water,
                                   # dries in air over ~140 ticks
        self.bathe_value = 0.25    # learned relief value of bathing: a weak
                                   # prior (water plausibly relieves the
                                   # dirty feeling); each completed bath's
                                   # felt relief moves it toward 1 (discovery)
        self._deep_stint = 0       # consecutive ticks in deep water
        self.last_cell = cell_of(self.x, self.y)
        self.stuck = 0
        # biography / measurement records
        self.stats = {"distance": 0.0, "eats": 0, "drinks": 0, "pricks": 0,
                      "cold_shocks": 0, "gathers": 0, "places": 0, "sleeps": 0,
                      "wasted_arrivals": 0,
                      "bathes": 0, "swims": 0, "swim_ticks": 0,
                      "unions": 0}
        # ADAM & EVE P5: lived-practice counters per skill. Level is a PURE
        # function of these (skill_level) — the only way up, no decay.
        self.practice = {"forage": 0, "swim": 0, "build": 0}
        # P6: seeds carried (from eaten fruit) — plantable
        self.seeds = 0
        self.sleep_episodes = []               # (start_tick, x, y)
        self.hunger_onset = None               # tick when hunger crossed 60
        self.hunger_latencies = []             # (day, latency_ticks)
        self.thicket_log = []                  # (tick, thicket_id, kind) kind: enter|prick — for G3b within life runs
        self._inside_thicket = set()
        self.journal = None                    # attached by Sim
        # Phase 2: the mind rides on the same single PRNG stream
        self.mind = Mind(rng)
        self.mind_events = {}                  # last tick's outcomes (set by Sim)
        self._raw = None                       # raw percept stashed for the mind

    # ---------------- learning ----------------
    def _learn_outcome(self, x, y, outcome, lr=0.25):
        cx, cy = cell_of(x, y)
        v = self.place_value[cy][cx]
        self.place_value[cy][cx] = max(-1.0, min(1.0, v + lr * (outcome - v)))

    def grime(self):
        """How dirty he is, 0 (clean) .. 100 (filthy) — the deficit side
        of the cleanliness state (Stage 2, WATER-GATE)."""
        return 100.0 - self.cleanliness

    def skill_level(self, name):
        """P5: skill level 0..1, pure function of lived practice —
        same curve as her swim 1.0 and his forage curve."""
        return 1.0 - math.exp(-max(0.0, self.practice.get(name, 0)) / 400.0)

    def swim_skill(self):
        """Swimming skill in [0, 1], DERIVED from lifetime swim_ticks
        (persisted in stats — no new state, no migration; her past swims
        count toward it). Asymptotic learning: 1 - exp(-ticks/TAU), so
        early swims teach the most and skill never exceeds 1."""
        return 1.0 - math.exp(-self.stats["swim_ticks"] / SWIM_SKILL_TAU)

    def swim_slow_factor(self):
        """Deep-water movement factor: SWIM_SLOW_FACTOR (0.30) at skill 0,
        rising linearly to SWIM_SLOW_SKILLED (0.80) at skill 1."""
        return SWIM_SLOW_FACTOR + self.swim_skill() * (SWIM_SLOW_SKILLED - SWIM_SLOW_FACTOR)

    def swim_cost_relief(self):
        """Fraction of swim energy cost remaining: 1.0 at skill 0 down to
        (1 - SWIM_COST_RELIEF) at skill 1. The W5 hunger:thirst asymmetry
        ratio is preserved — only magnitude scales."""
        return 1.0 - SWIM_COST_RELIEF * self.swim_skill()

    def fear_at(self, x, y, world):
        """Highest relevant learned fear for a location."""
        f = 0.0
        for th in world.thickets:
            if math.hypot(th.x - x, th.y - y) < th.r + 9.0:
                f = max(f, self.fear.get(f"thicket:{th.id}", 0.0))
        if is_deep_water(x, y):
            f = max(f, self.fear.get("deep_water", 0.0))
        return f

    # ---------------- perception intake ----------------
    def observe(self, percept, tick):
        # When the mind steers, raw transduction is NOT written straight
        # into food memory: the mind's belief layer does that, weighted by
        # the monitor. Otherwise (lesions / mind off) Phase-1 intake.
        if self.mind is not None and self.mind.active and self.mind.steers:
            self._raw = percept
            return
        for (kind, oid, dx, dy, dist, extra) in percept.visible:
            if kind == "tree":
                self.food_memory[oid] = [self.x + dx, self.y + dy, extra, tick]

    # ---------------- policy ----------------
    def _est_fruit(self, mem, tick):
        _, _, fruit_seen, tick_seen = mem
        return min(6.0, fruit_seen + (tick - tick_seen) / 3200.0)

    def _travel(self, tx, ty):
        return math.hypot(tx - self.x, ty - self.y) / WALK_SPEED

    def _score_target(self, base, tx, ty, world):
        cx, cy = cell_of(tx, ty)
        s = base * (1.0 + 0.18 * self.place_value[cy][cx])
        # route risk: fear is evaluated along the PATH, not only at the goal —
        # a feared place on the way taxes the whole intention (animals avoid
        # routes through feared ground, not just feared destinations).
        f = self.fear_at(tx, ty, world)
        dist = math.hypot(tx - self.x, ty - self.y)
        steps = max(1, int(dist // 4.0))
        for i in range(1, steps):
            px = self.x + (tx - self.x) * i / steps
            py = self.y + (ty - self.y) * i / steps
            f = max(f, self.fear_at(px, py, world))
        if f > 0.0:
            s *= (1.0 - 0.85 * f)
        return s

    def _best_explore_cell(self, world):
        """Most novel reachable cell (novelty x learned value, fear-discounted).
        Returns (quality_0_to_1, tx, ty) or None."""
        best = None
        for cy in range(GRID_H):
            for cx in range(GRID_W):
                nov = self.novelty[cy][cx]
                if nov < 0.35:
                    continue
                tx, ty = cx * CELL + CELL / 2, cy * CELL + CELL / 2
                if not is_passable(tx, ty):
                    continue
                if math.hypot(tx - self.x, ty - self.y) < 5.0:
                    # never target the cell he already stands in: its
                    # centre sits inside the body's arrival radius, the
                    # intent completes instantly, and — novelty decaying
                    # only on cell ENTRY — the same cell is re-issued
                    # forever (measured paralysis in sparse worlds,
                    # Phase-2 P-E regression diagnosis)
                    continue
                sc = nov * (1.0 + 0.3 * self.place_value[cy][cx]) / (1.0 + self._travel(tx, ty) / 90.0)
                sc *= (1.0 - 0.85 * self.fear_at(tx, ty, world))
                if best is None or sc > best[0]:
                    best = (sc, tx, ty)
        return best

    def choose_intent(self, world, percept, tick, light):
        d = self.drives.d
        cands = []  # (score, order, intent)
        order = 0

        def add(score, intent):
            nonlocal order
            cands.append((score, order, intent))
            order += 1

        # --- flee an active hazard contact zone: run to a point well clear
        # of the hazard (a fixed short hop deadlocks against the arrival radius) ---
        for th in world.thickets:
            dist = math.hypot(th.x - self.x, th.y - self.y)
            if dist < th.r + 1.0 and self.fear.get(f"thicket:{th.id}", 0.0) > 0.3:
                ux = (self.x - th.x) / max(dist, 0.01)
                uy = (self.y - th.y) / max(dist, 0.01)
                add(500.0, ("flee", th.x + ux * (th.r + 10.0), th.y + uy * (th.r + 10.0), th.id))

        # --- sleep ---
        # sleep intent: strong pressure any time, moderate pressure at night
        if d["sleep"] > 70.0 or (light < 0.25 and d["sleep"] > 45.0):
            sx, sy = SHELTER_POS
            sc = self.drives.urgency("sleep") * 3.2 / (1.0 + self._travel(sx, sy) / 60.0)
            add(self._score_target(sc, sx, sy, world), ("sleep", sx, sy, 0))

        # --- hunger ---
        food_known = False
        if d["hunger"] > 30.0:
            best = None
            for tid in sorted(self.food_memory.keys()):
                mem = self.food_memory[tid]
                est = self._est_fruit(mem, tick)
                if est < 0.6:
                    continue
                tx, ty = mem[0], mem[1]
                gain = min(d["hunger"], FRUIT_SATIATION * min(3.0, est))
                sc = self.drives.urgency("hunger") * gain / (self._travel(tx, ty) + 12.0)
                sc = self._score_target(sc, tx, ty, world)
                if best is None or sc > best[0]:
                    best = (sc, ("eat", tx, ty, tid))
            if best is not None:
                add(best[0], best[1])
                food_known = True
            else:
                # no known food: head for the meadow and look (smell helps up close)
                sc = self.drives.urgency("hunger") * 0.55 / (1.0 + self._travel(128.0, 95.0) / 60.0)
                add(sc, ("seek_food", 128.0, 95.0, -1))
                # desperation widens the search (a real foraging mechanism, not
                # a script): a starving animal that knows no food searches the
                # unknown instead of pacing at the empty known place.
                if d["hunger"] > 70.0:
                    cell = self._best_explore_cell(world)
                    if cell is not None:
                        add(self.drives.urgency("hunger") * cell[0] * 0.9,
                            ("search", cell[1], cell[2], -1))

        # --- thirst ---
        if d["thirst"] > 30.0:
            ry = max(10.0, min(170.0, self.y))
            from world import river_center_x
            tx = river_center_x(ry)
            sc = self.drives.urgency("thirst") * min(d["thirst"], DRINK_RELIEF) / (self._travel(tx, ry) + 8.0)
            add(self._score_target(sc, tx, ry, world), ("drink", tx, ry, -1))

        # --- hygiene: bathing (Stage 2, WATER-GATE) — a body-state
        # deficit with one specific remedy, priced as the drink remedy
        # is priced, carrying bathe_value: what his own completed baths
        # taught him bathing is worth (weak prior 0.25). The target is
        # the wading shallows on his side of the river (bath_spot). ---
        grime_now = self.grime()
        if grime_now >= 40.0:
            bx, by = bath_spot(self.x, self.y)
            sc = self.drives.urgency("comfort") * min(grime_now, 70.0) \
                * (0.2 + 0.8 * self.bathe_value) / (self._travel(bx, by) + 10.0)
            add(self._score_target(sc, bx, by, world), ("bathe", bx, by, -1))

        # --- safety / comfort: shelter restores both ---
        if d["safety"] > 58.0 or d["comfort"] > 55.0:
            sx, sy = SHELTER_POS
            sc = (self.drives.urgency("safety") + self.drives.urgency("comfort")) * 1.6 / (1.0 + self._travel(sx, sy) / 60.0)
            add(self._score_target(sc, sx, sy, world), ("shelter", sx, sy, 0))

        # --- social (lantern presence — Guide-stub) ---
        if d["social"] > 52.0:
            lx, ly = LANTERN_POS
            sc = self.drives.urgency("social") * 2.4 / (1.0 + self._travel(lx, ly) / 60.0)
            add(sc, ("social", lx, ly, 0))

        # --- creation ---
        if d["creation"] > 48.0:
            if len(self.carry) >= 1:
                bx, by = BUILD_SITE
                sc = self.drives.urgency("creation") * 2.2 / (1.0 + self._travel(bx, by) / 60.0)
                add(sc, ("place", bx, by, -1))
            else:
                # nearest visible or any known material
                tgt = None
                for (kind, oid, dx, dy, dist, extra) in percept.visible:
                    if kind == "material" and (tgt is None or dist < tgt[0]):
                        tgt = (dist, self.x + dx, self.y + dy, oid)
                if tgt is not None:
                    sc = self.drives.urgency("creation") * 1.8 / (1.0 + tgt[0] / WALK_SPEED / 60.0)
                    add(sc, ("gather", tgt[1], tgt[2], tgt[3]))
                else:
                    sc = self.drives.urgency("creation") * 0.5 / (1.0 + self._travel(205.0, 95.0) / 60.0)
                    add(sc, ("seek_material", 205.0, 95.0, -1))

        # --- curiosity: explore the most novel reachable cell ---
        if d["curiosity"] > 42.0:
            best = self._best_explore_cell(world)
            if best is not None:
                add(self.drives.urgency("curiosity") * best[0] * 1.15, ("explore", best[1], best[2], -1))

        if not cands:
            return ("wander", self.x, self.y, -1)
        cands.sort(key=lambda c: (-c[0], c[1]))
        return cands[0][2]

    # ---------------- per-tick behaviour ----------------
    def step(self, world, percept, tick, light, rng, journal):
        d = self.drives.d
        events = {"pricked": False, "cold_shock": False, "ate": False, "drank": False,
                  "slept_here": False, "near_hazard": False, "gathered": False, "placed": False,
                  "bathed": False, "swam": False}
        # --- sensory bookkeeping: thicket entries / novelty / visits ---
        cx, cy = cell_of(self.x, self.y)
        if (cx, cy) != self.last_cell:
            self.visits[cy][cx] += 1
            # experiencing novelty feeds the curiosity drive (SPEC §5)
            self.drives.d["curiosity"] -= self.novelty[cy][cx] * 7.0
            self.novelty[cy][cx] *= 0.25
            self.last_cell = (cx, cy)
        for th in world.thickets:
            inside = math.hypot(th.x - self.x, th.y - self.y) < th.r
            if inside and th.id not in self._inside_thicket:
                self._inside_thicket.add(th.id)
                self.thicket_log.append((tick, th.id, "enter"))
            elif not inside:
                self._inside_thicket.discard(th.id)
            if math.hypot(th.x - self.x, th.y - self.y) < th.r + 2.0:
                events["near_hazard"] = True
        if is_deep_water(self.x, self.y):
            events["near_hazard"] = True
            self._deep_stint += 1
            self.stats["swim_ticks"] += 1
            self.practice["swim"] += 1    # P5: lived practice
            if not self.in_deep:
                self.in_deep = True
                events["cold_shock"] = True
                self.stats["cold_shocks"] += 1
                d["comfort"] += 22.0
                d["safety"] += 18.0
                self.fear["deep_water"] = 1.0
                self._learn_outcome(self.x, self.y, -0.7)
                journal.log(tick, "cold_shock", "الماء العميق بارد جدًا — صدمة برد", x=self.x, y=self.y)
        else:
            # Stage 2: a deep-water stint of >= 8 ticks that ENDS is a
            # swim — he entered the channel, swam, and got out (the exit
            # is always possible: water is never a trap, SPEC §2.5).
            # Shorter dips are bank splashes, not swims.
            if self._deep_stint >= 8:
                events["swam"] = True
                self.stats["swims"] += 1
                journal.log(tick, "swim", "سبح في النهر وخرج", x=self.x, y=self.y)
            self._deep_stint = 0
            self.in_deep = False
        # --- thorn pricks (world physics, SPEC §2.5) ---
        for th in world.thickets:
            if math.hypot(th.x - self.x, th.y - self.y) < th.r and rng.chance(0.10):
                events["pricked"] = True
                self.stats["pricks"] += 1
                self.cleanliness = max(0.0, self.cleanliness - 1.5)  # thorn dirt (Stage 2)
                d["safety"] += 30.0
                d["comfort"] += 14.0
                self.fear[f"thicket:{th.id}"] = 1.0
                self._learn_outcome(th.x, th.y, -1.0)
                self.thicket_log.append((tick, th.id, "prick"))
                journal.log(tick, "prick", "وخزة شوك — ألم وتعلّم خوف", x=self.x, y=self.y)
                if self.asleep:
                    # a sting wakes a sleeping animal — no sleeping inside thorns
                    self.asleep = False
                    journal.log(tick, "wake", "أيقظته الوخزة", x=self.x, y=self.y)
                break

        # --- water & hygiene physiology (Stage 2, WATER-GATE) ---
        zone = depth_zone(self.x, self.y)
        # wetness: water soaks at once, rain wets slowly, air dries
        if zone != "dry":
            self.wet = min(1.0, self.wet + 0.25)
        elif world.weather in ("rain", "storm") and not percept.in_shelter:
            self.wet = min(1.0, self.wet + 0.02)
        else:
            self.wet = max(0.0, self.wet - 1.0 / 140.0)
        # dirt accrual: living soils him — slowly at rest, faster in the
        # forest and at the thickets' edge (work dirt is added at the
        # gather/place sites below). Dev-calibrated (WATER-GATE dev
        # pass 1): the first rates (~2 net/day after drink-trip rinsing)
        # could not carry grime to the felt range inside a week; these
        # give ~12-15 net/day for an active life.
        dirt = 0.002 if self.asleep else 0.016
        if in_forest(self.x, self.y):
            dirt += 0.030
        if any(math.hypot(th.x - self.x, th.y - self.y) < th.r + 1.0
               for th in world.thickets):
            dirt += 0.050
        self.cleanliness = max(0.0, self.cleanliness - dirt)
        # water rinses as it touches him: wading a little, swimming more.
        # The deliberate bath (the acting branch) is the thorough version.
        if zone == "shallow":
            self.cleanliness = min(100.0, self.cleanliness + 0.10)
        elif zone == "deep":
            self.cleanliness = min(100.0, self.cleanliness + 0.25)
        # grime is FELT, not merely stored: it loads the comfort drive —
        # his interoception reports a dirty body as discomfort (never as
        # danger: hygiene has no vital band and cannot pin a drive by
        # itself beyond what comfort's own dynamics allow). A wet body
        # in the cold dark is likewise uncomfortable, mildly.
        if zone == "dry":
            d["comfort"] += 0.55 * (self.grime() / 100.0) * (0.35 if self.asleep else 1.0)
            if self.wet > 0.5 and light < 0.2 and not percept.in_shelter:
                d["comfort"] += 0.12
        # swimming is work: the channel costs energy and water beyond
        # the base metabolic rates (the measured W5 cost asymmetry)
        if zone == "deep" and not self.asleep:
            # skill relieves the cost (never below half); the measured W5
            # hunger:thirst asymmetry ratio is preserved at every level.
            relief = self.swim_cost_relief()
            d["hunger"] += 0.055 * relief
            d["thirst"] += 0.022 * relief

        # --- the mind's cycle (Phase 2): perception -> beliefs -> workspace
        # -> decision. Runs before the sleep/act branches so consolidation
        # and recording happen on the first asleep tick too. ---
        if self.mind is not None and self.mind.active:
            self.mind.cycle(self, world, percept, tick, light, rng, journal)

        if self.asleep:
            events["slept_here"] = True
            quality = 1.0 if percept.in_shelter else 0.55
            if world.weather in ("rain", "storm") and not percept.in_shelter:
                quality = 0.4
            # sleeping somewhere good teaches its value (place memory, G3c)
            self._learn_outcome(self.x, self.y, 0.35 * quality, lr=0.01)
            if d["sleep"] <= 12.0:
                self.asleep = False
                journal.log(tick, "wake", "استيقظ", x=self.x, y=self.y)
            return events

        # --- acting (multi-tick actions) ---
        if self.acting is not None:
            kind, left, target_id = self.acting
            left -= 1
            if kind == "eat":
                if left <= 0:
                    if target_id <= -1000:
                        # eating from a fruit pile (Phase-2 memory target)
                        pile = pile_by_id(world, -1000 - target_id)
                        if pile is not None and pile.fruit > 0:
                            # the outcome settles his monitor's claims
                            if self.mind is not None:
                                self.mind.monitor.verify(f"pile:{pile.id}", pile.fruit, tick)
                            pile.fruit -= 1
                            d["hunger"] -= FRUIT_SATIATION
                            self.stats["eats"] += 1
                            events["ate"] = True
                            # P6: every fruit carries a seed
                            self.seeds = min(6, getattr(self, "seeds", 0) + 1)
                            self._learn_outcome(pile.x, pile.y, 1.0)
                            if self.hunger_onset is not None:
                                self.hunger_latencies.append((self.hunger_onset[0], tick - self.hunger_onset[1]))
                                self.hunger_onset = None
                            journal.log(tick, "eat", "أكل من كومة فاكهة", x=self.x, y=self.y)
                        self.acting = None
                        self.intent = None
                        return events
                    tree = world.trees[target_id]
                    if tree.fruit > 0:
                        # the outcome settles his monitor's claims
                        if self.mind is not None:
                            self.mind.monitor.verify(f"tree:{target_id}", tree.fruit, tick)
                        tree.fruit -= 1
                        d["hunger"] -= FRUIT_SATIATION
                        self.stats["eats"] += 1
                        events["ate"] = True
                        # P6: every fruit carries a seed
                        self.seeds = min(6, getattr(self, "seeds", 0) + 1)
                        self._learn_outcome(tree.x, tree.y, 1.0)
                        self.food_memory[target_id] = [tree.x, tree.y, tree.fruit, tick]
                        if self.hunger_onset is not None:
                            self.hunger_latencies.append((self.hunger_onset[0], tick - self.hunger_onset[1]))
                            self.hunger_onset = None
                        journal.log(tick, "eat", "أكل فاكهة", x=self.x, y=self.y)
                    self.acting = None
                    self.intent = None
                else:
                    self.acting = (kind, left, target_id)
                return events
            if kind == "bathe":
                # Bathing is done IN the water and takes time: each tick
                # scrubs a share of the grime away. The felt comfort
                # relief at completion is the reinforcer that teaches
                # bathe_value — nobody tells him bathing works; the
                # relief does (discovery through the comfort economy).
                self.cleanliness = min(100.0, self.cleanliness + 5.0)
                self.wet = 1.0
                if self.cleanliness >= 99.5 or left <= 0:
                    self.acting = None
                    self.intent = None
                    self.stats["bathes"] += 1
                    events["bathed"] = True
                    d["comfort"] = max(0.0, d["comfort"] - 18.0)
                    self.bathe_value += 0.30 * (1.0 - self.bathe_value)
                    self._learn_outcome(self.x, self.y, 0.85)
                    journal.log(tick, "bathe", "استحمّ في النهر", x=self.x, y=self.y)
                else:
                    self.acting = (kind, left, target_id)
                return events
            self.acting = None if left <= 0 else (kind, left, target_id)
            if left > 0:
                return events

        # --- choose / refresh intent ---
        self.intent_age += 1
        if self.intent is None or self.intent_age > 10:
            # Phase 3 instruction channel (PHASE3-GATE §0/§4): a fresh
            # host directive — set only by her own comprehension of a
            # Host instruction — outranks the cycle's choice, UNLESS a
            # vital drive is in emergency (>= 88): survival outranks
            # the Host's word (SPEC §2.3). Cleared on arrival or expiry.
            # With host_directive None (every pre-Phase-3 harness) this
            # branch never fires and behaviour is unchanged.
            directive = getattr(self, "host_directive", None)
            if directive is not None:
                dk, dtx, dty, dpayload, dexpiry = directive
                vital_emergency = max(d["hunger"], d["thirst"], d["sleep"]) >= 88.0
                arrived = math.hypot(dtx - self.x, dty - self.y) <= 3.0
                if tick > dexpiry or arrived or vital_emergency:
                    self.host_directive = None
                    directive = None
                else:
                    self.intent = (dk, dtx, dty, dpayload)
            if directive is None:
                if (self.mind is not None and self.mind.active and self.mind.steers
                        and self.mind.chosen_intent is not None):
                    self.intent = self.mind.chosen_intent
                else:
                    self.intent = self.choose_intent(world, percept, tick, light)
            self.intent_age = 0
        kind, tx, ty, payload = self.intent

        # sleep onset: intent is sleep and (arrived or too exhausted to continue)
        # onset radius == the movement arrival radius (3.0): the old 2.5 left a
        # dead zone (2.5 < dist <= 3.0) where he counted as "arrived" (stopped
        # walking) but could not fall asleep — seed 97's lost night in repair
        # round 2: parked at dist 2.87 from home, intent sleep, awake until
        # S hit the 96 anywhere-threshold ~280 ticks later.
        if kind == "sleep" and (math.hypot(tx - self.x, ty - self.y) < 3.0 or d["sleep"] >= SLEEP_PRESSURE_ANYWHERE):
            self.asleep = True
            self.stats["sleeps"] += 1
            self.sleep_episodes.append((tick, self.x, self.y))
            events["slept_here"] = True
            journal.log(tick, "sleep", "نام", x=self.x, y=self.y)
            self.intent = None
            return events
        if d["sleep"] >= SLEEP_PRESSURE_ANYWHERE + 3.0:  # collapse anywhere
            self.asleep = True
            self.stats["sleeps"] += 1
            self.sleep_episodes.append((tick, self.x, self.y))
            events["slept_here"] = True
            journal.log(tick, "sleep", "نام حيث أدركه التعب", x=self.x, y=self.y)
            return events

        # --- move toward target ---
        dist = math.hypot(tx - self.x, ty - self.y)
        arrival = 2.2 if kind in ("eat", "gather") else (1.2 if kind == "flee" else 3.0)
        if kind == "bathe":
            arrival = 1.0   # he must actually wade in, not stop at the bank
        if dist > arrival:
            # depth-zoned water movement (Stage 2): the deep channel is
            # swum (slowest, costliest), the shallows waded
            speed = WALK_SPEED * (self.swim_slow_factor() if is_deep_water(self.x, self.y)
                                  else (WATER_SLOW_FACTOR if percept.in_water else 1.0))
            step = min(speed, dist)
            ux, uy = (tx - self.x) / dist, (ty - self.y) / dist
            nx = self.x + ux * step
            ny = self.y + uy * step
            # learned local avoidance: never step INTO a feared hazard's margin
            # if a deflected step keeps him out (reactive, uses learned fear only)
            if kind != "flee":
                def enters_feared(px, py):
                    for th in world.thickets:
                        if self.fear.get(f"thicket:{th.id}", 0.0) > 0.4:
                            now = math.hypot(th.x - self.x, th.y - self.y)
                            new = math.hypot(th.x - px, th.y - py)
                            if new < th.r + 1.5 and now >= th.r + 1.5:
                                return True
                    return False
                if enters_feared(nx, ny):
                    for ang in (0.96, -0.96, 1.92, -1.92):
                        ca, sa = math.cos(ang), math.sin(ang)
                        rx, ry = ux * ca - uy * sa, ux * sa + uy * ca
                        cxp, cyp = self.x + rx * step, self.y + ry * step
                        if is_passable(cxp, cyp) and not enters_feared(cxp, cyp):
                            nx, ny = cxp, cyp
                            break
            moved = False
            for cand in ((nx, ny), (nx, self.y), (self.x, ny)):
                if is_passable(cand[0], cand[1]):
                    self.stats["distance"] += math.hypot(cand[0] - self.x, cand[1] - self.y)
                    self.x, self.y = cand
                    moved = True
                    break
            if not moved:
                self.stuck += 1
                if self.stuck > 30:
                    self.intent = None
                    self.stuck = 0
            else:
                self.stuck = 0
            return events

        # --- arrival actions ---
        if kind == "eat":
            if payload <= -1000:
                pile = pile_by_id(world, -1000 - payload)
                if pile is not None and pile.fruit > 0:
                    self.acting = ("eat", EAT_TICKS_PER_FRUIT, payload)
                else:
                    if self.mind is not None and pile is not None:
                        self.mind.note_obstacle("eat", tick, pile.x, pile.y,
                                                journal)
                    self.intent = None
            elif payload == -1:
                # position-only food target (an episode's place): eat what
                # is actually here — a pile or a tree within reach
                done = False
                for pile in world.piles:
                    if pile.fruit > 0 and math.hypot(pile.x - self.x, pile.y - self.y) < 3.5:
                        self.acting = ("eat", EAT_TICKS_PER_FRUIT, -1000 - pile.id)
                        done = True
                        break
                if not done:
                    for tree in world.trees:
                        if tree.fruit > 0 and math.hypot(tree.x - self.x, tree.y - self.y) < 3.5:
                            self.acting = ("eat", EAT_TICKS_PER_FRUIT, tree.id)
                            done = True
                            break
                if not done:
                    self.intent = None
            else:
                tree = world.trees[payload]
                if tree.fruit > 0:
                    self.acting = ("eat", EAT_TICKS_PER_FRUIT, payload)
                else:
                    # arriving to eat and finding nothing settles claims too
                    if self.mind is not None:
                        self.mind.monitor.verify(f"tree:{payload}", 0, tick)
                    self.food_memory[payload] = [tree.x, tree.y, 0, tick]
                    self.stats["wasted_arrivals"] = self.stats.get("wasted_arrivals", 0) + 1
                    # obstacle memory (Phase 4): she went, found nothing —
                    # remembered. Twice in 3 days teaches her to look away.
                    if self.mind is not None:
                        self.mind.note_obstacle("eat", tick, tree.x, tree.y,
                                                journal)
                    self.intent = None
        elif kind == "drink":
            if dist_to_river(self.x, self.y) < 6.5:
                d["thirst"] -= DRINK_RELIEF
                self.stats["drinks"] += 1
                events["drank"] = True
                self._learn_outcome(self.x, self.y, 0.8)
                journal.log(tick, "drink", "شرب من النهر", x=self.x, y=self.y)
            self.intent = None
        elif kind == "bathe":
            # the bath itself is a multi-tick act in the water; if the
            # arrival point somehow is not in the water, the intent
            # simply ends (he re-decides next refresh)
            if is_water(self.x, self.y):
                self.acting = ("bathe", BATHE_TICKS, -1)
            self.intent = None
        elif kind == "gather":
            for m in list(world.materials):
                if m.id == payload:
                    world.materials.remove(m)
                    self.carry.append(m.kind)
                    self.stats["gathers"] += 1
                    self.practice["forage"] += 1    # P5: lived practice
                    events["gathered"] = True
                    self.cleanliness = max(0.0, self.cleanliness - 1.2)  # work dirt (Stage 2)
                    d["creation"] -= 6.0
                    journal.log(tick, "gather", f"جمع {m.kind}", x=self.x, y=self.y)
                    break
            self.intent = None
        elif kind == "place":
            if self.carry:
                kind_item = self.carry.pop()
                # Phase 4: she builds WHERE the intent points — usually the
                # usual site, sometimes her own chosen spot (creativity).
                # Structures cluster around the chosen base.
                base_x, base_y = tx, ty
                if payload == -2:
                    # Stage 4 structured-build slot: place EXACTLY on the
                    # project's slot — no grid jitter, the pattern survives.
                    px, py = base_x, base_y
                else:
                    near = sum(1 for s in world.structures
                               if math.hypot(s["x"] - base_x, s["y"] - base_y) < 12.0)
                    px, py = (base_x + (near % 5) * 1.6,
                              base_y + (near // 5) * 1.6)
                world.structures.append({"kind": kind_item, "x": px, "y": py})
                self.stats["places"] += 1
                self.practice["build"] += 1    # P5: lived practice
                events["placed"] = True
                self.cleanliness = max(0.0, self.cleanliness - 0.8)  # work dirt (Stage 2)
                d["creation"] -= 30.0
                self._learn_outcome(self.x, self.y, 0.7)
                journal.log(tick, "place", f"وضع {kind_item} في بنائه", x=self.x, y=self.y)
            self.intent = None
        elif kind == "plant":
            # P6: body capability (no mind scoring yet — she may imitate it)
            from world import Crop
            if getattr(self, "seeds", 0) > 0 and is_passable(self.x, self.y):
                world.crops.append(Crop(len(world.crops), self.x, self.y,
                                        planter="fajr"))
                self.seeds -= 1
            self.intent = None
        elif kind == "water":
            # P6: body capability
            for c in world.crops:
                if math.hypot(c.x - self.x, c.y - self.y) < 5.0 \
                        and c.stage != "mature":
                    c.water = min(100.0, c.water + 50.0)
                    break
            self.intent = None
        elif kind == "harvest":
            # P6: body capability
            for c in world.crops:
                if math.hypot(c.x - self.x, c.y - self.y) < 3.0 \
                        and c.stage == "mature" and c.fruit > 0:
                    n = c.fruit
                    c.fruit = 0
                    c.regrow = 0
                    d["hunger"] = max(0.0, d["hunger"] - FRUIT_SATIATION * n)
                    self.stats["eats"] += n
                    self.seeds = min(6, getattr(self, "seeds", 0) + n)
                    events["ate"] = True
                    journal.log(tick, "harvest", "حصد من زرعه", x=self.x, y=self.y)
                    break
            self.intent = None
        else:
            # seek/explore/social/shelter/flee/wander: arrival ends the intent
            self.intent = None
        return events

    # ---------------- (de)serialisation ----------------
    def to_dict(self):
        return {
            "x": self.x, "y": self.y,
            "drives": self.drives.to_dict(),
            "asleep": self.asleep,
            "carry": list(self.carry),
            "food_memory": {str(k): list(v) for k, v in self.food_memory.items()},
            "fear": dict(self.fear),
            "place_value": self.place_value,
            "novelty": self.novelty,
            "visits": self.visits,
            "intent": list(self.intent) if self.intent else None,
            "intent_age": self.intent_age,
            "acting": list(self.acting) if self.acting else None,
            "in_deep": self.in_deep,
            # Stage 2 water & hygiene state — part of the canonical
            # state (a resumed life must continue the same body)
            "cleanliness": self.cleanliness,
            "wet": self.wet,
            "bathe_value": self.bathe_value,
            "deep_stint": self._deep_stint,
            "last_cell": list(self.last_cell),
            "stuck": self.stuck,
            "stats": dict(self.stats),
            # P5: lived-practice counters (the only source of skill level)
            "practice": dict(self.practice),
            # P6: seeds carried
            "seeds": self.seeds,
            "sleep_episodes": [list(e) for e in self.sleep_episodes],
            "hunger_onset": list(self.hunger_onset) if self.hunger_onset else None,
            "hunger_latencies": [list(e) for e in self.hunger_latencies],
            "thicket_log": [list(e) for e in self.thicket_log],
            "inside_thicket": sorted(self._inside_thicket),
            "mind": self.mind.to_dict() if self.mind is not None else None,
        }

    @classmethod
    def from_dict(cls, dd, rng):
        f = cls.__new__(cls)
        f.x, f.y = dd["x"], dd["y"]
        f.drives = Drives.from_dict(dd["drives"])
        f.asleep = dd["asleep"]
        f.carry = list(dd["carry"])
        f.food_memory = {int(k): list(v) for k, v in dd["food_memory"].items()}
        f.fear = dict(dd["fear"])
        f.place_value = [list(r) for r in dd["place_value"]]
        f.novelty = [list(r) for r in dd["novelty"]]
        f.visits = [list(r) for r in dd["visits"]]
        f.intent = tuple(dd["intent"]) if dd["intent"] else None
        f.intent_age = dd["intent_age"]
        f.acting = tuple(dd["acting"]) if dd["acting"] else None
        f.in_deep = dd["in_deep"]
        # Stage 2 state; checkpoints written before Stage 2 predate it
        f.cleanliness = float(dd.get("cleanliness", 100.0))
        f.wet = float(dd.get("wet", 0.0))
        f.bathe_value = float(dd.get("bathe_value", 0.25))
        f._deep_stint = int(dd.get("deep_stint", 0))
        f.last_cell = tuple(dd["last_cell"])
        f.stuck = dd["stuck"]
        f.stats = dict(dd["stats"])
        # P5: practice counters; checkpoints that predate P5 seed them
        # from the real lived-history stats (her past counts toward it)
        pr = dd.get("practice")
        if pr:
            f.practice = {k: int(pr.get(k, 0)) for k in ("forage", "swim", "build")}
        else:
            f.practice = {"forage": int(f.stats.get("gathers", 0)),
                          "swim": int(f.stats.get("swim_ticks", 0)),
                          "build": int(f.stats.get("places", 0))}
        # P6: seeds; old checkpoints start at 0
        f.seeds = int(dd.get("seeds", 0))
        f.sleep_episodes = [tuple(e) for e in dd["sleep_episodes"]]
        f.hunger_onset = tuple(dd["hunger_onset"]) if dd["hunger_onset"] else None
        f.hunger_latencies = [tuple(e) for e in dd["hunger_latencies"]]
        f.thicket_log = [tuple(e) for e in dd["thicket_log"]]
        f._inside_thicket = set(dd["inside_thicket"])
        f.journal = None
        # Phase-2 mind: restored when present; a Phase-1 checkpoint gets a
        # fresh mind (documented in PHASE2-REPORT — v1 states predate it)
        f.mind = Mind.from_dict(dd["mind"], rng) if dd.get("mind") else Mind(rng)
        f.mind_events = {}
        f._raw = None
        return f
