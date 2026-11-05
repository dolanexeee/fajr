"""Simulation core: one deterministic tick loop binding world + body + drives,
the life journal, gate metrics, and JSON checkpoints.

Checkpoint contract (GATE G5): the canonical state (everything that can
influence the future — tick, PRNG, world dynamics, Fajr, metrics) serialises
to JSON; sha256 of the canonical dump is the state hash. A save -> load ->
continue run must end with the identical hash as the uninterrupted run.
"""
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TICKS_PER_DAY, LANTERN_POS, SOCIAL_RADIUS
from drives import VITAL_BANDS
from fajr import Fajr
from adam import Adam, SPAWN_POS
from bond import (PairBond, UNION_R, OBSERVE_R, BOND_UP_MUTUAL,
                   BOND_UP_CLOSE, BOND_DECAY, FAMILY_RELIEF_CLOSE,
                   FAMILY_RELIEF_MUTUAL, BOND_MILESTONES,
                   CONCEIVE_NEED, GESTATION_TICKS)
from culture import Culture, SKILLS as CULTURE_SKILLS, ERA_GENERATION
from child import Child, FEED_R
from prng import PRNG
from senses import perceive
from world import World, light_level, is_deep_water, is_passable


def fmt_time(tick_of_day):
    return f"{tick_of_day // 60:02d}:{tick_of_day % 60:02d}"


class Journal:
    """His life log, written continuously. Entries are data, not narration
    imposed on him: (tick, day, time, type, arabic label, position)."""

    def __init__(self, base_count=0):
        self.entries = []
        self.base = base_count

    def log(self, tick, type_, text, x=None, y=None):
        self.entries.append({
            "tick": tick, "day": tick // TICKS_PER_DAY + 1,
            "time": fmt_time(tick % TICKS_PER_DAY), "type": type_, "text": text,
            "x": None if x is None else round(x, 1), "y": None if y is None else round(y, 1),
        })

    @property
    def total_count(self):
        return self.base + len(self.entries)


class Sim:
    def __init__(self, seed=11, layout="default", mind_config=None):
        self.seed = seed
        self.layout = layout
        self.rng = PRNG(seed * 1000003 + 77)
        self.world = World(seed, layout)
        self.fajr = Fajr(self.rng)
        self.adam = None               # ADAM & EVE P2: spawns when she is ready
        self.bond = PairBond()         # ADAM & EVE P3: the pair's bond
        self.child = None              # ADAM & EVE P4: the offspring
        self.culture = Culture()       # ADAM & EVE P5: cumulative knowledge
        self._adam_just_ate = False    # transient, reset each tick
        self._apply_mind_config(mind_config)
        self.journal = Journal()
        self.fajr.journal = self.journal
        self.tick = 0
        self.metrics = {
            "ticks_total": 0, "ticks_in_band": 0,
            "sleep_ticks": 0, "sleep_ticks_night": 0,
            "max_pinned": {k: 0 for k in VITAL_BANDS},
            "pin_run": {k: 0 for k in VITAL_BANDS},
            "distance_by_day": [],
            "day_start_distance": 0.0,
            "valence_hourly": [],
        }
        self.journal.log(0, "awakening", "البداية — صحا فجر في الوادي قرب المأوى، بلا ذاكرة سابقة",
                         x=self.fajr.x, y=self.fajr.y)

    def _apply_mind_config(self, cfg):
        """Phase-2 lesion/configuration switches (battery use). Keys:
        master_off, workspace_off, single_query, monitor_constant,
        episodic_off, schema_off, feedforward_off, topdown_off,
        routing_off (v4-3 ablation: decision routing disabled)."""
        self.mind_config = dict(cfg) if cfg else None
        if not cfg:
            return
        m = self.fajr.mind
        if cfg.get("master_off"):
            m.active = False
            m.steers = False
        if cfg.get("workspace_off"):
            m.workspace.lesion_off = True
            m.steers = False       # agency reverts to Phase-1 arbitration
        if cfg.get("single_query"):
            m.workspace.lesion_single_query = True
        if cfg.get("monitor_constant"):
            m.monitor.lesion_constant = True
        if cfg.get("episodic_off"):
            m.episodic.lesion_off = True
        if cfg.get("schema_off"):
            m.workspace.schema.enabled = False
        if cfg.get("feedforward_off"):
            m.perception.lesion_feedforward = True
        if cfg.get("topdown_off"):
            m.perception.lesion_topdown = True
        if cfg.get("routing_off"):
            m.routing_enabled = False

    # ---------------- one tick ----------------
    def step(self):
        tick = self.tick
        tod = tick % TICKS_PER_DAY
        light = light_level(tod)
        w = self.world
        f = self.fajr
        w.update(tick, light, self.rng)

        # world sounds (hearing channel)
        sounds = []
        if w.weather == "storm" and self.rng.chance(0.004):
            sounds.append("thunder")
            f.drives.d["safety"] += 6.0
            self.journal.log(tick, "thunder", "رعد", x=f.x, y=f.y)
        if 380 <= tod <= 620 and (80 < f.x < 250):
            if self.rng.chance(0.002):
                sounds.append("birds")

        percept = perceive(w, f, light, sounds)
        f.observe(percept, tick)
        # transient link for the union intent (never serialised)
        f.partner = self.adam
        events = f.step(w, percept, tick, light, self.rng, self.journal)
        f.mind_events = events   # the mind reads last tick's outcomes

        # physiology
        ctx = {
            "asleep": f.asleep, "light": light,
            "in_shelter": percept.in_shelter,
            "in_rain_exposed": w.weather in ("rain", "storm") and not percept.in_shelter,
            "near_lantern": math.hypot(LANTERN_POS[0] - f.x, LANTERN_POS[1] - f.y) < SOCIAL_RADIUS,
            "near_hazard": events["near_hazard"],
            "cold_water": is_deep_water(f.x, f.y),
            # ADAM & EVE P3: bond feeds her family drive (binding design)
            "bond": self.bond.value, "mature": 1.0,
        }
        prev_hunger = f.drives.d["hunger"]
        f.drives.update(ctx)
        # hunger-episode onset (for G3a latencies)
        if prev_hunger < 60.0 <= f.drives.d["hunger"] and f.hunger_onset is None:
            f.hunger_onset = (tick // TICKS_PER_DAY + 1, tick)
        # fear slow decay + extinction by safe exposure
        for key in sorted(f.fear.keys()):
            f.fear[key] = max(0.0, f.fear[key] * (1.0 - 0.00002))
        # novelty refreshes on a forgetting timescale of ~9 days: a place must
        # stay unvisited that long before it counts as novel again (a faster
        # refresh treadmills the search around home and the frontier never
        # pushes outward — measured in the food_probe assay, 2026-10-22).
        for row in f.novelty:
            for i in range(len(row)):
                if row[i] < 1.0:
                    row[i] = min(1.0, row[i] + 0.000008)

        # ---- Adam (ADAM & EVE P2): enters when she is ready, lives beside her
        if self.adam is None and self.fajr.mind._kinship_ready:
            self.spawn_adam()
        if self.adam is not None:
            _eats0 = self.adam.stats["eats"]
            self.adam.step(self.world, self.fajr, tick, light, self.rng,
                           self.journal, bond=self.bond.value, mature=1.0)
            self._adam_just_ate = self.adam.stats["eats"] > _eats0
            # ---- pair bond (ADAM & EVE P3): lived closeness ----
            self._pair_tick(tick)
            # ---- imitation (ADAM & EVE P3/P5): she teaches by living ----
            self._imitation_tick(tick)
            # ---- culture (ADAM & EVE P5): era milestones ----
            self._culture_tick(tick)
            # ---- discovery (ADAM & EVE P6): observed emergence ----
            self._discovery_tick(tick)
            # ---- offspring (ADAM & EVE P4): gestation, birth, growth ----
            self._pregnancy_tick(tick)
        if self.child is not None:
            self.child.other_parent = self.adam  # transient, never serialised
            self.child.step(self.world, self.fajr, tick, light, self.rng,
                            self.journal, bond=self.bond.value,
                            mature=self.child.maturity())
            self._feed_tick(tick)
            self._growth_tick(tick)

        # ---- metrics ----
        m = self.metrics
        m["ticks_total"] += 1
        if all(f.drives.d[k] < band for k, band in VITAL_BANDS.items()):
            m["ticks_in_band"] += 1
        for k in VITAL_BANDS:
            if f.drives.d[k] >= 100.0:
                m["pin_run"][k] += 1
                m["max_pinned"][k] = max(m["max_pinned"][k], m["pin_run"][k])
            else:
                m["pin_run"][k] = 0
        if f.asleep:
            m["sleep_ticks"] += 1
            if light < 0.25:
                m["sleep_ticks_night"] += 1
        if tick % 60 == 0:
            m["valence_hourly"].append(round(f.drives.valence(), 4))
        if (tick + 1) % TICKS_PER_DAY == 0:
            m["distance_by_day"].append(round(f.stats["distance"] - m["day_start_distance"], 2))
            m["day_start_distance"] = f.stats["distance"]
            self.journal.log(tick, "day_summary",
                             f"نهاية اليوم {tick // TICKS_PER_DAY + 1}: أكل {f.stats['eats']}، شرب {f.stats['drinks']}، "
                             f"نام {f.stats['sleeps']} مرة، بنى {f.stats['places']}، قطع {m['distance_by_day'][-1]} وحدة",
                             x=f.x, y=f.y)
        self.tick += 1

    def run_ticks(self, n):
        for _ in range(n):
            self.step()

    def spawn_adam(self):
        """ADAM & EVE P2: the male enters at the valley edge, zero knowledge.
        Called once, the first tick after she is marked ready (P1)."""
        x, y = SPAWN_POS
        if not is_passable(x, y):
            # deterministic nudge: scan outward until passable
            found = None
            for r in (2.0, 4.0, 6.0, 10.0, 16.0):
                for k in range(12):
                    ang = (k / 12.0) * 2.0 * math.pi
                    cx, cy = x + math.cos(ang) * r, y + math.sin(ang) * r
                    if is_passable(cx, cy):
                        found = (cx, cy)
                        break
                if found:
                    break
            if found:
                x, y = found
        self.adam = Adam(self.rng, x, y)
        self.journal.log(self.tick, "adam_entered",
                         "دخل رجل الوادي — لا يعرف شيئًا", x=x, y=y)

    def _pair_tick(self, tick):
        """ADAM & EVE P3: one tick of pair life.

        - Closeness (within UNION_R) relieves the family drive in both.
        - Mutual union choice (BOTH seeking each other) grows the bond
          faster and accrues P4's conception counter.
        - Milestones journaled behaviorally.
        Imitation teaching is a separate _imitation_tick (observation
        radius, not closeness radius).
        """
        f, a = self.fajr, self.adam
        bond = self.bond
        dist = math.hypot(f.x - a.x, f.y - a.y)
        if dist < UNION_R:
            bond.close_ticks += 1
            f_chose = (f.mind.chosen_intent is not None
                       and f.mind.chosen_intent[0] == "union")
            # P4-fix: arrival fulfils his choice, it doesn't erase it —
            # seeking_her persists past the intent clearing on arrival
            a_chose = (a.intent is not None and a.intent[0] in ("union", "follow")) \
                or getattr(a, "seeking_her", False)
            # Mutual = chosen togetherness (2026-11-04 revision): requiring
            # SIMULTANEOUS active union intents was too strict — 25 live days
            # yielded 0 mutual ticks because real togetherness includes
            # doing things side by side. Now: together, both feel the family
            # need, both awake, and at least one actively seeking the other.
            # Presence maintained despite alternatives IS a choice.
            f_want = f.drives.d["family"] > 40.0
            a_want = a.drives.d["family"] > 40.0
            awake = not f.asleep and not a.asleep
            mutual = (f_want and a_want and awake and (f_chose or a_chose))
            # relief: mild for mere closeness, strong for mutual choice —
            # the drive must be ABLE to reach the union threshold, or the
            # choice at the heart of the design never happens
            relief = FAMILY_RELIEF_MUTUAL if mutual else FAMILY_RELIEF_CLOSE
            for agent in (f, a):
                agent.drives.d["family"] = max(
                    0.0, agent.drives.d["family"] - relief)
            if mutual:
                bond.union_ticks += 1
                bond.conceive_ticks += 1
                bond.value = min(1.0, bond.value + BOND_UP_MUTUAL)
                # P4 conception: the CONSEQUENCE of their sustained chosen
                # togetherness — never an external trigger. The counter
                # grows ONLY on mutual ticks (together + both want + awake
                # + at least one seeking). No "bond > 0.8 -> spawn" anywhere.
                if (not bond.pregnant and self.child is None
                        and bond.conceive_ticks >= CONCEIVE_NEED):
                    bond.pregnant = True
                    bond.gestation_left = GESTATION_TICKS
                    bond.conceive_ticks = 0
                    self.journal.log(tick, "pregnancy", "بدا عليها الحمل",
                                     x=f.x, y=f.y)
                # throttled union episodes -> her kinship dreams (P1 theme)
                if bond.union_ticks % 240 == 0:
                    f.stats["unions"] = f.stats.get("unions", 0) + 1
                    f.mind.episodic.record(tick, "union", f.x, f.y, [], {},
                                           f.drives.valence(), 0.3,
                                           outcome=0.6, note="union")
            else:
                bond.value = min(1.0, bond.value + BOND_UP_CLOSE)
        else:
            bond.value = max(0.0, bond.value - BOND_DECAY)
        for thresh, key, text in BOND_MILESTONES:
            if bond.value >= thresh and key not in bond.milestones:
                bond.milestones.append(key)
                self.journal.log(tick, "bond", text, x=f.x, y=f.y)

    def _imitation_tick(self, tick):
        """She teaches by living; they learn by watching (P3/P5).

        Within OBSERVATION radius (not closeness radius), her completed
        acts grant PRACTICE via Culture.transmit: the learner closes a
        fraction of the gap toward her level with a small transmission
        loss. Teaching alone never grants a level outright; practice
        never decays — knowledge only ratchets upward. A child learner
        in child stage learns at double speed through Child's
        practice_skill (P5: shoulders of parents).
        The Guide NEVER teaches them — by construction, no code path does.
        """
        f = self.fajr
        learners = []
        if self.adam is not None:
            learners.append(self.adam)
        if self.child is not None:
            learners.append(self.child)
        if not learners:
            return
        ev = f.mind_events or {}
        acts = []
        if ev.get("ate") or ev.get("gathered"):
            acts.append("forage")
        if ev.get("placed"):
            acts.append("build")
        if ev.get("swam"):
            acts.append("swim")
        if not acts:
            return
        learned_any = False
        for a in learners:
            if math.hypot(f.x - a.x, f.y - a.y) > OBSERVE_R:
                continue
            for skill in acts:
                grant = self.culture.transmit(a.practice[skill],
                                              f.skill_level(skill))
                if grant:
                    a.practice_skill(skill, grant)
                    self.culture.teach_events += 1
                    learned_any = True
        if learned_any and tick - self.bond.last_imitate_tick > 720:
            self.bond.last_imitate_tick = tick
            lx = self.adam.x if self.adam is not None else f.x
            ly = self.adam.y if self.adam is not None else f.y
            self.journal.log(tick, "imitation", "يتعلم منها بمشاهدتها",
                             x=lx, y=ly)

    # ---------------- ADAM & EVE P5: perpetual learning ----------------
    def _culture_tick(self, tick):
        """Tribe era milestones on collective capability (checked
        every 360 ticks — cheap, and eras are rare)."""
        if tick % 360 != 0:
            return
        f = self.fajr
        levels = [{s: f.skill_level(s) for s in CULTURE_SKILLS}]
        if self.adam is not None:
            levels.append({s: self.adam.skill(s) for s in CULTURE_SKILLS})
        if self.child is not None:
            levels.append({s: self.child.skill(s) for s in CULTURE_SKILLS})
        self.culture.check_skill_eras(levels, tick, self.journal, f.x, f.y)

    # ---------------- ADAM & EVE P6: open evolution ----------------
    def _discovery_tick(self, tick):
        """P6: OBSERVED emergence, never scripted. When a PLANTED crop is
        harvested, the tribe has discovered farming — registered as
        permanent knowledge (never decays: no dark ages), then the era
        fires AFTER the fact, like the swim era did. The SIM doesn't know
        "farming"; it sees plant->water->harvest->eat and names what
        happened, the way we named the hut."""
        for agent in (self.adam, self.child):
            if agent is None:
                continue
            lh = getattr(agent, "_last_harvest", None)
            if lh is None:
                continue
            agent._last_harvest = None
            _, planter = lh
            if planter is None:
                continue  # wild crop — not a discovery
            if self.culture.register_discovery(
                    "farming", "اكتشف الزراعة في الوادي",
                    tick, self.journal, agent.x, agent.y, by=planter):
                agent.curiosity.known.add("farming")
                self.culture.mark("era_farming", "بدأ عصر الزراعة في الوادي",
                                  tick, self.journal, agent.x, agent.y)
        # discoveries transmit by observation (P5 spirit, P6 content)
        if self.culture.knows("farming"):
            self._teach_discovery_tick(tick)

    def _teach_discovery_tick(self, tick):
        """Watching a knower do farming verbs teaches the discovery.
        Throttled journaling; the knowledge itself never decays."""
        teachers = [a for a in (self.adam, self.child)
                    if a is not None and "farming" in a.curiosity.known]
        learners = [a for a in (self.adam, self.child)
                    if a is not None and "farming" not in a.curiosity.known]
        if not teachers or not learners:
            return
        for t in teachers:
            for lrn in learners:
                if t is lrn:
                    continue
                if math.hypot(t.x - lrn.x, t.y - lrn.y) > 22.0:
                    continue
                if t.intent is not None and t.intent[0] in ("plant", "water", "harvest"):
                    lrn.curiosity.known.add("farming")
                    if tick - self.bond.last_imitate_tick > 1440:
                        self.bond.last_imitate_tick = tick
                        self.journal.log(tick, "imitation",
                                         "تعلم الزراعة بمشاهدتها",
                                         x=lrn.x, y=lrn.y)
                    return

    # ---------------- ADAM & EVE P4: offspring ----------------
    def _pregnancy_tick(self, tick):
        """Gestation countdown (fixed physics) -> birth event."""
        bond = self.bond
        if not bond.pregnant:
            return
        bond.gestation_left -= 1
        if bond.gestation_left > 0:
            return
        bond.pregnant = False
        f, a = self.fajr, self.adam
        x, y = (f.x + a.x) / 2.0, (f.y + a.y) / 2.0
        if not is_passable(x, y):
            found = None
            for r in (2.0, 4.0, 8.0, 14.0):
                for k in range(12):
                    ang = (k / 12.0) * 2.0 * math.pi
                    cx, cy = x + math.cos(ang) * r, y + math.sin(ang) * r
                    if is_passable(cx, cy):
                        found = (cx, cy)
                        break
                if found:
                    break
            if found:
                x, y = found
        sex = "male" if self.rng.chance(0.5) else "female"
        self.child = Child(self.rng, x, y, sex)
        self.journal.log(tick, "birth", "وُلد طفل في الوادي", x=x, y=y)
        # P5: a new generation is a tribe era (our reading, honest label)
        self.culture.mark(ERA_GENERATION[0], ERA_GENERATION[1],
                          tick, self.journal, x, y)

    def _feed_tick(self, tick):
        """An infant cannot gather: an adult's completed meal nearby
        shares food (physics)."""
        from config import FRUIT_SATIATION
        c = self.child
        if c.stage() != "infant":
            return
        fed = False
        ev_f = self.fajr.mind_events or {}
        if ev_f.get("ate") and math.hypot(self.fajr.x - c.x,
                                          self.fajr.y - c.y) < FEED_R:
            fed = True
        if (not fed and getattr(self, "_adam_just_ate", False)
                and self.adam is not None
                and math.hypot(self.adam.x - c.x,
                               self.adam.y - c.y) < FEED_R):
            fed = True
        if not fed:
            return
        c.drives.d["hunger"] = max(
            0.0, c.drives.d["hunger"] - FRUIT_SATIATION * 0.8)
        if tick - self.bond.last_feed_tick > 1440:
            self.bond.last_feed_tick = tick
            self.journal.log(tick, "feeding", "أطعموا الطفل", x=c.x, y=c.y)

    def _growth_tick(self, tick):
        """Stage transitions are journaled once each (physics)."""
        c = self.child
        now = c.stage()
        if now != c._stage:
            c._stage = now
            if now == "child":
                self.journal.log(tick, "grew", "كبر الطفل — صار يتعلم بسرعة",
                                 x=c.x, y=c.y)
            elif now == "adult":
                self.journal.log(tick, "grew", "صار الطفل بالغًا",
                                 x=c.x, y=c.y)

    def run_days(self, days):
        self.run_ticks(days * TICKS_PER_DAY)

    # ---------------- checkpoint ----------------
    def state_dict(self):
        return {
            "version": 2, "seed": self.seed, "layout": self.layout, "tick": self.tick,
            "prng": self.rng.to_dict(),
            "world": self.world.to_dict(),
            "fajr": self.fajr.to_dict(),
            "adam": self.adam.to_dict() if self.adam is not None else None,
            "bond": self.bond.to_dict(),
            "child": self.child.to_dict() if self.child is not None else None,
            "culture": self.culture.to_dict(),
            "metrics": self.metrics,
            "journal_count": self.journal.total_count,
        }

    @staticmethod
    def canonical_hash(state) -> str:
        # Normalise through a JSON round-trip BEFORE hashing so the
        # hash is always taken over the disk-normal form. Without this,
        # a state dict with integer keys (e.g. perception metrics
        # err_day, keyed by day number) dumps with numerically sorted
        # keys in memory but reloads with string keys that sort
        # lexicographically — the two blobs diverge the moment a key
        # reaches two digits (day 10), and every checkpoint written
        # after that fails verification on load. (Guardian repair,
        # 2026-10-23: state content untouched; see
        # bridge/CHECKPOINT-REPAIR-20261023.md.)
        state = json.loads(json.dumps(state, ensure_ascii=False))
        blob = json.dumps(state, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def save(self, path):
        state = self.state_dict()
        payload = {"hash": self.canonical_hash(state), "state": state}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        state = payload["state"]
        if cls.canonical_hash(state) != payload["hash"]:
            raise ValueError("checkpoint hash mismatch — file corrupted or tampered")
        sim = cls.__new__(cls)
        sim.seed = state["seed"]
        sim.layout = state["layout"]
        sim.tick = state["tick"]
        sim.rng = PRNG.from_dict(state["prng"])
        sim.world = World.from_dict(state["world"])
        sim.fajr = Fajr.from_dict(state["fajr"], sim.rng)
        adam_d = state.get("adam")
        sim.adam = Adam.from_dict(adam_d, sim.rng) if adam_d else None
        sim.bond = PairBond.from_dict(state.get("bond", {}))
        child_d = state.get("child")
        sim.child = Child.from_dict(child_d, sim.rng) if child_d else None
        sim.culture = Culture.from_dict(state.get("culture", {}))
        sim._adam_just_ate = False
        sim.mind_config = None
        sim.metrics = state["metrics"]
        sim.journal = Journal(base_count=state["journal_count"])
        sim.fajr.journal = sim.journal
        return sim
