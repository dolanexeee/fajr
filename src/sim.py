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
from prng import PRNG
from senses import perceive
from world import World, light_level, is_deep_water


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

    def run_days(self, days):
        self.run_ticks(days * TICKS_PER_DAY)

    # ---------------- checkpoint ----------------
    def state_dict(self):
        return {
            "version": 2, "seed": self.seed, "layout": self.layout, "tick": self.tick,
            "prng": self.rng.to_dict(),
            "world": self.world.to_dict(),
            "fajr": self.fajr.to_dict(),
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
        sim.mind_config = None
        sim.metrics = state["metrics"]
        sim.journal = Journal(base_count=state["journal_count"])
        sim.fajr.journal = sim.journal
        return sim
