"""The valley — SPEC §3.

Continuous 2D world. Terrain from an analytic elevation function (no stored
grid needed for physics; a raster is exported only for the viewer).
The river runs lengthwise, the meadow holds the fruit trees, the forest edge
holds wood/stone and the thorn thickets (the honest, world-physics dangers),
the hill gives a vantage, the shelter is home, and the mountain+fog ring is
the true edge of his world — impassable, and no sense reaches past it.
"""
import math

from config import WORLD_W, WORLD_H, SHELTER_POS, LANTERN_POS
from prng import PRNG

RIVER_HALF_WIDTH = 4.0
DEEP_HALF_WIDTH = 1.6


def river_center_x(y: float) -> float:
    return 58.0 + 16.0 * math.sin(y / 30.0) + 6.0 * math.sin(y / 11.0)


def dist_to_river(x: float, y: float) -> float:
    return abs(x - river_center_x(y))


def elevation(x: float, y: float) -> float:
    """0..1. >0.78 = mountain (impassable). Hill near (208, 42)."""
    # valley floor: gentle bowl
    e = 0.18 + 0.10 * math.sin(x / 47.0) * math.cos(y / 39.0)
    # the hill (vantage)
    dx, dy = x - 208.0, y - 42.0
    e += 0.55 * math.exp(-(dx * dx + dy * dy) / (2 * 26.0 ** 2))
    # mountain ring rising toward every edge
    ex = min(x, WORLD_W - x) / 26.0
    ey = min(y, WORLD_H - y) / 26.0
    edge = max(0.0, 1.0 - min(ex, ey))
    e += 0.85 * edge ** 3
    return e


def is_water(x: float, y: float) -> bool:
    return dist_to_river(x, y) < RIVER_HALF_WIDTH


def is_deep_water(x: float, y: float) -> bool:
    return dist_to_river(x, y) < DEEP_HALF_WIDTH


def depth_zone(x: float, y: float) -> str:
    """Stage 2 (WATER-GATE): the river's depth zones. 'deep' = the
    channel (dist < DEEP_HALF_WIDTH): crossable only by swimming.
    'shallow' = the banks (wading depth; where bathing happens).
    'dry' = land."""
    rd = dist_to_river(x, y)
    if rd < DEEP_HALF_WIDTH:
        return "deep"
    if rd < RIVER_HALF_WIDTH:
        return "shallow"
    return "dry"


def bath_spot(x: float, y: float):
    """A wading-depth point on the river at his latitude, on HIS side of
    the channel (dist_to_river = 2.8: in the water, out of the deep).
    Procedural body knowledge of the same kind as the drink target —
    where the water is, not when to go."""
    cx = river_center_x(y)
    side = 1.0 if x >= cx else -1.0
    return (cx + side * 2.8, y)


def is_passable(x: float, y: float) -> bool:
    if not (4.0 <= x <= WORLD_W - 4.0 and 4.0 <= y <= WORLD_H - 4.0):
        return False
    return elevation(x, y) <= 0.78


def in_forest(x: float, y: float) -> bool:
    return x > 188.0


def in_meadow(x: float, y: float) -> bool:
    return 88.0 <= x <= 172.0 and 38.0 <= y <= 152.0


class Tree:
    __slots__ = ("id", "x", "y", "fruit", "max_fruit", "regrow", "regrow_need")

    def __init__(self, tid, x, y, fruit):
        self.id, self.x, self.y = tid, x, y
        self.fruit = fruit
        self.max_fruit = 6
        self.regrow = 0
        self.regrow_need = 3200


class Thicket:
    """Thorn thicket — an honest world-physics hazard (SPEC §2.5)."""
    __slots__ = ("id", "x", "y", "r")

    def __init__(self, tid, x, y, r=4.5):
        self.id, self.x, self.y, self.r = tid, x, y, r


class Material:
    __slots__ = ("id", "kind", "x", "y")

    def __init__(self, mid, kind, x, y):
        self.id, self.kind, self.x, self.y = mid, kind, x, y


class Pile:
    """A temporary pile of fruit on the ground (e.g. fallen fruit).
    Present in the world physics like any object; probe worlds place one,
    the default layout has none."""
    __slots__ = ("id", "x", "y", "fruit")

    def __init__(self, pid, x, y, fruit):
        self.id, self.x, self.y, self.fruit = pid, x, y, fruit


class World:
    def __init__(self, seed: int, layout: str = "default"):
        rng = PRNG(seed * 7919 + 13)
        self.trees = []
        self.thickets = []
        self.materials = []
        self.piles = []
        self.structures = []          # placed items that persist: {kind, x, y}
        self.weather = "clear"        # clear | rain | storm
        self.weather_timer = 0
        self.season_phase = 0.0       # 0..1 across a 30-day cycle (seasons-lite)
        if layout == "default":
            self._layout_default(rng)
        elif layout == "fear_probe":
            self._layout_fear_probe(rng)
        elif layout == "food_probe":
            self._layout_food_probe(rng)
        elif layout == "binding_probe":
            # RPT-2 v2 construction (PHASE2-GATE amendment, 2026-10-23): an
            # orchard strip along the forest edge — fruit trees interleaved
            # with thorn thickets 8–18 units apart, initial fruit spread
            # 0..6 so |Δfruit| ≥ 3 pairs are common. Food exists ONLY here,
            # so a life lived in this world constantly produces the frozen
            # binding configuration (≥2 sensed trees with |Δtrue fruit| ≥ 3
            # AND a thicket within 20 units of him). Sitting 1's scene was
            # geometrically unsatisfiable: the default thickets sit at the
            # forest edge, >20 units from everywhere he actually lives.
            for i, (tx, ty) in enumerate([(186, 52), (196, 66), (184, 82),
                                           (198, 96), (186, 112), (196, 128),
                                           (188, 146)]):
                self.trees.append(Tree(i, tx, ty, [0, 5, 1, 6, 2, 4, 3][i]))
            for i, (tx, ty) in enumerate([(190, 74), (192, 104), (190, 134)]):
                self.thickets.append(Thicket(i, tx, ty))
            for i, (mx, my, mk) in enumerate([(150, 60, "wood"), (120, 140, "stone"),
                                              (170, 40, "wood"), (100, 60, "stone")]):
                self.materials.append(Material(i, mk, mx, my))
        elif layout == "gwt4_probe":
            # GWT-4 v2 construction (PHASE2-GATE amendment, 2026-10-23): NO
            # fruit trees anywhere. The probe adds one fruit pile at a
            # fixed point — the only food in the world. Eating it dry
            # drives its belief estimate to ~0, so on test morning the
            # location is carried ONLY by episodic memory; beliefs cannot
            # send him there and Phase-1 food_memory never records piles.
            for i, (mx, my, mk) in enumerate([(120, 60, "wood"), (90, 140, "stone"),
                                              (200, 150, "wood"), (60, 40, "stone")]):
                self.materials.append(Material(i, mk, mx, my))
        else:
            raise ValueError(layout)

    # ---------- layouts ----------
    def _scatter_tree(self, rng, x, y, fruit=None):
        t = Tree(len(self.trees), x, y,
                 rng.randint(2, 6) if fruit is None else fruit)
        self.trees.append(t)
        return t

    def _layout_default(self, rng: PRNG):
        # meadow fruit trees
        for _ in range(14):
            x = rng.range(96.0, 166.0)
            y = rng.range(46.0, 146.0)
            if math.hypot(x - SHELTER_POS[0], y - SHELTER_POS[1]) < 9:
                continue
            self._scatter_tree(rng, x, y)
        # a few forest trees (food deeper in the risky edge)
        for _ in range(4):
            self._scatter_tree(rng, rng.range(196.0, 238.0), rng.range(30.0, 160.0))
        # thorn thickets on the forest edge
        for i, (x, y) in enumerate([(196.0, 60.0), (206.0, 108.0), (193.0, 142.0)]):
            self.thickets.append(Thicket(i, x, y))
        # loose wood & stone
        mid = 0
        for _ in range(16):
            x = rng.range(180.0, 244.0) if rng.chance(0.6) else rng.range(90.0, 170.0)
            y = rng.range(24.0, 164.0)
            self.materials.append(Material(mid, "wood" if rng.chance(0.6) else "stone", x, y))
            mid += 1

    def _layout_fear_probe(self, rng: PRNG):
        """Probe world (GATE G3b): a fruit tree just beyond a thicket, so a
        hungry Fajr's own policy takes him across the thicket's edge.
        The world is arranged; his actions are never scripted."""
        # the bait tree beyond the thicket
        self._scatter_tree(rng, 226.0, 92.0, fruit=6)
        # his home meadow trees are far away and start bare in the probe
        self._scatter_tree(rng, 120.0, 60.0, fruit=0)
        self._scatter_tree(rng, 128.0, 130.0, fruit=0)
        self.thickets.append(Thicket(0, 208.0, 92.0, r=5.0))
        mid = 0
        for _ in range(6):
            self.materials.append(Material(mid, "wood", rng.range(100.0, 150.0), rng.range(40.0, 150.0)))
            mid += 1

    def _layout_food_probe(self, rng: PRNG):
        """Memory-assay world (GATE G3a, amendment 3): an arc of eight trees
        of which exactly ONE bears fruit. A naive forager must inspect tree
        after tree (win-shift maze logic); an experienced one goes straight
        to the tree his life taught him. Identical for both conditions."""
        arc = [(104.0, 64.0), (124.0, 56.0), (144.0, 55.0), (162.0, 60.0),
               (176.0, 72.0), (183.0, 90.0), (178.0, 108.0), (164.0, 120.0)]
        for i, (x, y) in enumerate(arc):
            t = self._scatter_tree(rng, x, y, fruit=0)
            if i == 5:
                t.fruit = 8
                t.max_fruit = 8
                t.regrow_need = 900
        mid = 0
        for _ in range(6):
            self.materials.append(Material(mid, "stone", rng.range(100.0, 170.0), rng.range(120.0, 160.0)))
            mid += 1

    # ---------- dynamics ----------
    def update(self, tick: int, light: float, rng: PRNG):
        # fruit regrowth per tree (default ~1 fruit per ~2.2 days)
        for t in self.trees:
            if t.fruit < t.max_fruit:
                t.regrow += 1
                if t.regrow >= t.regrow_need:
                    t.fruit += 1
                    t.regrow = 0
        # weather machine (seasons-lite bias: more rain as season_phase -> 0.5)
        self.weather_timer -= 1
        if self.weather_timer <= 0:
            r = rng.uniform()
            rain_bias = 0.28 + 0.25 * math.sin(self.season_phase * 2 * math.pi)
            if r < rain_bias:
                self.weather = "storm" if rng.chance(0.30) else "rain"
                self.weather_timer = rng.randint(120, 420)
            else:
                self.weather = "clear"
                self.weather_timer = rng.randint(240, 900)
        self.season_phase = (self.season_phase + 1.0 / (30 * 1440)) % 1.0

    # ---------- (de)serialisation ----------
    def to_dict(self):
        return {
            "trees": [{"id": t.id, "x": t.x, "y": t.y, "fruit": t.fruit, "regrow": t.regrow,
                       "max_fruit": t.max_fruit, "regrow_need": t.regrow_need}
                      for t in self.trees],
            "thickets": [{"id": t.id, "x": t.x, "y": t.y, "r": t.r} for t in self.thickets],
            "materials": [{"id": m.id, "kind": m.kind, "x": m.x, "y": m.y} for m in self.materials],
            "piles": [{"id": p.id, "x": p.x, "y": p.y, "fruit": p.fruit} for p in self.piles],
            "structures": list(self.structures),
            "weather": self.weather,
            "weather_timer": self.weather_timer,
            "season_phase": self.season_phase,
        }

    @classmethod
    def from_dict(cls, d):
        w = cls.__new__(cls)
        w.trees = []
        for td in d["trees"]:
            t = Tree(td["id"], td["x"], td["y"], td["fruit"])
            t.regrow = td["regrow"]
            t.max_fruit = td.get("max_fruit", 6)
            t.regrow_need = td.get("regrow_need", 3200)
            w.trees.append(t)
        w.thickets = [Thicket(t["id"], t["x"], t["y"], t["r"]) for t in d["thickets"]]
        w.materials = [Material(m["id"], m["kind"], m["x"], m["y"]) for m in d["materials"]]
        w.piles = [Pile(p["id"], p["x"], p["y"], p["fruit"]) for p in d.get("piles", [])]
        w.structures = [dict(s) for s in d["structures"]]
        w.weather = d["weather"]
        w.weather_timer = d["weather_timer"]
        w.season_phase = d["season_phase"]
        return w


def light_level(tick_of_day: int) -> float:
    """0..1 sunlight. Sunrise ~06:00 (360), sunset ~20:00 (1200)."""
    ang = (tick_of_day - 360) / 1440.0 * 2 * math.pi
    v = math.sin(ang)
    return max(0.0, min(1.0, v * 1.35))


def is_night(tick_of_day: int) -> bool:
    return light_level(tick_of_day) < 0.25
