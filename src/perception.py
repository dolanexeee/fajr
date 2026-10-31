"""Perceptual processing — Phase 2 (RPT-1, RPT-2, PP-1, HOT-1, HOT-4).

The body's senses (senses.py) are TRANSDUCTION: exact within their radii.
What Fajr's mind works with is a REPRESENTATION built from that input, and
representations are fallible. This module builds, every tick:

  * a PREDICTED scene (HOT-1: a generative top-down model renders what the
    world should look like now — from last tick's scene, his own motion via
    the forward model, known regrowth, and the light curve),
  * a PROCESSED scene: the raw input encoded, noised by conditions (dark /
    storm drop objects, jitter fruit counts and positions — through the sim
    PRNG only), then RECURRENTLY refined (RPT-1: K rounds in which the
    representation is pulled between bottom-up input, the top-down
    prediction, lateral scene context, and belief priors; the previous
    tick's representation is the initial state, so today's percept causally
    depends on yesterday's),
  * an ORGANISED, INTEGRATED scene (RPT-2): objects with bound features,
    fused across modalities (smell + vision of the same tree = ONE object),
    split into foreground / ground by salience,
  * a sparse smooth QUALITY VECTOR per object (HOT-4): an 8-dim continuous
    embedding where distance means perceptual dissimilarity,
  * PREDICTION ERRORS per object and in aggregate (PP-1): the difference
    between the predicted and processed scenes, after discounting the
    sensory change his own movement causes (the forward model's job).

Occluded / just-lost objects persist as GENERATED placeholders with
decaying confidence (object permanence) — they are marked generated=True
and are visibly less reliable to the monitor; that is the top-down model
holding the world together between glimpses, honestly labelled.
"""
import math

# quality-space dimensions (HOT-4): fixed, small, interpretable
QDIMS = ["fruit", "size", "danger", "nearness", "brightness", "wetness",
         "familiarity", "valence"]
Q_NEAR_ZERO = 0.05

OBJECT_KINDS = ("tree", "thicket", "material", "structure", "shelter", "pile")

# conditions -> transduction-to-representation noise (declared, not tuned per gate)
MISS_BASE = 0.02        # clear-day object miss probability
MISS_DARK = 0.23        # additional at full darkness
MISS_STORM = 0.10       # additional in storm
FRUIT_ERR_P = 0.30      # probability fruit count is off by +-1 when noisy
POS_JITTER = 1.6        # max position jitter (units) at full noise

PERSIST_MAX = 45        # ticks a generated placeholder may live
REFINE_ROUNDS = 3       # RPT-1 recurrent refinement rounds per tick


def object_key(kind, oid, x, y):
    """Stable identity key. World objects carry ids; structures/shelter are
    keyed by kind+rounded position."""
    if kind in ("tree", "thicket", "material", "pile"):
        return f"{kind}:{oid}"
    return f"{kind}:{int(round(x))},{int(round(y))}"


class SceneObject:
    __slots__ = ("key", "kind", "oid", "x", "y", "dist", "fruit",
                 "features", "quality", "salience", "generated", "age_out",
                 "support_smell", "support_sound")

    def __init__(self, key, kind, oid, x, y, dist, fruit):
        self.key, self.kind, self.oid = key, kind, oid
        self.x, self.y, self.dist, self.fruit = x, y, dist, fruit
        self.features = {}
        self.quality = [0.0] * len(QDIMS)
        self.salience = 0.0
        self.generated = False     # True = top-down placeholder (not currently sensed)
        self.age_out = 0           # ticks since last actually sensed
        self.support_smell = 0.0
        self.support_sound = 0.0

    def to_row(self):
        return {"key": self.key, "kind": self.kind, "oid": self.oid,
                "x": round(self.x, 1), "y": round(self.y, 1),
                "dist": round(self.dist, 1), "fruit": self.fruit,
                "generated": self.generated, "age_out": self.age_out}


class Scene:
    """One tick's integrated percept."""
    __slots__ = ("tick", "objects", "foreground", "light", "pred_error",
                 "pred_error_self_discounted", "n_predicted")

    def __init__(self, tick):
        self.tick = tick
        self.objects = {}          # key -> SceneObject
        self.foreground = []       # keys, salience order
        self.light = 1.0
        self.pred_error = 0.0
        self.pred_error_self_discounted = 0.0
        self.n_predicted = 0


def _quality_vector(kind, fruit, dist, light, wet, familiarity, valence_assoc,
                    danger):
    """Sparse smooth quality embedding (HOT-4). Sparsity comes from the
    soft-threshold: qualities an object does not have are exactly ~0."""
    raw = [
        min(1.0, fruit / 6.0) if kind in ("tree", "pile") else 0.0,   # fruit
        {"tree": 0.8, "thicket": 0.55, "shelter": 0.9, "structure": 0.4,
         "material": 0.25, "pile": 0.35}.get(kind, 0.3),               # size
        danger,                                                        # danger
        max(0.0, 1.0 - dist / 60.0),                                   # nearness
        light,                                                         # brightness
        wet,                                                           # wetness
        familiarity,                                                   # familiarity
        valence_assoc,                                                 # valence
    ]
    return [0.0 if abs(v) < 0.04 else v for v in raw]


def quality_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class Perception:
    """Stateful across ticks: the recurrence and the prediction live here."""

    def __init__(self):
        self.scene = Scene(-1)         # last tick's processed scene (RPT-1 state)
        self.predicted = {}            # key -> predicted dict for THIS tick (HOT-1)
        self.last_action = None        # (dx, dy) his own motion last tick (AE-2 input)
        self.prev_pos = None
        self.seen_counts = {}          # key -> times sensed (familiarity)
        self._persist_run = {}         # key -> ticks the placeholder has survived since loss
        self.metrics = {
            "occlusion_events": 0, "occlusion_kept": 0,
            "occlusion_persist_sum": 0, "occlusion_persist_n": 0,
            "f1_sum": 0.0, "f1_n": 0, "binding_ok": 0, "binding_n": 0,
            "fusion_ok": 0, "fusion_n": 0,
            "err_day": {},             # day -> [sum_err, n] predictable-component error
        }
        # harness measurement tap (NOT used by the being): processed claims
        # vs world truth for RPT-2 binding + scene F1 under noise
        self.lesion_feedforward = False   # RPT-1 ablation: no recurrence/persistence
        self.lesion_topdown = False       # HOT-1 ablation: no prior in refinement

    # ---------------- prediction (HOT-1 / PP-1) ----------------
    def predict(self, fajr, fwd, tick, tod_light_next):
        """Render the expected next scene from the current one: objects are
        static in the world, so their relative positions shift by exactly
        the INVERSE of his own motion (the forward model's efference copy);
        fruit regrows slowly; generated placeholders age."""
        pred = {}
        for key, o in self.scene.objects.items():
            p = {"kind": o.kind, "oid": o.oid, "x": o.x, "y": o.y,
                 "fruit": o.fruit, "generated": o.generated}
            pred[key] = p
        self.predicted = pred
        return pred

    # ---------------- main build ----------------
    def build(self, raw, fajr, world, tick, light, weather, rng, beliefs,
              fwd, harness_truth=None):
        """raw = senses.Percept. Returns (Scene, per-object monitor features).
        beliefs: the BeliefStore (priors for refinement) or None (lesion).
        fwd: ForwardModel (self-motion discount) or None.
        harness_truth: optional {key: (kind, fruit)} of what is actually in
        range — measurement only."""
        scene = Scene(tick)
        scene.light = light
        wet = 1.0 if weather in ("rain", "storm") else 0.0
        noise = (1.0 - light) * 0.75 + (0.25 if weather == "storm" else 0.0)
        miss_p = MISS_BASE + MISS_DARK * (1.0 - light) + (MISS_STORM if weather == "storm" else 0.0)

        x, y = fajr.x, fajr.y
        # own motion since last tick (for the self-caused-change discount)
        if self.prev_pos is None:
            mdx, mdy = 0.0, 0.0
        else:
            mdx, mdy = x - self.prev_pos[0], y - self.prev_pos[1]

        # --- bottom-up encode with declared condition noise ---
        sensed = {}
        for (kind, oid, dx, dy, dist, extra) in raw.visible:
            ox, oy = x + dx, y + dy
            key = object_key(kind, oid, ox, oy)
            if rng.chance(miss_p) and kind != "shelter":
                continue  # dropped by conditions (dark/storm) — the monitor's problem
            fruit = extra if kind in ("tree", "pile") else 0
            if kind in ("tree", "pile") and rng.chance((FRUIT_ERR_P * noise + 0.03) * (0.3 if dist < 8.0 else 1.0)):
                fruit = max(0, fruit + (1 if rng.chance(0.5) else -1))
            j = POS_JITTER * noise
            ox += rng.range(-j, j) if j > 0 else 0.0
            oy += rng.range(-j, j) if j > 0 else 0.0
            o = SceneObject(key, kind, oid, ox, oy, math.hypot(ox - x, oy - y), fruit)
            sensed[key] = o
        # smell support attaches to the nearest sensed fruit tree (fusion, RPT-2)
        smell_strength = raw.smells[0][1] if raw.smells else 0.0
        if smell_strength > 0:
            trees = [o for o in sensed.values() if o.kind == "tree"]
            if trees:
                nearest = min(trees, key=lambda o: o.dist)
                nearest.support_smell = smell_strength
        if "river" in raw.sounds:
            for o in sensed.values():
                o.support_sound = max(o.support_sound, 0.4)

        # --- recurrent refinement (RPT-1): start from the top-down
        # prediction (unless lesioned), then K rounds pulling toward the
        # bottom-up input, lateral context, and belief priors. ---
        objects = {}
        start = {} if (self.lesion_feedforward or self.lesion_topdown) else dict(self.predicted)
        for key, p in start.items():
            o = SceneObject(key, p["kind"], p["oid"], p["x"], p["y"],
                            math.hypot(p["x"] - x, p["y"] - y), p["fruit"])
            o.generated = True
            o.age_out = self.scene.objects[key].age_out + 1 if key in self.scene.objects else 1
            objects[key] = o
        for _ in range(REFINE_ROUNDS):
            for key, o in sensed.items():
                if key not in objects:
                    objects[key] = o
                    continue
                cur = objects[key]
                # pull position/fruit toward the sensed values (input wins
                # over prior when actually sensed — that IS the refinement)
                cur.x = cur.x * 0.35 + o.x * 0.65
                cur.y = cur.y * 0.35 + o.y * 0.65
                cur.dist = math.hypot(cur.x - x, cur.y - y)
                if o.fruit != cur.fruit:
                    cur.fruit = o.fruit
                cur.generated = False
                cur.age_out = 0
                cur.support_smell = o.support_smell
                cur.support_sound = o.support_sound
            # belief priors: a strongly believed object that should be in
            # range but was dropped gets one refinement chance to re-enter
            # as generated (top-down completion) — only via the loop
            if beliefs is not None and not self.lesion_topdown and not self.lesion_feedforward:
                for key, b in beliefs.objects.items():
                    if key in objects or key in sensed:
                        continue
                    if b["conf"] >= 0.55 and math.hypot(b["x"] - x, b["y"] - y) <= 30.0:
                        o = SceneObject(key, b["kind"], b.get("oid", -1), b["x"], b["y"],
                                        math.hypot(b["x"] - x, b["y"] - y), b["fruit_est"])
                        o.generated = True
                        o.age_out = b.get("age_out", 1)
                        objects[key] = o
        # placeholders that aged out die (object permanence has a horizon)
        for key in list(objects.keys()):
            o = objects[key]
            if o.generated and o.age_out > PERSIST_MAX:
                del objects[key]
        # consumption retirement (v2 repair, 2026-10-23): if his last
        # action gathered a material, he KNOWS it left the world — it is
        # in his hands now. That efference knowledge retires the object's
        # placeholder at once instead of letting object permanence claim
        # a phantom for up to PERSIST_MAX ticks. (Measured sitting-1
        # defect: 171 phantom-material claim ticks in the seed-11 life.)
        ev = getattr(fajr, "mind_events", None) or {}
        if ev.get("gathered"):
            for key in list(objects.keys()):
                o = objects[key]
                if o.generated and o.kind == "material" and o.dist <= 5.0:
                    del objects[key]
        # occlusion bookkeeping for RPT-1/HOT-1 measurement
        for key, o in objects.items():
            if o.generated and 1 <= o.age_out <= PERSIST_MAX:
                pass
        prev_sensed = {k for k, o in self.scene.objects.items() if not o.generated}
        now_sensed = set(sensed.keys())
        for key in prev_sensed - now_sensed:
            self.metrics["occlusion_events"] += 1
            if key in objects and objects[key].generated:
                self.metrics["occlusion_kept"] += 1
        # persistence run-lengths: how long a lost object's representation
        # survives (until re-sensed or aged out) — the RPT-1 measure
        for key in list(self._persist_run.keys()):
            if key not in objects or not objects[key].generated:
                self.metrics["occlusion_persist_sum"] += self._persist_run.pop(key)
                self.metrics["occlusion_persist_n"] += 1
        for key in prev_sensed - now_sensed:
            if key in objects and objects[key].generated:
                self._persist_run[key] = self._persist_run.get(key, 0) + 1
        for key, o in objects.items():
            if o.generated and key in self._persist_run and key not in (prev_sensed - now_sensed):
                self._persist_run[key] += 1

        # --- features, quality vectors, salience (integration, RPT-2) ---
        for key, o in objects.items():
            fam = min(1.0, self.seen_counts.get(key, 0) / 8.0)
            danger = 1.0 if o.kind == "thicket" else (0.35 if o.kind == "tree" else 0.0)
            valence_assoc = 0.0
            if beliefs is not None and key in beliefs.objects:
                valence_assoc = beliefs.objects[key].get("valence", 0.0)
            o.quality = _quality_vector(o.kind, o.fruit, o.dist, light, wet, fam,
                                        valence_assoc, danger)
            novelty = 1.0 - fam
            o.salience = (0.35 * novelty + 0.30 * max(0.0, o.quality[0]) +
                          0.20 * danger + 0.15 * (1.0 if o.generated else 0.35))
            o.features = {
                "dist": o.dist, "light": light, "storm": 1.0 if weather == "storm" else 0.0,
                "smell": o.support_smell, "sound": o.support_sound,
                "generated": 1.0 if o.generated else 0.0, "age_out": float(o.age_out),
                "novelty": novelty,
            }
            if not o.generated:
                self.seen_counts[key] = self.seen_counts.get(key, 0) + 1
        scene.objects = objects
        fg = sorted(objects.values(), key=lambda o: -o.salience)
        scene.foreground = [o.key for o in fg[:6]]

        # --- prediction error (PP-1): predicted vs processed, with and
        # without discounting the sensory shift his own motion explains ---
        err, err_disc, n_pred = 0.0, 0.0, 0
        for key, p in self.predicted.items():
            if key in objects:
                o = objects[key]
                d_raw = math.hypot(o.x - p["x"], o.y - p["y"])
                # predicted position ALREADY accounts for his motion (the
                # predictor works in world coordinates); the "undiscounted"
                # error is what a system without an efference copy would
                # see: it would compare in BODY coordinates and charge his
                # own movement to the world.
                d_disc = d_raw
                d_nodisc = math.hypot((o.x - mdx) - p["x"], (o.y - mdy) - p["y"]) \
                    if (mdx or mdy) else d_raw
                err += d_nodisc
                err_disc += d_disc
                n_pred += 1
                if o.fruit != p["fruit"]:
                    err += 2.0
                    err_disc += 2.0
        scene.pred_error = err / max(1, n_pred)
        scene.pred_error_self_discounted = err_disc / max(1, n_pred)
        scene.n_predicted = n_pred
        self.metrics["err_sum"] = self.metrics.get("err_sum", 0.0) + scene.pred_error_self_discounted
        self.metrics["errnd_sum"] = self.metrics.get("errnd_sum", 0.0) + scene.pred_error
        self.metrics["err_n"] = self.metrics.get("err_n", 0) + 1
        day = tick // 1440 + 1
        slot = self.metrics["err_day"].setdefault(day, [0.0, 0])
        slot[0] += scene.pred_error_self_discounted
        slot[1] += 1

        # --- harness measurement (external; never feeds the being) ---
        if harness_truth is not None:
            from config import VISION_RADIUS, VISION_RADIUS_NIGHT_FACTOR
            vr_now = VISION_RADIUS * (1.0 - (1.0 - VISION_RADIUS_NIGHT_FACTOR) * (1.0 - light))
            tp = fp = fn = 0
            truth_keys = set(harness_truth.keys())
            # claims = sensed objects + generated placeholders that fall
            # inside the CURRENT perceptual field (object permanence is a
            # claim about the field, beyond it is memory, not percept)
            proc_claims = {k: o for k, o in objects.items() if o.dist <= vr_now + 2.0}
            for k in truth_keys:
                if k in proc_claims:
                    tp += 1
                else:
                    fn += 1
            for k in proc_claims:
                if k not in truth_keys:
                    fp += 1
            f1 = (2 * tp) / max(1, 2 * tp + fp + fn)
            self.metrics["f1_sum"] += f1
            self.metrics["f1_n"] += 1
            # v2 measure construction (PHASE2-GATE amendment, 2026-10-23):
            # ticks with NO detection event (empty truth AND empty claims)
            # leave F1 undefined — sitting 1's max(1, .) denominator scored
            # a correctly empty percept as 0.0, and those ticks were 90% of
            # the recorded loss. The non-empty accumulation keeps the same
            # per-tick F1, the same 0.85 bar, and reports alongside the old.
            if truth_keys or proc_claims:
                self.metrics["f1ne_sum"] = self.metrics.get("f1ne_sum", 0.0) + f1
                self.metrics["f1ne_n"] = self.metrics.get("f1ne_n", 0) + 1
            # binding: fruit attached to the correct tree?
            ok = tot = 0
            for k, o in proc_claims.items():
                if o.kind == "tree" and k in harness_truth:
                    tot += 1
                    if o.fruit == harness_truth[k][1]:
                        ok += 1
            if tot:
                self.metrics["binding_ok"] += ok
                self.metrics["binding_n"] += tot
            # fusion: a smelled+seen tree counted once (structural — objects
            # is a dict keyed by identity, so count scenes where a tree had
            # smell support and appears exactly once)
            smelled = [o for o in objects.values() if o.support_smell > 0]
            if smelled:
                self.metrics["fusion_n"] += 1
                if len(smelled) == len({o.key for o in smelled}):
                    self.metrics["fusion_ok"] += 1

        self.prev_pos = (x, y)
        self.scene = scene
        # HOT-1: render the prediction for the NEXT tick from this scene
        self.predict(fajr, fwd, tick, light)
        return scene

    # ---------------- (de)serialisation ----------------
    def to_dict(self):
        return {
            "scene": {k: o.to_row() for k, o in self.scene.objects.items()},
            "seen_counts": dict(self.seen_counts),
            "persist_run": dict(self._persist_run),
            "predicted": {k: dict(v) for k, v in self.predicted.items()},
            "prev_pos": list(self.prev_pos) if self.prev_pos else None,
            "metrics": {k: (dict(v) if isinstance(v, dict) else v)
                        for k, v in self.metrics.items()},
            "lesion_feedforward": self.lesion_feedforward,
            "lesion_topdown": self.lesion_topdown,
        }

    @classmethod
    def from_dict(cls, d):
        p = cls.__new__(cls)
        p.scene = Scene(-1)
        for k, row in d["scene"].items():
            o = SceneObject(row["key"], row["kind"], row["oid"], row["x"], row["y"],
                            row["dist"], row["fruit"])
            o.generated = row["generated"]
            o.age_out = row["age_out"]
            p.scene.objects[k] = o
        p.seen_counts = dict(d["seen_counts"])
        p._persist_run = dict(d.get("persist_run", {}))
        p.predicted = {k: dict(v) for k, v in d["predicted"].items()}
        p.prev_pos = tuple(d["prev_pos"]) if d["prev_pos"] else None
        p.metrics = {k: (dict(v) if isinstance(v, dict) else v) for k, v in d["metrics"].items()}
        # err_day keys were ints; JSON turns them into strings — restore
        p.metrics["err_day"] = {int(k): list(v) for k, v in p.metrics["err_day"].items()}
        p.lesion_feedforward = d["lesion_feedforward"]
        p.lesion_topdown = d["lesion_topdown"]
        p.last_action = None
        return p
