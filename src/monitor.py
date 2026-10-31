"""Metacognitive monitor — Phase 2 (HOT-2, Perceptual Reality Monitoring).

A second-order system that reads FIRST-ORDER perceptual activations and
outputs, per scene object, a reliability scalar in [0,1]: "how likely is
this percept to be veridical?" It is a small logistic model over declared
features, learned ONLINE from the being's own verification events only:

  * a claim is VERIFIED when a later, cleaner percept (closer range, better
    light) or an action outcome (an eat attempt) settles it;
  * the claim a percept makes is its exact fruit COUNT (0 for absent);
    the label is whether that count matched the verification. (An
    earlier boundary semantics — "fruit > 0" — made labels ~always true
    and the monitor structurally unable to learn distrust; corrected
    during Phase-2 engineering, before the frozen battery.)

The monitor never sees world ground truth. (The battery harness separately
records reliability-vs-truth pairs for calibration scoring; that tape is
measurement, not learning.)

Lesion mode (battery P-B): reliability is a constant 0.5 — the monitor
exists but cannot discriminate, which is exactly the deficit being probed.

v2 repair (2026-10-23, PHASE2-GATE amendment): sitting 1's monitor reached
a reliability gap of 0.123 against the 0.15 bar, and a formal Bayes-ceiling
analysis of its 11 features (results/hot2_ceiling_analysis.md) put the
feature class's ceiling at ~0.147 — but the SAME analysis showed the class
was missing its strongest signal: whether a claim has been CONSISTENTLY
re-sensed. Two features were therefore added to the BEING (the bar and the
tape did not move): agree_streak (consecutive sensed ticks agreeing on the
same fruit count) and claim_age (how long the current count has stood).
The repaired model's ceiling on the same protocol is ~0.296 — headroom the
online learner can genuinely use. Claim registration density was also
raised (200 -> 60 ticks) so the learner trains on more, harder claims,
and the learning rate 0.05 -> 0.08.
"""
import math

FEATURES = ["pred_err", "smell", "sound", "temporal", "dist", "light",
            "storm", "generated", "novelty", "dev", "dark_far",
            "agree_streak", "claim_age"]
W0 = {  # declared initial weights (a sane prior, then learned)
    "pred_err": -0.9, "smell": 0.8, "sound": 0.4, "temporal": 1.1,
    "dist": -0.7, "light": 0.9, "storm": -0.8, "generated": -1.2,
    "novelty": -0.3, "dev": -0.6, "dark_far": -0.5,
    "agree_streak": 0.9, "claim_age": 0.3,
}
BIAS0 = 0.4
LR = 0.08
VERIFY_WINDOW = 1500    # ticks a claim may await verification (claims about
                        # a tree seen at night are settled when he next
                        # comes close in good light, or eats there)


def _sigmoid(z):
    if z < -30.0:
        return 0.0
    if z > 30.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


class Monitor:
    def __init__(self):
        self.w = dict(W0)
        self.bias = BIAS0
        self.lesion_constant = False
        self.pending = {}   # key -> {tick, reliability, fruit_claim, features}
        self.n_trained = 0
        self.last = {}      # key -> reliability this tick (for the viewer)
        self.tape = []      # (reliability, label) per settled claim — scoring only
        self._obs = {}      # key -> {fruit, since, streak}: the current
                            # claim's consistency history (v2 features)
        self._claim_since = {}  # key -> tick the current pending claim was
                            # FIRST registered (v2: claim_age's clock —
                            # decoupled from the 60-tick re-registration,
                            # per the ceiling analysis's recommendation)

    def features_of(self, obj, pred_err_obj, temporal_consistency, belief_fruit=None,
                    tick=0):
        f = obj.features
        # deviation from the standing belief: regrowth is slow, so a
        # percept that jumps far from what he believes is suspect
        # (perceptual reality monitoring uses exactly such context cues)
        dev = 0.0
        if belief_fruit is not None and obj.kind in ("tree", "pile"):
            dev = min(1.0, abs(obj.fruit - belief_fruit) / 3.0)
        # v2 consistency features: a count that has been re-sensed
        # unchanged, tick after tick, is a different epistemic object
        # from a first glimpse — the monitor now gets to see that.
        # claim_age runs on the claim's own clock (first registration of
        # the standing claim), not on percept noise or re-registration.
        streak = 0.0
        age = 0.0
        ob = self._obs.get(obj.key)
        if ob is not None and ob["fruit"] == obj.fruit:
            streak = min(1.0, ob["streak"] / 6.0)
        t0 = self._claim_since.get(obj.key)
        if t0 is not None:
            age = min(1.0, (tick - t0) / 300.0)
        return {
            "pred_err": min(1.0, pred_err_obj / 3.0),
            "smell": f["smell"], "sound": f["sound"],
            "temporal": temporal_consistency,
            "dist": min(1.0, f["dist"] / 40.0),
            "light": f["light"], "storm": f["storm"],
            "generated": f["generated"], "novelty": f["novelty"],
            "dev": dev,
            "dark_far": (1.0 - f["light"]) * min(1.0, f["dist"] / 40.0),
            "agree_streak": streak, "claim_age": age,
        }

    def reliability(self, feats):
        if self.lesion_constant:
            return 0.5
        z = self.bias + sum(self.w[k] * feats[k] for k in FEATURES)
        return _sigmoid(z)

    def assess(self, scene_objects, pred_err_by_key, temporal, tick, belief_fruit=None):
        """scene_objects: {key: SceneObject}. Returns {key: reliability}
        and registers verifiable claims (fruit-bearing kinds).

        A claim, once registered, PERSISTS until verified, expired, or
        200 ticks old: re-registering every tick would mean only the
        freshest, easiest claims are ever settled, and the monitor could
        never learn what a claim made in the dark is worth."""
        out = {}
        for key, o in scene_objects.items():
            # v2: update the claim-consistency history BEFORE features are
            # computed, so this tick's sensing counts toward its own claim
            if o.kind in ("tree", "pile") and not o.generated:
                ob = self._obs.get(key)
                if ob is not None and ob["fruit"] == o.fruit:
                    ob["streak"] += 1
                else:
                    self._obs[key] = {"fruit": o.fruit, "since": tick, "streak": 1}
            bf = belief_fruit.get(key) if belief_fruit else None
            feats = self.features_of(o, pred_err_by_key.get(key, 0.0),
                                     temporal.get(key, 0.0), bf, tick=tick)
            r = self.reliability(feats)
            out[key] = r
            if o.kind in ("tree", "pile") and not o.generated:
                old = self.pending.get(key)
                if old is None or tick - old["tick"] >= 60:
                    if old is None or old["fruit_count"] != o.fruit:
                        # a genuinely new claim starts its age clock;
                        # re-registering the SAME claim does not
                        self._claim_since[key] = tick
                    self.pending[key] = {
                        "tick": tick,
                        "reliability": r, "fruit_claim": 1 if o.fruit > 0 else 0,
                        "fruit_count": o.fruit, "features": feats,
                    }
        self.last = out
        return out

    def verify(self, key, actual_fruit, tick):
        """A cleaner observation or an action outcome settles a claim.
        actual_fruit: the fruit count as now reliably known (or None if the
        object proved absent -> fruit claim false)."""
        p = self.pending.pop(key, None)
        if p is None:
            return
        self._claim_since.pop(key, None)
        if tick - p["tick"] > VERIFY_WINDOW:
            return
        label = 1 if (actual_fruit is not None and actual_fruit == p["fruit_count"]) else 0
        # measurement tape: the reliability this claim carried vs how it
        # settled — the calibration population the monitor actually serves
        self.tape.append((round(p["reliability"], 3), label))
        if self.lesion_constant:
            return  # a lesioned monitor cannot learn either
        feats = p["features"]
        pred = self.reliability(feats)
        err = label - pred
        for k in FEATURES:
            self.w[k] += LR * err * feats[k]
        self.bias += LR * err
        self.n_trained += 1

    def expire(self, tick):
        for key in list(self.pending.keys()):
            if tick - self.pending[key]["tick"] > VERIFY_WINDOW:
                del self.pending[key]
                self._claim_since.pop(key, None)
        for key in list(self._obs.keys()):
            if tick - self._obs[key]["since"] > VERIFY_WINDOW and key not in self.pending:
                del self._obs[key]

    def to_dict(self):
        return {"w": dict(self.w), "bias": self.bias,
                "lesion_constant": self.lesion_constant,
                "pending": {k: dict(v) for k, v in self.pending.items()},
                "n_trained": self.n_trained, "last": dict(self.last),
                "obs": {k: dict(v) for k, v in self._obs.items()},
                "claim_since": dict(self._claim_since)}

    @classmethod
    def from_dict(cls, d):
        m = cls.__new__(cls)
        m.w = {**W0, **dict(d["w"])}
        m.bias = d["bias"]
        m.lesion_constant = d["lesion_constant"]
        m.pending = {k: dict(v) for k, v in d["pending"].items()}
        m.n_trained = d["n_trained"]
        m.last = dict(d["last"])
        m.tape = []
        m._obs = {k: dict(v) for k, v in d.get("obs", {}).items()}
        m._claim_since = dict(d.get("claim_since", {}))
        return m
