"""Belief state — Phase 2 (HOT-3, scoped to his world, named as scoped).

A store of what Fajr currently takes the world to be, separate from both
raw perception and from Phase-1's associative structures:

  objects[key] = {kind, oid, x, y, fruit_est, fruit_var, danger,
                  conf, valence, last_seen, updates, low_rel_updates}
  vars         = small set of world/self propositions with confidences
                 (water_at_river, home_is_shelter, thorn_hurts, ...)

Beliefs are updated from the processed scene WEIGHTED BY MONITOR
RELIABILITY — that weighting is the HOT-3 disposition made mechanical:
a percept the monitor distrusts moves beliefs only slightly, and instead
raises an EPISTEMIC flag (needs re-observation) that the attention system
can act on. Beliefs — not ground truth — are what goal evaluation reads.

Scope honesty: belief formation here is general only within his world's
ontology (objects, places, his body). It is not open-ended belief
formation; the gate scores it SCOPED by design (PHASE2-GATE I-HOT-3).
"""
import math

LOW_REL = 0.4
HIGH_REL = 0.7


class BeliefStore:
    def __init__(self):
        self.objects = {}
        self.vars = {
            "water_at_river": {"p": 0.5, "support": 0},
            "home_is_shelter": {"p": 0.5, "support": 0},
            "thorn_hurts": {"p": 0.5, "support": 0},
            "night_is_dark": {"p": 0.9, "support": 0},
        }
        self.epistemic_flags = {}   # key -> tick when flagged for re-observation
        self.stats = {"w_low_sum": 0.0, "w_low_n": 0, "w_high_sum": 0.0, "w_high_n": 0,
                      "revisions": 0, "reobserve_marks": 0}

    # ---------------- update from a processed scene ----------------
    def update_from_scene(self, scene, reliability, tick, pred_err_mag):
        """pred_err_mag: |scene prediction error| (PP-1 coupling: bigger
        surprise, bigger updates where reliability allows)."""
        for key, o in scene.objects.items():
            r = reliability.get(key, 0.5)
            # v4 flag lifecycle (GATE v4-2): a CLEAN re-sensing of a
            # flagged object answers the epistemic question the flag
            # posed — the flag clears. (Flags previously never expired:
            # the old observe branch re-targeted the alphabetically
            # first flag forever, one of its two fatal defects.)
            if not o.generated and r >= HIGH_REL and key in self.epistemic_flags:
                del self.epistemic_flags[key]
            # update weight: reliability-gated, surprise-amplified.
            # v3 precision weighting (repair round 2, GATE v3-1): the
            # surprise term is itself scaled BY reliability. Sitting 2
            # measured why: with the unscaled term, the low-reliability
            # class concentrates on the darkest/stormiest ticks — exactly
            # where aggregate prediction error is largest — so raw
            # surprise inflated distrusted updates and the low/high mean
            # weight ratio ROSE to 0.5341 as the monitor improved.
            # Precision weighting is the principled form: a percept moves
            # belief in proportion to its calibrated precision, surprise
            # included — w = (0.08 + 0.55r) * (1 + min(1.5, pred_err) * r).
            # v4 calibration note (dev, architecture round 3): a
            # steepened curve (0.05 + 0.62 r^1.6) was tried and REJECTED
            # by dev measurement — it improved the HOT-3(b) margin
            # (0.3675 -> 0.2155, margin it did not need) while P-B's
            # dynamic-truth median fell 1.309 -> 1.039 with a seed
            # inversion: amplifying the monitor's reliability contrast
            # also amplifies its estimation noise under a moving truth.
            # The v3 curve stands.
            w = (0.08 + 0.55 * r) * (1.0 + min(1.5, pred_err_mag) * r)
            if r < LOW_REL:
                self.stats["w_low_sum"] += w
                self.stats["w_low_n"] += 1
            elif r >= HIGH_REL:
                self.stats["w_high_sum"] += w
                self.stats["w_high_n"] += 1
            b = self.objects.get(key)
            if b is None:
                self.objects[key] = {
                    "kind": o.kind, "oid": o.oid, "x": o.x, "y": o.y,
                    "fruit_est": float(o.fruit), "fruit_var": 1.0,
                    "danger": 1.0 if o.kind == "thicket" else 0.0,
                    "conf": r, "valence": 0.0, "last_seen": tick,
                    "updates": 1, "low_rel_updates": 1 if r < LOW_REL else 0,
                    "age_out": 0,
                }
                continue
            old_fruit = b["fruit_est"]
            b["x"] = b["x"] * (1 - w) + o.x * w
            b["y"] = b["y"] * (1 - w) + o.y * w
            if o.kind in ("tree", "pile"):
                b["fruit_est"] = b["fruit_est"] * (1 - w) + float(o.fruit) * w
                b["fruit_var"] = max(0.05, b["fruit_var"] * (1 - w) + (float(o.fruit) - b["fruit_est"]) ** 2 * w)
            if abs(b["fruit_est"] - old_fruit) > 0.8:
                self.stats["revisions"] += 1
            # v4 revision tracking (GATE v4-2, surprise channel): a
            # single update that moves the fruit estimate by >= 1.5 is
            # a belief REVISION — recorded on the belief itself so the
            # epistemic action system can price verifying it.
            if o.kind in ("tree", "pile") and abs(b["fruit_est"] - old_fruit) >= 1.5:
                b["last_rev_tick"] = tick
                b["last_rev_mag"] = round(abs(b["fruit_est"] - old_fruit), 3)
                # the revising percept's own reliability: a revision
                # wrought by a trustworthy percept is knowledge, by an
                # untrustworthy one a new question (v4 epistemic
                # pricing reads this — GATE v4-2 calibration)
                b["last_rev_rel"] = round(r, 3)
            b["conf"] = max(b["conf"] * 0.92, r) if not o.generated else b["conf"] * 0.985
            b["last_seen"] = tick if not o.generated else b["last_seen"]
            b["age_out"] = 0 if not o.generated else b.get("age_out", 0) + 1
            b["updates"] += 1
            if r < LOW_REL:
                b["low_rel_updates"] += 1
                # the disposition: distrusted percepts about FOOD raise a
                # re-observation flag instead of quietly rewriting belief
                if o.kind in ("tree", "pile") and key not in self.epistemic_flags:
                    self.epistemic_flags[key] = tick
                    self.stats["reobserve_marks"] += 1
        # age beliefs about objects not in the scene at all
        for key, b in self.objects.items():
            if key not in scene.objects:
                b["age_out"] = b.get("age_out", 0) + 1
                b["conf"] *= 0.9995

    def set_var(self, name, p, support_bump=1):
        v = self.vars.setdefault(name, {"p": 0.5, "support": 0})
        v["p"] = max(0.0, min(1.0, p))
        v["support"] += support_bump

    def revise_var(self, name, evidence_p, weight):
        v = self.vars.setdefault(name, {"p": 0.5, "support": 0})
        v["p"] = v["p"] * (1 - weight) + evidence_p * weight
        v["support"] += 1

    # ---------------- queries used by modules ----------------
    def fruit_beliefs(self):
        """{key: belief} for fruit-bearing objects with fruit_est above the
        action threshold — what his goal system may aim at."""
        return {k: b for k, b in self.objects.items()
                if b["kind"] in ("tree", "pile") and b["fruit_est"] >= 0.6}

    def mean_update_weights(self):
        low = self.stats["w_low_sum"] / max(1, self.stats["w_low_n"])
        high = self.stats["w_high_sum"] / max(1, self.stats["w_high_n"])
        return low, high

    def to_dict(self):
        return {"objects": {k: dict(v) for k, v in self.objects.items()},
                "vars": {k: dict(v) for k, v in self.vars.items()},
                "epistemic_flags": dict(self.epistemic_flags),
                "stats": dict(self.stats)}

    @classmethod
    def from_dict(cls, d):
        b = cls.__new__(cls)
        b.objects = {k: dict(v) for k, v in d["objects"].items()}
        b.vars = {k: dict(v) for k, v in d["vars"].items()}
        b.epistemic_flags = dict(d["epistemic_flags"])
        b.stats = dict(d["stats"])
        return b
