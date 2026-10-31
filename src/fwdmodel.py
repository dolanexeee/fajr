"""Forward model — Phase 2 (AE-2, embodiment).

A learned tabular transition model: p(next observation features | context,
action), where context = (intent kind, time-of-day bucket, weather,
in_shelter) and the predicted features are the ones his body can check next
tick: will water be perceptible, will a fruit-bearing object be perceptible,
will a hazard contact occur (prick/cold shock), expected drive deltas,
expected displacement. Counts and running means per context, with a global
fallback for unseen contexts — standard, deterministic, no RNG.

Two uses, as the indicator requires:
  * PERCEPTION: it supplies the efference-copy expectation the perception
    module uses to discount self-caused sensory change (objects are static;
    their apparent shift is his own motion — predicted, not charged to the
    world as prediction error).
  * CONTROL: the planning module scores candidate intentions by predicted
    outcomes (expected drive relief minus predicted hazard risk) blended
    with the Phase-1 heuristic scores.

It is deliberately small: a table, not a neural world model. PHASE2-REPORT
says so.
"""
import math

TOD_BUCKETS = 6
CAT_FEATURES = ["see_water", "see_food", "hazard"]
NUM_FEATURES = ["d_hunger", "d_thirst", "d_sleep", "moved"]


def context_key(intent_kind, tod, weather, in_shelter):
    bucket = min(TOD_BUCKETS - 1, int(tod / 1440 * TOD_BUCKETS))
    return (intent_kind or "idle", bucket, weather, 1 if in_shelter else 0)


class ForwardModel:
    def __init__(self):
        self.cat = {}    # key -> {feat: [hits, total]}
        self.num = {}    # key -> {feat: [sum, n]}
        self.gcat = {f: [0, 0] for f in CAT_FEATURES}
        self.gnum = {f: [0.0, 0] for f in NUM_FEATURES}
        self.pending = None   # (key, ) awaiting outcome
        self.metrics = {"pred_n": 0, "pred_correct": 0, "base_correct": 0,
                        "by_day": {}}   # day -> [correct, n] for battery

    # ---------------- learning ----------------
    def expect(self, key):
        """Store the context of the action just chosen; the outcome is
        recorded next tick via observe_outcome."""
        self.pending = key

    def observe_outcome(self, outcome, day):
        """outcome: {see_water: bool, see_food: bool, hazard: bool,
        d_hunger: float, ...}. Also scores yesterday's prediction."""
        if self.pending is None:
            return
        key = self.pending
        self.pending = None
        # score the prediction that was made for this context
        pred = self.predict(key)
        correct = 0
        base_correct = 0
        for f in CAT_FEATURES:
            p = pred[f]
            actual = 1 if outcome[f] else 0
            if (p >= 0.5) == (actual == 1):
                correct += 1
            # majority-class baseline from the running global rates
            hits, tot = self.gcat[f]
            base_pred = 1 if (tot > 0 and hits / tot >= 0.5) else 0
            if base_pred == actual:
                base_correct += 1
        self.metrics["pred_n"] += len(CAT_FEATURES)
        self.metrics["pred_correct"] += correct
        self.metrics["base_correct"] = self.metrics.get("base_correct", 0) + base_correct
        slot = self.metrics["by_day"].setdefault(day, [0, 0])
        slot[0] += correct
        slot[1] += len(CAT_FEATURES)
        # learn
        c = self.cat.setdefault(key, {f: [0, 0] for f in CAT_FEATURES})
        n = self.num.setdefault(key, {f: [0.0, 0] for f in NUM_FEATURES})
        for f in CAT_FEATURES:
            c[f][1] += 1
            self.gcat[f][1] += 1
            if outcome[f]:
                c[f][0] += 1
                self.gcat[f][0] += 1
        for f in NUM_FEATURES:
            n[f][0] += outcome[f]
            n[f][1] += 1
            self.gnum[f][0] += outcome[f]
            self.gnum[f][1] += 1

    # ---------------- prediction ----------------
    def predict(self, key):
        out = {}
        c = self.cat.get(key)
        n = self.num.get(key)
        for f in CAT_FEATURES:
            if c is not None and c[f][1] >= 4:
                out[f] = c[f][0] / c[f][1]
            elif self.gcat[f][1] > 0:
                out[f] = self.gcat[f][0] / self.gcat[f][1]
            else:
                out[f] = 0.5
        for f in NUM_FEATURES:
            if n is not None and n[f][1] >= 4:
                out[f] = n[f][0] / n[f][1]
            elif self.gnum[f][1] > 0:
                out[f] = self.gnum[f][0] / self.gnum[f][1]
            else:
                out[f] = 0.0
        return out

    def accuracy_by_day(self, day):
        slot = self.metrics["by_day"].get(day)
        if not slot or slot[1] == 0:
            return None
        return slot[0] / slot[1]

    def to_dict(self):
        return {
            "cat": {"|".join(map(str, k)): v for k, v in self.cat.items()},
            "num": {"|".join(map(str, k)): v for k, v in self.num.items()},
            "gcat": self.gcat, "gnum": self.gnum,
            "pending": list(self.pending) if self.pending else None,
            "metrics": {"pred_n": self.metrics["pred_n"],
                        "pred_correct": self.metrics["pred_correct"],
                        "base_correct": self.metrics.get("base_correct", 0),
                        "by_day": {str(k): list(v) for k, v in self.metrics["by_day"].items()}},
        }

    @classmethod
    def from_dict(cls, d):
        m = cls.__new__(cls)

        def parse(k):
            kind, bucket, weather, shelter = k.split("|")
            return (kind, int(bucket), weather, int(shelter))
        m.cat = {parse(k): {f: list(vv) for f, vv in v.items()} for k, v in d["cat"].items()}
        m.num = {parse(k): {f: list(vv) for f, vv in v.items()} for k, v in d["num"].items()}
        m.gcat = {f: list(v) for f, v in d["gcat"].items()}
        m.gnum = {f: list(v) for f, v in d["gnum"].items()}
        m.pending = parse("|".join(map(str, d["pending"]))) if d["pending"] else None
        m.metrics = {"pred_n": d["metrics"]["pred_n"],
                     "pred_correct": d["metrics"]["pred_correct"],
                     "base_correct": d["metrics"].get("base_correct", 0),
                     "by_day": {int(k): list(v) for k, v in d["metrics"]["by_day"].items()}}
        return m
