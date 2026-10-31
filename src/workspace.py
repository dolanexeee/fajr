"""Global workspace + attention — Phase 2 (GWT-1..4, AST-1).

A LIDA-flavoured cognitive cycle, run once per tick:

  Stage 1 — PERCEPTUAL COMPETITION. Six specialist modules, each with its
  own state, independently propose CONTENT coalitions from the same inputs
  (scene, drives, affect, beliefs, memory, last broadcast):
      vision, interoception, affect, memory, beliefs, motor.
  The workspace holds at most CAPACITY=3 coalitions — a real bottleneck:
  typically 6-10 proposals compete, most are excluded every cycle.
  Winners are BROADCAST: the same coalition list is delivered to every
  module (receipts counted per module — the GWT-3 audit) and conditions
  the next cycle (memory cueing reads it; planning's goal order follows
  the needs it contains; believed objects in it consolidate).

  Stage 2 — ACTION SELECTION (in mind.py, conditioned on the broadcast).

  Between the stages, GWT-4: if the broadcast shows a need whose target
  information is missing, the attention CONTROLLER queries modules IN
  SUCCESSION — memory (recall), then beliefs (is it reliable?), then
  planning (evaluate) — each answer entering the workspace before the
  decision. The sequence is driven by the workspace's current state.

AST-1 — the attention SCHEMA is a learned linear-softmax model that
PREDICTS next cycle's winning module from current features. It is
consulted BEFORE competition: if it predicts attention will be captured
away from an active goal, goal-serving modules are boosted. That is a
predictive model of attention used for control (beyond the non-predictive
schema of Wilterson & Graziano 2021) — scoped to his attention in his
world, and scored SCOPED (PHASE2-GATE I-AST-1).
"""
import math

MODULES = ["vision", "interoception", "affect", "memory", "beliefs", "motor"]
CAPACITY = 3


class Coalition:
    __slots__ = ("source", "kind", "salience", "target_key", "x", "y", "payload")

    def __init__(self, source, kind, salience, target_key=None, x=None, y=None, payload=None):
        self.source, self.kind, self.salience = source, kind, salience
        self.target_key, self.x, self.y = target_key, x, y
        self.payload = payload

    def row(self):
        return {"source": self.source, "kind": self.kind,
                "salience": round(self.salience, 3), "target": self.target_key}


# ---------------------------------------------------------------- schema
SCHEMA_FEATURES = 10   # [bias handled separately] see _schema_features

class AttentionSchema:
    """Predictive model of his own attention (AST-1, scoped)."""

    def __init__(self):
        self.W = [[0.0] * SCHEMA_FEATURES for _ in MODULES]
        self.b = [0.0] * len(MODULES)
        self.lr = 0.06
        self.enabled = True       # battery lesion: schema off (no boost, still logs)
        self.metrics = {"n": 0, "correct": 0, "majority_correct": 0,
                        "class_counts": {m: 0 for m in MODULES}}

    def _softmax(self, z):
        m = max(z)
        e = [math.exp(v - m) for v in z]
        s = sum(e)
        return [v / s for v in e]

    def predict(self, feats):
        z = [sum(w * f for w, f in zip(self.W[i], feats)) + self.b[i]
             for i in range(len(MODULES))]
        p = self._softmax(z)
        best = max(range(len(MODULES)), key=lambda i: p[i])
        return MODULES[best], p[best], p

    def learn(self, feats, actual_module):
        pred, prob, _ = self.predict(feats)
        y = MODULES.index(actual_module)
        # majority-class baseline tracking
        majority = max(self.metrics["class_counts"].items(), key=lambda kv: kv[1])[0]
        self.metrics["n"] += 1
        if pred == actual_module:
            self.metrics["correct"] += 1
        if majority == actual_module:
            self.metrics["majority_correct"] += 1
        self.metrics["class_counts"][actual_module] += 1
        if not self.enabled:
            return
        # online softmax gradient (deterministic)
        z = [sum(w * f for w, f in zip(self.W[i], feats)) + self.b[i]
             for i in range(len(MODULES))]
        p = self._softmax(z)
        for i in range(len(MODULES)):
            g = p[i] - (1.0 if i == y else 0.0)
            for j in range(SCHEMA_FEATURES):
                self.W[i][j] -= self.lr * g * feats[j]
            self.b[i] -= self.lr * g

    def accuracy(self):
        n = max(1, self.metrics["n"])
        return self.metrics["correct"] / n, self.metrics["majority_correct"] / n

    def to_dict(self):
        return {"W": [list(r) for r in self.W], "b": list(self.b),
                "enabled": self.enabled, "metrics": {
                    "n": self.metrics["n"], "correct": self.metrics["correct"],
                    "majority_correct": self.metrics["majority_correct"],
                    "class_counts": dict(self.metrics["class_counts"])}}

    @classmethod
    def from_dict(cls, d):
        s = cls.__new__(cls)
        s.W = [list(r) for r in d["W"]]
        s.b = list(d["b"])
        s.lr = 0.06
        s.enabled = d["enabled"]
        s.metrics = {"n": d["metrics"]["n"], "correct": d["metrics"]["correct"],
                     "majority_correct": d["metrics"]["majority_correct"],
                     "class_counts": dict(d["metrics"]["class_counts"])}
        return s


def schema_features(ctx):
    """Features for predicting the NEXT cycle's attention winner."""
    urg = ctx["top_urgency"] / 20.0
    feats = [
        min(1.0, urg),                                   # drive pressure
        ctx["light"],                                    # light
        ctx["arousal"],                                  # arousal
        ctx["distractor_novelty"],                       # best distractor novelty
        1.0 if ctx["goal_active"] else 0.0,              # goal pursuit active
        ctx["hour_frac"],                                # time of day
    ]
    prev = ctx["prev_winner"]
    for m in MODULES[:4]:                                # one-hot of previous winner (first 4)
        feats.append(1.0 if prev == m else 0.0)
    return feats  # 10 features


# ---------------------------------------------------------------- workspace
class Workspace:
    def __init__(self):
        self.contents = []            # winning coalitions (<= CAPACITY)
        self.schema = AttentionSchema()
        self.lesion_off = False       # battery P-A: no workspace at all
        self.lesion_single_query = False  # battery GWT-4 lesion
        self.metrics = {
            "cycles": 0, "proposals_sum": 0, "excluded_salient": 0,
            "winner_counts": {m: 0 for m in MODULES},
            "entry_counts": {m: 0 for m in MODULES},
            "propose_cycles": {m: 0 for m in MODULES},
            "broadcast_receipts": {m: 0 for m in MODULES},
            "queries": 0, "task_cycles": 0, "queries_in_task": 0,
            "capture_events": 0,
        }
        self._capture_run = 0
        # v4 ecology (GATE v4-1): per-module winner fatigue — inhibition
        # of return at the module level. A module that won entry last
        # cycle bids at 0.60 strength; the factor recovers x1.4 per
        # cycle it stays out. (Dev calibration trail: 0.55 made every
        # module ubiquitous and diluted the cue-specificity measures;
        # 0.68 let a multi-voice module lock the broadcast alongside
        # the slot cap's absence; 0.60 + the 2-slot cap is the frozen
        # combination.) No module may hold the broadcast back-to-back
        # at full strength indefinitely (sitting 3: beliefs held ~2.0
        # entries/cycle and motor starved below the GWT-1 floor).
        self._ffactor = {m: 1.0 for m in MODULES}
        self.last_queries = []        # query names in the last cycle (viewer/debug)
        self.chain_completed = False  # v2 (2026-10-23): full query chain ran THIS cycle

    # ---------------- stage 1 ----------------
    def compete(self, proposals, schema_boost=None):
        """schema_boost: {module: factor} from the attention schema."""
        scored = []
        present = set()
        for c in proposals:
            present.add(c.source)
            # v4-1: fatigue applies to the BASE salience, before the
            # attention schema's deliberate control boosts
            s = c.salience * self._ffactor.get(c.source, 1.0)
            if schema_boost and c.source in schema_boost:
                s *= schema_boost[c.source]
            scored.append((s, c))
        for src in present:
            self.metrics["propose_cycles"][src] += 1
        scored.sort(key=lambda t: -t[0])
        # v4 slot cap (GATE v4-1 calibration): at most 2 of the 3
        # broadcast slots per module. The broadcast is a coalition
        # ACROSS modules by design; dev showed both monopoly failure
        # modes — beliefs ~2.0 entries/cycle with no fatigue (sitting
        # 3), memory ~2.4/cycle once its bids carried retrieval scores
        # (round-3 dev) — each collapsing the winner distribution the
        # attention schema must learn (AST-1a seed-11 dev ratio fell
        # to 1.18 under the second).
        winners = []
        per_src = {}
        for _, c in scored:
            if per_src.get(c.source, 0) >= 2:
                continue
            per_src[c.source] = per_src.get(c.source, 0) + 1
            winners.append(c)
            if len(winners) >= CAPACITY:
                break
        m = self.metrics
        m["cycles"] += 1
        m["proposals_sum"] += len(proposals)
        if winners:
            win_s = scored[0][0]
            for s, c in scored[CAPACITY:]:
                if s >= 0.5 * win_s and win_s > 0:
                    m["excluded_salient"] += 1
                    break
            m["winner_counts"][winners[0].source] += 1
            for c in winners:
                m["entry_counts"][c.source] += 1
        # v4-1: fatigue update — winners tire, the excluded recover
        won_src = {c.source for c in winners}
        for mod in MODULES:
            if mod in won_src:
                self._ffactor[mod] = 0.60
            else:
                self._ffactor[mod] = min(1.0, self._ffactor[mod] * 1.4)
        self.contents = winners
        # broadcast: delivered to every module (receipts audited)
        for mod in MODULES:
            m["broadcast_receipts"][mod] += 1
        return winners

    def query(self, name):
        self.metrics["queries"] += 1
        self.last_queries.append(name)

    def note_task_cycle(self, n_queries):
        self.metrics["task_cycles"] += 1
        self.metrics["queries_in_task"] += n_queries

    def note_attention(self, winner_source, goal_active, focus_is_distractor):
        """Distractor-capture bookkeeping: during goal pursuit, vision
        winning with a novel NON-goal object for >=3 straight cycles is a
        capture event."""
        if goal_active and focus_is_distractor and winner_source == "vision":
            self._capture_run += 1
            if self._capture_run == 3:
                self.metrics["capture_events"] += 1
        else:
            self._capture_run = 0

    def to_dict(self):
        return {"contents": [c.row() for c in self.contents],
                "schema": self.schema.to_dict(),
                "lesion_off": self.lesion_off,
                "lesion_single_query": self.lesion_single_query,
                "metrics": {"cycles": self.metrics["cycles"],
                            "proposals_sum": self.metrics["proposals_sum"],
                            "excluded_salient": self.metrics["excluded_salient"],
                            "winner_counts": dict(self.metrics["winner_counts"]),
                            "entry_counts": dict(self.metrics["entry_counts"]),
                            "propose_cycles": dict(self.metrics["propose_cycles"]),
                            "broadcast_receipts": dict(self.metrics["broadcast_receipts"]),
                            "queries": self.metrics["queries"],
                            "task_cycles": self.metrics["task_cycles"],
                            "queries_in_task": self.metrics["queries_in_task"],
                            "capture_events": self.metrics["capture_events"]},
                "capture_run": self._capture_run,
                "ffactor": dict(self._ffactor),
                "last_queries": list(self.last_queries)}

    @classmethod
    def from_dict(cls, d):
        w = cls.__new__(cls)
        w.contents = []
        for r in d["contents"]:
            w.contents.append(Coalition(r["source"], r["kind"], r["salience"], r["target"]))
        w.schema = AttentionSchema.from_dict(d["schema"])
        w.lesion_off = d["lesion_off"]
        w.lesion_single_query = d["lesion_single_query"]
        w.metrics = {"cycles": d["metrics"]["cycles"],
                     "proposals_sum": d["metrics"]["proposals_sum"],
                     "excluded_salient": d["metrics"]["excluded_salient"],
                     "winner_counts": dict(d["metrics"]["winner_counts"]),
                     "entry_counts": dict(d["metrics"]["entry_counts"]),
                     "propose_cycles": dict(d["metrics"].get("propose_cycles", {m: 0 for m in MODULES})),
                     "broadcast_receipts": dict(d["metrics"]["broadcast_receipts"]),
                     "queries": d["metrics"]["queries"],
                     "task_cycles": d["metrics"]["task_cycles"],
                     "queries_in_task": d["metrics"]["queries_in_task"],
                     "capture_events": d["metrics"]["capture_events"]}
        w._capture_run = d["capture_run"]
        w._ffactor = {m: float(d.get("ffactor", {}).get(m, 1.0)) for m in MODULES}
        w.last_queries = list(d["last_queries"])
        return w
