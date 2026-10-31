"""Episodic + semantic memory — Phase 2 (SPEC §6; battery probe P-C).

EPISODES are autobiographical records: (tick, place, event kind, scene
digest, drive snapshot, affect snapshot, outcome). They are written at
salient moments, CONSOLIDATED during sleep (semantic facts distilled,
weak traces pruned), and RETRIEVED by cue (goal + place + affect) — the
memory module's proposals come from here, which is what makes an effect
like probe P-C (a one-time seen-but-not-eaten fruit pile steering him a
day later) a memory effect rather than an associative-grid effect:
Phase-1 structures never record "I SAW food at L"; episodes do.

SEMANTIC facts are distilled regularities: tree fruit rates, place safety,
propositions that mirror belief vars but are grounded in episode counts.
"""
import math

EPISODE_KINDS = ("eat", "drink", "prick", "cold_shock", "sleep", "wake",
                 "saw_food", "saw_pile", "gather", "place", "storm",
                 "surprise", "shelter_rest", "first_visit",
                 "bathe", "swim")   # Stage 2 (WATER-GATE)
MAX_EPISODES = 4000
# v4 (GATE v4-4a): kinds that SERVE a goal without being the goal kind
# itself — a seen-but-not-eaten pile is food-location information for
# the eat goal exactly as an eat record is, and it is far rarer.
GOAL_SERVANTS = {"eat": ("saw_pile", "saw_food"),
                 "sleep": ("shelter_rest",)}


class EpisodicMemory:
    def __init__(self):
        self.episodes = []          # list of dicts, chronological
        self.semantic = {}          # key -> {"value": float, "support": int, "tick": int}
        self.last_consolidated_tick = 0
        self.consolidations = 0
        self.lesion_off = False     # battery P-C / P-A lesion: no recording, no retrieval
        self.metrics = {"recorded": 0, "retrievals": 0, "retrieval_hits": 0}

    # ---------------- recording ----------------
    def record(self, tick, kind, x, y, scene_digest, drives_snap, valence,
               arousal, outcome=0.0, note=""):
        if self.lesion_off:
            return None
        ep = {"tick": tick, "day": tick // 1440 + 1, "kind": kind,
              "x": round(x, 1), "y": round(y, 1), "scene": scene_digest,
              "drives": drives_snap, "valence": round(valence, 3),
              "arousal": round(arousal, 3), "outcome": outcome, "note": note,
              "strength": 1.0}
        self.episodes.append(ep)
        self.metrics["recorded"] += 1
        if len(self.episodes) > MAX_EPISODES:
            # prune the weakest third (strength = recency x |outcome| x affect)
            self.episodes.sort(key=lambda e: e["strength"])
            self.episodes = self.episodes[len(self.episodes) // 3:]
            self.episodes.sort(key=lambda e: e["tick"])
        return ep

    # ---------------- retrieval ----------------
    def retrieve(self, cue_kinds, cue_x=None, cue_y=None, goal_kind=None, limit=5,
                 with_scores=False):
        """Score episodes by kind match, goal relevance, recency, affect
        salience and (optional) place proximity. Returns top episodes.

        v4 selection (GATE v4-4a): kind-diverse — the best episode of
        each distinct CUE-MATCHED kind is selected before remaining
        slots fill by score. Each cued kind is a separate retrieval
        cue; a store where one kind has a thousand records must not
        let them monopolise the channel, because the sole record of a
        rare kind is often the only carrier of its information (the
        P-C failure: one saw_pile record crowded out of the top-3 by
        routine eat records, sitting 3 §5)."""
        if self.lesion_off or not self.episodes:
            return []
        scored = []
        max_tick = self.episodes[-1]["tick"]
        # v4 (GATE v4-4a): kind base-rates in the store, for the
        # distinctiveness term below
        kind_counts = {}
        for e in self.episodes:
            kind_counts[e["kind"]] = kind_counts.get(e["kind"], 0) + 1
        n_eps = len(self.episodes)
        servants = GOAL_SERVANTS.get(goal_kind, ()) if goal_kind else ()
        for ep in self.episodes:
            s = 0.0
            if ep["kind"] in cue_kinds:
                s += 1.0
                # von Restorff distinctiveness: among cued kinds, the
                # RARE kind carries the information the common kinds
                # cannot (sitting 3: the unique saw_pile record, score
                # ~1.6, was crowded out of the top-3 by fresh eat
                # records at ~2.3 — recency is all an eat record has
                # that a one-time sighting lacks, and it is not what
                # makes the sighting matter)
                s += 0.55 * (1.0 - kind_counts[ep["kind"]] / n_eps)
            if goal_kind and (ep["kind"] == goal_kind or ep["kind"] in servants):
                s += 0.8
            age = max(0, max_tick - ep["tick"])
            s += 0.5 * math.exp(-age / 6000.0)
            s += 0.35 * abs(ep["outcome"])
            s += 0.2 * ep["arousal"]
            if cue_x is not None:
                d = math.hypot(ep["x"] - cue_x, ep["y"] - cue_y)
                s += 0.4 * max(0.0, 1.0 - d / 120.0)
            if s > 0.55:
                scored.append((s, ep))
        scored.sort(key=lambda t: -t[0])
        picked, seen_kinds = [], set()
        for s, ep in scored:                      # pass 1: best per cued kind
            if ep["kind"] in cue_kinds and ep["kind"] not in seen_kinds:
                picked.append((s, ep))
                seen_kinds.add(ep["kind"])
                if len(picked) >= limit:
                    break
        if len(picked) < limit:                   # pass 2: fill by score
            chosen = {id(ep) for _, ep in picked}
            for s, ep in scored:
                if id(ep) not in chosen:
                    picked.append((s, ep))
                    if len(picked) >= limit:
                        break
        picked.sort(key=lambda t: -t[0])
        out = [ep for _, ep in picked]
        self.metrics["retrievals"] += 1
        if out:
            self.metrics["retrieval_hits"] += 1
        if with_scores:
            return picked
        return out

    # ---------------- consolidation (runs in sleep) ----------------
    def consolidate(self, tick):
        """Distil semantic facts from episodes since the last consolidation;
        strengthen affect-laden traces, decay the rest."""
        self.consolidations += 1
        new_eps = [e for e in self.episodes if e["tick"] > self.last_consolidated_tick]
        # tree fruit reliability from eat/saw episodes is handled via the
        # semantic keys the mind supplies; here: generic place outcomes
        for ep in new_eps:
            if ep["kind"] == "eat":
                self._bump("food:found", 1.0)
            if ep["kind"] == "prick":
                self._bump("thorn:hurts", 1.0)
            if ep["kind"] == "drink":
                self._bump("water:river", 1.0)
            if ep["kind"] == "bathe":
                self._bump("water:cleans", 1.0)
        for ep in self.episodes:
            ep["strength"] = ep["strength"] * 0.92 + 0.35 * abs(ep["outcome"]) + 0.15 * ep["arousal"]
        # v4 (GATE v4-4a): rare-kind traces are sole carriers — the
        # generic decay above would let a unique record's strength be
        # ground down by hundreds of routine ones and pruned at the
        # cap. Kinds with <= 2 records keep a strength floor.
        kind_counts = {}
        for e in self.episodes:
            kind_counts[e["kind"]] = kind_counts.get(e["kind"], 0) + 1
        for ep in self.episodes:
            if kind_counts[ep["kind"]] <= 2:
                ep["strength"] = max(ep["strength"], 0.9)
        self.last_consolidated_tick = tick
        return len(new_eps)

    def _bump(self, key, value):
        s = self.semantic.setdefault(key, {"value": 0.0, "support": 0, "tick": 0})
        s["value"] = s["value"] * 0.8 + value * 0.2
        s["support"] += 1

    def set_semantic(self, key, value, tick):
        s = self.semantic.setdefault(key, {"value": 0.0, "support": 0, "tick": 0})
        s["value"] = value
        s["support"] += 1
        s["tick"] = tick

    def get_semantic(self, key, default=None):
        s = self.semantic.get(key)
        return default if s is None else s["value"]

    def to_dict(self):
        return {"episodes": [dict(e) for e in self.episodes],
                "semantic": {k: dict(v) for k, v in self.semantic.items()},
                "last_consolidated_tick": self.last_consolidated_tick,
                "consolidations": self.consolidations,
                "lesion_off": self.lesion_off,
                "metrics": dict(self.metrics)}

    @classmethod
    def from_dict(cls, d):
        m = cls.__new__(cls)
        m.episodes = [dict(e) for e in d["episodes"]]
        m.semantic = {k: dict(v) for k, v in d["semantic"].items()}
        m.last_consolidated_tick = d["last_consolidated_tick"]
        m.consolidations = d["consolidations"]
        m.lesion_off = d["lesion_off"]
        m.metrics = dict(d["metrics"])
        return m
