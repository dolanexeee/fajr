"""Live wiring for the PROVEN novelty mechanism (LANGUAGE-GROUNDING Part B).

The NoveltyTracker + GroundedConcept in grounding.py are proven in
harness (test_grounding.py 17/17). This module connects them to her
LIVE perceptual path — additively:

  * every tick, each actually-sensed scene object is offered to the
    tracker via LiveNovelty.feed()
  * known kinds (perception.OBJECT_KINDS) return immediately from
    observe() — the tracker is verifiably DORMANT in her current
    valley, which contains no novel stimuli
  * if a genuinely novel kind ever appears, after K=5 exposures the
    concept crystallizes: the label is coined by the proven mechanism,
    promoted into her lexicon as a grounded 'novel-live' concept with
    full provenance, and recorded in the journal as a novel_word event

Nothing here changes any existing code path. The feed is read-only
over the scene; promotion and journal writes happen ONLY at
crystallization, which cannot occur while all percepts are known.

No LLM. Deterministic. Pure Python (+ numpy-free).
"""
from grounding import NoveltyTracker, NOVEL_K, NOVEL_GAIN
from perception import OBJECT_KINDS

# journal event type for a crystallized novel word (additive: new type,
# journal format untouched)
NOVEL_WORD_EVENT = "novel_word"


def _live_features(obj):
    """Coarse, stable bands for a scene object's signature.

    Bands (not raw floats) so the same object yields the same signature
    tick after tick — a concept must be STABLE to crystallize."""
    d = obj.dist
    band = "near" if d < 8.0 else ("mid" if d < 25.0 else "far")
    feats = {"dist": band}
    if obj.kind in ("tree", "pile"):
        feats["fruit"] = "yes" if obj.fruit > 0 else "no"
    return feats


class LiveNovelty:
    """Owns one NoveltyTracker for the live biography, with checkpoint
    (de)serialization and the crystallization side-effects (lexicon
    promotion + journal event)."""

    def __init__(self, tracker=None):
        self.tracker = tracker if tracker is not None else NoveltyTracker(
            known_kinds=OBJECT_KINDS, k=NOVEL_K, gain=NOVEL_GAIN)
        # labels promoted into the shared lexicon this session
        # (re-promoted from tracker state on checkpoint load)
        self.promoted = []

    # ---------------- live feed (called once per tick) ----------------
    def feed(self, scene, tick, journal, x=None, y=None):
        """Offer every actually-sensed object to the tracker.

        Returns the number of crystallizations this tick (0 in her
        current valley — always, until a novel kind appears)."""
        crystallized = 0
        for key in sorted(scene.objects.keys()):
            o = scene.objects[key]
            if o.generated:
                continue  # top-down placeholder, not a genuine percept
            concept, now = self.tracker.observe(
                tick, o.kind, _live_features(o))
            if now:
                self._on_crystallize(concept, tick, journal, x, y)
                crystallized += 1
        return crystallized

    # ---------------- crystallization side-effects ----------------
    def _on_crystallize(self, concept, tick, journal, x, y):
        kind = concept.signature[0]
        label = concept.label
        self._promote(concept, tick)
        journal.log(
            tick, NOVEL_WORD_EVENT,
            f"كلمة جديدة: «{label}» — مفهوم جديد من التجربة "
            f"(kind={kind}، {len(concept.exposures)} تعرضات، "
            f"provenance=novel-live)",
            x=x, y=y)

    def _promote(self, concept, tick):
        """Register the coined label as a grounded 'novel-live' concept.

        Idempotent: safe to call again on checkpoint resume."""
        from lexicon import WORDS
        from grounding import GROUNDING
        label = concept.label
        kind = concept.signature[0]
        if label in self.promoted:
            return
        wid = f"novel_{label}"
        if wid not in WORDS:
            WORDS[wid] = {
                "ar": label, "pos": "noun",
                "anchor": f"object:novel:{kind}",
                "provenance": "novel-live",
                "crystallized_tick": concept.crystallized_tick,
                "exposures": len(concept.exposures),
            }
        if label not in GROUNDING:
            feats = sorted(f"{k}={v}" for k, v in concept.signature[1])
            GROUNDING[label] = {
                "sensors": f"novel percept: kind={kind} ({', '.join(feats)})",
                "drives": "—",
                "action": "— (perceived; named at crystallization)",
                "learned_via": ("novel-live: K=%d exposures, label coined "
                                "from her own voice phonemes" % NOVEL_K),
                "emits_via": ("name_active_concept: crystallized AND "
                              "pattern currently present"),
                "status": "grounded",
                "provenance": "novel-live",
            }
        self.promoted.append(label)

    # ---------------- checkpoint (de)serialization ----------------
    def to_dict(self):
        concepts = []
        for c in self.tracker.concepts:
            concepts.append({
                "kind": c.signature[0],
                "features": sorted([list(kv) for kv in c.signature[1]]),
                "exposures": list(c.exposures),
                "strength": c.strength,
                "label": c.label,
                "crystallized": c.crystallized,
                "crystallized_tick": c.crystallized_tick,
            })
        return {
            "known_kinds": sorted(self.tracker.known_kinds),
            "k": self.tracker.k,
            "gain": self.tracker.gain,
            "concepts": concepts,
            "coined": sorted(self.tracker.coined),
            "promoted": list(self.promoted),
        }

    @classmethod
    def from_dict(cls, d):
        from grounding import GroundedConcept
        if not d:
            return cls()  # old checkpoint: fresh, dormant tracker
        tracker = NoveltyTracker(
            known_kinds=d["known_kinds"], k=d["k"], gain=d["gain"])
        for row in d["concepts"]:
            c = GroundedConcept(
                (row["kind"], frozenset((k, v) for k, v in row["features"])))
            c.exposures = list(row["exposures"])
            c.strength = row["strength"]
            c.label = row["label"]
            c.crystallized = row["crystallized"]
            c.crystallized_tick = row["crystallized_tick"]
            tracker.concepts.append(c)
        tracker.coined = set(d["coined"])
        live = cls(tracker)
        # re-promote crystallized concepts into this process's lexicon
        # (module-global WORDS does not survive a restart)
        for c in tracker.concepts:
            if c.crystallized and c.label:
                live._promote(c, c.crystallized_tick)
        live.promoted = list(d.get("promoted", live.promoted))
        return live

    # ---------------- inspection (for the dormancy proof) ----------------
    def status(self):
        t = self.tracker
        return {
            "concepts_tracked": len(t.concepts),
            "crystallized": sum(1 for c in t.concepts if c.crystallized),
            "labels_coined": sorted(t.coined),
            "promoted": list(self.promoted),
        }
