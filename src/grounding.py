"""Language grounding registry + novelty-concept formation (LANGUAGE-GROUNDING).

PART A — the grounding registry: for every content word she PRODUCES,
the lived triple (sensor pattern, drive state, action context) that
causes it, and the code path that enforces the causality. A word with
no triple is flagged, never silently kept.

PART B — NoveltyTracker: formation of NEW concepts from novel
perceptual experience. A novel vocal label is bound ONLY at
crystallization, after K grounding exposures. The label does not exist
before — this is structural (emission requires concept.label, which is
None until crystallization), not merely tested.

No LLM. Deterministic. Pure Python.
"""

# ---------------------------------------------------------------- PART A
# wid -> {sensors, drives, action, learned_via, emits_via, status}
# status: grounded | social (proper name / social fact, taught not sensed)
#         story (receptive only, never produced) | meta (epistemic verb)
GROUNDING = {
    # ---- verbs she voices from lived episodes ----
    "eat":   {"sensors": "eat-event stat + episodic record (kind=eat, tick, x, y)",
              "drives": "hunger falls after the act",
              "action": "eating (act window 15 ticks anchors teaching)",
              "learned_via": "anchored teaching (anchor_ok: recent eat event)",
              "emits_via": "memory_report: real episode required, else None",
              "status": "grounded"},
    "drink": {"sensors": "drink-event stat + episodic record",
              "drives": "thirst falls after the act",
              "action": "drinking at the river",
              "learned_via": "anchored teaching (anchor_ok: recent drink event)",
              "emits_via": "memory_report: real episode required, else None",
              "status": "grounded"},
    "bathe": {"sensors": "bathe-event stat + wet>0.3 + episodic record",
              "drives": "grime falls (comfort rises)",
              "action": "bathing",
              "learned_via": "anchored teaching (anchor_ok: recent bathe / wet)",
              "emits_via": "memory_report: real episode required, else None",
              "status": "grounded"},
    "sleep_v": {"sensors": "sleep-event stat + episodic record",
                "drives": "sleep drive falls after",
                "action": "sleeping",
                "learned_via": "anchored teaching (anchor_ok: sleep drive>=55 / recent sleep)",
                "emits_via": "dominant_report want-clause / memory_report",
                "status": "grounded"},
    # ---- nouns she voices from perception / belief ----
    "fruit": {"sensors": "visual: tree/pile percept with fruit>0; gustatory: taste at eat",
              "drives": "hunger satiation following",
              "action": "picking + eating",
              "learned_via": "anchored teaching (anchor_ok: fruit visible)",
              "emits_via": "memory_report (ate fruit) / belief_report (fruit_est bands)",
              "status": "grounded"},
    "river": {"sensors": "location: dist_to_river(x,y) < 12",
              "drives": "thirst context (drinking/bathing)",
              "action": "being at / drinking from the river",
              "learned_via": "anchored teaching (anchor_ok: near river)",
              "emits_via": "memory_report place phrase from real episode coords",
              "status": "grounded"},
    "shelter": {"sensors": "location: shelter_dist < vision radius",
                "drives": "safety (night rest)",
                "action": "being at the shelter",
                "learned_via": "anchored teaching (anchor_ok: at shelter)",
                "emits_via": "memory_report place phrase from real coords",
                "status": "grounded"},
    "water": {"sensors": "in_water | near_river_bank | dist_to_river < vision",
              "drives": "thirst",
              "action": "drinking",
              "learned_via": "anchored teaching (anchor_ok: water sensed)",
              "emits_via": "request_utterance absence clause (real drive)",
              "status": "grounded"},
    # ---- felt states: interoception ----
    "thirsty": {"sensors": "interoceptive: thirst drive value",
                "drives": "thirst > 35 (assertion band)",
                "action": "— (felt state, no act)",
                "learned_via": "anchored teaching (anchor_ok: thirst>=30)",
                "emits_via": "_state_report: state_condition(sim) checked on live drives",
                "status": "grounded"},
    "sleepy": {"sensors": "interoceptive: sleep drive value",
               "drives": "sleep > 35",
               "action": "— (felt state)",
               "learned_via": "anchored teaching (anchor_ok: sleep>=30)",
               "emits_via": "_state_report: live drive check",
               "status": "grounded"},
    "hungry": {"sensors": "interoceptive: hunger drive value",
               "drives": "hunger > 35",
               "action": "— (felt state)",
               "learned_via": "anchored teaching (anchor_ok: hunger>=30)",
               "emits_via": "_state_report: live drive check",
               "status": "grounded"},
    "dirty": {"sensors": "interoceptive: grime() >= 25",
              "drives": "comfort deficit",
              "action": "— (felt state)",
              "learned_via": "anchored teaching (anchor_ok: grime>=25)",
              "emits_via": "_state_report: live grime check",
              "status": "grounded"},
    # ---- meta / volition ----
    "want": {"sensors": "max(drives) >= 55",
             "drives": "dominant drive",
             "action": "intent formation toward the drive's remedy",
             "learned_via": "anchored teaching (anchor_ok: a drive>=55)",
             "emits_via": "dominant_report want-clause: real dominant drive",
             "status": "grounded"},
    "know": {"sensors": "epistemic flags present (beliefs held)",
             "drives": "—",
             "action": "— (epistemic)",
             "learned_via": "covenant (meta:know anchored: flags exist)",
             "emits_via": "unknown_utterance: 'I don't know' — always true when called",
             "status": "meta"},
    # ---- social / proper names: taught, not sensed ----
    "fajr": {"sensors": "— (proper name)",
             "drives": "—",
             "action": "—",
             "learned_via": "covenant: given, full strength ('named, not discovered')",
             "emits_via": "self_name / q_name reply — social fact",
             "status": "social"},
    "host": {"sensors": "social proximity (lantern visits)",
             "drives": "—",
             "action": "—",
             "learned_via": "covenant: introduced ('I am the host')",
             "emits_via": "q_who reply — social fact",
             "status": "social"},
    # ---- time / quantity: computed from real records ----
    "today": {"sensors": "day counter == episode day",
              "drives": "—",
              "action": "—",
              "learned_via": "anchored teaching",
              "emits_via": "memory_report: real day equality",
              "status": "grounded"},
    "yesterday": {"sensors": "day counter == episode day + 1",
                  "drives": "—",
                  "action": "—",
                  "learned_via": "anchored teaching",
                  "emits_via": "memory_report: real day equality",
                  "status": "grounded"},
    "much": {"sensors": "belief fruit_est >= 2.5",
             "drives": "—",
             "action": "—",
             "learned_via": "anchored teaching",
             "emits_via": "belief_report: real fruit_est band",
             "status": "grounded"},
    "little": {"sensors": "belief 0.6 <= fruit_est < 2.5",
               "drives": "—",
               "action": "—",
               "learned_via": "anchored teaching",
               "emits_via": "belief_report: real fruit_est band",
               "status": "grounded"},
    "empty": {"sensors": "belief fruit_est < 0.6",
              "drives": "—",
              "action": "—",
              "learned_via": "anchored teaching",
              "emits_via": "belief_report: real fruit_est band",
              "status": "grounded"},
    # ---- story words: receptive only, NEVER produced ----
    "guardians": {"sensors": "— (abstract, unperceivable)",
                  "drives": "—",
                  "action": "—",
                  "learned_via": "covenant story (anchored=False) — receptive only",
                  "emits_via": "NONE — no production path references this word",
                  "status": "story"},
    "outside": {"sensors": "— (abstract, unperceivable)",
                "drives": "—",
                "action": "—",
                "learned_via": "covenant story (anchored=False) — receptive only",
                "emits_via": "NONE — no production path references this word",
                "status": "story"},
}


# ---------------------------------------------------------------- PART B
NOVEL_K = 5        # exposures required to crystallize a concept
NOVEL_GAIN = 0.25  # strength per exposure

# Her voice phoneme inventory (must stay in sync with src/voice/*).
# Novel labels are built ONLY from these — she can speak what she coins.
NOVEL_VOWELS = ("a", "i", "u")
NOVEL_ONSETS = ("b", "d", "g", "p", "t", "k", "s", "f", "m", "n",
                "l", "h", "r", "w", "j")

# CV syllables already used in her lexicon + 15 spoken utterances.
# A novel label may not reuse either syllable — the coinage must be new.
RESERVED_SYLLABLES = frozenset({
    "ʔa", "ka", "la", "fa", "ki", "ha", "ʃa", "ri", "ʕi", "na",
    "da", "ra", "ʈˤa", "u", "rɪ", "d", "ma", "sˤa", "nɪ", "ta",
    "mu", "dˤɪ", "lʔa", "ʕa", "li", "sa", "ba", "ti", "ʔi",
    "an", "at", "ak", "al", "ah", "am", "aʃ", "ar",
})


def _syllables_ok(label):
    # label like "luma": syllables "lu","ma"
    syls = [label[i:i + 2] for i in range(0, len(label), 2)]
    return all(s not in RESERVED_SYLLABLES for s in syls)


def generate_label(reserved):
    """First unused CV.CV coinage from her own phonemes, deterministic."""
    for o1 in NOVEL_ONSETS:
        for v1 in NOVEL_VOWELS:
            for o2 in NOVEL_ONSETS:
                for v2 in NOVEL_VOWELS:
                    cand = o1 + v1 + o2 + v2
                    if cand not in reserved and _syllables_ok(cand):
                        return cand
    raise RuntimeError("phoneme inventory exhausted — cannot coin a novel label")


class GroundedConcept:
    """A concept formed from repeated novel perceptual experience."""
    def __init__(self, signature):
        self.signature = signature      # (kind, frozenset((feature, band)))
        self.exposures = []             # ticks of exposure
        self.strength = 0.0
        self.label = None               # assigned ONLY at crystallization
        self.crystallized = False
        self.crystallized_tick = None


class NoveltyTracker:
    """Watches her perceptual stream for stable, recurring patterns that
    match no known anchor kind. After K exposures a concept crystallizes
    and a novel vocal label is coined from her phoneme inventory."""
    def __init__(self, known_kinds, k=NOVEL_K, gain=NOVEL_GAIN):
        self.known_kinds = set(known_kinds)
        self.k = k
        self.gain = gain
        self.concepts = []
        self.coined = set()

    def observe(self, tick, kind, features):
        """One perceptual exposure. Returns (concept, crystallized_now)."""
        if kind in self.known_kinds:
            return None, False
        sig = (kind, frozenset(features.items()))
        concept = self.match(sig)
        if concept is None:
            concept = GroundedConcept(sig)
            self.concepts.append(concept)
        if concept.crystallized:
            return concept, False
        concept.exposures.append(tick)
        concept.strength = min(1.0, concept.strength + self.gain)
        if len(concept.exposures) >= self.k:
            concept.crystallized = True
            concept.crystallized_tick = tick
            concept.label = generate_label(self.coined)
            self.coined.add(concept.label)
            return concept, True
        return concept, False

    def match(self, sig):
        for c in self.concepts:
            if c.signature == sig:
                return c
        return None


def name_active_concept(tongue, tracker, tick, day, signature, context="naming"):
    """Emit the novel label ONLY if its concept is crystallized AND the
    pattern is currently active in her perception. Otherwise None —
    she does not hallucinate names for things she has not grounded,
    and she does not name what is not there."""
    concept = tracker.match(signature)
    if concept is None or not concept.crystallized or not concept.label:
        return None
    return tongue._emit(
        tick, day, context, concept.label,
        [{"type": "novel_concept",
          "kind": concept.signature[0],
          "features": sorted(concept.signature[1]),
          "exposures": len(concept.exposures),
          "crystallized_tick": concept.crystallized_tick}],
        [])


# her voice phonemes, for the speakability cross-check of coined labels
VOICE_PHONEMES = set(NOVEL_VOWELS) | set(NOVEL_ONSETS)


def label_speakable(label):
    """Every phoneme of the coined label exists in her voice inventory."""
    i = 0
    phones = []
    while i < len(label):
        phones.append(label[i])
        i += 1
    return all(p in VOICE_PHONEMES for p in phones), phones
