# -*- coding: utf-8 -*-
"""Culture — ADAM & EVE P5: perpetual learning (التعلم المطلق).

HONESTY NOTE (read before touching):
- Skill level = 1 - exp(-practice / TAU): a PURE function of LIVED
  practice. There is no other way to gain skill. No free levels, no
  admin grants, no decay (practice never decreases — knowledge only
  ratchets upward).
- Teaching transmission: on a teaching event the learner's practice
  moves TOWARD the teacher's level with a small transmission loss
  (target = teacher_level * (1 - LOSS)). Teaching alone can never push
  the learner past that target — surpassing the master requires lived
  practice. Knowledge survives the teacher: it lives in the learner.
- Children learn parent-known skills at double speed (shoulders).
- Tribe "era" milestones fire when collective capability crosses a
  threshold. Era NAMES are OUR reading (like the hut reading), journaled
  behaviorally.

RNG: none needed here (deterministic formulas). The shared SplitMix64
PRNG is used by callers for any stochastic choice.
"""
import math

SKILL_TAU = 400.0       # same curve as her swim 1.0 and his forage curve
TRANSMIT_FRAC = 0.08    # fraction of the gap closed per teaching event
TRANSMIT_LOSS = 0.05    # learner approaches teacher * 0.95
TEACH_ERA_NEED = 50     # teaching events that open the "teaching era"

SKILLS = ("forage", "swim", "build")


def skill_level(practice):
    """0..1, pure function of lived practice. The ONLY way up."""
    return 1.0 - math.exp(-max(0.0, practice) / SKILL_TAU)


def practice_for_level(level):
    """Inverse: practice ticks needed to reach a level."""
    level = min(max(level, 0.0), 0.999999)
    return -SKILL_TAU * math.log(max(1e-9, 1.0 - level))


# (skill, threshold, era_key, journal_text) — thresholds on the tribe's
# best level in that skill ("collective capability", our reading)
ERA_SKILLS = [
    ("swim", 0.8, "era_swim", "بدأ عصر السباحة في الوادي"),
    ("forage", 0.8, "era_forage", "بدأ عصر الجمع في الوادي"),
    ("build", 0.8, "era_build", "بدأ عصر البناء في الوادي"),
]
ERA_TEACH = ("era_teach", TEACH_ERA_NEED, "بدأ عصر التعليم في الوادي")
ERA_GENERATION = ("era_generation", "بدأ عصر الجيل الجديد في الوادي")


class Culture:
    """The tribe's cumulative knowledge. Lives on Sim, persisted."""

    def __init__(self):
        self.eras = []            # journaled era keys, in order
        self.teach_events = 0     # cumulative teaching transmissions
        # P6: discoveries — tribal knowledge that NEVER decays.
        # Registered only when OBSERVED (a planted crop harvested, etc.),
        # never scripted. No dark ages: this list only grows.
        self.discoveries = []     # [{key, text, tick, by}]

    def register_discovery(self, key, text, tick, journal, x, y, by=None):
        """P6: a stable discovery becomes permanent tribal knowledge.
        Idempotent. Returns True on first registration."""
        if any(d["key"] == key for d in self.discoveries):
            return False
        self.discoveries.append({"key": key, "text": text, "tick": tick,
                                 "by": by})
        journal.log(tick, "discovery", text, x=x, y=y)
        return True

    def knows(self, key):
        return any(d["key"] == key for d in self.discoveries)

    def transmit(self, learner_practice, teacher_level):
        """Practice granted by ONE teaching event. The learner closes
        TRANSMIT_FRAC of the gap toward teacher*(1-LOSS), and the grant
        is clamped so teaching alone NEVER crosses that ceiling —
        surpassing the master requires lived practice."""
        target = teacher_level * (1.0 - TRANSMIT_LOSS)
        if skill_level(learner_practice) >= target or target <= 0:
            return 0
        gap = practice_for_level(target) - learner_practice
        if gap <= 0:
            return 0
        # floor (not ceil): the grant never crosses the ceiling, even
        # fractionally — the last sliver stays unclosed, asymptotically
        grant = min(max(2, int(gap * TRANSMIT_FRAC)), int(gap))
        return grant

    def mark(self, key, text, tick, journal, x, y):
        """Journal an era once (idempotent)."""
        if key in self.eras:
            return False
        self.eras.append(key)
        journal.log(tick, "era", text, x=x, y=y)
        return True

    def check_skill_eras(self, tribe_levels, tick, journal, x, y):
        """tribe_levels: iterable of {skill: level} dicts, one per agent."""
        best = {s: 0.0 for s in SKILLS}
        for lv in tribe_levels:
            for s in SKILLS:
                best[s] = max(best[s], lv.get(s, 0.0))
        for skill, thr, key, text in ERA_SKILLS:
            if key not in self.eras and best[skill] >= thr:
                self.mark(key, text, tick, journal, x, y)
        if (ERA_TEACH[0] not in self.eras
                and self.teach_events >= ERA_TEACH[1]):
            self.mark(ERA_TEACH[0], ERA_TEACH[2], tick, journal, x, y)

    def to_dict(self):
        return {"eras": list(self.eras), "teach_events": self.teach_events,
                "discoveries": [dict(d) for d in self.discoveries]}

    @classmethod
    def from_dict(cls, d):
        c = cls()
        c.eras = list(d.get("eras", []))
        c.teach_events = int(d.get("teach_events", 0))
        c.discoveries = [dict(x) for x in d.get("discoveries", [])]
        return c
