"""Aspirations — Phase 4 (التكامل الفعلي, stage 2).

Self-chosen goals grown from dream themes. When a dream theme keeps
recurring, it becomes an aspiration: a standing preference that biases
intent scoring toward themed acts. Aspirations are NOT drives (no
physiology behind them) and NOT orders (no host gives them) — they are
hers, grown from her own nights.

Mechanism:
- birth: dreamer.theme_weights[theme] >= BIRTH_W and no live aspiration
  of that theme -> born, journaled ("طموح جديد").
- bias: in _decide, matching intent scores x (1 + BONUS * strength).
  The survival filter still runs AFTER the bonus, so lethal needs
  always preempt aspirations. Aspirations bias; they never command.
- fulfill: the themed act's stat grows since birth -> satisfied,
  journaled ("حققت طموحها"), retired with a small satisfaction note.
- decay: unfulfilled aspirations fade over days; fulfilled ones leave
  a trace in theme_weights (she remembers what mattered).

Deterministic: sim PRNG only. Persisted via to_dict/from_dict.
"""

BIRTH_W = 1.5          # theme weight needed to birth an aspiration
BONUS = 0.35           # score multiplier per unit strength
DECAY_PER_DAY = 0.25   # strength lost per day unfulfilled
MAX_LIVE = 3           # she holds at most 3 aspirations at once
MOOD_BIRTH_GAIN = 0.20  # good mood lowers the birth threshold

# dream theme -> intent kinds it favours (must match _decide's kinds)
THEME_INTENTS = {
    "swim":    ("bathe",),
    "eat":     ("eat", "seek_food"),
    "drink":   ("drink",),
    "gather":  ("gather", "seek_material"),
    "build":   ("place",),
    "shelter": ("sleep", "shelter"),
    "explore": ("search", "explore", "wander"),
    "storm":   ("shelter", "sleep"),
    # ADAM & EVE P3+: her own dream-borne kinship longing biases her
    # normal "union" choice — exactly like hunger->eat.
    "kinship": ("union",),
}

# dream theme -> fajr.stats key whose growth fulfills it
# ("distance" needs a delta threshold instead of +1)
THEME_STAT = {
    "swim": "swims", "eat": "eats", "drink": "drinks",
    "gather": "gathers", "build": "places", "shelter": "sleeps",
    "explore": "distance", "storm": "sleeps",
    # union episodes counter (incremented by Sim at each recorded
    # union episode, see sim._pair_tick) — a kinship aspiration is
    # fulfilled by lived togetherness.
    "kinship": "unions",
}
DISTANCE_FULFILL = 60.0

ASPIRATION_TEXT = {
    "swim":    "طموح جديد: تريد أن تسبح أكثر",
    "eat":     "طموح جديد: تريد أن تجد فاكهة أطيب",
    "drink":   "طموح جديد: تريد ماءً أعذب",
    "gather":  "طموح جديد: تريد أن تجمع أشياء جميلة",
    "build":   "طموح جديد: تريد أن تبني شيئًا أكبر",
    "shelter": "طموح جديد: تريد مأوى أدفأ",
    "explore": "طموح جديد: تريد أن تستكشف ما وراء التل",
    "storm":   "طموح جديد: تريد أن تكون آمنة دائمًا",
    "kinship": "طموح جديد: تريد أن تكون مع زوجها أكثر",
}
FULFILL_TEXT = {
    "swim":    "حققت طموحها: سبحت كما حلمت",
    "eat":     "حققت طموحها: أكلت كما حلمت",
    "drink":   "حققت طموحها: شربت كما حلمت",
    "gather":  "حققت طموحها: جمعت كما حلمت",
    "build":   "حققت طموحها: بنت كما حلمت",
    "shelter": "حققت طموحها: نامت آمنة كما حلمت",
    "explore": "حققت طموحها: استكشفت كما حلمت",
    "storm":   "حققت طموحها: وجدت الأمان كما حلمت",
    "kinship": "حققت طموحها: عاشت مع زوجها كما حلمت",
}


class AspirationEngine:
    def __init__(self):
        self.aspirations = []   # {id, theme, strength, born_tick, born_day,
                                #  stat_key, stat_at_birth, status}
        self._next_id = 1
        self.metrics = {"born": 0, "fulfilled": 0, "faded": 0}

    # ---------------- per-tick upkeep (called from mind.cycle) ----------------
    def update(self, dreamer, f, tick, journal, mood=0.0):
        day = tick // 1440 + 1
        live = [a for a in self.aspirations if a["status"] == "live"]
        # birth from dream themes (good mood lowers the bar)
        threshold = BIRTH_W * (1.0 - MOOD_BIRTH_GAIN * mood)
        if len(live) < MAX_LIVE:
            for theme, w in dreamer.theme_weights.items():
                if w >= threshold and not any(a["theme"] == theme for a in live):
                    stat_key = THEME_STAT[theme]
                    a = {"id": self._next_id, "theme": theme,
                         "strength": min(1.0, w / (BIRTH_W * 2)),
                         "born_tick": tick, "born_day": day,
                         "stat_key": stat_key,
                         "stat_at_birth": f.stats.get(stat_key, 0.0),
                         "status": "live"}
                    self._next_id += 1
                    self.aspirations.append(a)
                    live.append(a)
                    self.metrics["born"] += 1
                    journal.log(tick, "aspiration",
                                ASPIRATION_TEXT[theme], x=f.x, y=f.y)
                    # the dream that birthed it is spent
                    dreamer.theme_weights[theme] = 0.0
                    if len(live) >= MAX_LIVE:
                        break
        # fulfill / decay (decay ticks once per day, not per tick)
        for a in live:
            now = f.stats.get(a["stat_key"], 0.0)
            grew = now - a["stat_at_birth"]
            done = grew >= (DISTANCE_FULFILL if a["stat_key"] == "distance" else 1.0)
            if done:
                a["status"] = "fulfilled"
                self.metrics["fulfilled"] += 1
                journal.log(tick, "aspiration", FULFILL_TEXT[a["theme"]],
                            x=f.x, y=f.y)
                continue
            age_days = day - a["born_day"]
            if day != a.get("last_decay_day", a["born_day"]):
                a["last_decay_day"] = day
                a["strength"] = max(0.0, a["strength"] - DECAY_PER_DAY)
            if age_days >= 4 and a["strength"] <= 0.05:
                a["status"] = "faded"
                self.metrics["faded"] += 1

    # ---------------- decision bias (called from _decide) ----------------
    def bonus(self, intent_kind):
        mult = 1.0
        for a in self.aspirations:
            if a["status"] == "live" and intent_kind in THEME_INTENTS.get(a["theme"], ()):
                mult *= 1.0 + BONUS * a["strength"]
        return mult

    def live_themes(self):
        return [a["theme"] for a in self.aspirations if a["status"] == "live"]

    # ---------------- persistence ----------------
    def to_dict(self):
        return {"aspirations": [dict(a) for a in self.aspirations],
                "next_id": self._next_id,
                "metrics": dict(self.metrics)}

    @classmethod
    def from_dict(cls, d):
        e = cls.__new__(cls)
        e.aspirations = [dict(a) for a in d.get("aspirations", [])]
        e._next_id = d.get("next_id", 1)
        e.metrics = dict(d.get("metrics", {"born": 0, "fulfilled": 0, "faded": 0}))
        return e
