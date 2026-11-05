"""Dreams — Phase 4 (التكامل الفعلي, stage 1).

Mechanistic dreaming: at sleep onset, after episodic consolidation, the
DreamWeaver recombines recent strong episodes into novel scenes.

A dream is NOT a memory. It lives in a separate store, carries provenance
(the source episode ticks it was woven from), and is journaled with type
"dreams". Honesty charter (frozen): dreams are NEVER passed to
episodic.record, NEVER retrieved as memories, and her recount of a dream
is claimed as {"type": "dream"} — never as a witnessed fact.

Dream -> aspiration link: each dream's theme feeds theme_weights, which
the aspiration engine (stage 2) reads to grow self-chosen goals.

Randomness: the sim PRNG only (deterministic, checkpoint-safe).
"""

# episode kind -> dream theme
KIND_THEME = {
    "eat": "eat", "saw_food": "eat", "saw_pile": "eat",
    "drink": "drink", "bathe": "swim", "swim": "swim",
    "gather": "gather", "place": "build",
    "sleep": "shelter", "wake": "shelter", "shelter_rest": "shelter",
    "storm": "storm", "prick": "storm", "cold_shock": "storm",
    "surprise": "explore", "first_visit": "explore",
    # ADAM & EVE: union episodes (P3+) dream in the kinship theme
    "union": "kinship",
}

# journal fragments per theme: (recombined element A, element B)
# The journal is the biographer's record (third-person neutral), not her speech.
THEME_FRAGMENTS = {
    "swim":    ("تسبح في نهر لا نهاية له", "الماء دافئ ويحملها بعيدًا"),
    "eat":     ("فاكهة بحجم رأسها تتدلى من السماء", "تأكل ولا تشبع"),
    "drink":   ("نهر من ماء صافٍ يغني", "تشرب فيرتوي الوادي كله"),
    "gather":  ("تجمع حجارة تضيء في الظلام", "تملأ يديها ولا تثقلان"),
    "build":   ("تبني برجًا يلامس الغيوم", "كل حجر تضعه يغني لها"),
    "shelter": ("مأوى من ضوء القمر", "تنام فيه ولا برد ولا خوف"),
    "storm":   ("عاصفة من أوراق ملونة", "تركض فيها وتضحك ولا تبتل"),
    "explore": ("وادٍ جديد خلف التل", "كل خطوة فيه تكشف لونًا لم تره"),
    # ADAM & EVE P1: authored scaffolding — the theme becomes reachable
    # once kinship episodes exist (union kind, P3+); the words are hers
    # via the P1 teaching.
    "kinship":  ("اثنان يمشيان معًا في الوادي", "بينهما طفل صغير يركض ويضحك"),
}

RECOUNT_TEMPLATES = {
    # her own voice (first person, feminine) — only with words she KNOWS
    "swim":    "حلمت أني أسبح في نهر كبير",
    "eat":     "حلمت أني آكل فاكهة كثيرة",
    "drink":   "حلمت أني أشرب ماءً صافيًا",
    "gather":  "حلمت أني أجمع حجارة تضيء",
    "build":   "حلمت أني أبني برجًا عاليًا",
    "shelter": "حلمت أني أنام في مأوى من ضوء",
    "storm":   "حلمت بعاصفة جميلة",
    "explore": "حلمت أني أذهب إلى وادٍ جديد",
    "kinship": "حلمت أني مع زوج وعائلة",
}

# theme -> word ids the recount needs her to know
RECOUNT_WORDS = {
    "swim":    ("dream", "swim", "river"),
    "eat":     ("dream", "eat", "fruit", "much"),
    "drink":   ("dream", "drink", "water"),
    "gather":  ("dream", "gather", "stone"),
    "build":   ("dream", "build"),
    "shelter": ("dream", "sleep_v", "shelter"),
    "storm":   ("dream",),
    "explore": ("dream", "go", "valley"),
    "kinship": ("dream", "pair", "family"),
}

VIVID_RECOUNT_THRESHOLD = 0.45   # only vivid dreams get a morning recount
DREAM_WINDOW_TICKS = 3 * 1440    # weave from the last 3 days of episodes
MOOD_VIVID_GAIN = 0.25           # good mood -> more vivid dreams


class DreamWeaver:
    def __init__(self):
        self.dreams = []          # list of dream dicts — NEVER episodes
        self.theme_weights = {}   # theme -> float (aspiration fuel)
        self.unrecounted_id = None
        self._next_id = 1
        self.metrics = {"woven": 0, "recounted": 0}

    # ---------------- weaving (runs at sleep onset) ----------------
    def weave(self, episodic, tick, rng, mood=0.0):
        """Recombine recent strong episodes into one dream scene.
        Returns the dream dict, or None if there is too little to dream of.
        Mood colors the night: good mood -> more vivid dreams."""
        eps = [e for e in episodic.episodes
               if tick - e["tick"] <= DREAM_WINDOW_TICKS]
        if len(eps) < 2:
            return None
        weights = [max(0.05, e["strength"]) * (0.5 + max(0.0, e["arousal"]))
                   for e in eps]
        k = min(len(eps), 2 + int(rng.uniform() * 3))  # 2..4 sources
        sources = _weighted_sample(eps, weights, k, rng)
        # theme: the strongest source's theme (ties -> first)
        sources.sort(key=lambda e: e["strength"], reverse=True)
        theme = KIND_THEME.get(sources[0]["kind"], "explore")
        vividness = sum(max(0.0, e["arousal"]) for e in sources) / len(sources)
        vividness = min(1.0, 0.25 + 0.75 * vividness)
        vividness = min(1.0, max(0.0, vividness * (1.0 + MOOD_VIVID_GAIN * mood)))
        frag_a, frag_b = THEME_FRAGMENTS[theme]
        # recombine: with some probability swap in the second source's theme
        # fragment — this is what makes it a DREAM, not a replay
        if len(sources) > 1 and rng.uniform() < 0.5:
            theme2 = KIND_THEME.get(sources[1]["kind"], "explore")
            if theme2 != theme:
                frag_b = THEME_FRAGMENTS[theme2][1]
        dream = {
            "id": self._next_id,
            "tick": tick,
            "day": tick // 1440 + 1,
            "type": "dream",                       # NEVER "episode"
            "sources": [e["tick"] for e in sources],  # provenance
            "theme": theme,
            "vividness": round(vividness, 3),
            "journal_text": f"حلم — {frag_a}، {frag_b}",
            "recounted": False,
            "named": False,               # host named the experience for her
        }
        self._next_id += 1
        self.dreams.append(dream)
        self.theme_weights[theme] = self.theme_weights.get(theme, 0.0) + vividness
        if vividness >= VIVID_RECOUNT_THRESHOLD and self.unrecounted_id is None:
            self.unrecounted_id = dream["id"]
        self.metrics["woven"] += 1
        return dream

    def get(self, dream_id):
        for d in self.dreams:
            if d["id"] == dream_id:
                return d
        return None

    def mark_recounted(self, dream_id):
        d = self.get(dream_id)
        if d is not None:
            d["recounted"] = True
            self.metrics["recounted"] += 1
        if self.unrecounted_id == dream_id:
            self.unrecounted_id = None

    # ---------------- persistence ----------------
    def to_dict(self):
        return {"dreams": [dict(d) for d in self.dreams],
                "theme_weights": dict(self.theme_weights),
                "unrecounted_id": self.unrecounted_id,
                "next_id": self._next_id,
                "metrics": dict(self.metrics)}

    @classmethod
    def from_dict(cls, d):
        w = cls.__new__(cls)
        w.dreams = [dict(x) for x in d.get("dreams", [])]
        w.theme_weights = dict(d.get("theme_weights", {}))
        w.unrecounted_id = d.get("unrecounted_id")
        w._next_id = d.get("next_id", 1)
        w.metrics = dict(d.get("metrics", {"woven": 0, "recounted": 0}))
        return w


def _weighted_sample(items, weights, k, rng):
    """Deterministic weighted sample without replacement (Efraimidis-Spirakis)."""
    scored = []
    for it, w in zip(items, weights):
        u = rng.uniform()
        key = u ** (1.0 / max(1e-9, w))
        scored.append((key, it))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [it for _, it in scored[:k]]
