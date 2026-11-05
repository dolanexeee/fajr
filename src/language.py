"""Her tongue — Phase 3 (PHASE3-GATE §1, §3).

Her personal language state and the two directions of her language:

  PRODUCTION: a compositional generator over the words she KNOWS,
  rendering speech acts from her real state — drives and body (felt
  bands), beliefs (with the monitor-gated confidence that produced
  them; hedged by band; her beliefs may be wrong and are reported as
  hers), episodes, self-model, current intent. Every utterance is
  logged with structured grounding claims resolvable to those sources.

  COMPREHENSION: a parser over the SAME lexicon/grammar, checked
  against HER known set. A content word she has not learned makes the
  parse fail on that word; the failure is reported, never papered over.

No LLM. Deterministic. All state here is hers and serialisable.
"""
import math

from lexicon import (WORDS, FORM_MAP, GLUE_N, tokenize, word_surface,
                     state_condition, state_value, place_word_at)
from config import SHELTER_POS, LANTERN_POS
from world import dist_to_river

TEACH_GAIN = 0.6          # PHASE3-GATE §1 (frozen)
USE_GAIN = 0.25
DECAY_PER_TICK = 0.00012
KNOW_THRESHOLD = 0.5

FEM_NOUNS = {"tree", "fruit", "pile", "forest", "family"}

DRIVE_STATE_WORDS = {     # drive -> (high-state word, low-state word)
    "hunger": ("hungry", "satiated"),
    "thirst": ("thirsty", None),
    "sleep": ("sleepy", None),
    "safety": ("afraid", None),
    "comfort": (None, "comfortable"),
}
DRIVE_VERBS = {"hunger": "eat", "thirst": "drink", "sleep": "sleep_v",
               "safety": None, "comfort": None}
EPISODE_VERBS = {"eat": "eat", "drink": "drink", "sleep": "sleep_v",
                 "bathe": "bathe", "swim": "swim", "gather": "gather",
                 "place": "build"}
OBJECT_WORD_BY_KIND = {"tree": "tree", "thicket": "thicket",
                       "pile": "pile", "structure": "structure",
                       "shelter": "shelter"}


def drive_band(v):
    return "low" if v < 35.0 else ("mid" if v <= 65.0 else "high")


class Tongue:
    def __init__(self):
        self.words = {}            # wid -> {strength, taught_tick, episodes, uses}
        self.learning_log = []     # word-learning episodes (her teaching memory)
        self.taught_beliefs = []   # {tick, proposition, text, source}
        self.belief_provenance = {}  # proposition name -> "taught"
        self.utterances = []       # {tick, day, context, text, claims}
        self.heard_unknown = []    # {tick, surface} words said to her, unknown to her

    # ---------------- knowledge state ----------------
    def known(self, wid):
        w = self.words.get(wid)
        return bool(w) and w["strength"] >= KNOW_THRESHOLD

    def recognises(self, wid):
        """Receptive knowledge: she can UNDERSTAND a word well below
        the strength at which she would produce it herself (0.25 vs
        the 0.50 production threshold). Comprehension is checked
        against this; production never is."""
        w = self.words.get(wid)
        return bool(w) and w["strength"] >= 0.25

    def strength(self, wid):
        w = self.words.get(wid)
        return 0.0 if w is None else w["strength"]

    def learn(self, wid, tick, gain=TEACH_GAIN, anchored=True, source="host"):
        w = self.words.setdefault(wid, {"strength": 0.0, "taught_tick": tick,
                                        "episodes": 0, "uses": 0})
        w["strength"] = min(1.0, w["strength"] + gain)
        w["taught_tick"] = tick
        w["episodes"] += 1
        ep = {"tick": tick, "word": wid, "surface": word_surface(wid),
              "anchored": anchored, "source": source,
              "strength_after": round(w["strength"], 3)}
        self.learning_log.append(ep)
        return ep

    def _use(self, wid):
        w = self.words.get(wid)
        if w is not None:
            w["strength"] = min(1.0, w["strength"] + USE_GAIN)
            w["uses"] += 1

    def decay(self, ticks=1):
        f = (1.0 - DECAY_PER_TICK) ** ticks
        for w in self.words.values():
            w["strength"] *= f

    # ---------------- utterance plumbing ----------------
    def _emit(self, tick, day, context, text, claims, used_words):
        for wid in used_words:
            self._use(wid)
        u = {"tick": tick, "day": day, "context": context,
             "text": text, "claims": claims}
        self.utterances.append(u)
        return u

    # ================================================================ #
    # PRODUCTION                                                        #
    # ================================================================ #
    def _state_report(self, sim, state_wid):
        """A report of one felt state, if the word is known and the
        state truly holds. Returns (text, claims, used) or None."""
        if not self.known(state_wid):
            return None
        if not state_condition(state_wid, sim):
            return None
        anchor = WORDS[state_wid]["anchor"]
        _, src, direction = anchor.split(":")
        value = state_value(state_wid, sim)
        text = f"أنا {word_surface(state_wid)}"
        if direction == "high" and value >= 80.0:
            text += " جدًا"
        claims = [{"type": "state", "word": state_wid, "src": src,
                   "value": round(value, 1), "band": drive_band(value)}]
        return text, claims, [state_wid]

    def dominant_report(self, sim, context="opener"):
        """Her unprompted account of her strongest felt need (with a
        want-clause when she has the words)."""
        f = sim.fajr
        best = None
        for drive, (hi, lo) in DRIVE_STATE_WORDS.items():
            for wid in (hi, lo):
                if wid and self._state_report(sim, wid):
                    v = f.drives.d[drive]
                    sal = v if WORDS[wid]["anchor"].endswith("high") else (100.0 - v)
                    if best is None or sal > best[0]:
                        best = (sal, wid, drive)
        for wid in ("dirty", "clean", "wet"):
            if self._state_report(sim, wid):
                v = f.grime() if wid != "wet" else f.wet * 100.0
                sal = v if wid != "clean" else 60.0
                if best is None or sal > best[0]:
                    best = (sal, wid, "comfort")
        if best is None:
            return None
        _, wid, drive = best
        text, claims, used = self._state_report(sim, wid)
        verb = DRIVE_VERBS.get(drive)
        if verb and self.known("want") and self.known(verb):
            text += f". {WORDS['want']['pres1']} أن {WORDS[verb]['pres1']}"
            claims.append({"type": "intent", "verb": verb})
            used += ["want", verb]
        return self._emit(sim.tick, sim.tick // 1440 + 1, context, text, claims, used)

    def nearest_belief(self, sim, kind):
        f = sim.fajr
        cands = [b for b in f.mind.beliefs.objects.values() if b["kind"] == kind]
        if not cands:
            return None
        return min(cands, key=lambda b: math.hypot(b["x"] - f.x, b["y"] - f.y))

    def belief_report(self, sim, belief, context="reply"):
        """What she holds about one believed object, hedged by the
        belief's own confidence band (PHASE3-GATE §3)."""
        kind = belief["kind"]
        wid = OBJECT_WORD_BY_KIND.get(kind)
        if wid is None or not self.known(wid):
            return None
        conf = belief["conf"]
        hedge = "plain" if conf >= 0.70 else ("think" if conf >= 0.45 else "unsure")
        noun = word_surface(wid)
        claims = [{"type": "belief", "key": f"{kind}:{belief['oid']}",
                   "fruit_est": round(belief["fruit_est"], 2),
                   "conf": round(conf, 3), "hedge": hedge}]
        used = [wid]
        fruit_claim = None
        if kind in ("tree", "pile") and self.known("fruit"):
            fe = belief["fruit_est"]
            if fe >= 2.5:
                q = " كثيرة" if self.known("much") else ""
                fruit_claim = f"فيها فاكهة{q}"
                if q:
                    used.append("much")
            elif fe >= 0.6:
                q = " قليلة" if self.known("little") else ""
                fruit_claim = f"فيها فاكهة{q}"
                if q:
                    used.append("little")
            elif self.known("empty"):
                fruit_claim = "فارغة"
                used.append("empty")
            if fruit_claim:
                used.append("fruit")
                claims[0]["fruit_band"] = ("much" if fe >= 2.5 else
                                           "little" if fe >= 0.6 else "empty")
        if hedge == "unsure":
            # an unsure belief is reported as uncertainty, not as fact
            text = f"لست متأكدة من {noun}"
            claims[0]["asserted"] = False
        else:
            prefix = "أظن أن " if hedge == "think" else ""
            if fruit_claim == "فارغة":
                text = f"{prefix}{noun} فارغة"
            elif fruit_claim:
                text = f"{prefix}{noun} {fruit_claim}"
            else:
                pw = place_word_at(belief["x"], belief["y"])
                if pw and self.known(pw):
                    text = f"{prefix}{noun} عند {word_surface(pw)}"
                    claims.append({"type": "place", "word": pw,
                                   "x": belief["x"], "y": belief["y"]})
                    used.append(pw)
                else:
                    text = f"{prefix}رأيت {noun}"
            claims[0]["asserted"] = True
        return self._emit(sim.tick, sim.tick // 1440 + 1, context, text, claims, used)

    def memory_report(self, sim, episode, context="reply"):
        verb = EPISODE_VERBS.get(episode["kind"])
        if verb is None or not self.known(verb):
            return None
        today = sim.tick // 1440 + 1
        parts, used = [], [verb]
        if episode["day"] == today and self.known("today"):
            parts.append(word_surface("today"))
            used.append("today")
        elif episode["day"] == today - 1 and self.known("yesterday"):
            parts.append(word_surface("yesterday"))
            used.append("yesterday")
        parts.append(WORDS[verb]["past1"])
        if episode["kind"] == "eat" and self.known("fruit"):
            parts.append(word_surface("fruit"))
            used.append("fruit")
        pw = place_word_at(episode["x"], episode["y"])
        if pw and self.known(pw):
            parts.append(f"عند {word_surface(pw)}")
            used.append(pw)
        claims = [{"type": "memory", "episode_tick": episode["tick"],
                   "kind": episode["kind"], "day": episode["day"],
                   "x": episode["x"], "y": episode["y"]}]
        return self._emit(sim.tick, today, context, " ".join(parts), claims, used)

    def request_utterance(self, sim, drive):
        """PHASE3-GATE §5 frozen form: state + absence + ask."""
        hi, _ = DRIVE_STATE_WORDS[drive]
        if not self.known(hi):
            return None
        miss_wid = "fruit" if drive == "hunger" else "water"
        if not self.known(miss_wid):
            return None
        text, claims, used = self._state_report(sim, hi)
        text += f". لا توجد {word_surface(miss_wid)}"
        claims.append({"type": "absence", "word": miss_wid, "drive": drive})
        used.append(miss_wid)
        ask = "need" if self.known("need") else ("want" if self.known("want") else None)
        if ask:
            text += f". {WORDS[ask]['pres1']} {word_surface(miss_wid)}"
            claims.append({"type": "request", "drive": drive, "word": miss_wid})
            used.append(ask)
        return self._emit(sim.tick, sim.tick // 1440 + 1, "request",
                          text, claims, used)

    def unknown_utterance(self, sim, context="reply", surface=None):
        if not self.known("know"):
            return None
        if surface:
            text = f"لا أعرف كلمة «{surface}»"
            claims = [{"type": "unknown_answer", "word_surface": surface}]
        else:
            text = "لا أعرف"
            claims = [{"type": "unknown_answer", "word_surface": None}]
        return self._emit(sim.tick, sim.tick // 1440 + 1, context,
                          text, claims, ["know"])

    def self_name(self, sim, context="reply"):
        if not self.known("fajr"):
            return None
        return self._emit(sim.tick, sim.tick // 1440 + 1, context, "أنا فجر",
                          [{"type": "self", "field": "name", "value": "فجر"}],
                          ["fajr"])

    # ================================================================ #
    # COMPREHENSION                                                     #
    # ================================================================ #
    def parse(self, text):
        """Parse a line spoken TO her. Returns a meaning dict. Content
        words are word ids; `unknown` lists raw tokens in NO word's
        forms (unknown to the language itself); whether a parsed word
        is known TO HER is the caller's check against self.known()."""
        raw = tokenize(text)
        content, unknown = [], []
        has = set()
        for tok_raw, tok in raw:
            has.add(tok)
            if tok in GLUE_N:
                continue
            hit = FORM_MAP.get(tok)
            if hit:
                content.append((hit[0], hit[1], tok_raw))
            else:
                unknown.append(tok_raw)
        wids = [c[0] for c in content]
        forms = {c[0]: c[1] for c in content}
        out = {"kind": "unknown", "content": content, "unknown": unknown,
               "text": text}
        # ---- teaching formats (checked first; PHASE3-GATE §2) ----
        for wid, form, _s in content:
            if wid == "fajr" and "اسمك" in text:
                out.update(kind="teaching", word="fajr")
                return out
            if wid == "host" and forms.get("host") == "ar" and len(wids) <= 2:
                out.update(kind="teaching", word="host")
                return out
            if WORDS[wid]["pos"] == "state" and "الان" in has:
                out.update(kind="teaching", word=wid)
                return out
            if WORDS[wid]["pos"] == "verb" and form in ("pres2f", "pres1"):
                out.update(kind="teaching", word=wid)
                return out
            if WORDS[wid]["pos"] == "noun" and wid not in ("fajr", "host") \
                    and ("هذا" in text or "هذه" in text):
                out.update(kind="teaching", word=wid)
                return out
        # ---- instruction: a feminine imperative leads ----
        for wid, form, _s in content:
            if form == "impf":
                place = next((w for w in wids if w in
                              ("river", "shelter", "hill", "forest", "lantern", "valley")), None)
                obj = next((w for w in wids if WORDS[w]["pos"] == "noun"), None)
                out.update(kind="instruction", verb=wid, place=place, object=obj)
                return out
        # ---- questions ----
        if "هل" in text or any(t in has for t in ("هل",)):
            for wid in wids:
                if WORDS[wid]["pos"] == "state":
                    out.update(kind="q_state", word=wid)
                    return out
            for wid in wids:
                if wid in OBJECT_WORD_BY_KIND.values():
                    out.update(kind="q_belief", word=wid)
                    return out
            for wid in wids:
                if WORDS[wid]["pos"] == "noun":
                    out.update(kind="q_belief", word=wid)
                    return out
        if ("ما" in text or "ماذا" in text) and ("هذا" in text or "هذه" in text):
            out.update(kind="q_identify")
            return out
        if "اسمك" in text:
            out.update(kind="q_name")
            return out
        if "اين" in has or "أين" in text:
            for wid in wids:
                if WORDS[wid]["pos"] == "noun":
                    out.update(kind="q_where", word=wid)
                    return out
        if "ماذا" in text:
            for wid, form, _s in content:
                if WORDS[wid]["pos"] == "verb" and form in ("past1", "pres2f"):
                    out.update(kind="q_memory", word=wid)
                    return out
            out.update(kind="q_memory", word=None)
            return out
        if "من" in text and "أنا" in text or "من أنا" in text:
            out.update(kind="q_who")
            return out
        # ---- fact: noun + في + place ----
        nouns = [w for w in wids if WORDS[w]["pos"] == "noun"]
        places = [w for w in wids if w in
                  ("river", "shelter", "hill", "forest", "lantern", "valley")]
        if nouns and places and ("في" in text or "عند" in text):
            out.update(kind="fact", subject=nouns[0], place=places[0])
            return out
        if any(g in has for g in ("مرحبا", "اهلا", "سلام")):
            out.update(kind="greeting")
            return out
        if content and not unknown:
            out.update(kind="unparsed")
        return out

    # ---------------- serialisation ----------------
    def to_dict(self):
        return {"words": {k: dict(v) for k, v in self.words.items()},
                "learning_log": [dict(e) for e in self.learning_log],
                "taught_beliefs": [dict(b) for b in self.taught_beliefs],
                "belief_provenance": dict(self.belief_provenance),
                "utterances": [dict(u) for u in self.utterances],
                "heard_unknown": [dict(u) for u in self.heard_unknown]}

    @classmethod
    def from_dict(cls, d):
        t = cls.__new__(cls)
        t.words = {k: dict(v) for k, v in d["words"].items()}
        t.learning_log = [dict(e) for e in d["learning_log"]]
        t.taught_beliefs = [dict(b) for b in d["taught_beliefs"]]
        t.belief_provenance = dict(d["belief_provenance"])
        t.utterances = [dict(u) for u in d["utterances"]]
        t.heard_unknown = [dict(u) for u in d["heard_unknown"]]
        return t
