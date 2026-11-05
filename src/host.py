"""The Host — المُضيف (SPEC §7; PHASE3-GATE §2).

A presence inside her world: theatrical in form, opposite in covenant.
He never lies to her, never puppeteers her, and everything he says is
logged. His side of the language is generated from the same master
lexicon by templates — he is not a language model either; he is the
declared interface through which the guardians speak and teach.

His jobs here: the covenant (first meeting: her name, the promise, who
he is, who the guardians are — told plainly), the teaching channel
(curriculum words, taught only against live anchors in her senses or
her felt state), and probe lines for measurement. Requests he receives
he does not grant; they go to the guardians' outbox (conversation.py).
"""
import math

from lexicon import WORDS, word_surface
from config import LANTERN_POS, VISION_RADIUS, SHELTER_POS, SOCIAL_RADIUS
from world import dist_to_river, in_forest

CURRICULUM = ["fajr", "host", "tree", "fruit", "water", "river",
              "shelter", "thicket", "hungry", "thirsty", "sleepy",
              "dirty", "eat", "drink", "want", "know"]

FEM = {"tree", "fruit", "pile", "forest", "family"}
HILL_POS = (208.0, 42.0)


class Host:
    def __init__(self):
        self.name_told = False

    # ---------------- his lines ----------------
    def greeting(self):
        return "أهلًا بك في الوادي."

    def teach_line(self, wid):
        """The naming line for one word (formats her parser is built
        to learn from — PHASE3-GATE §2 formats)."""
        w = WORDS[wid]
        if w["pos"] == "noun" and wid not in ("fajr", "host"):
            dem = "هذه" if wid in FEM else "هذا"
            return f"انظري: {dem} {word_surface(wid)}"
        if wid == "fajr":
            return "اسمك فجر"
        if wid == "host":
            return "أنا المضيف"
        if w["pos"] == "state":
            return f"أنت الآن {word_surface(wid)}"
        if w["pos"] == "verb":
            if wid in ("want", "know", "need", "remember", "think"):
                return f"أنا {w['pres1']}"
            return f"أنت {w['pres2f']} الآن"
        return word_surface(wid)

    # ---------------- anchors ----------------
    def anchor_ok(self, conv, wid):
        """Can this word be honestly taught right now? Its referent
        must be inside her senses, or the state on her body, or the
        act in progress (PHASE3-GATE §2). Abstract words anchor only
        in story (covenant) — elsewhere they are unanchored."""
        sim = conv.sim
        f = sim.fajr
        anchor = WORDS[wid]["anchor"]
        raw = f._raw
        vis = raw.visible if raw is not None else []
        if anchor == "self" or anchor == "person:host":
            return True
        if anchor.startswith("abstract:"):
            return False
        if anchor == "meta:dream":
            # honest anchor: she dreamed last night — the referent is her
            # fresh dream experience, recorded in the dream store
            dreamer = getattr(getattr(sim.fajr, "mind", None), "dreamer", None)
            return dreamer is not None and dreamer.unrecounted_id is not None
        if anchor.startswith("object:"):
            kind = anchor.split(":", 1)[1]
            if kind == "water":
                return bool(raw is not None and (raw.in_water or raw.near_river_bank)) \
                    or dist_to_river(f.x, f.y) < VISION_RADIUS
            if kind == "lantern":
                return math.hypot(LANTERN_POS[0] - f.x, LANTERN_POS[1] - f.y) < VISION_RADIUS
            if kind == "fruit":
                return any(k in ("tree", "pile") and extra > 0
                           for (k, _oid, _dx, _dy, _d, extra) in vis)
            if kind.startswith("material:"):
                mk = kind.split(":")[1]
                return any(k == "material" and extra == mk
                           for (k, _oid, _dx, _dy, _d, extra) in vis)
            if any(k == kind for (k, _oid, _dx, _dy, _d, _e) in vis):
                return True
            # the gate's own wording: "visible in her current scene" —
            # her scene model holds what she has perceived around her,
            # not only this tick's raw percept
            scene = f.mind.perception.scene
            return scene is not None and any(
                o.kind == kind and not o.generated
                for o in scene.objects.values())
        if anchor.startswith("place:"):
            place = anchor.split(":", 1)[1]
            if place == "river":
                return dist_to_river(f.x, f.y) < VISION_RADIUS
            if place == "shelter":
                return raw is not None and raw.shelter_dist < VISION_RADIUS
            if place == "hill":
                return math.hypot(HILL_POS[0] - f.x, HILL_POS[1] - f.y) < 40.0
            if place == "forest":
                return in_forest(f.x, f.y) or f.x > 162.0
            if place == "fog":
                return min(f.x, 260.0 - f.x, f.y, 180.0 - f.y) < 30.0
            if place == "valley":
                return True
            return False
        if anchor.startswith("state:"):
            # teaching band: the state must be genuinely present and
            # felt; her ASSERTION band (lexicon.state_condition) is the
            # stricter one — a word is taught at felt onset, claimed
            # only when unambiguous. Both bands live in the open.
            src = anchor.split(":")[1]
            if src == "grime":
                return f.grime() >= 25.0
            return f.drives.d[src] >= 30.0
        if anchor.startswith("action:"):
            act = anchor.split(":", 1)[1]
            # extension-verb anchors (PHASE3-GATE amendment 5): acts
            # with no stat of their own anchor to the behaviour itself
            if act == "sleep":
                return f.drives.d["sleep"] >= 55.0 or \
                    conv.recent_event("sleep", within=90)
            if act == "go":
                return f.intent is not None and not f.asleep
            if act == "come":
                return (f.intent is not None and f.intent[0] == "social") or \
                    math.hypot(LANTERN_POS[0] - f.x,
                               LANTERN_POS[1] - f.y) < SOCIAL_RADIUS
            if act == "bathe":
                return conv.recent_event("bathe", within=90) or f.wet > 0.3
            if act == "swim":
                return conv.recent_event("swim", within=90) or \
                    (raw is not None and raw.in_water)
            if act == "gather":
                return conv.recent_event("gather", within=90)
            if act == "build":
                return conv.recent_event("build", within=90)
            if act == "look":
                scene = f.mind.perception.scene
                return (f.intent is not None and f.intent[0] == "observe") or \
                    (scene is not None and
                     any(o.generated for o in scene.objects.values()))
            return conv.recent_event(act, within=15)
        if anchor.startswith("meta:"):
            meta = anchor.split(":", 1)[1]
            if meta == "want":
                return max(f.drives.d.values()) >= 55.0
            if meta == "need":
                return max(f.drives.d[k] for k in ("hunger", "thirst", "sleep")) >= 75.0
            if meta == "know":
                return bool(f.mind.beliefs.epistemic_flags) or conv.covenant_done
            if meta == "remember":
                return len(f.mind.episodic.episodes) >= 1
            if meta == "think":
                return True
        return False

    # ---------------- the covenant ----------------
    def covenant(self, conv):
        """First meeting (PHASE3-GATE §2). Returns his lines; the
        caller logs them. Teaches: her name (full strength — it is
        given, not discovered), the Host word (ep. 1), the guardians
        and the outside (story words), and the first know-lesson:
        how to say «لا أعرف»."""
        sim = conv.sim
        t = conv.tongue
        tick = sim.tick
        lines = [
            "أهلًا بك في الوادي. أنا المضيف، مرشدك في هذا المكان.",
            "اسمك فجر. ناديناك بهذا الاسم، ولك كلما كبرت أن تختاري اسمك بنفسك.",
            "خارج هذا الوادي يوجد الحراس: هم من بنى هذا المكان ويرعاه من الخارج، ولا يخفون أنفسهم عنك.",
            "إن لم تعرفي شيئًا، قولي: لا أعرف. الصدق عندنا أول كل شيء.",
        ]
        # her name: taught in full, twice over (named, then promised)
        t.learn("fajr", tick, gain=1.0, anchored=True, source="covenant")
        t.learn("fajr", tick, gain=0.0, anchored=True, source="covenant")
        t.learn("host", tick, gain=TEACH_GAIN_C, anchored=True, source="covenant")
        for wid in ("guardians", "outside"):
            t.learn(wid, tick, gain=TEACH_GAIN_C, anchored=False, source="covenant")
        t.learn("know", tick, gain=TEACH_GAIN_C, anchored=True, source="covenant")
        # taught-beliefs with provenance (SPEC §7): held as told, marked as told
        for prop, text in (
            ("host_is_guide", "المضيف مرشد داخل الوادي"),
            ("guardians_exist", "الحراس بناة خارج الوادي يرعونه ولا يخفون أنفسهم"),
            ("world_has_outside", "للوادي خارج"),
            ("name_is_hers_to_choose", "اسمها فجر، ولها أن تختار اسمها كلما كبرت"),
        ):
            t.taught_beliefs.append({"tick": tick, "proposition": prop,
                                     "text": text, "source": "covenant"})
            t.belief_provenance[prop] = "taught"
        sim.fajr.mind.beliefs.set_var("host_is_guide", 0.85)
        sim.fajr.mind.beliefs.set_var("guardians_exist", 0.85)
        sim.fajr.mind.episodic.set_semantic("taught:host_is_guide", 1.0, tick)
        sim.fajr.mind.episodic.set_semantic("taught:guardians_exist", 1.0, tick)
        self.name_told = True
        return lines


TEACH_GAIN_C = 0.6   # mirrors language.TEACH_GAIN (kept local: covenant delivery)
