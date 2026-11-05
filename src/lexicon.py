"""The shared master lexicon + grammar for Phase 3 (PHASE3-GATE §1).

ONE inventory of content words and ONE grammar, used by both her
generator and her parser — she understands exactly the language she can
speak, and nothing else. Words are data: surface form, part of speech,
and a grounding ANCHOR saying what in her world/body a word refers to.

Anchors:
  self / person:host / abstract:*     persons and taught-only concepts
  object:<percept kind>               things her senses deliver
  place:<name>                        named regions of her valley
  state:<drive>:<high|low>            felt interoceptive bands
  action:<intent/event kind>          things she does
  meta:*                              epistemic/volitional verbs
  quant:* / time:*                    quantity and day-reference words

Grammar glue (particles, pronouns, demonstratives, question particles
هل/ما/أين) is scaffolding, not knowledge: it carries no referent, so it
is not taught and not counted as knowing. Content words are knowledge.
"""
import re

# verb forms: pres1 = her first-person present («آكل»), past1 = her
# first-person past («أكلت»), impf = feminine imperative («كلي»),
# pres2f = second-person feminine present («تأكلين», the Host's naming
# form when he teaches a verb from her live action).
WORDS = {
    # ---- persons ----
    "fajr":      {"ar": "فجر", "pos": "noun", "anchor": "self"},
    "host":      {"ar": "المضيف", "pos": "noun", "anchor": "person:host"},
    "guardians": {"ar": "الحراس", "pos": "noun", "anchor": "abstract:guardians"},
    "outside":   {"ar": "خارج", "pos": "noun", "anchor": "abstract:outside"},
    # ---- kinship concepts (ADAM & EVE P1: taught-only, like guardians) ----
    "pair":      {"ar": "زوج", "pos": "noun", "anchor": "abstract:kinship"},
    "family":    {"ar": "عائلة", "pos": "noun", "anchor": "abstract:kinship"},
    "newlife":   {"ar": "حياة جديدة", "pos": "noun", "anchor": "abstract:kinship"},
    # ---- maker concepts (2026-11-04, his order: Guide introduces Mustafa;
    #      taught-only like kinship words — the Guide names, she learns) ----
    "maker":     {"ar": "صانع", "pos": "noun", "anchor": "abstract:maker"},
    "mustafa":   {"ar": "مصطفى", "pos": "noun", "anchor": "abstract:maker"},
    # ---- perceivable objects & places ----
    "tree":      {"ar": "شجرة", "pos": "noun", "anchor": "object:tree"},
    "fruit":     {"ar": "فاكهة", "pos": "noun", "anchor": "object:fruit"},
    "pile":      {"ar": "كومة", "pos": "noun", "anchor": "object:pile"},
    "thicket":   {"ar": "شوك", "pos": "noun", "anchor": "object:thicket"},
    "wood":      {"ar": "خشب", "pos": "noun", "anchor": "object:material:wood"},
    "stone":     {"ar": "حجر", "pos": "noun", "anchor": "object:material:stone"},
    "structure": {"ar": "بناء", "pos": "noun", "anchor": "object:structure"},
    "lantern":   {"ar": "مصباح", "pos": "noun", "anchor": "object:lantern"},
    "water":     {"ar": "ماء", "pos": "noun", "anchor": "object:water"},
    "shelter":   {"ar": "مأوى", "pos": "noun", "anchor": "place:shelter"},
    "river":     {"ar": "نهر", "pos": "noun", "anchor": "place:river"},
    "hill":      {"ar": "تل", "pos": "noun", "anchor": "place:hill"},
    "forest":    {"ar": "غابة", "pos": "noun", "anchor": "place:forest"},
    "fog":       {"ar": "ضباب", "pos": "noun", "anchor": "place:fog"},
    "valley":    {"ar": "الوادي", "pos": "noun", "anchor": "place:valley"},
    # ---- felt states (feminine surfaces: she speaks of herself) ----
    "hungry":      {"ar": "جائعة", "pos": "state", "anchor": "state:hunger:high", "masc": "جائع"},
    "satiated":    {"ar": "شبعانة", "pos": "state", "anchor": "state:hunger:low", "masc": "شبعان"},
    "thirsty":     {"ar": "عطشانة", "pos": "state", "anchor": "state:thirst:high", "masc": "عطشان"},
    "sleepy":      {"ar": "نعسانة", "pos": "state", "anchor": "state:sleep:high", "masc": "نعسان"},
    "afraid":      {"ar": "خائفة", "pos": "state", "anchor": "state:safety:high", "masc": "خائف"},
    "comfortable": {"ar": "مرتاحة", "pos": "state", "anchor": "state:comfort:low", "masc": "مرتاح"},
    "dirty":       {"ar": "وسخة", "pos": "state", "anchor": "state:grime:high", "masc": "وسخ"},
    "clean":       {"ar": "نظيفة", "pos": "state", "anchor": "state:grime:low", "masc": "نظيف"},
    "wet":         {"ar": "مبللة", "pos": "state", "anchor": "state:wet:high", "masc": "مبلل"},
    # ---- verbs ----
    "eat":      {"ar": "أكل", "pos": "verb", "anchor": "action:eat",
                 "pres1": "آكل", "past1": "أكلت", "impf": "كلي", "pres2f": "تأكلين"},
    "drink":    {"ar": "شرب", "pos": "verb", "anchor": "action:drink",
                 "pres1": "أشرب", "past1": "شربت", "impf": "اشربي", "pres2f": "تشربين"},
    "sleep_v":  {"ar": "نام", "pos": "verb", "anchor": "action:sleep",
                 "pres1": "أنام", "past1": "نمت", "impf": "نامي", "pres2f": "تنامين"},
    "go":       {"ar": "ذهب", "pos": "verb", "anchor": "action:go",
                 "pres1": "أذهب", "past1": "ذهبت", "impf": "اذهبي", "pres2f": "تذهبين"},
    "come":     {"ar": "جاء", "pos": "verb", "anchor": "action:come",
                 "pres1": "آتي", "past1": "جئت", "impf": "تعالي", "pres2f": "تأتين"},
    "bathe":    {"ar": "استحم", "pos": "verb", "anchor": "action:bathe",
                 "pres1": "أستحم", "past1": "استحممت", "impf": "استحمي", "pres2f": "تستحمين"},
    "swim":     {"ar": "سبح", "pos": "verb", "anchor": "action:swim",
                 "pres1": "أسبح", "past1": "سبحت", "impf": "اسبحي", "pres2f": "تسبحين"},
    "gather":   {"ar": "جمع", "pos": "verb", "anchor": "action:gather",
                 "pres1": "أجمع", "past1": "جمعت", "impf": "اجمعي", "pres2f": "تجمعين"},
    "build":    {"ar": "بنى", "pos": "verb", "anchor": "action:build",
                 "pres1": "أبني", "past1": "بنيت", "impf": "ابني", "pres2f": "تبنين"},
    "look":     {"ar": "نظر", "pos": "verb", "anchor": "action:look",
                 "pres1": "أنظر", "past1": "نظرت", "impf": "انظري", "pres2f": "تنظرين"},
    "know":     {"ar": "عرف", "pos": "verb", "anchor": "meta:know",
                 "pres1": "أعرف", "past1": "عرفت", "impf": None, "pres2f": "تعرفين"},
    "want":     {"ar": "أريد", "pos": "verb", "anchor": "meta:want",
                 "pres1": "أريد", "past1": "أردت", "impf": None, "pres2f": "تريدين"},
    "need":     {"ar": "أحتاج", "pos": "verb", "anchor": "meta:need",
                 "pres1": "أحتاج", "past1": "احتجت", "impf": None, "pres2f": "تحتاجين"},
    "remember": {"ar": "تذكر", "pos": "verb", "anchor": "meta:remember",
                 "pres1": "أتذكر", "past1": "تذكرت", "impf": None, "pres2f": "تتذكرين"},
    "think":    {"ar": "ظن", "pos": "verb", "anchor": "meta:think",
                 "pres1": "أظن", "past1": "ظننت", "impf": None, "pres2f": "تظنين"},
    "dream":    {"ar": "حلم", "pos": "verb", "anchor": "meta:dream",
                 "pres1": "أحلم", "past1": "حلمت", "impf": None, "pres2f": "تحلمين"},
    # ---- quantity / time ----
    "much":      {"ar": "كثيرة", "pos": "quant", "anchor": "quant:much", "masc": "كثير"},
    "little":    {"ar": "قليلة", "pos": "quant", "anchor": "quant:little", "masc": "قليل"},
    "empty":     {"ar": "فارغة", "pos": "quant", "anchor": "quant:empty", "masc": "فارغ"},
    "today":     {"ar": "اليوم", "pos": "time", "anchor": "time:today"},
    "yesterday": {"ar": "أمس", "pos": "time", "anchor": "time:yesterday"},
    "tomorrow":  {"ar": "غدًا", "pos": "time", "anchor": "time:tomorrow"},
}

CURRICULUM = ["fajr", "host", "tree", "fruit", "water", "river",
              "shelter", "thicket", "hungry", "thirsty", "sleepy",
              "dirty", "eat", "drink", "want", "know"]

# grammar glue (normalised forms): scaffolding, never knowledge
GLUE = ["في", "من", "إلى", "الى", "أن", "ان", "أنا", "انا", "أنت", "انت",
        "أنتِ", "هذا", "هذه", "هنا", "هناك", "لا", "نعم", "الآن", "الان",
        "هل", "ما", "أين", "اين", "ماذا", "من", "اسمك", "اسمي",
        "عند", "عنده", "مع", "هو", "هي", "أنتَ", "لك", "لي",
        "فعلت", "تفعلينه", "شيء", "شيا", "مرحبا", "أهلا", "اهلا", "سلام",
        "بك", "كلما", "كبرت", "إن", "لم", "قولي", "يوجد", "توجد", "مرشدك",
        "مرشد", "بنفسك", "بهذا", "الاسم", "فيها", "فيه"]
GLUE_N = set()


_DIACRITICS = re.compile(r"[ً-ٰٟ]")  # harakat .. superscript alef


def normalise(tok):
    """Arabic normalisation for parsing (generation always emits the
    canonical surfaces). Returns the normalised token string."""
    t = _DIACRITICS.sub("", tok)
    t = t.replace("ـ", "")                       # tatweel
    t = t.strip("«»؟?.!,؛;:")
    t = t.replace("إ", "ا").replace("أ", "ا").replace("آ", "ا")
    t = t.replace("ة", "ه").replace("ى", "ي")
    return t


def tokenize(text):
    """Split text into (raw, normalised) token pairs."""
    out = []
    for part in re.split(r"\s+", text.strip()):
        if part:
            out.append((part, normalise(part)))
    return out


def _build_form_map():
    m = {}
    for wid, w in WORDS.items():
        forms = [("ar", w["ar"])]
        if w.get("masc"):
            forms.append(("ar", w["masc"]))
        # impf before pres1: for build, normalised pres1 «أبني» and
        # impf «ابني» collide — the imperative reading must win, it
        # is the instruction channel's form
        for key in ("impf", "pres1", "past1", "pres2f"):
            if w.get(key):
                forms.append((key, w[key]))
        for form_kind, surface in forms:
            n = normalise(surface)
            m.setdefault(n, (wid, form_kind))
            # definite variant for nouns that do not already carry ال
            if w["pos"] in ("noun", "state", "quant") and not surface.startswith("ال"):
                m.setdefault(normalise("ال" + surface), (wid, form_kind))
    return m


FORM_MAP = _build_form_map()
for g in GLUE:
    GLUE_N.add(normalise(g))


def word_surface(wid):
    return WORDS[wid]["ar"]


# ------------------------------------------------------- felt-state truth
# The semantic bands of state words (PHASE3-GATE §7): the generator
# asserts a state word only when its condition holds; the grounding
# checker uses THESE SAME functions against sim ground truth.
def state_value(state_wid, sim):
    src = WORDS[state_wid]["anchor"].split(":")[1]
    f = sim.fajr
    if src == "grime":
        return f.grime()
    if src == "wet":
        return f.wet * 100.0
    return f.drives.d[src]


def state_condition(state_wid, sim):
    anchor = WORDS[state_wid]["anchor"]
    _, src, direction = anchor.split(":")
    v = state_value(state_wid, sim)
    if src == "grime":
        return v >= 25.0 if direction == "high" else v <= 20.0
    if src == "wet":
        return v >= 50.0
    if direction == "high":
        return v > 35.0
    return v < 25.0 if src == "hunger" else v < 35.0


# ------------------------------------------------------------ place names
PLACE_WORDS = ("river", "shelter", "forest", "hill", "lantern", "valley")
HILL_POS = (208.0, 42.0)


def place_word_at(x, y):
    """The named place a point belongs to (nearest anchor wins)."""
    from config import SHELTER_POS, LANTERN_POS
    from world import dist_to_river, in_forest
    if dist_to_river(x, y) < 12.0:
        return "river"
    if (x - SHELTER_POS[0]) ** 2 + (y - SHELTER_POS[1]) ** 2 < 144.0:
        return "shelter"
    if (x - LANTERN_POS[0]) ** 2 + (y - LANTERN_POS[1]) ** 2 < 36.0:
        return "lantern"
    if in_forest(x, y):
        return "forest"
    if (x - HILL_POS[0]) ** 2 + (y - HILL_POS[1]) ** 2 < 900.0:
        return "hill"
    return "valley"
