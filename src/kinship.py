# -*- coding: utf-8 -*-
"""Kinship teaching — ADAM & EVE P1.

HONESTY NOTE (read before touching):
- KINSHIP_WORDS and KINSHIP_LINES are authored scaffolding, exactly like
  THEME_FRAGMENTS in dreams.py. The Guide names the concepts; what is HERS
  is the learning (strength via the normal Tongue.learn path), the later
  USE of the words in her own utterances/dreams, and the readiness flag.
- Abstract words anchor only in story (covenant-like moment) — see
  host.anchor_ok. This teaching is delivered as one such deliberate
  moment, once, ever. It is NEVER repeated.
- Journal language stays behavioral: «علمها المضيف» records that teaching
  happened, never that she "understands reproduction" as a human would.
"""

from lexicon import word_surface

# word ids taught (must exist in lexicon.WORDS)
KINSHIP_WORDS = ["pair", "family", "newlife"]

# the Guide's lines — his voice, conceptual level only
KINSHIP_LINES = {
    "pair": "انظري: هذا زوج — اثنان معًا، يكمل أحدهما الآخر",
    "family": "وهذه عائلة — زوج يحمي ويُحمى، ومنهما تأتي حياة جديدة",
    "newlife": "الحياة الجديدة تأتي من الزوج — طفل يكبر في الوادي",
}

TEACH_GAIN_KIN = 0.6  # same as conversation.TEACH_GAIN (kept local: no import cycle)


def teach_kinship(conv):
    """One-time teaching of the kinship words by the Guide.

    Follows the exact normal teaching path (transcript line + journal
    'teaching' entry + tongue.learn + semantic memory), so the words
    enter her vocabulary through the same honest machinery as every
    other word she knows. Sets the mind readiness flag — the behavioral
    marker P2's spawn checks.
    Returns the list of word ids taught.
    """
    sim = conv.sim
    tick = sim.tick
    f = sim.fajr
    taught = []
    for wid in KINSHIP_WORDS:
        line = KINSHIP_LINES[wid]
        conv._log_line("host", line)
        sim.journal.log(tick, "teaching", f"علمها المضيف: {line}",
                        x=f.x, y=f.y)
        ep = conv.tongue.learn(wid, tick, gain=TEACH_GAIN_KIN,
                               anchored=True, source="teaching")
        f.mind.episodic.set_semantic(f"word:{wid}", ep["strength_after"], tick)
        taught.append(wid)
    # behavioral marker: she is ready (P2 checks this before Adam spawns)
    f.mind._kinship_ready = True
    sim.journal.log(tick, "readiness", "تعلمت كلمات الزوج والعائلة — تستعد",
                    x=f.x, y=f.y)
    return taught
