"""Honest interaction — Phase 1 item 5 (INTERACTION-CHARTER.md).

Mechanistic honesty for her turns. Three instruments, no LLM:

  decline_utterance  — her refusal within her means. She has no
      ability-word; bare "no" is her refusal as well as her
      negation. The structured claim (for the log, not spoken)
      records what was asked and why she cannot.
  question_places    — place words named in a question.
  episodes_matching  — real episodes filtered by kind and, when the
      question names places, by place. A memory question naming a
      place NEVER answers from an episode elsewhere: that would be
      confabulation by misattribution.
  verify_memory_claim — a memory claim is honest only if its
      episode_tick exists in her episodic store.

Pure Python. Deterministic. All state hers.
"""
from lexicon import PLACE_WORDS, place_word_at


def decline_utterance(tongue, sim, verb_wid, reason):
    """She cannot do it: decline, never silence disguised as
    compliance, never a pretended attempt. Works without any
    learned word — refusal predates vocabulary."""
    tick = sim.tick
    day = tick // 1440 + 1
    claims = [{"type": "refusal", "verb": verb_wid, "reason": reason}]
    return tongue._emit(tick, day, "reply", "لا", claims, [])


def question_places(parsed):
    """Place words named in a question (content words only)."""
    return [wid for wid, _form, _s in parsed["content"]
            if wid in PLACE_WORDS]


def episodes_matching(sim, kinds, places):
    """Her real episodes, filtered by kind and — when places are
    named — by the place each episode happened at."""
    eps = sim.fajr.mind.episodic.episodes
    pool = [e for e in eps if e["kind"] in kinds]
    if places:
        pool = [e for e in pool
                if place_word_at(e["x"], e["y"]) in places]
    return pool


def verify_memory_claim(claim, sim):
    """A memory claim is honest only if its episode_tick is a real
    entry in her episodic store. Anything else is confabulation."""
    if claim.get("type") != "memory":
        return True
    tick = claim.get("episode_tick")
    return any(e["tick"] == tick
               for e in sim.fajr.mind.episodic.episodes)
