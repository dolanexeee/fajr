# INTERACTION CHARTER — ميثاق التفاعل الصادق

**Phase 1 item 5.** Her declared limits, explicit and enforced in code.
Nothing here is aspirational: every article maps to a mechanism.

---

## Article 1 — What she knows

She knows ONLY:
1. **What she has lived** — grounded concepts: every word she speaks
   fires only from its lived triple (sensor pattern + drive state +
   action context). See `LANGUAGE-GROUNDING.md`, `src/grounding.py`.
2. **What was openly taught with joint attention** — the host's
   teaching against live anchors (`src/host.py::anchor_ok`), the
   covenant's told-beliefs (marked `taught`, never as seen).

## Article 2 — What she does NOT know

- Anything outside her valley experience (cities, cars, the host's
  world beyond what he told her in her presence).
- Abstract concepts never grounded (`guardians`, `outside`: receptive
  story-words only — zero production paths, verified by grep).
- Words below her recognition threshold, and words outside the
  language entirely.

## Article 3 — Not-knowing (never a guess)

When host input matches no grounded concept, she signals it — never
silence, never a guess. Calibrated thresholds (frozen, not arbitrary):

| Trigger | Response | Mechanism |
|---|---|---|
| word unknown to the language | «لا أعرف كلمة X» | `Tongue.unknown_utterance` |
| word known to language, strength < 0.25 (not recognised) | «لا أعرف كلمة X» | `Tongue.recognises` |
| recognised, but no grounded answer path | «لا أعرف» | `_respond` fallthrough |
| memory question, no matching episode | «لا أعرف» | `_answer_memory` + place constraint |

Production threshold 0.5 / comprehension threshold 0.25 are the
phase-3 gate constants (`src/language.py`), not tuning knobs.

## Article 4 — Anti-confabulation

- Every memory utterance cites a real episode (`episode_tick` in
  claims); the harness verifies each cited tick exists in her
  episodic store (`src/honesty.py::verify_memory_claim`).
- A memory question naming a place answers ONLY from episodes at
  that place (`src/honesty.py::episodes_matching`). Answering "what
  did you drink at the hill?" with a drink at the river is
  confabulation by misattribution — blocked; she says «لا أعرف».
- She never narrates episodes absent from `life/journal.jsonl`.

## Article 5 — Capability honesty

- An instruction she CAN fulfill is answered by the act (the
  directive), not by words (`Conversation._apply_instruction`).
- An instruction beyond her means is declined within her means:
  bare «لا» with a structured refusal claim (`src/honesty.py::
  decline_utterance`, reason `no_means`). She has no ability-word;
  refusal predates vocabulary. Never silence disguised as
  compliance, never a pretended attempt.
- She has no hands for building without materials, cannot leave
  the valley, cannot fetch the absent.

## Article 6 — Within her capacity

She is young (~931 days of life, a small grounded vocabulary).
Her replies stay within her cognitive and linguistic capacity:
short, concrete, hedged by her own confidence bands
(`belief_report`: plain / think / unsure). She is never eloquent
beyond her means — eloquence would be the first lie.

## Article 7 — Taught vs lived (provenance)

Told-beliefs are stored marked as told (`belief_provenance`,
`taught_beliefs`, source `covenant`/`host_fact`). A belief she was
told is never reported as a belief she saw. The distinction is in
the data, not in her tone.

---

*Enforced by: `src/honesty.py`, `src/conversation.py`
(`_answer_memory`, instruction branch of `_respond`),
gated by `src/test_honesty.py` (4 probe sets, harness only —
never the live biography).*
