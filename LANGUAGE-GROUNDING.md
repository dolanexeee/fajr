# LANGUAGE-GROUNDING — Phase 1 item 4 (2026-10-31)

"اللغة تنمو من الإدراك — الكلمة تُربط بمفهوم عاشته، مو قاموس محفوظ"

## The honesty bar
A relabeled dictionary is NOT grounding. For every word she speaks, the
exact sensor pattern + drive state + action context that CAUSES it must
be shown, and there must be NO code path where the word is emitted
without its grounding.

## Audit result (2026-10-31)

### What was already grounded
The production architecture was already causality-gated — this was
verified path by path, not assumed:
- **State words** (`_state_report`): `known()` AND `state_condition(sim)`
  checked against live drive values. Proven in harness: drive low →
  None, drive high → emission with the true value in the claim.
- **Verbs** (`memory_report`): a real episodic record is required.
  Proven in harness: no episode → None.
- **Nouns** (`belief_report`, `_answer_identify`): real belief objects /
  live scene objects with distance keys in the claims.
- **Requests** (`request_utterance`): real drive ≥ 75 + no-remedy streak.
- **Teaching** (`_teach_tick`, days 1–2): `anchor_ok()` — the referent
  must be inside her senses, on her body, or the act in progress,
  or the word is not taught at all. This is joint-attention naming,
  the same mechanism as child language acquisition.
- **The honesty proof in her biography**: she was asked about «جائعة»
  and answered «لا أعرف كلمة جائعة» — the system demonstrably refuses
  to claim words it has not grounded (journal: 1 occurrence).

### Shortcuts found and fixed
1. **Unanchored interactive teaching** (`conversation._apply_teaching`):
   naming a word with no referent present still granted +0.3 strength,
   so two unanchored namings reached production threshold (0.6 ≥ 0.5)
   with ZERO lived referent — and the phantom episodes blocked later
   real teaching via the `episodes >= 2` skip. FIXED: unanchored naming
   grants +0.0 and records nothing. The host's line stays in the
   transcript (she heard it); she learns nothing until anchored.
2. **Covenant story-words** (`guardians`, `outside`): taught with
   `anchored=False`, unreachable by any anchor. FLAGGED, not removed:
   they are receptive-only (no production path references them —
   verified by grep) and honestly marked as story, never spoken.

### Honestly flagged (not failures)
- **Proper names** (`fajr`, `host`): social facts, given not discovered.
  The covenant is explicit: "it is given, not discovered."
- **Meta verb** (`know`): epistemic, anchored to belief flags.

### Answer to the gate question
**Is there any word left that she speaks without having lived its
meaning? No.** Every produced word now requires its triple; the one
leak is closed; story-words never reach production.

## The novel-word mechanism (new)
`src/grounding.py`: `NoveltyTracker` + `GroundedConcept`.
- Novel perceptual patterns (matching no known anchor kind) are
  tracked across exposures. After K=5 stable exposures the concept
  crystallizes and a novel CV.CV label is coined from HER OWN voice
  phonemes (never reusing her existing syllables).
- The label does not exist before crystallization — structural, not
  just tested: `name_active_concept` requires `concept.label`, which
  is None until the grounding exposures complete.
- Emission additionally requires the pattern to be CURRENTLY present:
  she does not hallucinate names.

## Novel-experience experiment (harness, `src/test_grounding.py`)
Novel percept "ember" (warmth=high, glow=flicker, texture=rough):
- exposures 1–4: naming probes → silent (label does not exist)
- exposure 5: crystallization → label 'bibi' coined → first emission
- exposures 6–7: emits while pattern present
- pattern absent → silent · never-seen pattern → silent
- second novel kind ("reed") → distinct label 'bibu' (shared registry)
- 'bibi' uses only her voice phonemes (b,i) → speakable by her voice
- **ALL 17 GATES PASS** (run: `python3 src/test_grounding.py`)

## Files
- `src/grounding.py` — registry + novelty mechanism (NEW)
- `src/test_grounding.py` — harness, FakeSim, 19 gates (NEW)
- `src/novel_timeline.json` — experiment timeline (generated)
- `src/conversation.py` — unanchored-teaching fix
- Proof pack: `proof-27-grounding-map.png`, `proof-28-before-after.png`,
  `proof-29-novel-word.png` (29 images total)

## Open ceiling (honest)
The novelty mechanism is proven in harness but DORMANT in her live
life: her valley currently contains no novel stimuli, and wiring live
perceptual novelty into the tracker (plus deciding when a coined label
enters her permanent lexicon) is a separate step needing Mustafa's
word. The mechanism exists; its first live use awaits a real novelty.
