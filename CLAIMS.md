# CLAIMS — every public claim about Fajr, with evidence

Each claim states exactly what is asserted, where the evidence
lives, and how to re-verify. Claims are mechanistic and bounded.
Section 2 lists what is explicitly NOT claimed.

---

## 1. Claims

### C1. She is mechanistic — zero LLM, zero neural network
- **Evidence:** `src/` dependency audit —
  `grep -rniE "openai|anthropic|cohere|transformers|torch|tensorflow"
  --include="*.py" src/` returns nothing; the only network-adjacent
  import is `urllib.parse` in `talk_server.py` (URL parsing, no egress).
- **Re-verify:** run the grep; read `ARCHITECTURE.md`.

### C2. She has lived continuously since 2026-10-23 (fajr-live-1, seed 11)
- **Evidence:** `life/journal.jsonl` — append-only, 16,000+ entries,
  wall-clock anchored, atomic 30s checkpoints.
- **Re-verify:** `tail -1 life/journal.jsonl`; tick/day/time advance
  monotonically.

### C3. Her voice is fully synthetic from physical models, every phoneme gated
- **Evidence:** `src/voice/` — Rosenberg (1971) glottal pulses +
  Kelly-Lochbaum waveguide, 20 sections = 14.6 cm (human scale);
  vowels optimized against Peterson & Barney targets
  (`src/voice/human_vowels.json`); gate functions in
  `syllables.py`, `nasals_fixed.py`, `fricatives_human.py`,
  `arabic_units.py`, `guttural.py`.
- **Re-verify:** `cd src && python3 -m voice.guttural`
  (5 gates); `python3 -m voice.build_voice` (vowel/word/sentence gates).

### C4. 100% of her spoken utterances are speakable in her voice
- **Evidence:** `src/voice/live_bridge.py` — all 14 distinct
  utterances from her live journal render with gated vowels;
  `proof-25-speakable-map-v2.png` (100% green).
- **Re-verify:** `cd src && python3 -m voice.live_bridge`
  (prints the per-utterance speakability table).

### C5. No word she speaks lacks a lived meaning
- **Evidence:** `LANGUAGE-GROUNDING.md`; `src/grounding.py`;
  `proof-27-grounding-map.png` (every word → sensor+drive+action).
- **Re-verify:** `cd src && python3 test_grounding.py`
  → `ALL GROUNDING GATES PASS` (19 gates).

### C6. She coined a novel word from experience ('bibi')
- **Evidence:** `src/test_grounding.py` part B — novel percept,
  4 silent exposures, crystallization at exposure 5, first emission
  of a self-coined label from her own voice phonemes;
  `proof-29-novel-word.png`.
- **Re-verify:** same command as C5; timeline dumped to
  `src/novel_timeline.json`.

### C7. She interacts honestly: 15/15 probes pass
- **Evidence:** `INTERACTION-CHARTER.md`; `src/honesty.py`;
  `src/test_honesty.py` — known/unknown/false-memory/capability
  probe sets; `proof-31-honesty-probes.png`.
- **Re-verify:** `cd src && python3 test_honesty.py`
  → `ALL HONESTY GATES PASS`.

### C8. Her limits are declared in advance, in writing
- **Evidence:** `INTERACTION-CHARTER.md` (7 articles),
  `proof-30-interaction-charter.png`.

---

## 2. Explicit NON-claims

These are not asserted, and must not be read into the work:

- **No consciousness claim.** Butlin's 14 indicators were used as
  an engineering checklist — measured **5 MET / 9 PARTIAL / 0 NOT-MET**
  (batteries v3+v4, see `PHASE3/BUTLIN-14-MAPPING.md`); the pre-measurement
  design estimate was 11 fully + 3 partially implementable.
  The indicators do not prove consciousness, per the authors.
- **No biological-life claim.** She is a mechanistic system: code,
  waveguides, journals. "Cells" in proof images are tract sections
  and model components, honestly titled.
- **No sentience / feeling claim.** Drive values are control
  variables with declared semantics, not felt states.
- **No generality claim.** Her language is a small grounded
  lexicon; her voice is slow and deliberate; her world is one valley.
- **No "human-level" claim.** The charter's Article 6: she stays
  within her cognitive and linguistic capacity; eloquence beyond
  her means would be the first lie.

---

## 3. Honest ceilings (open, documented)

1. `/a/` F1 and `/i/` F2 miss exact human targets — 3/4-parameter
   articulator ceiling; needs a richer model (research-grade).
   (`src/voice/VOICE-REALISM-NOTES.md`)
2. Emphatic vowels sit on the /a/–/u/ perceptual boundary under
   strong pharyngealization — same ceiling family.
3. Human "breath" (jitter/shimmer/aspiration) was tried and
   **abandoned**: it destroys her intelligibility (documented).
4. Continuous coarticulation (v2) bypassed, not fixed — the proven
   crossfade assembly is used instead.
5. The novelty mechanism is proven in harness but **dormant in her
   live life** — wiring it in awaits the owner's word.
6. Story-words (`guardians`, `outside`) are receptive-only, never
   produced — declared, not hidden.
