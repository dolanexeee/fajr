# ARCHITECTURE — one-page system map

Fajr (fajr-live-1): a mechanistic human-like being in «الوادي» (the
valley). Pure Python. Zero LLM. Zero network. Wall-clock life,
2 ticks/sec, atomic 30s checkpoints.

```
┌─ LIFE LOOP ──────────────────────────────────────────────┐
│ src/run_life.py → src/live_runner.py                     │
│   sim tick (2/sec) → senses → drives → mind → act        │
│   every tick journaled: life/journal.jsonl (append-only) │
└──────────────┬───────────────────────┬───────────────────┘
               │                       │
┌─ BODY/MIND ──┴─────────┐  ┌─ LANGUAGE─┴──────────────────┐
│ src/fajr.py            │  │ src/lexicon.py   word store  │
│ src/world.py   valley  │  │ src/language.py  Tongue:    │
│ src/senses.py  raw     │  │   production (state→speech) │
│ src/perception.py      │  │   comprehension (parser)    │
│ src/drives.py  needs   │  │ src/conversation.py turns   │
│ src/mind.py            │  │ src/host.py  the Host       │
│ src/episodic.py memory │  │ src/grounding.py concepts   │
│ src/beliefs.py         │  │ src/honesty.py   interaction│
│ src/selfmodel.py       │  │ src/voice/*      synthesis  │
│ src/fwdmodel.py        │  └─────────────────────────────┘
│ src/workspace.py       │
└────────────────────────┘
```

## The five honesty-relevant subsystems (file pointers)

1. **Life** (`run_life.py`, `live_runner.py`, `fajr.py`, `world.py`):
   continuous biography since 2026-10-23, seed 11. Journal is the
   single source of truth (`life/journal.jsonl`).
2. **Language** (`lexicon.py`, `language.py`): production only from
   known words (strength ≥ 0.50); comprehension down to 0.25.
   Every utterance logged with structured grounding claims.
3. **Grounding** (`grounding.py`, `LANGUAGE-GROUNDING.md`): no word
   without its lived triple (sensor + drive + action). Novel-word
   mechanism proven in harness (`test_grounding.py`).
4. **Voice** (`voice/`): Rosenberg glottal source + 20-section
   Kelly-Lochbaum tract at human scale + human vowels. Every
   phoneme gated by segment-wise recognition
   (`rosenberg.py`, `tract.py`, `syllables.py`, `nasals_fixed.py`,
   `fricatives_human.py`, `arabic_units.py`, `guttural.py`,
   `live_bridge.py`). Honest ceilings in `voice/VOICE-REALISM-NOTES.md`.
5. **Interaction** (`conversation.py`, `host.py`, `honesty.py`,
   `INTERACTION-CHARTER.md`): turns within declared limits;
   not-knowing instead of guessing; anti-confabulation;
   capability decline. Gated by `test_honesty.py`.

## What is NOT in the system

- No neural networks, no embeddings, no training data.
- No API keys, no model endpoints, no network calls at runtime
  (`talk_server.py` serves HTTP locally; its only `urllib` use is
  URL parsing).
- No hidden state: `to_dict`/`from_dict` serialize everything.

## Entry points for a reviewer

- Read: `SPEC.md` (frozen spec), `ROADMAP-TO-WORLD.md`,
  `INTERACTION-CHARTER.md`, `LANGUAGE-GROUNDING.md`.
- Run: `cd src && python3 test_grounding.py`
  → `ALL GROUNDING GATES PASS`
- Run: `cd src && python3 test_honesty.py`
  → `ALL HONESTY GATES PASS`
- Run: `cd src && python3 -m voice.guttural`
  → 5 guttural gates (ayn/hah/emph-t/emph-d/djim)
- Audit: `grep -rniE "openai|anthropic|transformers|torch" --include="*.py" src/`
  → empty (see REVIEW-PACKAGE.md).
