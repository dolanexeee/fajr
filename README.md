# Fajr — a mechanistic human-like being

**An honest alternative in the LLM era: no neural networks, no black
boxes, every claim gated.**

Fajr («فجر», dawn) is a human-like artificial creature — explicitly not
a human-level one — built from inspectable mechanisms: a continuous
life loop with drives and episodic memory, a physically synthesized
voice, language grounded in lived sensorimotor experience, and
interaction limits declared in advance and enforced in code.

This repository is the Phase 1 (technical maturity) release candidate.
It has not been published yet; the paper draft (`PAPER-DRAFT.md`) has
not been submitted; no external review has been received. Everything
below is verifiable on your machine.

---

## The honesty charter (read first)

1. **No false memories.** She never narrates an episode absent from her
   journal.
2. **No ungrounded words.** No word she speaks lacks a sensorimotor
   meaning she has lived.
3. **No inflated claims.** Every capability is gated by a falsifiable
   test; every ceiling is documented in writing (`CLAIMS.md` §3,
   `src/voice/VOICE-REALISM-NOTES.md`).

What is explicitly **not** claimed: consciousness, biological life,
sentience, generality, human-level ability (`CLAIMS.md` §2).

## What is here

| Path | What |
|---|---|
| `src/run_life.py`, `src/live_runner.py` | life loop (2 Hz tick: senses → drives → mind → act) |
| `src/world.py`, `src/fajr.py`, `src/senses.py`, `src/perception.py` | valley, body, senses |
| `src/drives.py`, `src/mind.py`, `src/episodic.py`, `src/beliefs.py`, `src/selfmodel.py` | needs, memory, beliefs |
| `src/language.py`, `src/lexicon.py` | the Tongue: speech production & comprehension |
| `src/conversation.py`, `src/host.py` | turns with the human host |
| `src/grounding.py` | every word bound to (sensor, drive, action); novel-word mechanism |
| `src/honesty.py` | not-knowing, anti-confabulation, capability decline |
| `src/voice/` | Rosenberg glottal source + Kelly–Lochbaum waveguide + gated phonemes |
| `src/test_grounding.py` | 19 grounding checks |
| `src/test_honesty.py` | 15 honesty probes |
| `PAPER-DRAFT.md` | technical paper (draft, unsubmitted) |
| `CLAIMS.md` | every claim with evidence pointers + explicit non-claims |
| `REVIEW-PACKAGE.md` | what an external reviewer is asked to attest |
| `REPRODUCE.md` | step-by-step reproduction of the 5 key results |

## What is NOT here (deliberately)

- **`life/journal.jsonl`** — her private biography (16,000+ entries).
  The speakability audit degrades gracefully without it; everything
  else runs fully offline.
- **Operational files** — tunnel keys, PINs, systemd units, checkpoints.
  Never public.
- **Proof images / audio** — the owner's to share.

## Quickstart

```bash
python3 --version            # 3.x
python3 -c "import numpy, scipy, matplotlib; print('deps ok')"

cd src
python3 test_grounding.py    # expect: ALL GROUNDING GATES PASS
python3 test_honesty.py      # expect: ALL HONESTY GATES PASS
python3 -m voice.guttural   # expect: 5 PASS lines
python3 -m voice.build_voice # expect: Gates: s2=True s3=True s4=True
```

Dependencies: Python 3 + `numpy` + `scipy` + `matplotlib`. Nothing else.
No network access is used or needed — run it air-gapped if you like.

## Verify "no hidden LLM" yourself

```bash
grep -rniE "openai|anthropic|cohere|huggingface|transformers|torch|tensorflow|onnx|gguf" \
  --include="*.py" src/
# must print nothing
```

## License

MIT — see `LICENSE`. Use it, fork it, build stranger things with it.
If you publish results, cite the paper draft and keep the honesty
charter: one inflated claim kills the credibility this was built on.

## Status

Phase 1 complete 2026-10-31. External code review pending (reviewer
TBD). Paper unsubmitted. This tree is a release *candidate*, prepared
for review — not a finished publication.
