# REVIEW PACKAGE — Phase 1 exit gate

**Purpose.** An external reviewer (researcher/engineer, chosen by
Mustafa) inspects the full code and attests: no hidden LLM, no
deception, results reproducible. This package contains everything
needed. The reviewer's identity and the sending of the package are
Mustafa's decision — this file prepares, it does not send.

**Project root:** `~/workspace/projects/alsakin-inhabitant/`
(also: the live biography `life/journal.jsonl`, the proof pack of
31 real-data images, the voice audio).

---

## 1. What to review (in order)

1. `ARCHITECTURE.md` — one-page system map (10 min).
2. `SPEC.md` — the frozen spec: what was promised (20 min).
3. `CLAIMS.md` — every claim with evidence pointers + explicit
   non-claims (20 min).
4. `src/` — full code. Start with `honesty.py` (120 lines),
   `grounding.py`, `language.py`, `conversation.py`, then
   `voice/` (`rosenberg.py`, `tract.py`, `syllables.py`).
5. `INTERACTION-CHARTER.md`, `LANGUAGE-GROUNDING.md`,
   `src/voice/VOICE-REALISM-NOTES.md` — the honest records.

## 2. The three attestations requested

- [ ] **No hidden LLM.** No neural network, no model API, no
      prompt, no embedding anywhere in the runtime path.
- [ ] **No deception.** Every claim in `CLAIMS.md` is supported;
      the ceilings in `CLAIMS.md §3` are real; nothing is faked,
      papered over, or exaggerated.
- [ ] **Reproducible.** The commands in §4 produce the stated
      outputs on a fresh checkout (Python 3 + numpy/scipy/
      matplotlib only).

## 3. How to verify "no hidden LLM" (do this first)

```bash
cd ~/workspace/projects/alsakin-inhabitant
# 3a. model/API imports — must print nothing
grep -rniE "openai|anthropic|cohere|huggingface|transformers|torch|tensorflow|onnx|gguf" --include="*.py" src/
# 3b. network egress — inspect every hit by hand
grep -rniE "urllib|requests|http\.|socket" --include="*.py" src/ \
  | grep -vi "requests he\|requests go\|requests to"
#     expected: only talk_server.py — http.server + urllib.parse for
#     the LOCAL talk surface (no outbound calls anywhere)
# 3c. runtime network check (optional, strong)
#     run any gate below with networking disabled — all pass offline
# 3d. secrets/keys — must print nothing
grep -rniE "sk-|api[_-]?key|bearer" --include="*.py" src/ \
  | grep -vi "keyboard" | grep -vi "task-relevant"
```

## 4. How to reproduce the key results (exact commands)

```bash
cd ~/workspace/projects/alsakin-inhabitant/src
python3 --version            # 3.x
python3 -c "import numpy, scipy, matplotlib; print('deps ok')"

# 4a. language grounding — 17 gates
python3 test_grounding.py
# expected last line: ALL GROUNDING GATES PASS
# (also writes src/novel_timeline.json — the 'bibi' experiment data)

# 4b. honest interaction — 15 probes
python3 test_honesty.py
# expected last line: ALL HONESTY GATES PASS
# (also writes src/honesty_probes.json)

# 4c. voice: 5 guttural phoneme gates (ayn/hah/emph-t/emph-d/djim)
python3 -m voice.guttural
# expected: 5 PASS lines with gate signatures

# 4d. voice: vowel/word/sentence gates (the original ص2–ص4 battery)
python3 -m voice.build_voice
# expected: ALL DONE. Gates: s2=True s3=True s4=True

# 4e. voice: speakability of her real utterances
python3 -m voice.live_bridge
# expected: per-utterance table, 100% speakable, missing=[]
# (reads life/journal.jsonl — the live biography; on a fresh
# checkout without it, this step degrades gracefully)
```

All five are deterministic (seeded PRNG in `src/prng.py`).

## 5. What the reviewer is NOT asked to do

- Not asked to judge consciousness, sentience, or moral status —
  the project makes no such claims (`CLAIMS.md §2`).
- Not asked to endorse any worldview or public narrative.
- Not asked to keep secrets: the code, the journal, and the
  proof images are inspectable as presented.

## 6. Delivering a verdict

The reviewer replies in any form with, for each attestation in
§2: **confirmed / not confirmed / cannot determine**, plus the
exact file and line behind any "not confirmed". One inflated
claim found = credibility failed; the charter requires it to be
reported, not softened.

---

*Prepared 2026-10-31. Reviewer: TBD by Mustafa. Package status:
ready to send on his word — do not send without it.*
