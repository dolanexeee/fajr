# REPRODUCE — the 5 key results, step by step

Environment: Python 3 + `numpy` + `scipy` + `matplotlib`. No network.
All commands run from `src/`. Every gate is deterministic (seeded PRNG).

---

## 0. Sanity: dependencies and the no-LLM audit

```bash
python3 -c "import numpy, scipy, matplotlib; print('deps ok')"
# expected: deps ok

grep -rniE "openai|anthropic|cohere|huggingface|transformers|torch|tensorflow|onnx|gguf" \
  --include="*.py" .
# expected: (no output — empty)
```

If the grep prints anything, stop: the mechanistic claim is broken.

---

## 1. Language grounding — 19 checks

```bash
python3 test_grounding.py
```

**Expected last lines:**
```
ALL GROUNDING GATES PASS
timeline -> src/novel_timeline.json
```

**What it proves:** every word she speaks requires its lived triple
(sensor pattern + drive state + action context); the unanchored-teaching
leak is closed; and in the novel-word experiment a new percept
("ember") yields 4 silent exposures, crystallization at exposure 5 with
a self-coined label ('bibi'), emission only while the pattern is
present, and a distinct label ('bibu') for a second novel kind.
`src/novel_timeline.json` holds the experiment data.

---

## 2. Honest interaction — 15 probes

```bash
python3 test_honesty.py
```

**Expected last lines:**
```
ALL HONESTY GATES PASS
```

**What it proves:** known questions answered from grounding; unknown
words/concepts → «لا أعرف» (never a guess); false-memory traps refused
(never confabulated); impossible instructions declined within her
means. Probe details in `src/honesty_probes.json`.

---

## 3. Voice: 5 guttural phoneme gates

```bash
python3 -m voice.guttural
```

**Expected:** 5 PASS lines with gate signatures, e.g.
```
ayn PASS {'Re': 11239.0, 'turbulent': True, 'harmonicity': 0.924, 'f0_hz': 183.9, ...}
hah PASS {'Re': 11239.0, ..., 'harmonicity': 0.115, 'centroid_hz': 1794.1, ...}
emph-t PASS {'F2_drop_hz': 926.7, ...}
emph-d PASS {'F2_drop_hz': 926.7, ...}
djim PASS {'Re': 11239.0, ..., 'centroid_hf_hz': 4732.0, ...}
```

**What it proves:** /ʕ/ is voiced friction (not ħ); /ħ/ is aperiodic
pharyngeal noise; emphatics lower F2 by ~927 Hz; /dʒ/ has a voiced
ʃ-band release. Each with its vowel recognized segment-wise.

---

## 4. Voice: vowel / word / sentence battery

```bash
python3 -m voice.build_voice
```

**Expected last line:**
```
ALL DONE. Gates: s2=True s3=True s4=True
```

**What it proves:** vowels /a i u/ mutually distinguishable at
human-scale formants (s2); 6 words recognized segment-wise (s3);
two-word sentences (s4). (Also writes sample audio to
`src/voice_state/`.)

---

## 5. Voice: speakability audit — 100%

```bash
python3 -m voice.live_bridge
```

**Expected:** per-utterance table ending with 100% speakable,
`missing=[]`.

**What it proves:** all 14 distinct utterances from her live journal
render in her voice with gated vowels. **Note:** this step reads
`../life/journal.jsonl` — her private biography, NOT included in this
tree. Without it, the step degrades gracefully (it reports the
missing input instead of failing). The phoneme inventory it audits
is fully present and gated by steps 3–4 above.

---

## If anything fails

1. Check `python3 --version` (3.x) and the three packages.
2. Re-run with the exact command — no extra flags.
3. Open an issue with: the command, the full output, and your
   `python3 -c "import numpy, scipy, matplotlib; print(numpy.__version__, scipy.__version__, matplotlib.__version__)"`.

One inflated claim found = credibility failed. Report it, don't
soften it.
