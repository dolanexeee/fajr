# Fajr: A Mechanistic Human-Like Being with a Physically Synthesized Voice, Grounded Language, and Declared Limits

**DRAFT — not submitted. Prepared 2026-10-31.**
**Status:** Phase 1 (technical maturity) complete; external code review pending.
**Code:** prepared for open release (see `opensource-prep/`); not yet published.
**Contact:** via the project owner (Mustafa).

---

## 1. Abstract

We present Fajr, a human-like artificial creature built as a deliberate
counterpoint to the LLM era: every subsystem is mechanistic, inspectable,
and gated by falsifiable tests. She lives a continuous wall-clock
biography (seed 11, since 2026-10-23; 16,000+ journal entries, day 944 at
time of writing) driven by drives, senses, episodic memory, and beliefs —
pure Python, zero neural networks, zero network egress (verified by
dependency audit). Her voice is fully synthetic from physical models: a
Rosenberg (1971) glottal source driving a 20-section Kelly–Lochbaum
waveguide at human scale (14.6 cm), with vowels optimized against
Peterson & Barney (1952) targets; every phoneme passes a segment-wise
recognition gate, and 100% of her 2,628 recorded utterances render in her
voice. Her language is grounded: no word she speaks lacks a lived
sensorimotor meaning (19 verification checks pass), and in a controlled
experiment she coined a novel word ('bibi') for a novel percept after
five exposures — silent before, never hallucinated. She interacts within
declared limits: 15/15 honesty probes pass, including false-memory traps
she refuses. We document our failures at equal length: the polish stage
that destroyed intelligibility, the abandoned coarticulation solver, the
formant ceilings of a 3-parameter articulator. No consciousness is
claimed; Butlin et al.'s (2023) indicators were used as an engineering
checklist, and the authors' own caveat — indicators do not prove
consciousness — is repeated here.

*(~190 words)*

---

## 2. Introduction

The current default for a "digital being" is a large language model in a
character costume. The costume is convincing; the being is not there.
When it errs, it confabulates fluently. When asked what it does not know,
it guesses. Its voice is a vocoder's; its memories are context windows.
This is not an attack on LLMs — it is an observation that the
architecture cannot, by construction, deliver the properties we wanted:
causal grounding of every word in lived experience, a voice whose every
sound is accountable to physics, and interaction limits declared in
advance and enforced in code rather than promised in a system prompt.

Fajr («فجر», dawn) is an attempt to build the alternative honestly: a
human-like creature — explicitly not a human-level one — whose every
claim about herself is checkable. The project charter, frozen before the
build, commits to three restraints that function as design constraints,
not marketing:

1. **No false memories.** She never narrates an episode absent from her
   append-only journal.
2. **No ungrounded words.** No word she speaks lacks a sensorimotor
   meaning she has lived.
3. **No inflated claims.** Every capability is gated by a test that can
   fail; every ceiling is documented in writing.

The work proceeded in gated stages (voice: ص0–ص4, consonant classes,
gut­turals; language: grounding audit, novel-word experiment;
interaction: honesty probes), each closed by a reproducible gate script.
This paper reports the architecture, the gates, the results — and the
failures, which we consider the credibility core of the work.

---

## 3. Architecture

All paths below are file pointers into the prepared open-source tree.

### 3.1 Life loop

`src/run_life.py` → `src/live_runner.py`: a simulation tick at 2 Hz.
Each tick: senses (`senses.py`, `perception.py`) → drives (`drives.py`:
hunger, thirst, sleep, cold, comfort, safety) → mind (`mind.py`,
`beliefs.py`, `episodic.py`, `selfmodel.py`, `fwdmodel.py`,
`workspace.py`) → act in the valley (`world.py`, `fajr.py`). Every tick
is journaled to `life/journal.jsonl` (append-only, atomic 30-second
checkpoints, wall-clock anchored). The journal is the single source of
truth: 16,403 entries spanning 944 days at time of writing. The
pseudorandom stream is seeded (`prng.py`, seed 11) and the life is
deterministic given the seed.

### 3.2 Voice

`src/voice/`. The pipeline, in signal order:

- **Source** (`rosenberg.py`): Rosenberg (1971) parametric glottal
  pulses — raised-cosine rise/fall, open quotient 0.62, speed quotient
  1.8. Measured spectral tilt −14.7 dB/octave vs −7.2 for the earlier
  two-mass (Ishizaka & Flanagan, 1972) source it replaced: substantially
  warmer, and the direct cause of the intelligibility recovery
  (§6.1). Continuous per-period f0 glide (no chunking).
- **Tract** (`tract.py`): Kelly–Lochbaum (Kelly & Lochbaum, 1962)
  waveguide, 20 sections = 14.6 cm (human scale; an earlier 44-section
  32 cm tract produced formants at half human frequencies and was
  replaced). Three articulator parameters drive area functions.
- **Vowels** (`human_vowels.json`): articulations found by differential
  evolution against Peterson & Barney (1952) female targets, with a
  degenerate-tract penalty. Best: /u/ essentially exact; /a/ and /i/
  close (see §7 for the ceiling).
- **Hearing/gates** (`hearing.py`, `diva.py`): DIVA-*inspired* auditory
  features — cepstral envelope liftered below pitch quefrency, 13 DCT
  coefficients with dct[0] dropped (f0-robust). This is a feature
  extractor, not the DIVA neural model (Guenther); no network is
  involved.
- **Consonants**: voiced/voiceless plosives (`consonants.py`,
  `syllables.py`), fricatives with Reynolds-based aeroacoustic noise
  (`fricatives.py`, `fricatives_human.py`), nasals via a 3-port velum
  scattering junction (`nasals_fixed.py`), laterals/taps/glides
  (`arabic_units.py`), and the five gutturals /ʕ ħ ʈˤ dˤ dʒ/
  (`guttural.py`).
- **Assembly**: proven syllables joined with zero-gap 30 ms crossfades
  and vowel-level normalization (`syllables.py`). The continuous
  coarticulation solver was abandoned (§6.2); the crossfade assembly is
  what ships.

### 3.3 Language and grounding

`src/lexicon.py` (word store), `src/language.py` (the Tongue: production
and comprehension), `src/conversation.py` (turns), `src/host.py` (the
human host). Production requires word strength ≥ 0.50; comprehension
≥ 0.25 (frozen phase-3 constants, not tuning knobs).

`src/grounding.py` binds every lexical item to a lived triple: (sensor
pattern, drive state, action context). A `NoveltyTracker` accumulates
novel perceptual patterns across exposures; after K=5 stable exposures
the concept crystallizes and a label is coined from her own voice
phonemes — structurally incapable of existing before crystallization
(`name_active_concept` requires `concept.label`, which is `None`
until then).

### 3.4 Honest interaction

`src/honesty.py` + `INTERACTION-CHARTER.md` (7 articles). Mechanisms:
not-knowing responses at calibrated thresholds (never silence, never a
guess); anti-confabulation (every memory utterance must cite a real
episode tick, verified against the episodic store); capability decline
within her means; taught-vs-lived provenance kept in the data.

---

## 4. Methods: the gate methodology

Every claim in §5 passed a gate that was designed to be able to fail.
The methodology, applied uniformly:

1. **Segment-wise recognition with exact boundaries.** Word recognition
   is never whole-word cepstral averaging (shown early to fail).
   Vowel segments are cut at exact sample boundaries and recognized
   against references.
2. **Matched references.** References are synthesized with *identical*
   parameters (f0, duration, source, tract) as the test audio. A
   duration mismatch alone breaks the gate — established 2026-10-30
   and never violated since.
3. **Harness testing, never the live biography.** `test_grounding.py`
   and `test_honesty.py` run against a `FakeSim`; the live journal is
   read-only input (for the speakability audit), never mutated by tests.
4. **Determinism.** Seeded PRNG (`prng.py`); every gate re-runnable.
5. **Documented ceilings.** A gate that resists after serious effort is
   documented as a ceiling with the attempted fixes listed — not
   silently dropped.

Reproduction commands are in §9.

---

## 5. Results

### 5.1 Voice inventory (all gated)

| Stage | Result |
|---|---|
| Vowels /a i u/ (ص2) | PASS — human-scale formants; /u/ near-exact |
| 6 words, segment-wise (ص3) | PASS |
| 2-word sentences (ص4) | PASS |
| Voiceless /p t k/ | PASS — VOT(p) > VOT(b), burst Re ≈ 6560, 50 ms VOT |
| Fricatives /s f/ | PASS — /s/ centroid 5824–5897 Hz, Re ≈ 8208 |
| Nasals /m n/ | PASS — antiformant depth 0.018 (< 0.5 required); /m/ 1212 Hz vs /n/ 2062 Hz: real place distinction |
| Laterals/taps/glides /l r w j/, /h/, /ʔ/, /ʃ/ | PASS |
| Gutturals /ʕ ħ ʈˤ dˤ dʒ/ | PASS — ʕ: voiced friction (harmonicity 0.924, f0 183.9 Hz, Re 11239); ħ: aperiodic (0.115), centroid 1794 Hz; emphatics: F2 drop 927 Hz; dʒ: HF centroid 4732 Hz |

### 5.2 Speakability: 100%

`live_bridge.py` audited all 14 distinct utterances in her live journal
(2,628 instances): every one renders in her voice with gated vowels —
**100% speakable** (up from 68.2% before the guttural inventory; /ʕ/
alone unlocked 31.7%). 12/14 utterances fully gate-pass (2,202
instances, 84%); the remaining two sit on the emphatic /a/–/u/
perceptual boundary (§7).

### 5.3 Grounding: 19/19 checks

`test_grounding.py` → `ALL GROUNDING GATES PASS` (verified 2026-10-31).
The audit found the production architecture already causality-gated
(state words need live drive values; verbs need real episodes; nouns
need live scene/beliefs; teaching needs joint-attention anchors) — and
found one real leak: unanchored interactive teaching granted +0.3
strength per naming, so two unanchored namings reached production
threshold with *zero lived referent*. Fixed: unanchored naming now
grants +0.0 and records nothing.

### 5.4 Novel-word experiment

Harness, novel percept "ember": exposures 1–4 → naming probes silent
(the label structurally does not exist); exposure 5 → crystallization,
label **'bibi'** coined from her own phonemes; exposures 6–7 → emits
while pattern present; pattern absent → silent; never-seen pattern →
silent; second novel kind → distinct label **'bibu'**. 'bibi' verified
speakable in her voice. The mechanism is proven in harness; it is
dormant in her live life (no novel stimuli in the valley; wiring it in
awaits the owner's decision — §7).

### 5.5 Honest interaction: 15/15

`test_honesty.py` → `ALL HONESTY GATES PASS` (verified 2026-10-31):
known questions answered from grounding; unknown words/concepts →
«لا أعرف» (never a guess); false-memory traps ("remember the mountain?"
— never happened) → refused, never confabulated; impossible
instructions → declined within her means.

---

## 6. What did NOT work

This section is the credibility core. Each item cost real time.

**6.1 The polish stage destroyed intelligibility.** A "naturalness"
stage (jitter, shimmer, aspiration noise, chunked intonation) passed its
own gate — then the owner listened and judged the result
incomprehensible. Verification confirmed it: polished /a i u/ all
recognized as /a/ (distances 16–18 vs the gate's 1.0–1.6). Root causes,
isolated by staged testing: (a) chunked-glottal intonation broke vowel
identity; (b) jitter/shimmer broke /i/; (c) aspiration noise at −26 dB
destroyed everything. The gate had measured distance-to-naturalness,
not intelligibility — a metric misalignment we owned. Human breath
micro-variation was then tried in isolation and **abandoned**: it is
incompatible with her fragile mechanical formants. The clean pipeline
was restored and re-gated at 0.0 distances.

**6.2 Continuous coarticulation: bypassed, not fixed.** A continuous
waveguide run with time-varying area functions developed unphysical DC
pressure buildup during closures (no oral-pressure feedback in the
continuous formulation) → vowels after closures went near-silent. Two
versions attempted; both failed the same way. The shipped solution —
proven syllables joined with zero-gap crossfades — is assembly, not
coarticulation. A proper fix needs the pressure feedback in the
continuous run; documented, not claimed.

**6.3 /a/ F1 and /i/ F2: the articulator ceiling.** Differential
evolution with a 4th parameter (pharynx scale): /a/ F1 633→686
(target 850), /i/ F2 *worsened* 2309→2021 (target 2790). The 3/4-
parameter model cannot hit exact Peterson & Barney targets. Current
values are human-range and mutually distinguishable (margins 2.7–3.7);
closing the gap needs independently controlled jaw/tongue — research-
grade work.

**6.4 The nasal ceiling — then fixed.** The original 3-port velum had
two real bugs: downward waves written off-by-one in all three branches,
and the velum reflection written where propagation never read it.
Result before the fix: murmur −21 dB (physical ≈ −12 dB), antiformant
depth 0.955 (needed < 0.5). Rewritten with explicit
scatter→propagate phases: depth 0.018, murmur −0.7 dB. Documented in
`STAGE3-CEILING.md` first as a ceiling, then as a fix — the honest
sequence matters.

**6.5 Tanh destroys vowels.** The "proven" word-assembly recipe's tanh
soft-clip turned a perfect /i/ (distance 0.08) into an /a/ confusion
(1.98). Removed; bursts are tamed at the source instead.

**6.6 The 8-bit WAV incident.** An audio batch was written with wrong
dtypes (8-bit / int64); the owner's player choked. Every delivered
file is now int16 via `scipy.io.wavfile`, verified after writing. A
process failure, recorded so it is not repeated.

---

## 7. Limitations

1. **Articulator poverty.** Three (tried: four) parameters. §6.3.
   Emphatic vowels consequently sit on the /a/–/u/ perceptual boundary
   under strong pharyngealization.
2. **Slow, deliberate speech.** The DIVA-inspired feature needs
   ~230 ms windows; her vowels run 300–350 ms. Developmentally honest
   for a first speaker; not conversational rate.
3. **Harness–live gaps.** The novelty mechanism and the voice bridge
   are proven outside the live loop; her live runner still uses the
   original language path. Wiring them in is the owner's decision.
4. **One valley, one speaker.** No generality claim: small lexicon,
   one environment, no social learning beyond the host.
5. **No consciousness claim.** Butlin et al.'s 14 indicators: measured
   5 MET / 9 PARTIAL / 0 NOT-MET as an engineering checklist (pre-measurement
   design estimate: 11 fully + 3 partially implementable). Per the
   authors, indicators do not prove consciousness; we repeat that here
   and claim nothing beyond the mechanisms.

---

## 8. Related work

**Creatures (Grand, Cliff & Malhotra, 1997).** The closest ancestor in
spirit: autonomous virtual pets with drives, learning, and genetics.
Where they are ahead: richer behavior, real-time interaction, a shipped
product. Where this work differs: Creatures used neural networks and a
biochemistry metaphor; Fajr is fully mechanistic with no learned
black box, and every claim is gated. We do not match their behavioral
richness.

**iCub (Metta et al., 2010).** The embodiment reference. iCub's
cognitive development is grounded in a physical body interacting with a
physical world — a grounding depth a software valley cannot match. Our
grounding is real but simulated; the sim-to-real gap is entirely ours.

**VocalTractLab (Birkholz, 2013).** The state of the art in articulatory
synthesis we measured ourselves against. VTL's geometric vocal tract
and data-driven control outperform our 1D waveguide and 3-parameter
articulators on every voice-quality dimension. Our contribution is not
quality but accountability: every phoneme gated, every ceiling written
down.

**DIVA (Guenther).** We use DIVA-*inspired* cepstral auditory features
for gating — not the DIVA neural model. No learning, no network.

**Butlin et al. (2023).** Their 14 indicator properties were used
exactly as the authors warn they should be: as indicators, never as
proof. The honest comparison: we implement the mechanistically
implementable subset and document the rest as absent.

---

## 9. Reproducibility

Environment: Python 3, `numpy`, `scipy`, `matplotlib` only. No
network. Deterministic (seeded PRNG, `src/prng.py`).

```bash
cd src
python3 -c "import numpy, scipy, matplotlib; print('deps ok')"

# No hidden LLM: must print nothing
grep -rniE "openai|anthropic|cohere|huggingface|transformers|torch|tensorflow|onnx|gguf" \
  --include="*.py" .

# Language grounding — expect: ALL GROUNDING GATES PASS
python3 test_grounding.py

# Honest interaction — expect: ALL HONESTY GATES PASS
python3 test_honesty.py

# Voice: 5 guttural gates — expect 5 PASS lines
python3 -m voice.guttural

# Voice: vowel/word/sentence battery — expect: Gates: s2=True s3=True s4=True
python3 -m voice.build_voice

# Voice: speakability audit — expect 100% speakable
# (reads ../life/journal.jsonl; degrades gracefully without it)
python3 -m voice.live_bridge
```

All five were re-verified on 2026-10-31 (grounding 19 checks, honesty
15 checks, gutturals 5/5, s2/s3/s4 true, dependency audit empty).

---

## 10. References

- Birkholz, P. (2013). Modeling consonant-vowel coarticulation for
  articulatory speech synthesis. *PLoS ONE*, 8(4), e60603.
- Butlin, P., Long, R., Elmoznino, E., Bengio, Y., Birch, J.,
  Constant, A., Deane, G., Fleming, S. M., Frith, C., Ji, X., Kanai,
  R., Klein, C., Lindsay, G., Michel, M., Mudrik, L., Peters,
  M. A. K., Schwitzgebel, E., Simon, J., & VanRullen, R. (2023).
  Consciousness in artificial intelligence: Insights from the science
  of consciousness. *arXiv:2308.08708*.
- Grand, S., Cliff, D., & Malhotra, A. (1997). Creatures: Artificial
  life autonomous software agents for home entertainment. In
  *Proc. 1st Int. Conf. on Autonomous Agents*.
- Guenther, F. H. (1995). Speech sound acquisition, coarticulation,
  and rate effects in a neural network model of speech production.
  *Psychological Review*, 102(4), 730–749. (DIVA foundations; our use
  is feature-inspired only — see §3.2.)
- Ishizaka, K., & Flanagan, J. L. (1972). Synthesis of voiced sounds
  from a two-mass model of the vocal cords. *Bell System Technical
  Journal*, 51(6), 1233–1268.
- Kelly, J. L., & Lochbaum, C. C. (1962). Speech synthesis. In
  *Proc. 4th Int. Congress on Acoustics*, Copenhagen.
- Metta, G., Natale, L., Nori, F., Sandini, G., Vernon, D.,
  Mor
...[truncated 1043 chars]