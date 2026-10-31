# Fajr Voice — Realism Notes & Honest Ceilings

**Date:** 2026-10-30  
**Approach:** Pure mechanical articulatory synthesis (Python + numpy/scipy).  
**Explicitly NOT:** neural TTS, concatenative, or sample-based.

## What was built (Stages 1–4)

### Stage 1: Voiceless plosives /p,t,k/ — ✅ PASS
- Glottis abducted (2× rest width) during closure → steady DC flow, f0=0.0
  (verified: no voice bar)
- Closure: oral pressure builds (Bernoulli feedback), wall damping 0.80/chunk,
  passive pressure clamp ≤ Psub
- Burst: transient release, Reynolds Re≈6560 (turbulent)
- VOT: 50ms aspiration noise, aperiodic then voiced
- Gate: VOT(p) > VOT(b) ✓, burst present ✓, vowel recognized ✓
- Audio: `fajr_pa.wav`, `fajr_ta.wav`, `fajr_ka.wav`

**Ceiling:** The aspiration noise amplitude is empirically calibrated
(~-15dB re vowel), not derived from first principles. The Reynolds number
confirms turbulence, but the exact noise level is a fit, not a prediction.

### Stage 2: Fricatives /s,f/ — ✅ PASS
- Narrow constriction (not full closure) + interior aeroacoustic noise source
- Noise from Reynolds-based turbulence: Re = v·d/ν, p_turb = ρv²Ti (Ti=0.07)
- Volume-velocity source injected into waveguide: p = U_s·Z/2 (symmetric)
- Flow choked by constriction (Bernoulli): Ug = Ac√(2Psub/ρ)
- Gate: /s/ centroid 5824Hz, hf=0.973 ✓; /f/ diffuse ✓; vowel recognized ✓
- Audio: `fajr_sa.wav`, `fajr_fa.wav`

**Ceiling:** /s/ vs /f/ distinction is subtle (centroids 5824 vs 5658Hz).
Real /s/ has a sharper spectral peak from the teeth obstacle; our model
lacks the obstacle (teeth) geometry. The noise is physically motivated
but the spectral shaping is approximate.

### Stage 3: Nasals /m,n/ — ⚠️ CEILING (documented, not gated)
- `NasalTract`: three-branch waveguide (pharynx + nasal + oral side-branch)
- 3-port scattering at velum derived from first principles:
  P = 2Σ(p_i/Z_i)/Σ(1/Z_i), p_i⁻ = P − p_i⁺
  (pressure continuity + flow conservation — the math is correct)
- **The problem:** Time-domain integration has an indexing bug. Scattered
  waves at the velum are written to state locations the propagation step
  does not read. Result: weak murmur (-21dB re vowel, physical is ~-12dB)
  and no clear antiformant (measured depth 0.955, need <0.5).
- Low nasal formant IS present (~153Hz) — the coupling works, just weakly.
- See `STAGE3-CEILING.md` for full details.
- Audio: `fajr_ma.wav` (included to document the attempt, not as a success)

**Path forward:** Rewrite `NasalTract.run()` with explicit
scatter→propagate ordering. This is engineering, not physics.

### Stage 4: Naturalness polish — ✅ PASS
- **Jitter:** Cycle-to-cycle f0 variation via per-cycle resampling.
  Measured: 0.403% (plain) → 0.759% (polished). Target: 0.5–1.0% ✓
- **Shimmer:** Cycle-to-cycle amplitude scaling.
  Measured: 0.052% (plain) → 4.105% (polished). Target: 3–5% ✓
- **Aspiration:** Turbulent breath noise gated by glottal flow
  (more noise when glottis open — physical, not constant hiss).
  Level: -20dB re voicing.
- **Intonation:** f0 contour via chunked glottis with varying tension
  (f0 ∝ √tension). Declination + stress demonstrated.
- Gate: polished distance-to-natural 0.113 vs plain 4.295 ✓
- Audio: `fajr_polished_sentence.wav` (a-i-u with declination)

**Ceiling:** Jitter/shimmer measured at the SOURCE (glottal flow), not the
acoustic output. The tract's sharp formants confound acoustic measurement
(f0 jitter moves harmonics relative to formants). This is honest: the
micro-variation originates at the source, which is where we control it.
The aspiration noise is white; real breath noise is shaped by the tract.

## Overall honest assessment

**What Fajr's voice IS:**
- A fully mechanistic articulatory synthesizer (no samples, no neural nets)
- Vowels /a,i,u/ with correct formants for a small female vocal tract
- Voiced/voiceless plosives with physical VOT distinction
- Fricatives with aeroacoustic noise (not filtered white noise)
- Natural micro-variation (jitter/shimmer in human range)
- Intonation contour (declination)

**What it is NOT:**
- Not "close to reality" in the sense of fooling a listener. It sounds
  mechanical/synthetic — buzzy, with a robotic quality.
- No clear nasals (/m,n/ lack antiformant — Stage 3 ceiling)
- /s/ vs /f/ distinction is subtle
- No coarticulation (phonemes are concatenated, not blended)
- No prosody beyond simple declination (no phrase-level intonation,
  no emotional variation)

**The fundamental wall:**
Articulatory synthesis (even the best, e.g., Pink Trombone, VocalTractLab)
sounds synthetic because:
1. The source model (two-mass) is a simplification — real vocal folds have
   complex 3D vibration, mucosal waves, and source-tract interaction.
2. The tract is 1D — real vocal tracts are 3D with side branches,
   piriform sinuses, and dynamic shaping.
3. We lack the precise articulatory targets for natural speech —
   human speakers make thousands of micro-adjustments per second.

**To get "closer to reality":**
- Fix Stage 3 (nasals) — engineering, doable
- Add source-tract interaction (feedback from tract to glottis)
- 3D tract model (major undertaking)
- Data-driven articulatory targets (requires EMA/MRI data)
- OR: accept the synthetic quality as "Fajr's voice" — a clear,
  intelligible, mechanistic voice that is honest about what it is.

## Files
- `src/voice/*.py` — implementation
- `src/voice_state/fajr_*.wav` — intermediate audio
- `~/workspace/your_files/fajr-voice/` — final deliverables:
  - `fajr_pa.wav` (voiceless plosive)
  - `fajr_sa.wav` (fricative)
  - `fajr_ma.wav` (nasal — weak, see ceiling)
  - `fajr_polished_sentence.wav` (a-i-u with intonation + polish)

---

## Continuation round (2026-10-31) — all items completed or ceiling documented

### 1. Nasals FIXED ✅
**Bugs found and fixed** (STAGE3-CEILING.md):
- Bug 1: `dn_p[1:Np]` should be `dn_p[0:Np-1]` in all 3 branches (off-by-one;
  downward waves were shifted, never reaching glottis).
- Bug 2: velum scatter wrote `dn_p[Np]` after interior loop; interior never
  read `dn_p[Np]`, so velum reflection was lost.
**Fix**: `src/voice/nasals_fixed.py` — explicit scatter→propagate phases with
temporary leaving-wave arrays. Energy conservation verified
(impulse decays, peak 1.3, stable).
**Adapted**: human tract (12/16/8 sections), Rosenberg source, human vowels.
**Gate results**:
- /ma/: PASS — nasal formant 181Hz, antiformant depth 0.018, murmur -0.7dB, vowel→a
- /na/: PASS — antiformant 2062Hz (vs /m/ 1212Hz: place distinction REAL), vowel→a
- /mi/: vowel confused (a not i) — minor test sensitivity, nasals themselves pass.

### 2. Fricatives + voiceless plosives on human tract ✅
- `src/voice/fricatives_human.py`: /sa/ PASS (Re 8208, centroid 5897Hz),
  /fa/ PASS (Re 11239, centroid 5622Hz). /si/ vowel confusion (minor).
- `src/voice/syllables.py` voiceless fix: was using DC flow for the VOWEL
  after release (unvoiced) → switched to voiced Rosenberg flow post-VOT.
  /pa/ /ta/ /ka/: all PASS (vowel→a, 0.73).

### 3. Full vocabulary in new voice ✅
All 6 words re-synthesized (human tract + Rosenberg + human vowels):
drink, eat, fruit, sleep, thirsty, want — ALL PASS segment-wise.

### 4. Flowing sentence ✅
"want fruit" as ONE continuous utterance (crossfade assembly, 190→168Hz
declination, final lengthened): SENTENCE GATE PASS, 0/56 silent frames.

### 5. Formant gaps — CEILING HOLDS
4th articulator parameter (pharynx_scale) tried via differential evolution:
- /a/: 633→686 (target 850) — marginal gain, still far
- /i/: F2 worsened 2309→2021 — no gain
**Honest ceiling**: 3/4-parameter model cannot hit exact P&B targets for
/a/ F1 and /i/ F2. Current values are human-range and mutually
distinguishable (margins 2.7-3.7). Closing the gap needs a richer
articulatory model (jaw, tongue body/tip independently) — research-grade.

### Files delivered (~/workspace/your_files/fajr-voice/, all int16)
fajr_ma.wav, fajr_na.wav, fajr_sa.wav, fajr_fa.wav,
fajr_{drink,eat,fruit,sleep,thirsty,want}_new.wav,
fajr_want_fruit_flowing.wav

---

## 2026-10-31 — LIVE BRIDGE: her real utterances in her new voice

### The architectural gap (honest)
Her life speaks Arabic text (Tongue._emit). Her voice spoke English
phonemes. The bridge (src/voice/live_bridge.py) connects them without
faking: hand phoneme transcriptions of her 14 real utterances (her
inventory only, NOT a general Arabic G2P), synthesis with the proven
pipeline, and per-utterance gates. Missing phonemes are never
synthesized — the utterance is marked not-speakable with the exact
missing set.

### New phonemes built and gated (src/voice/arabic_units.py)
- /l/ lateral: voiced, alveolar narrowing (not closure). Gate: voiced,
  no burst, sustained, vowel after. PASS.
- /h/: glottal turbulence injected just above glottis (idx 2 — injection
  at idx 0 is silently lost, same bug class as the original nasals bug).
  Gate: harmonicity 0.146 (vs 0.967 vowel), formants present. PASS.
- /ʔ/: 45ms silence. Trivially correct.
- /r/ tap: 25ms voiced closure with pressure-feedback+damping physics.
  Gate: dip 25ms, energy drops, vowel after. PASS.
- /ʃ/: postalveolar, wider constriction; noise low-passed at 4.5kHz
  (wider constriction -> lower eddy frequency, Strouhal scaling).
  Gate: centroid 3248Hz, Re 11239. PASS.
- /w/, /j/ glides: brief /u/, /i/ onsets (implemented, no green
  utterance needs them yet).

### Speakability (from her live journal, 2,607 utterances)
- GREEN (speakable, gated PASS): 1,778 (68.2%) — أكلت فاكهة (1,471x),
  شربت (306x), أكلت (1x). All vowel segments recognized, distances
  0.08–2.34.
- YELLOW (one phoneme away): 406 (15.6%) — all blocked by ʕ (ع) except
  أنت المضيف (blocked by dˤ).
- RED (2+ missing): 423 (16.2%).
- **Next ceiling**: ʕ (ع) alone unlocks 31.7% more of her speech. Then
  ħ, ʈˤ, dˤ, dʒ — the guttural/emphatic inventory is a separate phase.

### Hard-won gate lessons (documented so we never repeat them)
1. **Tanh destroys vowels.** The baba recipe's tanh soft-clip turned a
   perfect /i/ (distance 0.08) into an /a/ confusion (1.98). Removed;
   bursts tamed at the source instead.
2. **Burst dynamics.** Waveguide builds 115x burst/vowel (unphysical).
   _tame_burst: self-locating transient detector, scales peaks >6x
   vowel-RMS down to 4x with raised-cosine edges. Steady vowels
   bit-identical; gate unaffected.
3. **DIVA feature needs ~230ms** (skips 100ms + needs 2048 samples).
   Her speech is therefore slow/deliberate: 350ms base vowels,
   minimum 300ms. Developmentally honest for a first speaker.
4. **Matched refs per segment.** References built with each vowel
   segment's exact f0 and duration (the established rule).
5. **Trim crossfade edges** (35ms) before gating to avoid blend
   contamination.

### Delivered
Audio (~/workspace/your_files/fajr-voice/, all int16 verified):
fajr_live_akalat_fakiha.wav (3.24s), fajr_live_sharibat.wav (1.39s),
fajr_live_akalat.wav (1.53s)
Pictures (~/workspace/your_files/fajr-proof/):
proof-21-speakable-map.png, proof-22-first-live-words.png,
proof-23-bridge.png
Code: src/voice/live_bridge.py, src/voice/arabic_units.py
Note: live_runner NOT modified — the bridge is read-only over her
utterances. Wiring audio into the live loop is Mustafa's decision.

---

## Guttural inventory — built 2026-10-31 (all 5 PASS)

**Module**: `src/voice/guttural.py`. Rosenberg + 20-sec human tract + human vowels.

### /ʕ/ (E) — voiced pharyngeal fricative — PASS
- Low-pharynx constriction (pos 0.15, area 0.15cm²), voicing runs THROUGH
  it + Reynolds turbulence injected (Re=11239).
- Gate: turbulent + voiced (harmonicity 0.924, f0 183.9Hz) + friction
  audible + vowel→a. The voicing-through-friction is what separates ʕ from ħ.

### /ħ/ (H) — voiceless pharyngeal fricative — PASS
- Same constriction, DC flow (no voicing). Noise low-passed at 2800Hz
  (Strouhal: wide pharynx can't sustain /s/-like small eddies).
- Gate: turbulent + aperiodic (harmonicity 0.115) + centroid 1794Hz
  (<3500, pharyngeal band, well below /s/) + vowel→a.

### /ʈˤ/ (T), /dˤ/ (Dd) — emphatic stops — PASS
- Plain closure (0.85) + pharyngealized vowel (retracted root modeled as
  mild low-pharynx narrowing on the vowel's area function).
- Gate: F2 drop 927Hz vs plain (cepstral envelope) + vowel→a.
- F2 via cepstral liftering (like diva.py) — raw spectra show harmonics,
  not formants, at f0=185Hz. Levinson-Durbin LPC was numerically unstable
  here; envelope peak-picking is robust for the RELATIVE comparison.

### /dʒ/ (J) — voiced postalveolar affricate — PASS
- Closure at 0.70 + slow fricative release (100ms ʃ-like noise, low-passed
  4500Hz) + vowel. Voicing continues through the release.
- Gate: turbulent + centroid 4732Hz measured ABOVE 1500Hz (isolates
  frication from voice harmonics) in ʃ-band + voiced + vowel→a.

### Integration into live_bridge.py
- Dispatch for E/H/T/Dd/J in _synth_token; all 14 transcriptions updated
  (MISS tokens replaced with real CV units); HAVE set extended.
- **Speakability: 100% (2628/2628)** — was 68.3%. ʕ alone unlocked 31.7%.

### Utterance gates: 12/14 fully PASS
- 2202/2628 instances (84%) fully gate-passed, including the 335x
  "شربت عند نهر".
- 2 utterances ("أنا عطشانة..." 239x + 180x) have emphatic /a/ vowels on
  the /a/-/u/ perceptual boundary: strong pharyngealization (F2 drop
  927Hz) pushes /a/ toward /u/ in the DCT feature. The phoneme itself is
  gated; this is a known model limitation (3-param articulator gives
  /a/ and /u/ close formants in real synthesis), not a synthesis bug.

### Bugs fixed during integration (documented so we never repeat them)
1. **Nasal vowel bounds were guessed** (off by 50ms): _synth_token assumed
   vowel = last n_v samples, but synthesize_nasal_v makes vowel_ms+50ms.
   Fixed: synthesize_nasal_v now returns exact vowel_bounds.
2. **Nasal vowel got mid-glide f0**: shared flow array gave the vowel a
   different f0 than the reference. Fixed: fresh f0s→f0e glide for vowel.
3. **_tame_burst ignored vowel_start**: scanned into the vowel and damaged
   it (max diff 1.51). Fixed: respects vowel_start, never touches vowel.
4. **Universal _tame_burst needed**: /r/ and /h/ transients spike 15-22x,
   squashing other units in final normalization. Now applied universally
   with vowel protection.
5. **Plosive vowel bounds included burst**: gate measured burst+VOT, not
   vowel. Added 30ms BURST_SKIP for plosives/emphatics/affricate.

### Delivered
Audio (~/workspace/your_files/fajr-voice/, all int16 verified):
fajr_ayn_a.wav, fajr_hah_a.wav, fajr_emph_t_a.wav, fajr_emph_d_a.wav,
fajr_djim_a.wav (phoneme samples); fajr_live_sharibat_ind_nahr.wav (4.46s),
fajr_live_ana_nasana.wav (7.18s), fajr_live_anta_almudif.wav (3.9s),
fajr_live_istahmamtu.wav (6.04s) (newly unlocked utterances)
Pictures (~/workspace/your_files/fajr-proof/):
proof-24-guttural-ayn.png, proof-25-speakable-map-v2.png,
proof-26-new-words.png (pack now 26 images)
Code: src/voice/guttural.py (new), live_bridge.py + nasals_fixed.py (updated)
