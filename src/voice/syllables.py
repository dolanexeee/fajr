"""Human-tract consonant-vowel syllables — the path to intelligible words.

Adapts the proven plosive physics (closure -> pressure buildup -> burst)
to the human-scale tract (20 sections, 14.6cm) with the warm Rosenberg
source and human-optimized vowel articulations.

Closure positions are fractions of tract length, so they scale automatically.
Oral cavity volume scales with tract size for correct pressure timing.
"""
import json
import numpy as np
from scipy.signal import resample_poly

from .tract import VocalTract, _areas_from_articulators, SR_TRACT
from .rosenberg import rosenberg_train, SR
from .diva import auditory_feature

HUMAN_N = 20
CLOSURE_POS = {"b": 0.97, "d": 0.85, "g": 0.55, "p": 0.97, "t": 0.85, "k": 0.55}

import os
_HV = json.load(open(os.path.join(os.path.dirname(__file__), 'human_vowels.json')))
HUMAN_ART = {v: tuple(_HV[v]['art']) for v in ('a', 'i', 'u')}


def _oral_pressure_feedback(ug48_hold, volume_cm3=27.0):
    """Same physics as consonants.py, scaled oral volume for human tract."""
    C_oral = volume_cm3 / (0.00114 * 35000.0 ** 2)
    Psub_dyn = 800.0 * 1.35 * 10.0
    dt = 1.0 / SR_TRACT
    Poral = 0.0
    out = np.empty_like(ug48_hold)
    for i in range(ug48_hold.size):
        scale = max(0.0, (Psub_dyn - Poral) / Psub_dyn)
        ug_s = ug48_hold[i] * scale
        out[i] = ug_s
        Poral += ug_s * dt / C_oral
    return out


def synthesize_cv(plosive="b", vowel="a", hold_ms=70, vowel_ms=280,
                  f0s=185, f0e=172):
    """Voiced/voiceless plosive + vowel on the human tract. Returns 16kHz."""
    pos = CLOSURE_POS[plosive]
    voiceless = plosive in ("p", "t", "k")
    hold_s = hold_ms / 1000.0
    vowel_s = vowel_ms / 1000.0
    total = hold_s + vowel_s + 0.15

    if voiceless:
        # abducted: steady DC flow (no voicing during closure)
        flow_hold = rosenberg_train(f0s, f0e, total) * 0.0 + 1.2e-4
        flow_hold = flow_hold * 1e6
        # vowel: VOICED flow (voicing starts after VOT)
        flow_vow = rosenberg_train(f0s, f0e, total) * 1e6
    else:
        flow_hold = rosenberg_train(f0s, f0e, total) * 1e6
        flow_vow = flow_hold
    ug48_hold_full = resample_poly(flow_hold, 3, 1)
    ug48_vow_full = resample_poly(flow_vow, 3, 1)
    n_hold48 = int(hold_s * SR_TRACT)

    ug48_hold = _oral_pressure_feedback(ug48_hold_full[:n_hold48])
    need48 = n_hold48 + int((vowel_s + 0.1) * SR_TRACT)
    ug48_rest = ug48_vow_full[n_hold48:need48]

    tract = VocalTract(n_sec=HUMAN_N)
    tract.set_areas(_areas_from_articulators(*HUMAN_ART[vowel], n=HUMAN_N))
    tract.reset()

    # phase 1: sealed
    tract.set_closure(pos, True)
    out_hold_parts = []
    chunk48 = int(0.01 * SR_TRACT)
    for i in range(0, n_hold48, chunk48):
        out_chunk = tract._run48(ug48_hold[i:i + chunk48])
        out_hold_parts.append(out_chunk)
        tract.up *= 0.85
        tract.dn *= 0.85
    out_hold = np.concatenate(out_hold_parts) if out_hold_parts else np.zeros(0)

    # phase 2: release + vowel
    tract.set_closure(pos, False)
    out_rest = tract._run48(ug48_rest)

    out48 = np.concatenate([out_hold, out_rest])
    if voiceless:
        # VOT aspiration: filtered noise burst after release
        n_vot = int(0.05 * SR_TRACT)
        rng = np.random.default_rng(3)
        noise = rng.standard_normal(n_vot) * 0.15
        # bandpass-ish: differentiate to whiten
        noise = np.diff(noise, prepend=0)
        start = n_hold48
        out48[start:start + n_vot] += noise * np.abs(out48).max()
    out16 = resample_poly(out48, 1, 3).astype(float)
    mx = np.abs(out16).max()
    return out16 / (mx + 1e-12)


def synthesize_word(syllables, gap_ms=120):
    """Syllables like [('b','a'),('b','a')] -> 'baba'."""
    gap = np.zeros(int(gap_ms / 1000 * SR))
    parts = []
    for i, (p, v) in enumerate(syllables):
        # gentle f0 declination across the word
        f0s = 190 - i * 8
        parts.append(synthesize_cv(p, v, f0s=f0s, f0e=f0s - 10))
        if i < len(syllables) - 1:
            parts.append(gap)
    y = np.concatenate(parts)
    return y / (np.abs(y).max() + 1e-12)
