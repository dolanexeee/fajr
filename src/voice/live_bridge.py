"""Live bridge: Fajr's real utterances -> her new voice.

Her life speaks Arabic text (Tongue._emit). Her voice speaks phonemes.
This module is the honest bridge between them.

Design:
- TRANSCRIPTIONS: hand phoneme transcriptions of her 14 real utterances
  (her inventory only — NOT a general Arabic G2P; that would be a claim
  we cannot gate).
- render_utterance(text): transcription -> synthesis units -> zero-gap
  30ms-crossfade assembly (the proven baba recipe) -> int16 audio.
- Missing phonemes are NEVER faked: the utterance is marked not-yet-
  speakable and the exact missing set is recorded.
- gate_utterance(text): every vowel segment recognized with exact
  boundaries against references built with identical parameters.

Phoneme key (ASCII):
  vowels: a i u
  plosives: b d g p t k      fricatives: s f sh
  nasals: m n                lateral: l        glottal fricative: h
  tap: r                     glottal stop: Q   glides: w j
  long vowel: V: (1.5x)      emphatics/gutturals NOT in inventory:
                             E=ʕ  H=ħ  T=ʈˤ  Dd=dˤ  J=dʒ  K=q ...
"""
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from .arabic_units import (
    unit_vowel, synthesize_lateral_v, synthesize_h_v, glottal_stop,
    synthesize_tap_v, synthesize_sh_v, glide_sound, unit_final_stop, SR,
)
from .guttural import (
    synthesize_ayn_v, synthesize_hah_v, synthesize_emphatic_cv,
    synthesize_djim_v,
)
from .syllables import synthesize_cv
from .fricatives_human import synthesize_fricative_v
from .nasals_fixed import synthesize_nasal_v
from .diva import auditory_feature

# ------------------------------------------------------------------ #
# Transcriptions. Each utterance -> list of tokens.
# Token forms:
#   ('V', v, scale)       vowel, scale = duration multiplier
#   ('CV', c, v, scale)   consonant+vowel syllable
#   ('Q',)                glottal stop
#   ('FINAL', c)          word-final consonant
#   ('GLIDE', g, v)       glide w/j into vowel v
#   ('MISS', symbol)      phoneme we do NOT have (never synthesized)
# ------------------------------------------------------------------ #
T = {
    # 1469x — fully speakable
    "أكلت فاكهة": [
        ("Q",), ("V", "a", 0.45), ("CV", "k", "a", 1.0), ("CV", "l", "a", 1.0),
        ("FINAL", "t"),
        ("CV", "f", "a", 1.5), ("CV", "k", "i", 1.0), ("CV", "h", "a", 1.0),
    ],
    # 335x — ʕ now built
    "شربت عند نهر": [
        ("CV", "sh", "a", 1.0), ("CV", "r", "i", 1.0), ("FINAL", "b"), ("FINAL", "t"),
        ("CV", "E", "i", 0.7), ("CV", "n", "a", 0.35), ("CV", "d", "a", 1.0),
        ("CV", "n", "a", 1.0), ("CV", "h", "a", 1.0), ("CV", "r", "a", 0.6),
    ],
    # 306x — fully speakable
    "شربت": [
        ("CV", "sh", "a", 1.0), ("CV", "r", "i", 1.0), ("FINAL", "b"), ("FINAL", "t"),
    ],
    # 239x — ʕ, ʈˤ, dʒ now built
    "أنا عطشانة جدًا. أريد أن أشرب": [
        ("Q",), ("V", "a", 1.0), ("CV", "n", "a", 1.0),
        ("CV", "E", "a", 1.0), ("CV", "T", "a", 1.5),
        ("CV", "n", "a", 1.0),
        ("CV", "J", "i", 0.7), ("CV", "d", "a", 0.5), ("CV", "d", "a", 0.8),
        ("CV", "n", "a", 0.8),
        ("Q",), ("V", "u", 1.0), ("CV", "r", "i", 1.5), ("FINAL", "d"),
        ("Q",), ("V", "a", 1.0), ("CV", "n", "a", 1.0),
        ("Q",), ("V", "a", 1.0), ("CV", "sh", "a", 1.0), ("CV", "r", "a", 1.0),
        ("FINAL", "b"),
    ],
    # 180x — ʕ, ʈˤ now built
    "أنا عطشانة. أريد أن أشرب": [
        ("Q",), ("V", "a", 1.0), ("CV", "n", "a", 1.0),
        ("CV", "E", "a", 1.0), ("CV", "T", "a", 1.5),
        ("CV", "n", "a", 1.0),
        ("Q",), ("V", "u", 1.0), ("CV", "r", "i", 1.5), ("FINAL", "d"),
        ("Q",), ("V", "a", 1.0), ("CV", "n", "a", 1.0),
        ("Q",), ("V", "a", 1.0), ("CV", "sh", "a", 1.0), ("CV", "r", "a", 1.0),
        ("FINAL", "b"),
    ],
    # 60x — ʕ now built
    "أنا نعسانة. أريد أن أنام": [
        ("Q",), ("V", "a", 1.0), ("CV", "n", "a", 1.0),
        ("CV", "n", "a", 1.0), ("CV", "E", "a", 1.0), ("CV", "s", "a", 1.5), ("CV", "n", "a", 1.0),
        ("Q",), ("V", "u", 1.0), ("CV", "r", "i", 1.5), ("FINAL", "d"),
        ("Q",), ("V", "a", 1.0), ("CV", "n", "a", 1.0),
        ("Q",), ("V", "a", 1.0), ("CV", "n", "a", 1.5), ("CV", "m", "a", 1.0),
    ],
    # 7x — ʕ now built
    "أكلت فاكهة عند مأوى": [
        ("Q",), ("V", "a", 0.45), ("CV", "k", "a", 1.0), ("CV", "l", "a", 1.0),
        ("FINAL", "t"),
        ("CV", "f", "a", 1.5), ("CV", "k", "i", 1.0), ("CV", "h", "a", 1.0),
        ("CV", "E", "i", 0.7), ("CV", "n", "a", 0.35), ("CV", "d", "a", 1.0),
        ("CV", "m", "a", 1.0), ("Q",), ("GLIDE", "w", "a", 1.0),
    ],
    # 3x — ʕ, dʒ now built
    "أنا نعسانة جدًا. أريد أن أنام": [
        ("Q",), ("V", "a", 1.0), ("CV", "n", "a", 1.0),
        ("CV", "n", "a", 1.0), ("CV", "E", "a", 1.0), ("CV", "s", "a", 1.5), ("CV", "n", "a", 1.0),
        ("CV", "J", "i", 0.7), ("CV", "d", "a", 0.5), ("CV", "d", "a", 0.8),
        ("CV", "n", "a", 0.8),
        ("Q",), ("V", "u", 1.0), ("CV", "r", "i", 1.5), ("FINAL", "d"),
        ("Q",), ("V", "a", 1.0), ("CV", "n", "a", 1.0),
        ("Q",), ("V", "a", 1.0), ("CV", "n", "a", 1.5), ("CV", "m", "a", 1.0),
    ],
    # 2x — dˤ now built
    "أنت المضيف": [
        ("Q",), ("V", "a", 1.0), ("CV", "n", "a", 0.35), ("FINAL", "t"), ("V", "a", 1.0),
        ("V", "a", 0.8), ("CV", "l", "a", 0.4), ("CV", "m", "u", 0.8),
        ("CV", "Dd", "i", 1.5), ("CV", "f", "a", 1.0),
    ],
    # 2x — ʕ now built
    "لا. أنا نعسانة. أريد أن أنام": [
        ("CV", "l", "a", 1.5),
        ("Q",), ("V", "a", 1.0), ("CV", "n", "a", 1.0),
        ("CV", "n", "a", 1.0), ("CV", "E", "a", 1.0), ("CV", "s", "a", 1.5), ("CV", "n", "a", 1.0),
        ("Q",), ("V", "u", 1.0), ("CV", "r", "i", 1.5), ("FINAL", "d"),
        ("Q",), ("V", "a", 1.0), ("CV", "n", "a", 1.0),
        ("Q",), ("V", "a", 1.0), ("CV", "n", "a", 1.5), ("CV", "m", "a", 1.0),
    ],
    # 1x — fully speakable
    "أكلت": [
        ("Q",), ("V", "a", 0.45), ("CV", "k", "a", 1.0), ("CV", "l", "a", 1.0),
        ("FINAL", "t"),
    ],
    # 1x — ʕ, dʒ now built
    "لا أعرف كلمة «جائعة»": [
        ("CV", "l", "a", 1.5),
        ("Q",), ("V", "a", 0.5), ("CV", "E", "a", 0.6), ("CV", "r", "i", 1.0), ("CV", "f", "a", 1.0),
        ("CV", "k", "a", 1.0), ("CV", "l", "i", 1.0), ("CV", "m", "a", 1.0),
        ("V", "a", 1.0),
        ("CV", "J", "a", 1.5), ("Q",), ("V", "i", 1.0), ("CV", "E", "a", 1.0),
    ],
    # 1x — ħ, ʕ now built
    "استحممت عند نهر": [
        ("V", "i", 0.8), ("CV", "s", "a", 0.35), ("FINAL", "t"), ("V", "a", 0.8),
        ("CV", "H", "a", 0.8), ("CV", "m", "a", 0.8), ("CV", "m", "a", 0.5), ("FINAL", "t"),
        ("CV", "E", "i", 0.7), ("CV", "n", "a", 0.35), ("CV", "d", "a", 1.0),
        ("CV", "n", "a", 1.0), ("CV", "h", "a", 1.0), ("CV", "r", "a", 0.6),
    ],
    # 1x — ʕ, ʈˤ now built
    "نعم، أنا عطشانة": [
        ("CV", "n", "a", 1.0), ("CV", "E", "a", 1.0), ("CV", "m", "a", 1.0),
        ("Q",), ("V", "a", 1.0), ("CV", "n", "a", 1.0),
        ("CV", "E", "a", 1.0), ("CV", "T", "a", 1.5),
        ("CV", "n", "a", 1.0),
    ],
}

# symbol -> human name for the missing-phoneme report
MISSING_NAMES = {
    "E": "ʕ (ع)", "H": "ħ (ح)", "T": "ʈˤ (ط)", "Dd": "dˤ (ض)",
    "J": "dʒ (ج)", "K": "q (ق)",
}

# inventory actually implemented
HAVE = set("aiu") | {"b", "d", "g", "p", "t", "k", "s", "f", "sh",
                     "m", "n", "l", "h", "r", "Q", "w", "j",
                     "E", "H", "T", "Dd", "J"}


# ------------------------------------------------------------------ #
# unit synthesis dispatch. Each returns (audio, vowel_bounds_relative)
# ------------------------------------------------------------------ #
# Minimum vowel duration for gateability: auditory_feature skips the first
# 100ms and needs 2048 samples after that (~228ms total). Her speech is
# therefore slow and deliberate — developmentally honest for a first
# speaker. Base vowel 350ms keeps trimmed segments above the 230ms
# the auditory feature needs. Faster speech is future work, not a faked gate.
MIN_VOWEL_MS = 300


def _tame_burst(y, vowel_start=None, sr=SR, max_ratio=4.0):
    """Tame plosive burst transients to a natural level.
    The waveguide builds unrealistic pressure in the closure (100x+
    burst/vowel measured); real speech is ~2-4x. Self-locating: the
    vowel steady-state is the last 200ms (reference RMS); any peak
    above 6x that RMS in the pre-vowel part is a transient and gets
    scaled to 4x with raised-cosine edges. The vowel itself is NEVER
    touched (scan ends at vowel_start)."""
    y = np.asarray(y, dtype=float).copy()
    n = len(y)
    if n < int(0.3 * sr):
        return y
    ref = y[-int(0.2 * sr):]
    v_rms = float(np.sqrt(np.mean(ref ** 2))) + 1e-12
    # NEVER scan the vowel: stop at vowel_start if given, else 80%
    if vowel_start is not None:
        scan_end = min(int(vowel_start), n)
    else:
        scan_end = int(n * 0.8)
    peak = float(np.abs(y[:scan_end]).max()) if scan_end > 0 else 0.0
    if peak <= 6.0 * v_rms:
        return y
    nw = int(0.04 * sr)
    nf = min(int(0.005 * sr), nw // 4)
    fade = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, nf))
    i = 0
    while i < scan_end:
        w0, w1 = i, min(i + nw, scan_end)
        wpk = float(np.abs(y[w0:w1]).max())
        if wpk > 6.0 * v_rms:
            scale = (max_ratio * v_rms) / (wpk + 1e-12)
            m = w1 - w0
            ramp = np.ones(m)
            f = min(nf, m // 4)
            if f > 0:
                ramp[:f] = 1.0 - (1.0 - scale) * fade[:f]
                ramp[-f:] = 1.0 - (1.0 - scale) * fade[:f][::-1]
                ramp[f:m - f] = scale
            else:
                ramp[:] = scale
            y[w0:w1] = y[w0:w1] * ramp
            i = w1
        else:
            i += int(0.01 * sr)
    return y


def _synth_token(tok, f0s, f0e):
    kind = tok[0]
    if kind == "V":
        _, v, scale = tok
        dur = max(int(350 * scale), MIN_VOWEL_MS)
        y, info = unit_vowel(v, dur_ms=dur, f0s=f0s, f0e=f0e)
        return y, info["vowel_bounds"]
    if kind == "CV":
        _, c, v, scale = tok
        vms = max(int(350 * scale), MIN_VOWEL_MS)
        kw = dict(f0s=f0s, f0e=f0e)
        # Burst/VOT skip: plosive vowels start with release transient +
        # aspiration; the steady vowel (what the gate checks) begins ~60ms
        # after release. Bounds skip it so gates measure the vowel, not
        # the burst.
        BURST_SKIP = int(0.030 * SR)
        if c in ("b", "d", "g", "p", "t", "k"):
            y = synthesize_cv(c, v, hold_ms=70,
                              vowel_ms=vms, **kw)
            n_v = int(vms / 1000 * SR)
            # tame the unrealistic burst transient (documented above)
            y = _tame_burst(y, len(y) - n_v)
            vs = len(y) - n_v + BURST_SKIP
            return y, [(v, vs, len(y))]
        if c in ("s", "f"):
            y, _ = synthesize_fricative_v(c, v, fric_ms=200,
                                          vowel_ms=vms, **kw)
            n_v = int(vms / 1000 * SR)
            y = _tame_burst(y)
            return y, [(v, len(y) - n_v, len(y))]
        if c == "sh":
            y, info = synthesize_sh_v(v, fric_ms=200,
                                      vowel_ms=vms, **kw)
            n_v = int(vms / 1000 * SR)
            y = _tame_burst(y)
            return y, [(v, len(y) - n_v, len(y))]
        if c in ("m", "n"):
            y, info = synthesize_nasal_v(c, v, murmur_ms=220,
                                         vowel_ms=vms, **kw)
            return y, info["vowel_bounds"]
        if c == "l":
            y, info = synthesize_lateral_v(v, lat_ms=180,
                                           vowel_ms=vms, **kw)
            return y, info["vowel_bounds"]
        if c == "h":
            y, info = synthesize_h_v(v, h_ms=150,
                                     vowel_ms=vms, **kw)
            return y, info["vowel_bounds"]
        if c == "r":
            y, info = synthesize_tap_v(v, tap_ms=25,
                                       vowel_ms=vms, **kw)
            return y, info["vowel_bounds"]
        if c == "E":
            y, info = synthesize_ayn_v(v, ayn_ms=200,
                                       vowel_ms=vms, **kw)
            return y, info["vowel_bounds"]
        if c == "H":
            y, info = synthesize_hah_v(v, hah_ms=200,
                                       vowel_ms=vms, **kw)
            return y, info["vowel_bounds"]
        if c == "T":
            y, info = synthesize_emphatic_cv("t", v, hold_ms=70,
                                             vowel_ms=vms, **kw)
            n_v = int(vms / 1000 * SR)
            y = _tame_burst(y, len(y) - n_v)
            vs, ve = info["vowel_bounds"][0][1], info["vowel_bounds"][0][2]
            return y, [(v, vs + BURST_SKIP, ve)]
        if c == "Dd":
            y, info = synthesize_emphatic_cv("d", v, hold_ms=70,
                                             vowel_ms=vms, **kw)
            n_v = int(vms / 1000 * SR)
            y = _tame_burst(y, len(y) - n_v)
            vs, ve = info["vowel_bounds"][0][1], info["vowel_bounds"][0][2]
            return y, [(v, vs + BURST_SKIP, ve)]
        if c == "J":
            y, info = synthesize_djim_v(v, hold_ms=60, fric_ms=100,
                                        vowel_ms=vms, **kw)
            n_v = int(vms / 1000 * SR)
            y = _tame_burst(y, len(y) - n_v)
            vs, ve = info["vowel_bounds"][0][1], info["vowel_bounds"][0][2]
            return y, [(v, vs + BURST_SKIP, ve)]
        raise ValueError(f"no synth for consonant {c}")
    if kind == "Q":
        y, _ = glottal_stop(45)
        return y, []
    if kind == "FINAL":
        _, c = tok
        y, _ = unit_final_stop(c, f0s=f0s, f0e=f0e)
        return y, []
    if kind == "GLIDE":
        _, g, v, scale = tok
        yg, _ = glide_sound(g, dur_ms=70, f0s=f0s, f0e=f0e)
        yv, vb = unit_vowel(v, dur_ms=max(int(200 * scale), 250),
                            f0s=f0s, f0e=f0e)
        y = np.concatenate([yg, yv])
        return y, [(v, len(yg), len(y))]
    raise ValueError(f"unknown token {tok}")


def missing_phonemes(text):
    """Exact set of phoneme symbols in this utterance we cannot synthesize."""
    return {tok[1] for tok in T[text] if tok[0] == "MISS"}


def speakability(text):
    """green = speakable now, yellow = one missing phoneme class,
    red = two or more missing."""
    m = missing_phonemes(text)
    if not m:
        return "green"
    return "yellow" if len(m) == 1 else "red"


# ------------------------------------------------------------------ #
# assembly: zero-gap crossfade, per-unit vowel-level normalization
# ------------------------------------------------------------------ #
def _xfade(a, b, ms=30):
    n = int(ms / 1000.0 * SR)
    n = min(n, len(a) // 2, len(b) // 2)
    if n <= 0:
        return np.concatenate([a, b])
    w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
    tail = a[-n:] * (1 - w) + b[:n] * w
    return np.concatenate([a[:-n], tail, b[n:]])


def render_utterance(text, f0_start=190.0, f0_end=168.0):
    """Render a speakable utterance. Returns (audio_int16_float, vowel_segs).
    vowel_segs = [(vowel, start_sample, end_sample)] absolute boundaries.
    Raises ValueError if the utterance has missing phonemes."""
    miss = missing_phonemes(text)
    if miss:
        raise ValueError(f"not speakable, missing: {sorted(miss)}")
    toks = [t for t in T[text] if t[0] != "MISS"]
    n = len(toks)
    pieces = []
    vsegs = []
    offset = 0
    for i, tok in enumerate(toks):
        f = i / max(1, n - 1)
        f0s = f0_start + (f0_end - f0_start) * f
        f0e = f0_start + (f0_end - f0_start) * min(1.0, f + 1.0 / max(1, n - 1))
        y, vb = _synth_token(tok, f0s, f0e)
        # normalize by vowel level when the unit has one, else gentle peak norm
        if vb:
            vlev = np.sqrt(np.mean(y[vb[0][1]:vb[0][2]] ** 2)) + 1e-12
            y = y / vlev * 0.5
        else:
            y = y / (np.abs(y).max() + 1e-12) * 0.35
        # tame transients UNIVERSALLY: tap releases, frication onsets, etc.
        # can spike 15-22x the vowel RMS; without this the final
        # max-normalization squashes quieter units to silence.
        # vowel_start protects the vowel: only pre-vowel transients tamed.
        vstart = vb[0][1] if vb else None
        y = _tame_burst(y, vstart)
        pieces.append((y, vb, offset, f0s, f0e))
        offset += len(y)
    # assemble with crossfade, tracking absolute vowel boundaries + f0
    out = pieces[0][0]
    abs_vsegs = [(v, a, b, pieces[0][3], pieces[0][4])
                 for (v, a, b) in pieces[0][1]]
    xf = int(30 / 1000.0 * SR)
    for (y, vb, _, f0s, f0e) in pieces[1:]:
        # shift previously recorded boundaries: they stay valid (only tail changes)
        out = _xfade(out, y, ms=30)
        base = len(out) - len(y)
        for (v, a, b) in vb:
            abs_vsegs.append((v, base + a, base + b, f0s, f0e))
    # NOTE: no tanh soft-clip and no global limiter. Tanh measurably
    # destroys vowel identity (/i/ at 0.08 without tanh confuses with /a/
    # at 1.98 with tanh); a global limiter touches vowel peaks above 0.9
    # and broke the gate. Burst dynamics are tamed at the source instead
    # (_tame_burst in _synth_token: transients -> max 4x vowel RMS, steady
    # vowels bit-identical). Final max-normalization only.
    out = out / (np.abs(out).max() + 1e-12)
    return out, abs_vsegs


def _build_refs(f0_start=190.0, f0_end=168.0):
    refs = {}
    for v in ("a", "i", "u"):
        y, _ = unit_vowel(v, dur_ms=300, f0s=f0_start, f0e=f0_end)
        refs[v] = auditory_feature(y / (np.abs(y).max() + 1e-12))
    return refs


def gate_utterance(text):
    """Every vowel segment recognized with exact boundaries.
    References built with the SAME f0 and duration as each test
    segment (the established rule: matched parameters or the gate
    measures the mismatch, not the vowel)."""
    miss = missing_phonemes(text)
    if miss:
        return False, {"missing": sorted(miss)}
    y, vsegs = render_utterance(text)
    ok = True
    per = []
    trim = int(0.035 * SR)  # exclude the 30ms crossfade blends at edges
    for (v, a, b, f0s, f0e) in vsegs:
        a2, b2 = a + trim, b - trim
        if b2 - a2 < int(0.23 * SR):
            a2, b2 = a, b
        seg = y[a2:b2]
        if len(seg) < int(0.23 * SR):
            per.append((v, "too_short", None))
            ok = False
            continue
        # matched reference: same vowel, same f0, same duration
        dur_ms = len(seg) / SR * 1000.0
        yref, _ = unit_vowel(v, dur_ms=dur_ms, f0s=f0s, f0e=f0e)
        refs = {}
        for rv in ("a", "i", "u"):
            yr, _ = unit_vowel(rv, dur_ms=dur_ms, f0s=f0s, f0e=f0e)
            refs[rv] = auditory_feature(yr / (np.abs(yr).max() + 1e-12))
        f = auditory_feature(seg / (np.abs(seg).max() + 1e-12))
        dists = {rv: float(np.linalg.norm(f - refs[rv])) for rv in refs}
        recog = min(dists, key=dists.get)
        good = (recog == v)
        ok = ok and good
        per.append((v, recog, round(min(dists.values()), 2), good))
    return ok, {"vowels": per, "n_vowels": len(vsegs)}


def utterance_stats():
    """(text, count) for her 14 utterances — filled by survey_utterances().

    Degrades gracefully when the private journal is absent (e.g. in the
    open-source tree, where life/ is excluded): reports the missing input
    and returns an empty counter instead of raising FileNotFoundError.
    """
    import json, os
    from collections import Counter
    J = os.path.join(os.path.dirname(__file__), "..", "..", "life", "journal.jsonl")
    c = Counter()
    try:
        fh = open(J)
    except FileNotFoundError:
        print(f"NOTE: journal not found at {J} — private biography excluded "
              f"from this tree; utterance table will be empty.")
        return c
    with fh:
        for line in fh:
            o = json.loads(line)
            if o["type"] == "utterance":
                c[o["text"]] += 1
    return c


if __name__ == "__main__":
    c = utterance_stats()
    for text, count in c.most_common():
        sp = speakability(text)
        miss = sorted(missing_phonemes(text))
        mnames = [MISSING_NAMES.get(m, m) for m in miss]
        print(f"{count:5d}  [{sp:6s}]  {text[:40]:42s}  missing={mnames}")


def latin_transcription(text):
    """Readable Latin rendering of the phoneme transcription
    (for displays where Arabic script renders disconnected)."""
    sym = {"Q": "ʔ", "E": "ʕ", "H": "ħ", "T": "ṭ", "Dd": "ḍ", "J": "ǧ"}
    out = []
    for tok in T[text]:
        k = tok[0]
        if k == "Q":
            out.append("ʔ")
        elif k == "V":
            _, v, sc = tok
            out.append(v.upper() if sc >= 1.4 else v)
        elif k == "CV":
            _, c, v, sc = tok
            vv = v.upper() if sc >= 1.4 else v
            out.append(c + vv)
        elif k == "FINAL":
            out.append(tok[1])
        elif k == "GLIDE":
            out.append(tok[1] + tok[2])
        elif k == "MISS":
            out.append("[" + sym.get(tok[1], tok[1]) + "]")
    return " ".join(out)
