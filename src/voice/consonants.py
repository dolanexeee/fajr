"""Consonants — plosives via tract closure + burst release.

A voiced plosive = complete closure at some point -> glottal flow keeps
pumping -> pressure builds behind the seal -> sudden release -> the
built-up wave radiates as the burst transient -> vowel follows.

All of this is pure Kelly-Lochbaum waveguide physics: closure sets
near-zero area (reflection ~ -1, full), release restores it instantly,
and the transient emerges from the wave state. No artificial noise.

Places of articulation (closure position, 0=glottis, 1=lips):
    b: 0.97  bilabial (lips)       front cavity ~0cm
    d: 0.85  alveolar (ridge)      front cavity ~2-3cm
    g: 0.55  velar (soft palate)   front cavity ~6-8cm
The burst spectrum differs by front-cavity resonances — the honest
acoustic cue humans use. Gate checks centroid(d) > centroid(g) > centroid(b).
"""
import numpy as np
from scipy.signal import resample_poly

from .tract import VocalTract, HER_VOWELS, SR_TRACT
from .glottis import TwoMassGlottis
from .hearing import SR
from .diva import auditory_feature

CLOSURE_POS = {"b": 0.97, "d": 0.85, "g": 0.55}


def _oral_pressure_feedback(ug48_hold, tension):
    """Honest source-tract interaction during closure.

    The sealed oral cavity is a compliance: dPoral/dt = Ug/C.
    As Poral rises toward subglottal pressure, transglottal pressure
    drops, so glottal flow scales by (Psub-Poral)/Psub. This feedback
    naturally bounds the buildup to <= Psub — no tuned envelope.
    """
    # oral cavity compliance C = V/(rho*c^2), V~60cm^3 (cgs)
    C_oral = 60.0 / (0.00114 * 35000.0 ** 2)  # cm^5/dyn
    Psub_dyn = 800.0 * tension * 10.0  # Pa -> dyn/cm^2
    dt = 1.0 / SR_TRACT
    Poral = 0.0
    out = np.empty_like(ug48_hold)
    for i in range(ug48_hold.size):
        scale = (Psub_dyn - Poral) / Psub_dyn
        if scale < 0.0:
            scale = 0.0
        ug_s = ug48_hold[i] * scale
        out[i] = ug_s
        Poral += ug_s * dt / C_oral
    return out


def _steady_glottal_flow(tension, total_s):
    g = TwoMassGlottis()
    g.set_tension(tension)
    g.reset()
    g.x1 = 1e-6
    ug = g.run(total_s)
    ug = ug[int(0.2 * SR):]  # skip onset transient
    # UNIT FIX: glottis outputs m^3/s (SI); tract expects cm^3/s (cgs)
    return ug * 1e6


def synthesize_cv(plosive="b", vowel="a", hold_ms=60, vowel_ms=300,
                  tension=1.35):
    """Voiced plosive + vowel, e.g. /ba/. Returns 16 kHz waveform.

    Timeline: [closure hold] -> sudden release -> [vowel].
    Glottis runs continuously (voiced); during closure the sealed tract
    builds pressure and radiates ~nothing — the burst comes at release.
    """
    pos = CLOSURE_POS[plosive]
    hold_s = hold_ms / 1000.0
    vowel_s = vowel_ms / 1000.0
    ug = _steady_glottal_flow(tension, 0.2 + hold_s + vowel_s + 0.1)
    ug48 = resample_poly(ug, 3, 1)
    n_hold48 = int(hold_s * SR_TRACT)

    # Physical: oral-pressure feedback bounds the buildup (see above).
    # No artificial envelope — the time constant emerges from compliance.
    ug48_hold = _oral_pressure_feedback(ug48[:n_hold48], tension)
    # take exactly what we need (flow is 0.2s-onset-skipped)
    need48 = n_hold48 + int((vowel_s + 0.1) * SR_TRACT)
    ug48_rest = ug48[n_hold48:need48]

    tract = VocalTract()
    tract.set_articulators(*HER_VOWELS[vowel])
    tract.reset()

    # phase 1: sealed — pressure builds behind the closure.
    # Wall damping: a sealed oral cavity has low Q (viscous/thermal wall
    # losses + tissue compliance). Without it the lossless waveguide
    # builds unphysical standing waves. Applied per 10ms chunk as
    # wave-state decay, time constant ~60ms (Q~5-10, physical).
    tract.set_closure(pos, True)
    out_hold_parts = []
    chunk48 = int(0.01 * SR_TRACT)
    for i in range(0, n_hold48, chunk48):
        out_chunk = tract._run48(ug48_hold[i:i + chunk48])
        out_hold_parts.append(out_chunk)
        tract.up *= 0.85
        tract.dn *= 0.85
    out_hold = np.concatenate(out_hold_parts) if out_hold_parts else np.zeros(0)
    # phase 2: sudden release — the burst transient radiates
    tract.set_closure(pos, False)
    out_rest = tract._run48(ug48_rest)

    out48 = np.concatenate([out_hold, out_rest])
    out16 = resample_poly(out48, 1, 3).astype(float)
    mx = np.abs(out16).max()
    return out16 / (mx + 1e-12), hold_ms


def _spectral_centroid(x, sr=SR):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    return float(np.sum(f * X) / (np.sum(X) + 1e-12))


# ------------------------------------------------- voiceless plosives /p,t,k/
VOICELESS_PLACE = {"p": "b", "t": "d", "k": "g"}  # same articulation places


def _abducted_flow(tension, total_s, abduction=2.0):
    """Steady DC flow through open glottis (voiceless).

    Moderate abduction (2x rest width) prevents the flutter instability:
    Bernoulli forces stay small, folds sit near rest, flow is steady.
    Wider would give excessive DC flow -> unphysical pressure buildup.
    Returns cm^3/s, onset-skipped.
    """
    g = TwoMassGlottis()
    g.set_tension(tension)
    g.x01 = 5e-4 * abduction
    g.x02 = 5e-4 * abduction
    g.reset()
    g.x1 = 1e-6
    ug = g.run(total_s)
    ug = ug[int(0.15 * SR):]
    return ug * 1e6


def _aspiration_noise(n48, ug_dc, area_glottis, seed=0):
    """Reynolds-based turbulence noise for aspiration (VOT).

    Aeroacoustics: Re = v*d/nu. Above Re_c=1800 the glottal jet is
    turbulent and radiates noise. The white-noise randomness IS the
    turbulence; the amplitude scale is calibrated so aspiration sits
    ~-12dB re the vowel (speech data: /p/ VOT aspiration 10-15dB
    below vowel level). Returns (noise_waveform, Re).
    """
    rng = np.random.default_rng(seed)
    NU = 0.15  # cm^2/s, kinematic viscosity of air
    v = ug_dc / max(area_glottis, 1e-9)       # cm/s
    d = np.sqrt(max(area_glottis, 1e-9))      # cm, characteristic diameter
    Re = v * d / NU
    if Re < 1800:
        return np.zeros(n48), Re
    # calibrated amplitude (see VOICE-REALISM-NOTES.md): empirically set
    # so aspiration AC sits ~-15dB re vowel. Note: the VOT region has
    # large DC (steady flow); gate measures AC-only after highpass.
    amp = 36.0 * (1.0 + 0.2 * (Re - 1800) / 1800.0)  # cm^3/s RMS
    return rng.standard_normal(n48) * amp, Re


def synthesize_cv_voiceless(plosive="p", vowel="a", hold_ms=30, vot_ms=50,
                            vowel_ms=300, tension=1.35, seed=0):
    """Voiceless plosive + vowel, e.g. /pa/. Returns 16kHz waveform.

    Timeline: [closure: abducted DC flow, silence] -> [burst] ->
    [VOT: aspiration noise through open tract] -> [vowel: voicing].
    Short hold (30ms): steady DC builds pressure fast; longer leaks.
    Stronger wall damping (0.80) for the high-pressure DC buildup.
    """
    place = VOICELESS_PLACE[plosive]
    pos = CLOSURE_POS[place]
    hold_s, vot_s, vowel_s = hold_ms / 1000.0, vot_ms / 1000.0, vowel_ms / 1000.0

    # 1. hold: abducted steady flow + oral pressure feedback
    ug_hold16 = _abducted_flow(tension, 0.15 + hold_s + 0.05)
    ug_hold16 = ug_hold16[:int(hold_s * SR)]
    ug_hold48 = resample_poly(ug_hold16, 3, 1)
    ug_hold48 = _oral_pressure_feedback(ug_hold48, tension)
    ug_dc = float(np.mean(ug_hold16))  # DC level for VOT continuation

    # 2. VOT: DC flow + Reynolds-based aspiration noise (at 48kHz)
    n_vot48 = int(vot_s * SR_TRACT)
    # abducted glottal area: full width 2*x01_abd * length (abduction=2.0)
    area_g = (2 * 5e-4 * 2.0 * 100) * 1.4  # cm^2 (x01 in cm * L in cm)
    noise48, Re = _aspiration_noise(n_vot48, ug_dc, area_g, seed)
    ug_vot48 = np.full(n_vot48, ug_dc) + noise48

    # 3. vowel: normal voicing
    ug_vow16 = _steady_glottal_flow(tension, 0.2 + vowel_s + 0.1)
    ug_vow16 = ug_vow16[:int((vowel_s + 0.1) * SR)]
    ug_vowel48 = resample_poly(ug_vow16, 3, 1)

    n_hold48 = ug_hold48.size
    tract = VocalTract()
    tract.set_articulators(*HER_VOWELS[vowel])
    tract.reset()

    # phase 1: sealed (stronger wall damping for DC pressure buildup)
    # PLUS pressure clamp: a passive tract cannot exceed subglottal
    # pressure (energy conservation). Prevents unphysical standing-wave
    # buildup in the lossless waveguide.
    Psub_dyn = 800.0 * tension * 10.0  # Pa -> dyn/cm^2
    tract.set_closure(pos, True)
    out_hold_parts = []
    chunk48 = int(0.01 * SR_TRACT)
    for i in range(0, n_hold48, chunk48):
        out_chunk = tract._run48(ug_hold48[i:i + chunk48])
        out_hold_parts.append(out_chunk)
        tract.up *= 0.80
        tract.dn *= 0.80
        # clamp to physical pressure limit
        np.clip(tract.up, -Psub_dyn, Psub_dyn, out=tract.up)
        np.clip(tract.dn, -Psub_dyn, Psub_dyn, out=tract.dn)
    out_hold = np.concatenate(out_hold_parts) if out_hold_parts else np.zeros(0)
    # phases 2+3: release -> burst, then VOT aspiration, then vowel
    tract.set_closure(pos, False)
    out_rest = tract._run48(np.concatenate([ug_vot48, ug_vowel48]))

    out48 = np.concatenate([out_hold, out_rest])
    out16 = resample_poly(out48, 1, 3).astype(float)
    # DC blocker: steady flows create inaudible DC offsets that waste
    # dynamic range. 2nd-order highpass at 40Hz (physical: ears/mics
    # don't respond to DC; sound is AC pressure).
    from scipy.signal import butter, filtfilt
    b_hp, a_hp = butter(2, 40.0 / (SR / 2.0), 'high')
    out16 = filtfilt(b_hp, a_hp, out16)
    mx = np.abs(out16).max()
    info = {"Re_aspiration": round(float(Re), 1), "hold_ms": hold_ms,
            "vot_ms": vot_ms}
    return out16 / (mx + 1e-12), info


def _measure_vot(y, hold_ms, sr=SR):
    """Voice Onset Time: ms from burst to first STABLE periodic frame.

    Sliding 30ms windows, autocorrelation f0 detection. Requires f0
    stable within ±10% across 3 consecutive windows — rejects spurious
    periodicity from filtered aspiration noise (erratic) vs true voicing
    (stable). Returns VOT in ms, or None if no voicing found.
    """
    from .hearing import f0_autocorr
    n_hold = int(hold_ms / 1000.0 * sr)
    after = y[n_hold:]
    win = int(0.03 * sr)
    hop = int(0.005 * sr)
    f0s = []
    starts = []
    for start in range(0, len(after) - win, hop):
        seg = after[start:start + win]
        # skip the burst transient itself (first 15ms)
        if start < int(0.015 * sr):
            f0s.append(0.0)
            starts.append(start)
            continue
        f0 = f0_autocorr(seg)
        f0s.append(f0 if 60.0 < f0 < 600.0 else 0.0)
        starts.append(start)
    # find first run of 3 stable windows
    for i in range(len(f0s) - 2):
        a, b, c = f0s[i], f0s[i + 1], f0s[i + 2]
        if a > 0 and b > 0 and c > 0:
            if abs(b - a) / a < 0.10 and abs(c - b) / b < 0.10:
                return round((starts[i] / sr) * 1000.0, 1)
    return None


def _periodicity_strength(x, sr=SR, fmin=130.0, fmax=260.0):
    """Strength of periodic (voiced) component, 0..1.

    Max normalized autocorrelation peak in the expected f0 range
    (default 130-260Hz, her voice). Narrow range rejects spurious peaks
    from formant resonances in filtered noise (formants >500Hz).
    Voicing -> high (~0.7+); aspiration noise / silence -> low (<0.3).
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    if np.max(np.abs(x)) < 1e-9:
        return 0.0
    n = x.size
    corr = np.correlate(x, x, mode="full")[n - 1:]
    corr /= (corr[0] + 1e-12)
    lo = int(sr / fmax)
    hi = min(int(sr / fmin), n - 1)
    if hi <= lo:
        return 0.0
    return float(np.max(corr[lo:hi]))
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    if np.max(np.abs(x)) < 1e-9:
        return 0.0
    n = x.size
    corr = np.correlate(x, x, mode="full")[n - 1:]
    corr /= (corr[0] + 1e-12)
    lo = int(sr / 600.0)
    hi = min(int(sr / 60.0), n - 1)
    if hi <= lo:
        return 0.0
    return float(np.max(corr[lo:hi]))


def gate_voiceless_plosive(plosive="p", vowel="a", vowel_refs=None):
    """Gate: silent closure + burst + noise-like VOT + vowel.

    The key distinction vs voiced: NO voice bar (tail aperiodic) and
    VOT is aspiration noise (aperiodic), not immediate voicing.
    Returns (ok, diagnostics).
    """
    y, info = synthesize_cv_voiceless(plosive, vowel)
    hold_ms = info["hold_ms"]
    n_hold = int(hold_ms / 1000.0 * SR)
    hold = y[:n_hold]
    after = y[n_hold:]

    tail = hold[int(-0.04 * SR):]
    rms_tail = float(np.sqrt(np.mean(tail ** 2)))
    per_tail = _periodicity_strength(tail)
    steady = after[int(-0.15 * SR):]
    rms_vowel = float(np.sqrt(np.mean(steady ** 2)))
    burst = after[:int(0.03 * SR)]
    peak_burst = float(np.abs(burst).max())
    # VOT window: 20-40ms, clean aspiration
    vot_seg = after[int(0.020 * SR):int(0.040 * SR)]
    rms_vot = float(np.sqrt(np.mean(vot_seg ** 2)))
    per_vot = _periodicity_strength(vot_seg)
    # voicing window: 60-90ms, should be periodic
    voice_seg = after[int(0.060 * SR):int(0.090 * SR)]
    per_voice = _periodicity_strength(voice_seg)

    diag = {
        "Re_aspiration": info["Re_aspiration"],
        "rms_tail": round(rms_tail, 4),
        "per_tail": round(per_tail, 3),
        "rms_vot": round(rms_vot, 4),
        "per_vot": round(per_vot, 3),
        "per_voice": round(per_voice, 3),
        "rms_vowel": round(rms_vowel, 4),
        "peak_burst": round(peak_burst, 4),
    }
    ok = True
    # 1. closure has NO voice bar (aperiodic) — the voiceless distinction
    c1 = per_tail < 0.35
    diag["no_voice_bar"] = bool(c1)
    ok &= c1
    # 2. burst present
    c2 = peak_burst > 3.0 * rms_tail
    diag["burst_present"] = bool(c2)
    ok &= c2
    # 3. VOT is aspiration NOISE (aperiodic, audible above floor)
    c3 = (per_vot < 0.35) and (rms_vot > 1.5 * rms_tail)
    diag["vot_aperiodic"] = bool(c3)
    ok &= c3
    # 4. voicing arrives later (periodic after VOT)
    c4 = per_voice > 0.5
    diag["voicing_after_vot"] = bool(c4)
    ok &= c4
    # 5. vowel recognized
    if vowel_refs is not None:
        f = auditory_feature(steady / (np.abs(steady).max() + 1e-12))
        dists = {v: float(np.linalg.norm(f - vowel_refs[v])) for v in vowel_refs}
        recog = min(dists, key=dists.get)
        diag["vowel_recognized"] = recog
        c6 = (recog == vowel)
        diag["vowel_ok"] = bool(c6)
        ok &= c6
    return ok, diag


def gate_plosive(plosive="b", vowel="a", vowel_refs=None):
    """Gate: closure silence + burst transient + vowel recognized.

    Returns (ok, diagnostics dict).
    """
    y, hold_ms = synthesize_cv(plosive, vowel)
    n_hold = int(hold_ms / 1000.0 * SR)
    hold = y[:n_hold]
    after = y[n_hold:]

    rms_hold = float(np.sqrt(np.mean(hold ** 2)))
    # voice bar: voicing decays through the closure as pressure equalizes
    # check the quiet tail (last 40 ms) — should be well below vowel level
    tail = hold[int(-0.04 * SR):]
    rms_tail = float(np.sqrt(np.mean(tail ** 2)))
    # steady vowel: last 150 ms
    steady = after[int(-0.15 * SR):]
    rms_vowel = float(np.sqrt(np.mean(steady ** 2)))
    # burst: first 30 ms after release
    burst = after[:int(0.03 * SR)]
    peak_burst = float(np.abs(burst).max())

    diag = {
        "rms_hold": round(rms_hold, 4),
        "rms_tail": round(rms_tail, 4),
        "rms_vowel": round(rms_vowel, 4),
        "peak_burst": round(peak_burst, 4),
    }
    ok = True
    # 1. closure tail is quiet relative to the vowel (voice bar decayed)
    c1 = rms_tail < 0.30 * rms_vowel
    diag["closure_quiet"] = bool(c1)
    ok &= c1
    # 2. burst transient stands out over the quiet tail
    c2 = peak_burst > 3.0 * rms_tail
    diag["burst_present"] = bool(c2)
    ok &= c2
    # 3. burst is broadband (plosive, not a tone)
    if len(burst) > 256:
        X = np.abs(np.fft.rfft(burst * np.hanning(len(burst))))
        Xn = X / (X.sum() + 1e-12)
        flatness = float(np.exp(np.mean(np.log(Xn + 1e-12))) / (np.mean(Xn) + 1e-12))
        diag["burst_flatness"] = round(flatness, 3)
        c3 = flatness > 0.15
        diag["burst_broadband"] = bool(c3)
        ok &= c3
    # 4. vowel recognized after the burst
    if vowel_refs is not None:
        f = auditory_feature(steady / (np.abs(steady).max() + 1e-12))
        dists = {v: float(np.linalg.norm(f - vowel_refs[v])) for v in vowel_refs}
        recog = min(dists, key=dists.get)
        diag["vowel_recognized"] = recog
        c4 = (recog == vowel)
        diag["vowel_ok"] = bool(c4)
        ok &= c4
    # 5. burst spectral centroid (for place-of-articulation ordering)
    # use first 10 ms only — the burst proper, before vowel onset colors it
    burst10 = after[:int(0.01 * SR)]
    diag["burst_centroid_hz"] = round(_spectral_centroid(burst10), 1)
    return ok, diag
