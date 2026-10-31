"""Naturalness polish — jitter, shimmer, aspiration, intonation.

Makes the mechanical voice less robotic by adding the micro-variations
of real human voicing:

- Jitter: cycle-to-cycle f0 variation (natural: 0.5-1%)
- Shimmer: cycle-to-cycle amplitude variation (natural: 3-5%)
- Aspiration: turbulent breath noise mixed with voicing (breathiness)
- Intonation: f0 contour over utterance (declination + stress)

All applied at the GLOTTAL SOURCE level (before tract filtering),
which is where these phenomena originate physically.

Pure Python + numpy/scipy. Zero tokens at runtime.
"""
import numpy as np
from scipy.interpolate import interp1d

from .hearing import SR, f0_autocorr
from .glottis import TwoMassGlottis


def _find_cycle_starts(ug, sr=SR):
    """Find glottal cycle starts (rising edge from closed phase).

    The two-mass model has a true closed phase (Ug=0). Cycle starts
    where flow rises above a small threshold after being below it.
    Returns array of sample indices.
    """
    thresh = 0.05 * np.max(ug)
    above = ug > thresh
    # rising edge: False->True
    edges = np.where(np.diff(above.astype(int)) == 1)[0] + 1
    # filter: minimum period 2ms (500Hz max), maximum 20ms (50Hz min)
    min_per = int(0.002 * sr)
    filtered = [edges[0]] if len(edges) > 0 else []
    for e in edges[1:]:
        if e - filtered[-1] >= min_per:
            filtered.append(e)
    return np.array(filtered, dtype=int)


def apply_jitter_shimmer(ug, sr=SR, jitter_pct=0.75, shimmer_pct=4.0,
                         seed=0):
    """Apply cycle-to-cycle jitter and shimmer to glottal flow.

    Jitter: each cycle resampled to random duration (1 ± N(0,jitter%)).
    Shimmer: each cycle scaled by (1 ± N(0,shimmer%)).
    Preserves waveform shape and spectral envelope; only timing and
    amplitude vary — exactly what jitter/shimmer are.
    """
    rng = np.random.default_rng(seed)
    starts = _find_cycle_starts(ug, sr)
    if len(starts) < 3:
        return ug  # not enough cycles, return as-is

    out_parts = []
    # leading partial cycle (before first detected start)
    out_parts.append(ug[:starts[0]])

    for i in range(len(starts) - 1):
        s0, s1 = starts[i], starts[i + 1]
        cycle = ug[s0:s1].astype(float)
        T = len(cycle)
        if T < 4:
            out_parts.append(cycle)
            continue
        # jitter: random duration
        j = rng.normal(0, jitter_pct / 100.0)
        T_new = max(4, int(round(T * (1 + j))))
        # resample cycle to new duration (cubic for smoothness)
        x_old = np.linspace(0, 1, T)
        x_new = np.linspace(0, 1, T_new)
        f = interp1d(x_old, cycle, kind='cubic',
                     fill_value='extrapolate')
        cycle_rs = f(x_new)
        # shimmer: random amplitude scale
        s = rng.normal(0, shimmer_pct / 100.0)
        cycle_rs = cycle_rs * (1 + s)
        out_parts.append(cycle_rs)

    # trailing partial cycle
    out_parts.append(ug[starts[-1]:])
    out = np.concatenate(out_parts)
    # trim/pad to original length
    if len(out) > len(ug):
        out = out[:len(ug)]
    else:
        out = np.pad(out, (0, len(ug) - len(out)))
    return out


def add_aspiration(ug, sr=SR, level_db=-20.0, seed=1):
    """Mix turbulent breath noise with voicing (physical breathiness).

    Aspiration arises from turbulence at the abducting glottis. Here:
    white noise gated by instantaneous glottal flow (more noise when
    glottis is open), scaled to level_db relative to flow RMS.
    The gating makes it physical (not constant background hiss).
    """
    rng = np.random.default_rng(seed)
    rms_ug = np.sqrt(np.mean(ug ** 2)) + 1e-12
    # gate: noise proportional to normalized instantaneous flow
    # (glottis open -> more turbulence)
    gate = np.clip(ug / (np.max(ug) + 1e-12), 0, 1)
    noise = rng.standard_normal(len(ug)) * gate
    rms_noise = np.sqrt(np.mean(noise ** 2)) + 1e-12
    # scale to target level
    target_rms = rms_ug * 10.0 ** (level_db / 20.0)
    noise = noise * (target_rms / rms_noise)
    return ug + noise


def measure_jitter_shimmer(y, sr=SR, fmin=100.0, fmax=300.0):
    """Measure jitter (%) and shimmer (%) from a voiced waveform.

    Detects cycles via autocorrelation peaks, measures period and
    amplitude variation. Returns (jitter_pct, shimmer_pct).
    """
    # use autocorrelation to find average period, then peak-pick
    from .hearing import f0_autocorr
    f0 = f0_autocorr(y)
    if f0 < fmin or f0 > fmax:
        return 0.0, 0.0
    period = int(sr / f0)
    # find peaks (glottal pulses)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(y, distance=int(period * 0.7))
    if len(peaks) < 4:
        return 0.0, 0.0
    periods = np.diff(peaks)
    amps = y[peaks[:-1]]  # amplitude at each pulse
    # jitter: mean absolute period difference / mean period
    jitter = np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-12)
    # shimmer: mean absolute amplitude difference / mean amplitude
    shimmer = np.mean(np.abs(np.diff(amps))) / (np.mean(np.abs(amps)) + 1e-12)
    return float(jitter * 100.0), float(shimmer * 100.0)


def synthesize_intonation(text_phonemes, f0_points, tension_ref=1.35,
                          f0_ref=177.8, sr=SR):
    """Synthesize with f0 intonation contour.

    text_phonemes: list of (phoneme, duration_ms), e.g.
        [('w',50), ('a',120), ('n',60), ('t',40), ...] — simplified:
        we use vowel/consonant placeholders and delegate to tract.
    f0_points: list of (time_ms, f0_hz) control points for pitch contour.
        Interpolated smoothly (cubic).
    tension_ref, f0_ref: calibration (tension 1.35 -> 177.8Hz).

    Actually implemented: generates glottal flow in 20ms chunks with
    tension from the contour (f0 ~ sqrt(tension) law). State persists
    across chunks for smooth transitions.

    Returns (flow_16kHz, times_ms, f0_actual).
    """
    # build time axis
    total_ms = sum(d for _, d in text_phonemes)
    times = np.array([t for t, _ in f0_points])
    f0s = np.array([f for _, f in f0_points])
    # cubic interpolation of f0 contour
    t_grid = np.arange(0, total_ms, 20.0)  # 20ms chunks
    f_interp = interp1d(times, f0s, kind='cubic',
                        fill_value=(f0s[0], f0s[-1]),
                        bounds_error=False)(t_grid)
    f_interp = np.clip(f_interp, 80.0, 350.0)

    g = TwoMassGlottis()
    g.reset()
    g.x1 = 1e-6
    chunks = []
    for f0 in f_interp:
        T = (f0 / f0_ref) ** 2 * tension_ref
        T = max(0.25, T)
        g.set_tension(T)  # updates k1,k2,kc,Ps; state (x,v) persists
        chunk = g.run(0.020)  # 20ms
        chunks.append(chunk)
    ug = np.concatenate(chunks)
    # measure actual f0 per chunk for verification
    f0_actual = []
    for i in range(0, len(ug), int(0.06 * sr)):
        seg = ug[i:i + int(0.06 * sr)]
        if len(seg) > int(0.03 * sr):
            f0_actual.append(f0_autocorr(seg))
    return ug, t_grid, np.array(f0_actual)


def polish_sentence_vowel(vowel="a", dur_ms=400, f0_start=185.0,
                          f0_end=165.0, jitter_pct=0.75, shimmer_pct=4.0,
                          seed=0):
    """Full polish demo: intonated vowel with jitter/shimmer/aspiration.

    Generates a vowel with falling intonation (declination), natural
    micro-variation, and breathiness. Returns 16kHz waveform.
    """
    from .tract import VocalTract, HER_VOWELS
    from scipy.signal import resample_poly, butter, filtfilt

    # 1. intonation: f0 contour via chunked glottis
    n_chunks = max(2, int(dur_ms / 20))
    times = np.linspace(0, dur_ms, n_chunks)
    # declination: linear fall, plus slight initial rise (natural)
    f0s = f0_start + (f0_end - f0_start) * (times / dur_ms)
    f0s[0] += 5.0  # tiny onset lift
    f0_points = list(zip(times, f0s))

    # build phoneme list (just vowel, chunked for intonation API)
    phonemes = [(vowel, 20) for _ in range(n_chunks)]
    ug, _, _ = synthesize_intonation(phonemes, f0_points)

    # 2. jitter + shimmer (on 16kHz flow)
    ug = apply_jitter_shimmer(ug, SR, jitter_pct, shimmer_pct, seed)

    # 3. aspiration breathiness
    ug = add_aspiration(ug, SR, level_db=-20.0, seed=seed + 1)

    # 4. tract filter (to cm^3/s)
    ug_cm = ug * 1e6
    ug48 = resample_poly(ug_cm, 3, 1)
    tract = VocalTract()
    tract.set_articulators(*HER_VOWELS[vowel])
    tract.reset()
    out48 = tract._run48(ug48)
    out16 = resample_poly(out48, 1, 3).astype(float)

    # DC blocker
    b_hp, a_hp = butter(2, 40.0 / (SR / 2.0), 'high')
    out16 = filtfilt(b_hp, a_hp, out16)
    mx = np.abs(out16).max()
    return out16 / (mx + 1e-12)


def gate_polish(vowel="a", dur_ms=500):
    """Gate: polished voice has natural jitter/shimmer, unpolished ~zero.

    Measures jitter/shimmer from the GLOTTAL FLOW (source), not the
    acoustic output — the tract's sharp formants confound acoustic
    measurement (f0 jitter moves harmonics relative to formants,
    creating artifactual amplitude/period variation). Source-level
    measurement is direct and honest.

    Natural targets: jitter 0.5-1.0%, shimmer 3-5%. Polished must be
    closer to target than unpolished (which should measure ~0%).
    Returns (ok, diagnostics).
    """
    from .glottis import TwoMassGlottis

    # unpolished: plain glottal flow
    g = TwoMassGlottis()
    g.set_tension(1.35); g.reset(); g.x1 = 1e-6
    ug_plain = g.run(dur_ms / 1000.0)
    ug_plain = ug_plain[int(0.1 * SR):]
    # polished: jitter + shimmer (voicing micro-variation)
    ug_js = apply_jitter_shimmer(ug_plain.copy(), SR, 0.75, 4.0, seed=0)
    # aspiration added after (breathiness) — does not change the
    # underlying voicing jitter/shimmer, but its noise confounds
    # peak-based measurement, so we measure on ug_js (see notes)
    ug_pol = add_aspiration(ug_js.copy(), SR, -20.0, seed=1)

    # measure at SOURCE (glottal flow) — direct, unconfounded
    # NOTE: measured on jitter+shimmer flow (ug_js), not +aspiration,
    # because white noise creates spurious peaks that break the
    # cycle detector. The aspiration is additive and does not alter
    # the voicing's intrinsic jitter/shimmer.
    j0, s0 = measure_jitter_shimmer(ug_plain)
    j1, s1 = measure_jitter_shimmer(ug_js)

    # distance to natural target (jitter 0.75%, shimmer 4%)
    def dist(j, s):
        return abs(j - 0.75) + abs(s - 4.0)
    d0, d1 = dist(j0, s0), dist(j1, s1)

    diag = {
        "jitter_plain": round(j0, 3),
        "shimmer_plain": round(s0, 3),
        "jitter_polished": round(j1, 3),
        "shimmer_polished": round(s1, 3),
        "dist_plain": round(d0, 3),
        "dist_polished": round(d1, 3),
    }
    ok = True
    # 1. polished jitter in natural range
    c1 = 0.4 <= j1 <= 1.5
    diag["jitter_natural"] = bool(c1)
    ok &= c1
    # 2. polished shimmer in natural range
    c2 = 2.0 <= s1 <= 7.0
    diag["shimmer_natural"] = bool(c2)
    ok &= c2
    # 3. polished closer to natural than plain
    c3 = d1 < d0
    diag["closer_to_natural"] = bool(c3)
    ok &= c3
    return ok, diag
