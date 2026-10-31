"""Arabic phoneme units on the human tract (20 sec, 14.6cm) + Rosenberg source.

Extends the proven English inventory (b,d,g,p,t,k,s,f,m,n + a,i,u)
with the Arabic sounds needed for Fajr's real utterances:

  /l/  alveolar lateral approximant (voiced, narrowed not closed)
  /h/  glottal fricative (turbulence at glottis, vowel-shaped tract)
  /ʔ/  glottal stop (brief silence)
  /r/  alveolar tap (single brief voiced closure — legitimate allophone)
  /ʃ/  postalveolar fricative (constriction further back than /s/)
  /w/  glide from /u/          /j/  glide from /i/

Each unit returns (audio_16k, info) with exact sample boundaries for gating.
Honest approximations are documented per-unit; what is NOT attempted
(ʕ ħ q emphatics θ ð dʒ x ɣ z) is listed in live_bridge.py as the ceiling.
"""
import numpy as np
from scipy.signal import resample_poly

from .tract import VocalTract, _areas_from_articulators, SR_TRACT
from .rosenberg import rosenberg_train, SR
from .diva import auditory_feature
from .syllables import HUMAN_ART, HUMAN_N
from .fricatives import _frication_noise, _constriction_flow, _spectral_centroid
from .hearing import f0_autocorr

VOWELS = ("a", "i", "u")


def _voiced_flow(f0s, f0e, dur_s):
    return rosenberg_train(f0s, f0e, dur_s)[:int(dur_s * SR)] * 1e6


def unit_vowel(vowel="a", dur_ms=280, f0s=185, f0e=175):
    """Pure vowel. Returns (audio, {'vowel_bounds': [(0, n)]})"""
    dur_s = dur_ms / 1000.0
    ug48 = resample_poly(_voiced_flow(f0s, f0e, dur_s + 0.05), 3, 1)
    t = VocalTract(n_sec=HUMAN_N)
    t.set_areas(_areas_from_articulators(*HUMAN_ART[vowel], n=HUMAN_N))
    t.reset()
    out48 = t._run48(ug48[:int(dur_s * SR_TRACT)])
    y = resample_poly(out48, 1, 3).astype(float)
    y = y / (np.abs(y).max() + 1e-12)
    return y, {"vowel_bounds": [(vowel, 0, len(y))]}


def synthesize_lateral_v(vowel="a", lat_ms=180, vowel_ms=280,
                         f0s=185, f0e=175):
    """ /l/: voiced flow through tract narrowed (not closed) at alveolar
    ridge. A 1-D waveguide cannot model lateral airflow; acoustically this
    is a voiced approximant with an alveolar constriction — the honest
    closest model. Gate: voiced, no burst, sustained energy, vowel after."""
    lat_s, vowel_s = lat_ms / 1000.0, vowel_ms / 1000.0
    total = lat_s + vowel_s + 0.05
    ug48 = resample_poly(_voiced_flow(f0s, f0e, total), 3, 1)
    n_lat48 = int(lat_s * SR_TRACT)

    t = VocalTract(n_sec=HUMAN_N)
    areas = _areas_from_articulators(*HUMAN_ART[vowel], n=HUMAN_N).copy()
    idx = int(0.85 * HUMAN_N)          # alveolar ridge
    areas[max(0, idx - 1):idx + 2] = 0.45   # narrowed, NOT closed
    t.set_areas(areas)
    t.reset()
    out_lat = t._run48(ug48[:n_lat48])

    t.set_areas(_areas_from_articulators(*HUMAN_ART[vowel], n=HUMAN_N))
    out_vow = t._run48(ug48[n_lat48:n_lat48 + int((vowel_s + 0.05) * SR_TRACT)])

    out48 = np.concatenate([out_lat, out_vow])
    y = resample_poly(out48, 1, 3).astype(float)
    y = y / (np.abs(y).max() + 1e-12)
    n_lat16 = int(lat_s * SR)
    return y, {"lat_bounds": (0, n_lat16),
               "vowel_bounds": [(vowel, n_lat16, len(y))]}


def gate_lateral(vowel="a", vowel_refs=None):
    y, info = synthesize_lateral_v(vowel)
    n_lat = info["lat_bounds"][1]
    lat = y[:n_lat]
    # voiced?
    f0 = f0_autocorr(lat)
    c1 = f0 > 60.0
    # no burst: onset transient small relative to steady lateral
    onset = np.abs(lat[:int(0.02 * SR)]).max()
    steady = np.abs(lat[int(0.05 * SR):]).max()
    c2 = onset < 3.0 * (steady + 1e-9)
    # sustained energy (approximant, not stop)
    h1 = np.mean(lat[int(0.03 * SR):int(0.09 * SR)] ** 2)
    h2 = np.mean(lat[int(0.09 * SR):int(0.15 * SR)] ** 2)
    c3 = (h2 + 1e-12) / (h1 + 1e-12) > 0.4
    diag = {"f0_hz": round(float(f0), 1), "voiced": bool(c1),
            "no_burst": bool(c2), "sustained": bool(c3)}
    ok = bool(c1 and c2 and c3)
    if vowel_refs is not None:
        vseg = y[info["vowel_bounds"][0][1]:info["vowel_bounds"][0][2]]
        vseg = vseg[int(-0.15 * SR):]
        f = auditory_feature(vseg / (np.abs(vseg).max() + 1e-12))
        dists = {v: float(np.linalg.norm(f - vowel_refs[v])) for v in vowel_refs}
        recog = min(dists, key=dists.get)
        diag["vowel_recognized"] = recog
        c4 = (recog == vowel)
        diag["vowel_ok"] = bool(c4)
        ok = ok and c4
    return ok, diag


def synthesize_h_v(vowel="a", h_ms=150, vowel_ms=280, f0s=185, f0e=175,
                   seed=0):
    """ /h/: glottal turbulence (no oral constriction) through the
    following vowel's tract shape. Noise injected at the glottis.
    Gate: aperiodic + vowel-like formants in the noise + vowel after."""
    h_s, vowel_s = h_ms / 1000.0, vowel_ms / 1000.0
    n_h48 = int(h_s * SR_TRACT)
    # breathy glottal flow: DC component larger than modal voicing
    ug_dc = _constriction_flow(0.5, 1080.0)
    ug_h48 = np.full(n_h48, ug_dc)
    noise48, Re, _ = _frication_noise(n_h48, ug_dc, 0.5, seed)

    t = VocalTract(n_sec=HUMAN_N)
    t.set_areas(_areas_from_articulators(*HUMAN_ART[vowel], n=HUMAN_N))
    t.reset()
    # NOTE: inject at idx 2, not 0 — the waveguide never propagates up[0]
    # (injection at the glottis junction is silently lost). Two sections
    # above the glottis is still the glottal region: glottal turbulence
    # exciting the vowel-shaped tract.
    Z2 = t.impedance_at(2)
    out_h = t._run48(ug_h48, inject={2: (noise48, Z2)})

    ug48_v = resample_poly(_voiced_flow(f0s, f0e, vowel_s + 0.1), 3, 1)
    t.set_areas(_areas_from_articulators(*HUMAN_ART[vowel], n=HUMAN_N))
    out_vow = t._run48(ug48_v[:int((vowel_s + 0.05) * SR_TRACT)])

    out48 = np.concatenate([out_h, out_vow])
    y = resample_poly(out48, 1, 3).astype(float)
    # the steady glottal flow leaves a DC offset (unphysical as sound:
    # steady flow makes no acoustic signal) — high-pass like fricatives_human
    from scipy.signal import butter, filtfilt
    b_hp, a_hp = butter(2, 40.0 / (SR / 2.0), 'high')
    y = filtfilt(b_hp, a_hp, y)
    y = y / (np.abs(y).max() + 1e-12)
    n_h16 = int(h_s * SR)
    return y, {"h_bounds": (0, n_h16),
               "vowel_bounds": [(vowel, n_h16, len(y))], "Re": Re}


def gate_h(vowel="a", vowel_refs=None):
    y, info = synthesize_h_v(vowel)
    n_h = info["h_bounds"][1]
    h = y[:n_h]
    # steady middle of the /h/ (avoid onset and the vowel-onset transition)
    a = int(0.04 * SR); b = min(int(0.12 * SR), n_h)
    seg = h[a:b]
    seg = seg - np.mean(seg)
    # aperiodic: no harmonic structure (autocorrelation peak low).
    # voiced vowels score ~0.97; turbulence ~0.14. Threshold 0.4 is clean.
    n = len(seg)
    ac = np.correlate(seg, seg, 'full')[n - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = int(SR / 400.0), int(SR / 70.0)
    harm = float(np.max(ac[lo:hi]))
    c1 = harm < 0.4
    # vowel-like formants: spectral peak in 500-2500 Hz
    X = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    fr = np.fft.rfftfreq(len(seg), 1.0 / SR)
    band = (fr >= 500) & (fr <= 2500)
    c2 = float(X[band].max()) > 3.0 * float(np.median(X[band] + 1e-12))
    diag = {"harmonicity": round(harm, 3), "aperiodic": bool(c1),
            "formants_present": bool(c2), "Re": round(float(info["Re"]), 1)}
    ok = bool(c1 and c2)
    if vowel_refs is not None:
        vseg = y[info["vowel_bounds"][0][1]:info["vowel_bounds"][0][2]]
        vseg = vseg[int(-0.15 * SR):]
        f = auditory_feature(vseg / (np.abs(vseg).max() + 1e-12))
        dists = {v: float(np.linalg.norm(f - vowel_refs[v])) for v in vowel_refs}
        recog = min(dists, key=dists.get)
        diag["vowel_recognized"] = recog
        c3 = (recog == vowel)
        diag["vowel_ok"] = bool(c3)
        ok = ok and c3
    return ok, diag


def glottal_stop(ms=45):
    """ /ʔ/: brief silence with raised-cosine edge fades (no clicks)."""
    n = int(ms / 1000.0 * SR)
    y = np.zeros(n)
    nf = min(int(0.005 * SR), n // 4)
    fade = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, nf))
    # silence is silence; fades only matter against neighbours in assembly
    return y, {"silence_bounds": (0, n)}


def synthesize_tap_v(vowel="a", tap_ms=25, vowel_ms=280, f0s=185, f0e=175):
    """ /r/ as alveolar tap: one very brief voiced closure. In fast
    speech this is the standard allophone of /r/ between vowels.
    Uses the same closure physics as synthesize_cv (oral-pressure
    feedback chokes the flow; state damped during the seal).
    Gate: dip 10-50ms, voicing around it, vowel recognized after."""
    from .syllables import _oral_pressure_feedback
    tap_s, vowel_s = tap_ms / 1000.0, vowel_ms / 1000.0
    total = 0.06 + tap_s + vowel_s + 0.05   # 60ms lead-in voicing
    ug48 = resample_poly(_voiced_flow(f0s, f0e, total), 3, 1)

    t = VocalTract(n_sec=HUMAN_N)
    t.set_areas(_areas_from_articulators(*HUMAN_ART[vowel], n=HUMAN_N))
    t.reset()
    n_lead48 = int(0.06 * SR_TRACT)
    out_lead = t._run48(ug48[:n_lead48])
    # closure: pressure feedback + damping (proven physics)
    n_tap48 = int(tap_s * SR_TRACT)
    ug_tap = _oral_pressure_feedback(ug48[n_lead48:n_lead48 + n_tap48])
    t.set_closure(0.85, True)
    out_tap_parts = []
    chunk48 = int(0.01 * SR_TRACT)
    for i in range(0, n_tap48, chunk48):
        out_tap_parts.append(t._run48(ug_tap[i:i + chunk48]))
        t.up *= 0.85
        t.dn *= 0.85
    out_tap = np.concatenate(out_tap_parts) if out_tap_parts else np.zeros(0)
    t.set_closure(0.85, False)
    out_vow = t._run48(ug48[n_lead48 + n_tap48:
                            n_lead48 + n_tap48 + int((vowel_s + 0.05) * SR_TRACT)])
    out48 = np.concatenate([out_lead, out_tap, out_vow])
    y = resample_poly(out48, 1, 3).astype(float)
    y = y / (np.abs(y).max() + 1e-12)
    n_lead16 = int(0.06 * SR)
    n_tap16 = int(tap_s * SR)
    vstart = n_lead16 + n_tap16
    return y, {"tap_bounds": (n_lead16, vstart),
               "vowel_bounds": [(vowel, vstart, len(y))]}


def gate_tap(vowel="a", vowel_refs=None):
    y, info = synthesize_tap_v(vowel)
    a, b = info["tap_bounds"]
    tap = y[a:b]
    dip_ms = (b - a) / SR * 1000.0
    c1 = 10.0 <= dip_ms <= 60.0
    # energy dips but voicing continues around it
    e_tap = float(np.mean(tap ** 2)) + 1e-12
    e_pre = float(np.mean(y[a - int(0.03 * SR):a] ** 2)) + 1e-12
    c2 = e_tap < 0.5 * e_pre
    diag = {"dip_ms": round(dip_ms, 1), "brief": bool(c1), "dips": bool(c2)}
    ok = bool(c1 and c2)
    if vowel_refs is not None:
        vseg = y[info["vowel_bounds"][0][1]:info["vowel_bounds"][0][2]]
        vseg = vseg[int(-0.15 * SR):]
        f = auditory_feature(vseg / (np.abs(vseg).max() + 1e-12))
        dists = {v: float(np.linalg.norm(f - vowel_refs[v])) for v in vowel_refs}
        recog = min(dists, key=dists.get)
        diag["vowel_recognized"] = recog
        c3 = (recog == vowel)
        diag["vowel_ok"] = bool(c3)
        ok = ok and c3
    return ok, diag


def synthesize_sh_v(vowel="a", fric_ms=200, vowel_ms=280, f0s=185, f0e=175,
                    seed=0):
    """ /ʃ/: postalveolar fricative — constriction further back and
    wider than /s/. Physics: a wider constriction means lower jet
    velocity, hence larger turbulent eddies and less high-frequency
    energy — so the injected noise is low-passed (Strouhal scaling),
    not white. Same proven waveguide machinery otherwise."""
    from scipy.signal import butter, filtfilt
    pos, area_c = 0.65, 0.15
    fric_s, vowel_s = fric_ms / 1000.0, vowel_ms / 1000.0
    Psub_Pa = 800.0 * 1.35
    ug_dc = _constriction_flow(area_c, Psub_Pa)
    n_fric48 = int(fric_s * SR_TRACT)
    ug_fric48 = np.full(n_fric48, ug_dc)
    noise48, Re, _ = _frication_noise(n_fric48, ug_dc, area_c, seed)
    # wider constriction -> lower eddy frequency: low-pass the source
    b_lp, a_lp = butter(2, 4500.0 / (SR_TRACT / 2.0), 'low')
    noise48 = filtfilt(b_lp, a_lp, noise48)

    flow_v = _voiced_flow(f0s, f0e, vowel_s + 0.1)
    t = VocalTract(n_sec=HUMAN_N)
    t.set_areas(_areas_from_articulators(*HUMAN_ART[vowel], n=HUMAN_N))
    idx = t.set_fricative(pos, area_c)
    t.reset()
    Zj = t.impedance_at(idx)
    L = min(len(ug_fric48), len(noise48))
    out_fric = t._run48(ug_fric48[:L], inject={idx: (noise48[:L], Zj)})

    t.set_areas(_areas_from_articulators(*HUMAN_ART[vowel], n=HUMAN_N))
    ug48_v = resample_poly(flow_v, 3, 1)
    out_vowel = t._run48(ug48_v[:int((vowel_s + 0.05) * SR_TRACT)])

    out48 = np.concatenate([out_fric, out_vowel])
    out16 = resample_poly(out48, 1, 3).astype(float)
    b_hp, a_hp = butter(2, 40.0 / (SR / 2.0), 'high')
    out16 = filtfilt(b_hp, a_hp, out16)
    mx = np.abs(out16).max()
    y = out16 / (mx + 1e-12)
    n_fric16 = int(fric_s * SR)
    return y, {"fric_bounds": (0, n_fric16),
               "vowel_bounds": [(vowel, n_fric16, len(y))],
               "info": {"Re": Re}}


def gate_sh(vowel="a", vowel_refs=None):
    y, info = synthesize_sh_v(vowel)
    n_fric = info["fric_bounds"][1]
    fric = y[:n_fric]
    steady = fric[int(0.03 * SR):]
    centroid = _spectral_centroid(fric)
    Re = info["info"]["Re"]
    c1 = Re > 1800.0
    c2 = 1800.0 <= centroid <= 4800.0   # lower than /s/ (>3500), distinct band
    diag = {"Re": round(float(Re), 1), "centroid_hz": round(float(centroid), 1),
            "turbulent": bool(c1), "sh_band": bool(c2)}
    ok = bool(c1 and c2)
    if vowel_refs is not None:
        vseg = y[info["vowel_bounds"][0][1]:info["vowel_bounds"][0][2]]
        vseg = vseg[int(-0.15 * SR):]
        f = auditory_feature(vseg / (np.abs(vseg).max() + 1e-12))
        dists = {v: float(np.linalg.norm(f - vowel_refs[v])) for v in vowel_refs}
        recog = min(dists, key=dists.get)
        diag["vowel_recognized"] = recog
        c3 = (recog == vowel)
        diag["vowel_ok"] = bool(c3)
        ok = ok and c3
    return ok, diag


def glide_sound(glide="w", dur_ms=70, f0s=185, f0e=180):
    """ /w/ = brief /u/, /j/ = brief /i/ — the glide onset. The bridge
    crossfades this into the following vowel (formant transition)."""
    v = "u" if glide == "w" else "i"
    y, _ = unit_vowel(v, dur_ms=dur_ms, f0s=f0s, f0e=f0e)
    return y, {"glide": glide}


def unit_final_stop(plosive="t", hold_ms=70, release_ms=70, f0s=180, f0e=172):
    """Word-final stop: closure + burst + short weak release."""
    from .syllables import synthesize_cv
    y = synthesize_cv(plosive, "a", hold_ms=hold_ms,
                      vowel_ms=release_ms, f0s=f0s, f0e=f0e)
    want = int((hold_ms + release_ms + 60) / 1000.0 * SR)
    y = y[:want]
    return y, {"final": plosive}
