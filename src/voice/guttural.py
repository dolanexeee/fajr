"""Guttural inventory on the human tract (20 sec, 14.6cm) + Rosenberg source.

The honest next ceiling from the live bridge: ʕ ħ ʈˤ dˤ dʒ.

  /ʕ/  (E)  voiced pharyngeal fricative — narrow low-pharynx constriction
             (pos 0.15) with voicing running THROUGH it + injected
             turbulence. Signature: voiced + friction + pharyngeal color.
  /ħ/  (H)  voiceless pharyngeal fricative — same constriction, DC flow
             (no voicing) during frication, like /s/ but pharyngeal.
  /ʈˤ/ (T)  emphatic t — plain /t/ closure + pharyngealized vowel
             (retracted tongue root modeled as mild low-pharynx narrowing
             on the vowel's area function). Signature: F2 lowering.
  /dˤ/ (Dd) emphatic d — same, voiced.
  /dʒ/ (J)  voiced postalveolar affricate — stop closure at 0.70 +
             slow fricative release (d into ʃ).

Each unit returns (audio_16k, info) with exact sample boundaries.
Gates follow the established pattern: physical signature + vowel
recognized segment-wise against matched references.
"""
import numpy as np
from scipy.signal import resample_poly, butter, filtfilt

from .tract import VocalTract, _areas_from_articulators, SR_TRACT
from .rosenberg import rosenberg_train, SR
from .diva import auditory_feature
from .syllables import HUMAN_ART, HUMAN_N, _oral_pressure_feedback
from .fricatives import _frication_noise, _constriction_flow, _spectral_centroid
from .hearing import f0_autocorr

PHARYNX_POS = 0.15
PHARYNX_AREA = 0.15


def _voiced_flow(f0s, f0e, dur_s):
    return rosenberg_train(f0s, f0e, dur_s)[:int(dur_s * SR)] * 1e6


def _highpass(y):
    b_hp, a_hp = butter(2, 40.0 / (SR / 2.0), 'high')
    return filtfilt(b_hp, a_hp, y)


def _harmonicity(x):
    """Peak normalized autocorrelation in the f0 range (70-400 Hz)."""
    x = x - np.mean(x)
    n = len(x)
    if n < int(SR / 70.0) + 10:
        return 0.0
    ac = np.correlate(x, x, 'full')[n - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = int(SR / 400.0), int(SR / 70.0)
    return float(np.max(ac[lo:hi]))


def _vowel_f2(y_vowel):
    """F2 estimate from the CEPSTRAL spectral envelope (f0-robust).
    Raw spectra show harmonics (f0=185Hz), not formants — liftering
    below the pitch quefrency (like diva.auditory_feature) recovers
    the vocal-tract envelope, where true formant peaks live.
    Used only for the RELATIVE emphatic-vs-plain comparison."""
    from scipy.signal import find_peaks
    y = np.asarray(y_vowel[int(-0.25 * SR):], dtype=float)
    y = y[int(0.10 * SR):]
    if y.size < 2048:
        y = np.pad(y, (0, 2048 - y.size))
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size))) + 1e-9
    logspec = np.log(spec)
    n = 2 * (logspec.size - 1)
    ceps = np.fft.irfft(logspec, n=n)
    # lifter: keep quefrency < 2.5ms (envelope), kill pitch ripple
    ceps[40:] = 0.0
    env = np.fft.rfft(ceps, n=n).real
    fr = np.fft.rfftfreq(n, 1.0 / SR)
    band = (fr >= 900.0) & (fr <= 2800.0)
    fb, eb = fr[band], env[band]
    peaks, _ = find_peaks(eb, prominence=0.15, distance=8)
    if len(peaks) == 0:
        return None
    best = peaks[int(np.argmax(eb[peaks]))]
    return round(float(fb[best]), 1)


def _pharyngeal_areas(vowel, narrow_area=0.6):
    """Vowel area function + mild low-pharynx narrowing (retracted root)."""
    areas = _areas_from_articulators(*HUMAN_ART[vowel], n=HUMAN_N).copy()
    idx = int(PHARYNX_POS * (HUMAN_N - 1))
    for j in (idx - 1, idx, idx + 1):
        if 0 <= j < HUMAN_N:
            areas[j] = min(areas[j], narrow_area)
    return areas


# ------------------------------------------------------------------ #
# /ʕ/ — voiced pharyngeal fricative
# ------------------------------------------------------------------ #
def synthesize_ayn_v(vowel="a", ayn_ms=200, vowel_ms=350,
                     f0s=185, f0e=175, seed=0):
    ayn_s, vowel_s = ayn_ms / 1000.0, vowel_ms / 1000.0
    total = ayn_s + vowel_s + 0.1
    # voiced flow runs THROUGH the constriction (this is what makes it ʕ)
    ug48 = resample_poly(_voiced_flow(f0s, f0e, total), 3, 1)
    n_ayn48 = int(ayn_s * SR_TRACT)

    # turbulence level from Bernoulli flow at the constriction
    ug_dc = _constriction_flow(PHARYNX_AREA, 1080.0)
    noise48, Re, _ = _frication_noise(n_ayn48, ug_dc, PHARYNX_AREA, seed)

    t = VocalTract(n_sec=HUMAN_N)
    t.set_areas(_areas_from_articulators(*HUMAN_ART[vowel], n=HUMAN_N))
    idx = t.set_fricative(PHARYNX_POS, PHARYNX_AREA)
    assert idx is not None and idx >= 2, f"inject idx {idx} unsafe"
    t.reset()
    Zj = t.impedance_at(idx)
    L = min(n_ayn48, len(noise48))
    out_ayn = t._run48(ug48[:L], inject={idx: (noise48[:L], Zj)})

    t.set_areas(_areas_from_articulators(*HUMAN_ART[vowel], n=HUMAN_N))
    out_vow = t._run48(ug48[L:L + int((vowel_s + 0.05) * SR_TRACT)])

    out48 = np.concatenate([out_ayn, out_vow])
    y = resample_poly(out48, 1, 3).astype(float)
    y = _highpass(y)
    y = y / (np.abs(y).max() + 1e-12)
    n_ayn16 = int(ayn_s * SR)
    return y, {"ayn_bounds": (0, n_ayn16),
               "vowel_bounds": [(vowel, n_ayn16, len(y))], "Re": Re,
               "idx": idx}


def gate_ayn(vowel="a", vowel_refs=None):
    y, info = synthesize_ayn_v(vowel)
    n_ayn = info["ayn_bounds"][1]
    ayn = y[:n_ayn]
    seg = ayn[int(0.05 * SR):int(0.16 * SR)]
    # 1) turbulent
    c1 = info["Re"] > 1800.0
    # 2) VOICED through the friction (this separates ʕ from ħ)
    harm = _harmonicity(seg)
    f0 = f0_autocorr(seg)
    c2 = harm > 0.5 and f0 > 60.0
    # 3) friction audible: noise energy vs vowel
    vseg = y[info["vowel_bounds"][0][1]:]
    vseg = vseg[int(-0.2 * SR):]
    e_ayn = float(np.mean(seg ** 2)) + 1e-12
    e_vow = float(np.mean(vseg ** 2)) + 1e-12
    c3 = e_ayn > 0.15 * e_vow
    diag = {"Re": round(float(info["Re"]), 1), "turbulent": bool(c1),
            "harmonicity": round(harm, 3), "f0_hz": round(float(f0), 1),
            "voiced": bool(c2), "friction_audible": bool(c3)}
    ok = bool(c1 and c2 and c3)
    if vowel_refs is not None:
        f = auditory_feature(vseg / (np.abs(vseg).max() + 1e-12))
        dists = {v: float(np.linalg.norm(f - vowel_refs[v])) for v in vowel_refs}
        recog = min(dists, key=dists.get)
        diag["vowel_recognized"] = recog
        c4 = (recog == vowel)
        diag["vowel_ok"] = bool(c4)
        ok = ok and c4
    return ok, diag


# ------------------------------------------------------------------ #
# /ħ/ — voiceless pharyngeal fricative
# ------------------------------------------------------------------ #
def synthesize_hah_v(vowel="a", hah_ms=200, vowel_ms=350,
                     f0s=185, f0e=175, seed=0):
    hah_s, vowel_s = hah_ms / 1000.0, vowel_ms / 1000.0
    ug_dc = _constriction_flow(PHARYNX_AREA, 1080.0)
    n_hah48 = int(hah_s * SR_TRACT)
    ug_hah48 = np.full(n_hah48, ug_dc)
    noise48, Re, _ = _frication_noise(n_hah48, ug_dc, PHARYNX_AREA, seed)
    # pharyngeal fricatives concentrate energy low: the wide pharynx
    # cannot sustain the small fast eddies of /s/ — low-pass the source
    # (Strouhal scaling, same principle as /ʃ/ but lower).
    b_lp, a_lp = butter(2, 2800.0 / (SR_TRACT / 2.0), 'low')
    noise48 = filtfilt(b_lp, a_lp, noise48)

    t = VocalTract(n_sec=HUMAN_N)
    t.set_areas(_areas_from_articulators(*HUMAN_ART[vowel], n=HUMAN_N))
    idx = t.set_fricative(PHARYNX_POS, PHARYNX_AREA)
    assert idx is not None and idx >= 2, f"inject idx {idx} unsafe"
    t.reset()
    Zj = t.impedance_at(idx)
    L = min(len(ug_hah48), len(noise48))
    out_hah = t._run48(ug_hah48[:L], inject={idx: (noise48[:L], Zj)})

    ug48_v = resample_poly(_voiced_flow(f0s, f0e, vowel_s + 0.1), 3, 1)
    t.set_areas(_areas_from_articulators(*HUMAN_ART[vowel], n=HUMAN_N))
    out_vow = t._run48(ug48_v[:int((vowel_s + 0.05) * SR_TRACT)])

    out48 = np.concatenate([out_hah, out_vow])
    y = resample_poly(out48, 1, 3).astype(float)
    y = _highpass(y)
    y = y / (np.abs(y).max() + 1e-12)
    n_hah16 = int(hah_s * SR)
    return y, {"hah_bounds": (0, n_hah16),
               "vowel_bounds": [(vowel, n_hah16, len(y))], "Re": Re}


def gate_hah(vowel="a", vowel_refs=None):
    y, info = synthesize_hah_v(vowel)
    n_hah = info["hah_bounds"][1]
    hah = y[:n_hah]
    seg = hah[int(0.05 * SR):int(0.16 * SR)]
    seg = seg - np.mean(seg)
    # 1) turbulent
    c1 = info["Re"] > 1800.0
    # 2) APERIODIC (this separates ħ from ʕ)
    harm = _harmonicity(seg)
    c2 = harm < 0.4
    # 3) pharyngeal color: centroid lower than /s/ (alveolar)
    centroid = _spectral_centroid(seg)
    c3 = centroid < 3500.0
    diag = {"Re": round(float(info["Re"]), 1), "turbulent": bool(c1),
            "harmonicity": round(harm, 3), "aperiodic": bool(c2),
            "centroid_hz": round(float(centroid), 1),
            "pharyngeal_band": bool(c3)}
    ok = bool(c1 and c2 and c3)
    if vowel_refs is not None:
        vseg = y[info["vowel_bounds"][0][1]:]
        vseg = vseg[int(-0.2 * SR):]
        f = auditory_feature(vseg / (np.abs(vseg).max() + 1e-12))
        dists = {v: float(np.linalg.norm(f - vowel_refs[v])) for v in vowel_refs}
        recog = min(dists, key=dists.get)
        diag["vowel_recognized"] = recog
        c4 = (recog == vowel)
        diag["vowel_ok"] = bool(c4)
        ok = ok and c4
    return ok, diag


# ------------------------------------------------------------------ #
# /ʈˤ/ /dˤ/ — emphatic stops (plain closure + pharyngealized vowel)
# ------------------------------------------------------------------ #
def synthesize_emphatic_cv(plosive="t", vowel="a", hold_ms=70,
                           vowel_ms=350, f0s=185, f0e=172):
    """Emphatic = plain stop burst + vowel with retracted tongue root.
    The pharyngealization lives on the VOWEL (that is where Arabic
    emphasis is acoustically realized: lowered F2)."""
    from .syllables import CLOSURE_POS
    plain = "t" if plosive == "t" else "d"
    pos = CLOSURE_POS[plain]
    voiceless = (plosive == "t")
    hold_s, vowel_s = hold_ms / 1000.0, vowel_ms / 1000.0
    total = hold_s + vowel_s + 0.15

    if voiceless:
        flow_hold = rosenberg_train(f0s, f0e, total) * 0.0 + 1.2e-4
        flow_hold = flow_hold * 1e6
        flow_vow = rosenberg_train(f0s, f0e, total) * 1e6
    else:
        flow_hold = rosenberg_train(f0s, f0e, total) * 1e6
        flow_vow = flow_hold
    ug48_hold_full = resample_poly(flow_hold, 3, 1)
    ug48_vow_full = resample_poly(flow_vow, 3, 1)
    n_hold48 = int(hold_s * SR_TRACT)
    ug48_hold = _oral_pressure_feedback(ug48_hold_full[:n_hold48])

    t = VocalTract(n_sec=HUMAN_N)
    # closure phase: plain vowel shape (burst is the plain burst)
    t.set_areas(_areas_from_articulators(*HUMAN_ART[vowel], n=HUMAN_N))
    t.reset()
    t.set_closure(pos, True)
    out_hold_parts = []
    chunk48 = int(0.01 * SR_TRACT)
    for i in range(0, n_hold48, chunk48):
        out_hold_parts.append(t._run48(ug48_hold[i:i + chunk48]))
        t.up *= 0.85
        t.dn *= 0.85
    out_hold = np.concatenate(out_hold_parts) if out_hold_parts else np.zeros(0)

    # vowel phase: PHARYNGEALIZED areas
    t.set_closure(pos, False)
    t.set_areas(_pharyngeal_areas(vowel))
    need48 = n_hold48 + int((vowel_s + 0.1) * SR_TRACT)
    out_rest = t._run48(ug48_vow_full[n_hold48:need48])

    out48 = np.concatenate([out_hold, out_rest])
    if voiceless:
        n_vot = int(0.05 * SR_TRACT)
        rng = np.random.default_rng(3)
        noise = np.diff(rng.standard_normal(n_vot), prepend=0) * 0.15
        out48[n_hold48:n_hold48 + n_vot] += noise * np.abs(out48).max()
    y = resample_poly(out48, 1, 3).astype(float)
    y = y / (np.abs(y).max() + 1e-12)
    n_hold16 = int(hold_s * SR)
    return y, {"closure_bounds": (0, n_hold16),
               "vowel_bounds": [(vowel, n_hold16, len(y))]}


def gate_emphatic(plosive="t", vowel="a", vowel_refs=None):
    from .syllables import synthesize_cv
    y, info = synthesize_emphatic_cv(plosive, vowel)
    n_hold = info["closure_bounds"][1]
    vseg = y[n_hold:]
    vseg = vseg[int(-0.25 * SR):]
    # plain counterpart for the F2 comparison
    plain = "t" if plosive == "t" else "d"
    yp = synthesize_cv(plain, vowel, hold_ms=70, vowel_ms=350)
    # find vowel start in plain (after hold)
    n_hold_p = int(0.07 * SR)
    pvseg = yp[n_hold_p:]
    pvseg = pvseg[int(-0.25 * SR):]
    f2_e = _vowel_f2(vseg)
    f2_p = _vowel_f2(pvseg)
    c1 = f2_e is not None and f2_p is not None and (f2_p - f2_e) > 80.0
    diag = {"F2_emphatic": f2_e, "F2_plain": f2_p,
            "F2_drop_hz": round(f2_p - f2_e, 1) if (f2_e and f2_p) else None,
            "emphasis": bool(c1)}
    ok = bool(c1)
    if vowel_refs is not None:
        f = auditory_feature(vseg / (np.abs(vseg).max() + 1e-12))
        dists = {v: float(np.linalg.norm(f - vowel_refs[v])) for v in vowel_refs}
        recog = min(dists, key=dists.get)
        diag["vowel_recognized"] = recog
        c2 = (recog == vowel)
        diag["vowel_ok"] = bool(c2)
        ok = ok and c2
    return ok, diag


# ------------------------------------------------------------------ #
# /dʒ/ — voiced postalveolar affricate (d into ʃ)
# ------------------------------------------------------------------ #
def synthesize_djim_v(vowel="a", hold_ms=60, fric_ms=100, vowel_ms=350,
                      f0s=185, f0e=172, seed=0):
    pos = 0.70  # postalveolar
    hold_s, fric_s, vowel_s = (hold_ms / 1000.0, fric_ms / 1000.0,
                               vowel_ms / 1000.0)
    total = hold_s + fric_s + vowel_s + 0.15
    flow = _voiced_flow(f0s, f0e, total)
    ug48_full = resample_poly(flow, 3, 1)
    n_hold48 = int(hold_s * SR_TRACT)
    n_fric48 = int(fric_s * SR_TRACT)

    # frication noise at the release constriction (ʃ-like, low-passed)
    area_c = 0.15
    ug_dc = _constriction_flow(area_c, 1080.0)
    noise48, Re, _ = _frication_noise(n_fric48, ug_dc, area_c, seed)
    b_lp, a_lp = butter(2, 4500.0 / (SR_TRACT / 2.0), 'low')
    noise48 = filtfilt(b_lp, a_lp, noise48)

    t = VocalTract(n_sec=HUMAN_N)
    t.set_areas(_areas_from_articulators(*HUMAN_ART[vowel], n=HUMAN_N))
    t.reset()
    # phase 1: voiced closure (with oral-pressure feedback)
    ug48_hold = _oral_pressure_feedback(ug48_full[:n_hold48])
    t.set_closure(pos, True)
    out_hold_parts = []
    chunk48 = int(0.01 * SR_TRACT)
    for i in range(0, n_hold48, chunk48):
        out_hold_parts.append(t._run48(ug48_hold[i:i + chunk48]))
        t.up *= 0.85
        t.dn *= 0.85
    out_hold = np.concatenate(out_hold_parts) if out_hold_parts else np.zeros(0)

    # phase 2: slow release into frication (affricate, not plosive burst)
    # narrow but not sealed; voiced flow continues + injected noise
    t.set_closure(pos, False)
    a = t._base_areas.copy()
    idx = int(np.clip(pos, 0, 1) * (HUMAN_N - 1))
    for j in (idx - 1, idx, idx + 1):
        if 0 <= j < HUMAN_N:
            a[j] = min(a[j], area_c)
    t._apply_areas(a)
    Zj = t.impedance_at(idx)
    ug48_fric = ug48_full[n_hold48:n_hold48 + n_fric48]
    L = min(len(ug48_fric), len(noise48))
    out_fric = t._run48(ug48_fric[:L], inject={idx: (noise48[:L], Zj)})

    # phase 3: vowel
    t.set_areas(_areas_from_articulators(*HUMAN_ART[vowel], n=HUMAN_N))
    start_v = n_hold48 + n_fric48
    out_vow = t._run48(ug48_full[start_v:start_v + int((vowel_s + 0.05) * SR_TRACT)])

    out48 = np.concatenate([out_hold, out_fric, out_vow])
    y = resample_poly(out48, 1, 3).astype(float)
    y = _highpass(y)
    y = y / (np.abs(y).max() + 1e-12)
    n_hold16 = int(hold_s * SR)
    n_fric16 = int(fric_s * SR)
    return y, {"closure_bounds": (0, n_hold16),
               "fric_bounds": (n_hold16, n_hold16 + n_fric16),
               "vowel_bounds": [(vowel, n_hold16 + n_fric16, len(y))],
               "Re": Re}


def gate_djim(vowel="a", vowel_refs=None):
    y, info = synthesize_djim_v(vowel)
    a, b = info["fric_bounds"]
    fric = y[a:b]
    seg = fric[int(0.02 * SR):]
    # 1) turbulent release
    c1 = info["Re"] > 1800.0
    # 2) postalveolar frication: isolate it from the voicing that runs
    #    through the affricate (voiced by definition) — measure the
    #    spectrum ABOVE the voice harmonics.
    from scipy.signal import butter, filtfilt
    b_hp2, a_hp2 = butter(2, 1500.0 / (SR / 2.0), 'high')
    seg_hf = filtfilt(b_hp2, a_hp2, seg)
    centroid = _spectral_centroid(seg_hf)
    c2 = 1800.0 <= centroid <= 4800.0
    # 3) voiced through it (affricate is voiced)
    harm = _harmonicity(seg)
    c3 = harm > 0.4
    diag = {"Re": round(float(info["Re"]), 1), "turbulent": bool(c1),
            "centroid_hf_hz": round(float(centroid), 1), "sh_band": bool(c2),
            "harmonicity": round(harm, 3), "voiced_release": bool(c3)}
    ok = bool(c1 and c2 and c3)
    if vowel_refs is not None:
        vseg = y[info["vowel_bounds"][0][1]:info["vowel_bounds"][0][2]]
        vseg = vseg[int(-0.2 * SR):]
        f = auditory_feature(vseg / (np.abs(vseg).max() + 1e-12))
        dists = {v: float(np.linalg.norm(f - vowel_refs[v])) for v in vowel_refs}
        recog = min(dists, key=dists.get)
        diag["vowel_recognized"] = recog
        c4 = (recog == vowel)
        diag["vowel_ok"] = bool(c4)
        ok = ok and c4
    return ok, diag


if __name__ == "__main__":
    from .arabic_units import unit_vowel
    refs = {}
    for v in ("a", "i", "u"):
        yr, _ = unit_vowel(v, dur_ms=300)
        refs[v] = auditory_feature(yr / (np.abs(yr).max() + 1e-12))
    for name, fn in [("ayn", gate_ayn), ("hah", gate_hah),
                     ("emph-t", lambda vowel_refs: gate_emphatic("t", vowel_refs=vowel_refs)),
                     ("emph-d", lambda vowel_refs: gate_emphatic("d", vowel_refs=vowel_refs)),
                     ("djim", gate_djim)]:
        try:
            ok, d = fn(vowel_refs=refs)
        except TypeError:
            ok, d = fn(refs)
        print(name, "PASS" if ok else "FAIL", d)
