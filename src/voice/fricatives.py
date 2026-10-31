"""
Fricatives /s,f/ — sustained turbulence noise at constriction.
Uses interior injection in the waveguide (see tract._run48 inject).
"""
import numpy as np
from scipy.signal import resample_poly

from .tract import VocalTract, HER_VOWELS, SR_TRACT
from .consonants import _steady_glottal_flow, _spectral_centroid
from .hearing import SR
from .diva import auditory_feature

# ------------------------------------------------------- fricatives /s,f/
FRICATIVE_CONFIG = {
    # pos: constriction location (0=glottis, 1=lips)
    # area: constriction area in cm^2 (narrow, NOT closed)
    "s": {"pos": 0.85, "area": 0.08},   # alveolar, narrow -> high-freq
    "f": {"pos": 0.95, "area": 0.15},   # labiodental, wider -> diffuse
}


def _frication_noise(n48, ug_dc, area_c, seed=0):
    """Reynolds-based turbulence noise at the constriction.

    Aeroacoustics: Re = v*d/nu with v=Ug/Ac at the constriction.
    Above Re_c=1800, the jet is turbulent. Turbulent pressure
    fluctuation p_turb ~ rho*v^2*Ti (Ti=0.07 turbulence intensity),
    converted to volume velocity via constriction impedance Zc.
    The white-noise randomness IS the turbulence; the amplitude is
    from aeroacoustics (NOT pre-recorded). Returns (noise, Re, info).
    """
    rng = np.random.default_rng(seed)
    NU = 0.15  # cm^2/s, kinematic viscosity
    RHO = 0.00114  # g/cm^3
    v = ug_dc / max(area_c, 1e-9)          # cm/s
    d = np.sqrt(max(area_c, 1e-9))         # cm
    Re = v * d / NU
    if Re < 1800:
        return np.zeros(n48), Re, {"turbulent": False}
    Ti = 0.07
    p_rms = RHO * v ** 2 * Ti              # dyn/cm^2
    Zc = RHO * 35000.0 / max(area_c, 1e-9)
    Us_rms = p_rms / Zc                   # cm^3/s
    info = {"turbulent": True, "v_cm_s": round(v, 1),
            "p_rms": round(p_rms, 1), "Us_rms": round(Us_rms, 3)}
    return rng.standard_normal(n48) * Us_rms, Re, info


def _constriction_flow(area_c, Psub_Pa=1080.0):
    """Physical DC flow through constriction (choked by Bernoulli).

    The narrow constriction (not the glottis) limits the flow:
    Ug = Ac * sqrt(2*Psub/rho). Prevents unphysical pressure drops.
    """
    RHO = 0.00114  # g/cm^3
    Psub_dyn = Psub_Pa * 10.0  # Pa -> dyn/cm^2
    return area_c * np.sqrt(2.0 * Psub_dyn / RHO)  # cm^3/s


def synthesize_fricative_v(fricative="s", vowel="a", fric_ms=200,
                           vowel_ms=300, tension=1.35, seed=0):
    """Fricative + vowel, e.g. /sa/. Returns 16kHz waveform.

    Timeline: [frication: abducted DC flow + constriction turbulence
    noise injected at the constriction, filtered by front cavity] ->
    [vowel: normal voicing]. The noise is sustained (not a transient),
    and its spectrum is shaped by the front-cavity resonances —
    the physical /s/ vs /f/ distinction.
    """
    cfg = FRICATIVE_CONFIG[fricative]
    pos, area_c = cfg["pos"], cfg["area"]
    fric_s, vowel_s = fric_ms / 1000.0, vowel_ms / 1000.0

    # DC flow is constriction-limited (Bernoulli choked), NOT glottis-limited.
    # The narrow constriction determines the flow; glottis is abducted.
    Psub_Pa = 800.0 * tension  # matches glottis set_tension
    ug_dc = _constriction_flow(area_c, Psub_Pa)
    n_fric48 = int(fric_s * SR_TRACT)
    ug_fric48 = np.full(n_fric48, ug_dc)

    # turbulence noise at constriction (48kHz)
    noise48, Re, ninfo = _frication_noise(n_fric48, ug_dc, area_c, seed)

    # vowel flow (voiced)
    ug_vow16 = _steady_glottal_flow(tension, 0.2 + vowel_s + 0.1)
    ug_vow16 = ug_vow16[:int((vowel_s + 0.1) * SR)]

    # tract with fricative constriction
    tract = VocalTract()
    tract.set_articulators(*HER_VOWELS[vowel])
    idx = tract.set_fricative(pos, area_c)
    tract.reset()
    Zj = tract.impedance_at(idx)

    # frication: DC flow (already at 48kHz) + interior noise at constriction
    L = min(len(ug_fric48), len(noise48))
    out_fric = tract._run48(ug_fric48[:L], inject={idx: (noise48[:L], Zj)})

    # vowel: remove constriction, voice
    tract.set_articulators(*HER_VOWELS[vowel])
    ug_vowel48 = resample_poly(ug_vow16, 3, 1)
    out_vowel = tract._run48(ug_vowel48)

    out48 = np.concatenate([out_fric, out_vowel])
    out16 = resample_poly(out48, 1, 3).astype(float)
    from scipy.signal import butter, filtfilt
    # DC blocker: zero-phase highpass (no transient) at 40Hz.
    # Removes inaudible DC from steady flows; sound is AC pressure.
    b_hp, a_hp = butter(2, 40.0 / (SR / 2.0), 'high')
    out16 = filtfilt(b_hp, a_hp, out16)
    mx = np.abs(out16).max()
    info = {"Re": round(float(Re), 1), "fric_ms": fric_ms,
            "noise_info": ninfo}
    return out16 / (mx + 1e-12), info


def _hf_ratio(x, sr=SR, cutoff=4000.0):
    """High-frequency energy ratio: E(>cutoff)/E(total)."""
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    hf = np.sum(X[f > cutoff] ** 2)
    tot = np.sum(X ** 2) + 1e-12
    return float(hf / tot)


def gate_fricative(fricative="s", vowel="a", vowel_refs=None):
    """Gate: sustained frication + correct spectral shape + vowel.

    /s/: high-freq dominated (alveolar, short front cavity -> peak 4-7kHz)
    /f/: diffuse/flat (labiodental, very short cavity)
    Must be SUSTAINED (not a transient like plosive burst).
    Returns (ok, diagnostics).
    """
    y, info = synthesize_fricative_v(fricative, vowel)
    n_fric = int(info["fric_ms"] / 1000.0 * SR)
    fric = y[:n_fric]
    after = y[n_fric:]

    # sustained? exclude 30ms onset transient (articulator formation),
    # compare first vs second half of STEADY-STATE energy
    steady_fric = fric[int(0.03 * SR):]
    h1 = steady_fric[:len(steady_fric) // 2]
    h2 = steady_fric[len(steady_fric) // 2:]
    e1 = float(np.mean(h1 ** 2))
    e2 = float(np.mean(h2 ** 2))
    sustained_ratio = e2 / (e1 + 1e-12)

    centroid = _spectral_centroid(fric)
    hf = _hf_ratio(fric)

    # vowel after
    steady = after[int(-0.15 * SR):]
    rms_vowel = float(np.sqrt(np.mean(steady ** 2)))
    rms_fric = float(np.sqrt(np.mean(fric ** 2)))

    diag = {
        "Re": info["Re"],
        "sustained_ratio": round(sustained_ratio, 3),
        "centroid_hz": round(centroid, 1),
        "hf_ratio": round(hf, 3),
        "rms_fric": round(rms_fric, 4),
        "rms_vowel": round(rms_vowel, 4),
    }
    ok = True
    # 1. turbulent (Re > 1800)
    c1 = info["Re"] > 1800
    diag["turbulent"] = bool(c1)
    ok &= c1
    # 2. sustained (not transient): second half >= 50% of first half energy
    c2 = sustained_ratio > 0.5
    diag["sustained"] = bool(c2)
    ok &= c2
    # 3. audible (above -30dB re vowel)
    c3 = rms_fric > 0.03 * rms_vowel
    diag["audible"] = bool(c3)
    ok &= c3
    # 4. spectral shape by fricative
    if fricative == "s":
        # /s/: high centroid (>3500Hz) and strong HF
        c4 = (centroid > 3500.0) and (hf > 0.35)
        diag["s_shape"] = bool(c4)
        ok &= c4
    else:  # 'f'
        # /f/: diffuse — lower centroid than /s/, flatter (HF present but
        # not dominant peak). We check it's frication-like but NOT /s/-like.
        c4 = (centroid < 6000.0) and (hf > 0.15)
        diag["f_shape"] = bool(c4)
        ok &= c4
    # 5. vowel recognized
    if vowel_refs is not None:
        f = auditory_feature(steady / (np.abs(steady).max() + 1e-12))
        dists = {v: float(np.linalg.norm(f - vowel_refs[v])) for v in vowel_refs}
        recog = min(dists, key=dists.get)
        diag["vowel_recognized"] = recog
        c5 = (recog == vowel)
        diag["vowel_ok"] = bool(c5)
        ok &= c5
    return ok, diag
