"""Fricatives + voiceless plosives on human tract (20 sec, 14.6cm).

Adapts fricatives.py and consonants.py voiceless code:
- Human tract + human vowel articulations
- Rosenberg warm source (not two-mass)
- Positions are fractions -> scale automatically.
"""
import numpy as np
from scipy.signal import resample_poly, butter, filtfilt

from .tract import VocalTract, _areas_from_articulators, SR_TRACT
from .rosenberg import rosenberg_train, SR
from .diva import auditory_feature
from .syllables import HUMAN_ART, HUMAN_N
from .fricatives import _frication_noise, _constriction_flow, _spectral_centroid, _hf_ratio

FRICATIVE_CONFIG = {
    "s": {"pos": 0.85, "area": 0.08},
    "f": {"pos": 0.95, "area": 0.15},
}


def synthesize_fricative_v(fricative="s", vowel="a", fric_ms=200,
                           vowel_ms=280, f0s=185, f0e=175, seed=0):
    """Fricative + vowel on human tract. Returns 16kHz."""
    cfg = FRICATIVE_CONFIG[fricative]
    pos, area_c = cfg["pos"], cfg["area"]
    fric_s, vowel_s = fric_ms / 1000.0, vowel_ms / 1000.0

    Psub_Pa = 800.0 * 1.35
    ug_dc = _constriction_flow(area_c, Psub_Pa)
    n_fric48 = int(fric_s * SR_TRACT)
    ug_fric48 = np.full(n_fric48, ug_dc)
    noise48, Re, ninfo = _frication_noise(n_fric48, ug_dc, area_c, seed)

    # vowel: Rosenberg voiced flow
    flow_v = rosenberg_train(f0s, f0e, vowel_s + 0.1)[:int(vowel_s * SR)] * 1e6

    tract = VocalTract(n_sec=HUMAN_N)
    tract.set_areas(_areas_from_articulators(*HUMAN_ART[vowel], n=HUMAN_N))
    idx = tract.set_fricative(pos, area_c)
    tract.reset()
    Zj = tract.impedance_at(idx)

    L = min(len(ug_fric48), len(noise48))
    out_fric = tract._run48(ug_fric48[:L], inject={idx: (noise48[:L], Zj)})

    tract.set_areas(_areas_from_articulators(*HUMAN_ART[vowel], n=HUMAN_N))
    out_vowel = tract._run48(resample_poly(flow_v, 3, 1))

    out48 = np.concatenate([out_fric, out_vowel])
    out16 = resample_poly(out48, 1, 3).astype(float)
    b_hp, a_hp = butter(2, 40.0 / (SR / 2.0), 'high')
    out16 = filtfilt(b_hp, a_hp, out16)
    mx = np.abs(out16).max()
    return out16 / (mx + 1e-12), {"Re": round(float(Re), 1), "fric_ms": fric_ms}


def gate_fricative(fricative="s", vowel="a", vowel_refs=None):
    y, info = synthesize_fricative_v(fricative, vowel)
    n_fric = int(info["fric_ms"] / 1000.0 * SR)
    fric = y[:n_fric]
    after = y[n_fric:]
    steady_fric = fric[int(0.03 * SR):]
    h1 = steady_fric[:len(steady_fric) // 2]
    h2 = steady_fric[len(steady_fric) // 2:]
    sustained_ratio = float(np.mean(h2 ** 2)) / (float(np.mean(h1 ** 2)) + 1e-12)
    centroid = _spectral_centroid(fric)
    hf = _hf_ratio(fric)
    steady = after[int(-0.15 * SR):]
    rms_vowel = float(np.sqrt(np.mean(steady ** 2)))
    rms_fric = float(np.sqrt(np.mean(fric ** 2)))
    diag = {
        "Re": info["Re"],
        "sustained_ratio": round(sustained_ratio, 3),
        "centroid_hz": round(centroid, 1),
        "hf_ratio": round(hf, 3),
    }
    ok = True
    c1 = info["Re"] > 1800; diag["turbulent"] = bool(c1); ok &= c1
    c2 = sustained_ratio > 0.5; diag["sustained"] = bool(c2); ok &= c2
    c3 = rms_fric > 0.03 * rms_vowel; diag["audible"] = bool(c3); ok &= c3
    if fricative == "s":
        c4 = (centroid > 3500.0) and (hf > 0.35)
        diag["s_shape"] = bool(c4); ok &= c4
    else:
        c4 = (centroid < 6000.0) and (hf > 0.15)
        diag["f_shape"] = bool(c4); ok &= c4
    if vowel_refs is not None:
        f = auditory_feature(steady / (np.abs(steady).max() + 1e-12))
        dists = {v: float(np.linalg.norm(f - vowel_refs[v])) for v in vowel_refs}
        recog = min(dists, key=dists.get)
        diag["vowel_recognized"] = recog
        c5 = (recog == vowel); diag["vowel_ok"] = bool(c5); ok &= c5
    return ok, diag
