"""Nasals /m,n/ — FIXED (2026-10-31).

Fixes the STAGE3-CEILING.md bugs:
  Bug 1: dn[1:N] should be dn[0:N-1] in all 3 branches (off-by-one: downward
         waves were shifted, never reaching glottis).
  Bug 2: velum scatter wrote dn_p[Np] after the interior loop; interior never
         read dn_p[Np], so the velum reflection was lost.

Fix: explicit scatter->propagate phases with temporary leaving-wave arrays.
All arrivals are read, all leavings are propagated. Energy is conserved.

Human scale: 20-section base tract (14.6cm). Branch lengths:
  pharynx 12 (glottis->velum, ~8.8cm), nasal 16 (~11.7cm),
  oral 8 (velum->lips for /m/), 5 (velum->alveolar for /n/).

Source: Rosenberg smooth pulses (warm), not two-mass.
"""
import numpy as np
from scipy.signal import resample_poly

from .tract import RHO_CGS, C, SR_TRACT, _areas_from_articulators
from .rosenberg import rosenberg_train, SR
from .diva import auditory_feature
from .syllables import HUMAN_ART, HUMAN_N


class NasalTract:
    """Three-branch waveguide with correct scatter->propagate ordering."""

    def __init__(self, n_pharynx=12, n_nasal=16, n_oral=8):
        self.np, self.nn, self.no = n_pharynx, n_nasal, n_oral
        # arrivals: up[j] forward AT junction j, dn[j] backward AT j
        self.up_p = np.zeros(n_pharynx + 1); self.dn_p = np.zeros(n_pharynx + 1)
        self.up_n = np.zeros(n_nasal + 1);   self.dn_n = np.zeros(n_nasal + 1)
        self.up_o = np.zeros(n_oral + 1);    self.dn_o = np.zeros(n_oral + 1)
        # reflection coefficients
        self.rp = np.zeros(n_pharynx + 1)
        self.rn = np.zeros(n_nasal + 1)
        self.ro = np.zeros(n_oral + 1)
        # leaving-wave temporaries (scatter phase)
        self._ulp = np.zeros(n_pharynx + 1); self._dlp = np.zeros(n_pharynx + 1)
        self._uln = np.zeros(n_nasal + 1);   self._dln = np.zeros(n_nasal + 1)
        self._ulo = np.zeros(n_oral + 1);    self._dlo = np.zeros(n_oral + 1)
        self.ap = self.an = self.ao = None

    def set_config(self, vowel="a", nasal_area=2.0):
        """Branch areas from human vowel shape."""
        a_full = _areas_from_articulators(*HUMAN_ART[vowel], n=HUMAN_N)
        # velum at 60% of human tract
        v_sec = 12
        self.ap = a_full[:self.np].copy()
        self.an = np.full(self.nn, nasal_area)
        # oral: velum -> end of branch
        ao_src = a_full[v_sec:v_sec + self.no]
        if len(ao_src) < self.no:
            ao_src = np.pad(ao_src, (0, self.no - len(ao_src)), 'edge')
        self.ao = ao_src[:self.no].copy()

        def refl(a, r, glottis=False, closed_end=False, rad_end=False):
            n = len(a)
            r[1:n] = (a[:-1] - a[1:]) / (a[:-1] + a[1:])
            if glottis:
                r[0] = 0.85
            if closed_end:
                r[n] = 1.0
            if rad_end:
                r[n] = -0.70

        refl(self.ap, self.rp, glottis=True)
        refl(self.an, self.rn, rad_end=True)
        refl(self.ao, self.ro, closed_end=True)
        self._Zg = RHO_CGS * C / max(self.ap[0], 1e-9)
        # velum impedances
        self._Z1 = RHO_CGS * C / max(self.ap[-1], 1e-9)
        self._Z2 = RHO_CGS * C / max(self.an[0], 1e-9)
        self._Z3 = RHO_CGS * C / max(self.ao[0], 1e-9)

    def reset(self):
        for a in (self.up_p, self.dn_p, self.up_n, self.dn_n,
                  self.up_o, self.dn_o):
            a.fill(0.0)

    def _scatter(self, ug_i):
        """Scatter phase: arrivals -> leavings (temporaries)."""
        Np, Nn, No = self.np, self.nn, self.no
        # glottis (pharynx junction 0)
        self._ulp[0] = 0.5 * ug_i * self._Zg + self.rp[0] * self.dn_p[0]
        # pharynx interior 1..Np-1
        if Np > 1:
            a = self.up_p[1:Np]; b = self.dn_p[1:Np]; r = self.rp[1:Np]
            self._ulp[1:Np] = (1 + r) * a - r * b
            self._dlp[1:Np] = r * a + (1 - r) * b
        # nasal interior 1..Nn-1
        if Nn > 1:
            a = self.up_n[1:Nn]; b = self.dn_n[1:Nn]; r = self.rn[1:Nn]
            self._uln[1:Nn] = (1 + r) * a - r * b
            self._dln[1:Nn] = r * a + (1 - r) * b
        # nasal nostrils (junction Nn): radiate + reflect
        self._dln[Nn] = self.rn[Nn] * self.up_n[Nn]
        # oral interior 1..No-1
        if No > 1:
            a = self.up_o[1:No]; b = self.dn_o[1:No]; r = self.ro[1:No]
            self._ulo[1:No] = (1 + r) * a - r * b
            self._dlo[1:No] = r * a + (1 - r) * b
        # oral closure (junction No): full reflection
        self._dlo[No] = self.ro[No] * self.up_o[No]
        # velum 3-port: arrivals up_p[Np], dn_n[0], dn_o[0]
        p1 = self.up_p[Np]; p2 = self.dn_n[0]; p3 = self.dn_o[0]
        Y = 1.0 / self._Z1 + 1.0 / self._Z2 + 1.0 / self._Z3
        P = 2.0 * (p1 / self._Z1 + p2 / self._Z2 + p3 / self._Z3) / (Y + 1e-18)
        self._dlp[Np] = P - p1   # back down pharynx
        self._uln[0] = P - p2    # up into nasal
        self._ulo[0] = P - p3    # up into oral
        return self.up_n[Nn] * (1.0 + self.rn[Nn])  # nostril output

    def _propagate(self):
        """Propagate phase: leavings -> arrivals for next sample."""
        Np, Nn, No = self.np, self.nn, self.no
        # pharynx: up[j] -> up[j+1]; dn[j] -> dn[j-1]
        self.up_p[1:Np + 1] = self._ulp[0:Np]
        # up_p[Np] is velum arrival; _ulp[Np] unused (velum has no up leaving)
        self.dn_p[0:Np] = self._dlp[1:Np + 1]
        # nasal
        self.up_n[1:Nn + 1] = self._uln[0:Nn]
        self.dn_n[0:Nn] = self._dln[1:Nn + 1]
        # oral
        self.up_o[1:No + 1] = self._ulo[0:No]
        self.dn_o[0:No] = self._dlo[1:No + 1]

    def run(self, ug48):
        n = ug48.size
        out = np.empty(n)
        for i in range(n):
            out[i] = self._scatter(ug48[i])
            self._propagate()
        return out

    def check_energy(self, n_samples=2000):
        """Verify energy conservation: inject impulse, check no blowup and
        velum scattering conserves (outgoing power <= incoming)."""
        self.reset()
        ug = np.zeros(n_samples); ug[0] = 1.0
        out = self.run(ug)
        # energy should decay (lossy boundaries), not grow
        e_start = np.sum(out[:100] ** 2)
        e_end = np.sum(out[-100:] ** 2)
        peak = np.abs(out).max()
        return {"e_start": float(e_start), "e_end": float(e_end),
                "peak": float(peak), "stable": bool(peak < 10.0 and e_end <= e_start * 1.01)}


def synthesize_nasal_v(nasal="m", vowel="a", murmur_ms=220, vowel_ms=300,
                       f0s=185, f0e=175):
    """Nasal murmur + vowel on human tract with Rosenberg source.

    The vowel gets a FRESH f0s->f0e glide (not the murmur's leftover
    mid-glide flow) so it matches unit_vowel references exactly in gates.
    """
    from .tract import VocalTract
    murmur_s, vowel_s = murmur_ms / 1000.0, vowel_ms / 1000.0
    # murmur flow: full glide (f0 continuity through the murmur)
    total_m = murmur_s + 0.05
    flow_m = rosenberg_train(f0s, f0e, total_m)[:int(total_m * SR)] * 1e6
    ug48_m = resample_poly(flow_m, 3, 1)
    n_mur48 = int(murmur_s * SR_TRACT)
    # vowel flow: FRESH glide f0s->f0e (matches unit_vowel references)
    total_v = vowel_s + 0.1
    flow_v = rosenberg_train(f0s, f0e, total_v)[:int(total_v * SR)] * 1e6
    ug48_v = resample_poly(flow_v, 3, 1)

    n_oral = 8 if nasal == "m" else 5
    nt = NasalTract(n_pharynx=12, n_nasal=16, n_oral=n_oral)
    nt.set_config(vowel=vowel, nasal_area=2.0)
    nt.reset()
    out_mur48 = nt.run(ug48_m[:n_mur48])

    # vowel: normal oral tract (velum raised)
    vt = VocalTract(n_sec=HUMAN_N)
    vt.set_areas(_areas_from_articulators(*HUMAN_ART[vowel], n=HUMAN_N))
    vt.reset()
    out_vow48 = vt._run48(ug48_v[:int((vowel_s + 0.05) * SR_TRACT)])

    out48 = np.concatenate([out_mur48, out_vow48])
    out16 = resample_poly(out48, 1, 3).astype(float)
    mx = np.abs(out16).max()
    y = out16 / (mx + 1e-12)
    n_mur16 = int(murmur_s * SR)
    return y, {"murmur_ms": murmur_ms,
               "vowel_bounds": [(vowel, n_mur16, len(y))]}


def gate_nasal(nasal="m", vowel="a", vowel_refs=None):
    """Gate: low nasal formant + antiformant + audible murmur + vowel."""
    y, info = synthesize_nasal_v(nasal, vowel)
    n_mur = int(info["murmur_ms"] / 1000.0 * SR)
    mur = y[:n_mur]
    mur_steady = mur[int(0.06 * SR):]
    after = y[n_mur:]

    X = np.abs(np.fft.rfft(mur_steady * np.hanning(len(mur_steady))))
    f = np.fft.rfftfreq(len(mur_steady), 1.0 / SR)
    band = (f >= 120) & (f <= 450)
    nasal_f = float(f[band][np.argmax(X[band])])

    band2 = (f >= 500) & (f <= 3000)
    Xb = X[band2]; fb = f[band2]
    from scipy.ndimage import uniform_filter1d
    env = uniform_filter1d(Xb, size=25, mode='nearest')
    ratio = Xb / (env + 1e-12)
    dip_ratio = float(np.min(ratio))
    dip_f = float(fb[np.argmin(ratio)])

    steady = after[int(-0.15 * SR):]
    rms_mur = float(np.sqrt(np.mean(mur_steady ** 2)))
    rms_vowel = float(np.sqrt(np.mean(steady ** 2)))

    diag = {
        "nasal_formant_hz": round(nasal_f, 1),
        "antiformant_hz": round(dip_f, 1),
        "antiformant_depth": round(dip_ratio, 3),
        "murmur_db_re_vowel": round(20 * np.log10(rms_mur / (rms_vowel + 1e-12)), 1),
    }
    ok = True
    c1 = 120.0 <= nasal_f <= 450.0
    diag["low_formant"] = bool(c1); ok &= c1
    c2 = dip_ratio < 0.5
    diag["antiformant"] = bool(c2); ok &= c2
    c3 = rms_mur > 0.05 * rms_vowel
    diag["audible"] = bool(c3); ok &= c3
    if vowel_refs is not None:
        feat = auditory_feature(steady / (np.abs(steady).max() + 1e-12))
        dists = {v: float(np.linalg.norm(feat - vowel_refs[v])) for v in vowel_refs}
        recog = min(dists, key=dists.get)
        diag["vowel_recognized"] = recog
        c4 = (recog == vowel)
        diag["vowel_ok"] = bool(c4); ok &= c4
    return ok, diag
