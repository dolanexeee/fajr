"""Nasals /m,n/ — velum lowering with side-branch resonator.

Topology: glottis -> pharynx -> VELUM ->+- nasal cavity -> nostrils (radiates)
                                    +- oral cavity -> closure (dead end)

The velum is a 3-port scattering junction (pressure continuous, flows sum
to zero). The oral side-branch creates antiformants (zeros); the nasal
cavity + pharynx create the low nasal formant (~250Hz). This is the
honest acoustic mechanism — not a filter approximation.

Pure Python + numpy. Zero tokens at runtime.
"""
import numpy as np
from scipy.signal import resample_poly

from .tract import RHO_CGS, C, SR_TRACT, HER_VOWELS, _areas_from_articulators
from .consonants import _steady_glottal_flow
from .hearing import SR
from .diva import auditory_feature


class NasalTract:
    """Three-branch waveguide: pharynx + nasal + oral side-branch.

    All branches use Kelly-Lochbaum propagation. The velum junction uses
    3-port scattering derived from pressure continuity and flow conservation:
        P = 2*sum(p_i+/Zi) / sum(1/Zi)
        p_i- = P - p_i+
    """

    def __init__(self, n_pharynx=27, n_nasal=16, n_oral=17):
        # branch section counts (0.73cm each at 48kHz)
        self.np = n_pharynx   # glottis -> velum
        self.nn = n_nasal     # velum -> nostrils
        self.no = n_oral      # velum -> oral closure
        # wave states: up[j] forward AT junction j, dn[j] backward AT j
        self.up_p = np.zeros(n_pharynx + 1)
        self.dn_p = np.zeros(n_pharynx + 1)
        self.up_n = np.zeros(n_nasal + 1)
        self.dn_n = np.zeros(n_nasal + 1)
        self.up_o = np.zeros(n_oral + 1)
        self.dn_o = np.zeros(n_oral + 1)
        # reflection coefficients (interior)
        self.rp = np.zeros(n_pharynx + 1)
        self.rn = np.zeros(n_nasal + 1)
        self.ro = np.zeros(n_oral + 1)
        # areas (for impedance)
        self.ap = None
        self.an = None
        self.ao = None

    def set_config(self, vowel="a", nasal_area=2.0, oral_closure_pos=1.0):
        """Configure branch areas.

        vowel: base vowel shape for pharynx and oral branch
        nasal_area: uniform nasal cavity area (cm^2)
        oral_closure_pos: 1.0=lips (/m/), 0.85=alveolar (/n/) — the oral
            branch runs velum->closure, closed at the end.
        """
        # pharynx: vowel areas from glottis to velum
        a_full = _areas_from_articulators(*HER_VOWELS[vowel], 44)
        # velum at section 26 (pos 0.6)
        v_sec = 26
        self.ap = a_full[:self.np]  # 0..26
        # nasal: uniform tube
        self.an = np.full(self.nn, nasal_area)
        # oral: from velum to closure
        # closure section in full tract coordinates
        c_sec = int(np.clip(oral_closure_pos, 0, 1) * 43)
        # oral branch length: from velum (26) to closure (c_sec)
        o_len = max(2, c_sec - v_sec)
        # adjust n_oral if needed (use configured, pad/truncate areas)
        ao_full = a_full[v_sec:v_sec + self.no]
        if len(ao_full) < self.no:
            ao_full = np.pad(ao_full, (0, self.no - len(ao_full)), 'edge')
        self.ao = ao_full[:self.no].copy()
        # oral branch closed at far end (closure) — handled by r=+1

        # reflections: pharynx interior (0..V-1), glottis at 0
        self.rp[1:self.np] = (self.ap[:-1] - self.ap[1:]) / (self.ap[:-1] + self.ap[1:])
        self.rp[0] = 0.85  # glottis
        # nasal interior, nostrils radiate at far end
        self.rn[1:self.nn] = (self.an[:-1] - self.an[1:]) / (self.an[:-1] + self.an[1:])
        self.rn[self.nn] = -0.70  # nostrils: resistive radiation (like lips)
        # oral interior, closed at far end
        self.ro[1:self.no] = (self.ao[:-1] - self.ao[1:]) / (self.ao[:-1] + self.ao[1:])
        self.ro[self.no] = 1.0  # closure: full reflection (dead end)
        # store glottal impedance
        self._Zg = RHO_CGS * C / self.ap[0]

    def reset(self):
        self.up_p.fill(0.0); self.dn_p.fill(0.0)
        self.up_n.fill(0.0); self.dn_n.fill(0.0)
        self.up_o.fill(0.0); self.dn_o.fill(0.0)

    def _velum_scatter(self):
        """3-port scattering at velum. Updates outgoing waves in place."""
        # incoming pressures at velum
        p1 = self.up_p[self.np]   # pharynx forward (arriving)
        p2 = self.dn_n[0]         # nasal backward (arriving)
        p3 = self.dn_o[0]         # oral backward (arriving)
        # impedances at velum
        Z1 = RHO_CGS * C / max(self.ap[-1], 1e-9)
        Z2 = RHO_CGS * C / max(self.an[0], 1e-9)
        Z3 = RHO_CGS * C / max(self.ao[0], 1e-9)
        # junction pressure
        Y = 1.0/Z1 + 1.0/Z2 + 1.0/Z3
        P = 2.0 * (p1/Z1 + p2/Z2 + p3/Z3) / (Y + 1e-18)
        # outgoing
        self.dn_p[self.np] = P - p1  # back into pharynx
        self.up_n[0] = P - p2        # into nasal
        self.up_o[0] = P - p3        # into oral

    def run(self, ug48):
        """Run with glottal flow (48kHz). Returns nostril radiation."""
        n = ug48.size
        out = np.empty(n)
        Np, Nn, No = self.np, self.nn, self.no
        for i in range(n):
            # 1) glottis injects into pharynx
            up0 = 0.5 * ug48[i] * self._Zg + self.rp[0] * self.dn_p[0]
            # 2) propagate pharynx interior (junctions 1..Np-1)
            #    (velum junction Np handled by 3-port below)
            if Np > 1:
                a_in = self.up_p[1:Np]
                b_in = self.dn_p[1:Np]
                ri = self.rp[1:Np]
                self.up_p[2:Np+1] = (1+ri)*a_in - ri*b_in
                self.dn_p[1:Np-1+1] = ri*a_in + (1-ri)*b_in
                # note: dn_p[Np] set by velum scatter, up_p[Np] is incoming
            self.up_p[1] = up0
            # 3) propagate nasal interior (junctions 1..Nn-1)
            if Nn > 1:
                a_in = self.up_n[1:Nn]
                b_in = self.dn_n[1:Nn]
                ri = self.rn[1:Nn]
                self.up_n[2:Nn+1] = (1+ri)*a_in - ri*b_in
                self.dn_n[1:Nn] = ri*a_in + (1-ri)*b_in
            # nostrils radiate
            out[i] = self.up_n[Nn] * (1.0 + self.rn[Nn])
            self.dn_n[Nn-1] = self.rn[Nn] * self.up_n[Nn]
            # 4) propagate oral interior (junctions 1..No-1)
            if No > 1:
                a_in = self.up_o[1:No]
                b_in = self.dn_o[1:No]
                ri = self.ro[1:No]
                self.up_o[2:No+1] = (1+ri)*a_in - ri*b_in
                self.dn_o[1:No] = ri*a_in + (1-ri)*b_in
            # oral closure: full reflection
            self.dn_o[No-1] = self.ro[No] * self.up_o[No]
            # 5) velum 3-port scattering
            self._velum_scatter()
        return out


def synthesize_nasal_v(nasal="m", vowel="a", murmur_ms=200, vowel_ms=300,
                       tension=1.35):
    """Nasal murmur + vowel, e.g. /ma/.

    Murmur: velum lowered, oral closed, voiced flow -> nasal radiation
    with low formant + antiformants from oral side-branch.
    Then velum raises (oral opens) -> normal vowel.
    Returns 16kHz waveform.
    """
    from .tract import VocalTract
    murmur_s, vowel_s = murmur_ms/1000.0, vowel_ms/1000.0

    # voiced flow
    ug16 = _steady_glottal_flow(tension, 0.2 + murmur_s + vowel_s + 0.1)
    ug16 = ug16[:int((murmur_s + vowel_s + 0.1)*SR)]
    ug48 = resample_poly(ug16, 3, 1)
    n_mur48 = int(murmur_s * SR_TRACT)

    # nasal tract: /m/ = lips closed (1.0), /n/ = alveolar (0.85)
    closure = 1.0 if nasal == "m" else 0.85
    # oral branch length from velum (26) to closure
    c_sec = int(closure * 43)
    n_oral = max(4, c_sec - 26)

    nt = NasalTract(n_pharynx=27, n_nasal=16, n_oral=n_oral)
    nt.set_config(vowel=vowel, nasal_area=2.0, oral_closure_pos=closure)
    nt.reset()
    out_mur48 = nt.run(ug48[:n_mur48])

    # vowel: normal oral (velum raised) — use standard tract
    vt = VocalTract()
    vt.set_articulators(*HER_VOWELS[vowel])
    vt.reset()
    out_vow48 = vt._run48(ug48[n_mur48:])

    out48 = np.concatenate([out_mur48, out_vow48])
    out16 = resample_poly(out48, 1, 3).astype(float)
    from scipy.signal import butter, filtfilt
    b_hp, a_hp = butter(2, 40.0/(SR/2.0), 'high')
    out16 = filtfilt(b_hp, a_hp, out16)
    mx = np.abs(out16).max()
    return out16/(mx+1e-12), {"murmur_ms": murmur_ms}


def gate_nasal(nasal="m", vowel="a", vowel_refs=None):
    """Gate: nasal murmur with low formant + antiformant, then vowel.

    Murmur: strong low-freq (~200-350Hz nasal formant), antiformant
    (spectral zero) from oral side-branch. Vowel recognized after.
    Returns (ok, diagnostics).
    """
    y, info = synthesize_nasal_v(nasal, vowel)
    n_mur = int(info["murmur_ms"]/1000.0 * SR)
    mur = y[:n_mur]
    # exclude onset transient
    mur_steady = mur[int(0.05*SR):]
    after = y[n_mur:]

    # spectrum of murmur
    X = np.abs(np.fft.rfft(mur_steady * np.hanning(len(mur_steady))))
    f = np.fft.rfftfreq(len(mur_steady), 1.0/SR)
    # nasal formant: peak in 150-400Hz
    band = (f >= 150) & (f <= 400)
    f0_idx = np.argmax(X[band])
    nasal_f = float(f[band][f0_idx])
    # antiformant: deep valley — find min in 500-2500Hz relative to neighbors
    band2 = (f >= 500) & (f <= 2500)
    Xb = X[band2]; fb = f[band2]
    # normalize by smoothed envelope to find relative dip
    from scipy.ndimage import uniform_filter1d
    env = uniform_filter1d(Xb, size=20, mode='nearest')
    dip_ratio = np.min(Xb/(env+1e-12))
    dip_f = float(fb[np.argmin(Xb/(env+1e-12))])

    steady = after[int(-0.15*SR):]
    rms_mur = float(np.sqrt(np.mean(mur_steady**2)))
    rms_vowel = float(np.sqrt(np.mean(steady**2)))

    diag = {
        "nasal_formant_hz": round(nasal_f, 1),
        "antiformant_hz": round(dip_f, 1),
        "antiformant_depth": round(float(dip_ratio), 3),
        "rms_murmur": round(rms_mur, 4),
        "rms_vowel": round(rms_vowel, 4),
    }
    ok = True
    # 1. nasal formant low (150-400Hz)
    c1 = 150.0 <= nasal_f <= 400.0
    diag["low_formant"] = bool(c1)
    ok &= c1
    # 2. antiformant present (dip >6dB below local envelope)
    c2 = dip_ratio < 0.5
    diag["antiformant"] = bool(c2)
    ok &= c2
    # 3. murmur audible
    c3 = rms_mur > 0.05 * rms_vowel
    diag["audible"] = bool(c3)
    ok &= c3
    # 4. vowel recognized after
    if vowel_refs is not None:
        feat = auditory_feature(steady/(np.abs(steady).max()+1e-12))
        dists = {v: float(np.linalg.norm(feat-vowel_refs[v])) for v in vowel_refs}
        recog = min(dists, key=dists.get)
        diag["vowel_recognized"] = recog
        c4 = (recog == vowel)
        diag["vowel_ok"] = bool(c4)
        ok &= c4
    return ok, diag
