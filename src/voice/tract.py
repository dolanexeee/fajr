"""Vocal tract — Kelly-Lochbaum waveguide (Pink-Trombone style).

The tract is N cylindrical sections; sound propagates as forward/backward
pressure waves scattering at each area discontinuity:
    r_j = (A[j] - A[j-1]) / (A[j] + A[j-1])
Glottal flow from the two-mass model is injected at the bottom;
the lips radiate with a resistive termination.

Articulatory control is 3-D: constriction location, constriction degree,
lip rounding — enough for the vowel triangle (DIVA babbling space).

Pure Python + numpy. Zero tokens at runtime.
"""
import numpy as np

from .hearing import SR

NSEC = 44          # tract sections (Pink-Trombone style)
SR_TRACT = 48000   # internal waveguide rate -> 44*350/48000 = 32cm tract
C = 35000.0        # cm/s, speed of sound
RHO_CGS = 0.00114  # g/cm^3


def _areas_from_articulators(loc, degree, rounding, n=NSEC):
    """Map 3 articulatory params to an area function (cm^2).

    loc:      0..1 constriction location (0=glottis end, 1=lips)
    degree:   0..1 constriction degree (0=open, 1=tight)
    rounding: 0..1 lip rounding (0=spread, 1=rounded)
    """
    x = np.linspace(0, 1, n)
    # baseline: neutral-ish tract, wider in the middle
    base = 2.0 + 2.0 * np.sin(np.pi * np.clip(x, 0, 1)) ** 0.7
    # tongue constriction: gaussian dip
    width = 0.10 + 0.10 * (1.0 - degree)
    dip = np.exp(-0.5 * ((x - loc) / width) ** 2)
    area = base * (1.0 - (0.15 + 0.80 * degree) * dip)
    # lips: last sections shrink with rounding
    lip = np.clip((x - 0.90) / 0.10, 0, 1)
    area = area * (1.0 - lip * (0.30 + 0.65 * rounding))
    return np.clip(area, 0.05, 12.0)


# canonical vowel presets (Fant-style area shapes) — adult guide targets
VOWELS = {
    "a": (0.30, 0.55, 0.15),   # pharyngeal constriction, open lips
    "i": (0.78, 0.80, 0.10),   # high-front palatal constriction, spread lips
    "u": (0.50, 0.65, 1.00),   # velar constriction, fully rounded lips
}

# Fajr's own discovered articulations (DIVA: her f0 needs tighter /u/)
# Found by systematic search 2026-10-30: her /u/ at (0.5,0.65,1.0) was
# recognized as /i/; (0.5,0.9,1.0) is recognized as /u/.
HER_VOWELS = {
    "a": (0.30, 0.55, 0.15),
    "i": (0.78, 0.80, 0.10),
    "u": (0.50, 0.90, 1.00),
}


class VocalTract:
    def __init__(self, n_sec=NSEC):
        self.n = n_sec
        self.up = np.zeros(n_sec + 1)    # forward waves at junctions
        self.dn = np.zeros(n_sec + 1)    # backward waves at junctions
        self.r = np.zeros(n_sec + 1)     # reflection coefficients
        self._base_areas = None          # areas without closure (for plosives)
        self.set_areas(_areas_from_articulators(0.5, 0.2, 0.2, n_sec))

    def set_areas(self, areas):
        a = np.asarray(areas, dtype=float)
        assert a.size == self.n
        self._base_areas = a.copy()
        self._apply_areas(a)

    def _apply_areas(self, a):
        # junction reflections r_j = (A_below - A_above)/(A_below + A_above)
        # (pressure-wave reflection from acoustic impedance Z = rho*c/A)
        self.r[1:-1] = (a[:-1] - a[1:]) / (a[:-1] + a[1:])
        self.r[0] = 0.85    # glottis: high reflection (nearly closed source)
        self.r[-1] = -0.70  # lips: resistive radiation
        self._a0 = a[0]
        self._areas = a.copy()  # stored for impedance calculations

    def set_fricative(self, pos, area):
        """Fricative constriction: narrow (not closed) at pos.

        Unlike set_closure (full seal), this leaves a narrow channel.
        Turbulent flow through it generates frication noise (injected
        separately at the constriction junction).
        Returns the junction index.
        """
        if self._base_areas is None:
            return None
        a = self._base_areas.copy()
        idx = int(np.clip(pos, 0, 1) * (self.n - 1))
        # narrow the constriction region (3 sections) to target area
        for j in (idx - 1, idx, idx + 1):
            if 0 <= j < self.n:
                a[j] = min(a[j], area)
        self._apply_areas(a)
        return idx

    def impedance_at(self, idx):
        """Characteristic impedance Z = rho*c/A at junction idx (cgs)."""
        if self._areas is None:
            return RHO_CGS * C / 2.0
        a = self._areas[min(max(idx, 0), self.n - 1)]
        return RHO_CGS * C / max(a, 1e-9)

    def set_closure(self, pos, closed):
        """Plosive closure: seal the tract at pos (0=glottis, 1=lips).

        When closed, 3 sections get near-zero area -> full reflection,
        pressure builds behind the seal. Sudden reopening releases it as
        the burst transient — pure waveguide physics, no injected noise.
        Call between _run48 chunks; wave state persists, so the release
        transient emerges naturally.
        """
        if self._base_areas is None:
            return
        a = self._base_areas.copy()
        if closed and pos is not None:
            idx = int(np.clip(pos, 0, 1) * (self.n - 1))
            for j in (idx - 1, idx, idx + 1):
                if 0 <= j < self.n:
                    a[j] = 1e-6  # tight seal; 3-section avoids NaN, tiny
                    # leakage acts as physical pressure relief
        self._apply_areas(a)

    def set_vowel(self, name):
        self.set_areas(_areas_from_articulators(*VOWELS[name], self.n))

    def set_articulators(self, loc, degree, rounding):
        self.set_areas(_areas_from_articulators(loc, degree, rounding, self.n))

    def reset(self):
        self.up.fill(0.0)
        self.dn.fill(0.0)

    def run(self, glottal_flow):
        """Filter glottal flow -> radiated pressure waveform (16 kHz out).

        The waveguide runs at 48 kHz internally (44 sections = 32 cm
        tract, Pink-Trombone style); input is upsampled x3, output
        downsampled back to 16 kHz.
        """
        from scipy.signal import resample_poly
        ug = np.asarray(glottal_flow, dtype=float)
        ug48 = resample_poly(ug, 3, 1)
        out48 = self._run48(ug48)
        return resample_poly(out48, 1, 3).astype(float)

    def _run48(self, ug, inject=None):
        """Waveguide run with optional interior injection.

        inject: dict {junction_idx: (noise_array, Z_j)} for aeroacoustic
        sources (e.g., frication at constriction). The noise is a volume
        velocity source: creates symmetric forward/backward pressure waves
        p = U_s * Z_j / 2 in each direction. Arrays must match ug length.
        """
        n = ug.size
        out = np.empty(n)
        # characteristic impedance at glottis (cgs): Z = rho*c/A
        Z = RHO_CGS * C / self._a0
        up = self.up   # up[j]: forward wave AT junction j (arrived from below)
        dn = self.dn   # dn[j]: backward wave AT junction j (arrived from above)
        r = self.r
        N = self.n
        ri = r[1:N]
        # precompute injection pressure scales
        inj = None
        if inject:
            inj = [(j, arr, Zj / 2.0) for j, (arr, Zj) in inject.items()]
        for i in range(n):
            # 1) glottis junction: source + reflection of arriving downgoing wave
            up0 = 0.5 * ug[i] * Z + r[0] * dn[0]
            # 1b) interior aeroacoustic sources (volume velocity -> pressure)
            if inj:
                for j, arr, pscale in inj:
                    p_src = arr[i] * pscale
                    up[j] += p_src
                    dn[j] += p_src
            # 2) interior junctions j=1..N-1
            a_in = up[1:N]      # arrived from below
            b_in = dn[1:N]      # arrived from above
            up_leave = (1.0 + ri) * a_in - ri * b_in   # going up from j
            dn_leave = ri * a_in + (1.0 - ri) * b_in   # going down from j
            # 3) lips: radiate + reflect
            out[i] = up[N] * (1.0 + r[N])
            dn_lip = r[N] * up[N]
            # 4) propagate one section:
            #    up[j] <= wave that left junction j-1 going up
            #    dn[j] <= wave that left junction j+1 going down
            up[1] = up0
            up[2:N + 1] = up_leave
            dn[0:N - 1] = dn_leave
            dn[N - 1] = dn_lip
        self.up, self.dn = up, dn
        return out
