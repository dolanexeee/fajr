"""Glottal source — Ishizaka & Flanagan (1972) two-mass model.

"Synthesis of voiced sounds from a two-mass model of the vocal cords",
Bell System Technical Journal 51: 1233-1268.

Two masses (lower m1, upper m2) with stiffness k1/k2, coupling kc,
damping r1/r2, driven by subglottal pressure Ps through Bernoulli flow.
Output: glottal volume-velocity waveform Ug(t) driving the vocal tract.

Pure Python + numpy. Zero tokens at runtime.
"""
import numpy as np

from .hearing import SR

RHO = 1.14  # air density, kg/m^3
KENT = 1.37  # glottal entrance loss coefficient (IF72 / Fulcher et al. 2012)


class TwoMassGlottis:
    """Ishizaka-Flanagan two-mass vocal fold model.

    Aerodynamics EXACTLY per Ishizaka & Flanagan (1972) via
    Fulcher, Scherer & Waddle (JASA 2012, PMC3477191):

      A1 = Lg(ξ0+x1),  A2 = Lg(ξ0+x2)                       [Eq 10]
      P1 = Psub(A1²−A2²)/(A1²+(kent−1)A2²),  P2 = 0          [Eq 11]
      Ug from Psub = (ρUg²/2)[(kent−1)/A1² + 1/A2²]          [Eq 8]
      F1 = P1·Lg·d1,  F2 = 0                                [Eq 4,5]

    The small-amplitude limit gives a flow-induced stiffness
    φ = 2·Lg·d1·Psub/(kent·ξ0) acting as φ(x1−x2): a
    non-symmetric (circulatory) coupling that destabilizes the
    synchronized modes above threshold pressure — a flutter
    instability, no flow lag required.
    """

    def __init__(self, tension=1.0, pressure=800.0):
        # --- mechanical parameters (Ishizaka & Flanagan 1972, SI) ---
        self.m1 = 1.25e-4      # kg, lower mass
        self.m2 = 2.5e-5       # kg, upper mass
        self.k10 = 80.0        # N/m, lower stiffness at tension=1
        self.k20 = 8.0         # N/m, upper stiffness at tension=1
        self.kc0 = 25.0        # N/m, coupling stiffness at tension=1
        zeta = 0.10
        self.r1 = 2 * zeta * np.sqrt(self.k10 * self.m1)
        self.r2 = 2 * zeta * np.sqrt(self.k20 * self.m2)
        # --- geometry ---
        self.L = 0.014         # m, fold length
        self.d1 = 0.0025       # m, lower fold thickness
        self.d2 = 0.0005       # m, upper fold thickness
        self.x01 = 5e-4        # m, rest half-width lower
        self.x02 = 5e-4        # m, rest half-width upper
        self.ag_min = 1e-9     # m^2, numerical floor
        # --- aerodynamics ---
        self.Ps = float(pressure)  # Pa, subglottal pressure
        self.set_tension(tension)
        self.reset()

    def _areas(self):
        ag1 = max(self.ag_min, self.L * (self.x01 + self.x1))
        ag2 = max(self.ag_min, self.L * (self.x02 + self.x2))
        return ag1, ag2

    def set_tension(self, tension, pressure=None):
        """Vocal-fold tension: scales stiffness (the f0 control knob).

        Higher tension raises the phonation threshold, so subglottal
        pressure is coupled up proportionally (as in real voicing:
        high notes need more lung pressure) unless given explicitly.
        """
        T = max(0.25, float(tension))
        self.k1 = self.k10 * T
        self.k2 = self.k20 * T
        self.kc = self.kc0 * T
        self.Ps = float(pressure) if pressure is not None else 800.0 * T

    def reset(self):
        self.x1 = 0.0
        self.v1 = 0.0
        self.x2 = 0.0
        self.v2 = 0.0

    def _flow_and_forces(self, dt):
        """IF72 aerodynamics, exact per PMC3477191 Eq 8/10/11.

        Closed phase (either mass at/below closure): the glottis is shut,
        Ug = 0, and the full subglottal pressure bears on the closure,
        blowing it open — the physical phonation cycle
        (closed → blown open → Bernoulli recoil → closed).
        """
        A1 = self.L * (self.x01 + self.x1)
        A2 = self.L * (self.x02 + self.x2)
        if A1 <= 0.0 or A2 <= 0.0:
            # glottis closed: no flow, full Psub on the closure
            return 0.0, self.Ps * self.L * self.d1, 0.0
        # glottal flow from Eq 8
        denom = (KENT - 1.0) / A1 ** 2 + 1.0 / A2 ** 2
        Ug = np.sqrt(2.0 * self.Ps / RHO / denom)
        # intraglottal pressures, Eq 11 and P2 = 0
        P1 = self.Ps * (A1 ** 2 - A2 ** 2) / (A1 ** 2 + (KENT - 1.0) * A2 ** 2)
        F1 = P1 * self.L * self.d1
        F2 = 0.0
        return Ug, F1, F2

    def _step(self, dt):
        Ug, F1, F2 = self._flow_and_forces(dt)
        # collision springs (folds meet at midline)
        if self.x1 < -self.x01:
            F1 += -(3.0 * self.k1) * (self.x1 + self.x01) - 2.0 * self.r1 * self.v1
        if self.x2 < -self.x02:
            F2 += -(3.0 * self.k2) * (self.x2 + self.x02) - 2.0 * self.r2 * self.v2
        a1 = (-self.k1 * self.x1 - self.kc * (self.x1 - self.x2)
              - self.r1 * self.v1 + F1) / self.m1
        a2 = (-self.k2 * self.x2 - self.kc * (self.x2 - self.x1)
              - self.r2 * self.v2 + F2) / self.m2
        # semi-implicit Euler
        self.v1 += a1 * dt
        self.x1 += self.v1 * dt
        self.v2 += a2 * dt
        self.x2 += self.v2 * dt
        return Ug

    def run(self, dur_s, substeps=2):
        """Run for dur_s seconds. Returns glottal flow waveform (m^3/s)."""
        n = int(dur_s * SR)
        out = np.empty(n)
        dt = 1.0 / SR / substeps
        for i in range(n):
            ug = 0.0
            for _ in range(substeps):
                ug = self._step(dt)
            out[i] = ug
        return out

    # ------------------------------------------------------------ control
    def calibrate(self, dur_s=1.0):
        """Measure f0 at tension=1; returns measured f0 (Hz)."""
        from .hearing import f0_autocorr
        self.set_tension(1.0)
        self.reset()
        self.x1 = 1e-6  # seed: exact equilibrium would sit still forever
        w = self.run(dur_s)
        steady = w[int(0.6 * SR):]  # skip onset transient
        return f0_autocorr(steady)

    def tension_for_f0(self, f_target, f_ref=None):
        """Tension giving f_target, via f0 ~ sqrt(tension) law."""
        if f_ref is None:
            f_ref = self.calibrate()
        if f_ref <= 0:
            raise RuntimeError("glottis did not oscillate at tension=1")
        return (f_target / f_ref) ** 2
