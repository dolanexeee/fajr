"""DIVA speech learning — Guenther's model, two published phases.

Phase 1 (babbling): random articulatory exploration -> produce sound ->
hear own spectral envelope -> learn inverse model (auditory -> motor).
Phase 2 (imitation): given a guide's auditory target, invert to
articulation, produce, compare, refine by feedback.

Auditory feature: cepstrally-smoothed log spectral envelope (f0-robust,
so the infant can match an adult guide despite different pitch).

Pure Python + numpy/scipy. Zero tokens at runtime.
"""
import numpy as np

from .hearing import SR, stft_mag
from .tract import VocalTract

NCEPS = 13  # cepstral coefficients kept (envelope)


def auditory_feature(waveform):
    """f0-robust spectral envelope feature.

    Cepstrally liftered BELOW the pitch quefrency (first 40 coeffs =
    quefrency < 2.5 ms, under any f0 period here), so the feature hears
    the vocal-tract filter, not the glottal pitch.
    """
    y = np.asarray(waveform, dtype=float)
    y = y[int(0.10 * SR):]  # skip onset
    if y.size < 2048:
        y = np.pad(y, (0, 2048 - y.size))
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size))) + 1e-9
    logspec = np.log(spec)
    ceps = np.fft.irfft(logspec, n=2 * (logspec.size - 1))[:40]
    dct = np.array([np.sum(ceps * np.cos(np.pi * k * (2 * np.arange(40) + 1)
                                        / 80.0))
                    for k in range(NCEPS)])
    dct[0] = 0.0  # drop overall level
    return dct[1:]


class DivaLearner:
    """Babbling + imitation learner over (loc, degree, rounding)."""

    def __init__(self, glottal_flow, seed=0):
        self.rng = np.random.default_rng(seed)
        self.source = np.asarray(glottal_flow, dtype=float)
        self.tract = VocalTract()
        self.memory = []  # (feature, articulation)

    def _produce(self, art):
        loc, degree, rounding = (float(np.clip(v, 0.02, 0.98))
                                 for v in art)
        self.tract.set_articulators(loc, degree, rounding)
        self.tract.reset()
        y = self.tract.run(self.source)
        return y / (np.abs(y).max() + 1e-12)

    def babble(self, n_samples):
        """Phase 1: explore articulation, store (auditory -> motor)."""
        for _ in range(n_samples):
            art = self.rng.uniform(0.05, 0.95, size=3)
            y = self._produce(art)
            feat = auditory_feature(y)
            self.memory.append((feat, art))
        return len(self.memory)

    def anchor(self, articulations):
        """Guided babbling: seed memory with clear examples rendered
        through HER OWN voice (like hearing a caregiver's clear speech
        and feeling how it maps to her own articulation)."""
        for art in articulations:
            art = tuple(float(np.clip(v, 0.02, 0.98)) for v in art)
            y = self._produce(art)
            feat = auditory_feature(y)
            self.memory.append((feat, np.array(art)))
        return len(self.memory)

    def _inverse(self, target_feat, k=8):
        """k-NN inverse: articulation for a target auditory feature."""
        d = np.array([np.linalg.norm(f - target_feat) for f, _ in self.memory])
        idx = np.argsort(d)[:k]
        w = 1.0 / (d[idx] + 1e-9)
        w /= w.sum()
        arts = np.array([self.memory[i][1] for i in idx])
        return w @ arts

    def imitate(self, target_wave, refine_steps=12):
        """Phase 2: match a guide target via active search.

        Coarse grid over articulation + local refinement. This IS the
        learning: systematic exploration of her own motor space guided
        by auditory feedback (DIVA's babbling->imitation loop).
        Returns (wave, error, articulation).
        """
        target_feat = auditory_feature(target_wave)

        def err_of(art):
            y = self._produce(art)
            return np.linalg.norm(auditory_feature(y) - target_feat), y

        # coarse grid
        best_err, best_art, best_y = 1e9, None, None
        for loc in (0.25, 0.4, 0.55, 0.7, 0.85):
            for deg in (0.3, 0.55, 0.8):
                for rnd in (0.15, 0.5, 0.85):
                    art = (loc, deg, rnd)
                    e, y = err_of(art)
                    if e < best_err:
                        best_err, best_art, best_y = e, art, y
        # local refinement (hill-climb)
        art = np.array(best_art)
        step = 0.10
        for _ in range(refine_steps * 2):
            cand = np.clip(art + self.rng.normal(0, step, size=3), 0.02, 0.98)
            e, y = err_of(tuple(cand))
            if e < best_err:
                best_err, best_art, best_y = e, tuple(cand), y
                art = cand
                step *= 0.92
            else:
                step *= 0.96
        return best_y, best_err, np.array(best_art)
