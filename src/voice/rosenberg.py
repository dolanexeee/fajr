"""Rosenberg-style smooth glottal pulse source — the pleasant alternative.

Why: the two-mass model's abrupt closure slams high-frequency energy into
the tract (harsh buzz). A C1-smooth pulse (raised-cosine rise/fall) has a
gentle spectral tilt — warm, not harsh — while keeping f0 and formants
intact. Fully mechanical: parametric equations, no recordings, no TTS.

Reference: Rosenberg (1971) parametric glottal flow.
"""
import numpy as np

SR = 16000


def rosenberg_train(f0_start=185.0, f0_end=165.0, dur_s=0.5,
                    oq=0.62, sq=1.8, seed=0):
    """Smooth glottal FLOW pulse train with continuous f0 glide.

    f0 glides linearly from f0_start to f0_end (no chunking — each period
    is generated with its own smoothly-varying T0, so no discontinuities).
    oq = open quotient (fraction of period with airflow)
    sq = speed quotient (opening time / closing time)
    Returns flow in m^3/s (SI), like TwoMassGlottis.
    """
    n = int(dur_s * SR)
    out = np.zeros(n)
    t = 0.0
    while t < dur_s:
        frac = t / dur_s
        f0 = f0_start + (f0_end - f0_start) * frac
        T0 = 1.0 / f0
        t_open = oq * T0 * sq / (1 + sq)
        t_close = oq * T0 / (1 + sq)
        i0 = int(t * SR)
        i1 = min(n, int((t + t_open + t_close) * SR))
        if i1 <= i0:
            break
        tt = (np.arange(i0, i1) - i0) / SR
        pulse = np.zeros_like(tt)
        m_open = tt < t_open
        m_close = ~m_open
        # raised-cosine rise (zero slope at both ends)
        pulse[m_open] = 0.5 * (1 - np.cos(np.pi * tt[m_open] / t_open))
        tc = tt[m_close] - t_open
        # raised-cosine fall
        pulse[m_close] = 0.5 * (1 + np.cos(np.pi * tc / t_close))
        # physiological peak flow scale (~2e-4 m^3/s for soft voice)
        out[i0:i1] += pulse * 2.0e-4
        t += T0
    return out


def spectral_tilt(y, sr=SR, f_lo=500.0, f_hi=4000.0):
    """Rough spectral tilt in dB/octave between f_lo and f_hi."""
    H = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    m = (freqs >= f_lo) & (freqs <= f_hi) & (H > 0)
    x = np.log2(freqs[m])
    yy = 20 * np.log10(H[m])
    slope = np.polyfit(x, yy, 1)[0]
    return float(slope)
