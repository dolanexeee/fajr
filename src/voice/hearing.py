"""Acoustic hearing upgrade — voice station, SPEC addendum.

Replaces string-label sounds ("river", "thunder") with REAL acoustic
signals. Every heard event is a waveform; the auditory front-end is a
magnitude spectrogram the learning loop can close on.

Pure Python + numpy. Zero tokens at runtime.
"""
import numpy as np

SR = 16000  # sample rate, Hz


# ---------------------------------------------------------------- front-end
def stft_mag(x, n_fft=512, hop=128):
    """Magnitude spectrogram, shape (frames, n_fft//2+1)."""
    x = np.asarray(x, dtype=float)
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    win = np.hanning(n_fft)
    n_frames = 1 + (x.size - n_fft) // hop
    out = np.empty((n_frames, n_fft // 2 + 1))
    for i in range(n_frames):
        frame = x[i * hop:i * hop + n_fft] * win
        out[i] = np.abs(np.fft.rfft(frame))
    return out


def spectral_centroid(spec):
    """Mean spectral centroid across frames (Hz)."""
    freqs = np.fft.rfftfreq(512, 1.0 / SR)
    num = (spec * freqs).sum(axis=1)
    den = spec.sum(axis=1) + 1e-12
    return float(np.mean(num / den))


def f0_autocorr(x, fmin=60.0, fmax=600.0):
    """Fundamental frequency by autocorrelation, Hz. Returns 0 if unvoiced."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    if np.max(np.abs(x)) < 1e-9:
        return 0.0
    n = x.size
    corr = np.correlate(x, x, mode="full")[n - 1:]
    corr /= (corr[0] + 1e-12)
    lo = int(SR / fmax)
    hi = min(int(SR / fmin), n - 1)
    if hi <= lo:
        return 0.0
    peak = lo + int(np.argmax(corr[lo:hi]))
    if corr[peak] < 0.3:
        return 0.0
    return float(SR / peak)


# ------------------------------------------------------- event synthesizers
def synth_river(dur_s, rng):
    """River murmur: band-shaped noise, gentle amplitude wander."""
    n = int(dur_s * SR)
    noise = rng.standard_normal(n)
    # simple one-pole lowpass + highpass to shape into murmur band
    b = np.zeros(n)
    alpha = 0.12
    for i in range(1, n):
        b[i] = b[i - 1] + alpha * (noise[i] - b[i - 1])
    murmur = noise - b  # crude highpass of the lowpassed noise
    wander = 0.6 + 0.4 * np.sin(2 * np.pi * 0.4 * np.arange(n) / SR
                                + rng.uniform(0, 6.28))
    murmur = murmur * wander
    murmur /= (np.max(np.abs(murmur)) + 1e-9)
    return (0.35 * murmur).astype(float)


def synth_thunder(dur_s, rng):
    """Distant thunder: low rumble burst with slow decay."""
    n = int(dur_s * SR)
    noise = rng.standard_normal(n)
    b = np.zeros(n)
    alpha = 0.02  # heavy lowpass -> rumble
    for i in range(1, n):
        b[i] = b[i - 1] + alpha * (noise[i] - b[i - 1])
    t = np.arange(n) / SR
    env = np.exp(-t / (dur_s * 0.45)) * (1 - np.exp(-t / 0.08))
    out = b * env
    out /= (np.max(np.abs(out)) + 1e-9)
    return (0.6 * out).astype(float)


def synth_bird(dur_s, rng):
    """Morning bird: a few downward FM chirps."""
    n = int(dur_s * SR)
    out = np.zeros(n)
    t = np.arange(n) / SR
    n_chirps = int(rng.integers(2, 5))
    for _ in range(n_chirps):
        start = rng.uniform(0, max(0.01, dur_s - 0.25))
        f0 = rng.uniform(2800, 4200)
        f1 = rng.uniform(1800, 2600)
        dur = rng.uniform(0.08, 0.18)
        m = (t >= start) & (t < start + dur)
        tt = t[m] - start
        phase = 2 * np.pi * (f0 * tt + (f1 - f0) * tt ** 2 / (2 * dur))
        env = np.sin(np.pi * tt / dur) ** 2
        out[m] += 0.5 * np.sin(phase) * env
    out /= (np.max(np.abs(out)) + 1e-9)
    return (0.3 * out).astype(float)


EVENT_SYNTH = {
    "river": synth_river,
    "thunder": synth_thunder,
    "bird": synth_bird,
}


def hear_event(name, dur_s=1.0, seed=0):
    """Waveform for a named world event. Unknown events -> silence."""
    rng = np.random.default_rng(seed)
    synth = EVENT_SYNTH.get(name)
    if synth is None:
        return np.zeros(int(dur_s * SR))
    return synth(dur_s, rng)
