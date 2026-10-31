"""Fine-tune human vowels with proper optimization (differential evolution)."""
import numpy as np
import sys
sys.path.insert(0, '/home/hatch/workspace/projects/alsakin-inhabitant/src')
from scipy.signal import find_peaks
from scipy.optimize import differential_evolution
from voice.tract import VocalTract, _areas_from_articulators

HUMAN_N = 20
# P&B women: F1, F2, F3
TARGETS = {
    'a': (850, 1220, 2810),
    'i': (310, 2790, 3310),
    'u': (370, 950, 2950),
}

def formants3(art, n=HUMAN_N):
    t = VocalTract(n_sec=n)
    t.set_areas(_areas_from_articulators(*art, n=n))
    t.reset()
    imp = np.zeros(2048); imp[0] = 1.0
    h = t._run48(imp)[:2048]
    H = np.abs(np.fft.rfft(h * np.hanning(len(h)), 8192))
    freqs = np.fft.rfftfreq(8192, 1 / 48000)
    peaks, _ = find_peaks(20 * np.log10(H + 1e-9), distance=10, prominence=1.5)
    return [float(freqs[p]) for p in peaks if freqs[p] < 4000][:3]

def make_loss(tgt):
    def loss(x):
        loc, deg, rnd = x
        # penalize degenerate (near-closed) tracts
        a = _areas_from_articulators(loc, deg, rnd, HUMAN_N)
        if a.min() < 0.3:
            return 1e3 + (0.3 - a.min()) * 1e3
        f = formants3((loc, deg, rnd))
        if len(f) < 3:
            return 1e9
        return sum(abs(f[i] - tgt[i]) / tgt[i] * w
                   for i, w in enumerate([3.0, 2.0, 1.0]))
    return loss

if __name__ == '__main__':
    import json
    results = {}
    for v, tgt in TARGETS.items():
        print(f'optimizing /{v}/ target {tgt}...', flush=True)
        res = differential_evolution(
            make_loss(tgt),
            bounds=[(0.05, 0.95), (0.1, 0.98), (0.0, 1.0)],
            maxiter=40, popsize=12, seed=7, tol=1e-4,
            workers=1, updating='deferred', disp=False)
        art = tuple(float(x) for x in res.x)
        f = formants3(art)
        results[v] = {'art': art, 'formants': [round(x) for x in f],
                      'loss': round(float(res.fun), 4)}
        print(f'  /{v}/ art={tuple(round(x,3) for x in art)} '
              f'F={[round(x) for x in f]} loss={res.fun:.4f}', flush=True)
    json.dump(results, open('/tmp/human_vowels_v2.json', 'w'), indent=1)
    print('saved')
