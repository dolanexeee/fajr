"""Find human-like vowel articulations for a human-scale tract.

Tract: 20 sections x 0.73cm = 14.6cm (young female).
Targets: Peterson & Barney adult female formants.
"""
import numpy as np
import sys
sys.path.insert(0, '/home/hatch/workspace/projects/alsakin-inhabitant/src')
from scipy.signal import find_peaks
from voice.tract import VocalTract, _areas_from_articulators

HUMAN_N = 20
TARGETS = {
    'a': (850, 1220),
    'i': (310, 2790),
    'u': (370, 950),
}

def formants_human(art, n=HUMAN_N):
    t = VocalTract(n_sec=n)
    t.set_areas(_areas_from_articulators(*art, n=n))
    t.reset()
    imp = np.zeros(2048); imp[0] = 1.0
    h = t._run48(imp)[:2048]
    H = np.abs(np.fft.rfft(h * np.hanning(len(h)), 8192))
    freqs = np.fft.rfftfreq(8192, 1 / 48000)
    peaks, _ = find_peaks(20 * np.log10(H + 1e-9), distance=10, prominence=1.5)
    form = [float(freqs[p]) for p in peaks if freqs[p] < 4000][:3]
    return form

def score(form, target):
    if len(form) < 2:
        return 1e9
    # weighted: F1 matters most, then F2
    return abs(form[0] - target[0]) / target[0] * 2.0 + \
           abs(form[1] - target[1]) / target[1] * 1.0

if __name__ == '__main__':
    import json
    results = {}
    for v, tgt in TARGETS.items():
        best = (1e9, None, None)
        for loc in np.linspace(0.15, 0.9, 10):
            for deg in np.linspace(0.2, 0.95, 10):
                for rnd in np.linspace(0.0, 1.0, 6):
                    art = (float(loc), float(deg), float(rnd))
                    f = formants_human(art)
                    s = score(f, tgt)
                    if s < best[0]:
                        best = (s, art, f)
        s, art, f = best
        results[v] = {'art': art, 'formants': [round(x) for x in f], 'score': round(s, 3)}
        print(f"/{v}/ target F1={tgt[0]} F2={tgt[1]} -> art={tuple(round(x,2) for x in art)} "
              f"F={[round(x) for x in f]} score={s:.3f}", flush=True)
    json.dump(results, open('/tmp/human_vowels.json', 'w'), indent=1)
    print('saved')
