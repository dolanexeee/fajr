"""Generate Fajr's voice and audio using the PROVEN configuration.

Uses the exact setup from the passing gate test (no FajrVoice class).
"""
import os
import json
import numpy as np
from scipy.io import wavfile

from voice.glottis import TwoMassGlottis
from voice.tract import VocalTract, HER_VOWELS
from voice.diva import auditory_feature
from voice.hearing import SR, f0_autocorr

VOICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "voice_state"))
os.makedirs(VOICE_DIR, exist_ok=True)

HER_WORDS = {
    "drink":   ("i", "a"),
    "eat":     ("a", "i"),
    "fruit":   ("a", "u"),
    "sleep":   ("u", "u"),
    "thirsty": ("i", "u"),
    "want":    ("a", "a"),
}

def make_source(tension, dur=0.5):
    g = TwoMassGlottis()
    g.set_tension(tension)
    g.reset()
    g.x1 = 1e-6
    s = g.run(dur)
    return s[int(0.2 * SR):]

def render_vowel(source, vowel, dur_s=0.30):
    t = VocalTract()
    t.set_vowel(vowel)
    t.reset()
    n = int(dur_s * SR)
    src = source[:n]  # never tile; use what we have
    y = t.run(src)
    return y / (np.abs(y).max() + 1e-12)

def render_art(source, art, dur_s=0.30):
    t = VocalTract()
    t.set_articulators(*art)
    t.reset()
    n = int(dur_s * SR)
    src = source[:n]  # never tile; use what we have
    y = t.run(src)
    return y / (np.abs(y).max() + 1e-12)

def main():
    # her voice (proven: tension 1.35 -> 177.8 Hz)
    src_her = make_source(1.35)
    f0_her = f0_autocorr(src_her)
    print(f"her f0: {f0_her:.1f} Hz", flush=True)
    # guide (proven: tension 0.72 -> 130.1 Hz)
    src_guide = make_source(0.72)
    f0_guide = f0_autocorr(src_guide)
    print(f"guide f0: {f0_guide:.1f} Hz", flush=True)

    # GATE ص2: her vowels recognized as guide targets
    refs = {v: auditory_feature(render_vowel(src_guide, v)) for v in ("a", "i", "u")}
    print("GATE ص2:", flush=True)
    ok2 = True
    for v in ("a", "i", "u"):
        f = auditory_feature(render_art(src_her, HER_VOWELS[v]))
        dists = {rv: float(np.linalg.norm(f - refs[rv])) for rv in refs}
        recog = min(dists, key=dists.get)
        ok = (recog == v)
        ok2 &= ok
        print(f"  her /{v}/ -> " + ", ".join(f"{k}={d:.2f}" for k, d in dists.items()) +
              f" => /{recog}/ [{'OK' if ok else 'FAIL'}]", flush=True)
    print(f"GATE ص2: {'PASS' if ok2 else 'FAIL'}", flush=True)

    def her_word(w):
        vowels = HER_WORDS[w]
        parts = [render_art(src_her, HER_VOWELS[v], 0.30) for v in vowels]
        gap = np.zeros(int(0.07 * SR))
        out = parts[0]
        for p in parts[1:]:
            out = np.concatenate([out, gap, p])
        return out / (np.abs(out).max() + 1e-12)

    # GATE ص3: her words recognized (segment-wise: each vowel in order)
    # Word structure: v1 (0.30s) + gap (0.07s) + v2 (0.30s); use exact bounds
    print("GATE ص3:", flush=True)
    ok3 = True
    vowel_refs = {v: auditory_feature(render_vowel(src_guide, v)) for v in ("a", "i", "u")}
    vlen = int(0.30 * SR)
    gaplen = int(0.07 * SR)
    for w in HER_WORDS:
        vowels = HER_WORDS[w]
        y = her_word(w)
        seg1 = y[:vlen]
        seg2 = y[vlen + gaplen:vlen + gaplen + vlen]
        ok_w = True
        for seg, v in ((seg1, vowels[0]), (seg2, vowels[1])):
            f = auditory_feature(seg / (np.abs(seg).max() + 1e-12))
            recog = min(vowel_refs, key=lambda rv: np.linalg.norm(f - vowel_refs[rv]))
            if recog != v:
                ok_w = False
        ok3 &= ok_w
        print(f"  her '{w}' {vowels} [{'OK' if ok_w else 'FAIL'}]", flush=True)
    print(f"GATE ص3: {'PASS' if ok3 else 'FAIL'}", flush=True)

    # GATE ص4: 2-word sentences, each word's vowels recognized in order
    def her_sentence(w1, w2):
        gap = np.zeros(int(0.20 * SR))
        out = np.concatenate([her_word(w1), gap, her_word(w2)])
        return out / (np.abs(out).max() + 1e-12)

    def check_word_seg(seg, w):
        vowels = HER_WORDS[w]
        s1 = seg[:vlen]
        s2 = seg[vlen + gaplen:vlen + gaplen + vlen]
        for s, v in ((s1, vowels[0]), (s2, vowels[1])):
            f = auditory_feature(s / (np.abs(s).max() + 1e-12))
            recog = min(vowel_refs, key=lambda rv: np.linalg.norm(f - vowel_refs[rv]))
            if recog != v:
                return False
        return True

    print("GATE ص4:", flush=True)
    ok4 = True
    wlen = vlen * 2 + gaplen  # one word length
    wordgap = int(0.20 * SR)
    for w1, w2 in [("want", "fruit"), ("drink", "thirsty")]:
        s = her_sentence(w1, w2)
        seg_w1 = s[:wlen]
        seg_w2 = s[wlen + wordgap:wlen + wordgap + wlen]
        ok_s = check_word_seg(seg_w1, w1) and check_word_seg(seg_w2, w2)
        ok4 &= ok_s
        print(f"  '{w1} {w2}' [{'OK' if ok_s else 'FAIL'}]", flush=True)
    print(f"GATE ص4: {'PASS' if ok4 else 'FAIL'}", flush=True)

    # save audio
    print("Saving audio:", flush=True)
    for w in HER_WORDS:
        path = os.path.join(VOICE_DIR, f"fajr_{w}.wav")
        wavfile.write(path, SR, (her_word(w) * 32767).astype(np.int16))
        print(f"  {path}", flush=True)
    for w1, w2 in [("want", "fruit"), ("drink", "thirsty")]:
        path = os.path.join(VOICE_DIR, f"fajr_{w1}_{w2}.wav")
        wavfile.write(path, SR, (her_sentence(w1, w2) * 32767).astype(np.int16))
        print(f"  {path}", flush=True)

    # save state
    state = {
        "f0_her": round(float(f0_her), 1),
        "f0_guide": round(float(f0_guide), 1),
        "gates": {"s2": bool(ok2), "s3": bool(ok3), "s4": bool(ok4)},
        "her_vowels": {k: list(v) for k, v in HER_VOWELS.items()},
        "her_words": HER_WORDS,
    }
    with open(os.path.join(VOICE_DIR, "fajr_voice.json"), "w") as f:
        json.dump(state, f, indent=1)
    print(f"ALL DONE. Gates: s2={ok2} s3={ok3} s4={ok4}", flush=True)

if __name__ == "__main__":
    main()
