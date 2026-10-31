"""Guide voice — the adult voice Fajr imitates (DIVA phase 2).

The guide renders Fajr's OWN template utterances (from language.py) with
a stable adult voice (low f0, canonical vowels). She learns to imitate
HER language, not an imported one — per the preflight design.

Pure Python + numpy. Zero tokens at runtime.
"""
import numpy as np

from .hearing import SR
from .glottis import TwoMassGlottis
from .tract import VocalTract

# Fajr's template words -> vowel sequences for the guide.
# (Her lexicon is small; each word gets a stable vowel pattern.)
WORD_VOWELS = {
    "fruit": ("a", "u"),
    "river": ("i", "a"),
    "shelter": ("u", "a"),
    "hungry": ("a", "i"),
    "thirsty": ("i", "u"),
    "tired": ("u", "i"),
    "fajr": ("a", "i"),   # her own name
}


class GuideVoice:
    """Stable adult voice for imitation targets."""

    def __init__(self, f0_target=130.0):
        self.glot = TwoMassGlottis()
        # direct tension (proven stable); f0_target ~130 -> tension ~0.72
        # calibrated: f0=153.8Hz at T=1, so T=(f0_target/153.8)^2
        T = (f0_target / 153.8) ** 2
        self.glot.set_tension(T)
        self.glot.reset()
        self.glot.x1 = 1e-6
        src = self.glot.run(0.5)
        self.source = src[int(0.2 * SR):]
        self.tract = VocalTract()

    def say_vowel(self, vowel, dur_s=0.35):
        n = int(dur_s * SR)
        src = self.source[:n] if self.source.size >= n else np.resize(self.source, n)
        self.tract.set_vowel(vowel)
        self.tract.reset()
        y = self.tract.run(src)
        return y / (np.abs(y).max() + 1e-12)

    def say_word(self, word):
        """Render one of Fajr's words as a vowel sequence."""
        vowels = WORD_VOWELS.get(word, ("a",))
        parts = [self.say_vowel(v, 0.30) for v in vowels]
        gap = np.zeros(int(0.06 * SR))
        out = parts[0]
        for p in parts[1:]:
            out = np.concatenate([out, gap, p])
        return out / (np.abs(out).max() + 1e-12)

    def say_sentence(self, words):
        """Sequence of words with pauses (composition target)."""
        gap = np.zeros(int(0.18 * SR))
        out = self.say_word(words[0])
        for w in words[1:]:
            out = np.concatenate([out, gap, self.say_word(w)])
        return out / (np.abs(out).max() + 1e-12)
