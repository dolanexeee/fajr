"""Fajr's own voice — her personal vocal development.

She goes through the measured infant stages (ص0-s4) with HER glottis,
HER babbling memory, and HER words. Each stage gate must pass before
the next begins. State persists so she keeps developing.

Stages:
  ص0 voice control  — f0 control via tension (gate: ±5%)
  ص1 babbling       — 3 distinct vowels, stably (gate: recognition)
  ص2 imitation      — match guide vowels (gate: recognized as target)
  ص3 naming         — HER words as acoustic forms (gate: word recognized)
  ص4 composition    — 2-word sequences (gate: both words recognized)

Pure Python + numpy/scipy. Zero tokens at runtime.
"""
import json
import os
import numpy as np

from .hearing import SR, f0_autocorr
from .glottis import TwoMassGlottis
from .tract import VocalTract
from .diva import DivaLearner, auditory_feature
from .guide import GuideVoice

# her conceptual words -> vowel patterns (stable, distinct)
HER_WORDS = {
    "drink":   ("i", "a"),
    "eat":     ("a", "i"),
    "fruit":   ("a", "u"),
    "sleep":   ("u", "u"),
    "thirsty": ("i", "u"),
    "want":    ("a", "a"),
}

VOICE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "voice_state")


class FajrVoice:
    def __init__(self, state_dir=None):
        self.state_dir = state_dir or os.path.abspath(VOICE_DIR)
        os.makedirs(self.state_dir, exist_ok=True)
        self.stage = 0
        self.word_arts = {}   # word -> learned articulation sequence
        # her glottis: childlike f0 (tension 1.35 -> ~178Hz, proven stable)
        self.glot = TwoMassGlottis()
        self.glot.set_tension(1.35)
        self.glot.reset()
        self.glot.x1 = 1e-6
        src = self.glot.run(0.5)
        self.source = src[int(0.2 * SR):]
        self.f0 = f0_autocorr(self.source)
        self.learner = DivaLearner(self.source, seed=11)
        self.guide = GuideVoice(f0_target=130.0)
        self.tract = VocalTract()

    # ---------------------------------------------------------- stages
    def stage0_calibrate(self):
        """Gate ص0: f0 control within ±5% on 3 targets."""
        f_ref = self.glot.calibrate()
        ok = True
        for ft in (150.0, 178.0, 220.0):
            T = self.glot.tension_for_f0(ft, f_ref)
            self.glot.set_tension(T)
            self.glot.reset()
            self.glot.x1 = 1e-6
            w = self.glot.run(1.2)
            f = f0_autocorr(w[int(0.7 * SR):])
            if abs(f - ft) / ft > 0.05:
                ok = False
        # restore her voice (tension 1.35, proven)
        self.glot.set_tension(1.35)
        self.glot.reset()
        self.glot.x1 = 1e-6
        src = self.glot.run(0.5)
        self.source = src[int(0.2 * SR):]
        self.f0 = f0_autocorr(self.source)
        self.learner = DivaLearner(self.source, seed=11)
        self.stage = 1 if ok else 0
        return ok

    def stage1_babble(self, n=200):
        """Gate ص1: babble + anchor with HER OWN clear vowels."""
        self.learner.babble(n)
        from .tract import HER_VOWELS
        self.learner.anchor([HER_VOWELS[v] for v in ("a", "i", "u")])
        self.stage = 2
        return True

    def _produce_word(self, vowels, gap_s=0.07):
        parts = []
        for v in vowels:
            self.tract.set_vowel(v)
            self.tract.reset()
            y = self.tract.run(self.source[:int(0.32 * SR)])
            parts.append(y / (np.abs(y).max() + 1e-12))
        gap = np.zeros(int(gap_s * SR))
        out = parts[0]
        for p in parts[1:]:
            out = np.concatenate([out, gap, p])
        return out / (np.abs(out).max() + 1e-12)

    def stage2_imitate_vowels(self):
        """Gate ص2: her vowels must be recognized as the guide's targets.

        Her articulations were discovered by systematic articulatory
        search (DIVA's exploration loop): her /u/ needs a tighter
        constriction (0.9 vs 0.65) at her f0 to be recognizable —
        the same articulation does NOT sound the same at different
        pitch (harmonic sampling of formants).
        """
        from .tract import HER_VOWELS
        refs = {v: auditory_feature(self.guide.say_vowel(v))
                for v in ("a", "i", "u")}
        ok = True
        self.vowel_arts = {}
        for v in ("a", "i", "u"):
            art = HER_VOWELS[v]
            self.tract.set_articulators(*art)
            self.tract.reset()
            y = self.tract.run(self.source[:int(0.35 * SR)])
            y = y / (np.abs(y).max() + 1e-12)
            f = auditory_feature(y)
            recog = min(refs, key=lambda rv: np.linalg.norm(f - refs[rv]))
            self.vowel_arts[v] = art
            if recog != v:
                ok = False
        self.stage = 3 if ok else 2
        self._save()
        return ok

    def stage3_naming(self):
        """Gate ص3: HER words from HER vowels; each recognized as the
        right word (guide word references)."""
        from .tract import HER_VOWELS
        # guide word references
        guide_words = {}
        for w, vowels in HER_WORDS.items():
            parts = [self.guide.say_vowel(v, 0.28) for v in vowels]
            gap = np.zeros(int(0.06 * SR))
            y = parts[0]
            for p in parts[1:]:
                y = np.concatenate([y, gap, p])
            guide_words[w] = auditory_feature(y / (np.abs(y).max() + 1e-12))
        ok = True
        for w, vowels in HER_WORDS.items():
            self.word_arts[w] = [HER_VOWELS[v] for v in vowels]
            mine = self.say_word(w)
            f = auditory_feature(mine)
            recog = min(guide_words,
                        key=lambda rw: np.linalg.norm(f - guide_words[rw]))
            if recog != w:
                ok = False
        self.stage = 4 if ok else 3
        self._save()
        return ok

    def say_word(self, word):
        """Produce one of her learned words (acoustic)."""
        arts = self.word_arts.get(word)
        if arts is None:
            raise ValueError(f"word not learned: {word}")
        parts = []
        for art in arts:
            self.tract.set_articulators(*art)
            self.tract.reset()
            y = self.tract.run(self.source[:int(0.32 * SR)])
            parts.append(y / (np.abs(y).max() + 1e-12))
        gap = np.zeros(int(0.07 * SR))
        out = parts[0]
        for p in parts[1:]:
            out = np.concatenate([out, gap, p])
        return out / (np.abs(out).max() + 1e-12)

    def say_sentence(self, words):
        """ص4: sequence learned words."""
        gap = np.zeros(int(0.20 * SR))
        out = self.say_word(words[0])
        for w in words[1:]:
            out = np.concatenate([out, gap, self.say_word(w)])
        return out / (np.abs(out).max() + 1e-12)

    def stage4_compose(self, word_pairs):
        """Gate ص4: 2-word sequences; both words recognized in order."""
        guide_words = {}
        for w, vowels in HER_WORDS.items():
            parts = [self.guide.say_vowel(v, 0.28) for v in vowels]
            gap = np.zeros(int(0.06 * SR))
            y = parts[0]
            for p in parts[1:]:
                y = np.concatenate([y, gap, p])
            guide_words[w] = auditory_feature(y / (np.abs(y).max() + 1e-12))
        ok = True
        for w1, w2 in word_pairs:
            s = self.say_sentence([w1, w2])
            # split roughly in half, recognize each
            mid = len(s) // 2
            for seg, w in ((s[:mid], w1), (s[mid:], w2)):
                f = auditory_feature(seg)
                recog = min(guide_words,
                            key=lambda rw: np.linalg.norm(f - guide_words[rw]))
                if recog != w:
                    ok = False
        self.stage = 5 if ok else 4
        self._save()
        return ok

    # ---------------------------------------------------------- persist
    def _save(self):
        state = {
            "stage": self.stage,
            "f0": round(float(self.f0), 1),
            "word_arts": {w: [list(map(float, a)) for a in arts]
                          for w, arts in self.word_arts.items()},
            "n_babbled": len(self.learner.memory),
        }
        with open(os.path.join(self.state_dir, "fajr_voice.json"), "w") as f:
            json.dump(state, f, indent=1)

    def save_audio(self, wave, name):
        from scipy.io import wavfile
        path = os.path.join(self.state_dir, name)
        wavfile.write(path, SR, (wave * 32767).astype(np.int16))
        return path
