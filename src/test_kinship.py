# -*- coding: utf-8 -*-
"""Tests for ADAM & EVE P1 — kinship teaching. Gates K1-K5."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lexicon import WORDS, word_surface
from dreams import THEME_FRAGMENTS, RECOUNT_TEMPLATES, RECOUNT_WORDS, KIND_THEME
from language import Tongue
from kinship import KINSHIP_WORDS, KINSHIP_LINES, teach_kinship


def test_k1_lexicon():
    assert KINSHIP_WORDS == ["pair", "family", "newlife"]
    assert word_surface("pair") == "زوج"
    assert word_surface("family") == "عائلة"
    assert word_surface("newlife") == "حياة جديدة"
    for wid in KINSHIP_WORDS:
        assert WORDS[wid]["pos"] == "noun", wid
        assert wid in KINSHIP_LINES, wid
    print("K1 PASS: 3 kinship words in lexicon with Guide lines")


class _FakeEpisodic:
    def __init__(self):
        self.sem = {}

    def set_semantic(self, key, val, tick):
        self.sem[key] = val


class _FakeMind:
    def __init__(self):
        self.episodic = _FakeEpisodic()
        self._kinship_ready = False


class _FakeFajr:
    def __init__(self):
        self.x, self.y = 100.0, 50.0
        self.mind = _FakeMind()


class _FakeJournal:
    def __init__(self):
        self.entries = []

    def log(self, tick, type_, text, x=None, y=None):
        self.entries.append({"type": type_, "text": text})


class _FakeSim:
    def __init__(self):
        self.tick = 12345
        self.journal = _FakeJournal()
        self.fajr = _FakeFajr()


class _FakeConv:
    def __init__(self):
        self.sim = _FakeSim()
        self.tongue = Tongue()
        self.lines = []

    def _log_line(self, who, text):
        self.lines.append((who, text))


def test_k2_teach_kinship():
    conv = _FakeConv()
    taught = teach_kinship(conv)
    assert taught == ["pair", "family", "newlife"], taught
    # words enter her vocabulary through the normal learn path
    for wid in taught:
        assert conv.tongue.known(wid), f"{wid} not known after teaching"
        assert conv.sim.fajr.mind.episodic.sem.get(f"word:{wid}", 0) > 0
    # journal records the Guide's teaching, behaviorally
    teach_entries = [e for e in conv.sim.journal.entries
                     if e["type"] == "teaching"]
    assert len(teach_entries) == 3, len(teach_entries)
    assert all("علمها المضيف" in e["text"] for e in teach_entries)
    # behavioral marker set
    assert conv.sim.fajr.mind._kinship_ready is True
    assert any(e["type"] == "readiness"
               for e in conv.sim.journal.entries)
    # transcript lines are the Guide's voice
    assert [w for w, _ in conv.lines] == ["host"] * 3
    print("K2 PASS: teach_kinship — 3 words known, journal honest, ready flag set")


def test_k3_dream_theme():
    assert "kinship" in THEME_FRAGMENTS
    assert "kinship" in RECOUNT_TEMPLATES
    assert RECOUNT_WORDS["kinship"] == ("dream", "pair", "family")
    assert KIND_THEME["union"] == "kinship"
    # recount only needs words she can know via P1 teaching
    print("K3 PASS: kinship dream theme + recount wired")


def test_k4_conversation_persist():
    from sim import Sim
    from conversation import Conversation
    sim = Sim(seed=11)
    c = Conversation(sim)
    assert c._kinship_taught is False  # fresh: not taught
    c._kinship_taught = True
    d = c.to_dict()
    assert d["kinship_taught"] is True
    c2 = Conversation.from_dict(sim, d)
    assert c2._kinship_taught is True
    # old checkpoints without the key default to False (no crash)
    d2 = dict(d)
    del d2["kinship_taught"]
    c3 = Conversation.from_dict(sim, d2)
    assert c3._kinship_taught is False
    print("K4 PASS: kinship_taught round-trips; old checkpoints default False")


def test_k5_mind_persist():
    from prng import PRNG
    from mind import Mind
    rng = PRNG(12345)
    m = Mind(rng)
    assert m._kinship_ready is False  # fresh: not ready
    m._kinship_ready = True
    d = m.to_dict()
    assert d["kinship_ready"] is True
    m2 = Mind.from_dict(d, PRNG(999))
    assert m2._kinship_ready is True
    # old checkpoints without the key default to False (no crash)
    d2 = dict(d)
    del d2["kinship_ready"]
    m3 = Mind.from_dict(d2, PRNG(999))
    assert m3._kinship_ready is False
    print("K5 PASS: kinship_ready round-trips; old checkpoints default False")


if __name__ == "__main__":
    test_k1_lexicon()
    test_k2_teach_kinship()
    test_k3_dream_theme()
    test_k4_conversation_persist()
    test_k5_mind_persist()
    print("ALL P1 KINSHIP TESTS PASS (K1-K5)")
