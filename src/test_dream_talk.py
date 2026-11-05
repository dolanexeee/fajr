"""Conversation-level dream gates (Phase 4, stage 1).

Fake sim driving Conversation._dream_tick — never the live biography.
  C1 host names the first dream: teaches "dream", journal gets teaching,
     dream marked named, unrecounted cleared.
  C2 second dream: she recounts it herself with context "dream",
     claim type "dream" (never memory/fact), journal gets dream_recount.
  C3 recount uses only known words: if a needed word is unknown, no
     utterance is emitted (grounding gate), dream still cleared.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conversation import Conversation
from dreams import DreamWeaver

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


class FakeJournal:
    def __init__(self):
        self.entries = []

    def log(self, tick, type_, text, x=None, y=None):
        self.entries.append({"tick": tick, "type": type_, "text": text})


class FakeEpisodic:
    def set_semantic(self, *a):
        pass

    def record(self, *a, **k):
        pass


class FakeMind:
    def __init__(self):
        self.dreamer = DreamWeaver()
        self.episodic = FakeEpisodic()
        self.affect = type("A", (), {"valence": 0.5, "arousal": 0.3})()


class FakeFajr:
    def __init__(self):
        self.asleep = False
        self.x, self.y = 100.0, 100.0
        self.stats = {}
        self.mind = FakeMind()
        self._raw = None

    class _D:
        d = {"hunger": 10.0, "thirst": 10.0, "sleep": 10.0, "safety": 10.0}
    drives = _D()


class FakeSim:
    def __init__(self):
        self.tick = 9000
        self.fajr = FakeFajr()
        self.journal = FakeJournal()


_ARM_ID = [100]

def arm_dream(conv, theme="swim", vivid=0.8):
    _ARM_ID[0] += 1
    did = _ARM_ID[0]
    d = conv.sim.fajr.mind.dreamer
    dream = {"id": did, "tick": 8000, "day": 6, "type": "dream",
             "sources": [7000, 7100], "theme": theme, "vividness": vivid,
             "journal_text": "حلم — test", "recounted": False, "named": False}
    d.dreams.append(dream)
    d.unrecounted_id = did
    return dream


# C1: first dream -> host teaches the word
sim = FakeSim()
conv = Conversation(sim)
dream = arm_dream(conv)
check("C1-word-unknown-first", not conv.tongue.known("dream"))
conv._dream_tick()
check("C1-word-taught", conv.tongue.known("dream"))
check("C1-teaching-journaled",
      any(e["type"] == "teaching" and "حلم" in e["text"] for e in sim.journal.entries))
check("C1-dream-named", dream["named"] is True)
check("C1-unrecounted-cleared", conv.sim.fajr.mind.dreamer.unrecounted_id is None)
check("C1-no-self-recount-yet",
      not any(e["type"] == "dream_recount" for e in sim.journal.entries))

# C2: second dream -> she recounts it herself
dream2 = arm_dream(conv, theme="swim")
# teach the words the swim recount needs (she learned them long ago live)
for w in ("swim", "river"):
    conv.tongue.learn(w, sim.tick, gain=1.0, anchored=True, source="test")
conv._dream_tick()
recounts = [e for e in sim.journal.entries if e["type"] == "dream_recount"]
check("C2-recount-journaled", len(recounts) == 1, f"found {len(recounts)}")
u = conv.tongue.utterances[-1]
check("C2-context-dream", u["context"] == "dream")
check("C2-claim-typed-dream",
      u["claims"] and u["claims"][0]["type"] == "dream",
      f"claims={u['claims']}")
check("C2-no-memory-claim",
      not any(c.get("type") == "memory" for c in u["claims"]))
check("C2-text-arabic", "حلمت" in u["text"], f"text={u['text']}")
check("C2-marked-recounted", dream2["recounted"] is True)

# C3: grounding gate — unknown word blocks the utterance
sim3 = FakeSim()
conv3 = Conversation(sim3)
conv3.tongue.learn("dream", sim3.tick, gain=1.0, anchored=True, source="test")
dream3 = arm_dream(conv3, theme="eat")  # needs eat/fruit/much — unknown here
n_before = len(conv3.tongue.utterances)
conv3._dream_tick()
check("C3-no-utterance-when-ungrounded",
      len(conv3.tongue.utterances) == n_before)
check("C3-dream-cleared-anyway",
      conv3.sim.fajr.mind.dreamer.unrecounted_id is None)

if FAILS:
    print(f"\n{len(FAILS)} FAILURES: {FAILS}")
    sys.exit(1)
print("\nALL CONVERSATION DREAM GATES PASS")
