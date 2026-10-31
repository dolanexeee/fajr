"""A demonstration life (seed 7 — NOT a battery seed): 3 accompanied
days, then a scripted conversation with the Host. Every utterance she
produces is grounding-verified at its own tick by the battery's
Verifier. Writes results/sample_conversation.json and prints the
transcript. Run: python3 tests/sample_conversation.py
"""
import json
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "tests")

from sim import Sim
from conversation import Conversation
from phase3_battery import Verifier, wait_until

sim = Sim(seed=7)
conv = Conversation(sim)
ver = Verifier(sim, conv)


def run(n):
    for _ in range(n):
        sim.step()
        conv.step()
        ver.poll()


run(3 * 1440)

SCRIPT = [
    "مرحبًا فجر",
    "ما اسمك؟",
    "من أنا؟",
    "هل أنت جائعة؟",
    "هل الشجرة فيها فاكهة؟",
    "أين الشجرة؟",
    "ما هذه؟",
    "ماذا أكلت؟",
    "هل رأيت القمر؟",
    "اشربي",
    "اذهبي إلى المأوى",
]
exchanges = []
for line in SCRIPT:
    wait_until(sim, conv, ver, lambda: not sim.fajr.asleep, 1800)
    before = len(conv.tongue.utterances)
    u = conv.exchange(line)
    ver.poll()
    exchanges.append({"host": line,
                        "reply": u["text"] if u else None,
                        "claims": u["claims"] if u else []})
    run(30)

out = {"seed": 7, "note": "demonstration life — not a battery seed",
       "transcript": conv.transcript,
       "exchanges": exchanges,
       "verified_utterances": ver.records,
       "journal": [e for e in sim.journal.entries
                   if e["type"] in ("host_says", "utterance", "teaching",
                                    "request")],
       "outbox": conv.outbox,
       "known_words": sorted(w for w in conv.tongue.words
                             if conv.tongue.known(w))}
with open("results/sample_conversation.json", "w") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
bad = [r for r in ver.records if not all(c["faithful"] for c in r["checks"])]
print(f"utterances verified: {len(ver.records)}, unfaithful: {len(bad)}")
for e in exchanges:
    print("HOST:", e["host"])
    print("FAJR:", e["reply"])
print("wrote results/sample_conversation.json")
