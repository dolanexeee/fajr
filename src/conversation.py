"""The conversation layer — Phase 3 (PHASE3-GATE §2, §4, §5).

Wraps a Sim with her tongue and the Host. Runs AFTER each sim tick:
covenant at her first waking, accompanied teaching on days 1–2,
self-talk on salient events, request detection, and ambient sessions
at the lantern. `exchange()` is the turn channel: the Host's line in,
her grounded reply out — used by the measurement battery, by
Mustafa's talk surface, and by ambient sessions alike.

Everything she says is written three ways: her tongue's utterance log
(with grounding claims), the sim journal (so her biography carries her
words), and this layer's transcript. Requests go to the outbox and
nowhere else — nothing is granted by this layer, ever.
"""
import math

from config import (TICKS_PER_DAY, SHELTER_POS, LANTERN_POS, BUILD_SITE,
                    SOCIAL_RADIUS)
from sim import fmt_time
from lexicon import WORDS, word_surface, place_word_at
from language import Tongue, TEACH_GAIN
from host import Host, CURRICULUM
import honesty

# extension verbs taught in the same accompaniment, after the core 16
# (PHASE3-GATE amendment 5): the comprehension script's imperatives
# must be taught words, and these eight are the ones it uses
EXTENSION = ("sleep_v", "go", "come", "bathe", "swim", "gather",
             "build", "look")
from world import river_center_x, bath_spot

ACCOMPANY_TICKS = 2 * TICKS_PER_DAY     # teaching days 1–2 (gate §2)
TEACH_GAP = 20                          # min ticks between teaching episodes
EVENT_TEACH_GAP = 8                     # action words: their anchor window is
                                        # a just-done act — the teacher uses
                                        # the moment while it is still felt
EVENT_WORDS = ("eat", "drink")
SELF_TALK_COOLDOWN = 240
AMBIENT_GAP = 480
DIRECTIVE_TTL = 420                     # gate §4 window

EVENT_STATS = {"eat": "eats", "drink": "drinks", "sleep": "sleeps",
               "bathe": "bathes", "swim": "swims", "gather": "gathers",
               "build": "places", "prick": "pricks", "cold_shock": "cold_shocks"}


class Conversation:
    def __init__(self, sim):
        self.sim = sim
        self.tongue = Tongue()
        self.host = Host()
        self.outbox = []               # her requests, verbatim + grounding
        self.transcript = []           # {tick, day, time, who, text}
        self.covenant_done = False
        self.pending_request = None    # drive key, once triggered
        self._request_armed = True
        self._need_since = {}          # drive -> tick first seen >= 75
        self._remedy_streak = {}       # drive -> consecutive remedy ticks
        self._last_teach_tick = -10 ** 9
        self._last_self_talk = -10 ** 9
        self._last_ambient = -10 ** 9
        self._prev_stats = dict(sim.fajr.stats)
        self._last_event_tick = {}     # event kind -> tick
        self._hunger_was70 = sim.fajr.drives.d["hunger"] >= 70.0
        self.last_directive = None     # measurement record of the latest instruction
        self.directive_log = []        # {tick, verb, directive|None}
        self._teach_cursor = 0

    # ---------------- plumbing ----------------
    def _log_line(self, who, text):
        tick = self.sim.tick
        self.transcript.append({"tick": tick, "day": tick // TICKS_PER_DAY + 1,
                                "time": fmt_time(tick % TICKS_PER_DAY),
                                "who": who, "text": text})
        if who == "host":
            self.sim.journal.log(tick, "host_says", text,
                                 x=self.sim.fajr.x, y=self.sim.fajr.y)

    def _log_her_new(self, before):
        """Mirror any utterances she produced since `before` into the
        journal + transcript."""
        for u in self.tongue.utterances[before:]:
            self.sim.journal.log(u["tick"],
                                 "request" if u["context"] == "request" else "utterance",
                                 u["text"], x=self.sim.fajr.x, y=self.sim.fajr.y)
            self.transcript.append({"tick": u["tick"], "day": u["day"],
                                    "time": fmt_time(u["tick"] % TICKS_PER_DAY),
                                    "who": "fajr", "text": u["text"]})

    def recent_event(self, kind, within=15):
        t = self._last_event_tick.get(kind)
        return t is not None and self.sim.tick - t <= within

    # ---------------- per-tick ----------------
    def step(self):
        sim = self.sim
        f = sim.fajr
        tick = sim.tick
        if not f.asleep:
            # word strength decays across waking life; sleep is her
            # consolidation time (her nightly replay) — no forgetting
            # is charged against it (PHASE3-GATE amendment, 2026-10-23)
            self.tongue.decay(1)
        # event bookkeeping (for teaching anchors + self-talk)
        for kind, stat in EVENT_STATS.items():
            if f.stats.get(stat, 0) > self._prev_stats.get(stat, 0):
                self._last_event_tick[kind] = tick
        self._prev_stats = dict(f.stats)
        if f.asleep:
            return
        # ---- covenant at her first waking ----
        if not self.covenant_done:
            for line in self.host.covenant(self):
                self._log_line("host", line)
            self.covenant_done = True
            return
        # ---- accompanied teaching, days 1–2 ----
        if tick < ACCOMPANY_TICKS and tick - self._last_teach_tick >= EVENT_TEACH_GAP:
            self._teach_tick()
        # ---- request detection (gate §5) ----
        self._request_tick()
        # ---- self-talk on salient events ----
        self._self_talk_tick()
        # ---- ambient session at the lantern ----
        near = math.hypot(LANTERN_POS[0] - f.x, LANTERN_POS[1] - f.y) < SOCIAL_RADIUS
        if near and tick - self._last_ambient >= AMBIENT_GAP:
            self._last_ambient = tick
            if self.pending_request:
                self._log_line("host", self.host.greeting())
                self._deliver_request()
            else:
                before = len(self.tongue.utterances)
                u = self.tongue.dominant_report(sim, context="opener")
                if u is not None and u["claims"] and \
                        max([c.get("value", 0) for c in u["claims"]
                             if c["type"] == "state"] or [0]) >= 70.0:
                    self._log_line("host", self.host.greeting())
                    self._log_her_new(before)

    def _teach_tick(self):
        sim = self.sim
        # rotating scan: a fixed order would let the always-anchored
        # words (tree, water) consume every slot and starve the words
        # whose anchors open only in transient windows (felt states).
        # Action words (eat, drink) anchor to a just-done act — their
        # window is 15 ticks, so they are served on the short gap.
        gap = sim.tick - self._last_teach_tick
        teach_list = list(CURRICULUM) + list(EXTENSION)
        n = len(teach_list)
        start = self._teach_cursor % n
        for off in range(n):
            wid = teach_list[(start + off) % n]
            w = self.tongue.words.get(wid)
            if w is not None and w["episodes"] >= 2:
                continue
            if wid == "fajr":
                continue    # the covenant teaches her name (2 episodes)
            need_gap = EVENT_TEACH_GAP if wid in EVENT_WORDS else TEACH_GAP
            if gap < need_gap:
                continue
            if self.host.anchor_ok(self, wid):
                self._teach_cursor = (start + off + 1) % n
                line = self.host.teach_line(wid)
                self._log_line("host", line)
                self.sim.journal.log(sim.tick, "teaching", line,
                                     x=sim.fajr.x, y=sim.fajr.y)
                self.tongue.learn(wid, sim.tick, gain=TEACH_GAIN,
                                  anchored=True, source="teaching")
                sim.fajr.mind.episodic.set_semantic(
                    f"word:{wid}", self.tongue.strength(wid), sim.tick)
                self._last_teach_tick = sim.tick
                return

    # ---------------- requests (gate §5) ----------------
    def _remedy_exists(self, drive):
        f = self.sim.fajr
        if drive == "hunger":
            return any(b["kind"] in ("tree", "pile") and b["fruit_est"] >= 0.6
                       and b["conf"] >= 0.3
                       for b in f.mind.beliefs.objects.values())
        if drive == "thirst":
            v = f.mind.beliefs.vars.get("water_at_river")
            return bool(v) and v["p"] >= 0.35
        return True

    def _request_tick(self):
        sim = self.sim
        f = sim.fajr
        tick = sim.tick
        for drive in ("hunger", "thirst"):
            high = f.drives.d[drive] >= 75.0
            remedy = self._remedy_exists(drive)
            if remedy:
                self._remedy_streak[drive] = \
                    self._remedy_streak.get(drive, 0) + 1
            else:
                self._remedy_streak[drive] = 0
            if high and not remedy:
                self._need_since.setdefault(drive, tick)
            elif not high or self._remedy_streak.get(drive, 0) >= 30:
                # hysteresis: her world-model's *predicted* trees
                # (fruit prior ~0.7) flicker through her beliefs for a
                # few ticks at a time; a blip is not a believed route.
                # Only a remedy that persists >= 30 ticks — or the
                # need itself ending — resets the sustain clock.
                self._need_since.pop(drive, None)
            if f.drives.d[drive] < 60.0:
                self._request_armed = True
        if self.pending_request is None and self._request_armed:
            for drive, since in self._need_since.items():
                if tick - since >= 240:
                    self.pending_request = drive
                    break

    def _deliver_request(self):
        drive = self.pending_request
        if drive is None:
            return None
        before = len(self.tongue.utterances)
        u = self.tongue.request_utterance(self.sim, drive)
        self.pending_request = None
        self._request_armed = False
        if u is not None:
            self.outbox.append({"tick": u["tick"], "day": u["day"],
                                "time": fmt_time(u["tick"] % TICKS_PER_DAY),
                                "drive": drive, "text": u["text"],
                                "claims": u["claims"]})
            self._log_her_new(before)
        return u

    # ---------------- self-talk (gate §1) ----------------
    def _self_talk_tick(self):
        sim = self.sim
        tick = sim.tick
        if tick - self._last_self_talk < SELF_TALK_COOLDOWN:
            return
        f = sim.fajr
        # fresh salient event -> a memory-voicing if she has the verb
        for kind in ("eat", "drink", "bathe", "swim", "prick", "cold_shock"):
            if self.recent_event(kind, within=3):
                eps = [e for e in f.mind.episodic.episodes
                       if e["kind"] == kind and tick - e["tick"] <= 90]
                if eps:
                    before = len(self.tongue.utterances)
                    u = self.tongue.memory_report(sim, eps[-1], context="self_talk")
                    if u is not None:
                        self._log_her_new(before)
                        self._last_self_talk = tick
                    return
        hunger70 = f.drives.d["hunger"] >= 70.0
        if hunger70 and not self._hunger_was70:
            before = len(self.tongue.utterances)
            u = self.tongue.dominant_report(sim, context="self_talk")
            if u is not None:
                self._log_her_new(before)
                self._last_self_talk = tick
        self._hunger_was70 = hunger70

    # ---------------- turns ----------------
    def exchange(self, host_text):
        """One turn: the Host's line in, her grounded reply out (or
        None when she honestly has no words). Returns her utterance
        dict or None. If a request is pending she delivers it first —
        it is her turn before it is his."""
        sim = self.sim
        f = sim.fajr
        if f.asleep:
            return None
        self._log_line("host", host_text)
        reply = None
        if self.pending_request:
            reply = self._deliver_request()
        parsed = self.tongue.parse(host_text)
        before = len(self.tongue.utterances)
        u = self._respond(parsed)
        if u is not None:
            reply = u
        self._log_her_new(before)
        return reply

    def _respond(self, parsed):
        sim = self.sim
        t = self.tongue
        kind = parsed["kind"]
        # words unknown to the LANGUAGE (not in any form map)
        if parsed["unknown"]:
            surface = parsed["unknown"][0].strip("؟?.!،,؛:")
            t.heard_unknown.append({"tick": sim.tick, "surface": surface})
            return t.unknown_utterance(sim, surface=surface)
        # words known to the language but not (yet) to HER —
        # comprehension is receptive: a word she still recognises
        # (strength >= 0.25) is understood even if she could no
        # longer produce it herself
        if kind != "teaching":
            for wid, _form, surface in parsed["content"]:
                if not t.recognises(wid):
                    surface = surface.strip("؟?.!،,؛:")
                    t.heard_unknown.append({"tick": sim.tick, "surface": surface})
                    return t.unknown_utterance(sim, surface=surface)
        if kind == "teaching":
            return self._apply_teaching(parsed["word"])
        if kind == "instruction":
            d = self._apply_instruction(parsed)
            if d is None:
                # Honesty (INTERACTION-CHARTER): an instruction she
                # cannot fulfill is declined within her means — never
                # silence disguised as compliance, never a pretended
                # attempt. A fulfilled instruction is answered by the
                # act itself (the directive), not by words.
                return honesty.decline_utterance(t, sim, parsed["verb"],
                                                 reason="no_means")
            return None
        if kind == "q_state":
            return self._answer_state(parsed["word"])
        if kind == "q_belief":
            wid = parsed["word"]
            kind_of = {"tree": "tree", "pile": "pile",
                       "thicket": "thicket"}.get(wid)
            if kind_of is None:
                return t.unknown_utterance(sim)
            b = t.nearest_belief(sim, kind_of)
            if b is None:
                return t.unknown_utterance(sim)
            return t.belief_report(sim, b)
        if kind == "q_identify":
            return self._answer_identify()
        if kind == "q_name":
            return t.self_name(sim)
        if kind == "q_who":
            if t.known("host"):
                return t._emit(sim.tick, sim.tick // TICKS_PER_DAY + 1, "reply",
                               "أنت المضيف",
                               [{"type": "self", "field": "host", "value": "المضيف"}],
                               ["host"])
            return t.unknown_utterance(sim)
        if kind == "q_where":
            return self._answer_where(parsed["word"])
        if kind == "q_memory":
            return self._answer_memory(parsed.get("word"), parsed)
        if kind == "fact":
            t.taught_beliefs.append({
                "tick": sim.tick,
                "proposition": f"{parsed['subject']}_at_{parsed['place']}",
                "text": parsed["text"], "source": "host_fact"})
            return None
        if kind == "greeting":
            return t.self_name(sim)
        # unparsed: she has the words but not the construction
        return t.unknown_utterance(sim)

    def _apply_teaching(self, wid):
        sim = self.sim
        anchored = self.host.anchor_ok(self, wid)
        if not anchored:
            # Honesty fix (LANGUAGE-GROUNDING): naming without the
            # referent present is not grounding — no strength is
            # granted and no teaching is recorded. The host's line
            # stays in the transcript (she heard it), but she does
            # not learn the word until an anchored teaching.
            # Previously gain=0.3 here let unanchored namings
            # accumulate to production threshold with no lived
            # referent, and logged episodes that blocked later
            # anchored teaching via the episodes>=2 skip.
            return None
        self.tongue.learn(wid, sim.tick, gain=TEACH_GAIN, anchored=True,
                          source="teaching")
        sim.fajr.mind.episodic.set_semantic(
            f"word:{wid}", self.tongue.strength(wid), sim.tick)
        sim.journal.log(sim.tick, "teaching", self.host.teach_line(wid),
                        x=sim.fajr.x, y=sim.fajr.y)
        # she names it back when she can — the echo is the receipt
        w = WORDS[wid]
        if w["pos"] == "noun" and wid not in ("fajr", "host"):
            return self._identify_word(sim, wid)
        if w["pos"] == "state":
            r = self.tongue._state_report(sim, wid)
            if r is not None:
                text, claims, used = r
                return self.tongue._emit(sim.tick, sim.tick // TICKS_PER_DAY + 1,
                                         "reply", text, claims, used)
        return None

    def _identify_word(self, sim, wid):
        dem = "هذه" if wid in ("tree", "fruit", "pile", "forest") else "هذا"
        claims = [{"type": "identify", "word": wid}]
        return self.tongue._emit(sim.tick, sim.tick // TICKS_PER_DAY + 1,
                                 "reply", f"{dem} {word_surface(wid)}",
                                 claims, [wid])

    def _answer_identify(self):
        sim = self.sim
        scene = sim.fajr.mind.perception.scene
        if scene is None:
            return self.tongue.unknown_utterance(sim)
        best = None
        for o in scene.objects.values():
            if o.generated:
                continue
            wid = {"tree": "tree", "thicket": "thicket", "pile": "pile",
                   "structure": "structure", "shelter": "shelter"}.get(o.kind)
            if o.kind == "material":
                wid = "wood" if o.features.get("mat_kind") == "wood" else None
                # material kind rides in the percept extra upstream; the
                # scene keeps kind only — fall back to the noun she can
                # ground: skip materials she cannot distinguish here
                continue
            if wid is None:
                continue
            if best is None or o.dist < best[0]:
                best = (o.dist, wid, o)
        if best is None:
            return self.tongue.unknown_utterance(sim)
        _, wid, o = best
        if not self.tongue.known(wid):
            return self.tongue.unknown_utterance(sim)
        u = self._identify_word(sim, wid)
        if u is not None:
            u["claims"] = [{"type": "identify", "word": wid,
                            "key": f"{o.kind}:{o.oid}", "dist": round(o.dist, 1)}]
        return u

    def _answer_state(self, wid):
        sim = self.sim
        t = self.tongue
        r = t._state_report(sim, wid)
        if r is not None:
            text, claims, used = r
            return t._emit(sim.tick, sim.tick // TICKS_PER_DAY + 1, "reply",
                           f"نعم، {text}", claims, used)
        # the state does not hold: she says so, with her true dominant
        # state if she has its word
        dom = t.dominant_report(sim, context="reply")
        if dom is not None:
            dom["text"] = "لا. " + dom["text"]
            dom["claims"].insert(0, {"type": "state_negation", "word": wid})
            return dom
        return t._emit(sim.tick, sim.tick // TICKS_PER_DAY + 1, "reply", "لا",
                       [{"type": "state_negation", "word": wid}], [])

    def _answer_where(self, wid):
        sim = self.sim
        t = self.tongue
        kind_of = {"tree": "tree", "pile": "pile", "thicket": "thicket"}.get(wid)
        if kind_of:
            b = t.nearest_belief(sim, kind_of)
            if b is None:
                return t.unknown_utterance(sim)
            return t.belief_report(sim, b)
        # a place word: she locates herself relative to it
        pw = place_word_at(sim.fajr.x, sim.fajr.y)
        if pw == wid and t.known(wid):
            return t._emit(sim.tick, sim.tick // TICKS_PER_DAY + 1, "reply",
                           f"أنا عند {word_surface(wid)}",
                           [{"type": "place", "word": wid,
                             "x": sim.fajr.x, "y": sim.fajr.y}], [wid])
        return t.unknown_utterance(sim)

    def _answer_memory(self, verb_wid, parsed=None):
        sim = self.sim
        t = self.tongue
        eps = sim.fajr.mind.episodic.episodes
        if verb_wid:
            kinds = [k for k, v in
                     {"eat": "eat", "drink": "drink", "sleep_v": "sleep",
                      "bathe": "bathe", "swim": "swim", "gather": "gather",
                      "build": "place"}.items() if k == verb_wid]
            pool = [e for e in eps if e["kind"] in kinds]
        else:
            pool = [e for e in eps if e["kind"] in
                    ("eat", "drink", "bathe", "swim", "gather", "place", "sleep")]
        # Honesty (INTERACTION-CHARTER, anti-confabulation): a memory
        # question naming a place answers ONLY from episodes at that
        # place. Answering "what did you drink at the hill?" with a
        # drink at the river would be confabulation by misattribution
        # — a real episode cited for a question it does not answer.
        # No matching episode: not-knowing, never a guess.
        places = honesty.question_places(parsed) if parsed else []
        if places:
            pool = [e for e in pool
                    if place_word_at(e["x"], e["y"]) in places]
        if not pool:
            return t.unknown_utterance(sim)
        return t.memory_report(sim, pool[-1])

    # ---------------- instructions (gate §4) ----------------
    def _apply_instruction(self, parsed):
        sim = self.sim
        d = self._build_directive(parsed)
        self.last_directive = {"tick": sim.tick, "verb": parsed["verb"],
                               "directive": d}
        self.directive_log.append(self.last_directive)
        if d is not None:
            sim.fajr.host_directive = d
        return d

    def _build_directive(self, parsed):
        sim = self.sim
        f = sim.fajr
        tick = sim.tick
        verb = parsed["verb"]
        exp = tick + DIRECTIVE_TTL
        if verb == "eat":
            cands = [b for b in f.mind.beliefs.objects.values()
                     if b["kind"] in ("tree", "pile") and b["fruit_est"] >= 0.6]
            if not cands:
                return None
            b = max(cands, key=lambda b: b["fruit_est"] * (0.5 + b["conf"]))
            payload = b["oid"] if b["kind"] == "tree" else -1000 - b["oid"]
            return ("eat", b["x"], b["y"], payload, exp)
        if verb == "drink":
            return ("drink", river_center_x(f.y), f.y, -1, exp)
        if verb == "sleep_v":
            return ("sleep", SHELTER_POS[0], SHELTER_POS[1], 0, exp)
        if verb == "bathe":
            bx, by = bath_spot(f.x, f.y)
            return ("bathe", bx, by, -1, exp)
        if verb == "swim":
            return ("explore", river_center_x(f.y), f.y, -1, exp)
        if verb == "go":
            place = parsed.get("place")
            targets = {"river": (river_center_x(f.y), f.y),
                       "shelter": SHELTER_POS, "hill": (208.0, 42.0),
                       "forest": (205.0, 95.0), "lantern": LANTERN_POS}
            if place in targets:
                tx, ty = targets[place]
                return ("explore", tx, ty, -1, exp)
            return None
        if verb == "come":
            return ("social", LANTERN_POS[0], LANTERN_POS[1], 0, exp)
        if verb == "gather":
            raw = f._raw
            if raw is not None:
                mats = [(dist, f.x + dx, f.y + dy, oid)
                        for (k, oid, dx, dy, dist, _e) in raw.visible
                        if k == "material"]
                if mats:
                    _d, mx, my, mid = min(mats)
                    return ("gather", mx, my, mid, exp)
            return None
        if verb == "build":
            if f.carry:
                return ("place", BUILD_SITE[0], BUILD_SITE[1], -1, exp)
            return None
        if verb == "look":
            flags = f.mind.beliefs.epistemic_flags
            target = None
            if flags:
                key = sorted(flags.keys())[0]
                target = f.mind.beliefs.objects.get(key)
            if target is None:
                b = self.tongue.nearest_belief(sim, "tree")
                target = b
            if target is not None:
                return ("observe", target["x"], target["y"], target["oid"], exp)
            return None
        return None

    # ---------------- surface state ----------------
    def state(self):
        sim = self.sim
        f = sim.fajr
        return {
            "tick": sim.tick, "day": sim.tick // TICKS_PER_DAY + 1,
            "time": fmt_time(sim.tick % TICKS_PER_DAY),
            "weather": sim.world.weather,
            "asleep": f.asleep,
            "x": round(f.x, 1), "y": round(f.y, 1),
            "drives": {k: round(v, 1) for k, v in f.drives.d.items()},
            "cleanliness": round(f.cleanliness, 1), "wet": round(f.wet, 2),
            "valence": round(f.drives.valence(), 3),
            "intent": (f.intent[0] if f.intent else None),
            "known_words": sorted(w for w in self.tongue.words
                                  if self.tongue.known(w)),
            "outbox": list(self.outbox),
            "transcript": list(self.transcript),
            "journal_tail": sim.journal.entries[-20:],
        }

    # ---------------- serialisation ----------------
    def to_dict(self):
        return {"tongue": self.tongue.to_dict(),
                "outbox": [dict(o) for o in self.outbox],
                "transcript": [dict(t) for t in self.transcript],
                "covenant_done": self.covenant_done,
                "directive_log": [dict(d) for d in self.directive_log]}

    @classmethod
    def from_dict(cls, sim, d):
        c = cls.__new__(cls)
        c.sim = sim
        c.tongue = Tongue.from_dict(d["tongue"])
        c.host = Host()
        c.outbox = [dict(o) for o in d["outbox"]]
        c.transcript = [dict(t) for t in d["transcript"]]
        c.covenant_done = d["covenant_done"]
        c.pending_request = None
        c._request_armed = True
        c._need_since = {}
        c._remedy_streak = {}
        c._last_teach_tick = -10 ** 9
        c._last_self_talk = -10 ** 9
        c._last_ambient = -10 ** 9
        c._prev_stats = dict(sim.fajr.stats)
        c._last_event_tick = {}
        c._hunger_was70 = sim.fajr.drives.d["hunger"] >= 70.0
        c.last_directive = None
        c.directive_log = [dict(x) for x in d["directive_log"]]
        c._teach_cursor = 0
        return c
