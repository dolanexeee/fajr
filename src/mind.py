"""The mind — Phase 2 (SPEC §6). A LIDA-flavoured cognitive cycle laid over
the Phase-1 body, run once per tick from Fajr.step:

  perceive (perception.py: recurrent, predictive, fallible)
    -> monitor assesses reliability (monitor.py)
    -> beliefs update, reliability-gated (beliefs.py)
    -> six modules propose content coalitions (workspace.py)
    -> attention schema predicts + guards; workspace competes; broadcast
    -> if a broadcast need lacks target information: sequential queries
       memory -> beliefs -> planning (GWT-4)
    -> planning decides an intention, scored with the forward model
    -> learning: forward model outcomes, schema outcomes, self-model
       samples, episodic recording at salient events, consolidation in
       sleep.

Agency (AE-1) is arbitrated THROUGH this cycle. The body's reflexes
(flee, prick reaction, sleep collapse) stay in the body — subcortical,
as they should be; the ablation battery (P-A) checks that they persist
when the workspace is off and this whole cycle stops steering.

No LLM anywhere. All randomness flows through the sim's single PRNG.
"""
import math

from beliefs import BeliefStore, LOW_REL, HIGH_REL
from episodic import EpisodicMemory
from fwdmodel import ForwardModel, context_key
from monitor import Monitor
from perception import Perception, object_key
from selfmodel import SelfModel
from workspace import Workspace, Coalition, MODULES, schema_features
from config import SHELTER_POS, LANTERN_POS, BUILD_SITE
from world import river_center_x, bath_spot

VITAL = ("hunger", "thirst", "sleep", "safety")


class Affect:
    """Valence (from drives, as in Phase 1) + arousal (surprise + urgency
    + hazard). Feeds module salience; recorded into episodes."""

    def __init__(self):
        self.valence = 0.7
        self.arousal = 0.2

    def update(self, drives, pred_error, near_hazard):
        self.valence = drives.valence()
        urg = max(drives.urgency(k) for k in drives.d) / 20.0
        self.arousal = max(0.0, min(1.0,
                                   0.45 * min(1.0, pred_error / 2.5) +
                                   0.35 * min(1.0, urg) +
                                   0.35 * (1.0 if near_hazard else 0.0)))
        return self.valence, self.arousal


class Mind:
    def __init__(self, rng):
        self.perception = Perception()
        self.monitor = Monitor()
        self.beliefs = BeliefStore()
        self.fwd = ForwardModel()
        self.episodic = EpisodicMemory()
        self.selfmodel = SelfModel()
        self.workspace = Workspace()
        self.affect = Affect()
        self.steers = True            # False -> Phase-1 arbitration (lesions)
        self.active = True            # False -> mind entirely off (master lesion)
        self.routing_enabled = True   # v4-3: workspace-winner decision routing
                                      # (battery ablation switch, sim mind_config)
        self._tick = 0
        self.chosen_intent = None
        self._last_pressure = None    # (sleep pressure, tick) at last sample
        self._prev_scene_keys = {}    # key -> "sensed"|"generated" last tick
        self._anchored = {"hunger": False, "thirst": False}
        self._was_asleep = False
        self._prev_drives = None
        self._prev_feats = None
        self._prev_winner = None
        self._saw_pile_keys = set()
        self._storm_prev = False
        self._tree_obs = {}           # oid -> [fruit_seen, obs]
        self._endurance_pending = []  # [dist, pressure, prediction]
        self._last_sleep_count = 0
        self.metrics = {
            "cycles": 0, "task_cycles": 0,
            "fm_participated": 0, "decisions": 0,
        }
        # harness tapes (measurement only; excluded from checkpoint state)
        self.tape_monitor = []        # (reliability, correct) sampled
        self.decision_log = []        # eat-intent adoptions for P-B/AE-2

    # ------------------------------------------------------------------
    # main cycle
    # ------------------------------------------------------------------
    def cycle(self, body, world, raw, tick, light, rng, journal):
        f = body
        self._tick = tick
        weather = world.weather
        # ---------- forward-model outcome from LAST tick's expectation --
        scene_prev = self.perception.scene
        ev = getattr(f, "mind_events", {}) or {}
        outcome = {
            "see_water": ("river" in raw.sounds) or raw.in_water,
            "see_food": any(o.kind in ("tree", "pile") and o.fruit > 0
                            for o in scene_prev.objects.values()),
            "hazard": bool(ev.get("pricked") or ev.get("cold_shock")),
            "d_hunger": 0.0, "d_thirst": 0.0, "d_sleep": 0.0, "moved": 0.0,
        }
        if self._prev_drives is not None:
            outcome["d_hunger"] = f.drives.d["hunger"] - self._prev_drives["hunger"]
            outcome["d_thirst"] = f.drives.d["thirst"] - self._prev_drives["thirst"]
            outcome["d_sleep"] = f.drives.d["sleep"] - self._prev_drives["sleep"]
        if self.perception.prev_pos is not None:
            outcome["moved"] = math.hypot(f.x - self.perception.prev_pos[0],
                                          f.y - self.perception.prev_pos[1])
        self.fwd.observe_outcome(outcome, tick // 1440 + 1)
        self._prev_drives = dict(f.drives.d)

        # ---------- perception ----------
        truth = {}
        for (kind, oid, dx, dy, dist, extra) in raw.visible:
            truth[object_key(kind, oid, f.x + dx, f.y + dy)] = (kind, extra)
        # v2 wiring repair (2026-10-23): snapshot the prediction BEFORE
        # build() runs — build ends by overwriting perception.predicted
        # with next tick's prediction (a copy of the just-built scene),
        # so the pred_err computation below used to compare the scene
        # against itself and the monitor's pred_err input was identically
        # 0.0 in all 9,037 claims of the ceiling analysis. The monitor now
        # receives the genuine per-object prediction error.
        pred_snapshot = dict(self.perception.predicted)
        scene = self.perception.build(raw, f, world, tick, light, weather,
                                      rng, self.beliefs, self.fwd,
                                      harness_truth=truth)

        # ---------- monitor ----------
        pred_err_by_key = {}
        for key, p in pred_snapshot.items():
            if key in scene.objects:
                o = scene.objects[key]
                pred_err_by_key[key] = math.hypot(o.x - p["x"], o.y - p["y"])
        temporal = {}
        for key in scene.objects:
            prev = self._prev_scene_keys.get(key)
            temporal[key] = 1.0 if prev == "sensed" else (0.55 if prev == "generated" else 0.25)
        # FIRST settle earlier claims with this tick's cleaner percepts:
        # a close, well-lit view settles it; a clear view from mid-range
        # settles an older claim made in worse conditions (that asymmetry
        # is what lets the monitor learn what darkness is worth)
        for key, o in scene.objects.items():
            if o.kind in ("tree", "pile") and not o.generated:
                if (o.dist < 8.0 and light > 0.4) or (o.dist < 14.0 and light > 0.45):
                    self.monitor.verify(key, o.fruit, tick)
        reliability = self.monitor.assess(
            scene.objects, pred_err_by_key, temporal, tick,
            belief_fruit={k: b["fruit_est"] for k, b in self.beliefs.objects.items()})
        self.monitor.expire(tick)
        # harness tape: reliability vs world truth. Exact-count claims.
        # Sampled every tick in hard conditions (dark / storm — where the
        # errors live), every 10th tick otherwise.
        hard = light < 0.4 or weather == "storm"
        if hard or tick % 10 == 0:
            for key, o in scene.objects.items():
                if o.kind in ("tree", "pile") and not o.generated:
                    t = truth.get(key)
                    if t is not None:
                        correct = 1 if o.fruit == t[1] else 0
                        self.tape_monitor.append((round(reliability.get(key, 0.5), 3), correct,
                                                  1 if hard else 0))
        self._prev_scene_keys = {k: ("generated" if o.generated else "sensed")
                                 for k, o in scene.objects.items()}

        # ---------- beliefs ----------
        # consumption retirement, belief side (v2 repair, 2026-10-23): a
        # material he gathered last tick is in his hands, not in the
        # world — its belief is deleted outright (certain knowledge from
        # his own action), so the belief prior cannot re-seed the phantom
        # placeholder perception just retired.
        if ev.get("gathered"):
            for key, b in list(self.beliefs.objects.items()):
                if b["kind"] == "material" and math.hypot(
                        b["x"] - f.x, b["y"] - f.y) <= 5.0:
                    del self.beliefs.objects[key]
        self.beliefs.update_from_scene(scene, reliability, tick,
                                       scene.pred_error_self_discounted)
        # sync the body's Phase-1 food memory from BELIEFS (his decisions
        # read beliefs now, not raw transduction) — only while steering;
        # under a steering lesion the raw intake in Fajr.observe governs
        if self.steers:
            for key, b in self.beliefs.objects.items():
                if b["kind"] == "tree" and b["conf"] >= 0.30:
                    f.food_memory[b["oid"]] = [b["x"], b["y"],
                                                int(round(b["fruit_est"])), b["last_seen"]]
        for key, b in self.beliefs.objects.items():
            o = scene.objects.get(key)
            if b["kind"] == "tree" and o is not None and not o.generated:
                slot = self._tree_obs.setdefault(b["oid"], [0, 0])
                slot[1] += 1
                if b["fruit_est"] >= 0.6:
                    slot[0] += 1

        # ---------- affect ----------
        near_hazard = any(o.kind == "thicket" and o.dist < 10.0
                          for o in scene.objects.values())
        self.affect.update(f.drives, scene.pred_error_self_discounted, near_hazard)

        # ---------- self-model upkeep ----------
        # relief is FELT at once: eating, drinking and waking force an
        # immediate interoceptive sample, not the next hourly one
        force = bool(ev.get("ate") or ev.get("drank")) or (self._was_asleep and not f.asleep)
        self._was_asleep = f.asleep
        self.selfmodel.sample(f.drives.d, tick, rng, asleep=f.asleep, force=force)
        # felt extremes are veridical anchors (declared mechanism) —
        # fired on the upward CROSSING only: an anchor is an event,
        # not a state, or the bias would slam on every tick
        for drv in ("hunger", "thirst"):
            if f.drives.d[drv] >= 88.0 and not self._anchored[drv]:
                self.selfmodel.anchor(drv, f.drives.d[drv], tick)
                self._anchored[drv] = True
            elif f.drives.d[drv] < 70.0:
                self._anchored[drv] = False
        sleeps_now = f.stats["sleeps"]
        if sleeps_now > self._last_sleep_count:
            self._last_sleep_count = sleeps_now
            # sleep onset: consolidation + home learning + anchors
            n_new = self.episodic.consolidate(tick)
            if f.sleep_episodes:
                _, sx, sy = f.sleep_episodes[-1]
                if math.hypot(sx - SHELTER_POS[0], sy - SHELTER_POS[1]) < 10.0:
                    self.selfmodel.note_home_sleep(SHELTER_POS)
                    self.beliefs.revise_var("home_is_shelter", 1.0, 0.25)
            if f.drives.d["sleep"] >= 90.0:
                self.selfmodel.anchor("sleep", f.drives.d["sleep"], tick)
            for oid, (seen, obs) in self._tree_obs.items():
                if obs >= 3:
                    self.episodic.set_semantic(f"tree:{oid}:fruit_rate", seen / obs, tick)
            journal.log(tick, "consolidation",
                        f"رسّخ {n_new} ذكرى أثناء النوم", x=f.x, y=f.y)

        # ---------- episodic recording ----------
        self._record_events(f, scene, tick, ev, weather, journal)

        if f.asleep:
            self.chosen_intent = None
            return

        # workspace lesion (P-A): no competition, no broadcast, no queries,
        # no steering — the cycle above still perceives/records/learns,
        # and the body's Phase-1 arbitration drives behaviour
        if self.workspace.lesion_off:
            self.chosen_intent = None
            return

        # ---------- modules propose (stage 1) ----------
        proposals = self._propose(f, world, scene, reliability, tick, light)
        # attention schema: predict this cycle's winner from LAST cycle's
        # features; learn from the outcome; guard an active goal
        goal_active = f.intent is not None and f.intent[0] not in ("wander",)
        boost = None
        if self._prev_feats is not None:
            pred_mod, prob, _ = self.workspace.schema.predict(self._prev_feats)
            boost = {}
            # stabilisation: a confident self-prediction mildly reinforces
            # the module it expects to win (attention self-consistency)
            if prob > 0.60:
                boost[pred_mod] = 1.12
            # capture guard: during goal pursuit, if the schema expects a
            # non-goal module (vision/affect) to capture attention next,
            # goal-serving modules are boosted instead
            if goal_active and pred_mod in ("vision", "affect"):
                boost = {"interoception": 1.3, "memory": 1.3, "beliefs": 1.2,
                         "motor": 1.15, "planning": 1.3}
            if boost:
                self.metrics["schema_boosts"] = self.metrics.get("schema_boosts", 0) + 1
            else:
                boost = None
        winners = self.workspace.compete(proposals, schema_boost=boost)
        winner_src = winners[0].source if winners else "motor"
        if self._prev_feats is not None:
            self.workspace.schema.learn(self._prev_feats, winner_src)
        # distractor-capture bookkeeping
        focus = winners[0] if winners else None
        focus_is_distractor = bool(
            focus and focus.source == "vision" and focus.target_key and
            scene.objects.get(focus.target_key) is not None and
            scene.objects[focus.target_key].features["novelty"] > 0.6 and
            not (f.intent and f.intent[0] in ("eat", "gather")))
        self.workspace.note_attention(winner_src, goal_active, focus_is_distractor)
        self._prev_winner = winner_src

        # ---------- sequential queries (GWT-4) ----------
        need = self._dominant_need(f)
        self.workspace.last_queries = []
        queries = self._run_queries(f, world, scene, winners, need, tick)
        # v2 chain validation (GWT-4 repair, 2026-10-23): the completed
        # sequence is recorded on the workspace; _decide treats far
        # episode-only targets as admissible only when it ran in full.
        self.workspace.chain_completed = (
            queries == ["memory", "beliefs", "planning"])
        if queries:
            self.workspace.note_task_cycle(len(queries))
            self.metrics["task_cycles"] += 1

        # ---------- decision (stage 2) ----------
        intent = self._decide(f, world, scene, winners, need, tick, light)
        self.chosen_intent = intent
        # forward-model expectation for the chosen course of action
        tod = tick % 1440
        key = context_key(intent[0] if intent else "idle", tod, weather,
                          raw.in_shelter)
        self.fwd.expect(key)
        # schema features for the NEXT cycle's prediction
        top_urg = max(f.drives.urgency(k) for k in f.drives.d)
        distract = 0.0
        for o in scene.objects.values():
            if focus is None or o.key != focus.target_key:
                distract = max(distract, o.features["novelty"])
        self._prev_feats = schema_features({
            "top_urgency": top_urg, "light": light, "arousal": self.affect.arousal,
            "distractor_novelty": distract, "goal_active": goal_active,
            "hour_frac": (tod / 1440.0), "prev_winner": winner_src})
        self.metrics["cycles"] += 1

    # ------------------------------------------------------------------
    # episodic triggers
    # ------------------------------------------------------------------
    def _record_events(self, f, scene, tick, ev, weather, journal):
        digest = [scene.objects[k].kind for k in scene.foreground[:3]]
        snap = {k: round(f.drives.d[k], 1) for k in VITAL}
        val, aro = self.affect.valence, self.affect.arousal

        def rec(kind, outcome=0.0, note="", x=None, y=None):
            # an episode binds WHERE ITS EVENT HAPPENED: acts of the
            # body (eat, drink, prick) happen at his position; a
            # SIGHTING happens at the object seen (v4, GATE v4-4a —
            # the journal already logged it so; the episode stored his
            # standing point instead, so episodic steering aimed at
            # where he stood when he saw the pile, ~10 units short of
            # the pile itself)
            self.episodic.record(tick, kind, f.x if x is None else x,
                                 f.y if y is None else y, digest, snap, val, aro,
                                 outcome=outcome, note=note)
        if ev.get("ate"):
            rec("eat", 0.8)
            self.selfmodel.note_outcome("food", 0.8)
        if ev.get("drank"):
            rec("drink", 0.6)
            self.beliefs.revise_var("water_at_river", 1.0, 0.3)
        # Stage 2 (WATER-GATE): bathing and swimming are autobiographical
        # events like any other — a good bathing place is remembered at
        # its own coordinates, and the fact "water cleans" is revisable
        if ev.get("bathed"):
            rec("bathe", 0.7)
            self.beliefs.revise_var("water_cleans", 1.0, 0.3)
        if ev.get("swam"):
            rec("swim", 0.15)
        if ev.get("pricked"):
            rec("prick", -0.9)
            self.beliefs.revise_var("thorn_hurts", 1.0, 0.4)
        if ev.get("cold_shock"):
            rec("cold_shock", -0.7)
        if ev.get("gathered"):
            rec("gather", 0.3)
        if ev.get("placed"):
            rec("place", 0.5)
            self.selfmodel.note_outcome("building", 0.5)
        # first sight of a fruit pile: a one-time kind of event (P-C's seed)
        for key, o in scene.objects.items():
            if o.kind == "pile" and not o.generated and key not in self._saw_pile_keys:
                self._saw_pile_keys.add(key)
                rec("saw_pile", 0.4, note=key, x=o.x, y=o.y)
                journal.log(tick, "saw_pile", "رأى كومة فاكهة", x=o.x, y=o.y)
        if weather == "storm" and not self._storm_prev:
            rec("storm", -0.2)
        self._storm_prev = weather == "storm"
        if scene.pred_error_self_discounted > 2.2:
            rec("surprise", -0.1)

    # ------------------------------------------------------------------
    # stage 1: module proposals
    # ------------------------------------------------------------------
    def _propose(self, f, world, scene, reliability, tick, light):
        props = []
        d = f.drives.d
        # --- vision: salient scene contents (an empty field still gets
        # a scanning bid — the module is active, it has nothing to show) ---
        if not scene.foreground:
            props.append(Coalition("vision", "scan", 0.18))
        for key in scene.foreground[:4]:
            o = scene.objects[key]
            sal = 0.25 + 0.55 * o.salience
            if o.kind == "thicket" and o.dist < 12.0:
                sal += 0.25
            props.append(Coalition("vision", "attend", sal, target_key=key,
                                   x=o.x, y=o.y))
        # --- interoception: needs. The top needs bid every cycle (a
        # module that is silent when drives are low is not monitoring);
        # salience carries the urgency. ---
        ranked = sorted(d.items(), key=lambda kv: -f.drives.urgency(kv[0]))
        for name, deficit in ranked:
            if deficit > 38.0 or name in (ranked[0][0], ranked[1][0]):
                sal = min(1.0, (deficit / 100.0) ** 1.5 * 1.6)
                props.append(Coalition("interoception", f"need:{name}", max(0.05, sal)))
        # --- affect: its state is always a content; salience follows it.
        # Its scale is commensurate with the other modules' (a flood of
        # affect after a hazard SHOULD be able to win the workspace). ---
        props.append(Coalition("affect", "affect_state",
                               0.25 + 0.75 * self.affect.arousal
                               + 1.1 * max(0.0, 0.65 - self.affect.valence)
                               + 0.4 * max(0.0, self.affect.valence - 0.85)))
        # --- memory: retrieval cued by broadcast + scene + needs. The
        # module bids EVERY cycle (with whatever the cues return); when
        # no need cue exists it bids its most salient recent episodes at
        # low salience — a module that only speaks when shouted at is
        # not a parallel specialist. ---
        cue = self._memory_cue(f, scene)
        kinds = cue[0] if cue is not None else ["eat", "prick", "place", "saw_pile"]
        goal = cue[1] if cue is not None else None
        # v4 ecology (GATE v4-1 calibration): memory speaks at FULL
        # strength when the BROADCAST cued it — last cycle's workspace
        # contents are the workspace's half of the retrieval loop, and
        # that loop is the global-workspace claim. A drive state alone
        # (a need with no broadcast corroboration) earns a damped bid:
        # drive-cued memory at full strength every waking cycle made
        # the module the habitual top winner (~0.94 majority baseline),
        # collapsing the winner entropy the attention schema learns
        # from (AST-1a dev ratio 1.00 under that regime).
        cue_from_broadcast = cue[2] if cue is not None else False
        eps = self.episodic.retrieve(kinds, cue_x=f.x, cue_y=f.y,
                                     goal_kind=goal, limit=3, with_scores=True)
        for sc_r, ep in eps:
            # v4 (GATE v4-4a): the bid carries the RETRIEVAL SCORE —
            # the memory system's own relevance signal for this cue —
            # not only the record's outcome magnitude. A zero-outcome
            # sighting that is the unique answer to the current cue
            # must be able to outbid a routine record that merely also
            # matches (its absence is what starved P-C's carrier).
            # Dev calibration: the bid's LEVEL sits in the ecology's
            # sitting-3 range (best cued episode ~0.55-0.70) — at
            # higher levels (0.22/unit score term) the module became
            # the habitual top winner in both tiny construction stores
            # and mature ones, collapsing winner entropy (AST-1a dev
            # ratio ~1.0). The score term's job is to ORDER memory's
            # own coalitions and lift the unique carrier into a
            # broadcast slot, not to crown the module.
            sal = 0.22 + 0.10 * min(2.5, sc_r) + 0.22 * abs(ep["outcome"]) \
                + 0.12 * ep["arousal"]
            if cue is None:
                sal *= 0.45
            elif not cue_from_broadcast:
                sal *= 0.6
            props.append(Coalition("memory", f"episode:{ep['kind']}", sal,
                                   x=ep["x"], y=ep["y"], payload=ep))
        # --- beliefs: epistemic flags + the strongest task-relevant
        # belief. Like memory, it bids every cycle. ---
        for key, t0 in list(self.beliefs.epistemic_flags.items()):
            b = self.beliefs.objects.get(key)
            if b is not None:
                props.append(Coalition("beliefs", "uncertain", 0.5,
                                       target_key=key, x=b["x"], y=b["y"]))
        fb = self.beliefs.fruit_beliefs()
        if fb and d["hunger"] > 25.0:
            key, b = max(fb.items(), key=lambda kv: kv[1]["fruit_est"] * kv[1]["conf"])
            props.append(Coalition("beliefs", "belief:food",
                                   0.3 + 0.35 * b["conf"],
                                   target_key=key, x=b["x"], y=b["y"]))
        elif d["sleep"] > 50.0:
            props.append(Coalition("beliefs", "belief:home",
                                   0.3 + 0.35 * self.beliefs.vars["home_is_shelter"]["p"],
                                   x=SHELTER_POS[0], y=SHELTER_POS[1]))
        elif self.beliefs.objects:
            # standing bid: the strongest current belief about the world
            key, b = max(self.beliefs.objects.items(), key=lambda kv: kv[1]["conf"])
            props.append(Coalition("beliefs", "belief:world",
                                   0.22 + 0.2 * b["conf"],
                                   target_key=key, x=b["x"], y=b["y"]))
        # --- motor: immediate affordances + the ongoing action (and a
        # standing readiness bid when idle — the module is never off).
        # v4-1: the execute bid is denominated in the plan's stakes —
        # the motor coalition reports the ongoing course of action, and
        # its salience rises with the urgency of the need that plan
        # serves (a flat 0.3 priced carrying out a starvation plan the
        # same as idling, and the module starved below the GWT-1 floor
        # once beliefs' bids grew — sitting 3 §7.1). ---
        top_urg = max(f.drives.urgency(k) for k in f.drives.d)
        exec_sal = 0.30 + 0.30 * min(1.0, top_urg / 14.0)
        if f.intent is not None:
            props.append(Coalition("motor", f"execute:{f.intent[0]}", exec_sal))
        elif f.acting is not None:
            props.append(Coalition("motor", f"execute:{f.acting[0]}", exec_sal))
        else:
            props.append(Coalition("motor", "ready", 0.2))
        for key, o in scene.objects.items():
            if o.kind in ("tree", "pile") and o.dist < 4.0 and o.fruit > 0 and d["hunger"] > 25.0:
                props.append(Coalition("motor", "afford:eat", 0.55, target_key=key, x=o.x, y=o.y))
            if o.kind == "material" and o.dist < 4.0:
                props.append(Coalition("motor", "afford:gather", 0.4, target_key=key, x=o.x, y=o.y))
        if d["sleep"] > 88.0:
            props.append(Coalition("motor", "afford:collapse", 0.8))
        return props

    def _memory_cue(self, f, scene):
        """Retrieval happens when a need lacks a visible target, or the
        broadcast carried a matching cue last cycle (GWT-3's causal path:
        broadcast content changes what memory returns)."""
        d = f.drives.d
        kinds = []
        goal = None
        if d["hunger"] > 45.0:
            kinds += ["eat", "saw_pile", "saw_food"]
            goal = "eat"
        if d["safety"] > 55.0:
            kinds += ["prick", "cold_shock"]
        if d["sleep"] > 60.0:
            kinds += ["sleep", "shelter_rest"]
            goal = "sleep"
        # Stage 2: a dirty body (felt as comfort load) cues bathing
        # memories — where bathing worked before becomes retrievable
        if d["comfort"] > 55.0 or (100.0 - f.cleanliness) >= 50.0:
            kinds += ["bathe"]
        # broadcast cue: a coalition kind from last cycle primes its kindred
        from_broadcast = False
        for c in self.workspace.contents:
            if c.kind.startswith("episode:"):
                kinds.append(c.kind.split(":", 1)[1])
                from_broadcast = True
            if c.kind == "need:hunger":
                kinds += ["eat", "saw_pile"]
                from_broadcast = True
        if not kinds:
            return None
        return (list(dict.fromkeys(kinds)), goal, from_broadcast)

    def _dominant_need(self, f):
        d = f.drives.d
        best, best_u = None, -1.0
        for name, v in d.items():
            u = f.drives.urgency(name)
            if u > best_u:
                best, best_u = name, u
        return best

    # ------------------------------------------------------------------
    # GWT-4: sequential queries when a need lacks target information
    # ------------------------------------------------------------------
    def _run_queries(self, f, world, scene, winners, need, tick):
        queries = []
        ws = self.workspace
        if ws.lesion_single_query and len(queries) >= 1:
            return queries
        need_kinds = {c.kind for c in winners}
        # hunger/thirst with no visible target in the scene -> query chain
        def visible_food():
            return any(o.kind in ("tree", "pile") and o.fruit > 0 and not o.generated
                       for o in scene.objects.values())
        if need == "hunger" and "need:hunger" in need_kinds and not visible_food():
            # Q1 memory: recall where food was
            ws.query("memory")
            queries.append("memory")
            eps = self.episodic.retrieve(["eat", "saw_pile", "saw_food"],
                                         goal_kind="eat", limit=3)
            for ep in eps:
                ws.contents.append(Coalition("memory", f"episode:{ep['kind']}",
                                             0.6, x=ep["x"], y=ep["y"], payload=ep))
            if ws.lesion_single_query:
                return queries
            # Q2 beliefs: is the recalled/believed target reliable?
            ws.query("beliefs")
            queries.append("beliefs")
            # Q3 planning: evaluate (in _decide via the forward model)
            ws.query("planning")
            queries.append("planning")
        elif need == "sleep" and "need:sleep" in need_kinds:
            ws.query("memory")
            queries.append("memory")
            if ws.lesion_single_query:
                return queries
            ws.query("planning")
            queries.append("planning")
        return queries

    # ------------------------------------------------------------------
    # stage 2: decision through beliefs + forward model
    # ------------------------------------------------------------------
    def _decide(self, f, world, scene, winners, need, tick, light):
        d = f.drives.d
        cands = []  # (score, intent, fm_used, tag)  tag: "" | "routed" | "epistemic"

        def add(score, intent, fm_used=False, tag=""):
            cands.append((score, intent, fm_used, tag))

        def fm_adjust(intent_kind, base, tod, weather):
            key = context_key(intent_kind, tod, weather, False)
            pred = self.fwd.predict(key)
            adj = base
            if intent_kind == "eat":
                adj = base * (0.7 + 0.55 * pred["see_food"] - 0.5 * pred["hazard"])
            elif intent_kind == "drink":
                adj = base * (0.7 + 0.55 * pred["see_water"] - 0.4 * pred["hazard"])
            else:
                adj = base * (1.0 - 0.35 * pred["hazard"])
            return adj, True

        tod = tick % 1440
        weather = world.weather

        # survival preemption: at lethal thirst/hunger the sleep bid is
        # suppressed — measured failure (Phase-2 P-E, seed 97): after a
        # long trek he slept ~1,000 ticks at thirst 99+ because sleep's
        # score at travel ~2 beat drink's at travel ~85. Collapse sleep
        # remains possible via the body's own exhaustion reflex.
        critical = d["thirst"] >= 90.0 or d["hunger"] >= 95.0
        # survival gate (repair round, addendum 2026-10-23): at lethal
        # levels the mind serves survival ONLY. Measured on the v2
        # sitting's seed-42 life: thirst pinned at 100 for 1,192 ticks —
        # the shelter bid (safety+comfort, place-value-boosted near home)
        # and intermittent eat bids alternately outscored drink, whose
        # gain caps at min(thirst, 55); his position oscillated
        # 123<->146 for ~500 ticks until exhaustion collapse. The sleep
        # bid was already suppressed at these levels; every other
        # non-survival bid now is too (filter before selection, with a
        # fallback to the full list if no survival candidate exists).
        # The body's flee reflex and collapse reflex are untouched.
        crit_kinds = None
        if d["thirst"] >= 90.0 and d["hunger"] >= 95.0:
            crit_kinds = ("drink", "eat")
        elif d["thirst"] >= 90.0:
            crit_kinds = ("drink",)
        elif d["hunger"] >= 95.0:
            crit_kinds = ("eat",)

        # --- sleep ---
        if not critical and (d["sleep"] > 70.0 or (light < 0.25 and d["sleep"] > 45.0)):
            hx, hy = (self.selfmodel.home_pos or SHELTER_POS)
            sc = f.drives.urgency("sleep") * 3.2 / (1.0 + f._travel(hx, hy) / 60.0)
            # circadian gate: sleep PROPENSITY is S x C, as in the body's
            # own Borbely-lite physiology. v3 (GATE v3-6): C is the body's
            # ENTRAINED circadian process — an internal phase oscillator
            # anchored by dawn/dusk light cues (drives.sleep_gate),
            # including its evening wake-maintenance dip — not the raw
            # instantaneous light of sittings 1-2. An arbitration that
            # reads a memoryless light signal drifts out of phase with
            # the physiology it serves — measured in sitting 2 as the
            # G4 seed-11 sleep-phase precession (diag_g4.py).
            sc *= f.drives.sleep_gate()
            sc, used = fm_adjust("sleep", sc, tod, weather)
            add(sc, ("sleep", hx, hy, 0), used)
        # --- hunger: targets from BELIEFS + episodic recall ---
        top_food_xy = None   # v4-2: the eat branch's chosen target (epistemic relevance)
        if d["hunger"] > 30.0:
            best = None
            targets = []
            for key, b in self.beliefs.fruit_beliefs().items():
                tid = b["oid"] if b["kind"] == "tree" else -1000 - b["oid"]
                targets.append((b["x"], b["y"], tid, b["fruit_est"], b["conf"], "belief"))
            for c in winners:
                if c.source == "memory" and c.payload is not None:
                    ep = c.payload
                    if ep["kind"] in ("eat", "saw_pile", "saw_food"):
                        if ep["kind"] == "saw_pile" and str(ep.get("note", "")).startswith("pile:"):
                            tid = -1000 - int(ep["note"].split(":")[1])
                            # v2 chain validation (GWT-4 repair): a
                            # one-time pile record is carried ONLY by
                            # episodic memory — beliefs about it read
                            # "bare" once eaten dry. Such an
                            # UNCORROBORATED stale recall becomes an
                            # admissible steering target only in cycles
                            # where the full sequential query chain
                            # (memory -> beliefs -> planning) completed:
                            # the chain is what arbitrates a stale recall
                            # before the body commits to it. (Dev-check
                            # calibration, 2026-10-23: the original
                            # >60-unit distance condition let the
                            # single-query lesion close the distance by
                            # unguided search and then use the recall
                            # ungated at <60 units — success 1.0 in the
                            # lesion arm. Corroboration, not distance, is
                            # the criterion: if a live belief corroborated
                            # the site, the belief target above already
                            # carries it and this branch adds nothing.)
                            if not self.workspace.chain_completed:
                                continue
                            targets.append((ep["x"], ep["y"], tid, 3.0, 0.5, "episode"))
                        else:
                            # v2 (HOT-3 repair): a tree episode defers to
                            # the belief store's CURRENT estimate for that
                            # tree — same object, two traces: the episode
                            # is history, the belief is the living
                            # estimate. A corrupted belief therefore
                            # governs the ghost as well.
                            bo = None
                            for bk, bb in self.beliefs.objects.items():
                                if bb["kind"] == "tree" and math.hypot(
                                        bb["x"] - ep["x"], bb["y"] - ep["y"]) <= 6.0:
                                    bo = bb
                                    break
                            if bo is not None:
                                targets.append((ep["x"], ep["y"], bo["oid"],
                                                bo["fruit_est"], bo["conf"], "episode"))
                            elif not self.workspace.chain_completed:
                                # an episode with NO corroborating belief
                                # (e.g. an eat record at a since-emptied
                                # pile) is the same epistemic object as a
                                # pile record: a stale recall that only
                                # the completed query chain may arbitrate
                                # (dev-check calibration, 2026-10-23 — the
                                # day-1 eat episodes at the pile pinpointed
                                # it in every arm, chain or no chain)
                                continue
                            else:
                                targets.append((ep["x"], ep["y"], -1, 1.0, 0.30, "episode"))
            seen_pos = set()
            for (tx, ty, tid, est, conf, src) in targets:
                pk = (round(tx), round(ty))
                if pk in seen_pos:
                    continue
                seen_pos.add(pk)
                # v2 expected-value targeting (HOT-3 repair, 2026-10-23):
                # the score is proportional to the BELIEVED yield and its
                # confidence, discounted by travel time — no saturation
                # plateau in the realistic estimate range (sitting 1's
                # min(hunger, 34*min(3, est)) made every est >= ~2.2
                # identical, so the associative layer chose instead), and
                # NO _score_target place-value multiplier: at decision
                # distance the belief layer chooses the target. The
                # Phase-1 associative layer keeps its body-reflex roles
                # (visible-food choice in fajr.choose_intent, drink and
                # shelter scoring above/below).
                gain = d["hunger"] * est / 6.0
                sc = f.drives.urgency("hunger") * gain / (f._travel(tx, ty) + 12.0)
                sc *= (0.35 + 0.65 * conf)
                if best is None or sc > best[0]:
                    best = (sc, ("eat", tx, ty, tid))
            if best is not None:
                adj, used = fm_adjust("eat", best[0], tod, weather)
                add(adj, best[1], used)
                top_food_xy = (best[1][1], best[1][2])
                # log the adoption for the battery's wasted-arrival measure
                # (tree targets only; pile targets are one-time by design)
                if best[1][3] >= 0:
                    tree = world.trees[best[1][3]]
                    self.decision_log.append({
                        "tick": tick, "tree": best[1][3],
                        "belief_fruit": round(self.beliefs.objects.get(
                            f"tree:{best[1][3]}", {}).get("fruit_est", 0.0), 2),
                        "world_fruit": tree.fruit})
                    if len(self.decision_log) > 4000:
                        del self.decision_log[:2000]
            else:
                # no believed food: SEEK = go to the frontier (unexplored
                # cells), not to a hardcoded point of the home valley —
                # in any other layout that point is empty ground and the
                # search dies there (Phase-2 P-E regression diagnosis)
                cell = f._best_explore_cell(world)
                if cell is not None:
                    sc = f.drives.urgency("hunger") * 0.55 * (0.5 + 0.5 * cell[0])
                    add(sc, ("search", cell[1], cell[2], -1))
                else:
                    sc = f.drives.urgency("hunger") * 0.55 / (1.0 + f._travel(128.0, 95.0) / 60.0)
                    add(sc, ("seek_food", 128.0, 95.0, -1))
                if d["hunger"] > 70.0:
                    cell = f._best_explore_cell(world)
                    if cell is not None:
                        add(f.drives.urgency("hunger") * cell[0] * 0.9,
                            ("search", cell[1], cell[2], -1))
        # --- thirst (procedural body knowledge, as Phase 1) ---
        if d["thirst"] > 30.0:
            ry = max(10.0, min(170.0, f.y))
            tx = river_center_x(ry)
            sc = f.drives.urgency("thirst") * min(d["thirst"], 55.0) / (f._travel(tx, ry) + 8.0)
            adj, used = fm_adjust("drink", f._score_target(sc, tx, ry, world), tod, weather)
            add(adj, ("drink", tx, ry, -1), used)
        # --- safety / comfort ---
        if d["safety"] > 58.0 or d["comfort"] > 55.0:
            sx, sy = SHELTER_POS
            sc = (f.drives.urgency("safety") + f.drives.urgency("comfort")) * 1.6 / (1.0 + f._travel(sx, sy) / 60.0)
            sc, used = fm_adjust("shelter", sc, tod, weather)
            add(f._score_target(sc, sx, sy, world), ("shelter", sx, sy, 0), used)
        # --- hygiene: bathing (Stage 2, WATER-GATE). A body-state
        # deficit with one specific remedy, priced as the drink branch
        # prices thirst (the serving need's urgency x the removable
        # amount / travel) and carrying bathe_value — the relief
        # association his own completed baths taught him (weak prior
        # 0.25: nobody tells him bathing works; felt relief does).
        # Targets: a remembered bathing place if memory put one in the
        # broadcast, else the procedural bank shallows (bath_spot). ---
        grime_now = 100.0 - f.cleanliness
        if grime_now >= 40.0:
            urg_c = f.drives.urgency("comfort")
            bfactor = 0.2 + 0.8 * f.bathe_value
            bx, by = bath_spot(f.x, f.y)
            best_b = (urg_c * min(grime_now, 70.0) * bfactor / (f._travel(bx, by) + 10.0),
                      ("bathe", bx, by, -1))
            for c in winners:
                if c.source == "memory" and c.payload is not None \
                        and c.payload.get("kind") == "bathe":
                    sc2 = urg_c * min(grime_now, 70.0) * bfactor \
                        / (f._travel(c.payload["x"], c.payload["y"]) + 10.0)
                    if sc2 > best_b[0]:
                        best_b = (sc2, ("bathe", c.payload["x"], c.payload["y"], -1))
            add(best_b[0], best_b[1])
        # --- social (lantern presence — Guide is Phase 3) ---
        if d["social"] > 52.0:
            lx, ly = LANTERN_POS
            sc = f.drives.urgency("social") * 2.4 / (1.0 + f._travel(lx, ly) / 60.0)
            add(sc, ("social", lx, ly, 0))
        # --- creation ---
        if d["creation"] > 48.0:
            if len(f.carry) >= 1:
                bx, by = BUILD_SITE
                sc = f.drives.urgency("creation") * 2.2 / (1.0 + f._travel(bx, by) / 60.0)
                add(sc, ("place", bx, by, -1))
            else:
                tgt = None
                for key, o in scene.objects.items():
                    if o.kind == "material" and not o.generated and (tgt is None or o.dist < tgt[0]):
                        tgt = (o.dist, o.x, o.y, o.oid)
                if tgt is not None:
                    sc = f.drives.urgency("creation") * 1.8 / (1.0 + tgt[0] / 1.15 / 60.0)
                    add(sc, ("gather", tgt[1], tgt[2], tgt[3]))
                else:
                    # frontier search, as with food (no hardcoded point)
                    cell = f._best_explore_cell(world)
                    if cell is not None:
                        add(f.drives.urgency("creation") * 0.5 * (0.5 + 0.5 * cell[0]),
                            ("seek_material", cell[1], cell[2], -1))
                    else:
                        sc = f.drives.urgency("creation") * 0.5 / (1.0 + f._travel(205.0, 95.0) / 60.0)
                        add(sc, ("seek_material", 205.0, 95.0, -1))
        # --- curiosity ---
        if d["curiosity"] > 42.0:
            best = f._best_explore_cell(world)
            if best is not None:
                add(f.drives.urgency("curiosity") * best[0] * 1.15,
                    ("explore", best[1], best[2], -1))
        # --- epistemic: value-of-information re-observation (v4,
        # GATE v4-2 — replaces the Phase-2 flat-curiosity branch, which
        # produced ZERO observe adoptions in the sitting-3 census: its
        # gates (curiosity > 45, non-vital dominant need) almost never
        # aligned, its flat curiosity x 0.75 price could not compete in
        # the urgency economy, and its target was the alphabetically
        # first of a set of flags that never expired). One candidate:
        # the food belief whose unresolved uncertainty is worth most,
        # priced like the eat branch prices food (served urgency x
        # gain / (travel + 12)) with the gain denominated in expected
        # information. Three declared channels per belief:
        #   flagged   — a low-reliability percept raised the question;
        #   surprise  — a recent update revised the estimate by >= 1.5;
        #   routine   — staleness alone, weak, doubled when the belief
        #               is the target he is about to rely on.
        # A candidate whose "uncertain" coalition is IN the broadcast
        # is doubled: what attention selected steers the epistemic act.
        # Suppressed under sleep pressure (sleep's own bid economy
        # decides the night) and, structurally, at lethal levels by
        # the survival filter below. ---
        if d["sleep"] <= 65.0:
            urg_served = max(f.drives.urgency("hunger"),
                             1.4 * f.drives.urgency("curiosity"))
            best_epi = None
            for key, b in self.beliefs.objects.items():
                if b["kind"] not in ("tree", "pile"):
                    continue
                u = min(1.0, math.sqrt(max(0.0, b["fruit_var"])) / 1.6)
                stale = min(1.0, (tick - b["last_seen"]) / 2400.0)
                u_eff = min(1.0, u + 0.35 * stale)
                eig = 0.0
                ft = self.beliefs.epistemic_flags.get(key)
                if ft is not None:
                    rec = max(0.25, 1.0 - (tick - ft) / 900.0)
                    eig = max(eig, (0.65 + 0.35 * u_eff) * rec)
                # settled-belief gate (v4 calibration, dev round 3): a
                # belief the being already holds with confidence >= 0.75
                # and no open flag is SETTLED — the surprise and routine
                # channels price verifying it at a quarter. Epistemic
                # effort belongs to the doubtful; spending it re-checking
                # what trustworthy percepts already settled is what made
                # the first dev pass fire 45 high-class re-observations
                # against 13 low-class ones (ratio 1.26 vs the >= 2
                # discrimination the behaviour exists to show).
                settled = 0.25 if (ft is None and b["conf"] >= 0.75) else 1.0
                # decision relevance (v4 calibration): the surprise and
                # routine channels serve a pending EAT decision; when
                # no hunger is at stake they are idle curiosity checks
                # and price at 0.4 — the deliberate events the battery
                # counts were dominated by idle re-checks of fruitful
                # (high-reliability-class) trees across every dev
                # setting, holding the low/high rate ratio at 1.2-1.7
                # against the behaviour's discriminating purpose. The
                # flag channel is exempt: an open question is a
                # question regardless of appetite.
                relevance = 1.0 if f.drives.urgency("hunger") >= 2.0 else 0.4
                rt = b.get("last_rev_tick")
                if rt is not None and 0 <= tick - rt <= 600:
                    # priced by the revising percept's reliability: a
                    # clean revision is settled knowledge (x~0.25 at
                    # r=0.9); a murky one is a live question (x~0.85)
                    rev_trust = 1.15 - b.get("last_rev_rel", 0.5)
                    eig = max(eig, settled * relevance * 0.30 * rev_trust
                              * min(1.0, b.get("last_rev_mag", 0.0) / 3.0)
                              * max(0.25, 1.0 - (tick - rt) / 900.0))
                routine = settled * relevance * 0.30 * u_eff * stale
                if top_food_xy is not None and math.hypot(
                        b["x"] - top_food_xy[0], b["y"] - top_food_xy[1]) <= 6.0:
                    routine *= 2.0
                eig = max(eig, routine)
                if eig < 0.08:
                    continue
                sc = urg_served * (6.0 * eig) / (f._travel(b["x"], b["y"]) + 12.0)
                for c in winners:
                    if c.source == "beliefs" and c.kind == "uncertain" \
                            and c.target_key == key:
                        sc *= 2.0
                        break
                if best_epi is None or sc > best_epi[0]:
                    best_epi = (sc, ("observe", b["x"], b["y"], b["oid"]))
            if best_epi is not None:
                self.metrics["epi_candidates"] = self.metrics.get("epi_candidates", 0) + 1
                add(best_epi[0], best_epi[1], False, "epistemic")
        # --- workspace routing (v4, GATE v4-3): the broadcast
        # winner's content is a first-class decision input. When
        # VISION tops the workspace with a sensed fruit-bearing food
        # object, that percept becomes an eat candidate priced by the
        # same expected-value formula as the belief path — gain from
        # the SENSED fruit count, discounted by the monitor's
        # reliability for the percept this tick — so what attention
        # selected can steer the body in every life (the AST-1(b)
        # filing showed the winner previously had no causal path to
        # the eat decision at all). The survival filter below is
        # intent-kind based, so a routed eat competes inside it on
        # its score like any other eat: routing changes where options
        # come from; the survival rule stays absolute. Ablated by
        # mind_config routing_off. ---
        if self.routing_enabled and winners and d["hunger"] > 30.0:
            w0 = winners[0]
            if w0.source == "vision" and w0.target_key:
                o = scene.objects.get(w0.target_key)
                if o is not None and not o.generated \
                        and o.kind in ("tree", "pile") and o.fruit >= 1:
                    rel = self.monitor.last.get(o.key, 0.5)
                    gain = d["hunger"] * o.fruit / 6.0
                    sc = f.drives.urgency("hunger") * gain / (f._travel(o.x, o.y) + 12.0)
                    sc *= (0.35 + 0.65 * rel)
                    sc, used_r = fm_adjust("eat", sc, tod, weather)
                    oid = o.oid if o.kind == "tree" else -1000 - o.oid
                    add(sc, ("eat", o.x, o.y, oid), used_r, "routed")
                    self.metrics["routed_candidates"] = \
                        self.metrics.get("routed_candidates", 0) + 1
        if not cands:
            return ("wander", f.x, f.y, -1)
        if crit_kinds:
            surv = [c for c in cands if c[1][0] in crit_kinds]
            if surv:
                cands = surv
        # (A plan-commitment hysteresis — incumbent eat target x1.3 —
        # was tried in dev and REMOVED: target flips in this dynamics
        # regime happen through belief updates on arrival, not
        # mid-route score noise, so the margin never once changed an
        # argmax in a full probe life; byte-identical trajectory.
        # Dead mechanisms do not ship. Recorded in ARCH3-DEV-REPORT.)
        cands.sort(key=lambda c: -c[0])
        score, intent, used, tag = cands[0]
        if tag == "routed":
            self.metrics["routed_wins"] = self.metrics.get("routed_wins", 0) + 1
        elif tag == "epistemic":
            self.metrics["epi_wins"] = self.metrics.get("epi_wins", 0) + 1
        self.metrics["decisions"] += 1
        if used:
            self.metrics["fm_participated"] += 1
        # endurance bookkeeping (P-D): far commitments judged on his
        # ESTIMATED sleep pressure (his judgement, not ground truth)
        dist = math.hypot(intent[1] - f.x, intent[2] - f.y)
        if dist > 55.0:
            pred = self.selfmodel.reachable(dist, self.selfmodel.estimate("sleep", tick))
            self._endurance_pending.append([dist, self.selfmodel.estimate("sleep", tick), pred, intent[1], intent[2]])
        # resolve endurance predictions: arrived -> outcome by whether he
        # got there before sleeping; slept first -> failed prediction check
        still = []
        for (dd, pp, pred, tx, ty) in self._endurance_pending:
            if math.hypot(tx - f.x, ty - f.y) < 4.0:
                self.selfmodel.log_endurance(dd, pp, pred, True)
            else:
                still.append([dd, pp, pred, tx, ty])
        self._endurance_pending = still[-6:]
        return intent

    # ------------------------------------------------------------------
    # viewer / report snapshot
    # ------------------------------------------------------------------
    def _settled_calibration(self):
        """Calibration of the monitor on the claims it has settled so
        far (its own verification stream) — for the viewer's readout."""
        tape = self.monitor.tape
        if len(tape) < 30:
            return None
        bins = {}
        for r, c in tape:
            bins.setdefault(min(4, int(r * 5)), []).append((r, c))
        ece, n = 0.0, len(tape)
        for rows in bins.values():
            conf = sum(r for r, _ in rows) / len(rows)
            acc = sum(c for _, c in rows) / len(rows)
            ece += len(rows) / n * abs(conf - acc)
        rc = [r for r, c in tape if c == 1]
        ri = [r for r, c in tape if c == 0]
        return {"n": n, "ece": round(ece, 3),
                "mean_rel_correct": round(sum(rc) / max(1, len(rc)), 3),
                "mean_rel_incorrect": round(sum(ri) / max(1, len(ri)), 3)}

    def snapshot(self):
        ws = self.workspace
        return {
            "workspace": [c.row() for c in ws.contents],
            "queries_last": list(ws.last_queries),
            "attention": {"winner": self._prev_winner,
                          "schema_accuracy": [round(v, 3) for v in ws.schema.accuracy()]},
            "beliefs": {
                "objects": {k: {"kind": b["kind"], "fruit_est": round(b["fruit_est"], 1),
                                "conf": round(b["conf"], 2)}
                            for k, b in list(self.beliefs.objects.items())[:24]},
                "vars": {k: round(v["p"], 2) for k, v in self.beliefs.vars.items()},
                "epistemic_flags": len(self.beliefs.epistemic_flags),
            },
            "monitor": {"last": {k: round(v, 2) for k, v in list(self.monitor.last.items())[:8]},
                        "n_trained": self.monitor.n_trained,
                        "tape_n": len(self.tape_monitor),
                        "settled_calibration": self._settled_calibration()},
            "memory": {"episodes": len(self.episodic.episodes),
                       "semantic_facts": len(self.episodic.semantic),
                       "consolidations": self.episodic.consolidations},
            "self_model": {"drive_est": {k: round(self.selfmodel.estimate(k, self._tick), 1)
                                         for k in self.selfmodel.drive_est},
                           "home_conf": round(self.selfmodel.home_conf, 2),
                           "sleep_rise_rate": round(self.selfmodel.rise["sleep"], 3)},
            "affect": {"valence": round(self.affect.valence, 3),
                       "arousal": round(self.affect.arousal, 3)},
            "fwd_accuracy": (None if self.fwd.metrics["pred_n"] == 0 else
                             round(self.fwd.metrics["pred_correct"] / self.fwd.metrics["pred_n"], 3)),
        }

    # ------------------------------------------------------------------
    # (de)serialisation
    # ------------------------------------------------------------------
    def to_dict(self):
        return {
            "perception": self.perception.to_dict(),
            "monitor": self.monitor.to_dict(),
            "beliefs": self.beliefs.to_dict(),
            "fwd": self.fwd.to_dict(),
            "episodic": self.episodic.to_dict(),
            "selfmodel": self.selfmodel.to_dict(),
            "workspace": self.workspace.to_dict(),
            "affect": {"valence": self.affect.valence, "arousal": self.affect.arousal},
            "steers": self.steers,
            "active": self.active,
            "routing_enabled": self.routing_enabled,
            "last_pressure": list(self._last_pressure) if self._last_pressure else None,
            "chosen_intent": list(self.chosen_intent) if self.chosen_intent else None,
            "prev_scene_keys": dict(self._prev_scene_keys),
            "anchored": dict(self._anchored),
            "was_asleep": self._was_asleep,
            "prev_drives": self._prev_drives,
            "prev_feats": self._prev_feats,
            "prev_winner": self._prev_winner,
            "saw_pile_keys": sorted(self._saw_pile_keys),
            "storm_prev": self._storm_prev,
            "tree_obs": {str(k): list(v) for k, v in self._tree_obs.items()},
            "endurance_pending": [list(r) for r in self._endurance_pending],
            "last_sleep_count": self._last_sleep_count,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, d, rng):
        m = cls.__new__(cls)
        m.perception = Perception.from_dict(d["perception"])
        m.monitor = Monitor.from_dict(d["monitor"])
        m.beliefs = BeliefStore.from_dict(d["beliefs"])
        m.fwd = ForwardModel.from_dict(d["fwd"])
        m.episodic = EpisodicMemory.from_dict(d["episodic"])
        m.selfmodel = SelfModel.from_dict(d["selfmodel"])
        m.workspace = Workspace.from_dict(d["workspace"])
        m.affect = Affect()
        m.affect.valence = d["affect"]["valence"]
        m.affect.arousal = d["affect"]["arousal"]
        m.steers = d["steers"]
        m.active = d.get("active", True)
        m.routing_enabled = d.get("routing_enabled", True)
        m._tick = 0
        lp = d.get("last_pressure")
        m._last_pressure = tuple(lp) if lp else None
        m.chosen_intent = tuple(d["chosen_intent"]) if d["chosen_intent"] else None
        m._prev_scene_keys = dict(d["prev_scene_keys"])
        m._anchored = dict(d.get("anchored", {"hunger": False, "thirst": False}))
        m._was_asleep = d.get("was_asleep", False)
        m._prev_drives = d["prev_drives"]
        m._prev_feats = d["prev_feats"]
        m._prev_winner = d["prev_winner"]
        m._saw_pile_keys = set(d["saw_pile_keys"])
        m._storm_prev = d["storm_prev"]
        m._tree_obs = {int(k): list(v) for k, v in d["tree_obs"].items()}
        m._endurance_pending = [list(r) for r in d["endurance_pending"]]
        m._last_sleep_count = d["last_sleep_count"]
        m.metrics = dict(d["metrics"])
        m.tape_monitor = []
        m.decision_log = []
        return m
