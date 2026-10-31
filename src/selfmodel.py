"""Self-model — Phase 2 (SPEC §6; battery probe P-D).

His estimates of HIMSELF, kept apart from ground truth and used by his
planning:

  * drive estimates: his interoception is NOISY (felt = true + noise,
    sampled hourly). Between samples the estimate is PROJECTED forward
    with a learned per-drive rise rate (learned from consecutive felt
    samples — never from ground truth). The bias is corrected only at
    anchor events he can feel directly (collapse into sleep, extreme
    hunger/thirst), never by reading truth.
  * endurance: reachable() projects sleep pressure over a trip using the
    learned sleep rise rate; planning calls it with the ESTIMATED
    pressure, so the judgement is his, fallible as he is.
  * home confidence and preferences: where home is (confidence grows
    with sleeps there), and which activities have felt good.
"""
import math

FELT_NOISE = 6.0        # interoceptive noise sigma (declared)
EST_EMA = 0.5           # smoothing of felt samples
RISE0 = {"hunger": 0.055, "thirst": 0.075, "sleep": 0.11}   # declared priors
RISE_SPAN = 480         # rise rates are learned over >= this many ticks of
                        # felt history (short spans are noise-dominated)


class SelfModel:
    def __init__(self):
        self.drive_est = {"hunger": 35.0, "thirst": 30.0, "sleep": 15.0}  # at est_tick
        self.drive_bias = {"hunger": 0.0, "thirst": 0.0, "sleep": 0.0}
        self.est_tick = 0
        self.rise = dict(RISE0)          # points per tick, learned
        self._felt_hist = []             # [(tick, {drive: felt}, asleep)]
        self.last_sample_tick = -10 ** 9
        self.home_conf = 0.5
        self.home_pos = None
        self.preferences = {}            # key -> running mean outcome
        self.metrics = {"samples": 0, "anchors": 0,
                        "endurance_queries": 0, "endurance_log": []}

    # ---------------- estimates ----------------
    def estimate(self, drive, tick, asleep=False):
        """Current best estimate: base projected by the learned rise.
        During sleep the sleep drive RECOVERS (falls) — there is no
        upward projection for it while asleep."""
        dt = max(0, tick - self.est_tick)
        if asleep and drive == "sleep":
            return max(0.0, min(100.0, self.drive_est[drive]))
        return max(0.0, min(100.0, self.drive_est[drive] + self.rise[drive] * dt))

    def sample(self, drives_true, tick, rng, force=False, asleep=False):
        """Hourly (or forced) felt sample: learn rise rates from long felt
        spans, then smooth the base estimate toward the felt value."""
        if not force and tick - self.last_sample_tick < 60:
            return
        self.last_sample_tick = tick
        felt_now = {k: drives_true[k] + rng.gauss(0.0, FELT_NOISE)
                    for k in self.drive_est}
        # rise learning, per drive: the NEWEST clean span of >= RISE_SPAN
        # ticks ending now. A span is clean for a drive if no consecutive
        # sample pair inside it drops sharply (a relief event — meal,
        # drink, sleep — is an action outcome, not physiology, and would
        # corrupt the rate). Sleep additionally requires the span awake.
        hist = self._felt_hist
        if hist:
            for k in self.drive_est:
                for i in range(len(hist) - 1, -1, -1):
                    t0, f0, _a0 = hist[i]
                    if tick - t0 < RISE_SPAN:
                        continue
                    slice_ = hist[i:] + [(tick, felt_now, asleep)]
                    clean = True
                    for j in range(len(slice_) - 1):
                        if slice_[j + 1][1][k] - slice_[j][1][k] < -10.0:
                            clean = False
                            break
                        if k == "sleep" and slice_[j][2]:
                            clean = False
                            break
                    if not clean or (k == "sleep" and asleep):
                        continue
                    delta = felt_now[k] - f0[k]
                    if delta > -6.0:
                        rate = delta / (tick - t0)
                        if 0.0 <= rate < 0.40:
                            self.rise[k] = self.rise[k] * 0.75 + rate * 0.25
                    break
        self._felt_hist.append((tick, felt_now, asleep))
        if len(self._felt_hist) > 40:
            self._felt_hist = self._felt_hist[-40:]
        for k in self.drive_est:
            # no upward projection of sleep pressure while asleep: the
            # base simply tracks the felt recovery
            if asleep and k == "sleep":
                projected = self.drive_est[k]
            else:
                projected = self.drive_est[k] + self.rise[k] * max(0, tick - self.est_tick)
            self.drive_est[k] = max(0.0, min(100.0,
                                        projected * (1 - EST_EMA) + (felt_now[k] + self.drive_bias[k]) * EST_EMA))
        self.est_tick = tick
        self.metrics["samples"] += 1

    def anchor(self, drive, true_value, tick):
        """An outcome he can feel directly (collapse, extreme states)
        corrects the bias for that drive."""
        err = true_value - self.estimate(drive, tick)
        self.drive_bias[drive] = max(-15.0, min(15.0, self.drive_bias[drive] + 0.4 * err))
        self.drive_est[drive] = true_value
        self.est_tick = tick
        self.metrics["anchors"] += 1

    # ---------------- endurance ----------------
    def reachable(self, distance, sleep_pressure_est):
        """Can he cover `distance` before (estimated) pressure would hit
        ~99? Judged on his estimate and his learned rise rate."""
        from config import WALK_SPEED
        travel_ticks = distance / WALK_SPEED
        projected = sleep_pressure_est + self.rise["sleep"] * travel_ticks
        self.metrics["endurance_queries"] += 1
        return projected <= (99.0 - 12.0)

    def log_endurance(self, distance, sleep_pressure, prediction, outcome):
        self.metrics["endurance_log"].append(
            [round(distance, 1), round(sleep_pressure, 1), 1 if prediction else 0,
             1 if outcome else 0])

    # ---------------- preferences / home ----------------
    def note_outcome(self, key, outcome):
        cur = self.preferences.get(key)
        self.preferences[key] = outcome if cur is None else cur * 0.8 + outcome * 0.2

    def note_home_sleep(self, pos):
        self.home_pos = list(pos)
        self.home_conf = min(1.0, self.home_conf + 0.12)

    def to_dict(self):
        return {"drive_est": dict(self.drive_est), "drive_bias": dict(self.drive_bias),
                "est_tick": self.est_tick, "rise": dict(self.rise),
                "felt_hist": [[t, dict(f), a] for (t, f, a) in self._felt_hist],
                "last_sample_tick": self.last_sample_tick,
                "home_conf": self.home_conf, "home_pos": self.home_pos,
                "preferences": dict(self.preferences),
                "metrics": {"samples": self.metrics["samples"],
                            "anchors": self.metrics["anchors"],
                            "endurance_queries": self.metrics["endurance_queries"],
                            "endurance_log": [list(r) for r in self.metrics["endurance_log"]]}}

    @classmethod
    def from_dict(cls, d):
        s = cls.__new__(cls)
        s.drive_est = dict(d["drive_est"])
        s.drive_bias = dict(d["drive_bias"])
        s.est_tick = d["est_tick"]
        s.rise = dict(d["rise"])
        s._felt_hist = [(t, dict(f), a) for (t, f, a) in d["felt_hist"]]
        s.last_sample_tick = d["last_sample_tick"]
        s.home_conf = d["home_conf"]
        s.home_pos = d["home_pos"]
        s.preferences = dict(d["preferences"])
        s.metrics = dict(d["metrics"])
        s.metrics["endurance_log"] = [list(r) for r in d["metrics"]["endurance_log"]]
        return s
