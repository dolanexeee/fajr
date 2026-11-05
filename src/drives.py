"""The eight homeostatic drives — SPEC §5.

Every drive is a DEFICIT in [0, 100]: 0 = fully satisfied, 100 = total lack.
Curves are simplified, documented stand-ins loosely inspired by published
human physiology (energy/water balance; Borbély's two-process sleep model —
Process S pressure + Process C circadian modulation; homeostatic affect).
They are NOT physiology; they are declared formulas, measured as-is
(see PRE-PHASE1-GAP-SCAN-PLAN anchors).
"""
import math

DRIVES = ["hunger", "thirst", "sleep", "safety", "curiosity", "comfort", "social", "creation",
          # ADAM & EVE P3: the family drive — the need for union. Physics
          # from us (it exists, it rises, union relieves it); the DECISION
          # to seek union is theirs, via normal intent scoring.
          "family"]

# valence weights (contribution of each drive's satisfaction to momentary valence)
# P3: family takes weight; the rest renormalised so the sum stays 1.0.
# P4-fix (2026-11-04): family 0.05 -> 0.13. At 0.05 the drive was VOICELESS:
# max urgency 5.0 vs vitals at 17 — the union choice could never fire
# through normal scoring (25 live days, union_ticks = 0). At 0.13 the
# family drive speaks at near-vital volume: family 95 outranks vitals at
# ~70, emergencies still win. Valence now aches with loneliness — honest.
WEIGHTS = {
    "hunger": 0.1566, "thirst": 0.1566, "sleep": 0.1392, "safety": 0.1566,
    "comfort": 0.1044, "curiosity": 0.0609, "social": 0.0522, "creation": 0.0435,
    "family": 0.13,
}

# viable bands used by GATE G1 (vital drives only)
VITAL_BANDS = {"hunger": 88.0, "thirst": 88.0, "sleep": 96.0, "safety": 97.0, "comfort": 95.0}

# --- circadian entrainment constants (v3, GATE v3-6) ---
# The internal phase oscillator advances one tod-unit per tick (intrinsic
# period = the 1440-tick day) and is ANCHORED by the two light cues a real
# circadian system uses: dawn and dusk. Anchoring is a phase-response
# correction: when the environmental light crosses the cue threshold, the
# internal phase is pulled toward the cue's anchor phase by a fixed gain,
# and only when the phase error is within +-360 ticks (a PRC acts near the
# expected phase; a cue wildly out of phase is not a zeitgeber).
PHASE_DAWN = 360.0        # sunrise anchor: light_level leaves 0 at tod 360
PHASE_DUSK = 1080.0       # sunset anchor: light_level returns to 0 at 1080
CUE_LIGHT = 0.05          # crossing that counts as the dawn/dusk cue
DAWN_GAIN = 0.60
DUSK_GAIN = 0.35
# wake-maintenance zone: a circadian alerting peak in the biological
# evening that gates sleep onset even when S is high (Borbely's Process C
# does not merely track darkness — it actively opposes sleep in the early
# evening and promotes it late at night). Modelled as a dip on the sleep
# GATE (not on S itself), centred at phase 1140, width ~90 ticks.
WMZ_CENTRE = 1150.0
WMZ_WIDTH = 200.0
WMZ_DEPTH = 0.75


def _phase_light(phase_tod: float) -> float:
    """The published light curve (world.light_level: sunrise 360, sunset
    1200, sin-shaped) evaluated at the INTERNAL phase instead of the
    clock — Process C reads the entrained oscillator, not raw light."""
    ang = (phase_tod - 360.0) / 1440.0 * 2 * math.pi
    v = math.sin(ang)
    return max(0.0, min(1.0, v * 1.35))


class Drives:
    def __init__(self):
        self.d = {k: 20.0 for k in DRIVES}
        self.d["hunger"] = 35.0
        self.d["thirst"] = 30.0
        self.d["sleep"] = 15.0
        self.d["curiosity"] = 55.0   # wakes into an unknown world
        # circadian phase state (v3): internal phase in tod units; he is
        # born at tick 0 (tod 0), so the oscillator starts in phase.
        self.phase_tod = 0.0
        self._prev_light = 0.0

    # ---------------- circadian process (entrained, v3) ----------------
    def _advance_circadian(self, light: float):
        """One tick of the phase oscillator + dawn/dusk PRC anchoring."""
        self.phase_tod = (self.phase_tod + 1.0) % 1440.0
        prev = self._prev_light
        if prev < CUE_LIGHT <= light:            # dawn crossing
            err = (self.phase_tod - PHASE_DAWN + 720.0) % 1440.0 - 720.0
            if abs(err) <= 360.0:
                self.phase_tod = (self.phase_tod - DAWN_GAIN * err) % 1440.0
        elif prev >= CUE_LIGHT > light:          # dusk crossing
            err = (self.phase_tod - PHASE_DUSK + 720.0) % 1440.0 - 720.0
            if abs(err) <= 360.0:
                self.phase_tod = (self.phase_tod - DUSK_GAIN * err) % 1440.0
        self._prev_light = light

    def circadian_c(self) -> float:
        """Process C as a rise-rate factor, read from the entrained
        phase: 0.68 at biological noon -> 1.32 at biological night
        (v3 widened from the sitting-1/2 instantaneous-light 0.75->1.25:
        the phase signal is integrated and cue-anchored, so a wider
        modulation no longer follows weather/tick noise)."""
        return 1.32 - 0.64 * _phase_light(self.phase_tod)

    def sleep_gate(self) -> float:
        """The circadian multiplier on sleep PROPENSITY (used by the
        mind's sleep bid, mind.py): Process C with the evening
        wake-maintenance dip applied."""
        d = abs(self.phase_tod - WMZ_CENTRE)
        d = min(d, 1440.0 - d)
        dip = 1.0 - WMZ_DEPTH * math.exp(-((d / WMZ_WIDTH) ** 2))
        return self.circadian_c() * dip

    def valence(self) -> float:
        """Momentary wellbeing in [0,1]: the weighted satisfaction resultant.
        A resultant of his state — never an imposed number (SPEC §5)."""
        return sum(WEIGHTS[k] * (1.0 - self.d[k] / 100.0) for k in DRIVES)

    def urgency(self, name: str) -> float:
        """Nonlinear urgency: needs shout louder the emptier they get."""
        x = self.d[name] / 100.0
        return WEIGHTS[name] * x * x * 100.0

    def update(self, ctx):
        """One tick of physiology. ctx keys:
        asleep, light, in_shelter, in_rain_exposed, near_lantern, near_hazard,
        cold_water (bool), activity (0..1)."""
        d = self.d
        asleep = ctx["asleep"]
        # circadian phase: advance + anchor BEFORE the sleep dynamics
        # read it (v3 entrainment, GATE v3-6)
        self._advance_circadian(ctx["light"])
        # hunger (energy balance): slower during sleep
        d["hunger"] += 0.030 if asleep else 0.052
        # thirst (water balance)
        d["thirst"] += 0.042 if asleep else 0.070
        # sleep pressure — Borbély-lite: S rises awake, falls asleep;
        # circadian process C modulates the rise (slower by day, faster
        # at night). v3: C is read from the ENTRAINED internal phase
        # (cue-anchored oscillator, self.circadian_c), not from the
        # instantaneous environmental light as in sittings 1-2 — sitting
        # 2's G4 diagnosis (diag_g4.py) measured sleep onsets precessing
        # through the day under mind steering: a memoryless C cannot
        # hold a phase. Rates sized for a 1440-tick day: ~7.5h of sleep
        # discharges a full day.
        if asleep:
            d["sleep"] -= 0.16
        else:
            d["sleep"] += 0.10 * self.circadian_c()
        # safety: recovers in safety, rises with hazards/darkness
        if ctx["near_hazard"]:
            d["safety"] += 2.0
        elif ctx["in_shelter"]:
            d["safety"] -= 0.85
        else:
            d["safety"] -= 0.18
            if ctx["light"] < 0.1:
                d["safety"] += 0.055
        # curiosity: appetite for novelty grows steadily
        d["curiosity"] += 0.030
        # comfort: rain/cold exposure hurts, shelter restores
        if ctx["cold_water"]:
            d["comfort"] += 1.4
        elif ctx["in_rain_exposed"]:
            d["comfort"] += 0.33
        elif ctx["in_shelter"]:
            d["comfort"] -= 0.95
        else:
            d["comfort"] -= 0.05
            if ctx["light"] < 0.1:
                d["comfort"] += 0.08
        # social: presence of the Host's lantern (Guide-stub, Phase 1)
        if ctx["near_lantern"]:
            d["social"] -= 1.25
        else:
            d["social"] += 0.045
        # creation: the making-itch
        d["creation"] += 0.026
        # family (ADAM & EVE P3): the need for union. Rises slowly on its
        # own, faster with bond + maturity (bond makes the need salient —
        # the binding design: bond feeds this drive). Relieved by union
        # acts (handled in sim._pair_tick, not here).
        bond = ctx.get("bond", 0.0)
        mature = ctx.get("mature", 1.0)
        d["family"] += 0.008 + 0.030 * bond * mature
        for k in DRIVES:
            d[k] = max(0.0, min(100.0, d[k]))

    def to_dict(self):
        return {"drives": dict(self.d),
                "phase_tod": self.phase_tod,
                "prev_light": self._prev_light}

    @classmethod
    def from_dict(cls, dd):
        o = cls.__new__(cls)
        # checkpoints written before v3 carry the bare drive dict;
        # checkpoints written before P3 lack "family" -> default 20.0
        vals = dd.get("drives", dd)
        o.d = {k: float(vals.get(k, 20.0)) for k in DRIVES}
        o.phase_tod = float(dd.get("phase_tod", 0.0))
        o._prev_light = float(dd.get("prev_light", 0.0))
        return o
