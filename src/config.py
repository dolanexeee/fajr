"""World & body constants for Phase 1.

Time: 1 tick = 1 simulated minute. His day = 1440 ticks.
At life speed his day lasts 12 real minutes (SPEC §3), i.e. the live runner
advances 2 ticks per real second. Tests run unthrottled.
"""

# --- world geometry (continuous 2D, units ~= a few metres) ---
WORLD_W = 260.0
WORLD_H = 180.0
CELL = 10.0                      # place-memory grid cell size
GRID_W = int(WORLD_W // CELL)    # 26
GRID_H = int(WORLD_H // CELL)    # 18

SHELTER_POS = (150.0, 95.0)      # the shelter / home (SPEC §3)
LANTERN_POS = (146.0, 99.0)      # the Host's lantern — Guide-stub presence (Phase 1 stub for §7)
BUILD_SITE = (158.0, 88.0)       # where placed materials accumulate into structures

TICKS_PER_DAY = 1440

# --- senses (SPEC §4 — honest limits, no super-senses) ---
VISION_RADIUS = 26.0
VISION_RADIUS_NIGHT_FACTOR = 0.55
HEARING_RADIUS = 42.0
SMELL_RADIUS = 32.0
SOCIAL_RADIUS = 9.0              # lantern presence radius

# --- body ---
WALK_SPEED = 1.15                # units per tick
WATER_SLOW_FACTOR = 0.45
EAT_TICKS_PER_FRUIT = 3
FRUIT_SATIATION = 34.0
DRINK_RELIEF = 55.0

# --- water depth zones & hygiene (Stage 2, WATER-GATE) ---
SWIM_SLOW_FACTOR = 0.30      # movement factor in the deep channel (swimming);
                             # the shallows keep WATER_SLOW_FACTOR (wading)
BATHE_TICKS = 25             # duration of a deliberate bath, in ticks

# --- swimming skill (expansion 2026-11-01) ---
# Skill is DERIVED from lifetime swim_ticks (persisted in stats): no new
# state, no migration, past swims count. Asymptotic: 1 - exp(-ticks/TAU).
SWIM_SKILL_TAU = 400.0       # swim_ticks for skill to reach ~63%
SWIM_SLOW_SKILLED = 0.80     # deep-water slow factor at full skill (from 0.30)
SWIM_COST_RELIEF = 0.5       # fraction of swim energy cost removed at skill=1
                             # (W5 cost asymmetry ratio hunger:thirst is preserved
                             #  at all skill levels — only magnitude scales)
