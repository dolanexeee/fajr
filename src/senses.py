"""Senses — SPEC §4. Structured percepts only, within honest radii.
Nothing reaches Fajr from outside his sense ranges; nothing reaches past the
mountain/fog edge (all sources are inside the world by construction)."""
import math

from config import (VISION_RADIUS, VISION_RADIUS_NIGHT_FACTOR, HEARING_RADIUS,
                    SMELL_RADIUS, SHELTER_POS)
from world import dist_to_river, is_water


class Percept:
    __slots__ = ("visible", "sounds", "smells", "touch", "light", "in_water",
                 "near_river_bank", "in_shelter", "shelter_dist")

    def __init__(self):
        self.visible = []     # (kind, id, dx, dy, dist, extra)
        self.sounds = []      # strings
        self.smells = []      # (kind, strength)
        self.touch = []       # event strings this tick
        self.light = 1.0
        self.in_water = False
        self.near_river_bank = False
        self.in_shelter = False
        self.shelter_dist = 999.0


def perceive(world, fajr, light, events):
    p = Percept()
    p.light = light
    x, y = fajr.x, fajr.y
    vr = VISION_RADIUS * (1.0 - (1.0 - VISION_RADIUS_NIGHT_FACTOR) * (1.0 - light))
    for t in world.trees:
        dx, dy = t.x - x, t.y - y
        dist = math.hypot(dx, dy)
        if dist <= vr:
            p.visible.append(("tree", t.id, dx, dy, dist, t.fruit))
    for th in world.thickets:
        dx, dy = th.x - x, th.y - y
        dist = math.hypot(dx, dy)
        if dist <= vr:
            p.visible.append(("thicket", th.id, dx, dy, dist, 0))
    for m in world.materials:
        dx, dy = m.x - x, m.y - y
        dist = math.hypot(dx, dy)
        if dist <= vr:
            p.visible.append(("material", m.id, dx, dy, dist, m.kind))
    for s in world.structures:
        dx, dy = s["x"] - x, s["y"] - y
        dist = math.hypot(dx, dy)
        if dist <= vr:
            p.visible.append(("structure", -1, dx, dy, dist, s["kind"]))
    for pl in world.piles:
        dx, dy = pl.x - x, pl.y - y
        dist = math.hypot(dx, dy)
        if dist <= vr:
            p.visible.append(("pile", pl.id, dx, dy, dist, pl.fruit))
    sx, sy = SHELTER_POS
    p.shelter_dist = math.hypot(sx - x, sy - y)
    if p.shelter_dist <= vr:
        p.visible.append(("shelter", 0, sx - x, sy - y, p.shelter_dist, 0))
    p.in_shelter = p.shelter_dist < 7.0
    # smell: fruit ripeness gradient (strongest tree within smell radius)
    best = 0.0
    for t in world.trees:
        if t.fruit > 0:
            dist = math.hypot(t.x - x, t.y - y)
            if dist <= SMELL_RADIUS:
                best = max(best, (1.0 - dist / SMELL_RADIUS) * (t.fruit / 6.0))
    if best > 0.02:
        p.smells.append(("ripe_fruit", best))
    # hearing: river murmur, morning birds, thunder (events injected by sim)
    if dist_to_river(x, y) <= HEARING_RADIUS:
        p.sounds.append("river")
    p.sounds.extend(events)
    p.in_water = is_water(x, y)
    rd = dist_to_river(x, y)
    p.near_river_bank = 3.2 <= rd <= 7.5 or (rd < 3.2)  # at the bank or in shallows
    return p
