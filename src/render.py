"""Render valley snapshots (PNG) from a live Sim state with PIL.
Used for the viewer's snapshot images — always drawn from actual sim state,
never hand-made."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image, ImageDraw

from config import WORLD_W, WORLD_H, SHELTER_POS, LANTERN_POS, BUILD_SITE
from world import elevation, is_water, in_forest, in_meadow

SCALE = 3


def render(sim, path, caption=""):
    W, H = int(WORLD_W * SCALE), int(WORLD_H * SCALE)
    img = Image.new("RGB", (W, H), (0, 0, 0))
    px = img.load()
    light = sim.fajr_light if hasattr(sim, "fajr_light") else 1.0
    from world import light_level
    tod = sim.tick % 1440
    light = light_level(tod)
    for gy in range(int(WORLD_H)):
        for gx in range(int(WORLD_W)):
            e = elevation(gx + 0.5, gy + 0.5)
            if is_water(gx + 0.5, gy + 0.5):
                col = (52, 96, 158)
            elif e > 0.78:
                col = (128, 122, 116)
            elif in_forest(gx + 0.5, gy + 0.5):
                col = (52 + int(e * 40), 96, 58)
            elif in_meadow(gx + 0.5, gy + 0.5):
                col = (104, 148, 74)
            else:
                col = (96 + int(e * 60), 128 + int(e * 30), 72)
            # night darkening + dawn/dusk warmth
            f = 0.22 + 0.78 * light
            col = tuple(int(c * f) for c in col)
            for sy in range(SCALE):
                for sx in range(SCALE):
                    px[gx * SCALE + sx, (int(WORLD_H) - 1 - gy) * SCALE + sy] = col
    d = ImageDraw.Draw(img)

    def P(x, y):
        return (x * SCALE, (WORLD_H - y) * SCALE)

    # trees
    for t in sim.world.trees:
        cx, cy = P(t.x, t.y)
        r = 3 + t.fruit
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(34, 92, 40), outline=(20, 60, 24))
        d.ellipse([cx - 1.5, cy - 1.5, cx + 1.5, cy + 1.5], fill=(110, 70, 34))
    # thickets
    for th in sim.world.thickets:
        cx, cy = P(th.x, th.y)
        r = th.r * SCALE
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(150, 60, 50), width=2)
        d.text((cx - 4, cy - 4), "x", fill=(200, 90, 80))
    # materials & structures
    for m in sim.world.materials:
        cx, cy = P(m.x, m.y)
        d.rectangle([cx - 2, cy - 2, cx + 2, cy + 2], fill=(160, 130, 90) if m.kind == "wood" else (150, 150, 150))
    for s in sim.world.structures:
        cx, cy = P(s["x"], s["y"])
        d.rectangle([cx - 3, cy - 3, cx + 3, cy + 3], fill=(178, 150, 104), outline=(90, 70, 40))
    # shelter + lantern + build site
    cx, cy = P(*SHELTER_POS)
    d.rectangle([cx - 9, cy - 7, cx + 9, cy + 7], fill=(196, 164, 110), outline=(80, 58, 30), width=2)
    d.polygon([(cx - 11, cy - 7), (cx + 11, cy - 7), (cx, cy - 15)], fill=(140, 84, 52))
    cx, cy = P(*LANTERN_POS)
    d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(255, 214, 90), outline=(200, 160, 40))
    # Fajr
    fx, fy = P(sim.fajr.x, sim.fajr.y)
    d.ellipse([fx - 7, fy - 7, fx + 7, fy + 7], outline=(255, 255, 255), width=2)
    d.ellipse([fx - 4, fy - 4, fx + 4, fy + 4], fill=(255, 140, 40))
    if sim.fajr.asleep:
        d.text((fx + 9, fy - 12), "Z z", fill=(220, 220, 255))
    if caption:
        d.rectangle([0, H - 18, W, H], fill=(20, 20, 24))
        d.text((6, H - 14), caption, fill=(235, 235, 235))
    img.save(path)
    return path
