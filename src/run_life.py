"""Run Fajr's life.

Batch (evidence runs):
  python3 run_life.py --seed 11 --days 7 --tag seed11
    -> results/life_seed11_checkpoint.json, results/life_seed11_journal.json,
       viewer/state.json + viewer/index.html (embedded snapshot),
       viewer/snapshots/{dawn,noon,dusk,night}.png, journal_sample.md

Live (his ongoing biography — resumes the same life from life/fajr_life.json):
  python3 run_life.py --live [--seed 11]
    advances 2 ticks per real second (SPEC §3: his day = 12 real minutes),
    refreshes the viewer export every few seconds, checkpoints regularly.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SHELTER_POS, LANTERN_POS, BUILD_SITE, TICKS_PER_DAY
from sim import Sim, fmt_time
from world import elevation, is_water, in_forest, in_meadow, light_level
import render

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_raster():
    cells = []
    for gy in range(90):
        for gx in range(130):
            x, y = gx * 2 + 1, gy * 2 + 1
            flags = (1 if is_water(x, y) else 0) | (2 if in_forest(x, y) else 0) | (4 if in_meadow(x, y) else 0)
            cells.append([round(elevation(x, y) * 255), flags])
    return cells


RASTER = None


def export_state(sim):
    global RASTER
    if RASTER is None:
        RASTER = build_raster()
    tod = sim.tick % TICKS_PER_DAY
    f = sim.fajr
    return {
        "tick": sim.tick, "day": sim.tick // TICKS_PER_DAY + 1, "time": fmt_time(tod),
        "light": round(light_level(tod), 3), "weather": sim.world.weather,
        "season_phase": round(sim.world.season_phase, 3),
        "raster": RASTER,
        "world": {
            "trees": [{"x": t.x, "y": t.y, "fruit": t.fruit} for t in sim.world.trees],
            "thickets": [{"x": t.x, "y": t.y, "r": t.r} for t in sim.world.thickets],
            "structures": sim.world.structures,
            "shelter": list(SHELTER_POS), "lantern": list(LANTERN_POS),
        },
        "fajr": {"x": round(f.x, 1), "y": round(f.y, 1), "asleep": f.asleep,
                 "drives": {k: round(v, 1) for k, v in f.drives.d.items()},
                 "valence": round(f.drives.valence(), 3),
                 "intent": (f.intent[0] if f.intent else None),
                 "carry": list(f.carry)},
        "mind": (f.mind.snapshot() if getattr(f, "mind", None) is not None else None),
        "stats": dict(f.stats),
        "journal_tail": sim.journal.entries[-16:],
    }


def write_viewer(sim, viewer_dir):
    os.makedirs(viewer_dir, exist_ok=True)
    state = export_state(sim)
    with open(os.path.join(viewer_dir, "state.json"), "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False)
    tpl_path = os.path.join(viewer_dir, "template.html")
    with open(tpl_path, encoding="utf-8") as fh:
        html = fh.read()
    html = html.replace("/*__STATE__*/null/*__END__*/", json.dumps(state, ensure_ascii=False))
    with open(os.path.join(viewer_dir, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(html)


def write_journal_sample(sim, path):
    entries = sim.journal.entries
    picked, seen_types = [], set()
    for e in entries:
        if e["type"] in ("awakening", "day_summary") or e["type"] not in seen_types:
            picked.append(e)
            seen_types.add(e["type"])
    lines = ["# فجر — مقتطف من سجل حياته (7 أيام، بذرة 11)", "",
             "سجل آلي كامل كتبته المحاكاة أثناء حياته — لا نصّ مُعدّ سلفًا.", ""]
    for e in picked:
        lines.append(f"- **اليوم {e['day']} · {e['time']}** — {e['text']}")
    f = sim.fajr
    lines += ["", f"**حصيلة الأسبوع:** أكل {f.stats['eats']} مرة، شرب {f.stats['drinks']}، "
                  f"نام {f.stats['sleeps']} مرة، جمع {f.stats['gathers']}، بنى {f.stats['places']}، "
                  f"وخزات شوك {f.stats['pricks']}، صدمات ماء بارد {f.stats['cold_shocks']}، "
                  f"قطع {f.stats['distance']:.0f} وحدة مشيًا."]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def batch(seed, days, tag):
    sim = Sim(seed=seed)
    results = os.path.join(ROOT, "results")
    viewer = os.path.join(ROOT, "viewer")
    snaps = os.path.join(viewer, "snapshots")
    os.makedirs(results, exist_ok=True)
    os.makedirs(snaps, exist_ok=True)
    targets = {1 * TICKS_PER_DAY + 390: "dawn", 1 * TICKS_PER_DAY + 750: "noon",
               1 * TICKS_PER_DAY + 1155: "dusk", 2 * TICKS_PER_DAY + 90: "night"}
    total = days * TICKS_PER_DAY
    for i in range(total):
        sim.step()
        if sim.tick in targets:
            name = targets[sim.tick]
            render.render(sim, os.path.join(snaps, f"{name}.png"),
                          caption=f"Fajr - day {sim.tick // TICKS_PER_DAY + 1} {fmt_time(sim.tick % TICKS_PER_DAY)} - {name} - weather {sim.world.weather}")
    sim.save(os.path.join(results, f"life_{tag}_checkpoint.json"))
    with open(os.path.join(results, f"life_{tag}_journal.json"), "w", encoding="utf-8") as fh:
        json.dump(sim.journal.entries, fh, ensure_ascii=False, indent=1)
    write_viewer(sim, viewer)
    write_journal_sample(sim, os.path.join(ROOT, "journal_sample.md"))
    print(f"batch done: seed={seed} days={days} tick={sim.tick} journal={sim.journal.total_count} entries")


def live(seed):
    life_dir = os.path.join(ROOT, "life")
    os.makedirs(life_dir, exist_ok=True)
    ckpt = os.path.join(life_dir, "fajr_life.json")
    if os.path.exists(ckpt):
        sim = Sim.load(ckpt)
        print(f"resumed his biography at tick {sim.tick} (day {sim.tick // TICKS_PER_DAY + 1})")
    else:
        sim = Sim(seed=seed)
        print("a new life begins")
    viewer = os.path.join(ROOT, "viewer")
    last_export, last_save = 0.0, 0.0
    try:
        while True:
            t0 = time.time()
            sim.step()
            sim.step()  # 2 ticks per real second -> his day = 12 real minutes
            now = time.time()
            if now - last_export > 3:
                write_viewer(sim, viewer)
                last_export = now
            if now - last_save > 60:
                sim.save(ckpt)
                last_save = now
            dt = time.time() - t0
            if dt < 1.0:
                time.sleep(1.0 - dt)
    except KeyboardInterrupt:
        sim.save(ckpt)
        print(f"paused — biography saved at tick {sim.tick}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--tag", default="seed11")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    if a.live:
        live(a.seed)
    else:
        batch(a.seed, a.days, a.tag)
