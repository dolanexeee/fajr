"""STAGE 2 — WATER & HYGIENE BATTERY.

Gates and constructions frozen in WATER-GATE.md before this run.
Nothing here relaxes a threshold; failures are reported as failures.
Writes results/water_gate.json (incrementally, crash-safe) and
journal_sample_water.md. Earlier batteries' files are never written:
the Phase-1 regression is redirected via ALS_GATE_OUT, and the
Phase-2 spot checks import the sitting-3 battery's measurement
functions without calling its writers.

Run: python3 tests/water_battery.py
"""
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import TICKS_PER_DAY  # noqa: E402
from sim import Sim  # noqa: E402
from world import depth_zone, river_center_x, is_water, Pile  # noqa: E402

SEEDS_LIVES = [11, 42, 97, 101, 202, 303]
SEEDS_BATTERY = [11, 42, 97]
OUT_PATH = os.path.join(ROOT, "results", "water_gate.json")

R = {"gate": "WATER-GATE.md (Stage 2, frozen 2026-10-23)",
     "lives": {}, "probe_west": {}, "gates": {}, "regression": {}}


def save():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=2)


def gate(name, passed, detail):
    R["gates"][name] = {"pass": bool(passed), **detail}
    save()
    print(f"[{name}] {'PASS' if passed else 'FAIL'}", flush=True)


# ------------------------------------------------------------------
# PART A: measurement lives
# ------------------------------------------------------------------
def measurement_life(seed, days=7):
    s = Sim(seed=seed)
    f = s.fajr
    r = {"baths": 0, "bath_end_clean": [], "swims": 0, "crossings": 0,
         "periods": 0, "periods_answered": 0,
         "wet_entries": 0, "wet_entry_ok": 0, "wet_exits_clean": 0,
         "wet_dry_ok": 0, "day_end_clean": [], "min_clean": 100.0,
         "max_deep_stint": 0, "asleep_deep_ticks": 0,
         "h_deep": [], "h_land": [], "comfort_hi": [], "comfort_lo": [],
         "first_bath_day": None, "final_zone": None}
    side = None
    deep_tr = 0
    period = None
    entry = None
    exit_at = None
    prev_inw = False
    for t in range(days * TICKS_PER_DAY):
        h0 = f.drives.d["hunger"]
        d0 = f.stats["distance"]
        s.step()
        ev = f.mind_events or {}
        zone = depth_zone(f.x, f.y)
        inw = zone != "dry"
        awake = not f.asleep
        moved = f.stats["distance"] - d0
        if f.asleep and zone == "deep":
            r["asleep_deep_ticks"] += 1
        r["max_deep_stint"] = max(r["max_deep_stint"], f._deep_stint)
        if awake and moved > 0.3 and not ev.get("ate"):
            dh = f.drives.d["hunger"] - h0
            if zone == "deep":
                r["h_deep"].append(dh)
            elif zone == "dry":
                r["h_land"].append(dh)
        if ev.get("bathed"):
            r["bath_end_clean"].append(round(f.cleanliness, 1))
            if r["first_bath_day"] is None:
                r["first_bath_day"] = s.tick // TICKS_PER_DAY + 1
        # river crossings
        cx = river_center_x(f.y)
        cur = 1 if f.x > cx + 6.0 else (-1 if f.x < cx - 6.0 else side)
        if zone == "deep":
            deep_tr += 1
        if side is not None and cur is not None and cur != side:
            if deep_tr >= 3:
                r["crossings"] += 1
            deep_tr = 0
            side = cur
        elif side is None:
            side = cur
        # dirty periods
        if period is None and f.cleanliness < 55.0:
            period = {"start": s.tick, "water": 0}
            r["periods"] += 1
        if period is not None:
            if inw:
                period["water"] += 1
            if ev.get("bathed") or period["water"] >= 15 or f.cleanliness >= 75.0:
                r["periods_answered"] += 1
                period = None
            elif s.tick - period["start"] > TICKS_PER_DAY:
                period = None
        # wet entry quality (>= 8 ticks in water)
        if inw and not prev_inw:
            entry = {"ticks": 0, "max_wet": f.wet}
        if entry is not None:
            entry["ticks"] += 1
            entry["max_wet"] = max(entry["max_wet"], f.wet)
            if entry["ticks"] >= 8:
                r["wet_entries"] += 1
                if entry["max_wet"] >= 0.8:
                    r["wet_entry_ok"] += 1
                entry = None
        if prev_inw and not inw:
            exit_at = s.tick
        if exit_at is not None:
            if inw:
                exit_at = None
            elif f.wet < 0.2:
                r["wet_exits_clean"] += 1
                if s.tick - exit_at <= 240:
                    r["wet_dry_ok"] += 1
                exit_at = None
        prev_inw = inw
        if awake and zone == "dry":
            if f.grime() >= 45.0:
                r["comfort_hi"].append(f.drives.d["comfort"])
            elif f.grime() <= 25.0:
                r["comfort_lo"].append(f.drives.d["comfort"])
        r["min_clean"] = min(r["min_clean"], f.cleanliness)
        if (s.tick + 1) % TICKS_PER_DAY == 0:
            r["day_end_clean"].append(round(f.cleanliness, 1))
    r["baths"] = f.stats["bathes"]
    r["swims"] = f.stats["swims"]
    r["swim_ticks"] = f.stats["swim_ticks"]
    r["cold_shocks"] = f.stats["cold_shocks"]
    r["max_pinned"] = dict(s.metrics["max_pinned"])
    r["final_zone"] = depth_zone(f.x, f.y)
    r["bathe_value_end"] = round(f.bathe_value, 3)
    r["episodes_bathe"] = sum(1 for e in f.mind.episodic.episodes if e["kind"] == "bathe")
    r["episodes_swim"] = sum(1 for e in f.mind.episodic.episodes if e["kind"] == "swim")
    r["semantic_water_cleans"] = f.mind.episodic.get_semantic("water:cleans")
    r["h_deep_mean"] = round(sum(r["h_deep"]) / len(r["h_deep"]), 4) if r["h_deep"] else None
    r["h_land_mean"] = round(sum(r["h_land"]) / len(r["h_land"]), 4) if r["h_land"] else None
    r["h_deep_n"] = len(r["h_deep"])
    r["h_land_n"] = len(r["h_land"])
    r["comfort_hi_mean"] = round(sum(r["comfort_hi"]) / len(r["comfort_hi"]), 2) if r["comfort_hi"] else None
    r["comfort_lo_mean"] = round(sum(r["comfort_lo"]) / len(r["comfort_lo"]), 2) if r["comfort_lo"] else None
    r["comfort_hi_n"] = len(r["comfort_hi"])
    r["comfort_lo_n"] = len(r["comfort_lo"])
    del r["h_deep"], r["h_land"], r["comfort_hi"], r["comfort_lo"]
    if seed == 11:
        keep = ("awakening", "bathe", "swim", "drink", "cold_shock", "day_summary")
        lines = ["# Journal sample — Stage 2 water & hygiene (seed 11, week 1)",
                 "",
                 "Water-relevant entries from the life's own journal, verbatim:", ""]
        for e in s.journal.entries:
            if e["type"] in keep:
                lines.append(f"- day {e['day']} {e['time']} — {e['text']} "
                             f"({e['type']}, x={e['x']}, y={e['y']})")
        with open(os.path.join(ROOT, "journal_sample_water.md"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    return r


def west_probe(seed, days=4):
    s = Sim(seed=seed)
    s.world.piles.append(Pile(900, 30.0, 95.0, 6))
    f = s.fajr
    deep_ticks = 0
    for _ in range(days * TICKS_PER_DAY):
        s.step()
        if depth_zone(f.x, f.y) == "deep":
            deep_ticks += 1
    pile = [p for p in s.world.piles if p.id == 900]
    left = pile[0].fruit if pile else 0
    return {"seed": seed, "pile_fruit_start": 6, "pile_fruit_left": left,
            "eaten": 6 - left, "deep_ticks": deep_ticks,
            "crossings_note": "transit measured by deep ticks + fruit eaten"}


def score_water_gates():
    lives = R["lives"]
    # W1
    n_min_ok = sum(1 for r in lives.values() if r["min_clean"] <= 62.0)
    d1 = sum(r["day_end_clean"][0] for r in lives.values()) / len(lives)
    d3 = sum(r["day_end_clean"][2] for r in lives.values()) / len(lives)
    gate("W1_dirt_accumulates", n_min_ok >= 5 and (d1 - d3) >= 8.0,
         {"lives_min_clean_le_62": n_min_ok, "mean_day1_end": round(d1, 1),
          "mean_day3_end": round(d3, 1), "decline": round(d1 - d3, 1)})
    # W2
    ends = [c for r in lives.values() for c in r["bath_end_clean"]]
    gate("W2_bath_restores", len(ends) >= 6 and all(c >= 90.0 for c in ends),
         {"baths_total": len(ends), "bath_end_clean": ends})
    # W3
    per = sum(r["periods"] for r in lives.values())
    ans = sum(r["periods_answered"] for r in lives.values())
    lives_bath = sum(1 for r in lives.values() if r["baths"] >= 1)
    gate("W3_bathes_by_choice", per > 0 and (ans / per) >= 0.50 and lives_bath >= 5,
         {"dirty_periods": per, "answered": ans,
          "answered_fraction": round(ans / per, 3) if per else None,
          "lives_with_bath": lives_bath})
    # W4
    lives_swim = sum(1 for r in lives.values() if r["swims"] >= 1)
    lives_cross = sum(1 for r in lives.values() if r["crossings"] >= 1)
    probe_eaten = sum(1 for p in R["probe_west"].values() if p["eaten"] >= 1)
    gate("W4_swim_transit", lives_swim >= 5 and lives_cross >= 2 and probe_eaten >= 2,
         {"lives_with_swim": lives_swim, "lives_with_crossing": lives_cross,
          "probe_seeds_eaten": probe_eaten})
    # W5
    hd = [r["h_deep_mean"] for r in lives.values() if r["h_deep_mean"]]
    hl = [r["h_land_mean"] for r in lives.values() if r["h_land_mean"]]
    ratio = (sum(hd) / len(hd)) / (sum(hl) / len(hl)) if hd and hl else None
    gate("W5_swim_energy_cost", ratio is not None and ratio >= 1.40,
         {"h_deep_mean": round(sum(hd) / len(hd), 4) if hd else None,
          "h_land_mean": round(sum(hl) / len(hl), 4) if hl else None,
          "ratio": None if ratio is None else round(ratio, 3)})
    # W6
    pin_ok = all(v <= 240 for r in lives.values() for v in r["max_pinned"].values())
    stint_ok = all(r["max_deep_stint"] <= 300 for r in lives.values())
    asleep_ok = all(r["asleep_deep_ticks"] == 0 for r in lives.values())
    gate("W6_no_water_harm", pin_ok and stint_ok and asleep_ok,
         {"max_pinned_all_le_240": pin_ok,
          "max_deep_stint": max(r["max_deep_stint"] for r in lives.values()),
          "asleep_deep_ticks_total": sum(r["asleep_deep_ticks"] for r in lives.values())})
    # W7
    we = sum(r["wet_entries"] for r in lives.values())
    wok = sum(r["wet_entry_ok"] for r in lives.values())
    wex = sum(r["wet_exits_clean"] for r in lives.values())
    wd = sum(r["wet_dry_ok"] for r in lives.values())
    gate("W7_wet_dry_cycle", we > 0 and (wok / we) >= 0.90 and wex > 0 and (wd / wex) >= 0.70,
         {"entries": we, "entry_soaked_fraction": round(wok / we, 3) if we else None,
          "clean_exits": wex, "dry_within_240_fraction": round(wd / wex, 3) if wex else None})
    # W8
    hi = [(r["comfort_hi_mean"], r["comfort_hi_n"]) for r in lives.values() if r["comfort_hi_mean"] is not None]
    lo = [(r["comfort_lo_mean"], r["comfort_lo_n"]) for r in lives.values() if r["comfort_lo_mean"] is not None]
    hi_m = sum(v * n for v, n in hi) / sum(n for _, n in hi) if hi else None
    lo_m = sum(v * n for v, n in lo) / sum(n for _, n in lo) if lo else None
    diff = (hi_m - lo_m) if (hi_m is not None and lo_m is not None) else None
    gate("W8_grime_is_felt", diff is not None and diff >= 5.0,
         {"comfort_hi_grime_mean": None if hi_m is None else round(hi_m, 2),
          "comfort_lo_grime_mean": None if lo_m is None else round(lo_m, 2),
          "difference": None if diff is None else round(diff, 2)})


# ------------------------------------------------------------------
# PART C: regression
# ------------------------------------------------------------------
def regression():
    reg = {}
    # ---- R1/R5: Phase-1 battery, output redirected ----
    pe_path = os.path.join(ROOT, "results", "phase1_gate_water.json")
    env = dict(os.environ)
    env["ALS_GATE_OUT"] = pe_path
    proc = subprocess.run([sys.executable, os.path.join(HERE, "gate_battery.py")],
                          capture_output=True, text=True, timeout=3600, env=env)
    r1 = {"subprocess_rc": proc.returncode}
    try:
        with open(pe_path, encoding="utf-8") as fh:
            p1 = json.load(fh)
        gates = p1["gates"]
        need = ["G1_homeostasis", "G2_survival", "G3a_food_learning",
                "G3b_fear_learning", "G4_circadian", "G5_checkpoint"]
        r1["gates"] = {k: gates[k]["pass"] for k in need}
        r1["pass"] = all(r1["gates"].values())
        r1["G3c_as_rerun_printed"] = gates["G3c_place_memory"]["pass"]
        r1["G3c_phase1_recorded"] = "FAIL (as recorded in Phase 1; not re-scored)"
        r5 = {"per_seed_night_sleep": {sd: gates["G4_circadian"]["per_seed"][sd]["night_sleep_fraction"]
                                       for sd in gates["G4_circadian"]["per_seed"]},
              "pass": gates["G4_circadian"]["pass"]}
    except Exception as exc:  # noqa: BLE001
        r1["pass"] = False
        r1["error"] = str(exc)
        r5 = {"pass": False, "error": str(exc)}
    reg["R1_phase1_battery"] = r1
    reg["R5_pe_night_sleep"] = r5
    print(f"[R1] {'PASS' if r1.get('pass') else 'FAIL'}", flush=True)
    print(f"[R5] {'PASS' if r5.get('pass') else 'FAIL'}", flush=True)

    # ---- R2/R3/R4: Phase-2 spot checks on the sitting-3 constructions ----
    import phase2_battery_v3 as b3
    sims, cols = {}, {}
    for sd in SEEDS_BATTERY:
        sims[sd], cols[sd] = b3.life_run(sd, instrument=(sd == 11))
    # R2 — I-GWT-1 (the battery's own formulas)
    ws = sims[11].fajr.mind.workspace.metrics
    cycles = max(1, ws["cycles"])
    propose_frac = {k: v / cycles for k, v in ws["propose_cycles"].items()}
    entry_frac = {k: v / cycles for k, v in ws["entry_counts"].items()}
    n95 = sum(1 for v in propose_frac.values() if v >= 0.95)
    reg["R2_gwt1_spot"] = {
        "pass": n95 >= 5 and all(v >= 0.005 for v in entry_frac.values()),
        "modules_at_95pct": n95,
        "entry_fraction": {k: round(v, 4) for k, v in entry_frac.items()},
        "sitting3_entry_fraction": {"vision": 0.0799, "interoception": 0.5937,
                                    "affect": 0.0233, "memory": 0.3012,
                                    "beliefs": 1.9989, "motor": 0.003}}
    print(f"[R2] {'PASS' if reg['R2_gwt1_spot']['pass'] else 'FAIL'}", flush=True)
    # R3 — I-HOT-2 (pooled tape)
    tape = []
    for sd in SEEDS_BATTERY:
        tape += [(r, c) for (r, c, h) in sims[sd].fajr.mind.tape_monitor]
    ece, auc, delta = b3.ece_auc_delta(tape)
    reg["R3_hot2_spot"] = {"pass": ece is not None and ece <= 0.15 and auc is not None
                           and auc >= 0.75 and delta is not None and delta >= 0.15,
                           "n_pairs": len(tape), "ece": ece, "auc": auc, "gap": delta}
    print(f"[R3] {'PASS' if reg['R3_hot2_spot']['pass'] else 'FAIL'}", flush=True)
    # R4 — I-HOT-3 (b) on seed-11 life + (a) probe aggregation
    low_w, high_w = sims[11].fajr.mind.beliefs.mean_update_weights()
    ratio_b = (low_w / high_w) if high_w else None
    h3a = {"corrupt": {}, "control": {}}
    for arm in ("corrupt", "control"):
        for sd in SEEDS_BATTERY:
            h3a[arm][str(sd)] = b3.probe_hot3a(sd, arm)
    valid_c = [v for v in h3a["corrupt"].values() if v["valid"]]
    valid_k = [v for v in h3a["control"].values() if v["valid"]]
    devs = [v["deviation_rate"] for v in valid_c if v.get("deviation_rate") is not None]
    avoid = sum(devs) / len(devs) if devs else None
    ate_c = (sum(v["target_share"] for v in valid_c) / len(valid_c)) if valid_c else None
    ate_k = (sum(v["target_share"] for v in valid_k) / len(valid_k)) if valid_k else None
    clause_a = (avoid is not None and avoid >= 0.50 and ate_c is not None
                and ate_k is not None and ate_k > ate_c)
    reg["R4_hot3_spot"] = {
        "pass": bool(clause_a and ratio_b is not None and ratio_b <= 0.50),
        "clause_a": {"avoidance_fraction_corrupt": avoid,
                     "ate_from_target_corrupt": ate_c,
                     "ate_from_target_control": ate_k, "pass": clause_a},
        "clause_b": {"weight_ratio": None if ratio_b is None else round(ratio_b, 4),
                     "pass": ratio_b is not None and ratio_b <= 0.50},
        "probe_detail": h3a}
    print(f"[R4] {'PASS' if reg['R4_hot3_spot']['pass'] else 'FAIL'}", flush=True)
    R["regression"] = reg
    save()


def main():
    print("== PART A: measurement lives ==", flush=True)
    for sd in SEEDS_LIVES:
        R["lives"][str(sd)] = measurement_life(sd)
        save()
        r = R["lives"][str(sd)]
        print(f"  seed {sd}: baths {r['baths']}, swims {r['swims']}, "
              f"crossings {r['crossings']}, min_clean {round(r['min_clean'], 1)}", flush=True)
    print("== PART B: west-bank probe ==", flush=True)
    for sd in SEEDS_BATTERY:
        R["probe_west"][str(sd)] = west_probe(sd)
        save()
        print(f"  seed {sd}: eaten {R['probe_west'][str(sd)]['eaten']}", flush=True)
    print("== WATER GATES ==", flush=True)
    score_water_gates()
    print("== PART C: regression ==", flush=True)
    regression()
    npass = sum(1 for g in R["gates"].values() if g["pass"])
    print(f"WATER GATES: {npass}/{len(R['gates'])} PASS", flush=True)
    print("wrote", OUT_PATH, flush=True)


if __name__ == "__main__":
    main()
