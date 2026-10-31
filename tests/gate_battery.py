"""PHASE 1 GATE BATTERY — thresholds frozen in GATE.md before this run.
Nothing here relaxes a threshold; failures are reported as failures.
Writes results/phase1_gate.json."""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from config import TICKS_PER_DAY, SHELTER_POS  # noqa: E402
from sim import Sim  # noqa: E402

SEEDS = [11, 42, 97]
RESULTS = {"seeds": SEEDS, "gates": {}}


def life_run(seed):
    s = Sim(seed=seed)
    s.run_days(7)
    return s


def gate_g1_g2_g4(sims):
    g1, g2, g4 = {}, {}, {}
    for seed, s in sims.items():
        m = s.metrics
        frac = m["ticks_in_band"] / m["ticks_total"]
        g1[str(seed)] = {"in_band_fraction": round(frac, 4), "pass": frac >= 0.95}
        no_death_spiral = all(v <= 240 for v in m["max_pinned"].values())
        moved_daily = all(d >= 30.0 for d in m["distance_by_day"])
        g2[str(seed)] = {"max_pinned_ticks": dict(m["max_pinned"]),
                         "distance_by_day": m["distance_by_day"],
                         "no_death_spiral": no_death_spiral, "moved_daily": moved_daily,
                         "pass": no_death_spiral and moved_daily}
        nf = m["sleep_ticks_night"] / max(1, m["sleep_ticks"])
        g4[str(seed)] = {"night_sleep_fraction": round(nf, 4), "sleep_ticks": m["sleep_ticks"],
                         "pass": nf >= 0.65}
    RESULTS["gates"]["G1_homeostasis"] = {"per_seed": g1, "pass": all(v["pass"] for v in g1.values())}
    RESULTS["gates"]["G2_survival"] = {"per_seed": g2, "pass": all(v["pass"] for v in g2.values())}
    RESULTS["gates"]["G4_circadian"] = {"per_seed": g4, "pass": all(v["pass"] for v in g4.values())}


def gate_g3c(sims):
    per = {}
    for seed, s in sims.items():
        eps = [(t // TICKS_PER_DAY + 1, math.hypot(x - SHELTER_POS[0], y - SHELTER_POS[1]))
               for (t, x, y) in s.fajr.sleep_episodes]
        late = [d for (day, d) in eps if 2 <= day <= 7]
        frac_near = (sum(1 for d in late if d < 10.0) / len(late)) if late else 0.0
        d12 = [d for (day, d) in eps if day <= 2]
        d57 = [d for (day, d) in eps if day >= 5]
        mean12 = sum(d12) / len(d12) if d12 else None
        mean57 = sum(d57) / len(d57) if d57 else None
        per[str(seed)] = {"episodes_days2_7": len(late), "fraction_near_shelter": round(frac_near, 3),
                          "mean_dist_days1_2": None if mean12 is None else round(mean12, 1),
                          "mean_dist_days5_7": None if mean57 is None else round(mean57, 1),
                          "pass": frac_near >= 0.60 and mean57 is not None and mean12 is not None and mean57 < mean12}
    RESULTS["gates"]["G3c_place_memory"] = {"per_seed": per, "pass": all(v["pass"] for v in per.values())}


PROBE_START = (144.0, 98.0)   # G3a food_probe start: no tree within any sense range


def _latency_run(sim, timeout=6000):
    """Run until a hunger latency is recorded (onset crossing 60 -> eat)."""
    start_eats = len(sim.fajr.hunger_latencies)
    for _ in range(timeout):
        sim.step()
        if len(sim.fajr.hunger_latencies) > start_eats:
            return sim.fajr.hunger_latencies[-1][1]
    return None


def gate_g3a():
    per = {}
    for seed in SEEDS:
        # naive probe: fresh being in the assay world, no food memory
        s = Sim(seed=seed, layout="food_probe")
        s.fajr.x, s.fajr.y = PROBE_START
        s.fajr.drives.d["hunger"] = 59.0
        # v2 protocol fix (2026-10-23, PHASE2-GATE amendment): snapshot the
        # naive trial's FULL drive vector — the experienced arm is set to
        # exactly these values so the ONLY difference between arms is
        # memory. Sitting 1 inherited the training life's end-state drives
        # (sleep/thirst/...), so seed 97's "experienced" latency measured
        # phase inheritance, not learning.
        naive_drives = dict(s.fajr.drives.d)
        naive = _latency_run(s)
        # experienced probe: same assay world lived in for 4 days first
        # (training life sized to the assay — GATE amendment 4)
        s2 = Sim(seed=seed, layout="food_probe")
        s2.run_days(4)
        bait = s2.world.trees[5]
        bait.fruit = 8  # refill the reward arm before the trial (standard maze practice)
        knows_bait = 5 in s2.fajr.food_memory and s2.fajr.food_memory[5][2] > 0
        s2.fajr.x, s2.fajr.y = PROBE_START
        s2.fajr.drives.d.update(naive_drives)
        s2.fajr.hunger_onset = None
        s2.fajr.asleep = False
        s2.fajr.intent = None
        s2.fajr.acting = None
        exp = _latency_run(s2) if knows_bait else None
        if naive and exp:
            improvement = (naive - exp) / naive
            per[str(seed)] = {"naive_ticks": naive, "experienced_ticks": exp,
                              "improvement": round(improvement, 3), "pass": improvement >= 0.20}
        else:
            per[str(seed)] = {"naive_ticks": naive, "experienced_ticks": exp,
                              "knows_bait_tree": knows_bait, "pass": False}
    RESULTS["gates"]["G3a_food_learning"] = {"per_seed": per, "pass": all(v["pass"] for v in per.values())}


def gate_g3b():
    per = {}
    for seed in (5, 6, 7):
        s = Sim(seed=seed, layout="fear_probe")
        th = s.world.thickets[0]
        entries_pre = entries_post = 0
        min_pre, min_post = 1e9, 1e9
        first_prick = None
        cond_tick = None
        total = 4 * TICKS_PER_DAY
        seen_entries = 0
        for i in range(total):
            s.step()
            d = math.hypot(th.x - s.fajr.x, th.y - s.fajr.y)
            pricks = [e for e in s.fajr.thicket_log if e[2] == "prick"]
            if pricks and first_prick is None:
                first_prick = pricks[0][0]
            if len(pricks) >= 2 and cond_tick is None:
                cond_tick = pricks[1][0]
            entries_now = sum(1 for e in s.fajr.thicket_log if e[2] == "enter")
            new_entries = entries_now - seen_entries
            seen_entries = entries_now
            if cond_tick is None:
                entries_pre += new_entries
                min_pre = min(min_pre, d)
            else:
                entries_post += new_entries
                min_post = min(min_post, d)
        if first_prick is None:
            per[str(seed)] = {"conditioned": False, "pass": False}
            continue
        if cond_tick is None:
            cond_tick = first_prick
        pre_ticks = max(1, first_prick)
        post_ticks = max(1, total - cond_tick)
        rate_pre = entries_pre / pre_ticks * 1000
        rate_post = entries_post / post_ticks * 1000
        drop = (rate_pre - rate_post) / rate_pre if rate_pre > 0 else 0.0
        per[str(seed)] = {"conditioned": True, "pricks": s.fajr.stats["pricks"],
                          "entry_rate_pre_per_1000": round(rate_pre, 2),
                          "entry_rate_post_per_1000": round(rate_post, 2),
                          "entry_drop": round(drop, 3),
                          "min_approach_pre": round(min_pre, 1), "min_approach_post": round(min_post, 1),
                          "pass": drop >= 0.60 and min_post > min_pre}
    RESULTS["gates"]["G3b_fear_learning"] = {"per_seed": per, "pass": all(v["pass"] for v in per.values())}


def gate_g5(tmpdir):
    per = {}
    for seed in SEEDS:
        a = Sim(seed=seed)
        a.run_days(7)
        hash_a = Sim.canonical_hash(a.state_dict())
        b = Sim(seed=seed)
        save_tick = 2 * TICKS_PER_DAY + (seed * 137) % TICKS_PER_DAY
        b.run_ticks(save_tick)
        path = os.path.join(tmpdir, f"ckpt_{seed}.json")
        b.save(path)
        hash_at_save = Sim.canonical_hash(b.state_dict())
        c = Sim.load(path)
        hash_loaded = Sim.canonical_hash(c.state_dict())
        fields_equal = (c.state_dict() == b.state_dict())
        c.run_ticks(7 * TICKS_PER_DAY - save_tick)
        hash_c = Sim.canonical_hash(c.state_dict())
        per[str(seed)] = {"save_tick": save_tick, "fields_equal_at_load": fields_equal,
                          "hash_match": hash_a == hash_c and hash_at_save == hash_loaded,
                          "pass": (hash_a == hash_c) and fields_equal}
    RESULTS["gates"]["G5_checkpoint"] = {"per_seed": per, "pass": all(v["pass"] for v in per.values())}


def supplementary_displacement_probe():
    """NOT part of the frozen battery — causal evidence for place memory:
    on day 4 at 15:00 the (probe-copy) being is displaced to the far NW
    corner; we measure whether that night's sleep begins back at the
    shelter (a return) rather than on the spot."""
    per = {}
    for seed in SEEDS:
        s = Sim(seed=seed)
        s.run_days(3)
        s.run_ticks(900)  # day 4, 15:00
        s.fajr.x, s.fajr.y = 36.0, 40.0
        s.fajr.asleep = False
        s.fajr.intent = None
        before = len(s.fajr.sleep_episodes)
        returned = None
        for _ in range(2000):
            s.step()
            if len(s.fajr.sleep_episodes) > before:
                t, x, y = s.fajr.sleep_episodes[-1]
                d = math.hypot(x - SHELTER_POS[0], y - SHELTER_POS[1])
                returned = {"sleep_dist_to_shelter": round(d, 1), "returned_home": d < 10.0}
                break
        per[str(seed)] = returned if returned else {"returned_home": False, "note": "no sleep within window"}
    RESULTS["supplementary"] = {"displacement_probe": per}


def main():
    os.makedirs("/tmp/alsakin_gate", exist_ok=True)
    sims = {seed: life_run(seed) for seed in SEEDS}
    gate_g1_g2_g4(sims)
    gate_g3c(sims)
    gate_g3a()
    gate_g3b()
    gate_g5("/tmp/alsakin_gate")
    supplementary_displacement_probe()
    RESULTS["all_pass"] = all(g["pass"] for g in RESULTS["gates"].values())
    # v2 plumbing (2026-10-23): ALS_GATE_OUT redirects the report so a
    # regression re-run never overwrites the recorded Phase-1 file.
    out = os.environ.get("ALS_GATE_OUT") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results", "phase1_gate.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(RESULTS, fh, ensure_ascii=False, indent=2)
    for name, g in RESULTS["gates"].items():
        print(f"{name}: {'PASS' if g['pass'] else 'FAIL'}")
    print("ALL:", "PASS" if RESULTS["all_pass"] else "FAIL")
    print("wrote", out)


if __name__ == "__main__":
    main()
