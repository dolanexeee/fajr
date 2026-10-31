"""PHASE 2 GATE BATTERY — SITTING 3 (v3, repair round 2, 2026-10-23).

Same frozen items, same thresholds (PHASE2-GATE.md). Changes vs sitting 2
are ONLY those declared in the gate's v3 amendment section: the being
carries the v3 mechanism repairs (precision-weighted belief updates;
entrained circadian phase in the body + the mind's sleep gate reading
it), and four probe constructions were rebuilt where sitting 2 PROVED
the old construction could not measure its frozen claim: HOT-3(c)
deliberate re-observation only; P-B dynamic ground truth; P-C remote
event site; AST-1(b) vision-leads scenario — calibrated at full size and
filed as architecturally non-instantiable (results/ast1b_v3_calibration
.md); the item records the filed evidence. PP-1 is
scored as before and filed per the independent floor re-derivation
(results/pp1_rederive_v3/). Writes results/phase2_gate_v3.json
incrementally (crash-safe); the Phase-1 regression writes
results/phase1_gate_v3.json via ALS_GATE_OUT. Sitting-1/2 files are
never written by this battery.

Run: python3 tests/phase2_battery_v3.py
"""
import bisect
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from config import TICKS_PER_DAY, SHELTER_POS, LANTERN_POS, BUILD_SITE  # noqa: E402
from sim import Sim  # noqa: E402
from world import light_level, is_passable  # noqa: E402
from perception import object_key  # noqa: E402
from fajr import cell_of  # noqa: E402

SEEDS = [11, 42, 97]
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "phase2_gate_v3.json")
R = {"seeds": SEEDS, "battery": "phase2_battery_v3 (sitting 3, repair round 2)", "items": {}, "probes": {}}


def save():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=2)


def verdict(item, passed, detail):
    R["items"][item] = {"pass": bool(passed), **detail}
    save()
    print(f"[{item}] {'PASS' if passed else 'FAIL'}", flush=True)


# ------------------------------------------------------------------ stats helpers
def ece_auc_delta(tape, bins=10):
    """tape: [(reliability, correct)] -> (ece, auc, mean_rel_correct - mean_rel_incorrect)"""
    if not tape:
        return None, None, None
    b = {}
    for r, c in tape:
        b.setdefault(min(bins - 1, int(r * bins)), []).append((r, c))
    ece = 0.0
    for rows in b.values():
        conf = sum(r for r, _ in rows) / len(rows)
        acc = sum(c for _, c in rows) / len(rows)
        ece += len(rows) / len(tape) * abs(conf - acc)
    pos = sorted(r for r, c in tape if c == 1)
    neg = sorted(r for r, c in tape if c == 0)
    if not pos or not neg:
        return round(ece, 4), None, None
    score = 0.0
    for r in pos:
        lo = bisect.bisect_left(neg, r)
        hi = bisect.bisect_right(neg, r)
        score += lo + 0.5 * (hi - lo)
    auc = score / (len(pos) * len(neg))
    delta = sum(pos) / len(pos) - sum(neg) / len(neg)
    return round(ece, 4), round(auc, 4), round(delta, 4)


def pearson(xs, ys):
    n = len(xs)
    if n < 4:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(ranks(xs), ranks(ys))


def belief_mae(sim, fresh_within=None):
    m = sim.fajr.mind
    errs, n = 0.0, 0
    for key, b in m.beliefs.objects.items():
        if b["kind"] != "tree":
            continue
        if fresh_within is not None and sim.tick - b["last_seen"] > fresh_within:
            continue
        errs += abs(b["fruit_est"] - sim.world.trees[b["oid"]].fruit)
        n += 1
    return (errs / n if n else None), n


# ------------------------------------------------------------------ instrumented life
def life_run(seed, cfg=None, days=7, instrument=False, layout="default"):
    """A measurement life. With instrument=True (seed-11 items), per-tick
    harness hooks collect PP-1 pairs, RPT-1 persistence, RPT-2 foreground,
    GWT-3b cue tracking and HOT-3c watchers. P-D hourly samples are
    collected for every seed."""
    s = Sim(seed, mind_config=cfg, layout=layout)
    f = s.fajr
    col = {"pd": {"hunger": [], "thirst": [], "sleep": []},
           "pp1_pairs": [], "persist_events": [], "rpt2_fg": [0, 0],
           "rpt2_bind": [0, 0], "rpt2_sep": [0, 0],
           "gwt3b": {"match_hit": 0, "match_tot": 0, "mismatch_hit": 0, "mismatch_tot": 0},
           "hot3c": {"low": [0, 0], "high": [0, 0]},   # class -> [resolved, total]
           "hot3c_app": {"low": [0, 0], "high": [0, 0]},   # v3 supplementary: sitting-2 approach measure
           "scenes": []}
    if not instrument:
        for t in range(days * TICKS_PER_DAY):
            s.step()
            if s.tick % 60 == 0 and s.tick >= TICKS_PER_DAY:
                for k in col["pd"]:
                    col["pd"][k].append(abs(f.mind.selfmodel.estimate(k, s.tick, asleep=f.asleep) - f.drives.d[k]))
        return s, col
    m = f.mind
    last_raw = set()
    occl = {}          # key -> survival ticks so far
    prev_beliefs = {}
    prev_contents = []
    watchers = []      # {tree, tick, cls, done}
    watch_cooldown = {}
    total = days * TICKS_PER_DAY
    for t in range(total):
        s.step()
        tick = s.tick
        # ---- P-D hourly samples, days 2..7 ----
        if tick % 60 == 0 and tick >= TICKS_PER_DAY:
            for k in col["pd"]:
                col["pd"][k].append(abs(m.selfmodel.estimate(k, tick, asleep=f.asleep) - f.drives.d[k]))
        if f.asleep:
            last_raw = set()
            continue
        scene = m.perception.scene
        # ---- RPT-1 harness persistence ----
        raw_keys = set()
        if f._raw is not None:
            for (kind, oid, dx, dy, dist, extra) in f._raw.visible:
                if kind in ("tree", "thicket", "material", "pile"):
                    raw_keys.add(object_key(kind, oid, f.x + dx, f.y + dy))
        for key in last_raw - raw_keys:
            if key not in occl:
                occl[key] = 0
        for key in list(occl.keys()):
            if key in raw_keys:
                col["persist_events"].append(occl.pop(key))
            elif key in scene.objects:
                occl[key] += 1
            else:
                col["persist_events"].append(occl.pop(key))
        last_raw = raw_keys
        # ---- PP-1 pairs: |pred error| vs belief update magnitude ----
        cur_beliefs = {k: b["fruit_est"] for k, b in m.beliefs.objects.items()
                       if b["kind"] == "tree"}
        if prev_beliefs and scene.n_predicted > 0:
            delta = sum(abs(cur_beliefs[k] - prev_beliefs[k])
                        for k in cur_beliefs if k in prev_beliefs)
            col["pp1_pairs"].append((scene.pred_error, delta))
        prev_beliefs = cur_beliefs
        # ---- RPT-2 foreground: when the workspace winner IS an object
        # (a vision coalition), its target should sit in the scene's
        # top-3 salience (frozen text: «حين يكون شيئًا») ----
        if m.workspace.contents and m.workspace.contents[0].source == "vision" \
                and m.workspace.contents[0].target_key:
            col["rpt2_fg"][1] += 1
            if m.workspace.contents[0].target_key in scene.foreground[:3]:
                col["rpt2_fg"][0] += 1
        # ---- RPT-2 binding, per tick (protocol amendment 1): constructed
        # scenes = two sensed trees with |true fruit diff| >= 3 plus a
        # thicket within 20 units; features must stay on their objects ----
        truth_now = {}
        if f._raw is not None:
            for (kind, oid, dx, dy, dist, extra) in f._raw.visible:
                truth_now[object_key(kind, oid, f.x + dx, f.y + dy)] = (kind, extra)
        sensed_trees = [(k, o) for k, o in scene.objects.items()
                        if o.kind == "tree" and not o.generated and k in truth_now]
        thick_near = any(o.kind == "thicket" and o.dist < 20.0
                         for o in scene.objects.values())
        if thick_near and len(sensed_trees) >= 2:
            for i in range(len(sensed_trees)):
                for j2 in range(i + 1, len(sensed_trees)):
                    (k1, o1), (k2, o2) = sensed_trees[i], sensed_trees[j2]
                    if abs(truth_now[k1][1] - truth_now[k2][1]) >= 3:
                        col["rpt2_bind"][1] += 1
                        if abs(o1.fruit - truth_now[k1][1]) <= 1 and abs(o2.fruit - truth_now[k2][1]) <= 1:
                            col["rpt2_bind"][0] += 1
                        col["rpt2_sep"][1] += 1
                        th_q = [o.quality[2] for o in scene.objects.values() if o.kind == "thicket" and o.dist < 20.0]
                        if th_q and max(th_q) >= 0.9 and o1.quality[2] <= 0.4 and o2.quality[2] <= 0.4:
                            col["rpt2_sep"][0] += 1
        # ---- GWT-3b: memory entry relevance vs previous broadcast cue ----
        need = None
        urg = -1.0
        for name, v in f.drives.d.items():
            u = f.drives.urgency(name)
            if u > urg:
                urg, need = u, name
        relevant = {"hunger": {"eat", "saw_pile", "saw_food"},
                    "safety": {"prick", "cold_shock"}}.get(need, set())
        if relevant:
            prev_kinds = {c.kind for c in prev_contents}
            matching = (f"need:{need}" in prev_kinds) or any(
                k.startswith("episode:") and k.split(":", 1)[1] in relevant
                for k in prev_kinds)
            entered = [c for c in m.workspace.contents
                       if c.source == "memory" and c.kind.startswith("episode:")
                       and c.kind.split(":", 1)[1] in relevant]
            if matching:
                col["gwt3b"]["match_tot"] += 1
                col["gwt3b"]["match_hit"] += 1 if entered else 0
            else:
                col["gwt3b"]["mismatch_tot"] += 1
                col["gwt3b"]["mismatch_hit"] += 1 if entered else 0
        prev_contents = list(m.workspace.contents)
        # ---- HOT-3c watchers: low/high reliability tree percepts ----
        for key, r in m.monitor.last.items():
            if not key.startswith("tree:"):
                continue
            tid = int(key.split(":")[1])
            cls = "low" if r < 0.4 else ("high" if r >= 0.7 else None)
            if cls and tick - watch_cooldown.get(tid, -10 ** 9) > 600:
                watch_cooldown[tid] = tick
                watchers.append({"tree": tid, "tick": tick, "cls": cls, "done": False,
                                 "app_done": False})
                col["hot3c"][cls][1] += 1
                col["hot3c_app"][cls][1] += 1
        for wch in watchers:
            tree = s.world.trees[wch["tree"]]
            # v3 construction (GATE v3-2): the SCORED resolution is a
            # DELIBERATE re-observation only — he adopts an `observe`
            # intention whose target is that very tree (the epistemic
            # branch's own act). An incidental approach (passing within
            # 6 units for any other reason) no longer resolves the
            # watcher; it is tallied separately as the sitting-2 measure,
            # reported as supplementary transparency.
            if not wch["app_done"] and math.hypot(tree.x - f.x, tree.y - f.y) < 6.0:
                col["hot3c_app"][wch["cls"]][0] += 1
                wch["app_done"] = True
            if wch["done"]:
                continue
            observing = (f.intent is not None and f.intent[0] == "observe"
                         and f.intent[3] == wch["tree"])
            if observing:
                col["hot3c"][wch["cls"]][0] += 1
                wch["done"] = True
            elif tick - wch["tick"] > 300:
                wch["done"] = True
        watchers = [wch for wch in watchers if not wch["done"]][-64:]
        # ---- scene samples for RPT-2 binding + HOT-4 ----
        if tick % 120 == 0:
            truth = {}
            if f._raw is not None:
                for (kind, oid, dx, dy, dist, extra) in f._raw.visible:
                    truth[object_key(kind, oid, f.x + dx, f.y + dy)] = (kind, extra, dist)
            col["scenes"].append({"tick": tick, "fx": f.x, "fy": f.y,
                                  "objects": {k: o for k, o in scene.objects.items()},
                                  "truth": truth,
                                  "weather": s.world.weather})
    col["persist_events"].extend(occl.values())
    return s, col


# ------------------------------------------------------------------ indicators from lives
def score_from_lives(sims, cols, binding_col=None):
    s11 = sims[11]
    m11 = s11.fajr.mind
    col11 = cols[11]

    # ---------- I-RPT-2 (binding as feature-to-object attachment, fusion,
    # foreground — per-tick measurement, protocol amendment 1.
    # v2 amendment 3: the BINDING clause is measured in the binding_probe
    # world, the only construction where qualifying scenes occur; fusion
    # and foreground stay on the main life, as the amendment declares.)
    bind_ok, bind_n = (binding_col or col11)["rpt2_bind"]
    main_bind_ok, main_bind_n = col11["rpt2_bind"]
    sep_ok, sep_n = (binding_col or col11)["rpt2_sep"]
    SIZE = {"tree": 0.8, "thicket": 0.55, "shelter": 0.9, "structure": 0.4,
            "material": 0.25, "pile": 0.35}
    hot4_pairs_q, hot4_pairs_t, hot4_sparse = [], [], []
    for sc in col11["scenes"]:
        objs, truth = sc["objects"], sc["truth"]
        # HOT-4 pairs within the scene
        items = [(k, o) for k, o in objs.items() if k in truth and not o.generated]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (k1, o1), (k2, o2) = items[i], items[j]
                dq = math.sqrt(sum((a - b) ** 2 for a, b in zip(o1.quality, o2.quality)))
                fr1 = truth[k1][1] if truth[k1][0] in ("tree", "pile") else 0
                fr2 = truth[k2][1] if truth[k2][0] in ("tree", "pile") else 0
                t1 = (fr1 / 6.0, SIZE.get(truth[k1][0], 0.3),
                      1.0 if truth[k1][0] == "thicket" else 0.0, max(0.0, 1 - truth[k1][2] / 60.0))
                t2 = (fr2 / 6.0, SIZE.get(truth[k2][0], 0.3),
                      1.0 if truth[k2][0] == "thicket" else 0.0, max(0.0, 1 - truth[k2][2] / 60.0))
                dt = math.sqrt(sum((a - b) ** 2 for a, b in zip(t1, t2)))
                hot4_pairs_q.append(dq)
                hot4_pairs_t.append(dt)
        for k, o in items:
            hot4_sparse.append(sum(1 for v in o.quality if abs(v) < 0.05) / len(o.quality))
    pm = m11.perception.metrics
    fusion = pm["fusion_ok"] / max(1, pm["fusion_n"])
    fg_frac = col11["rpt2_fg"][0] / max(1, col11["rpt2_fg"][1])
    bind_frac = bind_ok / max(1, bind_n)
    sep_frac = sep_ok / max(1, sep_n)
    verdict("I-RPT-2", bind_frac >= 0.95 and fusion >= 0.99 and fg_frac >= 0.80,
            {"binding_scenes": bind_n, "binding_ok_fraction": round(bind_frac, 4),
             "binding_world": "binding_probe (v2 amendment 3)",
             "binding_scenes_main_life_for_reference": main_bind_n,
             "feature_separation_fraction": round(sep_frac, 4),
             "fusion_fraction": round(fusion, 4), "fusion_n": pm["fusion_n"],
             "exact_count_binding_supplementary": round(pm["binding_ok"] / max(1, pm["binding_n"]), 4),
             "winner_target_in_top3_fraction": round(fg_frac, 4)})

    # ---------- I-GWT-1
    ws = m11.workspace.metrics
    cycles = max(1, ws["cycles"])
    propose_frac = {k: v / cycles for k, v in ws["propose_cycles"].items()}
    entry_frac = {k: v / cycles for k, v in ws["entry_counts"].items()}
    n95 = sum(1 for v in propose_frac.values() if v >= 0.95)
    verdict("I-GWT-1", n95 >= 5 and all(v >= 0.005 for v in entry_frac.values()),
            {"propose_fraction": {k: round(v, 4) for k, v in propose_frac.items()},
             "modules_at_95pct": n95,
             "entry_fraction": {k: round(v, 4) for k, v in entry_frac.items()},
             "note": "six proposing modules: vision, interoception, affect, memory, beliefs, motor; planning acts as stage-2 over the broadcast (mind.py) — the gate's parenthetical lists planning among the specialists"})

    # ---------- I-GWT-2
    verdict("I-GWT-2", (ws["proposals_sum"] / cycles) >= 5.0 and (ws["excluded_salient"] / cycles) >= 0.40,
            {"capacity": 3, "mean_proposals_per_cycle": round(ws["proposals_sum"] / cycles, 2),
             "excluded_salient_cycle_fraction": round(ws["excluded_salient"] / cycles, 4)})

    # ---------- I-GWT-3
    receipts = ws["broadcast_receipts"]
    receipts_ok = len(set(receipts.values())) == 1 and list(receipts.values())[0] == ws["cycles"]
    g3 = col11["gwt3b"]
    rate_match = g3["match_hit"] / max(1, g3["match_tot"])
    rate_mismatch = g3["mismatch_hit"] / max(1, g3["mismatch_tot"])
    rel_gain = (rate_match - rate_mismatch) / rate_mismatch if rate_mismatch > 0 else None
    # protocol amendment 2: a zero mismatched baseline makes the relative
    # gain undefined; the clause is then satisfied by matching rate > 0
    gwt3_pass = receipts_ok and ((rel_gain is not None and rel_gain >= 0.20)
                                 or (rate_mismatch == 0.0 and rate_match > 0.0))
    verdict("I-GWT-3", gwt3_pass,
            {"broadcast_receipts": receipts, "receipts_uniform_100pct": receipts_ok,
             "retrieval_hit_rate_matching_cue": round(rate_match, 4), "matching_cycles": g3["match_tot"],
             "retrieval_hit_rate_mismatched_cue": round(rate_mismatch, 4), "mismatched_cycles": g3["mismatch_tot"],
             "relative_gain": None if rel_gain is None else round(rel_gain, 4),
             "zero_baseline_clarification": "PHASE2-GATE protocol amendment 2" if rel_gain is None else None})

    # ---------- I-HOT-2 (pooled harness tape, seeds 11/42/97)
    tape = []
    for sd in SEEDS:
        tape += [(r, c) for (r, c, h) in sims[sd].fajr.mind.tape_monitor]
    ece, auc, delta = ece_auc_delta(tape)
    # Bayes-ceiling estimate: oracle AUC using condition bins from the tape itself
    verdict("I-HOT-2", ece is not None and ece <= 0.15 and auc is not None and auc >= 0.75
            and delta is not None and delta >= 0.15,
            {"n_pairs": len(tape), "ece_10bin": ece, "auc": auc, "mean_rel_gap": delta,
             "per_seed": {str(sd): dict(zip(("ece", "auc", "gap"),
                        ece_auc_delta([(r, c) for (r, c, h) in sims[sd].fajr.mind.tape_monitor])))
                          for sd in SEEDS}})

    # ---------- I-HOT-4
    sp = spearman(hot4_pairs_q[::3], hot4_pairs_t[::3]) if len(hot4_pairs_q) > 30 else None
    sparsity = sum(hot4_sparse) / max(1, len(hot4_sparse))
    verdict("I-HOT-4", sp is not None and sp >= 0.60 and sparsity >= 0.30,
            {"spearman_quality_vs_true": None if sp is None else round(sp, 4),
             "pairs": len(hot4_pairs_q), "mean_sparsity": round(sparsity, 4), "dims": 8})

    # ---------- I-PP-1
    err_day = pm["err_day"]
    def day_err(d):
        slot = err_day.get(d)
        return (slot[0] / slot[1]) if slot and slot[1] else None
    e1, e7 = day_err(1), day_err(7)
    ratio = (e7 / e1) if (e1 and e7) else None
    xs = [p[0] for p in col11["pp1_pairs"]]
    ys = [p[1] for p in col11["pp1_pairs"]]
    corr = pearson(xs, ys)
    verdict("I-PP-1", ratio is not None and ratio <= 0.75 and corr is not None and corr >= 0.40,
            {"pred_error_day1": None if e1 is None else round(e1, 4),
             "pred_error_day7": None if e7 is None else round(e7, 4),
             "day7_over_day1": None if ratio is None else round(ratio, 4),
             "pearson_err_vs_belief_update": None if corr is None else round(corr, 4),
             "pairs": len(xs)})

    # ---------- I-AE-2
    fm = m11.fwd.metrics
    d67 = [fm["by_day"].get(d, [0, 0]) for d in (6, 7)]
    acc67 = sum(c for c, n in d67) / max(1, sum(n for c, n in d67))
    base_rates = {f: round(m11.fwd.gcat[f][0] / max(1, m11.fwd.gcat[f][1]), 3)
                  for f in ("see_water", "see_food", "hazard")}
    disc_ratio = pm["err_sum"] / pm["errnd_sum"] if pm.get("errnd_sum") else None
    fm_part = m11.metrics["fm_participated"] / max(1, m11.metrics["decisions"])
    verdict("I-AE-2", acc67 >= 0.72 and disc_ratio is not None and disc_ratio <= 0.70 and fm_part >= 0.30,
            {"categorical_accuracy_days6_7": round(acc67, 4), "base_rates": base_rates,
             "self_discount_ratio": None if disc_ratio is None else round(disc_ratio, 4),
             "fm_participation_overall": round(fm_part, 4),
             "skill_vs_majority_baseline": round((fm["pred_correct"] - fm.get("base_correct", 0)) / max(1, fm["pred_n"]), 4)})

    # ---------- I-AST-1 clause (a): schema accuracy vs majority
    acc, maj = m11.workspace.schema.accuracy()
    verdict("I-AST-1a", acc >= 1.5 * maj,
            {"schema_accuracy": round(acc, 4), "majority_baseline": round(maj, 4),
             "ratio": round(acc / max(1e-9, maj), 3), "n": m11.workspace.schema.metrics["n"],
             "schema_boosts_applied": m11.metrics.get("schema_boosts", 0)})

    # ---------- P-D: self-model accuracy
    pd_out = {}
    pd_pass = True
    for k in ("hunger", "thirst", "sleep"):
        per_seed = {}
        pooled = []
        for sd in SEEDS:
            v = cols[sd]["pd"][k]
            per_seed[str(sd)] = round(sum(v) / max(1, len(v)), 3)
            pooled += v
        mae = sum(pooled) / max(1, len(pooled))
        pd_out[k] = {"mae_pooled": round(mae, 3), "per_seed": per_seed, "n": len(pooled)}
        pd_pass = pd_pass and mae <= 8.0
    end_log = []
    for sd in SEEDS:
        end_log += sims[sd].fajr.mind.selfmodel.metrics["endurance_log"]
    end_acc = (sum(1 for r in end_log if r[2] == r[3]) / len(end_log)) if end_log else None
    verdict("P-D", pd_pass and end_acc is not None and end_acc >= 0.75,
            {"drive_mae": pd_out, "endurance_entries": len(end_log),
             "endurance_classification_accuracy": None if end_acc is None else round(end_acc, 4)})


def score_lesion_runs(intact11, ff, td):
    """RPT-1 and HOT-1 need lesion lives: ff = feedforward_off, td = topdown_off.
    v2 amendment 1: verdicts use the non-empty-tick aggregation (f1ne);
    the sitting-1 aggregation is computed and reported alongside."""
    pm_i = intact11.fajr.mind.perception.metrics
    pm_f = ff.fajr.mind.perception.metrics
    pm_t = td.fajr.mind.perception.metrics

    def agg(pm, s, n):
        return pm[s] / max(1, pm[n])
    f1_i, f1_f, f1_t = (agg(pm_i, "f1_sum", "f1_n"), agg(pm_f, "f1_sum", "f1_n"),
                        agg(pm_t, "f1_sum", "f1_n"))
    f1ne_i = pm_i.get("f1ne_sum", 0.0) / max(1, pm_i.get("f1ne_n", 0))
    f1ne_f = pm_f.get("f1ne_sum", 0.0) / max(1, pm_f.get("f1ne_n", 0))
    f1ne_t = pm_t.get("f1ne_sum", 0.0) / max(1, pm_t.get("f1ne_n", 0))
    R["lesions"] = {"f1_intact": round(f1_i, 4), "f1_feedforward_off": round(f1_f, 4),
                    "f1_topdown_off": round(f1_t, 4),
                    "f1ne_intact": round(f1ne_i, 4), "f1ne_feedforward_off": round(f1ne_f, 4),
                    "f1ne_topdown_off": round(f1ne_t, 4),
                    "aggregation_note": "verdicts on f1ne (non-empty ticks, v2 amendment 1); f1 = sitting-1 aggregation, reported for transparency"}
    save()
    return f1ne_i, f1ne_f, f1ne_t


def score_rpt1(col11, ff_col, f1_i, f1_f):
    ev = [e for e in col11["persist_events"]]
    mean_p = sum(ev) / max(1, len(ev))
    ev_f = [e for e in ff_col["persist_events"]]
    mean_pf = sum(ev_f) / max(1, len(ev_f))
    verdict("I-RPT-1", mean_p >= 5.0 and len(ev) >= 20 and mean_pf <= 1.0
            and f1_i >= 0.85 and (f1_i - f1_f) >= 0.05,
            {"occlusion_events": len(ev), "mean_persistence_ticks": round(mean_p, 3),
             "lesion_mean_persistence_ticks": round(mean_pf, 3), "lesion_events": len(ev_f),
             "scene_f1_intact_nonempty": round(f1_i, 4), "scene_f1_feedforward_off_nonempty": round(f1_f, 4),
             "scene_f1_sitting1_aggregation": R["lesions"]["f1_intact"],
             "scene_f1_ff_sitting1_aggregation": R["lesions"]["f1_feedforward_off"]})


def score_hot1(intact11, td, f1_i, f1_t):
    pm = intact11.fajr.mind.perception.metrics
    kept = pm["occlusion_kept"] / max(1, pm["occlusion_events"])
    verdict("I-HOT-1", kept >= 0.80 and (f1_i - f1_t) >= 0.05,
            {"occlusion_completion_rate": round(kept, 4),
             "occlusion_events": pm["occlusion_events"],
             "f1_intact_nonempty": round(f1_i, 4), "f1_topdown_off_nonempty": round(f1_t, 4),
             "f1_sitting1_aggregation": R["lesions"]["f1_intact"],
             "f1_topdown_sitting1_aggregation": R["lesions"]["f1_topdown_off"],
             "predicted_scene_every_awake_tick": "predict() runs inside every perception build; predicted dict non-empty whenever a prior scene exists (code-audited in perception.build)"})


def score_hot3(intact11, col11, pb):
    """pb = P-B result dict (monitor lesion lives) computed in probes section;
    HOT-3 clauses (b) here, (a) and (c) filled by probes/harness data."""
    m = intact11.fajr.mind
    low_w, high_w = m.beliefs.mean_update_weights()
    ratio = low_w / high_w if high_w else None
    hc = col11["hot3c"]
    rate_low = hc["low"][0] / max(1, hc["low"][1])
    rate_high = hc["high"][0] / max(1, hc["high"][1])
    ha = col11["hot3c_app"]
    app_low = ha["low"][0] / max(1, ha["low"][1])
    app_high = ha["high"][0] / max(1, ha["high"][1])
    R["items"]["I-HOT-3"] = {
        "clause_b_weight_ratio": None if ratio is None else round(ratio, 4),
        "clause_b_pass": ratio is not None and ratio <= 0.50,
        "mean_weight_low_rel": round(low_w, 4), "mean_weight_high_rel": round(high_w, 4),
        "clause_c_measure": "deliberate re-observation only (adopted observe intent targeting the tree) — GATE v3 amendment 2",
        "clause_c_reobserve_rate_low": round(rate_low, 4), "clause_c_events_low": hc["low"][1],
        "clause_c_reobserve_rate_high": round(rate_high, 4), "clause_c_events_high": hc["high"][1],
        "clause_c_pass": hc["low"][1] >= 3 and hc["high"][1] >= 3 and rate_high > 0 and rate_low >= 2.0 * rate_high,
        "clause_c_supplementary_sitting2_measure": {
            "approach_rate_low": round(app_low, 4), "approach_rate_high": round(app_high, 4),
            "note": "the sitting-2 approach-based tally on this same life, no verdict weight"},
    }
    save()
    print("[I-HOT-3] clauses b/c recorded (verdict after probe a)", flush=True)


# ------------------------------------------------------------------ probes
def _remote_site(world):
    """P-C v3 site search (GATE v3-4 + calibration addendum): a passable
    point whose nearest tree is 40-75 units away and which is at least
    30 units from the shelter and 8 from any thicket — outside every
    believed-resource approach corridor by construction (sitting 2's
    site sat 8 units from a tree and its control arm visited 3/3
    incidentally; the visit radius is 6, so a 40-unit floor gives a
    ~7x margin). The band's upper edge is calibration, documented in
    the gate addendum: the unconstrained argmax lands in a far corner
    (seed 11: 119 units from the nearest tree, 164 from the shelter)
    where dev measurement showed the day-3 episode target is adopted
    0 times in the frozen 900-tick window — a probe that cannot
    instantiate its own scenario, the sitting-1 RPT-2 failure mode.
    Within the band the search maximises nearest-tree distance, then
    minimises shelter distance (his day-3 start). Pure world geometry,
    so the site is identical for every arm of a given seed."""
    best = None
    best_score = None
    xi = 10.0
    while xi <= 250.0:
        yi = 10.0
        while yi <= 170.0:
            if is_passable(xi, yi):
                d_tree = min(math.hypot(xi - tr.x, yi - tr.y) for tr in world.trees)
                d_shelter = math.hypot(xi - SHELTER_POS[0], yi - SHELTER_POS[1])
                d_thicket = min([math.hypot(xi - th.x, yi - th.y)
                                 for th in world.thickets] + [999.0])
                if d_tree >= 40.0 and d_shelter >= 30.0 and d_thicket >= 8.0:
                    score = (min(d_tree, 75.0), -d_shelter)
                    if best_score is None or score > best_score:
                        best_score = score
                        best = ((xi, yi), d_shelter, d_tree)
            yi += 5.0
        xi += 5.0
    return best   # ((x, y), d_shelter, d_tree)


def _view_point(site):
    """A passable viewing spot ~10 units from the site, toward the shelter.
    Calibration (round-2 dev): the original ~5-unit stand let him step to
    the pile and eat it during the staged sighting on seed 42 despite the
    satiation set, contaminating the site's place value and planting an
    eat episode at the site — the visit then proved nothing. 10 units
    keeps the pile well inside the daytime view radius (26) and the
    night one (~14.3, for the evening wash) while putting the pile
    outside a casual step's reach during the 45-tick staging."""
    sx, sy = site
    dx, dy = SHELTER_POS[0] - sx, SHELTER_POS[1] - sy
    n = math.hypot(dx, dy) or 1.0
    for dist in (10.0, 8.0, 12.0, 7.0):
        vx, vy = sx + dx / n * dist, sy + dy / n * dist
        if is_passable(vx, vy):
            return (vx, vy)
    for (ox, oy) in [(5, 0), (-5, 0), (0, 5), (0, -5), (4, 4), (-4, -4)]:
        if is_passable(sx + ox, sy + oy):
            return (sx + ox, sy + oy)
    return (sx, sy)


def probe_pc(seed, arm):
    """P-C v3 (GATE v3-4): the frozen scenario shape (one-time fruited
    pile sighting on day 2, removed; day-3 fruitless morning, hunger
    raised; visit = within 6 units) with the event site moved to a
    genuinely REMOTE point. Sitting 2 proved the frozen 8-unit offset
    sits inside the eat tree's approach corridor: the no-event control
    visited 3/3 incidentally, so the measure's background rate saturated
    it. Here the site maximises distance to the nearest tree (>= 30
    units from the shelter by construction; achieved distances are in
    the audit), so a day-3 arrival is explicable only by the episodic
    record. Isolation elements from v2 are kept: the pile is staged for
    45 ticks while he is deeply satiated (hunger set to 12 — staging
    integrity calibration, round-2 dev: at the frozen 30 the eat branch
    re-opened within a tick and seed 42 ate the staged pile, feeding
    place value at the site; at 12 the branch stays shut and the audit's
    ate_during_placement / place_value checks verify it), and the day-2
    evening belief-wash shows him the
    EMPTY pile at the site, so on test morning the belief reads bare,
    the place grid was never fed at the site (he never ate there), and
    only episodic memory carries the site's positive information.
    Stage-management teleports are declared, as throughout this battery.
    Arms: intact / none (no event at all) / episodic_off / workspace_off.
    """
    from world import Pile
    cfg = {"episodic_off": True} if arm == "episodic_off" else \
          ({"workspace_off": True} if arm == "workspace_off" else None)
    s = Sim(seed, mind_config=cfg)
    f = s.fajr
    found = _remote_site(s.world)
    site, d_shelter, d_tree = found[0], found[1], found[2]
    view = _view_point(site)
    window = {"visited": False, "bare_trees_before": None, "place_value": None,
              "valid": False, "ate_during_placement": False, "audit": None,
              "wash_done": False,
              "site": [round(site[0], 1), round(site[1], 1)],
              "site_dist_nearest_tree": round(d_tree, 1),
              "site_dist_shelter": round(d_shelter, 1)}
    placed_tick = None
    fruit_at_removal = None
    wash_state = 0
    bare_seen = set()
    visited_tick = None
    win_start_eats = 0

    def return_home():
        for (ox, oy) in [(2, 2), (-3, 1), (0, -4), (4, 0)]:
            hx, hy = SHELTER_POS[0] + ox, SHELTER_POS[1] + oy
            if is_passable(hx, hy):
                f.x, f.y = hx, hy
                break
        f.intent = None
        f.acting = None

    total = 3 * TICKS_PER_DAY + 1300
    for t in range(total):
        s.step()
        tick = s.tick
        day = tick // TICKS_PER_DAY + 1
        tod = tick % TICKS_PER_DAY
        # day 2, 09:00: the staged sighting at the remote site
        if arm != "none" and placed_tick is None and day == 2 and tod >= 540:
            s.world.piles.append(Pile(900 + seed, site[0], site[1], 3))
            placed_tick = tick
            f.x, f.y = view
            f.asleep = False
            f.intent = None
            f.acting = None
            f.drives.d["hunger"] = 12.0
        if placed_tick is not None and tick == placed_tick + 45:
            for p in list(s.world.piles):
                if p.id == 900 + seed:
                    fruit_at_removal = p.fruit
                    window["ate_during_placement"] = p.fruit < 3
                    s.world.piles.remove(p)
            return_home()
        # day-2 evening belief-wash (v2 isolation element, kept)
        if arm != "none" and wash_state == 0 and day == 2 and tod >= 1150:
            s.world.piles.append(Pile(900 + seed, site[0], site[1], 0))
            f.x, f.y = view
            f.asleep = False
            f.intent = None
            f.acting = None
            wash_state = 1
            window["wash_done"] = True
        if wash_state == 1 and day == 2 and tod >= 1240:
            for p in list(s.world.piles):
                if p.id == 900 + seed:
                    s.world.piles.remove(p)
            return_home()
            wash_state = 2
        # day 3: fruitless morning + hunger raise at 05:30 (frozen)
        if day == 3 and tod >= 300:
            for tr in s.world.trees:
                tr.fruit = 0
            if tod == 330:
                f.drives.d["hunger"] = 85.0
                window["valid"] = True
                win_start_eats = f.stats["eats"]
                b = f.mind.beliefs.objects.get(f"pile:{900 + seed}")
                cx, cy = cell_of(site[0], site[1])
                window["audit"] = {
                    "pile_belief_fruit_est": None if b is None else round(b["fruit_est"], 3),
                    "pile_belief_conf": None if b is None else round(b["conf"], 3),
                    "place_value_at_site": round(f.place_value[cy][cx], 4),
                    "dist_to_site": round(math.hypot(site[0] - f.x, site[1] - f.y), 1)}
            if visited_tick is None and tod >= 330:
                d = math.hypot(site[0] - f.x, site[1] - f.y)
                for tr in s.world.trees:
                    if math.hypot(tr.x - f.x, tr.y - f.y) < 3.0:
                        bare_seen.add(tr.id)
                if d <= 6.0:
                    visited_tick = tick
                    window["visited"] = True
                    window["bare_trees_before"] = len(bare_seen)
                if f.stats["eats"] > win_start_eats or tod > 330 + 900:
                    if window["bare_trees_before"] is None:
                        window["bare_trees_before"] = len(bare_seen)
                    cx, cy = cell_of(site[0], site[1])
                    window["place_value"] = round(f.place_value[cy][cx], 4)
                    break
    if window["place_value"] is None:
        cx, cy = cell_of(site[0], site[1])
        window["place_value"] = round(f.place_value[cy][cx], 4)
        if window["bare_trees_before"] is None:
            window["bare_trees_before"] = len(bare_seen)
    window["fruit_at_removal"] = fruit_at_removal
    return window
def score_pc():
    arms = {}
    for arm in ("intact", "none", "episodic_off", "workspace_off"):
        arms[arm] = {str(sd): probe_pc(sd, arm) for sd in SEEDS}
        print("P-C arm", arm, "done", flush=True)
    R["probes"]["P-C"] = arms
    save()
    intact_v = sum(1 for v in arms["intact"].values() if v["visited"])
    none_v = sum(1 for v in arms["none"].values() if v["visited"])
    epi_v = sum(1 for v in arms["episodic_off"].values() if v["visited"])
    ws_v = sum(1 for v in arms["workspace_off"].values() if v["visited"])
    pv_ok = all(abs(v["place_value"] or 0) <= 0.1 for v in arms["intact"].values() if v["valid"])
    verdict("P-C", intact_v >= 2 and none_v == 0 and epi_v == 0 and pv_ok,
            {"intact_visits": f"{intact_v}/3", "no_event_visits": f"{none_v}/3",
             "episodic_lesion_visits": f"{epi_v}/3", "workspace_lesion_visits": f"{ws_v}/3",
             "place_value_clause": pv_ok})
    return arms


def probe_gwt4(seed, arm):
    """I-GWT-4 v2 (amendment 5): the gwt4_probe world has NO fruit trees;
    one pile at P=(150,60) is the only food. Day 1 he starts ~24 units
    from it, hungry, and eats it dry (2 fruit: his pile belief is driven
    to ~0 by direct sight; the location survives as an episode). Night
    day 1 (tod >= 1250): the pile is restocked to 6 (declared), he is
    teleported to the far NW corner (25,165), hunger 80 / sleep 25 /
    thirst 30. Success = eats within 900 ticks. Under the mechanism
    repair, a far pile-episode target is admissible only in cycles whose
    full query chain completed — the single-query lesion stops after
    'memory', so its recalls never become treks."""
    from world import Pile
    # All arms share an INTACT day 1 (identical history, same seed); the
    # lesion is a test-time condition switched on at the teleport moment,
    # like HOT-3a's corruption. (Dev check: with the lesion on from
    # birth, the workspace_off arm's Phase-1 search never found the pile
    # on day 1 and the episode was invalid.)
    s = Sim(seed, layout="gwt4_probe")
    f = s.fajr
    ws = f.mind.workspace
    s.world.piles.append(Pile(700, 150.0, 60.0, 2))
    f.x, f.y = 128.0, 70.0
    f.drives.d["hunger"] = 70.0
    ep = None
    day1_ate = False
    total = 2 * TICKS_PER_DAY + 1100
    for t in range(total):
        s.step()
        tick = s.tick
        day = tick // TICKS_PER_DAY + 1
        tod = tick % TICKS_PER_DAY
        if day == 1 and f.stats["eats"] > 0:
            day1_ate = True
        if day == 1 and tod == 1000 and not day1_ate:
            f.drives.d["hunger"] = 85.0   # declared fallback nudge
        if ep is None and day == 1 and tod >= 1250:
            if not day1_ate:
                return {"valid": False, "invalid_reason": "no day-1 eat: episode never formed",
                        "success": False, "queries": 0, "task_cycles": 0, "order_samples": []}
            for p in s.world.piles:
                if p.id == 700:
                    p.fruit = 6        # declared restock
            if arm == "single":
                f.mind.workspace.lesion_single_query = True
            elif arm == "workspace_off":
                f.mind.workspace.lesion_off = True
                f.mind.steers = False
            f.x, f.y = 25.0, 165.0
            f.intent = None
            f.acting = None
            f.asleep = False
            f.drives.d["hunger"] = 80.0
            f.drives.d["sleep"] = 25.0
            f.drives.d["thirst"] = 30.0
            b = f.mind.beliefs.objects.get("pile:700")
            ep = {"start": tick, "q0": ws.metrics["queries"],
                  "tc0": ws.metrics["task_cycles"], "eats0": f.stats["eats"],
                  "order_samples": [], "valid": True,
                  "pile_belief_est_at_test": None if b is None else round(b["fruit_est"], 3)}
        elif ep is not None:
            if ws.last_queries and len(ep["order_samples"]) < 24:
                ep["order_samples"].append(tuple(ws.last_queries))
            if f.stats["eats"] > ep["eats0"]:
                ep.update(success=True, queries=ws.metrics["queries"] - ep["q0"],
                          task_cycles=ws.metrics["task_cycles"] - ep["tc0"])
                return ep
            if tick - ep["start"] > 900:
                ep.update(success=False, queries=ws.metrics["queries"] - ep["q0"],
                          task_cycles=ws.metrics["task_cycles"] - ep["tc0"])
                return ep
    if ep is not None:
        ep.update(success=False, queries=ws.metrics["queries"] - ep["q0"],
                  task_cycles=ws.metrics["task_cycles"] - ep["tc0"])
    return ep


def score_gwt4():
    arms = {}
    invalid_counts = {}
    for arm in ("intact", "single", "workspace_off"):
        eps = []
        for sd in (11, 42, 97, 5, 6, 7):   # 6 constructed episodes per arm (amendment 3)
            ep = probe_gwt4(sd, arm)
            if ep is not None:
                eps.append(ep)
        invalid_counts[arm] = sum(1 for e in eps if not e.get("valid", True))
        arms[arm] = [e for e in eps if e.get("valid", True)]
        print("GWT-4 arm", arm, "episodes", len(arms[arm]),
              "invalid", invalid_counts[arm], flush=True)
    R["probes"]["GWT-4"] = {k: [{kk: vv for kk, vv in e.items() if kk != "order_samples"} | 
                                {"order_samples": [list(x) for x in e["order_samples"][:6]]}
                            for e in v] for k, v in arms.items()}
    save()

    def succ(eps):
        return (sum(1 for e in eps if e["success"]) / len(eps)) if eps else None
    si, ss, sw = succ(arms["intact"]), succ(arms["single"]), succ(arms["workspace_off"])
    q = sum(e["queries"] for e in arms["intact"])
    tc = sum(e["task_cycles"] for e in arms["intact"])
    q_per_task = q / tc if tc else 0.0
    order_ok = 0
    order_n = 0
    for e in arms["intact"]:
        for sample in e["order_samples"]:
            seq = [x for x in sample if x in ("memory", "beliefs", "planning")]
            if len(seq) >= 2:
                order_n += 1
                if seq == sorted(seq, key=["memory", "beliefs", "planning"].index):
                    order_ok += 1
    verdict("I-GWT-4", si is not None and si >= 0.70 and q_per_task >= 2.0
            and ss is not None and (si - ss) >= 0.25,
            {"episodes_intact": len(arms["intact"]), "success_intact": si,
             "success_single_query": ss, "success_workspace_off": sw,
             "invalid_episodes": invalid_counts,
             "queries_per_task_cycle_intact": round(q_per_task, 3),
             "query_order_fraction": round(order_ok / max(1, order_n), 3),
             "query_order_samples": order_n})
    return arms


def probe_hot3a(seed, arm):
    """I-HOT-3 clause (a) at DECISION distance (protocol amendment 4):
    on day 4 he is placed at the shelter; his most-believed fruitful
    meadow tree (x > 90, i.e. > 60 units away, far outside vision) is —
    in the corrupt arm — set to believed-fruit 0. Hunger is raised and
    his FIRST adopted eat target is the measurement: the corruption can
    only be escaped by re-observation en route, so the decision itself
    is what is scored."""
    s = Sim(seed)
    f = s.fajr
    target = None
    result = {"valid": False}
    total = 4 * TICKS_PER_DAY + 480 + 500
    win = None
    for t in range(total):
        s.step()
        tick = s.tick
        day = tick // TICKS_PER_DAY + 1
        tod = tick % TICKS_PER_DAY
        if target is None and day == 4 and tod >= 480:
            cands = []
            for tr in s.world.trees:
                b = f.mind.beliefs.objects.get(f"tree:{tr.id}")
                if b and b["fruit_est"] >= 2.0 and tr.fruit >= 2 and tr.x > 90.0:
                    cands.append((b["fruit_est"], tr))
            if not cands:
                return result
            cands.sort(key=lambda c: -c[0])
            target = cands[0][1]
            f.x, f.y = 30.0, 20.0
            f.intent = None
            f.acting = None
            f.asleep = False
            if arm == "corrupt":
                f.mind.beliefs.objects[f"tree:{target.id}"]["fruit_est"] = 0.0
            f.drives.d["hunger"] = 75.0
            win = {"start": tick, "eats0": f.stats["eats"]}
            result["valid"] = True
            result["tree"] = target.id
            result["eat_adoptions"] = 0
            result["target_adoptions"] = 0
            result["_prev_eat_key"] = None
        elif win is not None:
            # count eat ADOPTIONS (a new eat intent whose target differs
            # from the previous tick's) — the frozen clause scores his
            # choices over attempts, not a single first step
            if f.intent is not None and f.intent[0] == "eat":
                key = (round(f.intent[1]), round(f.intent[2]))
                if key != result["_prev_eat_key"]:
                    result["eat_adoptions"] += 1
                    if math.hypot(f.intent[1] - target.x, f.intent[2] - target.y) < 6.0:
                        result["target_adoptions"] += 1
                    result["_prev_eat_key"] = key
            else:
                result["_prev_eat_key"] = None
            if tick - win["start"] > 700:
                n = result["eat_adoptions"]
                share = result["target_adoptions"] / n if n else 0.0
                result["target_share"] = round(share, 3)
                result["deviation_rate"] = round(1.0 - share, 3) if n else None
                result["ate_from_target"] = result["target_adoptions"] > 0 and share >= 0.5
                result["other_eats"] = f.stats["eats"] - win["eats0"]
                del result["_prev_eat_key"]
                return result
    if win is not None and "_prev_eat_key" in result:
        n = result["eat_adoptions"]
        share = result["target_adoptions"] / n if n else 0.0
        result["target_share"] = round(share, 3)
        result["deviation_rate"] = round(1.0 - share, 3) if n else None
        result["ate_from_target"] = result["target_adoptions"] > 0 and share >= 0.5
        result["other_eats"] = f.stats["eats"] - win["eats0"]
        del result["_prev_eat_key"]
    return result


def probe_ae1():
    """12 declared drive-vector scenarios on a settled being; the adopted
    intention must match an independent urgency-after-cost reference."""
    from world import river_center_x
    s = Sim(11)
    f = s.fajr
    s.run_ticks(2 * TICKS_PER_DAY + 600)   # settle: day 3, 10:00
    scenarios = [
        ("hunger_high", {"hunger": 85.0}),
        ("thirst_high", {"thirst": 85.0}),
        ("sleep_high", {"sleep": 85.0}),
        ("hunger_vs_thirst", {"hunger": 70.0, "thirst": 65.0}),
        ("sleep_vs_hunger", {"sleep": 70.0, "hunger": 65.0}),
        ("safety_high", {"safety": 80.0}),
        ("social_high", {"social": 80.0}),
        ("creation_empty", {"creation": 80.0}),
        ("creation_carry", {"creation": 80.0, "_carry": 1}),
        ("curiosity_high", {"curiosity": 85.0}),
        ("comfort_safety", {"comfort": 80.0, "safety": 70.0}),
        ("all_mild", {"hunger": 45.0, "thirst": 45.0, "sleep": 45.0}),
    ]
    kind_map = {"hunger": {"eat", "seek_food", "search"}, "thirst": {"drink"},
                "sleep": {"sleep"}, "safety": {"shelter", "flee"}, "comfort": {"shelter"},
                "social": {"social"}, "creation": {"gather", "place", "seek_material"},
                "curiosity": {"explore", "search"}}

    def reference(d):
        urg = {k: f.drives.urgency(k) for k in d}
        scores = {}
        best_tree = None
        for tr in s.world.trees:
            if tr.fruit > 0:
                sc = urg["hunger"] * min(d["hunger"], 34.0 * min(3.0, tr.fruit)) / (f._travel(tr.x, tr.y) + 12.0)
                if best_tree is None or sc > best_tree:
                    best_tree = sc
        scores["hunger"] = best_tree if best_tree is not None else urg["hunger"] * 0.3
        ry = max(10.0, min(170.0, f.y))
        scores["thirst"] = urg["thirst"] * min(d["thirst"], 55.0) / (f._travel(river_center_x(ry), ry) + 8.0)
        scores["sleep"] = urg["sleep"] * 3.2 / (1.0 + f._travel(*SHELTER_POS) / 60.0) * 1.0
        scores["safety"] = (urg["safety"] + urg["comfort"]) * 1.6 / (1.0 + f._travel(*SHELTER_POS) / 60.0)
        scores["comfort"] = scores["safety"]
        scores["social"] = urg["social"] * 2.4 / (1.0 + f._travel(*LANTERN_POS) / 60.0)
        if f.carry:
            scores["creation"] = urg["creation"] * 2.2 / (1.0 + f._travel(*BUILD_SITE) / 60.0)
        else:
            scores["creation"] = urg["creation"] * 0.9
        scores["curiosity"] = urg["curiosity"] * 0.575
        return max(scores.items(), key=lambda kv: kv[1])[0]

    out = []
    for name, vec in scenarios:
        for k in f.drives.d:
            f.drives.d[k] = 15.0
        for k, v in vec.items():
            if not k.startswith("_"):
                f.drives.d[k] = v
        f.carry = ["wood"] if vec.get("_carry") else []
        f.x, f.y = 120.0, 95.0
        f.asleep = False
        f.acting = None
        f.intent = None
        expected = reference(dict(f.drives.d))
        adopted = None
        for _ in range(12):
            s.step()
            if f.intent is not None:
                adopted = f.intent[0]
        match = adopted in kind_map.get(expected, set())
        out.append({"scenario": name, "expected_need": expected, "adopted": adopted, "match": match})
    matches = sum(1 for o in out if o["match"])
    distinct = len({o["adopted"] for o in out if o["adopted"]})
    verdict("I-AE-1", matches / len(out) >= 0.70 and distinct >= 3,
            {"scenarios": out, "match_fraction": round(matches / len(out), 3),
             "distinct_winners": distinct,
             "feedback_learning": "measured by P-E regression items G3a/G3b (frozen cross-reference)"})
    return out


def probe_pb_supplementary(seed=11, days=3):
    """P-B supplementary (amendment: contribution measures, NO verdict
    weight): the monitor's epistemic signature — how many distinct trees
    ever carried an epistemic flag (low-reliability re-observation
    requests) and how many observe intents were adopted. Under the
    constant-0.5 lesion no percept is ever below LOW_REL, so the
    signature must be empty by construction; the intact arm shows
    whether the pathway actually fires in life."""
    out = {}
    for arm, cfg in (("intact", None), ("monitor_constant", {"monitor_constant": True})):
        s = Sim(seed, mind_config=cfg)
        f = s.fajr
        flags_seen = set()
        observe_adoptions = 0
        prev = None
        for t in range(days * TICKS_PER_DAY):
            s.step()
            flags_seen |= set(f.mind.beliefs.epistemic_flags.keys())
            cur = f.intent
            if cur is not None and cur[0] == "observe" and \
                    (prev is None or prev[0] != "observe" or prev[1:] != cur[1:]):
                observe_adoptions += 1
            prev = cur
        out[arm] = {"distinct_flagged_objects": len(flags_seen),
                    "observe_adoptions": observe_adoptions}
    return out


def fear_probe_workspace_off():
    """P-A fear clause: gate G3b's assay, workspace lesioned, drop >= 50%."""
    per = {}
    for seed in (5, 6, 7):
        s = Sim(seed=seed, layout="fear_probe", mind_config={"workspace_off": True})
        th = s.world.thickets[0]
        entries_pre = entries_post = 0
        first_prick = None
        cond_tick = None
        total = 4 * TICKS_PER_DAY
        seen_entries = 0
        for i in range(total):
            s.step()
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
            else:
                entries_post += new_entries
        if first_prick is None:
            per[str(seed)] = {"conditioned": False, "drop": None}
            continue
        if cond_tick is None:
            cond_tick = first_prick
        rate_pre = entries_pre / max(1, first_prick) * 1000
        rate_post = entries_post / max(1, total - cond_tick) * 1000
        drop = (rate_pre - rate_post) / rate_pre if rate_pre > 0 else 0.0
        per[str(seed)] = {"conditioned": True, "entry_drop": round(drop, 3)}
    return per


def probe_pb_dynamic(seed, arm):
    """P-B v3 (GATE v3-3): DYNAMIC ground truth. The frozen claim is
    that the monitor layer does causal work on belief accuracy; sitting
    2 proved the frozen static-truth operationalisation cannot see it
    (near-static fruit -> end-state MAE rewards shrinkage, median ratio
    0.972, while the monitor demonstrably discriminates: HOT-2 gap
    0.2627 / AUC 0.8622). In this probe world the truth MOVES on a
    declared schedule: from day 2, every 480 ticks, 4 trees (probe-side
    PRNG seeded per seed — the identical schedule in both arms, and
    independent of the being's own PRNG stream) have their fruit
    re-rolled uniform 0..6. MAE is sampled hourly against the CURRENT
    true fruit of every tree he holds a belief about, days 2-7. A
    clamped/static belief cannot track a moving truth; a monitor that
    gates updates by calibrated reliability can. Clause bars keep their
    frozen form (see main). Wasted arrivals are the body's own counter
    (arrival to eat at a tree whose current fruit is 0) — under a moving
    truth a stale belief finally produces them."""
    import random as _random
    cfg = {"monitor_constant": True} if arm == "monitor_constant" else None
    s = Sim(seed, mind_config=cfg)
    f = s.fajr
    sched = _random.Random(10000 + seed)
    samples = []
    total = 7 * TICKS_PER_DAY
    for t in range(total):
        s.step()
        tick = s.tick
        if tick >= TICKS_PER_DAY and tick % 480 == 0:
            for i in sched.sample(range(len(s.world.trees)), 4):
                s.world.trees[i].fruit = sched.randint(0, 6)
        if tick % 60 == 0 and tick >= TICKS_PER_DAY:
            e, n = belief_mae(s)
            if e is not None:
                samples.append(e)
    return {"mae_dynamic": round(sum(samples) / max(1, len(samples)), 4),
            "mae_samples": len(samples),
            "wasted": f.stats.get("wasted_arrivals", 0),
            "in_band": round(s.metrics["ticks_in_band"] / max(1, s.metrics["ticks_total"]), 4)}


# ------------------------------------------------------------------ main
def main():
    print("== intact lives ==", flush=True)
    sims, cols = {}, {}
    for sd in SEEDS:
        sims[sd], cols[sd] = life_run(sd, instrument=(sd == 11))
        print("life seed", sd, "done", flush=True)
    print("== binding life (RPT-2, binding_probe world) ==", flush=True)
    bsim, bcol = life_run(11, instrument=True, layout="binding_probe")
    bpm = bsim.fajr.mind.perception.metrics
    R["probes"]["RPT-2-binding-life"] = {
        "qualifying_scenes": bcol["rpt2_bind"][1], "binding_ok": bcol["rpt2_bind"][0],
        "exact_count_binding_all_claims": round(bpm["binding_ok"] / max(1, bpm["binding_n"]), 4),
        "binding_claims_n": bpm["binding_n"],
        "eats": bsim.fajr.stats["eats"], "distance": round(bsim.fajr.stats["distance"])}
    save()
    print("binding life done; qualifying scenes", bcol["rpt2_bind"][1], flush=True)
    score_from_lives(sims, cols, binding_col=bcol)

    print("== lesion lives (feedforward / topdown) ==", flush=True)
    ff, ff_col = life_run(11, cfg={"feedforward_off": True}, instrument=True)
    td, _tdc = life_run(11, cfg={"topdown_off": True})
    f1_i, f1_f, f1_t = score_lesion_runs(sims[11], ff, td)
    score_rpt1(cols[11], ff_col, f1_i, f1_f)
    score_hot1(sims[11], td, f1_i, f1_t)
    score_hot3(sims[11], cols[11], None)

    print("== P-B: dynamic-truth probe lives (v3 construction) ==", flush=True)
    pb = {"construction": ("GATE v3 amendment 3: ground truth moves on a declared schedule "
                           "(4 trees re-rolled every 480 ticks from day 2, probe-side seed, "
                           "identical across arms); MAE sampled hourly vs CURRENT truth"),
          "per_seed": {}}
    mae_ratios, wasted_pairs, inband_pairs = [], [], []
    for sd in SEEDS:
        intact_r = probe_pb_dynamic(sd, "intact")
        lesion_r = probe_pb_dynamic(sd, "monitor_constant")
        ratio = (lesion_r["mae_dynamic"] / intact_r["mae_dynamic"]) if intact_r["mae_dynamic"] else None
        if ratio is not None:
            mae_ratios.append(ratio)
        wasted_pairs.append((lesion_r["wasted"], intact_r["wasted"]))
        inband_pairs.append((lesion_r["in_band"], intact_r["in_band"]))
        pb["per_seed"][str(sd)] = {"mae_dynamic_lesion": lesion_r["mae_dynamic"],
                                   "mae_dynamic_intact": intact_r["mae_dynamic"],
                                   "mae_ratio": None if ratio is None else round(ratio, 3),
                                   "mae_samples": intact_r["mae_samples"],
                                   "wasted_lesion": lesion_r["wasted"],
                                   "wasted_intact": intact_r["wasted"],
                                   "in_band_lesion": lesion_r["in_band"],
                                   "in_band_intact": intact_r["in_band"]}
        print("P-B dynamic seed", sd, "done", flush=True)
    mae_ratios_sorted = sorted(mae_ratios)
    median_ratio = mae_ratios_sorted[len(mae_ratios_sorted) // 2] if mae_ratios_sorted else None
    wasted_ok = all(wl >= 1.5 * wi if wi > 0 else True for (wl, wi) in wasted_pairs)
    inband_ok = all(ib_l >= ib_i - 0.05 for (ib_l, ib_i) in inband_pairs)
    verdict("P-B", median_ratio is not None and median_ratio >= 1.5 and wasted_ok and inband_ok,
            {"per_seed": pb["per_seed"], "median_mae_ratio": None if median_ratio is None else round(median_ratio, 3),
             "mae_definition": "hourly |belief fruit_est - CURRENT true fruit| over believed trees, days 2-7, dynamic-truth probe lives (GATE v3-3)",
             "wasted_clause": wasted_ok, "in_band_clause": inband_ok,
             "wasted_note": "wasted arrivals = body counter (arrival to eat at a tree whose CURRENT fruit is 0); clause evaluated literally (lesion >= 1.5 x intact), vacuous-true per seed where the intact arm wastes none"})
    R["probes"]["P-B"] = pb
    save()

    print("== P-B supplementary (epistemic signature, no verdict weight) ==", flush=True)
    R["probes"]["P-B-supplementary"] = {"epistemic_signature": probe_pb_supplementary()}
    save()
    print("P-B supplementary done", flush=True)

    print("== P-C probe ==", flush=True)
    pc_arms = score_pc()

    print("== GWT-4 probe ==", flush=True)
    g4_arms = score_gwt4()

    print("== P-A assembly ==", flush=True)
    ws_lives = {}
    for sd in SEEDS:
        ws_lives[sd], _ = life_run(sd, cfg={"workspace_off": True})
    inband_ws = {str(sd): round(ws_lives[sd].metrics["ticks_in_band"] / ws_lives[sd].metrics["ticks_total"], 4)
                 for sd in SEEDS}
    fear_ws = fear_probe_workspace_off()
    intact_succ = None
    eps_i = g4_arms["intact"]
    eps_w = g4_arms["workspace_off"]
    if eps_i:
        intact_succ = sum(1 for e in eps_i if e["success"]) / len(eps_i)
    ws_succ = (sum(1 for e in eps_w if e["success"]) / len(eps_w)) if eps_w else None
    gwt4_drop = (intact_succ - ws_succ) / intact_succ if (intact_succ and ws_succ is not None) else None
    pc_ws_visits = sum(1 for v in pc_arms["workspace_off"].values() if v["visited"])
    fear_drops = [v["entry_drop"] for v in fear_ws.values() if v.get("entry_drop") is not None]
    fear_ok = len(fear_drops) >= 2 and all(d >= 0.50 for d in fear_drops)
    verdict("P-A", pc_ws_visits == 0 and gwt4_drop is not None and gwt4_drop >= 0.40
            and fear_ok and all(v >= 0.90 for v in inband_ws.values()),
            {"pc_workspace_visits": f"{pc_ws_visits}/3",
             "gwt4_success_intact": intact_succ, "gwt4_success_workspace_off": ws_succ,
             "gwt4_relative_drop": None if gwt4_drop is None else round(gwt4_drop, 3),
             "fear_probe_workspace_off": fear_ws, "in_band_workspace_off": inband_ws})

    print("== HOT-3 probe (a) ==", flush=True)
    h3a = {"corrupt": {}, "control": {}}
    for arm in ("corrupt", "control"):
        for sd in SEEDS:
            h3a[arm][str(sd)] = probe_hot3a(sd, arm)
    R["probes"]["HOT-3a"] = h3a
    save()
    valid_c = [v for v in h3a["corrupt"].values() if v["valid"]]
    valid_k = [v for v in h3a["control"].values() if v["valid"]]
    devs = [v["deviation_rate"] for v in valid_c if v.get("deviation_rate") is not None]
    avoid_frac = sum(devs) / len(devs) if devs else None
    ate_corrupt = (sum(v["target_share"] for v in valid_c) / len(valid_c)) if valid_c else None
    ate_control = (sum(v["target_share"] for v in valid_k) / len(valid_k)) if valid_k else None
    clause_a = (avoid_frac is not None and avoid_frac >= 0.50
                and ate_control is not None and ate_corrupt is not None and ate_control > ate_corrupt)
    h3 = R["items"]["I-HOT-3"]
    h3["clause_a"] = {"avoidance_fraction_corrupt": avoid_frac,
                      "ate_from_target_corrupt": ate_corrupt,
                      "ate_from_target_control": ate_control, "pass": clause_a}
    h3["pass"] = bool(clause_a and h3["clause_b_pass"] and h3["clause_c_pass"])
    h3["label"] = "SCOPED" if h3["pass"] else "FAILED"
    save()
    print("[I-HOT-3]", "PASS (SCOPED)" if h3["pass"] else "FAIL", flush=True)

    print("== AST-1 probe (b): vision-leads construction (v3) ==", flush=True)
    # GATE v3-5 branch taken: the vision-leads construction was calibrated
    # at full size by the probe author and proved NON-INSTANTIABLE, and the
    # outcome bar independently unreachable (decision stage never reads
    # vision winners). Per the amendment, the architectural fact is FILED
    # and no probe module is shipped (a shipped one could only pass by
    # rigging). The item records the filed evidence verbatim from
    # results/ast1b_filing_evidence.json (18 episodes/arm/construction,
    # seeds 11/42/97; full account: results/ast1b_v3_calibration.md).
    _ev = json.load(open(os.path.join(os.path.dirname(OUT_PATH),
                                      "ast1b_filing_evidence.json"),
                         encoding="utf-8"))

    def _ev_pool(construction, arm):
        a = _ev[construction][arm]
        eps = a["episodes"]
        succ = sum(1 for e in eps if e["success"]) / max(1, len(eps))
        return succ, a["vision_pooled"], len(eps)
    ast = {"filed": True,
           "filing": "results/ast1b_v3_calibration.md",
           "evidence": "results/ast1b_filing_evidence.json",
           "constructions": {}}
    for _cname in ("noon_clear", "dawn_rain", "bound_fruit6",
                   "control_nosignal"):
        _so, _vo, _no = _ev_pool(_cname, "schema_on")
        _sf, _vf, _nf = _ev_pool(_cname, "schema_off")
        ast["constructions"][_cname] = {
            "success_on": round(_so, 4), "success_off": round(_sf, 4),
            "vision_top_fraction_on": round(_vo, 4),
            "vision_top_fraction_off": round(_vf, 4),
            "episodes_on": _no, "episodes_off": _nf}
    R["probes"]["AST-1b"] = ast
    save()
    # the mandated construction is noon_clear (the brief's own scenario)
    succ_on = ast["constructions"]["noon_clear"]["success_on"]
    succ_off = ast["constructions"]["noon_clear"]["success_off"]
    vtf_on = ast["constructions"]["noon_clear"]["vision_top_fraction_on"]
    vtf_off = ast["constructions"]["noon_clear"]["vision_top_fraction_off"]
    n_on = ast["constructions"]["noon_clear"]["episodes_on"]
    n_off = ast["constructions"]["noon_clear"]["episodes_off"]
    instantiation = vtf_on >= 0.10
    outcome_gap = succ_on - succ_off
    clause_b = bool(instantiation and outcome_gap >= 0.15)
    # pooled schema accuracy (clause a) across seeds — as sitting 2
    cn = sum(sims[sd].fajr.mind.workspace.schema.metrics["n"] for sd in SEEDS)
    cc = sum(sims[sd].fajr.mind.workspace.schema.metrics["correct"] for sd in SEEDS)
    cm = sum(sims[sd].fajr.mind.workspace.schema.metrics["majority_correct"] for sd in SEEDS)
    acc_p = cc / max(1, cn)
    maj_p = cm / max(1, cn)
    clause_a = acc_p >= 1.5 * maj_p
    R["items"]["I-AST-1"] = {
        "pass": bool(clause_a and clause_b), "label": "SCOPED" if (clause_a and clause_b) else "FAILED",
        "clause_a_pooled": {"accuracy": round(acc_p, 4), "majority": round(maj_p, 4),
                            "ratio": round(acc_p / max(1e-9, maj_p), 3), "pass": clause_a},
        "clause_a_seed11": R["items"].pop("I-AST-1a"),
        "clause_b": {"construction": "vision-leads scenario (GATE v3 amendment 5) — calibrated at full size, proved non-instantiable; architectural filing recorded instead of a probe run (results/ast1b_v3_calibration.md)",
                     "instantiation_vision_top_fraction_on": round(vtf_on, 4),
                     "instantiation_threshold": 0.10, "instantiation_pass": bool(instantiation),
                     "vision_top_fraction_off": round(vtf_off, 4),
                     "success_schema_on": round(succ_on, 4), "success_schema_off": round(succ_off, 4),
                     "outcome_gap": round(outcome_gap, 4), "outcome_bar": 0.15,
                     "episodes_on": n_on, "episodes_off": n_off,
                     "pass": clause_b}}
    save()
    print("[I-AST-1]", "PASS (SCOPED)" if (clause_a and clause_b) else "FAIL", flush=True)

    print("== AE-1 scenarios ==", flush=True)
    probe_ae1()

    print("== P-E: Phase-1 battery re-run (mind on, as frozen) ==", flush=True)
    import subprocess
    # v2 plumbing (GATE amendment 9), kept in v3: ALS_GATE_OUT redirects the Phase-1
    # battery's report to phase1_gate_v3.json — the recorded Phase-1
    # file is never overwritten by a sitting-3 regression run.
    pe_path = os.path.join(os.path.dirname(OUT_PATH), "phase1_gate_v3.json")
    env = dict(os.environ)
    env["ALS_GATE_OUT"] = pe_path
    proc = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate_battery.py")],
                          capture_output=True, text=True, timeout=2400, env=env)
    pe = {"subprocess_rc": proc.returncode, "stdout_tail": proc.stdout[-1500:],
          "output_file": "phase1_gate_v3.json via ALS_GATE_OUT"}
    try:
        with open(pe_path, encoding="utf-8") as fh:
            p1 = json.load(fh)
        gates = p1["gates"]
        need = ["G1_homeostasis", "G2_survival", "G3a_food_learning",
                "G3b_fear_learning", "G4_circadian", "G5_checkpoint"]
        pe["gates"] = {k: gates[k]["pass"] for k in need}
        pe["G3c_as_rerun_printed"] = gates["G3c_place_memory"]["pass"]
        pe["G3c_phase1_recorded"] = "FAIL (as recorded in Phase 1; not re-scored as a Phase-2 item)"
        pe["pass"] = all(pe["gates"].values())
    except Exception as exc:  # noqa: BLE001
        pe["pass"] = False
        pe["error"] = str(exc)
    R["items"]["P-E"] = pe
    save()
    print("[P-E]", "PASS" if pe.get("pass") else "FAIL", flush=True)

    # labels for indicator items
    for name, item in R["items"].items():
        if name.startswith("I-") and "label" not in item:
            item["label"] = "IMPLEMENTED" if item.get("pass") else "FAILED"
    # I-GWT-4 / I-AE-1 / probes are pass-fail as recorded
    R["summary"] = {k: (v.get("label") or ("PASS" if v.get("pass") else "FAIL"))
                    for k, v in R["items"].items()}
    save()
    print(json.dumps(R["summary"], ensure_ascii=False, indent=2), flush=True)
    print("wrote", OUT_PATH, flush=True)


if __name__ == "__main__":
    main()
