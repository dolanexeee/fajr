"""Engineering diagnostic — NOT the formal battery. Prints raw numbers."""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from sim import Sim
from config import TICKS_PER_DAY

def calibration(tape, hard_only=True):
    rows = [(r, c) for (r, c, h) in tape if (h == 1 or not hard_only)]
    if not rows:
        return None
    bins = {}
    for r, c in rows:
        b = min(4, int(r * 5))
        bins.setdefault(b, []).append((r, c))
    ece = 0.0
    for b, rs in bins.items():
        conf = sum(r for r, _ in rs) / len(rs)
        acc = sum(c for _, c in rs) / len(rs)
        ece += len(rs) / len(rows) * abs(conf - acc)
    rc = [r for r, c in rows if c == 1]
    ri = [r for r, c in rows if c == 0]
    return {"n": len(rows), "ece": round(ece, 3),
            "mean_rel_correct": round(sum(rc) / max(1, len(rc)), 3),
            "mean_rel_incorrect": round(sum(ri) / max(1, len(ri)), 3),
            "n_incorrect": len(ri),
            "base_acc": round(sum(c for _, c in rows) / len(rows), 3)}

def belief_mae(sim):
    m = sim.fajr.mind
    errs, n = 0.0, 0
    for key, b in m.beliefs.objects.items():
        if b["kind"] == "tree":
            t = sim.world.trees[b["oid"]]
            errs += abs(b["fruit_est"] - t.fruit)
            n += 1
    return (errs / n if n else None), n

def run(seed, days=7, cfg=None, label=""):
    t0 = time.time()
    s = Sim(seed, mind_config=cfg)
    sm_err = {"hunger": [], "thirst": [], "sleep": []}
    for t in range(days * TICKS_PER_DAY):
        s.step()
        if s.tick % 60 == 0:
            sm = s.fajr.mind.selfmodel
            for k in sm_err:
                sm_err[k].append(abs(sm.estimate(k, s.tick, asleep=s.fajr.asleep) - s.fajr.drives.d[k]))
    f = s.fajr
    m = f.mind
    mae, nb = belief_mae(s)
    wasted = sum(1 for d in m.decision_log if d["belief_fruit"] >= 0.6 and d["world_fruit"] == 0)
    print(f"== {label or 'intact'} seed {seed} ({time.time()-t0:.1f}s)")
    print("  stats", f.stats, "| in-band %.3f" % (s.metrics["ticks_in_band"] / s.metrics["ticks_total"]),
          "| night-sleep", (round(s.metrics["sleep_ticks_night"] / max(1, s.metrics["sleep_ticks"]), 3)))
    print("  perception: F1 %.3f binding %.3f fusion %.3f | persist mean %.1f kept %.2f | err day1 %.3f dayN %.3f" % (
        m.perception.metrics["f1_sum"] / max(1, m.perception.metrics["f1_n"]),
        m.perception.metrics["binding_ok"] / max(1, m.perception.metrics["binding_n"]),
        m.perception.metrics["fusion_ok"] / max(1, m.perception.metrics["fusion_n"]),
        m.perception.metrics["occlusion_persist_sum"] / max(1, m.perception.metrics["occlusion_persist_n"]),
        m.perception.metrics["occlusion_kept"] / max(1, m.perception.metrics["occlusion_events"]),
        (m.perception.metrics["err_day"].get(1, [0, 1])[0] / max(1, m.perception.metrics["err_day"].get(1, [0, 1])[1])),
        (m.perception.metrics["err_day"].get(days, [0, 1])[0] / max(1, m.perception.metrics["err_day"].get(days, [0, 1])[1]))))
    print("  monitor: trained", m.monitor.n_trained, "calib(hard tape)", calibration(m.tape_monitor),
          "\n           calib(settled claims)", calibration([(r, c, 1) for (r, c) in m.monitor.tape], hard_only=False))
    print("  beliefs: MAE", None if mae is None else round(mae, 3), "over", nb, "trees | wasted arrivals", f.stats.get("wasted_arrivals", 0),
          "decision_log", len(m.decision_log),
          "| weights low/high", [round(v, 3) for v in m.beliefs.mean_update_weights()],
          "| revisions", m.beliefs.stats["revisions"], "reobserve", m.beliefs.stats["reobserve_marks"],
          "flags_now", len(m.beliefs.epistemic_flags))
    wsm = m.workspace.metrics
    print("  workspace: cycles", wsm["cycles"], "avg proposals %.1f" % (wsm["proposals_sum"] / max(1, wsm["cycles"])),
          "| receipts", wsm["broadcast_receipts"], "| queries", wsm["queries"], "task_cycles", wsm["task_cycles"],
          "| captures", wsm["capture_events"], "| excluded_salient", wsm["excluded_salient"])
    acc, maj = m.workspace.schema.accuracy()
    print("  schema: acc %.3f majority %.3f n %d" % (acc, maj, m.workspace.schema.metrics["n"]))
    print("  selfmodel MAE", {k: round(sum(v) / max(1, len(v)), 2) for k, v in sm_err.items()},
          "| endurance log", len(m.selfmodel.metrics["endurance_log"]),
          "acc", (lambda L: round(sum(1 for r in L if r[2] == r[3]) / len(L), 3) if L else None)(m.selfmodel.metrics["endurance_log"]))
    print("  episodic:", len(m.episodic.episodes), "recorded", m.episodic.metrics["recorded"],
          "retrievals", m.episodic.metrics["retrievals"], "hits", m.episodic.metrics["retrieval_hits"],
          "consolidations", m.episodic.consolidations, "semantic", len(m.episodic.semantic))
    print("  fwd: overall %.3f skill %.3f day1 %s dayN %s | fm_participated %d/%d boosts %d" % (
        (m.fwd.metrics["pred_correct"] / max(1, m.fwd.metrics["pred_n"])),
        ((m.fwd.metrics["pred_correct"] - m.fwd.metrics.get("base_correct", 0)) / max(1, m.fwd.metrics["pred_n"])),
        m.fwd.accuracy_by_day(1), m.fwd.accuracy_by_day(days),
        m.metrics["fm_participated"], m.metrics["decisions"], m.metrics.get("schema_boosts", 0)))
    print("  hunger latencies", f.hunger_latencies[:8])
    return s

def auc(tape):
    rows = [(r, c) for (r, c, h) in tape]
    pos = sorted(r for r, c in rows if c == 1)
    neg = sorted(r for r, c in rows if c == 0)
    if not pos or not neg:
        return None
    import bisect
    score = 0.0
    for r in pos:
        lo = bisect.bisect_left(neg, r)
        hi = bisect.bisect_right(neg, r)
        score += lo + 0.5 * (hi - lo)
    return round(score / (len(pos) * len(neg)), 3)

if __name__ == "__main__":
    for sd in (11, 42, 97):
        s = run(sd)
        m = s.fajr.mind
        cal = calibration(m.tape_monitor, hard_only=False)
        print("  >> seed", sd, "HOT2 harness tape:", cal, "AUC", auc(m.tape_monitor),
              "| entry_counts", m.workspace.metrics["entry_counts"],
              "| propose_cycles", m.workspace.metrics["propose_cycles"])
    # checkpoint determinism
    a = Sim(42); a.run_days(7)
    b = Sim(42); b.run_days(3)
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "ck.json")
    b.save(p)
    b2 = Sim.load(p); b2.run_days(4)
    ha = Sim.canonical_hash(a.state_dict()); hb = Sim.canonical_hash(b2.state_dict())
    print("CHECKPOINT deterministic:", ha == hb)
    # lesion smoke
    for cfg, name in [({"workspace_off": True}, "workspace_off"), ({"monitor_constant": True}, "monitor_const"),
                      ({"episodic_off": True}, "episodic_off"), ({"master_off": True}, "master_off"),
                      ({"schema_off": True}, "schema_off"), ({"feedforward_off": True}, "ff_off"),
                      ({"topdown_off": True}, "topdown_off"), ({"single_query": True}, "single_query")]:
        s = Sim(11, mind_config=cfg); s.run_ticks(1500)
        print("lesion", name, "OK stats", s.fajr.stats)
