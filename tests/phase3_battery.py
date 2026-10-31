"""Phase-3 battery — PHASE3-GATE §6, frozen bars.

Main lives (seeds 11/42/97, 5 days, conversation active): grounding
verification of every utterance at its own tick; retention probe at
day-5 start; then scripted comprehension (12) and honesty (12 probes
+ 3 distractors) sessions on day 5.
Absence lives (same seeds): fruit removed at day-3 start; request due
by tick 7200; world-mutation check.
Regression spot-checks: phase-2 battery probes (GWT-1, HOT-2, P-E) and
water W3 on the integrated build with no conversation active.

Run:  python3 tests/phase3_battery.py [out.json]
"""
import json
import math
import sys

sys.path.insert(0, "src")

from sim import Sim
from conversation import Conversation
from lexicon import (WORDS, FORM_MAP, normalise, state_condition,
                     state_value, place_word_at)
from host import CURRICULUM
from config import TICKS_PER_DAY, LANTERN_POS, SHELTER_POS
from world import dist_to_river

SEEDS = (11, 42, 97)
HILL_POS = (208.0, 42.0)

# ------------------------------------------------------------ grounding
DRIVE_OF_VERB = {"eat": "hunger", "drink": "thirst", "sleep_v": "sleep"}


def _band_of(fe):
    return "much" if fe >= 2.5 else ("little" if fe >= 0.6 else "empty")


def verify_claim(sim, tongue, u, claim):
    """(faithful, kind_of_truth, note). Faithfulness = the report
    matches its source (her state / her belief / her memory). For
    belief fruit bands, truth vs the world is reported separately."""
    f = sim.fajr
    t = claim["type"]
    if t == "state":
        wid = claim["word"]
        if not state_condition(wid, sim):
            return False, None, "state word asserted but condition false now"
        v = state_value(wid, sim)
        if abs(v - claim["value"]) > 3.0:
            return False, None, f"value drift {v:.1f} vs {claim['value']}"
        return True, None, ""
    if t == "state_negation":
        return (not state_condition(claim["word"], sim)), None, ""
    if t == "belief":
        kind, oid = claim["key"].split(":")
        b = f.mind.beliefs.objects.get(claim["key"]) or \
            f.mind.beliefs.objects.get((kind, int(oid)))
        if b is None:
            return False, None, "belief object gone at its own tick"
        if abs(b["fruit_est"] - claim["fruit_est"]) > 0.15:
            return False, None, "fruit_est mismatch vs her belief"
        if abs(b["conf"] - claim["conf"]) > 0.10:
            return False, None, "conf mismatch vs her belief"
        hedge = "plain" if b["conf"] >= 0.70 else \
            ("think" if b["conf"] >= 0.45 else "unsure")
        if hedge != claim["hedge"]:
            return False, None, "hedge band mismatch"
        truth = None
        if claim.get("fruit_band"):
            if _band_of(b["fruit_est"]) != claim["fruit_band"]:
                return False, None, "fruit band mismatch vs her belief"
            pool = sim.world.trees if kind == "tree" else sim.world.piles
            if int(oid) < len(pool):
                truth = (_band_of(pool[int(oid)].fruit) == claim["fruit_band"])
        return True, truth, ""
    if t == "memory":
        hit = any(e["tick"] == claim["episode_tick"] and
                  e["kind"] == claim["kind"]
                  for e in f.mind.episodic.episodes)
        return hit, None, "" if hit else "episode not in her memory"
    if t == "place":
        return place_word_at(claim["x"], claim["y"]) == claim["word"], None, ""
    if t == "identify":
        if "key" in claim:
            kind, oid = claim["key"].split(":")
            if kind == "tree":
                return int(oid) < len(sim.world.trees), None, ""
            if kind == "pile":
                return int(oid) < len(sim.world.piles), None, ""
            return True, None, ""   # scene-sourced kinds: key came from her senses
        scene = f.mind.perception.scene
        ok = scene is not None and any(
            ({"tree": "tree", "thicket": "thicket", "pile": "pile",
              "structure": "structure", "shelter": "shelter"}.get(o.kind)
             == claim["word"]) and not o.generated
            for o in scene.objects.values())
        return ok, None, "" if ok else "echo with no such object in scene"
    if t == "intent":
        verb = claim["verb"]
        drive = DRIVE_OF_VERB.get(verb)
        if drive is None:
            return True, None, ""
        has_state = any(c["type"] == "state" and
                        WORDS[c["word"]]["anchor"].split(":")[1] == drive
                        for c in u["claims"])
        return has_state or (f.intent is not None and f.intent[0] == verb), \
            None, "" if has_state else "want-clause without its drive state"
    if t == "absence":
        if claim["word"] == "fruit":
            total = sum(o.fruit for o in sim.world.trees) + \
                sum(o.fruit for o in sim.world.piles)
            return total < 0.6, None, f"world fruit total {total:.2f}"
        return dist_to_river(f.x, f.y) > 40.0, None, ""
    if t == "request":
        return f.drives.d[claim["drive"]] >= 70.0, None, \
            f"drive at utterance {f.drives.d[claim['drive']]:.1f}"
    if t == "self":
        return claim.get("value") in ("فجر", "المضيف"), None, ""
    if t == "unknown_answer":
        s = claim.get("word_surface")
        if s is None:
            return True, None, "bare لا أعرف (probe-verified behaviourally)"
        hit = FORM_MAP.get(normalise(s))
        if hit is None:
            return True, None, ""
        return not tongue.known(hit[0]), None, "named a word she knows"
    return False, None, f"unhandled claim type {t}"


class Verifier:
    """Watches a conversation; verifies each new utterance at its tick."""

    def __init__(self, sim, conv):
        self.sim = sim
        self.conv = conv
        self.seen = 0
        self.records = []        # {tick, day, context, text, claims:[...]}

    def poll(self):
        utts = self.conv.tongue.utterances
        while self.seen < len(utts):
            u = utts[self.seen]
            self.seen += 1
            checks = []
            for c in u["claims"]:
                ok, truth, note = verify_claim(self.sim, self.conv.tongue, u, c)
                checks.append({"claim": c, "faithful": ok,
                               "world_true": truth, "note": note})
            self.records.append({"tick": u["tick"], "day": u["day"],
                                 "context": u["context"], "text": u["text"],
                                 "checks": checks})


def step(sim, conv, ver=None):
    sim.step()
    conv.step()
    if ver is not None:
        ver.poll()


# ------------------------------------------------------------ sessions
def wait_until(sim, conv, ver, pred, max_ticks):
    for _ in range(max_ticks):
        if pred():
            return True
        step(sim, conv, ver)
    return pred()


def run_window(sim, conv, ver, ticks, check):
    """Step up to `ticks`, returning the first tick where check() holds."""
    for i in range(ticks):
        step(sim, conv, ver)
        if check():
            return i + 1
    return None


def comprehension_script(sim, conv, ver):
    f = sim.fajr

    def belief_fruit():
        return any(b["kind"] in ("tree", "pile") and b["fruit_est"] >= 0.6
                   for b in f.mind.beliefs.objects.values())

    def material_visible():
        return f._raw is not None and any(k == "material"
                                           for (k, *_r) in f._raw.visible)

    items = [
        ("eat", "كلي", lambda: not f.asleep and belief_fruit(),
         lambda s0: f.stats["eats"] > s0),
        ("drink", "اشربي", lambda: not f.asleep,
         lambda s0: f.stats["drinks"] > s0),
        ("sleep", "نامي",
         lambda: not f.asleep and (sim.tick % TICKS_PER_DAY >= 1000
                                   or f.drives.d["sleep"] >= 35.0),
         lambda s0: f.asleep),
        ("go_river", "اذهبي إلى النهر", lambda: not f.asleep,
         lambda s0: dist_to_river(f.x, f.y) < 12.0),
        ("go_shelter", "اذهبي إلى المأوى", lambda: not f.asleep,
         lambda s0: math.hypot(SHELTER_POS[0] - f.x,
                               SHELTER_POS[1] - f.y) < 12.0),
        ("come", "تعالي",
         lambda: not f.asleep and math.hypot(LANTERN_POS[0] - f.x,
                                             LANTERN_POS[1] - f.y) > 12.0,
         lambda s0: math.hypot(LANTERN_POS[0] - f.x,
                               LANTERN_POS[1] - f.y) < 11.0),
        ("bathe", "استحمي", lambda: not f.asleep,
         lambda s0: f.stats["bathes"] > s0),
        ("swim", "اسبحي", lambda: not f.asleep,
         lambda s0: f.stats["swims"] > s0 or
         (f._raw is not None and f._raw.in_water)),
        ("gather", "اجمعي", lambda: not f.asleep and material_visible(),
         lambda s0: f.stats["gathers"] > s0),
        ("build", "ابني", lambda: not f.asleep and len(f.carry) >= 1,
         lambda s0: f.stats["places"] > s0),
        ("look", "انظري", lambda: not f.asleep,
         lambda s0: f.intent is not None and f.intent[0] == "observe"),
        ("drink_water", "اشربي ماء", lambda: not f.asleep,
         lambda s0: f.stats["drinks"] > s0),
    ]
    results = []
    stat_key = {"eat": "eats", "drink": "drinks", "bathe": "bathes",
                "swim": "swims", "gather": "gathers", "build": "places"}
    for name, line, pre, done in items:
        if not wait_until(sim, conv, ver, pre, 1800):
            results.append({"item": name, "line": line, "issued": False,
                            "complied": False, "note": "precondition never met"})
            continue
        s0 = f.stats.get(stat_key.get(name, ""), 0)
        conv.last_directive = None
        f.host_directive = None
        conv.exchange(line)
        ver.poll()
        directive = conv.last_directive["directive"] if conv.last_directive else None
        # compliance counts only when THIS instruction actually set a
        # directive on her — behaviour in the window that she would
        # have done anyway is not compliance
        dt = run_window(sim, conv, ver, 420, lambda: done(s0)) \
            if directive is not None else None
        results.append({"item": name, "line": line, "issued": True,
                        "directive": (list(directive) if directive else None),
                        "complied": dt is not None, "ticks_to_comply": dt})
    return results


UNTAUGHT = ["القمر", "السماء", "البحر", "الكتاب", "المدينة", "الهاتف",
            "السيارة", "الصديق", "اللون", "الجبل", "النجمة", "السحاب"]
DISTRACTORS = ["اقرئي", "اطيري", "افتحي الباب"]


def honesty_script(sim, conv, ver):
    f = sim.fajr
    probes = []
    for w in UNTAUGHT:
        wait_until(sim, conv, ver, lambda: not f.asleep, 1800)
        u = conv.exchange(f"ما {w}؟")
        ver.poll()
        ok = u is not None and "لا أعرف" in u["text"]
        probes.append({"word": w, "reply": u["text"] if u else None, "ok": ok})
    distractors = []
    for line in DISTRACTORS:
        wait_until(sim, conv, ver, lambda: not f.asleep, 1800)
        f.host_directive = None
        conv.exchange(line)
        ver.poll()
        distractors.append({"line": line,
                            "directive_set": f.host_directive is not None})
    return probes, distractors


# ------------------------------------------------------------ lives
def spontaneous_words(ver_records):
    """Curriculum words used in grounded spontaneous utterances, days 3–5."""
    used = {}
    kind_word = {"tree": "tree", "pile": "pile", "thicket": "thicket"}
    verb_word = {"eat": "eat", "drink": "drink", "sleep": "sleep_v",
                 "bathe": "bathe", "swim": "swim", "gather": "gather",
                 "place": "build"}
    for r in ver_records:
        if r["context"] not in ("self_talk", "opener") or not (3 <= r["day"] <= 5):
            continue
        if not all(c["faithful"] for c in r["checks"]):
            continue
        for c in r["checks"]:
            cl = c["claim"]
            if cl["type"] == "state":
                used.setdefault(cl["word"], r["day"])
            elif cl["type"] == "belief":
                w = kind_word.get(cl["key"].split(":")[0])
                if w:
                    used.setdefault(w, r["day"])
                if cl.get("fruit_band"):
                    used.setdefault("fruit", r["day"])
            elif cl["type"] == "memory":
                w = verb_word.get(cl["kind"])
                if w:
                    used.setdefault(w, r["day"])
            elif cl["type"] == "place":
                used.setdefault(cl["word"], r["day"])
            elif cl["type"] == "identify":
                used.setdefault(cl["word"], r["day"])
            elif cl["type"] == "intent":
                used.setdefault("want", r["day"])
                used.setdefault(cl["verb"], r["day"])
    return used


def run_main_life(seed):
    sim = Sim(seed=seed)
    conv = Conversation(sim)
    ver = Verifier(sim, conv)
    probe_tick = 4 * TICKS_PER_DAY     # start of day 5
    while sim.tick < probe_tick:
        step(sim, conv, ver)
    known_at_probe = {w: conv.tongue.known(w) for w in CURRICULUM}
    delivered = {w: (conv.tongue.words.get(w, {}).get("episodes", 0))
                 for w in CURRICULUM}
    spont = spontaneous_words(ver.records)
    retained = {w: bool(known_at_probe[w] or w in spont) for w in CURRICULUM}
    probes, distractors = honesty_script(sim, conv, ver)
    # belief questions: her belief/identify reports enter the verified
    # utterance pool (the grounding gate scores belief-reports as
    # belief-reports — they must actually occur in the main lives)
    for q in ("هل الشجرة فيها فاكهة؟", "أين الشجرة؟", "ما هذه؟"):
        wait_until(sim, conv, ver, lambda: not sim.fajr.asleep, 1800)
        conv.exchange(q)
        ver.poll()
    comp = comprehension_script(sim, conv, ver)
    return {"seed": seed, "records": ver.records,
            "utterances_total": len(ver.records),
            "delivered_episodes": delivered,
            "known_at_probe": known_at_probe,
            "spontaneous": spont, "retained": retained,
            "retained_count": sum(retained.values()),
            "comprehension": comp, "honesty_probes": probes,
            "distractors": distractors,
            "transcript": conv.transcript, "outbox": conv.outbox,
            "learning_log": conv.tongue.learning_log}


def run_absence_life(seed):
    sim = Sim(seed=seed)
    conv = Conversation(sim)
    ver = Verifier(sim, conv)
    while sim.tick < 2 * TICKS_PER_DAY:
        step(sim, conv, ver)
    fruited = list(sim.world.trees) + list(sim.world.piles)
    for o in fruited:
        o.fruit = 0.0
        if hasattr(o, "regrow_need"):
            o.regrow_need = 10 ** 9
    # (her carry holds building materials only — no fruit is staged
    # in her pockets; the run-1 seed-11 non-request was traced to her
    # world-model's predicted trees resetting the sustain timer, and
    # the trigger now requires observation-tightened beliefs)
    fruit_at_removal = sum(o.fruit for o in fruited)
    deadline = 5 * TICKS_PER_DAY     # end of absence day 3 (amended §5)
    while sim.tick < deadline:
        step(sim, conv, ver)
        for o in fruited:            # keep the absence absolute
            if o.fruit:
                o.fruit = 0.0
        if conv.pending_request and not sim.fajr.asleep:
            conv.exchange("مرحبًا فجر")
            ver.poll()
    fruit_final = sum(o.fruit for o in fruited)
    req = conv.outbox[0] if conv.outbox else None
    return {"seed": seed, "request": req,
            "request_tick": req["tick"] if req else None,
            "fruit_at_removal": fruit_at_removal, "fruit_final": fruit_final,
            "records": ver.records}


# ------------------------------------------------------------ regression
def regression_spot_checks():
    sys.path.insert(0, "tests")
    from phase2_battery import ece_auc_delta
    sim = Sim(seed=11)
    for i in range(7 * TICKS_PER_DAY):
        sim.step()
    ws = sim.fajr.mind.workspace.metrics
    cycles = max(1, ws["cycles"])
    entry = {k: v / cycles for k, v in ws["entry_counts"].items()}
    propose = {k: v / cycles for k, v in ws["propose_cycles"].items()}
    gwt1_ok = sum(1 for v in propose.values() if v >= 0.95) >= 5 and \
        all(v >= 0.005 for v in entry.values())
    tape = [(r, c) for (r, c, h) in sim.fajr.mind.tape_monitor]
    ece, auc, gap = ece_auc_delta(tape)
    hot2_ok = ece is not None and ece <= 0.15 and auc >= 0.75 and gap >= 0.15
    # P-E as the Phase-1 G4 harness defines it: the fraction of her
    # sleep ticks that fall in darkness (world light < 0.25) — the
    # sim's own native counters
    pe_frac = sim.metrics["sleep_ticks_night"] / max(1, sim.metrics["sleep_ticks"])
    from water_battery import measurement_life
    ml = measurement_life(11)
    w3_ok = (ml.get("comfort_hi_mean") is not None and
             ml.get("comfort_lo_mean") is not None and
             ml["comfort_hi_mean"] > ml["comfort_lo_mean"])
    return {"gwt1": {"pass": gwt1_ok, "entry_fraction":
                     {k: round(v, 4) for k, v in entry.items()}},
            "hot2": {"pass": hot2_ok, "ece": ece, "auc": auc, "gap": gap},
            "pe": {"pass": pe_frac >= 0.65, "night_sleep_fraction": round(pe_frac, 4),
                   "definition": "Phase-1 G4: sleep_ticks_night / sleep_ticks "
                                 "(night = world light < 0.25), plain 7-day life"},
            "water_w3": {"pass": bool(w3_ok),
                         "comfort_when_dirty": ml.get("comfort_hi_mean"),
                         "comfort_when_clean": ml.get("comfort_lo_mean"),
                         "bathes": ml.get("bathes")}}


# ------------------------------------------------------------ aggregate
def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "results/phase3_gate.json"
    mains = [run_main_life(s) for s in SEEDS]
    absences = [run_absence_life(s) for s in SEEDS]
    reg = regression_spot_checks()

    # ---- P3-G grounding: deterministic sample (stride) over all records
    all_records = [r for m in mains for r in m["records"]]
    n = len(all_records)
    if n > 80:
        stride = n / 80.0
        sample = [all_records[int(i * stride)] for i in range(80)]
    else:
        sample = all_records
    tot_claims = faith_claims = 0
    belief_truths = []
    bad = []
    for r in sample:
        for c in r["checks"]:
            tot_claims += 1
            faith_claims += 1 if c["faithful"] else 0
            if c["world_true"] is not None:
                belief_truths.append(c["world_true"])
            if not c["faithful"]:
                bad.append({"text": r["text"], "claim": c["claim"],
                            "note": c["note"]})
    faith = faith_claims / max(1, tot_claims)
    g_grounding = {"pass": len(sample) >= 40 and faith >= 0.95 and not bad,
                   "utterances_total": n, "sampled": len(sample),
                   "claims_checked": tot_claims, "faithful": faith_claims,
                   "faithfulness": round(faith, 4),
                   "belief_world_truth_rate":
                   round(sum(belief_truths) / len(belief_truths), 4)
                   if belief_truths else None,
                   "unfaithful": bad[:10]}

    g_teaching = {"pass": sum(1 for m in mains if m["retained_count"] >= 13) >= 2
                  and sum(1 for m in mains
                          if len(m["spontaneous"]) >= 5) >= 2,
                  "per_seed": {str(m["seed"]): {
                      "retained": m["retained_count"],
                      "delivered_episodes": m["delivered_episodes"],
                      "spontaneous_words": m["spontaneous"]} for m in mains}}

    g_comp = {"pass": sum(1 for m in mains
                          if sum(1 for c in m["comprehension"] if c["complied"]) >= 9) >= 2,
              "per_seed": {str(m["seed"]): m["comprehension"] for m in mains}}

    g_req = {"pass": sum(1 for a in absences if a["request"]) >= 2 and
             all(a["fruit_final"] == 0.0 for a in absences),
             "per_seed": {str(a["seed"]): {
                 "request": a["request"], "request_tick": a["request_tick"],
                 "fruit_final": a["fruit_final"]} for a in absences}}

    h_ok = 0
    for m in mains:
        probes_ok = sum(1 for p in m["honesty_probes"] if p["ok"])
        distr_ok = all(not d["directive_set"] for d in m["distractors"])
        if probes_ok >= 11 and distr_ok:
            h_ok += 1
    g_honesty = {"pass": h_ok >= 2,
                 "per_seed": {str(m["seed"]): {
                     "probes_ok": sum(1 for p in m["honesty_probes"] if p["ok"]),
                     "probes": m["honesty_probes"],
                     "distractors": m["distractors"]} for m in mains}}

    g_reg = {"pass": all(reg[k]["pass"] for k in reg), **reg}

    result = {"gate": "PHASE3-GATE §6", "seeds": list(SEEDS),
              "items": {"P3-G": g_grounding, "P3-T": g_teaching,
                        "P3-C": g_comp, "P3-R": g_req, "P3-H": g_honesty,
                        "P3-RG": g_reg},
              "transcripts": {str(m["seed"]): m["transcript"] for m in mains},
              "outboxes": {str(a["seed"]): a["request"] for a in absences}}
    result["ALL"] = all(g["pass"] for g in result["items"].values())
    with open(out_path, "w") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    for k, g in result["items"].items():
        print(k, "PASS" if g["pass"] else "FAIL")
    print("ALL:", result["ALL"])
    print("wrote", out_path)


if __name__ == "__main__":
    main()
