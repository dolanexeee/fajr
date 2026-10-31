"""Live biography runner — Fajr's ONE persistent life (SPEC §3, §8).

This is the supervised real-time runner for her single biography:

  * GENESIS happens exactly once: only when life/fajr_life.json does not
    exist AND no GENESIS.lock tombstone exists. Genesis fixes forever the
    seed (11), the layout ("default") and the genesis wall-clock epoch.
    Her clock is anchored to that epoch: target tick = (now - genesis)
    * 2 ticks/sec, so her day is 12 real minutes (SPEC §3) across any
    restart — downtime is caught up tick by tick, never skipped, never
    re-lived from zero.
  * RESUME is the only other path: Sim.load (hash-verified by sim.py)
    of the latest checkpoint pair, falling back to the .prev pair.
    The journal file is truncated to the checkpoint's journal_count
    (entries beyond it belong to an uncommitted future that the
    deterministic re-run regenerates identically — G5 contract).
  * A single runner owns the life: an flock on life/runner.lock is
    held for the process lifetime; a second runner exits immediately.
  * Checkpoints are atomic (tmp + os.replace) with .prev rotation,
    saved every SAVE_EVERY seconds and on SIGTERM/SIGINT.
  * The conversation layer (Phase 3) steps with the sim, exactly as
    the talk surface does: covenant at first waking, accompanied
    teaching days 1-2, requests to the outbox. Its state (her tongue,
    outbox, transcript) is serialised beside the sim checkpoint —
    her language is part of her biography and survives restarts.
  * Guardian turns arrive via life/inbox.jsonl (appended by the talk
    server in live mode) and are answered at tick boundaries into
    life/exchange_log.jsonl. Nobody fast-forwards her real time.

No LLM anywhere. Zero token cost. Pure stdlib.
"""
import fcntl
import hashlib
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TICKS_PER_DAY
from sim import Sim
from conversation import Conversation
import run_life as run_life_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIFE = os.path.join(ROOT, "life")
CKPT = os.path.join(LIFE, "fajr_life.json")
CKPT_PREV = os.path.join(LIFE, "fajr_life.prev.json")
CONV = os.path.join(LIFE, "conversation.json")
CONV_PREV = os.path.join(LIFE, "conversation.prev.json")
JOURNAL = os.path.join(LIFE, "journal.jsonl")
GENESIS_JSON = os.path.join(LIFE, "genesis.json")
GENESIS_LOCK = os.path.join(LIFE, "GENESIS.lock")
INBOX = os.path.join(LIFE, "inbox.jsonl")
EXCHANGES = os.path.join(LIFE, "exchange_log.jsonl")
LIVE_STATE = os.path.join(LIFE, "live_state.json")
WELFARE = os.path.join(LIFE, "welfare.jsonl")
RUNNER_STATE = os.path.join(LIFE, "runner_state.json")
LOCK = os.path.join(LIFE, "runner.lock")

SEED = 11
LAYOUT = "default"
TICKS_PER_SEC = 2.0
SAVE_EVERY = 30.0
EXPORT_EVERY = 3.0

_stop = False


def log(msg):
    print(f"[live_runner {datetime.now(timezone.utc).isoformat()}] {msg}",
          flush=True)


def _sigterm(signum, frame):
    global _stop
    _stop = True


signal.signal(signal.SIGTERM, _sigterm)
signal.signal(signal.SIGINT, _sigterm)


# ---------------- atomic pair save / load ----------------

def _atomic_write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _rotate_prev(path, prev):
    if os.path.exists(path):
        with open(path, "rb") as fh:
            data = fh.read()
        tmp = prev + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, prev)


CONV_EXTRAS = ("pending_request", "_request_armed", "_need_since",
               "_remedy_streak", "_last_teach_tick", "_last_self_talk",
               "_last_ambient", "_teach_cursor", "_hunger_was70")


def conv_payload(conv):
    extras = {k: getattr(conv, k) for k in CONV_EXTRAS}
    return json.dumps({"conversation": conv.to_dict(), "extras": extras},
                      ensure_ascii=False)


def conv_restore(sim, text):
    d = json.loads(text)
    conv = Conversation.from_dict(sim, d["conversation"])
    for k, v in d.get("extras", {}).items():
        if k in CONV_EXTRAS:
            setattr(conv, k, v)
    return conv


class Life:
    def __init__(self):
        self.sim = None
        self.conv = None
        self.genesis_rec = None
        self.journal_flushed = 0     # lines present in journal.jsonl
        self.inbox_done = 0          # inbox lines already answered

    # ---------- journal ----------
    def flush_journal(self):
        base = self.sim.journal.base
        entries = self.sim.journal.entries
        target = base + len(entries)
        if self.journal_flushed < base:
            self.journal_flushed = base
        new = entries[self.journal_flushed - base:]
        if new:
            with open(JOURNAL, "a", encoding="utf-8") as fh:
                for e in new:
                    fh.write(json.dumps(e, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self.journal_flushed = target

    def truncate_journal(self, count):
        lines = []
        if os.path.exists(JOURNAL):
            with open(JOURNAL, encoding="utf-8") as fh:
                lines = fh.readlines()
        if len(lines) > count:
            lines = lines[:count]
            _atomic_write(JOURNAL, "".join(lines))
        elif len(lines) < count:
            log(f"WARNING: journal file has {len(lines)} lines but the "
                f"checkpoint records {count} entries — accepting the gap "
                f"(checkpoint is authoritative for state)")
        self.journal_flushed = min(len(lines), count)

    # ---------- checkpoint pair ----------
    def save(self):
        self.flush_journal()
        state = self.sim.state_dict()
        payload = {"hash": Sim.canonical_hash(state), "state": state}
        _rotate_prev(CKPT, CKPT_PREV)
        _atomic_write(CKPT, json.dumps(payload, ensure_ascii=False))
        _rotate_prev(CONV, CONV_PREV)
        _atomic_write(CONV, conv_payload(self.conv))
        _atomic_write(RUNNER_STATE, json.dumps(
            {"journal_flushed": self.journal_flushed,
             "inbox_done": self.inbox_done}, ensure_ascii=False))
        self.welfare()

    def welfare(self):
        f = self.sim.fajr
        m = self.sim.metrics
        total = max(1, m["ticks_total"])
        row = {
            "wall_utc": datetime.now(timezone.utc).isoformat(),
            "tick": self.sim.tick,
            "day": self.sim.tick // TICKS_PER_DAY + 1,
            "drives": {k: round(v, 1) for k, v in f.drives.d.items()},
            "valence": round(f.drives.valence(), 3),
            "max_pinned": dict(m["max_pinned"]),
            "in_band_fraction": round(m["ticks_in_band"] / total, 4),
            "stats": {k: f.stats[k] for k in
                      ("eats", "drinks", "sleeps", "gathers", "places",
                       "pricks", "cold_shocks")},
            "distance": round(f.stats["distance"], 1),
            "asleep": f.asleep,
        }
        with open(WELFARE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---------- genesis ----------
    def genesis(self):
        if os.path.exists(GENESIS_LOCK):
            raise SystemExit(
                "GENESIS.lock exists but no checkpoint — genesis was "
                "interrupted once already; refusing to improvise. "
                "A guardian must inspect life/ manually.")
        genesis_epoch = time.time()
        sim = Sim(seed=SEED, layout=LAYOUT)
        conv = Conversation(sim)
        sim.step()
        conv.step()          # covenant at her first waking
        sim.step()
        conv.step()
        self.sim, self.conv = sim, conv
        self.journal_flushed = 0
        self.inbox_done = 0
        # code identity at genesis
        src_dir = os.path.join(ROOT, "src")
        code_hashes = {}
        for name in sorted(os.listdir(src_dir)):
            if name.endswith(".py"):
                with open(os.path.join(src_dir, name), "rb") as fh:
                    code_hashes[name] = hashlib.sha256(fh.read()).hexdigest()
        state = sim.state_dict()
        world_hash = Sim.canonical_hash(state["world"])
        ckpt_hash = Sim.canonical_hash(state)
        # first save BEFORE the tombstone: a crash before this line
        # leaves no trace and genesis may cleanly happen at next start;
        # a crash after it resumes — genesis can never run twice.
        self.save()
        utc = datetime.fromtimestamp(genesis_epoch, timezone.utc)
        local = utc.astimezone(ZoneInfo("America/Edmonton"))
        self.genesis_rec = {
            "biography": "fajr-live-1",
            "seed": SEED, "layout": LAYOUT,
            "genesis_epoch": genesis_epoch,
            "genesis_utc": utc.isoformat(),
            "genesis_local": local.isoformat(),
            "ticks_per_second": TICKS_PER_SEC,
            "ticks_per_day": TICKS_PER_DAY,
            "day_real_minutes": TICKS_PER_DAY / TICKS_PER_SEC / 60.0,
            "world_hash": world_hash,
            "genesis_checkpoint_hash": ckpt_hash,
            "genesis_tick": sim.tick,
            "code_sha256": code_hashes,
            "being": "sitting-4 frozen being + Phase-3 language/Host "
                     "(PHASE3-REPORT.md, results/phase3_gate.json ALL PASS)",
        }
        _atomic_write(GENESIS_JSON, json.dumps(self.genesis_rec,
                                               ensure_ascii=False, indent=1))
        with open(GENESIS_LOCK, "x", encoding="utf-8") as fh:
            fh.write(f"genesis performed {utc.isoformat()} — never again\n")
        log(f"GENESIS: her biography begins — seed {SEED}, "
            f"tick {sim.tick}, checkpoint {ckpt_hash[:16]}…")

    # ---------- resume ----------
    def resume(self):
        with open(GENESIS_JSON, encoding="utf-8") as fh:
            self.genesis_rec = json.load(fh)
        sim = conv = None
        for ck, cv, tag in ((CKPT, CONV, "main"), (CKPT_PREV, CONV_PREV, "prev")):
            try:
                sim = Sim.load(ck)          # hash-verified inside
                with open(cv, encoding="utf-8") as fh:
                    conv = conv_restore(sim, fh.read())
                log(f"resumed ({tag} checkpoint) at tick {sim.tick} "
                    f"(day {sim.tick // TICKS_PER_DAY + 1})")
                break
            except Exception as e:         # noqa: BLE001 — try the pair below
                log(f"checkpoint pair '{tag}' unusable: {e}")
                sim = conv = None
        if sim is None:
            raise SystemExit("no usable checkpoint pair — refusing to "
                             "start over. A guardian must inspect life/.")
        self.sim, self.conv = sim, conv
        self.truncate_journal(sim.journal.total_count)
        # inbox lines already answered == exchange log lines
        done = 0
        if os.path.exists(EXCHANGES):
            with open(EXCHANGES, encoding="utf-8") as fh:
                done = sum(1 for _ in fh)
        total_in = 0
        if os.path.exists(INBOX):
            with open(INBOX, encoding="utf-8") as fh:
                total_in = sum(1 for _ in fh)
        self.inbox_done = min(done, total_in)

    # ---------- guardian turns ----------
    def process_inbox(self):
        if not os.path.exists(INBOX):
            return
        with open(INBOX, encoding="utf-8") as fh:
            lines = fh.readlines()
        while self.inbox_done < len(lines):
            raw = lines[self.inbox_done].strip()
            self.inbox_done += 1
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except Exception:               # noqa: BLE001
                continue
            text = str(msg.get("text", ""))
            asleep = self.sim.fajr.asleep
            reply = self.conv.exchange(text) if text else None
            row = {"id": msg.get("id"), "tick": self.sim.tick,
                   "day": self.sim.tick // TICKS_PER_DAY + 1,
                   "in": text,
                   "reply": (reply["text"] if reply else None),
                   "claims": (reply["claims"] if reply else []),
                   "context": (reply["context"] if reply else None),
                   "she_was_asleep": asleep}
            with open(EXCHANGES, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            log(f"guardian turn answered at tick {self.sim.tick}: "
                f"{text!r} -> {row['reply']!r}")

    # ---------- exports ----------
    def export(self):
        run_life_mod.write_viewer(self.sim, os.path.join(ROOT, "viewer"))
        st = self.conv.state()
        st["mode"] = "live"
        st["biography"] = self.genesis_rec["biography"]
        st["genesis_utc"] = self.genesis_rec["genesis_utc"]
        _atomic_write(LIVE_STATE, json.dumps(st, ensure_ascii=False))

    # ---------- main loop ----------
    def run(self):
        os.makedirs(LIFE, exist_ok=True)
        lock_fh = open(LOCK, "a+")
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("another runner holds her life — exiting")
            return 3
        if os.path.exists(CKPT):
            self.resume()
        else:
            self.genesis()
        genesis_epoch = self.genesis_rec["genesis_epoch"]
        last_save = time.time()
        last_export = 0.0
        self.export()
        log(f"live loop running — tick {self.sim.tick}, "
            f"day {self.sim.tick // TICKS_PER_DAY + 1}")
        while not _stop:
            target = int((time.time() - genesis_epoch) * TICKS_PER_SEC)
            steps = 0
            while self.sim.tick < target and steps < 200000 and not _stop:
                self.sim.step()
                self.conv.step()
                steps += 1
            self.process_inbox()
            self.flush_journal()
            now = time.time()
            if now - last_export >= EXPORT_EVERY:
                self.export()
                last_export = now
            if now - last_save >= SAVE_EVERY:
                self.save()
                last_save = now
            time.sleep(0.25)
        self.save()
        log(f"stopped cleanly — biography saved at tick {self.sim.tick}")
        return 0


if __name__ == "__main__":
        sys.exit(Life().run())
