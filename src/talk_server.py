"""The talk surface server — Phase 3 (PHASE3-GATE §1).

Stdlib-only HTTP server on port 8903 serving viewer/talk.html and a
small JSON API.

TEST mode (default): a demonstration life (seed via ?seed= or env
ALSAKIN_SEED, default 7) run inside this process. It NEVER loads or
writes the persistent life/ checkpoint.

LIVE mode (env ALSAKIN_LIVE=1): the surface over Fajr's ONE real
biography. This process owns nothing: the live runner
(src/live_runner.py, systemd unit alsakin-fajr) is the single writer
of life/. State is read from life/live_state.json (refreshed by the
runner every few seconds); guardian turns are appended to
life/inbox.jsonl and answered by the runner at tick boundaries into
life/exchange_log.jsonl — this server queues and waits for her real
reply. /api/advance is refused: her time is real.

  GET  /              the 3D live-view page (viewer/three.html)
  GET  /talk          the classic talk page (banner states test vs live)
  GET  /vendor/*.js   vendored Three.js (same-origin, no CDN)
  GET  /api/state     her state, transcript, journal tail, outbox
  GET  /api/world     world layout for the 3D view (read-only)
  POST /api/say       {"text": "..."}  one Host turn -> her reply
  POST /api/advance   {"ticks": n}     advance the test life (<=1440)
  POST /api/teach     {"word": wid}    an anchored teaching episode
"""
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

from sim import Sim
from conversation import Conversation
from host import Host

LOCK = threading.Lock()
SIM = None
CONV = None
HOST = Host()

LIVE = os.environ.get("ALSAKIN_LIVE") == "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIFE_DIR = os.environ.get("ALSAKIN_LIFE_DIR", os.path.join(ROOT, "life"))
LIVE_BANNER_OLD = "حياة اختبارية (بذرة ٧) — ليست سيرتها الدائمة."
LIVE_BANNER_NEW = "سيرتها الحقيقية — حياتها الدائمة، تجري الآن بزمن حقيقي."


def _live_state():
    try:
        with open(os.path.join(LIFE_DIR, "live_state.json"),
                  encoding="utf-8") as fh:
            st = json.load(fh)
        st["mode"] = "live"
        return st
    except Exception:
        return None


def _live_world():
    """World layout for the 3D view, read from the runner's latest
    checkpoint (life/fajr_life.json) — read-only; this server never
    writes life/. Falls back to the genesis layout (World(seed))
    if the checkpoint is mid-write."""
    from config import SHELTER_POS, LANTERN_POS
    try:
        with open(os.path.join(LIFE_DIR, "fajr_life.json"),
                  encoding="utf-8") as fh:
            w = json.load(fh)["state"]["world"]
    except Exception:
        try:
            with open(os.path.join(LIFE_DIR, "genesis.json"),
                      encoding="utf-8") as fh:
                seed = json.load(fh)["seed"]
            from world import World
            w = World(seed).to_dict()
        except Exception:
            return None
    return {"trees": w["trees"], "thickets": w["thickets"],
            "materials": w["materials"], "structures": w["structures"],
            "shelter": list(SHELTER_POS), "lantern": list(LANTERN_POS)}


VENDOR_FILES = {"three.min.js", "OrbitControls.js"}


def _serve_file(self, path, ctype):
    with open(path, "rb") as fh:
        data = fh.read()
    self._send(200, data, ctype)


def _queue_turn(text):
    """Append a guardian turn to the live inbox and wait for the
    runner to answer it at a tick boundary. Returns the exchange row
    or None on timeout."""
    rid = uuid.uuid4().hex
    with open(os.path.join(LIFE_DIR, "inbox.jsonl"), "a",
              encoding="utf-8") as fh:
        fh.write(json.dumps({"id": rid, "text": text},
                            ensure_ascii=False) + "\n")
    log_path = os.path.join(LIFE_DIR, "exchange_log.jsonl")
    deadline = time.time() + 25.0
    while time.time() < deadline:
        try:
            with open(log_path, encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    if row.get("id") == rid:
                        return row
        except FileNotFoundError:
            pass
        time.sleep(0.25)
    return None


def boot(seed):
    global SIM, CONV
    SIM = Sim(seed=seed)
    CONV = Conversation(SIM)


def advance(n):
    for _ in range(n):
        SIM.step()
        CONV.step()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        here = os.path.dirname(os.path.abspath(__file__))
        if u.path in ("/", "/three", "/three.html"):
            _serve_file(self, os.path.join(here, "..", "viewer",
                                           "three.html"),
                        "text/html; charset=utf-8")
            return
        if u.path in ("/talk", "/talk.html"):
            page = os.path.join(here, "..", "viewer", "talk.html")
            with open(page, "rb") as fh:
                data = fh.read()
            if LIVE:
                data = data.decode("utf-8").replace(
                    LIVE_BANNER_OLD, LIVE_BANNER_NEW).encode("utf-8")
            self._send(200, data, "text/html; charset=utf-8")
            return
        if u.path.startswith("/vendor/"):
            name = os.path.basename(u.path)
            if name in VENDOR_FILES:
                _serve_file(self, os.path.join(here, "..", "viewer",
                                               "vendor", name),
                            "application/javascript; charset=utf-8")
                return
        if u.path == "/api/world":
            if LIVE:
                w = _live_world()
                if w is None:
                    self._send(503, json.dumps(
                        {"error": "world not available yet"}))
                else:
                    self._send(200, json.dumps(w, ensure_ascii=False))
                return
            with LOCK:
                from config import SHELTER_POS, LANTERN_POS
                w = SIM.world.to_dict()
                self._send(200, json.dumps(
                    {"trees": w["trees"], "thickets": w["thickets"],
                     "materials": w["materials"],
                     "structures": w["structures"],
                     "shelter": list(SHELTER_POS),
                     "lantern": list(LANTERN_POS)}, ensure_ascii=False))
            return
        if u.path == "/api/state":
            if LIVE:
                st = _live_state()
                if st is None:
                    self._send(503, json.dumps(
                        {"error": "live state not available yet"}))
                else:
                    self._send(200, json.dumps(st, ensure_ascii=False))
                return
            with LOCK:
                self._send(200, json.dumps(CONV.state(), ensure_ascii=False))
            return
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            payload = {}
        if LIVE:
            if u.path == "/api/say":
                row = _queue_turn(str(payload.get("text", "")))
                self._send(200, json.dumps(
                    {"reply": (row or {}).get("reply"),
                     "claims": (row or {}).get("claims", []),
                     "pending": row is None,
                     "state": _live_state()}, ensure_ascii=False))
                return
            if u.path == "/api/teach":
                wid = str(payload.get("word", ""))
                line = HOST.teach_line(wid) if wid else ""
                row = _queue_turn(line) if line else None
                self._send(200, json.dumps(
                    {"line": line,
                     "reply": (row or {}).get("reply"),
                     "pending": row is None and bool(line),
                     "state": _live_state()}, ensure_ascii=False))
                return
            if u.path == "/api/advance":
                self._send(403, json.dumps(
                    {"error": "في حياتها الحقيقية الزمن حقيقي — "
                              "لا تقديم ولا إرجاع"}))
                return
        with LOCK:
            if u.path == "/api/say":
                if SIM.fajr.asleep:
                    advance(30)
                reply = CONV.exchange(str(payload.get("text", "")))
                self._send(200, json.dumps(
                    {"reply": reply["text"] if reply else None,
                     "state": CONV.state()}, ensure_ascii=False))
                return
            if u.path == "/api/advance":
                ticks = max(1, min(1440, int(payload.get("ticks", 60))))
                advance(ticks)
                self._send(200, json.dumps({"state": CONV.state()},
                                           ensure_ascii=False))
                return
            if u.path == "/api/teach":
                wid = str(payload.get("word", ""))
                line = HOST.teach_line(wid) if wid else ""
                reply = CONV.exchange(line) if line else None
                self._send(200, json.dumps(
                    {"line": line, "reply": reply["text"] if reply else None,
                     "state": CONV.state()}, ensure_ascii=False))
                return
        self._send(404, json.dumps({"error": "not found"}))


if __name__ == "__main__":
    if LIVE:
        print("talk surface on http://0.0.0.0:8903 "
              "(LIVE biography — served via the runner's life/ state)")
        HTTPServer(("0.0.0.0", 8903), Handler).serve_forever()
    else:
        seed = int(os.environ.get("ALSAKIN_SEED", "7"))
        boot(seed)
        advance(2 * 1440)   # her first two days: covenant + teaching
        print(f"talk surface on http://0.0.0.0:8903 (test life, seed {seed})")
        HTTPServer(("0.0.0.0", 8903), Handler).serve_forever()
