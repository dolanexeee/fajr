#!/usr/bin/env bash
# Fajr — one-command reproduction of the 5 key results.
#
# Usage: ./reproduce.sh   (run from the repo root)
#
# What it does:
#   0. checks Python 3 + numpy/scipy/matplotlib (installs them if pip can)
#   1. no-LLM audit — the mechanistic claim breaks if this prints anything
#   2. language grounding — 19 gates
#   3. honest interaction — 15 probes
#   4. voice — 5 guttural phoneme gates
#   5. voice — vowel/word/sentence battery
#   6. voice — speakability audit (graceful without the private journal)
#
# Every gate is deterministic (seeded PRNG). No network needed after
# the optional pip install. See REPRODUCE.md for what each step proves.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

pass() { echo "  -> STEP PASS: $1"; }
fail() { echo "  -> STEP FAIL: $1"; echo "See REPRODUCE.md 'If anything fails'."; exit 1; }
need_marker() { # need_marker <file> <marker> <step-name>
    if grep -qF "$2" "$1"; then pass "$3"; else fail "$3 (missing marker: $2)"; fi
}

echo "=== 0. environment ==="
command -v python3 >/dev/null || fail "python3 not found"
python3 -c "import sys; assert sys.version_info[0] == 3" || fail "need Python 3"
if ! python3 -c "import numpy, scipy, matplotlib" 2>/dev/null; then
    echo "installing numpy/scipy/matplotlib via pip..."
    python3 -m pip install -q numpy scipy matplotlib \
        || fail "pip install failed — install numpy, scipy, matplotlib manually"
fi
python3 -c "import numpy, scipy, matplotlib; print('deps ok')"
pass "environment"

cd "$ROOT/src"

echo "=== 1. no-LLM audit (must print nothing) ==="
AUDIT_OUT="$(grep -rniE "openai|anthropic|cohere|huggingface|transformers|torch|tensorflow|onnx|gguf" \
    --include="*.py" . || true)"
[ -z "$AUDIT_OUT" ] && pass "no-LLM audit" || fail "no-LLM audit — found: $AUDIT_OUT"

echo "=== 2. language grounding — 19 gates ==="
python3 test_grounding.py > /tmp/fajr_repro_grounding.log 2>&1
need_marker /tmp/fajr_repro_grounding.log "ALL GROUNDING GATES PASS" "grounding (19 gates)"

echo "=== 3. honest interaction — 15 probes ==="
python3 test_honesty.py > /tmp/fajr_repro_honesty.log 2>&1
need_marker /tmp/fajr_repro_honesty.log "ALL HONESTY GATES PASS" "honesty (15 probes)"

echo "=== 4. voice — 5 guttural phoneme gates ==="
python3 -m voice.guttural > /tmp/fajr_repro_guttural.log 2>&1
GUTTURAL_PASS="$(grep -c " PASS " /tmp/fajr_repro_guttural.log || true)"
[ "$GUTTURAL_PASS" -ge 5 ] && pass "gutturals ($GUTTURAL_PASS PASS lines)" \
    || fail "gutturals (only $GUTTURAL_PASS PASS lines, need >= 5)"

echo "=== 5. voice — vowel/word/sentence battery ==="
python3 -m voice.build_voice > /tmp/fajr_repro_voice.log 2>&1
need_marker /tmp/fajr_repro_voice.log "ALL DONE. Gates: s2=True s3=True s4=True" \
    "voice battery (s2/s3/s4)"

echo "=== 6. voice — speakability audit ==="
python3 -m voice.live_bridge > /tmp/fajr_repro_bridge.log 2>&1
pass "speakability audit (exit 0 — graceful with or without the private journal)"

echo
echo "ALL REPRODUCTION STEPS PASS — 6/6"
echo "Logs: /tmp/fajr_repro_*.log"
