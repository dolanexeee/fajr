"""Continuous coarticulated synthesis v2 — precomputed area trajectory.

One unbroken waveguide run for a whole word. Articulators glide smoothly;
closures form/release within the ongoing airflow. No gaps, no clips.
"""
import numpy as np
from scipy.signal import resample_poly

from .tract import VocalTract, _areas_from_articulators, SR_TRACT
from .rosenberg import rosenberg_train, SR
from .syllables import HUMAN_ART, HUMAN_N, CLOSURE_POS

CHUNK_MS = 5


def _vowel_areas(v):
    return _areas_from_articulators(*HUMAN_ART[v], n=HUMAN_N)


def _closure_areas(base, pos, degree):
    a = base.copy()
    idx = int(np.clip(pos, 0, 1) * (HUMAN_N - 1))
    for j in (idx - 1, idx, idx + 1):
        if 0 <= j < HUMAN_N:
            a[j] = a[j] * (1.0 - degree) + 1e-6 * degree
    return np.clip(a, 1e-6, 12.0)


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return 3 * x ** 2 - 2 * x ** 3


def build_trajectory(timeline):
    """timeline: [(kind, param, dur_ms)] -> (n_chunks, N) area array.

    kind 'v': vowel; kind 'c': closure (shaped close/hold/release).
    Transitions between elements are smooth (25ms raised-cosine blends),
    closures ramp shut/open within their own span.
    """
    # chunk spans per element
    spans = []
    for kind, param, dur in timeline:
        spans.append(max(1, int(round(dur / CHUNK_MS))))
    total = sum(spans)
    traj = np.zeros((total, HUMAN_N))

    # anchor area function per element (midpoint target)
    anchors = []
    for (kind, param, dur), n_c in zip(timeline, spans):
        if kind == 'v':
            anchors.append(_vowel_areas(param))
        else:
            # closure: fully shut at midpoint
            base = _vowel_areas('a')
            anchors.append(_closure_areas(base, CLOSURE_POS[param], 1.0))

    # fill with smooth blends between anchors + shaped closures
    pos = 0
    for ei, ((kind, param, dur), n_c) in enumerate(zip(timeline, spans)):
        a0 = anchors[ei - 1] if ei > 0 else anchors[ei]
        a1 = anchors[ei]
        a2 = anchors[ei + 1] if ei < len(anchors) - 1 else anchors[ei]
        for k in range(n_c):
            if kind == 'c':
                # shaped closure within span: close 30ms / hold / release 15ms
                t_ms = k * CHUNK_MS
                close_ms, rel_ms = 30, 15
                if t_ms < close_ms:
                    deg = _smoothstep(t_ms / close_ms)
                elif t_ms < dur - rel_ms:
                    deg = 1.0
                else:
                    deg = 1.0 - _smoothstep((t_ms - (dur - rel_ms)) / rel_ms)
                base = _vowel_areas('a')
                # blend base toward neighbors at edges for smooth entry/exit
                traj[pos + k] = _closure_areas(base, CLOSURE_POS[param], deg)
            else:
                # vowel: smooth 25ms blend in from previous, out to next
                b = 25 / CHUNK_MS
                w_in = _smoothstep(k / b) if k < b else 1.0
                w_out = _smoothstep((n_c - 1 - k) / b) if (n_c - 1 - k) < b else 1.0
                w = min(w_in, w_out)
                # blend: (1-w)*neighbor + w*own, neighbor depends on side
                if k < n_c / 2:
                    traj[pos + k] = (1 - w) * a0 + w * a1
                else:
                    traj[pos + k] = (1 - w) * a2 + w * a1
        pos += n_c
    return traj


def synthesize_continuous(timeline, f0s=190, f0e=170):
    """One continuous waveguide run. Returns 16kHz waveform."""
    traj = build_trajectory(timeline)
    n_chunks = traj.shape[0]
    total_s = n_chunks * CHUNK_MS / 1000.0

    flow = rosenberg_train(f0s, f0e, total_s + 0.05)[:int(total_s * SR)] * 1e6
    ug48 = resample_poly(flow, 3, 1)
    chunk48 = int(CHUNK_MS / 1000 * SR_TRACT)

    tract = VocalTract(n_sec=HUMAN_N)
    tract.reset()
    out_parts = []
    for ci in range(n_chunks):
        tract._apply_areas(traj[ci])
        seg = ug48[ci * chunk48:(ci + 1) * chunk48]
        if len(seg) < chunk48:
            seg = np.pad(seg, (0, chunk48 - len(seg)))
        out_parts.append(tract._run48(seg))
        if traj[ci].min() < 1e-4:  # fully sealed: gentle wall loss
            tract.up *= 0.97
            tract.dn *= 0.97

    out48 = np.concatenate(out_parts)[:len(ug48)]
    out16 = resample_poly(out48, 1, 3).astype(float)
    mx = np.abs(out16).max()
    return out16 / (mx + 1e-12)


def word_baba():
    return synthesize_continuous([
        ('c', 'b', 85), ('v', 'a', 170),
        ('c', 'b', 85), ('v', 'a', 260),
    ], f0s=192, f0e=168)
