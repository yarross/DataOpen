"""Turn a capture from the stand into numbers (docs/LATENCY.md, 'на железе'): pair each stimulus edge with the response edge that follows it.

Input is a CSV from a logic analyser / oscilloscope export / camera frame log with two kinds of rows:

    channel,t            channel is `stim` (the stimulus: the marker in the frame, the mouse emulator's report) or `resp` (the answer:
                         the pointer moved, the USB IN packet, the GPIO of a test point); t is a time in `unit`
    unit: s | ms | us    or  frames  (then `fps` converts; a camera has a resolution of one frame period: reported as a +- bound)

A response is paired with the LAST stimulus before it that has not been answered; stimuli with no answer inside `window_ms` are counted as
missed (a lost report, a help that never came), not silently dropped. The statistics are the point: a mean without p95/p99/max hides exactly
what a person with slow reactions feels."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable, Optional

from .measure import stats

_SCALE = {"s": 1000.0, "ms": 1.0, "us": 0.001}


def read_edges(path: str | Path, unit: str = "us", fps: Optional[float] = None) -> tuple[list[float], list[float]]:
    """(stimulus times, response times) in milliseconds, each sorted."""
    if unit == "frames":
        if not fps or fps <= 0:
            raise ValueError("unit 'frames' needs --fps")
        scale = 1000.0 / fps
    elif unit in _SCALE:
        scale = _SCALE[unit]
    else:
        raise ValueError(f"unit must be one of s, ms, us, frames, not {unit!r}")
    stim, resp = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), 2):
            try:
                ch, t = row["channel"].strip().lower(), float(row["t"]) * scale
            except (KeyError, ValueError, AttributeError) as e:
                raise ValueError(f"{path}:{i}: expected columns channel,t ({e})") from e
            if ch == "stim":
                stim.append(t)
            elif ch == "resp":
                resp.append(t)
            else:
                raise ValueError(f"{path}:{i}: channel must be 'stim' or 'resp', not {ch!r}")
    return sorted(stim), sorted(resp)


def pair_edges(stim: Iterable[float], resp: Iterable[float], window_ms: float = 200.0) -> tuple[list[float], int, int]:
    """(delays in ms, stimuli without an answer, responses without a stimulus)."""
    s, r = sorted(stim), sorted(resp)
    delays: list[float] = []
    i = j = missed = orphans = 0
    while i < len(s):
        # the responses before this stimulus belong to nobody
        while j < len(r) and r[j] < s[i]:
            orphans += 1
            j += 1
        nxt = s[i + 1] if i + 1 < len(s) else float("inf")
        if j < len(r) and r[j] - s[i] <= window_ms and r[j] < nxt:
            delays.append(r[j] - s[i])
            j += 1
        else:
            missed += 1
        i += 1
    orphans += len(r) - j
    return delays, missed, orphans


def summarize(delays: list[float], missed: int = 0, orphans: int = 0, resolution_ms: float = 0.0) -> dict:
    """`resolution_ms`: the capture's own resolution (a camera: one frame period). Every number is +- that."""
    out = {"pairs": len(delays), "missed": missed, "orphan_responses": orphans, "resolution_ms": resolution_ms}
    out |= stats(delays)
    hist: dict[str, int] = {}
    for d in delays:
        k = f"{int(d // 1)}-{int(d // 1) + 1} ms"
        hist[k] = hist.get(k, 0) + 1
    out["histogram_1ms"] = dict(sorted(hist.items(), key=lambda kv: int(kv[0].split("-")[0])))
    return out


def analyze_file(path: str | Path, unit: str = "us", fps: Optional[float] = None, window_ms: float = 200.0) -> dict:
    stim, resp = read_edges(path, unit, fps)
    d, missed, orphans = pair_edges(stim, resp, window_ms)
    return summarize(d, missed, orphans, 1000.0 / fps if unit == "frames" and fps else 0.0)


def synth_capture(path: str | Path, n: int = 200, mean_ms: float = 12.0, jitter_ms: float = 3.0, miss: float = 0.0, seed: int = 1,
                  period_ms: float = 400.0) -> None:
    """A capture file with a known answer (for the tests and for trying the tool): delay = mean +- jitter, uniform; `miss` of the answers lost."""
    rng = random.Random(seed)
    rows = []
    for k in range(n):
        t0 = k * period_ms
        rows.append(("stim", t0))
        if rng.random() >= miss:
            rows.append(("resp", t0 + mean_ms + (rng.random() * 2 - 1) * jitter_ms))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["channel", "t"])
        for ch, t in rows:
            w.writerow([ch, f"{t:.4f}"])
