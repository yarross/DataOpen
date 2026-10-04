"""`dataopen runtime ...`: run the inference runtime, benchmark it, watch its output (docs/RUNTIME.md)."""
from __future__ import annotations

import json
import logging
import signal
import threading
import time

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 4


def _make_backend(a):
    from .backends import FallbackBackend, OrtBackend, RknnLiteBackend, ScriptedBackend
    kind = a.backend
    if kind == "mock":
        return ScriptedBackend(latency_ms=a.mock_latency_ms, jitter_ms=a.mock_jitter_ms)
    if not a.model:
        raise SystemExit("--model is required for this backend")
    if kind == "ort":
        b = OrtBackend(a.model, a.provider, a.conf, post=a.post, threads=a.threads)
        if a.fallback_model:
            return FallbackBackend(b, OrtBackend(a.fallback_model, "cpu", a.conf, post=a.post, threads=a.threads))
        return b
    if kind == "rknn":
        return RknnLiteBackend(a.model, a.core, a.conf, post=a.post)
    raise SystemExit(f"unknown backend {kind}")


def _make_source(a):
    from .frames import DirSource, SyntheticSource
    from .ring import RingConsumer
    if a.source == "synthetic":
        return SyntheticSource(a.fps, frames=a.frames)
    if a.source == "dir":
        return DirSource(a.input, a.fps, loop=a.loop)
    if a.source == "ring":
        return RingConsumer(a.input)
    raise SystemExit(f"unknown source {a.source}")


def _parse_cpus(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def _apply_scheduling(a) -> None:
    """Pin the process to the cores that are NOT running the NPU driver / capture path and (optionally) make it real-time. Both are
    best effort: a refusal (no CAP_SYS_NICE) is logged, the runtime still works."""
    import os
    log = logging.getLogger("dataopen.runtime")
    if getattr(a, "cpus", None):
        try:
            os.sched_setaffinity(0, _parse_cpus(a.cpus))
        except (OSError, ValueError, AttributeError) as e:
            log.warning("cannot pin to CPUs %s: %s", a.cpus, e)
    if getattr(a, "rt_priority", 0):
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(a.rt_priority))
        except (OSError, AttributeError) as e:
            log.warning("cannot enable SCHED_FIFO %d: %s (needs CAP_SYS_NICE / root)", a.rt_priority, e)


def _cmd_run(a) -> int:
    from .channel import ResultChannel
    from .loop import InferenceRuntime
    _apply_scheduling(a)
    chan = ResultChannel(a.uds, a.shm) if (a.uds or a.shm) else None
    rt = InferenceRuntime(_make_source(a), [_make_backend(a) for _ in range(a.workers)], chan, a.policy, a.max_age_ms,
                          a.stale_flag_ms, log_every_s=a.log_every, metrics_jsonl=a.metrics_jsonl)
    stop = threading.Event()
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stop.set())
    rt.start()
    t_end = time.monotonic() + a.duration if a.duration else None
    while not stop.is_set() and (t_end is None or time.monotonic() < t_end):
        if a.frames and rt.wait_idle(0.2):
            break
        stop.wait(0.1)
    rt.stop()
    print(json.dumps(rt.metrics.snapshot(), indent=2))
    return EXIT_OK


def _cmd_bench(a) -> int:
    """Closed-loop bench of the whole runtime path at a given input rate: latency percentiles, FPS, drops, queue depth."""
    from .frames import SyntheticSource
    from .loop import InferenceRuntime
    src = SyntheticSource(a.fps, frames=int(a.fps * a.duration))
    rt = InferenceRuntime(src, [_make_backend(a) for _ in range(a.workers)], None, a.policy, a.max_age_ms,
                          warmup=a.warmup)
    rt.start()
    rt.wait_idle(a.duration + 10)
    rt.stop()
    s = rt.metrics.snapshot()
    if a.json:
        print(json.dumps(s, indent=2))
    else:
        print(rt.metrics.line())
        for k, v in s["latency_ms"].items():
            print(f"  {k:14s} " + "  ".join(f"{q}={v[q]}" for q in ("mean", "p50", "p95", "p99", "max") if q in v))
        print(f"  dropped {s['dropped']}")
    return EXIT_OK


def _cmd_tail(a) -> int:
    from ..detector import structs
    from .channel import ShmLatest, UdsSubscriber
    n = 0
    if a.uds:
        sub = UdsSubscriber(a.uds)
        while not a.count or n < a.count:
            buf = sub.recv(2.0)
            if buf is None:
                continue
            arr = structs.validate(buf)
            print(f"frame {arr.frame_id} dets {arr.detection_count} flags {arr.flags}")
            n += 1
    else:
        shm, seq = ShmLatest(a.shm, create=False), 0
        while not a.count or n < a.count:
            r = shm.wait_new(seq, 2.0)
            if r is None:
                continue
            seq, buf, _ = r
            arr = structs.validate(buf)
            print(f"seq {seq} frame {arr.frame_id} dets {arr.detection_count} flags {arr.flags}")
            n += 1
    return EXIT_OK


def _cmd_build_post(a) -> int:
    from .postproc import build_c_library
    print(build_c_library(a.out))
    return EXIT_OK


def register(sub) -> None:
    r = sub.add_parser("runtime", help="inference runtime: run / bench / tail / build-post (docs/RUNTIME.md)")
    rs = r.add_subparsers(dest="rt_cmd", required=True)

    def common(p, bench=False):
        p.add_argument("--backend", default="ort", choices=["ort", "rknn", "mock"])
        p.add_argument("--model")
        p.add_argument("--provider", default="cpu", help="cpu | cuda | tensorrt | directml (ONNX Runtime)")
        p.add_argument("--core", default="auto", help="RKNN NPU core: auto|0|1|2|0_1|all")
        p.add_argument("--fallback-model", dest="fallback_model", help="CPU model used after repeated failures")
        p.add_argument("--conf", type=float, default=0.25)
        p.add_argument("--post", default="auto", choices=["auto", "c", "numpy"])
        p.add_argument("--threads", type=int, default=0)
        p.add_argument("--workers", type=int, default=1)
        p.add_argument("--policy", default="latest", help="latest | queue:N[:oldest|newest]")
        p.add_argument("--max-age-ms", dest="max_age_ms", type=float)
        p.add_argument("--mock-latency-ms", dest="mock_latency_ms", type=float, default=4.0)
        p.add_argument("--mock-jitter-ms", dest="mock_jitter_ms", type=float, default=0.0)
        p.add_argument("--fps", type=float, default=60.0)
        p.add_argument("--duration", type=float, default=5.0 if bench else 0.0, help="seconds (0 = until Ctrl-C / source ends)")

    p = rs.add_parser("run", help="run until stopped, publishing KeypointArrays")
    common(p)
    p.add_argument("--source", default="ring", choices=["ring", "dir", "synthetic"])
    p.add_argument("--input", help="ring socket path or image directory")
    p.add_argument("--frames", type=int)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--uds", help="publish every result on this UDS path")
    p.add_argument("--shm", help="publish the latest result in this shared-memory mailbox")
    p.add_argument("--cpus", help="pin the process to these cores, e.g. 4-7 (keep them off the capture / NPU-driver cores)")
    p.add_argument("--rt-priority", dest="rt_priority", type=int, default=0, help="SCHED_FIFO priority 1-99 (needs CAP_SYS_NICE)")
    p.add_argument("--stale-flag-ms", dest="stale_flag_ms", type=float)
    p.add_argument("--log-every", dest="log_every", type=float, default=5.0)
    p.add_argument("--metrics-jsonl", dest="metrics_jsonl")
    p.set_defaults(fn=_cmd_run)

    b = rs.add_parser("bench", help="synthetic frames at --fps through the whole runtime: latency / FPS / drops / queue depth")
    common(b, bench=True)
    b.add_argument("--warmup", type=int, default=3)
    b.add_argument("--json", action="store_true")
    b.set_defaults(fn=_cmd_bench)

    t = rs.add_parser("tail", help="print results from a running runtime (a minimal consumer / smoke check)")
    t.add_argument("--uds")
    t.add_argument("--shm")
    t.add_argument("--count", type=int, default=0)
    t.set_defaults(fn=_cmd_tail)

    c = rs.add_parser("build-post", help="compile the C post-processor (apollo_post.c) with the system compiler")
    c.add_argument("--out")
    c.set_defaults(fn=_cmd_build_post)


logging.getLogger("dataopen.runtime").addHandler(logging.NullHandler())
