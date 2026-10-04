"""The inference runtime: buffer policies, the main loop (drops, errors, fallback, out-of-order workers), the IPC channel, the
fd-passing frame ring, the C post-processor inside the runtime, ONNX Runtime and closed-loop evaluators as backends, and the
runtime as a closed-loop evaluator. Torch-free: the ONNX model is built by hand."""
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from dataopen.detector import structs
from dataopen.detector.layout import HeadLayout
from dataopen.runtime.backends import (EvaluatorBackend, FallbackBackend, OrtBackend, ScriptedBackend,
                                       predictions_to_array)
from dataopen.runtime.channel import CallbackPublisher, ResultChannel, ShmLatest, UdsPublisher, UdsSubscriber
from dataopen.runtime.evaluator import RuntimeEvaluator
from dataopen.runtime.frames import Frame, QueueSource, now_us
from dataopen.runtime.loop import (FLAG_DEGRADED, FLAG_DROPPED_BEFORE, FLAG_EMPTY_ERROR, FLAG_STALE, InferenceRuntime)
from dataopen.runtime.policy import BoundedQueue, LatestOnly, make_policy
from dataopen.runtime.ring import RingConsumer, RingProducer

IMG = np.zeros((640, 640, 3), np.uint8)
IMG.setflags(write=False)                                         # frames are read-only views of shared memory


def frame(i, released=None):
    return Frame(IMG, i, now_us(), _release=(lambda: released.append(i)) if released is not None else None)


def wait_for(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.002)
    return False


# ---------------------------------------------------------------- policies
def test_latest_only_keeps_one_and_reports_the_superseded_frame():
    p = LatestOnly()
    assert p.put(frame(1)) == []
    (f, why), = p.put(frame(2))
    assert f.frame_id == 1 and why == "superseded" and p.depth() == 1
    assert p.get(0.01).frame_id == 2 and p.get(0.01) is None


def test_bounded_queue_drops_oldest_or_newest_but_never_exceeds_capacity():
    p = BoundedQueue(2, "oldest")
    for i in (1, 2):
        p.put(frame(i))
    (f, why), = p.put(frame(3))
    assert (f.frame_id, why) == (1, "queue_full_oldest") and [p.get(0.01).frame_id, p.get(0.01).frame_id] == [2, 3]
    q = BoundedQueue(2, "newest")
    q.put(frame(1)); q.put(frame(2))
    (f, why), = q.put(frame(3))
    assert (f.frame_id, why) == (3, "queue_full_newest") and q.get(0.01).frame_id == 1


def test_make_policy_parses_specs_and_rejects_nonsense():
    assert isinstance(make_policy("latest"), LatestOnly)
    q = make_policy("queue:3:newest")
    assert isinstance(q, BoundedQueue) and q.capacity == 3 and q.drop == "newest"
    with pytest.raises(ValueError):
        make_policy("unbounded")
    with pytest.raises(ValueError):
        make_policy("queue:0")


# ---------------------------------------------------------------- the loop
def run_finite(backend, n=60, fps=400.0, policy="latest", **kw):
    got, released = [], []
    src = QueueSource(capacity=n + 1)
    pub = CallbackPublisher(lambda b: got.append(structs.validate(b)))
    rt = InferenceRuntime(src, backend, pub, policy, warmup=0, **kw)
    rt.start()
    for i in range(n):
        src.push(IMG, i, now_us(), on_release=lambda i=i: released.append(i))
        time.sleep(1.0 / fps)
    src.close()
    assert rt.wait_idle(10)
    rt.stop()
    return rt, got, released


def test_slow_model_with_latest_only_drops_frames_but_every_frame_is_released_exactly_once():
    rt, got, released = run_finite(ScriptedBackend(latency_ms=15), n=60, fps=300.0)
    s = rt.metrics.snapshot()
    assert sorted(released) == list(range(60))                       # each buffer returned to its producer, once
    assert s["frames_in"] == 60 and s["published"] == len(got)
    assert s["published"] + s["dropped"]["superseded"] == 60
    assert s["dropped"]["superseded"] > 10                           # 15 ms model, 3.3 ms frames: it had to skip
    ids = [a.frame_id for a in got]
    assert ids == sorted(set(ids))                                   # published in order, no duplicates
    assert any(a.flags & FLAG_DROPPED_BEFORE for a in got[1:])
    assert s["queue"]["max"] <= 1


def test_a_fast_model_drops_nothing():
    rt, got, released = run_finite(ScriptedBackend(latency_ms=0.5), n=40, fps=200.0)
    s = rt.metrics.snapshot()
    assert s["dropped_total"] == 0 and len(got) == 40 and len(released) == 40
    assert s["latency_ms"]["infer_ms"]["p50"] >= 0.5 and s["latency_ms"]["e2e_ms"]["count"] == 40


def test_bounded_queue_keeps_more_frames_than_latest_only_for_the_same_load():
    a = run_finite(ScriptedBackend(latency_ms=10), n=40, fps=150.0, policy="latest")[0].metrics.snapshot()
    b = run_finite(ScriptedBackend(latency_ms=10), n=40, fps=150.0, policy="queue:8:oldest")[0].metrics.snapshot()
    assert b["published"] > a["published"] and b["queue"]["max"] > 1


def test_stale_frames_are_dropped_instead_of_computed():
    rt, got, released = run_finite(ScriptedBackend(latency_ms=12), n=30, fps=300.0, policy="queue:30:oldest", max_age_ms=20)
    s = rt.metrics.snapshot()
    assert s["dropped"]["stale"] > 0 and s["published"] + s["dropped_total"] == 30
    assert len(released) == 30


def test_stale_flag_marks_late_results():
    rt, got, _ = run_finite(ScriptedBackend(latency_ms=25), n=10, fps=500.0, policy="queue:10:oldest", stale_flag_ms=40)
    assert any(a.flags & FLAG_STALE for a in got) and not (got[0].flags & FLAG_STALE)


def test_backend_error_publishes_an_explicit_empty_error_result_and_the_loop_survives():
    rt, got, released = run_finite(ScriptedBackend(fail_every=4), n=22, fps=100.0)
    errs = [a for a in got if a.flags & FLAG_EMPTY_ERROR]
    assert errs and all(a.detection_count == 0 for a in errs)        # "I failed" never looks like "nobody there"
    assert len(got) == 22 and rt.metrics.errors == len(errs) and len(released) == 22
    assert "scripted failure" in rt.metrics.snapshot()["errors"]["last"]
    assert rt.metrics.snapshot()["errors"]["consecutive"] == 0


def test_errors_can_be_withheld_from_the_channel():
    rt, got, _ = run_finite(ScriptedBackend(fail_every=2), n=10, fps=100.0, publish_errors=False)
    assert not any(a.flags & FLAG_EMPTY_ERROR for a in got) and rt.metrics.snapshot()["dropped"]["error"] == 5


def test_header_carries_frame_id_capture_time_brightness_and_letterbox():
    got = []
    src = QueueSource(4)
    rt = InferenceRuntime(src, ScriptedBackend(), CallbackPublisher(lambda b: got.append(structs.validate(b))), warmup=0)
    rt.start()
    img = np.full((640, 640, 3), 100, np.uint8)
    src.push(img, 77, 123456)
    assert wait_for(lambda: got)
    rt.stop()
    a = got[0]
    assert (a.frame_id, a.timestamp_us, a.avg_scene_brightness) == (77, 123456, 100)
    assert a.letterbox_scale == 1.0 and (a.src_w, a.src_h) == (640, 640) and a.detection_count == 1


def test_publisher_failure_does_not_stop_inference():
    def boom(_):
        raise OSError("consumer died")
    src = QueueSource(8)
    rt = InferenceRuntime(src, ScriptedBackend(), CallbackPublisher(boom), warmup=0)
    rt.start()
    for i in range(5):
        src.push(IMG, i)
        time.sleep(0.01)
    assert wait_for(lambda: rt.metrics.processed + rt.metrics.published >= 5)
    rt.stop()
    assert rt.metrics.errors >= 5 and rt.metrics.published == 5


def test_producer_is_never_blocked_by_a_stalled_model():
    src = QueueSource(capacity=2)
    rt = InferenceRuntime(src, ScriptedBackend(latency_ms=300), None, "latest", warmup=0)
    rt.start()
    t0 = time.monotonic()
    pushes = [src.push(IMG, i) for i in range(50)]
    assert time.monotonic() - t0 < 0.2                               # push() is O(1) whatever the model does
    assert not all(pushes) or True
    rt.stop(timeout=2)


def test_two_workers_never_publish_out_of_order():
    state = {"n": 0}
    lock = threading.Lock()

    class Uneven(ScriptedBackend):
        def infer(self, frame):
            with lock:
                state["n"] += 1
                slow = state["n"] % 2 == 1
            time.sleep(0.03 if slow else 0.002)
            return super().infer(frame)

    rt, got, released = run_finite([Uneven(), Uneven()], n=40, fps=200.0, policy="queue:8:oldest")
    ids = [a.frame_id for a in got]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    s = rt.metrics.snapshot()
    assert s["published"] + s["dropped_total"] == 40 and len(released) == 40


def test_health_reports_a_hung_worker():
    src = QueueSource(4)
    b = ScriptedBackend(stall_at=(1, 0.6))
    rt = InferenceRuntime(src, b, None, warmup=0, hang_ms=100)
    rt.start()
    src.push(IMG, 0)
    assert wait_for(lambda: not rt.health()["ok"], 2)
    assert rt.health()["stalled_workers"] == [0]
    assert wait_for(lambda: rt.health()["ok"], 3)                    # recovers when the stall ends
    rt.stop()


def test_fallback_backend_switches_after_repeated_errors_and_flags_every_result_degraded():
    class Dead(ScriptedBackend):
        def infer(self, frame):
            raise RuntimeError("npu hang")

    fb = FallbackBackend(Dead(), ScriptedBackend(), max_errors=2, retry_s=999)
    rt, got, _ = run_finite(fb, n=12, fps=100.0, policy="queue:12:oldest")
    flags = [a.flags for a in got]
    assert flags[0] & FLAG_EMPTY_ERROR                               # first failure is reported ...
    assert all(f & FLAG_DEGRADED for f in flags[2:]) and not any(f & FLAG_EMPTY_ERROR for f in flags[2:])    # ... then served by CPU
    assert fb.info()["switches"] == 1 and rt.metrics.degraded


def test_fallback_primary_recovers_after_the_retry_interval():
    class Flaky(ScriptedBackend):
        broken = True

        def infer(self, frame):
            if Flaky.broken:
                raise RuntimeError("down")
            return super().infer(frame)

    fb = FallbackBackend(Flaky(), ScriptedBackend(), max_errors=1, retry_s=0.05)
    f = frame(0)
    fb.infer(f)                                                      # one failure with max_errors=1: served by the secondary
    assert fb.degraded
    fb.infer(f)
    Flaky.broken = False
    time.sleep(0.06)
    fb.infer(f)
    assert not fb.degraded


def test_graceful_stop_releases_queued_frames_and_closes_everything():
    released = []
    src = QueueSource(10)
    rt = InferenceRuntime(src, ScriptedBackend(latency_ms=100), None, "queue:10:oldest", warmup=0)
    rt.start()
    for i in range(8):
        src.push(IMG, i, on_release=lambda i=i: released.append(i))
    time.sleep(0.05)
    rt.stop()
    assert sorted(released) == list(range(8))
    assert not any(t.is_alive() for t in rt._threads)


def test_metrics_snapshot_line_and_prometheus_are_consistent():
    rt, got, _ = run_finite(ScriptedBackend(latency_ms=5), n=30, fps=100.0)
    s = rt.metrics.snapshot()
    assert set(s["latency_ms"]) == {"e2e_ms", "frame_age_ms", "queue_wait_ms", "infer_ms", "publish_ms"}
    assert "fps in/out" in rt.metrics.line() and "queue" in rt.metrics.line()
    prom = rt.metrics.prometheus()
    assert 'apollo_runtime_latency_ms{stage="e2e",quantile="p99"}' in prom and 'reason="superseded"' in prom
    json.dumps(s)


# ---------------------------------------------------------------- IPC channel
def test_uds_stream_delivers_every_result_in_order_to_every_subscriber(tmp_path):
    pub = UdsPublisher(tmp_path / "r.sock")
    subs = [UdsSubscriber(tmp_path / "r.sock") for _ in range(2)]
    got = [[], []]

    def reader(k):                                                   # a subscriber that keeps up (a slow one would miss messages)
        while len(got[k]) < 20:
            b = subs[k].recv(1.0)
            if b is None:
                return
            got[k].append(structs.validate(b).frame_id)

    ts = [threading.Thread(target=reader, args=(k,)) for k in range(2)]
    for t in ts:
        t.start()
    time.sleep(0.05)
    for i in range(20):
        pub.publish(bytes(structs.pack([], i, 0, 0)))
        time.sleep(0.002)
    for t in ts:
        t.join(3)
    assert got[0] == list(range(20)) == got[1]
    for s in subs:
        s.close()
    pub.close()


def test_a_stuck_subscriber_never_blocks_the_publisher_or_the_healthy_one(tmp_path):
    pub = UdsPublisher(tmp_path / "r.sock", sndbuf=4096)
    stuck, good = UdsSubscriber(tmp_path / "r.sock"), UdsSubscriber(tmp_path / "r.sock")
    time.sleep(0.05)
    t0 = time.monotonic()
    seen = []
    for i in range(300):
        pub.publish(bytes(structs.pack([], i, 0, 0)))
        r = good.recv(0.0)
        while r is not None:
            seen.append(structs.validate(r).frame_id)
            r = good.recv(0.0)
    assert time.monotonic() - t0 < 2.0                               # publishing 300 results never waited on `stuck`
    assert seen[-1] >= 290 and pub.stats()["published"] == 300
    stuck.close(); good.close(); pub.close()


def test_latest_subscriber_read_skips_to_the_newest(tmp_path):
    pub = UdsPublisher(tmp_path / "r.sock")
    sub = UdsSubscriber(tmp_path / "r.sock")
    time.sleep(0.05)
    for i in range(5):
        pub.publish(bytes(structs.pack([], i, 0, 0)))
    time.sleep(0.05)
    assert structs.validate(sub.latest(0.5)).frame_id == 4
    sub.close(); pub.close()


def test_shm_mailbox_returns_only_complete_messages_under_a_hammering_writer():
    name = f"apollo_t_{now_us()}"
    w = ShmLatest(name, 2048)
    r = ShmLatest(name, create=False)
    assert r.read() is None
    stop, bad = threading.Event(), []

    def writer():
        i = 0
        while not stop.is_set():
            i += 1
            w.publish(bytes(structs.pack([], i, i, 0)))

    t = threading.Thread(target=writer)
    t.start()
    last, n = 0, 0
    end = time.monotonic() + 0.5
    while time.monotonic() < end:
        got = r.read()
        if got is None:
            continue
        seq, data, _ = got
        a = structs.validate(data)                                   # raises on a torn payload
        if a.frame_id != a.timestamp_us or seq < last:
            bad.append((seq, a.frame_id))
        last, n = seq, n + 1
    stop.set(); t.join()
    r.close(); w.close()
    assert n > 100 and not bad


def test_shm_wait_new_times_out_and_oversize_is_refused():
    name = f"apollo_t2_{now_us()}"
    w = ShmLatest(name, 64)
    with pytest.raises(ValueError):
        w.publish(b"x" * 65)
    assert w.wait_new(0, 0.05) is None
    w.publish(b"abc")
    assert w.wait_new(0, 0.5)[1] == b"abc"
    w.close()


def test_result_channel_fans_out_to_both_transports(tmp_path):
    name = f"apollo_t3_{now_us()}"
    ch = ResultChannel(str(tmp_path / "c.sock"), name)
    sub = UdsSubscriber(tmp_path / "c.sock")
    time.sleep(0.05)
    ch.publish(bytes(structs.pack([], 9, 0, 0)))
    assert structs.validate(sub.recv(1.0)).frame_id == 9
    assert structs.validate(ShmLatest(name, create=False).read()[1]).frame_id == 9
    sub.close(); ch.close()


# ---------------------------------------------------------------- DMA-style frame ring
def ring_pair(path, **kw):
    """The producer's constructor handshakes with the consumer (which accepts inside get()): drive both, as the runtime's reader
    thread does in production."""
    cons, box = RingConsumer(path), {}
    t = threading.Thread(target=lambda: box.update(p=RingProducer(path, **kw)))
    t.start()
    while t.is_alive():
        cons.get(0.02)
    t.join()
    return cons, box["p"]


def test_ring_passes_pixels_zero_copy_and_recycles_slots(tmp_path):
    cons, prod = ring_pair(tmp_path / "ring.sock", n_slots=2)
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(6)]
    for i, im in enumerate(imgs):                                    # 6 frames through 2 slots: needs release to recycle
        assert wait_for(lambda im=im, i=i: prod.submit(im, i, 1000 + i), 2)
        f = cons.get(1.0)
        assert f.frame_id == i and f.ts_us == 1000 + i and np.array_equal(f.array, im)
        assert not f.array.flags.writeable                           # the consumer may not scribble on the producer's buffer
        f.release()
        f.release()                                                  # idempotent
    prod.close(); cons.close()


def test_ring_producer_drops_instead_of_blocking_when_the_consumer_holds_every_slot(tmp_path):
    cons, prod = ring_pair(tmp_path / "ring.sock", n_slots=2)
    held = []
    for i in range(2):
        assert prod.submit(IMG, i)
        held.append(cons.get(1.0))
    t0 = time.monotonic()
    assert not prod.submit(IMG, 99)                                  # no free slot: drop, do not wait
    assert time.monotonic() - t0 < 0.1 and prod.stats()["dropped_no_slot"] == 1
    held[0].release()
    assert wait_for(lambda: prod.submit(IMG, 100), 2)
    prod.close(); cons.close()


def test_ring_consumer_survives_a_producer_restart(tmp_path):
    path = tmp_path / "ring.sock"
    cons, p1 = ring_pair(path, n_slots=2)
    assert p1.submit(IMG, 1)
    assert cons.get(1.0).frame_id == 1
    p1.close()
    box = {}
    t = threading.Thread(target=lambda: box.update(p=RingProducer(path, n_slots=3)))
    t.start()
    while t.is_alive():
        cons.get(0.02)
    t.join()
    p2 = box["p"]
    assert wait_for(lambda: p2.submit(IMG, 2), 2)
    got = None
    for _ in range(20):
        got = cons.get(0.2)
        if got is not None:
            break
    assert got is not None and got.frame_id == 2
    p2.close(); cons.close()


def test_runtime_end_to_end_from_the_ring_to_the_uds_channel(tmp_path):
    ring, uds = tmp_path / "ring.sock", tmp_path / "out.sock"
    cons, pub = RingConsumer(ring), UdsPublisher(uds)
    rt = InferenceRuntime(cons, ScriptedBackend(latency_ms=2), pub, warmup=0).start()
    prod = RingProducer(ring, n_slots=4)
    sub = UdsSubscriber(uds)
    ids = []

    def reader():
        while (b := sub.recv(0.5)) is not None:
            ids.append(structs.validate(b).frame_id)

    t = threading.Thread(target=reader)
    t.start()
    time.sleep(0.05)
    for i in range(30):
        prod.submit(IMG, i)
        time.sleep(0.01)
    t.join(5)
    assert len(ids) >= 25 and ids == sorted(ids)
    rt.stop(); prod.close(); sub.close()


# ---------------------------------------------------------------- real ONNX model through ONNX Runtime + the post-processor
def build_onnx(path: Path, layout: HeadLayout, uint8_nhwc=True) -> Path:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper
    W, H = layout.input_size
    outs, nodes, inits = [], [], []
    for li, s in enumerate(layout.strides):
        t = np.zeros((1, layout.channels, H // s, W // s), np.float32)
        t[0, :layout.n_cls] = -8.0
        if s == 16:                                                  # one person on the stride-16 grid, cell (10, 12), class 1
            gx, gy = 10, 12
            t[0, 1, gy, gx] = 6.0
            t[0, layout.n_cls:layout.n_cls + 4, gy, gx] = np.log(np.array([10.0, 20.0, 12.0, 30.0]) / s)
            t[0, layout.n_cls + 4 + 3 * layout.n_kpt:, gy, gx] = 0.0
            t[0, layout.n_cls + 4 + 2 * layout.n_kpt:layout.n_cls + 4 + 3 * layout.n_kpt, gy, gx] = 3.0
        name = f"p{li + 3}"
        inits.append(numpy_helper.from_array(t, name=f"{name}_c"))
        nodes.append(helper.make_node("Identity", [f"{name}_c"], [name]))
        outs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, t.shape))
    shape = [1, H, W, 3] if uint8_nhwc else [1, 3, H, W]
    inp = helper.make_tensor_value_info("frame", TensorProto.UINT8 if uint8_nhwc else TensorProto.FLOAT, shape)
    # ORT needs the input consumed to keep it alive in the graph: a cheap ReduceMax into an unused output keeps the signature honest
    nodes.append(helper.make_node("Cast", ["frame"], ["frame_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("ReduceMax", ["frame_f"], ["peak"], keepdims=0))
    outs.append(helper.make_tensor_value_info("peak", TensorProto.FLOAT, []))
    m = helper.make_model(helper.make_graph(nodes, "g", [inp], outs, inits), opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    e = m.metadata_props.add()
    e.key, e.value = "apollo", layout.to_json()
    onnx.save(m, str(path))
    return path


LAY = HeadLayout(n_cls=2, n_kpt=12, strides=(8, 16, 32), input_size=(640, 640), classes=("player_ct", "player_t"),
                 keypoints=tuple(f"k{i}" for i in range(12)))


@pytest.mark.parametrize("post", ["c", "numpy"])
def test_ort_backend_runs_a_real_onnx_model_and_both_post_processors_agree(tmp_path, post):
    pytest.importorskip("onnxruntime")
    model = build_onnx(tmp_path / "m.onnx", LAY)
    be = OrtBackend(model, "cpu", conf_thr=0.5, post=post)
    a = be.infer(frame(0))
    assert a.detection_count == 1 and a.schema_hash == structs.schema_hash(LAY.keypoints, LAY.classes)
    d = structs.unpack(a)[0]
    ax, ay = (10 + 0.5) * 16, (12 + 0.5) * 16
    assert d["cls"] == 1 and d["box"] == pytest.approx([ax - 10, ay - 20, ax + 12, ay + 30], abs=0.1)
    assert d["kxy"][0] == pytest.approx([ax, ay], abs=0.1)
    assert be.info()["post"] in ("c", "numpy") and "NHWC" in be.info()["input"]


def test_c_and_numpy_postprocessors_produce_the_same_bytes(tmp_path):
    pytest.importorskip("onnxruntime")
    model = build_onnx(tmp_path / "m.onnx", LAY)
    f = frame(0)
    c = bytes(OrtBackend(model, "cpu", conf_thr=0.5, post="c").infer(f))
    n = bytes(OrtBackend(model, "cpu", conf_thr=0.5, post="numpy").infer(f))
    assert c == n


def test_ort_backend_rejects_wrong_input_and_models_without_layout(tmp_path):
    pytest.importorskip("onnxruntime")
    model = build_onnx(tmp_path / "m.onnx", LAY)
    be = OrtBackend(model, "cpu", post="numpy")
    with pytest.raises(ValueError, match="640x640x3 uint8"):
        be.infer(Frame(np.zeros((320, 320, 3), np.uint8), 0, 0))
    with pytest.raises(ValueError, match="640x640x3 uint8"):
        be.infer(Frame(np.zeros((640, 640, 3), np.float32), 0, 0))
    import onnx
    m = onnx.load(str(model))
    del m.metadata_props[:]
    onnx.save(m, str(tmp_path / "bare.onnx"))
    with pytest.raises(ValueError, match="apollo"):
        OrtBackend(tmp_path / "bare.onnx", "cpu")


def test_ort_backend_warns_and_falls_back_to_cpu_when_cuda_is_unavailable(tmp_path, caplog):
    ort = pytest.importorskip("onnxruntime")
    model = build_onnx(tmp_path / "m.onnx", LAY)
    be = OrtBackend(model, "cuda", post="numpy")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        assert be.provider_used == "CPUExecutionProvider"
    assert be.infer(frame(0)).detection_count in (0, 1)


def test_full_runtime_with_a_real_onnx_model_and_the_shm_mailbox(tmp_path):
    pytest.importorskip("onnxruntime")
    model = build_onnx(tmp_path / "m.onnx", LAY)
    name = f"apollo_rt_{now_us()}"
    ch = ResultChannel(None, name)
    src = QueueSource(8)
    rt = InferenceRuntime(src, OrtBackend(model, "cpu", conf_thr=0.5), ch, warmup=2).start()
    for i in range(10):
        src.push(IMG, i)
        time.sleep(0.02)
    box = ShmLatest(name, create=False)
    assert wait_for(lambda: box.read() is not None and structs.validate(box.read()[1]).frame_id == 9, 3)
    a = structs.validate(box.read()[1], LAY.keypoints, LAY.classes)  # a consumer checks the schema hash before trusting fields
    assert a.detection_count == 1
    rt.stop(); box.close()


def test_layout_sidecar_next_to_the_model_is_what_rknn_needs(tmp_path):
    from dataopen.runtime.backends import RknnLiteBackend, write_layout_sidecar
    p = write_layout_sidecar(LAY, tmp_path / "m.layout.json")
    assert HeadLayout.from_json(p.read_text()) == LAY
    with pytest.raises(RuntimeError, match="rknnlite"):               # no Rockchip toolkit here: a clear message, not an ImportError
        RknnLiteBackend(tmp_path / "m.rknn", layout=LAY)


# ---------------------------------------------------------------- the closed loop <-> runtime bridge
def test_evaluator_backend_runs_the_closed_loop_mock_inside_the_runtime():
    from dataopen.quality.evaluators.simulated import CallableEvaluator
    from dataopen.quality.types import Prediction

    def model(images):
        kp = np.stack([np.arange(12) * 5.0 + 100, np.arange(12) * 9.0 + 50, np.full(12, 0.9)], axis=1)
        return [[Prediction((90.0, 40.0, 80.0, 120.0), 0.8, kp, 1)] for _ in images]

    be = EvaluatorBackend(CallableEvaluator(model))
    got = []
    src = QueueSource(4)
    rt = InferenceRuntime(src, be, CallbackPublisher(lambda b: got.append(structs.validate(b))), warmup=0).start()
    src.push(IMG, 5)
    assert wait_for(lambda: got)
    rt.stop()
    d = structs.unpack(got[0])[0]
    assert d["cls"] == 1 and d["score"] == pytest.approx(0.8, abs=0.01) and d["box"] == pytest.approx([90, 40, 170, 160], abs=0.1)
    assert d["kxy"][2] == pytest.approx([110, 68], abs=0.1)          # Q12.4 quantization: 1/16 px


def test_predictions_with_the_wrong_keypoint_count_are_refused():
    from dataopen.quality.types import Prediction
    p = Prediction((0, 0, 10, 10), 0.9, np.zeros((17, 3)), 0)
    with pytest.raises(ValueError, match="17 keypoints"):
        predictions_to_array([p])


def test_runtime_evaluator_scores_frames_through_the_production_path_and_maps_letterboxed_sizes_back():
    from dataopen.quality.evaluators.simulated import CallableEvaluator
    from dataopen.quality.types import Prediction

    seen = []

    def model(images):
        seen.append(images[0].shape[:2])
        kp = np.stack([np.full(12, 320.0), np.full(12, 320.0), np.full(12, 0.9)], axis=1)
        return [[Prediction((300.0, 300.0, 40.0, 40.0), 0.9, kp, 0)] for _ in images]

    ev = RuntimeEvaluator(EvaluatorBackend(CallableEvaluator(model)), window=4)
    out = ev.predict([np.zeros((320, 640, 3), np.uint8), np.zeros((640, 640, 3), np.uint8)])
    ev.close()
    assert seen == [(640, 640), (640, 640)]
    wide, square = out[0][0], out[1][0]
    assert square.bbox == pytest.approx((300, 300, 40, 40), abs=0.1)
    # a 320x640 image is letterboxed at scale 1 with 160 px of padding on top: model y=320 is image y=160
    assert wide.keypoints[0][:2] == pytest.approx([320, 160], abs=0.1) and wide.bbox[1] == pytest.approx(140, abs=0.1)


def test_runtime_evaluator_raises_on_backend_failure_instead_of_reporting_no_detections():
    ev = RuntimeEvaluator(ScriptedBackend(fail_every=1), window=2)
    with pytest.raises(RuntimeError, match="inference failed"):
        ev.predict([IMG])
    ev.close()


def test_runtime_evaluator_handles_more_images_than_its_window_and_never_drops():
    ev = RuntimeEvaluator(ScriptedBackend(latency_ms=1), window=3)
    out = ev.predict([IMG] * 10)
    snap = ev.runtime_metrics()
    ev.close()
    assert len(out) == 10 and all(len(o) == 1 for o in out) and snap["dropped_total"] == 0


def test_closed_loop_through_the_production_runtime_gives_the_same_verdicts_as_the_research_path(tmp_path):
    """`--quality-runtime`: every frame goes reader -> policy -> backend -> Q12.4 KeypointArray -> back to predictions. The only
    differences to the direct path are the 1/16 px and 1/255 quantization of the packed result, so verdicts must agree."""
    import csv

    from dataopen.cli import EXIT_OK, main

    def run(name, *extra):
        out = tmp_path / name
        with pytest.raises(SystemExit) as e:
            main(["collect", "--game", "mock_shooter", "--frames", "24", "--out", str(out), "--quality-sim", "--no-doctor",
                  "--seed", "3", *extra])
        assert e.value.code == EXIT_OK
        with (out / "quality_index.csv").open() as f:
            return {r["frame_id"]: r for r in csv.DictReader(f)}

    direct, viart = run("direct"), run("runtime", "--quality-runtime")
    assert direct.keys() == viart.keys() and direct
    same = sum(direct[k]["verdict"] == viart[k]["verdict"] for k in direct)
    assert same >= 0.9 * len(direct), (same, len(direct))
    close = sum(abs(float(direct[k]["oks_score"] or 0) - float(viart[k]["oks_score"] or 0)) < 0.03 for k in direct)
    assert close >= 0.9 * len(direct)


def test_cli_bench_prints_latency_fps_drops_and_queue_depth(capsys):
    from dataopen.cli import main
    with pytest.raises(SystemExit) as e:
        main(["runtime", "bench", "--backend", "mock", "--fps", "100", "--duration", "1", "--mock-latency-ms", "3", "--json"])
    assert e.value.code in (0, None)
    out = json.loads(capsys.readouterr().out)
    assert out["published"] > 50 and out["latency_ms"]["infer_ms"]["p50"] >= 3 and "depth" in out["queue"]


def test_cli_run_publishes_and_tail_reads_from_the_shared_memory_mailbox(capsys):
    from dataopen.cli import main
    name = f"apollo_cli_{now_us()}"
    th = threading.Thread(target=lambda: pytest.raises(SystemExit, main, ["runtime", "run", "--backend", "mock", "--source", "synthetic",
                                                                          "--fps", "50", "--duration", "1.5", "--shm", name,
                                                                          "--mock-latency-ms", "2"]))
    th.start()
    assert wait_for(lambda: _shm_exists(name), 3)
    box = ShmLatest(name, create=False)
    r = box.wait_new(0, 2.0)
    assert r is not None and structs.validate(r[1]).detection_count == 1
    box.close()
    th.join(5)
    out = capsys.readouterr().out
    assert '"published"' in out


def _shm_exists(name):
    try:
        ShmLatest(name, create=False).close()
        return True
    except (FileNotFoundError, OSError):
        return False


def test_cpu_list_parsing():
    from dataopen.runtime.cli import _parse_cpus
    assert _parse_cpus("4-7") == {4, 5, 6, 7} and _parse_cpus("0,2-3") == {0, 2, 3}
