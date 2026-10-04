import threading
import time

import numpy as np
import pytest

from dataopen.adapters.mock.server import MockServerOptions, serve_mock
from dataopen.adapters.remote import RemoteGameAdapter, RemoteOptions
from dataopen.core.imageio import read_image
from dataopen.core.models import CaptureRequest, FrameKind
from dataopen.core.randomization import DomainRandomizationController
from dataopen.core.transport import FileMailboxTransport
from dataopen.quality.shm import ShmRing, attach


def test_ring_acquire_release_backpressure_and_zero_copy_view():
    ring = ShmRing(slots=2, capacity_bytes=64 * 48 * 3)
    try:
        a, b = ring.acquire(), ring.acquire()
        assert ring.free_slots == 0
        with pytest.raises(TimeoutError, match="evaluator is not keeping up"):
            ring.acquire(timeout_s=0.1)
        w = attach(a.name)                                          # another "process" writes into the slot
        np.ndarray((48, 64, 3), np.uint8, buffer=w.buf)[:] = 200
        w.close()
        v = a.view(64, 48)
        assert v.shape == (48, 64, 3) and int(v.mean()) == 200       # read without a copy
        with pytest.raises(ValueError, match="slot holds"):
            a.view(640, 480)
        ring.release(a)
        assert ring.free_slots == 1
        del v
        ring.release(b)
    finally:
        ring.close()


def test_ring_grow_recreates_free_slots_with_a_larger_capacity():
    ring = ShmRing(slots=2, capacity_bytes=1000)
    try:
        ring.grow(5000)
        s = ring.acquire()
        assert s.capacity == 5000 and ring.capacity == 5000
        ring.release(s)
    finally:
        ring.close()


class Game:
    def __init__(self, directory, **opts):
        self.stop = threading.Event()
        self.t = threading.Thread(target=serve_mock, args=(directory, MockServerOptions(**opts), self.stop), daemon=True)
        self.t.start()

    def close(self):
        self.stop.set()
        self.t.join(3)


def snapshot_and_adapter(tmp_path, **game_opts):
    mb = tmp_path / "mb"
    g = Game(mb, **game_opts)
    a = RemoteGameAdapter(FileMailboxTransport(mb, default_timeout_s=10), RemoteOptions())
    a.connect()
    rz = DomainRandomizationController(0, a.parameter_space())
    scene = rz.sample_scene(0)
    a.spawner.spawn(scene)
    spec = rz.sample_frame(scene, 0, FrameKind.POSITIVE)
    snap = a.capture.capture(CaptureRequest("f0", spec, 320, 240))
    return g, a, snap


@pytest.mark.parametrize("shm", [True, False])
def test_peek_returns_exactly_the_pixels_that_commit_writes(tmp_path, shm):
    g, a, snap = snapshot_and_adapter(tmp_path, image_shm=shm)
    try:
        assert ("image_shm" in a.caps) is shm and "image_peek" in a.caps
        h = a.capture.peek_pixels(snap)
        assert h is not None and h.array.shape == (240, 320, 3)
        peeked = h.array.copy()
        if shm:
            assert a.ring.free_slots == a.options.shm_slots - 1       # the slot is held until released
        h.release()
        h.release()                                                   # idempotent
        if shm:
            assert a.ring.free_slots == a.options.shm_slots
        a.capture.commit(snap, tmp_path / "out" / "f0.png")           # peek must not consume the frame
        assert np.array_equal(read_image(tmp_path / "out" / "f0.png"), peeked)
        assert not list((tmp_path / "mb" / "staging").glob("peek_*"))   # staged file removed after reading
    finally:
        a.close()
        g.close()


def test_peek_is_unavailable_when_the_mod_does_not_offer_it(tmp_path):
    g, a, snap = snapshot_and_adapter(tmp_path, image_peek=False)
    try:
        assert a.capture.peek_pixels(snap) is None
        a.capture.discard(snap)
    finally:
        a.close()
        g.close()


def test_peek_of_an_unknown_frame_is_an_error_and_releases_the_slot(tmp_path):
    from dataopen.core.models import FrameSnapshot
    from dataopen.core.transport import RemoteError
    g, a, snap = snapshot_and_adapter(tmp_path)
    try:
        ghost = FrameSnapshot("nope", 0, snap.camera, [], meta={"image_mode": "engine"})
        free = a.ring.free_slots
        with pytest.raises(RemoteError, match="no pending image"):
            a.capture.peek_pixels(ghost)
        assert a.ring.free_slots == free                              # not leaked
        a.capture.discard(snap)
    finally:
        a.close()
        g.close()


def test_peek_is_fast_enough_to_not_matter(tmp_path):
    g, a, snap = snapshot_and_adapter(tmp_path)
    try:
        t0 = time.perf_counter()
        for _ in range(20):
            a.capture.peek_pixels(snap).release()
        assert (time.perf_counter() - t0) / 20 < 0.25                 # mailbox round trip dominates; no encode, no disk
        a.capture.discard(snap)
    finally:
        a.close()
        g.close()
