"""A whole device for the acceptance scenarios (docs/V1.md): made by the factory station, then run by the gateway against the real bridge core,
a simulated mouse and PC, a simulated person who calibrates, and a phone that only carries bytes."""
from __future__ import annotations

import struct
from pathlib import Path

from dataopen.bridge import protocol as BP
from dataopen.ctl import protocol as P
from dataopen.ctl.sim import SimLearner, World, dev_image
from dataopen.updates import dev as UD
from dataopen.updates import package as K

from prov_helpers import provisioned, vendor


def xy(report: bytes) -> int:
    """The X count of a 16-bit mouse report."""
    return struct.unpack_from("<h", report, 1)[0]


class Pilot:
    def __init__(self, tmp_path: Path, name: str = "pilot", **kw) -> None:
        self.tmp = Path(tmp_path)
        self.hsm, self.pub = vendor()
        self.dir, _, self.report = provisioned(self.tmp, name, hsm=self.hsm)
        self.forgets: list = []
        self.kw = dict(vendor_pub=self.hsm.pub, hw_id=self.hsm.hw_id, on_forget=lambda: self.forgets.append(1), fw_confirm_s=4,
                       learner=SimLearner(minutes=5.0, seed=2, speed=240.0), trial_s=20)
        self.kw.update(kw)
        self.w = World(self.dir, **self.kw)
        self.seq = 1000
        self.sender = UD.sender(self.tmp / "senders")

    # -- shortcuts
    @property
    def gw(self):
        return self.w.gw

    @property
    def ph(self):
        return self.w.phone

    @property
    def rig(self):
        return self.w.rig

    def connect(self):
        self.ph.connect()
        self.rig.run_until(lambda r: r.pc_conn in ("bridge", "direct"), 5000)
        return self.ph

    def press(self):
        """The button on the device (one press, one use)."""
        self.gw.physical_press(self.w.t)

    # -- the mouse and the PC
    def traffic(self, n: int = 200, dx: int = 3, dy: int = 1):
        """The person moves n ticks; returns the reports the PC received in that time as (route, in_x, out_x)."""
        n0 = len(self.rig.pc_reports)
        for _ in range(n):
            self.rig.move(dx, dy)
            self.rig.step()
        return [(r[1], xy(r[4]), xy(r[3])) for r in self.rig.pc_reports[n0:]]

    def scene(self, objs, t_capture: int) -> None:
        """A scene from the UI detector's side. On the device the detector's process and the gateway share ONE SPI master; here the scene
        enters the same bridge core as a second sender with its own sequence numbers (docs/V1.md: this is what is not integrated yet)."""
        self.seq += 1
        f = BP.scene_frame(t_capture, objs)
        f.seq = self.seq
        self.rig.b.link_rx(self.w.t, BP.pack_frame(f))

    def reach(self, target: float = 600.0, dist: float = 700.0, ms: int = 450, age_ms: int = 20, scene_every: int = 20):
        """A deliberate reach towards a target with the detector's scene arriving every 20 ms, `age_ms` old. Returns the lowest K, where the
        pointer ended, and every report (in, out)."""
        cum0 = self.rig.status().cum_x
        hist, ks = [], []
        prev = carry = 0.0
        n0 = len(self.rig.pc_reports)
        for k in range(ms + 200):
            tau = min(k / ms, 1.0)
            pos = dist * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
            carry += pos - prev
            prev = pos
            i = int(round(carry))
            carry -= i
            self.rig.move(i, 0)
            self.rig.step()
            st = self.rig.status()
            hist.append((self.w.t, st.cum_x - cum0))
            ks.append(st.k_q16 / 65536.0)
            if scene_every and k % scene_every == 0:
                t_cap = self.w.t - age_ms * 1000
                cum = next((c for t, c in reversed(hist) if t <= t_cap), 0)
                self.scene([BP.SceneObject(1, target - cum, 0.0, 24.0, None)], t_cap)
        reports = [(xy(r[4]), xy(r[3])) for r in self.rig.pc_reports[n0:]]
        return {"min_k": min(ks), "end": hist[-1][1], "reports": reports}

    # -- state
    def state(self) -> dict:
        return self.ph.get_state()

    def calibrate(self) -> None:
        """Calibration on a simulated person, then help on, kept."""
        assert self.ph.set("calib.running", True).type == P.T_ACK
        self.w.run(2000)
        assert self.ph.set("calib.running", False).type == P.T_ACK
        r = self.ph.set("assist.on", True)
        assert r.type == P.T_ACK, r.body
        self.w.run(1500)
        assert self.ph.confirm(True).type == P.T_ACK
        self.w.run(1500)

    # -- packages (channel B), from a sender the device does not know yet
    def package(self, **kw) -> bytes:
        self.seq += 1
        return K.build_package(self.sender, self.gw.identity.card(), self.seq, **kw)

    # -- images (channel A), signed by the same test manufacturer that made the board
    @staticmethod
    def image(version: int, min_version=None, payload=None) -> bytes:
        return dev_image(version, min_version, payload)
