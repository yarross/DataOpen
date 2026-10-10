"""OS transparency / zero PC footprint (docs/TRANSPARENCY.md): the computer and any software on it see an ORDINARY mouse.

What is proved here is the code and the simulation: simulated mice (boot, 16-bit, 12-bit with Report ID, a composite with a HID++-shaped
channel and a keyboard, a high-rate mouse), a simulated PC that enumerates and talks to whatever answers, the real C core. It is NOT proved with
a real mouse firmware, a real operating system, a corporate policy, an antivirus or a USB analyzer, and the electrical bypass is a model."""
import random
import re
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

from dataopen.bridge import protocol as P
from dataopen.bridge import sim_usb as U
from dataopen.bridge import transparency as T
from dataopen.bridge.cbridge import CCfg, PX_FORWARD, PX_LOCAL, PX_SERVE, REASONS, STATE, hid_parse
from dataopen.bridge.sim import Rig
from dataopen.bridge.sim_usb import SimMouse, SimPC

from test_bridge import MICE, c_mask, fixed_params, ok_axis, with_scene

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="no C compiler")

ROOT = Path(__file__).resolve().parents[1]
CSRC = ROOT / "src" / "dataopen" / "bridge" / "csrc"
DOC = ROOT / "docs" / "TRANSPARENCY.md"


def fresh(name, persona="overshooter", **kw):
    """A day in miniature: power on (the PC enumerates the real mouse directly), then the bridge engages."""
    asc, trm = fixed_params(persona)
    r = Rig(SimMouse(**MICE[name]), asc=asc, tremor=trm, **kw)
    r.run(500)
    assert r.transcripts.get("direct"), "the PC should have enumerated the real mouse first"
    assert r.engage_fast(), (STATE[r.status().state], REASONS[r.status().reason])
    return r


def state(r):
    s = r.status()
    return STATE[s.state], REASONS[s.reason]


# ------------------------------------------------------------------------------------------------------------ T1: nothing for the PC
def test_the_device_has_no_software_for_the_pc_in_its_scope():
    from dataopen.release import registry as R

    pc_side = [m.id for m in R.MODULES if m.status != R.DEFERRED and any(str(f).startswith(("adapters/", "src/dataopen/installer")) for f in m.paths)]
    assert pc_side == [], pc_side                                                           # no v1 module ships anything that runs on the PC
    # nothing the device does needs the PC to run our code: the control path is the phone (BLE), the video path never touches the PC's software
    for p in sorted((ROOT / "src" / "dataopen" / "bridge").glob("*.py")):
        if p.name == "transparency.py":
            continue                                                                         # it names them to say they are not part of the device
        text = p.read_text(encoding="utf-8")
        assert "installer" not in text and "adapters" not in text, p.name
    assert not re.search(r"\bdriver\b|\bservice\b.*\bPC\b", (CSRC / "usb_image.c").read_text(encoding="utf-8"), re.I)


def test_nothing_in_the_core_can_describe_a_device_of_its_own():
    """No descriptor, no VID/PID, no string, no usage page is written into the C sources: every byte of an identity the PC sees comes from
    the mouse's own image (usb_proxy.c serves only `im->` / `f->` fields)."""
    for p in CSRC.glob("*.c"):
        text = p.read_text(encoding="utf-8")
        assert not re.search(r"static\s+const\s+(uint8_t|char)\s+\w+\s*\[\s*\]", text), p.name       # no baked-in tables of bytes
        assert not re.search(r"\bidVendor\b|\bidProduct\b|\biManufacturer\b|\biSerial", text), p.name
    served = re.findall(r"\*data\s*=\s*([^;]+);", (CSRC / "usb_proxy.c").read_text(encoding="utf-8"))
    assert len(served) == 4 and all(re.match(r"(im|f)->", s.strip()) for s in served), served


# ------------------------------------------------------------------------------------------------------------ T6: nothing from the core on its own
def test_the_cores_pc_facing_functions_are_exactly_two():
    """The only functions of the core that hand bytes to the PC side are `bridge_pc_setup` (from the mouse's image) and `bridge_mouse_in` (a
    copy of the report, same length) - plus `bridge_merge`, which edits a pending report of the same endpoint. The only other byte-producer
    is the SPI frame to the MODULE. A new function that produces bytes fails this test until somebody classifies it."""
    header = (CSRC / "bridge.h").read_text(encoding="utf-8")
    producers = set()
    for m in re.finditer(r"^\w[\w\s\*]*?\b(bridge_\w+)\(([^)]*)\)\s*;", header, re.M):
        name, args = m.group(1), m.group(2)
        if re.search(r"(?<!const )uint8_t\s*\*\s*\w+|const uint8_t\s*\*\*|uint8_t\s+\w+\[\w+\]", args) and not re.search(r"const uint8_t\s+\w+\[", args):
            producers.add(name)
    assert producers == {"bridge_pc_setup", "bridge_mouse_in", "bridge_merge", "bridge_link_tx"}, producers


def test_a_quiet_mouse_means_a_silent_bridge_through_every_state_and_every_module_message():
    r = fresh("logi")
    r.pc_reports.clear()
    rng = random.Random(11)
    junk_kinds = [P.LK_NOP, P.LK_TELEM, P.LK_STATUS, P.LK_TSYNC_REPLY, P.LK_HELLO, P.LK_PARAMS_ASC, P.LK_PARAMS_TREMOR, P.LK_SCENE, P.LK_CMD, P.LK_TSYNC]
    for k in range(6000):
        if k % 7 == 0:
            r.b.link_rx(r.t, bytes(rng.randrange(256) for _ in range(128)))                      # garbage
        if k % 11 == 0:
            r.module.send(r.t, [P.Frame(rng.choice(junk_kinds), bytes(rng.randrange(256) for _ in range(rng.randrange(0, 40))))])
        if k % 500 == 100:
            r.module.send(r.t, [P.cmd_frame(rng.choice((0, 1, 2)))])                              # every command a module can send
        if k == 2000:
            r.panic(True)
        if k == 2100:
            r.panic(False)
        r.step()
    assert r.pc_reports == [] and r.dropped == 0                                                 # no mouse input: not one report to the PC, whatever happened
    n_enum = len(r.pc_enumerations)
    assert n_enum >= 2 and all(kind in ("direct", "bridge") for _, kind, _ in r.pc_enumerations)


def test_the_mouse_is_asked_nothing_the_pc_did_not_ask():
    r = Rig(SimMouse(**MICE["composite"]), asc=fixed_params()[0], tremor=fixed_params()[1])
    r.run(500)
    r.mouse.control_log.clear()
    assert r.engage_fast()
    log = [x[:6] for x in r.mouse.control_log]
    wanted = [(0x81, 6, 0x2200, f.num, r.b.img_rd_wanted(i), b"") for i, f in enumerate(f for f in r.mouse.ifaces if f.alt == 0)]
    probe, rest = log[: len(wanted)], log[len(wanted):]
    assert probe == wanted                                                                       # reading the report descriptors: what any host does
    pc = r.transcripts["bridge"]
    forwarded = []
    for bm, req, value, index, length, data, _st, _d in pc:
        kind, _ = r.b.pc_setup(bm, req, value, index, length)
        if kind == PX_FORWARD:
            forwarded.append((bm, req, value, index, length, data))
    assert rest == forwarded                                                                     # and every other request is one the PC made, in its order


# ------------------------------------------------------------------------------------------------------------ T2: the same mouse
SCENARIOS = ("assist", "soft_panic", "silent_module", "split")


@pytest.mark.parametrize("name", list(MICE))
def test_the_pc_sees_the_same_mouse_in_every_state(name):
    r = fresh(name)
    direct, first = r.transcripts["direct"], r.transcripts["bridge"]
    assert len(direct) > 30 and first == direct                                                  # every request, status and byte
    assert direct[1][7] == r.mouse.dev
    again = SimPC.enumerate(r.front)
    assert again == direct                                                                       # the PC may enumerate again at any time
    r.panic(True)
    r.run(30)
    r.panic(False)
    assert state(r)[0] == "PASSTHRU"
    assert SimPC.enumerate(r.front) == direct                                                    # a soft latch changes nothing the PC can read
    r.module = None
    r.run(6000)                                                                                  # the module goes silent: parameters and scene go stale
    assert SimPC.enumerate(r.front) == direct


def test_the_identity_the_pc_reads_is_the_mouses_own_for_any_vid_pid_and_strings():
    seen = set()
    for vid, pid, strings in ((0x046D, 0xC077, None), (0x1532, 0x0043, {1: "Razer", 2: "DeathAdder", 3: "XX123"}), (0x04F2, 0x0939, {1: "Ünïcode ✓", 2: "Мышь", 3: ""}),
                              (0x0001, 0xFFFF, {1: "a" * 30, 2: "b", 3: "c"})):
        asc, trm = fixed_params()
        r = Rig(SimMouse("m16", vid=vid, pid=pid, strings=strings), asc=asc, tremor=trm)
        r.run(500)
        assert r.engage_fast()
        dev = r.transcripts["bridge"][1][7]
        assert struct.unpack_from("<HH", dev, 8) == (vid, pid) and r.transcripts["bridge"] == r.transcripts["direct"]
        seen.add((vid, pid))
        assert r.status().vid == vid and r.status().pid == pid                                   # the core knows it (for the module over SPI), the PC reads no other
    assert len(seen) == 4


# ------------------------------------------------------------------------------------------------------------ T3: not options
def test_cloning_and_the_live_proxy_are_not_options():
    """The core's whole configuration is below. Nothing in it can switch the clone or the proxy off, replace an identity or add a descriptor;
    a new field has to be classified here."""
    fields = {n for n, _ in CCfg._fields_}
    assert fields == {"scope", "chord_mask", "chord_ms", "panic_long_ms", "rearm_ms", "engage_hold_ms", "probe_ms", "pc_cfg_ms", "auto_engage",
                      "soft_hold_ms", "param_ttl_ms", "scene_ttl_ms", "link_ttl_ms", "slow_poll_us", "split_poll_us", "lockin_ratio_q16",
                      "lockin_min_counts", "lockin_hold_ms", "budget_us", "overrun_limit", "status_period_ms", "domain_counts"}
    assert not any(re.search(r"clone|proxy|vid|pid|desc|string|usage|feature", n, re.I) for n in fields)


def test_the_proxy_routing_follows_the_rule_for_any_request():
    """Specification: SET_ADDRESS is local; GET_DESCRIPTOR for the device, configuration 0, and the HID class and report descriptors of a HID
    interface is served from the image; EVERYTHING ELSE is forwarded to the mouse. Checked against the C core on thousands of random requests."""
    r = fresh("composite")
    ifaces = {f.num for f in r.mouse.ifaces if f.cls == 3 and f.alt == 0}
    rng = random.Random(5)
    bms = [0x00, 0x01, 0x02, 0x20, 0x21, 0x22, 0x40, 0x41, 0x80, 0x81, 0x82, 0xA0, 0xA1, 0xC0, 0xC1, 0xE0, rng.randrange(256)]
    served = 0
    for _ in range(6000):
        bm, req = rng.choice(bms), rng.choice((0, 1, 2, 3, 5, 6, 6, 6, 8, 9, 10, 11, 0x0C, 0xFE, rng.randrange(256)))
        value = rng.choice((0x0100, 0x0200, 0x0201, 0x0300, 0x0301, 0x03EE, 0x0600, 0x0F00, 0x2100, 0x2200, 0x2201, 0x2300, rng.randrange(65536)))
        index, length = rng.choice((0, 1, 2, 3, 4, 0x0409, rng.randrange(65536))), rng.choice((0, 2, 8, 9, 18, 255, 4096, rng.randrange(65536)))
        if rng.random() < 0.4:                                                                   # aim at the descriptors the cache serves, so that both sides are well covered
            bm, req = rng.choice((0x80, 0x81)), 6
            value = rng.choice((0x0100, 0x0200, 0x2100, 0x2200))
            index = rng.choice(sorted(ifaces) + [7])
        want = PX_FORWARD
        if bm == 0x00 and req == 5:
            want = PX_LOCAL
        elif req == 6 and (value & 0xFF) == 0:
            if bm == 0x80 and (value >> 8) in (1, 2):
                want = PX_SERVE
            elif bm == 0x81 and (value >> 8) in (0x21, 0x22) and (index & 0xFF) in ifaces:
                want = PX_SERVE
        kind, data = r.b.pc_setup(bm, req, value, index, length)
        assert kind == want, (bm, req, value, index)
        if kind == PX_SERVE:
            served += 1
            assert len(data) <= length
    assert served > 800


@pytest.mark.parametrize("kw", [dict(bulk=True), dict(n_conf=2), dict(alt=True), dict(kind="abs"), dict(kind="vendor")])
def test_devices_the_proxy_can_not_mirror_exactly_are_never_engaged(kw):
    """The other half of 'mandatory': if the clone can not be exact the bridge stays out of the way - the mouse works directly, unassisted."""
    asc, trm = fixed_params()
    r = Rig(SimMouse(**{"kind": "m16", **kw}), asc=asc, tremor=trm)
    r.run(8000)
    assert state(r)[0] in ("HW_BYPASS", "PROBE") and r.route == "bypass" and r.pc_conn == "direct"
    assert all(kind == "direct" for _, kind, _ in r.pc_enumerations)                             # the PC never met anything but the real mouse


# ------------------------------------------------------------------------------------------------------------ T4/T5: X/Y only, only down
def pushed(r, ep, raw):
    r.mouse.push_in(ep, raw)
    r.step()


@pytest.mark.parametrize("name", ["m16", "logi", "composite", "hs"])
def test_only_xy_bits_change_and_only_downwards_for_any_report(name):
    """Random reports with random buttons, wheel, pan and every motion value, in every state of the bridge (assist, soft latch, split)."""
    r = with_scene(fresh(name, persona="tremor"))
    _, m = hid_parse(r.mouse.rds[0])
    f = m.m[0]
    kind = "logi" if name == "composite" else name if name != "hs" else "m16"
    mask = c_mask(f, f.len)
    rng = np.random.default_rng(8)
    lim = r.mouse.max_count
    n_changed = 0
    for phase in range(3):
        if phase == 1:
            r.panic(True)
            r.run(30)
            r.panic(False)
        if phase == 2:
            r.panic(True)
            r.run(2100)
            r.panic(False)                                                                       # back to ASSIST (a deliberate hold)
            r.run(100)
        r.pc_reports.clear()
        for _ in range(500):
            dx, dy = int(np.clip(rng.normal(0, 20), -lim, lim)), int(np.clip(rng.normal(0, 20), -lim, lim))
            r.move(dx, dy, int(rng.integers(0, 32)) if kind == "logi" else int(rng.integers(0, 8)), int(rng.integers(-3, 4)), int(rng.integers(-3, 4)))
            r.step()
        reps = [x for x in r.pc_reports if x[1] == "bridge" and x[2] == r.mouse.motion_ep]
        assert len(reps) == 500
        for _, _, _, out, raw in reps:
            assert len(out) == len(raw) == f.len
            assert all((a ^ b) & ~mk & 0xFF == 0 for a, b, mk in zip(raw, out, mask))            # buttons, wheel, pan, Report ID, vendor bits: as they came
            n_changed += out != raw
            for o, i in zip(_xy(kind, out, f), _xy(kind, raw, f)):
                assert ok_axis(o, i)
        if phase == 1:
            assert all(x[3] == x[4] for x in reps)                                               # the soft latch: bit for bit
    assert n_changed > 0                                                                          # and the help really edited something


def _xy(kind, report, f):
    if kind == "m16":
        return struct.unpack_from("<hh", report, 1)
    return U.get_bits(report, f.off_x, f.size_x, True), U.get_bits(report, f.off_y, f.size_y, True)


def test_every_other_interface_is_copied_unchanged():
    r = fresh("composite")
    hidpp_ep, kbd_ep = 0x82, 0x83                                                                # the composite mouse's HID++-shaped channel and keyboard
    rng = random.Random(2)
    sent = []
    for _ in range(400):
        ep, n = rng.choice(((hidpp_ep, 20), (kbd_ep, 8)))
        raw = bytes(rng.randrange(256) for _ in range(n))
        sent.append((ep, raw))
        pushed(r, ep, raw)
        if rng.random() < 0.5:
            r.move(rng.randrange(-20, 20), rng.randrange(-20, 20))                               # mixed with motion on the other interface
            r.step()
    got = [(x[2], x[3], x[4]) for x in r.pc_reports if x[2] in (hidpp_ep, kbd_ep)]
    assert len(got) == len(sent) and [(a, c) for a, b, c in got] == sent
    assert all(out == raw for _, out, raw in got)                                                # byte for byte, in order, none added, none lost
    s = r.status()
    assert s.other_reports >= 400 and s.invariant_viol == 0


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_one_report_out_for_every_report_in_through_every_state(scenario):
    name = "hs" if scenario == "split" else "logi"
    r = with_scene(fresh(name))
    n_in = 0
    if scenario == "soft_panic":
        r.panic(True)
        r.run(30)
        r.panic(False)
    if scenario == "silent_module":
        r.module = None
    r.pc_reports.clear()
    rng = np.random.default_rng(1)
    for k in range(3000):
        if rng.random() < 0.7:
            r.move(int(rng.normal(0, 8)), int(rng.normal(0, 8)), int(rng.integers(0, 16)))
            n_in += 1
        r.step()
        if scenario == "silent_module" and k == 1500:
            assert state(r)[0] == "PASSTHRU"                                                     # it fell to the pass-through by itself
    reps = [x for x in r.pc_reports if x[1] == "bridge"]
    assert len(reps) == n_in > 1000 and all(len(x[3]) == len(x[4]) for x in reps)


# ------------------------------------------------------------------------------------------------------------ T7/T8: bypass and the bus
def hidden_ids(r):
    """(VID, PID, serial index) of every device descriptor the PC read, in order, from the transcripts of all its enumerations."""
    out = []
    for _, kind, tr in r.pc_enumerations:
        dev = next(x for x in tr if x[0] == 0x80 and x[1] == 6 and x[2] == 0x0100 and x[4] == 18)[7]
        out.append((kind, struct.unpack_from("<HH", dev, 8), dev))
    return out


def test_the_pc_only_ever_enumerates_the_identity_of_the_real_mouse():
    """A hostile day: engage, long press, power loss, a hung firmware, a crash loop, a swapped mouse. Whatever happens the PC only ever
    reads descriptors that the real mouse has."""
    asc, trm = fixed_params()
    m = SimMouse("logi", hidpp=True)
    r = Rig(m, asc=asc, tremor=trm)
    r.run(500)
    assert r.engage_fast()
    r.panic(True)
    r.run(3100)
    r.panic(False)
    r.run(800)
    r.panic(True)
    r.run(2100)
    r.panic(False)
    assert r.engage_fast()
    r.kill_firmware()
    r.run(400)
    r.reboot_firmware()
    r.run(1500)
    r.power_off()
    r.run(800)
    ids = hidden_ids(r)
    kinds = [k for k, _, _ in ids]
    assert "direct" in kinds and "bridge" in kinds and len(ids) >= 5
    assert {i for _, i, _ in ids} == {(0x046D, 0xC077)}
    assert {bytes(d) for _, _, d in ids} == {bytes(m.dev)}
    assert all(tr == r.pc_enumerations[0][2] for _, kind, tr in r.pc_enumerations)               # every enumeration, direct or via the bridge, is the same transcript


def test_hard_bypass_is_the_real_mouse_again_with_the_same_transcript():
    r = fresh("composite")
    first = r.transcripts["direct"]
    r.panic(True)
    r.run(3100)
    r.panic(False)
    assert state(r)[0] == "HW_BYPASS" and r.route == "bypass"
    r.run(500)
    assert r.pc_conn == "direct" and r.transcripts["direct"] == first == r.transcripts["bridge"]
    r.pc_reports.clear()
    for _ in range(50):
        r.move(5, 2)
        r.step()
    assert len(r.pc_reports) == 50 and all(x[1] == "direct" and x[3] == x[4] for x in r.pc_reports)   # untouched: the mouse's own reports
    r.power_off()
    r.run(500)
    assert r.route == "bypass" and r.pc_conn == "direct"                                         # and without power: also the mouse itself


def test_soft_transitions_leave_no_trace_on_the_bus():
    r = fresh("logi")
    n_events = len([e for e in r.events if e[1].startswith("route")])
    n_enum = len(r.pc_enumerations)
    for _ in range(3):
        r.panic(True)
        r.run(30)
        r.panic(False)                                                                           # soft latch on
        r.run(200)
        r.panic(True)
        r.run(2100)
        r.panic(False)                                                                           # and off again
        r.run(200)
    r.module = None
    r.run(7000)                                                                                  # stale parameters: down to pass-through by itself
    assert state(r)[0] == "PASSTHRU"
    assert len([e for e in r.events if e[1].startswith("route")]) == n_events                   # no route change = no detach, no reset, no new enumeration
    assert len(r.pc_enumerations) == n_enum and r.pc_conn == "bridge"


# ------------------------------------------------------------------------------------------------------------ T9: the document
def test_every_property_names_tests_that_exist():
    names = set(re.findall(r"^def (test_\w+)", Path(__file__).read_text(encoding="utf-8"), re.M))
    names |= set(re.findall(r"^def (test_\w+)", (ROOT / "tests" / "test_bridge.py").read_text(encoding="utf-8"), re.M))
    names |= set(re.findall(r"^def (test_\w+)", (ROOT / "tests" / "test_bridge_failsafe.py").read_text(encoding="utf-8"), re.M))
    for p in T.PROPS:
        assert p.tests, p.id
        for t in p.tests:
            assert t in names, (p.id, t)
    assert [p.id for p in T.PROPS] == [f"T{i}" for i in range(1, 10)]
    assert {p.status for p in T.PROPS} >= {"sim", "code", "model"}


def test_the_document_is_current_and_names_the_boundaries():
    text = DOC.read_text(encoding="utf-8")
    block = re.compile(r"(<!-- fp:(\w+) -->)\n?(.*?)\n?(<!-- /fp:\2 -->)", re.S)
    rendered = block.sub(lambda m: m.group(1) + "\n" + T.TABLES[m.group(2)]() + "\n" + m.group(4), text)
    assert rendered == text, "run `dataopen bridge transparency --write`"
    assert {m.group(2) for m in block.finditer(text)} == set(T.TABLES)
    for h in ("## 0. Честный статус", "## 1. Свойства", "## 2. Что делает мост", "## 3. Чего ПК не видит", "## 4. Что ПК может заметить",
              "## 5. Тесты", "## 6. Стыковка с fail-safe", "## 7. Что не сделано"):
        assert h in text, h
    for honest in ("перенумерация", "LS", "каскад", "не доказывает", "не скрывает", "анализатор USB", "не проверено"):
        assert honest in text or honest in " ".join(" ".join(r) for r in T.VISIBLE), honest
    assert len(T.VISIBLE) >= 8 and len(T.INVISIBLE) >= 6
