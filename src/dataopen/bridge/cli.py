"""`dataopen bridge ...`: simulate the rig (fail-safe timeline, closed loop through the bridge), time and size the core, build it."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 4


def _params(persona: str, seed: int = 1):
    from ..assist.fixed import FixedParams
    from ..assist.params import AscParams
    from ..assist.sim_user import PERSONAS, build_profile
    from ..assist.tremor import TremorParams
    from ..assist.tremor_fixed import FixedTremorParams

    v = build_profile(PERSONAS[persona], seed=seed)
    return (
        AscParams.from_view(v),
        TremorParams.from_view(v),
        FixedParams.from_params(AscParams.from_view(v)),
        FixedTremorParams.from_params(TremorParams.from_view(v)),
    )


def _timeline(a) -> int:
    from .cbridge import REASONS, STATE
    from .sim import Rig
    from .sim_usb import SimMouse

    _, _, fa, ft = _params("overshooter")
    r = Rig(SimMouse(a.mouse), asc=fa, tremor=ft)
    rows: list[tuple[float, str]] = []

    def note(msg: str) -> None:
        s = r.status()
        st, why = (STATE[s.state], REASONS[s.reason]) if r.alive else ("(dead)", "-")        # a hung or unpowered firmware has no state
        rows.append((r.t / 1e6, f"{msg:44s} state={st:10s} reason={why:14s} route={r.route}"))

    def feed(ms: int) -> None:
        for _ in range(ms):
            r.move(3, 1)
            r.step()

    note("power on, mouse plugged (hardware bypass)")
    r.engage_fast()
    note("engaged: the PC enumerated the bridge")
    feed(300)
    r.panic(True)
    note("PANIC pressed (short)")
    feed(200)
    r.panic(False)
    feed(200)
    note("PANIC released: latched, no gap")
    r.panic(True)
    r.run(2100)
    r.panic(False)
    r.engage_fast()
    note("re-armed by a deliberate 2 s hold")
    feed(200)
    r.kill_firmware()
    r.run(150)
    note("firmware hung -> watchdog -> bypass")
    r.reboot_firmware(1)
    r.engage_fast(30000)
    note("firmware rebooted, engaged again")
    r.module.silent = True
    r.run(800)
    note("module silent for 800 ms")
    r.module.silent = False
    r.module.next_params = 0
    r.run(2000)
    note("module back")
    r.power_off()
    r.run(5)
    note("power lost -> bypass")
    if a.json:
        print(json.dumps([{"t_s": t, "event": m} for t, m in rows], indent=2))
    else:
        print(f"(re-enumeration after a switch is modelled as {r.reenum_us / 1000:.0f} ms: an ASSUMPTION; real PCs need 0.2 - 2 s)")
        for t, m in rows:
            print(f"{t:8.3f} s  {m}")
    return EXIT_OK


def _persona(a) -> int:
    import numpy as np

    from ..assist.sim_user import PERSONAS, run_trial, summarize
    from .sim import BridgeAssist

    names = list(PERSONAS) if a.persona == "all" else [a.persona]
    out = {}
    for n in names:
        ap, tp, _, _ = _params(n, a.seed)
        base, helped, trips = [], [], 0
        for i in range(a.trials):
            base.append(run_trial(PERSONAS[n], None, np.random.default_rng(i)))
            ba = BridgeAssist(ap, tp)
            helped.append(run_trial(PERSONAS[n], ba, np.random.default_rng(i)))
            trips += ba.b.status(ba.rig.t).lockin_trips > 0
        out[n] = {
            "without": summarize(base, 30.0),
            "through the bridge": summarize(helped, 30.0),
            "lock-in backstop fired in trials": trips,
        }
    if a.json:
        print(json.dumps(out, indent=2))
        return EXIT_OK
    cols = ("acquired", "t_acquire_ms", "overshoot_rate", "hold_rms_px", "final_err_px", "k_mean")
    print(f"{'persona':12s} {'':20s} " + " ".join(f"{c:>14s}" for c in cols))
    for n, r in out.items():
        for lab in ("without", "through the bridge"):
            print(f"{n:12s} {lab:20s} " + " ".join(f"{r[lab][c]:14.2f}" for c in cols))
        print(f"{'':12s} lock-in backstop fired in {r['lock-in backstop fired in trials']} of {a.trials} trials")
    return EXIT_OK


def _simulate(a) -> int:
    if a.what in ("failsafe", "all"):
        rc = _timeline(a)
        if rc:
            return rc
    if a.what in ("persona", "all"):
        return _persona(a)
    return EXIT_OK


def _build_exe(tmp: Path, exe_name: str, main_c: Path, extra: list[str]) -> Path:
    from ..assist.fixed import FixedParams
    from ..assist.tremor_fixed import FixedTremorParams
    from .cbridge import ASC_CSRC, CSRC, SOURCES
    from .seeds import write_seeds_header

    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        raise RuntimeError("no C compiler found")
    _, _, fa, ft = _params("overshooter")
    write_seeds_header(tmp / "seeds.h", fa if fa else FixedParams(), ft if ft else FixedTremorParams())
    exe = tmp / exe_name
    r = subprocess.run(
        [cc, "-O2", "-std=gnu99", f"-I{CSRC}", f"-I{ASC_CSRC}", f"-I{tmp}", str(main_c), *map(str, SOURCES), "-o", str(exe), *extra],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:1000])
    return exe


def _bench(a) -> int:
    from .cbridge import CSRC

    with tempfile.TemporaryDirectory() as d:
        try:
            exe = _build_exe(Path(d), "bridge_bench", CSRC / "bench_main.c", [])
        except RuntimeError as e:
            print(f"error: {e}")
            return EXIT_FAILED
        r = subprocess.run([str(exe), str(a.n)], capture_output=True, text=True)
        print(r.stdout, end="")
        print("(host numbers; a Cortex-M7 at 600 MHz is several times slower per call: measure on the target)")
        return EXIT_OK if r.returncode == 0 else EXIT_FAILED


def _size(a) -> int:
    from .cbridge import ASC_CSRC, CSRC

    clang = shutil.which("clang")
    size = shutil.which("llvm-size")
    if not clang or not size:
        print("error: needs clang (with the thumb backend) and llvm-size")
        return EXIT_FAILED
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        shim = tmp / "shim"
        shim.mkdir()
        (shim / "string.h").write_text(
            "#include <stddef.h>\nvoid *memcpy(void *, const void *, size_t);\nvoid *memset(void *, int, size_t);\n"
            "void *memmove(void *, const void *, size_t);\n"
        )
        total = [0, 0, 0]
        srcs = [CSRC / n for n in ("hid_desc.c", "usb_image.c", "usb_proxy.c", "link.c", "failsafe.c", "bridge.c")] + [
            ASC_CSRC / "asc_core.c",
            ASC_CSRC / "tremor_core.c",
        ]
        print(f"{'file':16s} {'text':>8s} {'data':>6s} {'bss':>6s}   (-Os, thumbv7em, cortex-m7)")
        for s in srcs:
            o = tmp / (s.name + ".o")
            r = subprocess.run(
                [
                    clang,
                    "--target=thumbv7em-none-eabihf",
                    "-mcpu=cortex-m7",
                    f"-O{a.opt}",
                    "-ffreestanding",
                    "-std=c99",
                    f"-isystem{shim}",
                    f"-I{CSRC}",
                    f"-I{ASC_CSRC}",
                    "-c",
                    str(s),
                    "-o",
                    str(o),
                ],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                print(f"error: {s.name}: {r.stderr[:300]}")
                return EXIT_FAILED
            line = subprocess.run([size, "--format=berkeley", str(o)], capture_output=True, text=True).stdout.splitlines()[-1].split()
            t, dd, bb = int(line[0]), int(line[1]), int(line[2])
            total = [total[0] + t, total[1] + dd, total[2] + bb]
            print(f"{s.name:16s} {t:8d} {dd:6d} {bb:6d}")
        print(f"{'total':16s} {total[0]:8d} {total[1]:6d} {total[2]:6d}")
        from .cbridge import CBridge

        print(
            f"RAM of one bridge instance (host layout): {CBridge().sizeof} bytes (telemetry ring 32 KB, USB image 8 KB, position ring 6 KB)"
        )
    return EXIT_OK


def _build(a) -> int:
    from .cbridge import build_library

    print(build_library(a.out))
    return EXIT_OK


DOC = Path(__file__).resolve().parents[3] / "docs" / "TRANSPARENCY.md"


def _transparency(a) -> int:
    """The properties of OS transparency, what the PC can and can not notice, and the freshness of docs/TRANSPARENCY.md."""
    import re

    from . import transparency as T

    block = re.compile(r"(<!-- fp:(\w+) -->)\n?(.*?)\n?(<!-- /fp:\2 -->)", re.S)

    def render(text: str) -> str:
        return block.sub(lambda m: m.group(1) + "\n" + T.TABLES[m.group(2)]() + "\n" + m.group(4), text)

    path = Path(a.path) if a.path else DOC
    if a.check or a.write:
        text = path.read_text(encoding="utf-8")
        current = render(text) == text and {m.group(2) for m in block.finditer(text)} == set(T.TABLES)
        if a.check:
            print("current" if current else f"{path} is stale: run `dataopen bridge transparency --write`")
            return EXIT_OK if current else EXIT_FAILED
        path.write_text(render(text), encoding="utf-8")
        print(f"written: {path}")
        return EXIT_OK
    if a.json:
        out = {"properties": [vars(p) for p in T.PROPS], "invisible": T.INVISIBLE, "visible": T.VISIBLE}
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(T.props_table() + "\n\n" + T.invisible_table() + "\n\n" + T.visible_table())
    return EXIT_OK


def register(sub) -> None:
    s = sub.add_parser("bridge", help="Assistive HID bridge: simulate / bench / size / build (docs/BRIDGE.md)")
    ss = s.add_subparsers(dest="bridge_cmd", required=True)
    a = ss.add_parser(
        "simulate", help="fail-safe timeline of the rig, and the closed-loop reach task through the bridge, on simulated people"
    )
    a.add_argument("what", nargs="?", default="all", choices=["all", "failsafe", "persona"])
    a.add_argument("--mouse", default="logi", choices=["boot", "m16", "logi"])
    a.add_argument("--persona", default="all", choices=["all", "steady", "overshooter", "tremor"])
    a.add_argument("--trials", type=int, default=20)
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--json", action="store_true")
    a.set_defaults(fn=_simulate)
    b = ss.add_parser("bench", help="time per mouse report / link frame of the C core (host numbers)")
    b.add_argument("-n", type=int, default=300000)
    b.set_defaults(fn=_bench)
    c = ss.add_parser("size", help="code size of the whole chain cross-compiled for a Cortex-M7 (needs clang with the thumb backend)")
    c.add_argument("--opt", default="s", choices=["s", "2", "z"])
    c.set_defaults(fn=_size)
    d = ss.add_parser("build", help="compile the C core with the system compiler")
    d.add_argument("--out")
    d.set_defaults(fn=_build)
    t = ss.add_parser("transparency", help="OS transparency: the properties and what the PC can and can not notice (docs/TRANSPARENCY.md)")
    t.add_argument("--json", action="store_true")
    t.add_argument("--check", action="store_true", help="is docs/TRANSPARENCY.md current")
    t.add_argument("--write", action="store_true", help="refresh the generated tables in docs/TRANSPARENCY.md")
    t.add_argument("--path")
    t.set_defaults(fn=_transparency)
