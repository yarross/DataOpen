"""The end-to-end latency budget of the assistive path (docs/LATENCY.md): stages, numbers, and where each number comes from.

Three groups, kept apart on purpose (summing them would hide what they are):

    A  the INPUT path: the person moves the mouse -> the report the PC receives. Every ms here is felt as a pointer that lags.
    B  the SCENE path: an object is on the cable -> the bridge knows it (as a target, captured at a known time). It does not delay the
       pointer; it decides how early the help can be there. The bridge compensates the pointer's motion since the capture, and drops a
       scene older than `scene_ttl_ms`.
    C  the POLICY: how fast the help is allowed to act once it can (guard, ramp, slew). Chosen smoothness, not pipeline latency.

Numbers are never typed twice: video timing, tracker/ASC/bridge constants are read from the modules that own them. Each stage carries the
BASIS of its number (`BASES`): what was computed from code, what was measured in the simulation, what is a documented host measurement,
what is an estimate and what has not been measured at all."""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass, replace
from typing import Optional

from ..assist.params import AscConfig
from ..runtime.frames import QueueSource
from ..ui.scene import SceneConfig
from ..video import timing as VT
from . import measure as M

BASES = {
    "calc": "расчёт из констант кода",
    "sim": "измерено в симуляции (виртуальное время)",
    "host": "замер на хосте по документации",
    "estimate": "оценка (рассуждение, не замер)",
    "unmeasured": "не измерено: зависит от железа, цифра это граница допущения",
}
FLOORS = {
    "physical": "физический нижний предел (скорость развёртки, протокол USB)",
    "arch": "ограничение архитектуры (можно изменить)",
    "hw": "зависит от железа",
    "none": "",
}
GROUPS = {"A": "ввод: мышь → отчёт в ПК", "B": "сцена: объект на кабеле → мост знает цель", "C": "политика: как быстро помощь вправе действовать"}

# inference of the UI network, ms (p50, p95): the CPU numbers are a documented host measurement (docs/UIDET.md, ONNX Runtime, 4 threads);
# the NPU numbers are a TARGET: nothing was run on an NPU
INFER_MS = {"cpu": (11.0, 16.0), "npu": (3.0, 4.0)}
UPDATE_MS = 0.3  # decode + NMS + tracker: an estimate (the host measurement is `UiService.trace.update_ms`)
CAPTURE_TAIL_MS = {"stream": 0.02, "frame": 0.5}  # work after the last needed line: VIDEO.md (host 9-34 us; whole-frame path 0.3..1 ms)
SPI_FRAME_BYTES, SPI_HZ_ASSUMED = 128, 10_000_000  # the link frame size is in code (link.h); the SPI clock is NOT defined anywhere
SPI_WAIT_MS = (0.0, 1.0, 2.0)  # how long a scene frame waits for the SPI master's turn: nothing in code defines it (no DRDY yet)
BRIDGE_TICK_MS = 1.0  # `bridge_poll` is called every ~1 ms (bridge.h)
SPLIT_WINDOW_MS = 1.0  # SPLIT: the chain ticks on a 1 ms grid over the reports of a window; each report uses the previous window's gain


@dataclass(frozen=True)
class Stage:
    id: str
    group: str
    name: str
    lo: float
    typ: float
    hi: float
    basis: str
    floor: str = "none"
    reduce: str = ""
    ref: str = ""

    def __post_init__(self) -> None:
        if self.basis not in BASES or self.floor not in FLOORS or self.group not in GROUPS:
            raise ValueError(self.id)
        if not self.lo <= self.typ <= self.hi:
            raise ValueError(f"{self.id}: lo <= typ <= hi must hold ({self.lo}, {self.typ}, {self.hi})")


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    mode: str = "1920x1080@144"
    capture: str = "stream"  # 'stream': the ROI is prepared while it arrives (VIDEO.md variant B); 'frame': the whole frame, then the work
    detector: str = "cpu"  # 'cpu' | 'npu'
    queue: str = "fifo"  # 'fifo' (as built) | 'latest' (UiService(latest_only=True))
    confirm_hits: Optional[int] = None  # None: the tracker's default (SceneConfig.confirm_hits)
    mouse_hz: float = 1000.0
    tremor_filter: bool = True  # the profile has the tremor stage switched on


SCENARIOS = (
    Scenario("as-built", "как собрано: CPU-детектор, очередь FIFO(4), 1080p144", capture="frame"),
    Scenario("cpu-latest", "CPU-детектор, только свежий кадр", capture="stream", queue="latest"),
    Scenario("npu-latest", "цель: NPU-детектор, только свежий кадр", capture="stream", detector="npu", queue="latest"),
    Scenario("npu-latest-c1", "NPU, свежий кадр, подтверждение за 1 кадр", capture="stream", detector="npu", queue="latest", confirm_hits=1),
    Scenario("cpu-60", "CPU-детектор, FIFO(4), 1080p60", mode="1920x1080@60", capture="frame"),
    Scenario("npu-4k60", "NPU, свежий кадр, 4K60", mode="3840x2160@60", capture="stream", detector="npu", queue="latest"),
    Scenario("slow-mouse", "мышь 125 Гц (SLOW), всё остальное — цель", detector="npu", queue="latest", mouse_hz=125.0),
)
BY_KEY = {s.key: s for s in SCENARIOS}


# ------------------------------------------------------------------------------------------------ constants read from their owners
def queue_capacity() -> int:
    return inspect.signature(QueueSource.__init__).parameters["capacity"].default


@functools.lru_cache(maxsize=None)
def _bridge_cfg() -> tuple:
    return tuple(sorted(_read_cfg().items()))


def bridge_cfg() -> dict:
    return dict(_bridge_cfg())


def _read_cfg() -> dict:
    """The bridge core's defaults, read from the C struct (needs the C compiler the tests use anyway); the literals are only a fallback."""
    try:
        from ..bridge.cbridge import CBridge

        c = CBridge().cfg
        return {k: int(getattr(c, k)) for k in ("scene_ttl_ms", "slow_poll_us", "split_poll_us", "budget_us", "overrun_limit")}
    except Exception:  # no C compiler: the same numbers as bridge.c, marked by the caller
        return {"scene_ttl_ms": 100, "slow_poll_us": 4000, "split_poll_us": 900, "budget_us": 300, "overrun_limit": 3}


def roi_for(sc: Scenario) -> tuple[VT.VideoMode, VT.Roi]:
    m = VT.mode(sc.mode)
    if sc.capture == "frame":
        return m, VT.Roi(0, 0, m.h_active, m.v_active)
    side = 640
    return m, VT.Roi(max(0, (m.h_active - side) // 2), max(0, (m.v_active - side) // 2), min(side, m.h_active), min(side, m.v_active))


def usb_mode(mouse_hz: float, cfg: Optional[dict] = None) -> str:
    """The bridge's choice from the mouse's poll interval (compute_mode in bridge.c): DIRECT / SPLIT / SLOW."""
    cfg = cfg or bridge_cfg()
    period_us = 1e6 / mouse_hz
    return "SLOW" if period_us > cfg["slow_poll_us"] else ("SPLIT" if period_us < cfg["split_poll_us"] else "DIRECT")


@functools.lru_cache(maxsize=None)
def _t_lo_ms() -> tuple[float, float]:
    """(the 'tremor' persona's, the largest among the personas') window T_lo in ms, from the profiles the simulator builds."""
    from ..assist.params import AscParams
    from ..assist.sim_user import PERSONAS, build_profile

    v = {k: AscParams.from_view(build_profile(p, seed=1)).t_lo_us / 1000.0 for k, p in PERSONAS.items()}
    return v["tremor"], max(v.values())


# ------------------------------------------------------------------------------------------------ stages
@functools.lru_cache(maxsize=None)
def _components(sc: Scenario, infer_ms: float, hits: int) -> dict[str, list[float]]:
    m = VT.mode(sc.mode)
    return M.pipe_timeline(M.Pipe(m.frame_time_us / 1000.0, infer_ms, UPDATE_MS, sc.queue, queue_capacity(), hits))


def _span(vals: list[float], hi_vals: list[float]) -> tuple[float, float, float]:
    return min(vals), sum(vals) / len(vals), max(hi_vals)


@functools.lru_cache(maxsize=None)
def _poll(hz: float) -> dict:
    return M.poll_resample(hz)["added_ms"]


@functools.lru_cache(maxsize=None)
def _chain() -> dict:
    return M.chain_delay("tremor")["delay_ms"]


def stages(sc: Scenario) -> list[Stage]:
    return list(_stages(sc))


@functools.lru_cache(maxsize=None)
def _stages(sc: Scenario) -> tuple:
    hits = sc.confirm_hits if sc.confirm_hits is not None else SceneConfig().confirm_hits
    cfg = bridge_cfg()
    m, roi = roi_for(sc)
    tail = CAPTURE_TAIL_MS[sc.capture]
    age = VT.data_age_us(m, roi)  # without the tail: that is B2
    b1 = age["age_of_content_us"] / 1000.0  # an object in the middle of the ROI
    b1_hi = (age["ready_us"] - m.row_done_us(roi.y)) / 1000.0  # an object in the first row of the ROI
    p50, p95 = INFER_MS[sc.detector]
    typ, hi = _components(sc, p50, hits), _components(sc, p95, hits)
    npu = sc.detector == "npu"
    out: list[Stage] = [
        Stage("B1", "B", "объект на кабеле → область (ROI) готова", 0.0, b1, b1_hi, "calc", "physical",
              "выбрать ROI выше по экрану или режим с большей частотой; нижнего предела нет: строки приходят со скоростью развёртки",
              "video.timing.data_age_us: возраст середины ROI к моменту готовности (верхняя граница: объект в первой строке ROI)"),
        Stage("B2", "B", "работа после последней строки (RGA/подготовка)", tail / 2, tail, tail * 2, "estimate", "hw",
              "потоковая подготовка строка за строкой оставляет только хвост одной строки", "VIDEO.md §5: хвост 9-34 мкс на хосте; весь кадр 0.3-1 мс"),
        Stage("B3", "B", "ожидание детектора (очередь)", *_span(typ["wait"], hi["wait"]), "calc", "arch",
              "брать только свежий кадр (UiService(latest_only=True)) и не копить кадры; убрать очередь глубже одного кадра",
              f"модель очереди: {sc.queue}, ёмкость {queue_capacity()}, кадр {m.frame_time_us / 1000:.2f} мс (runtime.frames.QueueSource)"),
        Stage("B4", "B", "вывод сети (инференс + decode)", p50, p50, p95, "unmeasured" if npu else "host", "hw",
              "NPU вместо CPU; меньший вход; сначала область вокруг указателя" if not npu else "цель NPU; нужен замер на RK3588",
              "docs/UIDET.md §2: ONNX Runtime CPU p50 11 мс, p95 16 мс" if not npu else "цель, не замер: для этой сети на NPU ничего не запускалось"),
        Stage("B5", "B", "трекер (сопоставление, сглаживание)", UPDATE_MS / 3, UPDATE_MS, UPDATE_MS * 3, "estimate", "none",
              "ничего существенного", "ui.service.LatencyTrace.update_ms покажет реальное значение на плате"),
        Stage("B6", "B", f"подтверждение цели (confirm_hits={hits})", *_span(typ["confirm"], hi["confirm"]), "calc", "arch",
              "подтверждать за 1 кадр для крупных целей или передавать цель сразу с меткой «не подтверждена»",
              "ui.scene.SceneConfig.confirm_hits: цель нужна в нескольких обработанных кадрах подряд"),
        Stage("B7", "B", "кадр сцены → мост (SPI)", SPI_FRAME_BYTES * 8 / SPI_HZ_ASSUMED * 1000, SPI_FRAME_BYTES * 8 / SPI_HZ_ASSUMED * 1000 + SPI_WAIT_MS[1],
              SPI_FRAME_BYTES * 8 / SPI_HZ_ASSUMED * 1000 + SPI_WAIT_MS[2], "unmeasured", "hw",
              "линия DRDY от моста: мастер передаёт сразу, а не по таймеру",
              f"кадр {SPI_FRAME_BYTES} байт (link.h); частота SPI и ожидание очереди в коде НЕ определены: допущено {SPI_HZ_ASSUMED // 1_000_000} МГц, 0-{SPI_WAIT_MS[2]:g} мс"),
        Stage("B8", "B", "сцена принята ядром моста", 0.0, BRIDGE_TICK_MS / 2, BRIDGE_TICK_MS, "estimate", "hw",
              "обрабатывать кадр в прерывании SPI, не по сетке 1 мс", "bridge.h: bridge_poll вызывается раз в ~1 мс; зависит от порта (прерывание или таймер)"),
    ]
    mouse_us = 1e6 / sc.mouse_hz
    mode = usb_mode(sc.mouse_hz, cfg)
    poll = _poll(sc.mouse_hz)
    split = SPLIT_WINDOW_MS if mode == "SPLIT" else 0.0
    chain_hi = cfg["budget_us"] / 1000.0
    out += [
        Stage("A1", "A", "цепочка ASC + тремор в ядре (вычисление)", 0.0006, 0.003, chain_hi, "estimate", "hw",
              "ничего: запас по `budget_us`; превышение переводит в PASSTHRU", "BRIDGE.md §7: хост 0.6 мкс; Cortex-M не измерен; верхняя граница = budget_us ядра"),
        Stage("A2", "A", "окно SPLIT (быстрая мышь)", 0.0, split / 2, split, "calc", "arch",
              "считать усиление по скользящему окну короче 1 мс", f"режим {mode} при опросе мыши {mouse_us:.0f} мкс (bridge.c compute_mode)"),
        Stage("A3", "A", "два опроса USB подряд (мост ← мышь, ПК ← мост)", max(0.0, poll["min"]), poll["mean"], poll["max"], "sim", "physical",
              "опрашивать мышь «точно вовремя» перед запросом ПК или быстрее собственного интервала мыши (нарушает ожидания спецификации USB)",
              f"measure.poll_resample({sc.mouse_hz:g} Гц): модель сеток опроса, случайные фазы"),
    ]
    if sc.tremor_filter:
        d = _chain()
        out.append(
            Stage("A4", "A", "фильтр тремора на намеренном движении", d[0.9], d[0.5], d[0.1], "sim", "arch",
                  "это цена подавления: меньшие постоянные времени фильтра или режим «только очень сильный тремор»; решает профиль",
                  "measure.chain_delay('tremor'): отставание выхода от входа на 90/50/10 % плавного броска 600 отсчётов (синтетический пользователь)"))
    # --- C: policy (not summed with A or B)
    a = AscConfig()
    k_floor = a.k_floor
    t_lo = _t_lo_ms()
    out += [
        Stage("C1", "C", "обнаружение начала движения (защита)", a.on_ms, a.on_ms, a.on_ms, "calc", "arch", "задаётся профилем", "AscConfig.on_ms"),
        Stage("C2", "C", "окно защиты T_lo от появления цели", a.t_min_ms, t_lo[0], t_lo[1], "calc", "arch",
              "задаётся профилем: max(100 мс, T_motor − 2σ); нужно, чтобы помощь не вмешивалась до реакции человека",
              "AscConfig.t_min_ms и t_lo_us профилей (assist.sim_user.build_profile): типичный и наибольший среди персон"),
        Stage("C3", "C", "нарастание помощи после открытия защиты", a.ramp_min_ms, a.ramp_min_ms * 1.5, a.ramp_max_ms, "calc", "arch",
              "задаётся профилем (clamp σ_T 20..80 мс)", "AscConfig.ramp_min_ms/ramp_max_ms"),
        Stage("C4", "C", "торможение: K падает не быстрее slew_per_s (до 0.5 / 0.3 / k_floor)", 0.5 / a.slew_per_s * 1000, 0.7 / a.slew_per_s * 1000,
              (1 - k_floor) / a.slew_per_s * 1000, "calc", "arch", "поднять slew_per_s: быстрее торможение, но резче для человека",
              f"AscConfig.slew_per_s={a.slew_per_s:g}/с; в симуляции (measure.scene_effect) K падает со скоростью до 6/с, на границе TTL до 7,2/с (причина не выяснена)"),
    ]
    return tuple(out)


def total(sc: Scenario, group: str) -> tuple[float, float, float]:
    ss = [s for s in stages(sc) if s.group == group]
    return sum(s.lo for s in ss), sum(s.typ for s in ss), sum(s.hi for s in ss)


def summary(sc: Scenario) -> dict:
    """The numbers a scenario is judged by."""
    cfg = bridge_cfg()
    a_lo, a_typ, a_hi = total(sc, "A")
    b_lo, b_typ, b_hi = total(sc, "B")
    a_pipe = [s for s in stages(sc) if s.group == "A" and s.id != "A4"]
    t_lo = AscConfig().t_min_ms
    return {"scenario": sc.key, "mode": usb_mode(sc.mouse_hz, cfg), "A_pipeline": (sum(s.lo for s in a_pipe), sum(s.typ for s in a_pipe), sum(s.hi for s in a_pipe)),
            "A_total": (a_lo, a_typ, a_hi), "B": (b_lo, b_typ, b_hi), "ttl_ms": cfg["scene_ttl_ms"], "ttl_margin_ms": cfg["scene_ttl_ms"] - b_hi,
            "guard_ms": t_lo, "guard_margin_ms": t_lo - b_hi}


def with_changes(sc: Scenario, **kw) -> Scenario:
    return replace(sc, **kw)


# ------------------------------------------------------------------------------------------------ the hardware stand
@dataclass(frozen=True)
class TestPoint:
    id: str
    where: str
    signal: str
    bounds: str


TESTPOINTS = (
    TestPoint("S0", "метка в кадре: узкий светлый прямоугольник в известной строке + фотодиод на экране", "кадр, в котором объект появился (t0)",
              "начало B; сравнение с L1 даёт B1 + B2"),
    TestPoint("L1", "GPIO SoM: «кадр готов» (после последней строки ROI и подготовки)", "фронт при отдаче кадра детектору", "B1 + B2"),
    TestPoint("L2", "GPIO SoM: «детектор взял кадр» и «детектор закончил»", "два фронта вокруг инференса", "B3 (до первого), B4 (между ними)"),
    TestPoint("L3", "GPIO SoM: «сцена отправлена» (после трекера и подтверждения)", "фронт перед записью кадра SPI", "B5 + B6"),
    TestPoint("L4", "GPIO моста: «кадр сцены принят» (в link_rx)", "фронт при приёме", "B7 + B8"),
    TestPoint("L5", "GPIO моста: «отчёт мыши принят» и «отчёт отправлен ПК» (завершение IN)", "два фронта вокруг bridge_mouse_in", "A1 + A2 (+ часть A3)"),
    TestPoint("U1", "анализатор USB (например Total Phase Beagle) на кабеле мышь↔мост и мост↔ПК", "метки пакетов IN на обеих сторонах", "A3"),
    TestPoint("M1", "эмулятор мыши (второй МК, подаёт отчёты в известные моменты) + тестовая программа на ПК с меткой времени", "известный момент отчёта → момент получения",
              "A целиком; сравнить с прямым подключением (мост в обходе)"),
    TestPoint("V1", "высокоскоростная камера (≥ 1000 кадр/с) на экране и на светодиоде эмулятора мыши", "момент движения указателя после появления цели", "B + C, как это видит человек"),
)
