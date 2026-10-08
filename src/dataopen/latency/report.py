"""Tables for docs/LATENCY.md, generated from the budget and from the simulation (markers `<!-- fp:NAME -->` ... `<!-- /fp:NAME -->`), so the
document can not drift from the code."""

from __future__ import annotations

import functools
import re
from pathlib import Path

from ..video import timing as VT
from . import budget as B
from . import measure as M


def _row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _ms(x: float) -> str:
    return "0" if x == 0 else f"{x:.3g}"


def _span(s: B.Stage) -> str:
    return f"{_ms(s.typ)} мс ({_ms(s.lo)}…{_ms(s.hi)})" if s.hi - s.lo > 1e-9 else f"{_ms(s.typ)} мс"


def stages_table(key: str, groups: str = "ABC") -> str:
    sc = B.BY_KEY[key]
    out = [_row("этап", "оценка задержки: типичная (от…до)", "источник цифры", "можно ли уменьшить"), _row(*["---"] * 4)]
    for s in B.stages(sc):
        if s.group not in groups:
            continue
        floor = f" **Нижний предел:** {B.FLOORS[s.floor]}." if s.floor in ("physical",) else ""
        out.append(_row(f"**{s.id}** {s.name}", _span(s), f"{B.BASES[s.basis]}. {s.ref}", f"{s.reduce}.{floor}" if s.reduce else floor.strip()))
    return "\n".join(out)


def stages_as_built() -> str:
    return stages_table("as-built")


def stages_target() -> str:
    return stages_table("npu-latest-c1")


def scenarios_table() -> str:
    out = [_row("сценарий", "A: конвейер ввода", "A + фильтр тремора", "B: сцена в мосту", "запас до TTL (худший случай)", "в пределах 10–12 мс?"),
           _row(*["---"] * 6)]
    for sc in B.SCENARIOS:
        s = B.summary(sc)
        ap, at, bb = s["A_pipeline"], s["A_total"], s["B"]
        verdict = []
        verdict.append("A: да" if ap[2] <= 12.0 else "A: нет")
        verdict.append("B: да" if bb[2] <= 12.0 else ("B: типично да" if bb[1] <= 12.0 else "B: нет"))
        ttl = f"{s['ttl_margin_ms']:.0f} мс" if s["ttl_margin_ms"] >= 0 else f"**{s['ttl_margin_ms']:.0f} мс: помощь пропадает**"
        out.append(_row(f"`{sc.key}`: {sc.title}", f"{_ms(ap[1])} ({_ms(ap[0])}…{_ms(ap[2])}) мс, режим {s['mode']}", f"{_ms(at[1])} ({_ms(at[0])}…{_ms(at[2])}) мс",
                        f"{_ms(bb[1])} ({_ms(bb[0])}…{_ms(bb[2])}) мс", ttl, ", ".join(verdict)))
    return "\n".join(out)


def policy_table() -> str:
    return stages_table("as-built", "C")


def floors_table() -> str:
    out = [_row("режим", "строка", "кадр", "ROI 640×640 готов", "возраст середины ROI", "весь кадр готов"), _row(*["---"] * 6)]
    for name in ("1920x1080@60", "1920x1080@144", "3840x2160@60"):
        m = VT.mode(name)
        roi = VT.Roi((m.h_active - 640) // 2, (m.v_active - 640) // 2, 640, 640)
        a = VT.data_age_us(m, roi)
        full = VT.data_age_us(m, VT.Roi(0, 0, m.h_active, m.v_active))
        out.append(_row(name, f"{m.line_time_us:.1f} мкс", f"{m.frame_time_us / 1000:.2f} мс", f"{a['ready_us'] / 1000:.2f} мс", f"{a['age_of_content_us'] / 1000:.2f} мс",
                        f"{full['ready_us'] / 1000:.2f} мс"))
    return "\n".join(out)


def usb_floor_table() -> str:
    out = [_row("мышь", "интервал опроса", "режим моста", "каскад опросов: в среднем / максимум", "ядро моста добавляет"), _row(*["---"] * 5)]
    for hz in (125, 500, 1000, 8000):
        p = M.poll_resample(hz)["added_ms"]
        out.append(_row(f"{hz} Гц", f"{1000 / hz:g} мс", B.usb_mode(hz), f"{_ms(p['mean'])} / {_ms(p['max'])} мс", "0 в симуляции (разрешение = шаг 1 мс)"))
    return "\n".join(out)


@functools.lru_cache(maxsize=None)
def _chain_runs() -> tuple:
    return tuple(M.chain_delay(p, a) for p, a in (("tremor", 0.0), ("tremor", 8.0), ("overshooter", 0.0), ("steady", 0.0)))


def chain_table() -> str:
    out = [_row("профиль", "фильтр тремора", "тремор в движении (отсчётов)", "отставание на 10 % / 50 % / 90 % пути, мс", "потеряно из движения", "наибольшее недоведение, отсчётов"),
           _row(*["---"] * 6)]
    for r, a in zip(_chain_runs(), (0.0, 8.0, 0.0, 0.0)):
        d = r["delay_ms"]
        out.append(_row(f"`{r['persona']}`", "включён" if r["tremor_filter"] else "выключен", f"{a:g}", f"{d[0.1]:.1f} / {d[0.5]:.1f} / {d[0.9]:.1f}",
                        f"{r['attenuation'] * 100:.0f} %", r["max_deficit_counts"]))
    return "\n".join(out)


@functools.lru_cache(maxsize=None)
def _scene_runs() -> tuple:
    return tuple(M.scene_effect(a) for a in (0, 50, 90, 110))


def scene_table() -> str:
    out = [_row("возраст сцены при отправке", "наименьший K (1 = помощи нет)", "тактов с торможением", "где остановился указатель, отсчётов", "сцена используется?"),
           _row(*["---"] * 5)]
    for r in _scene_runs():
        used = r["min_k"] < 1.0
        out.append(_row(f"{r['age_ms']:g} мс", f"{r['min_k']:.2f}", r["ticks_braked"], r["end_counts"],
                        "да" if used else f"**нет: старше TTL {r['ttl_ms']} мс**"))
    return "\n".join(out)


def pipe_table() -> str:
    out = [_row("детектор", "очередь", "кадр", "ожидание, мс", "цель подтверждена через, мс", "возраст кадра при взятии, мс"), _row(*["---"] * 6)]
    for label, infer, pol, frame in (("CPU 11 мс", 11.0, "fifo", 6.94), ("CPU 11 мс", 11.0, "latest", 6.94), ("CPU 16 мс", 16.0, "fifo", 6.94),
                                     ("CPU 16 мс", 16.0, "latest", 6.94), ("NPU 3 мс (цель)", 3.0, "fifo", 6.94), ("NPU 3 мс (цель)", 3.0, "latest", 6.94),
                                     ("CPU 11 мс", 11.0, "fifo", 16.67)):
        t = M.pipe_timeline(M.Pipe(frame, infer, B.UPDATE_MS, pol, B.queue_capacity(), 2))
        w, tot, age = M.stats(t["wait"]), M.stats(t["total"]), M.stats(t["taken_age"])
        out.append(_row(label, "FIFO(4)" if pol == "fifo" else "свежий кадр", f"{frame:g} мс", f"{w['mean']:.1f} (до {w['max']:.1f})",
                        f"{tot['mean']:.1f} (до {tot['max']:.1f})", f"{age['mean']:.1f} (до {age['max']:.1f})"))
    return "\n".join(out)


def testpoints_table() -> str:
    out = [_row("точка", "где", "сигнал", "какой промежуток ограничивает"), _row(*["---"] * 4)]
    out += [_row(f"**{t.id}**", t.where, t.signal, t.bounds) for t in B.TESTPOINTS]
    return "\n".join(out)


def bases_table() -> str:
    return "\n".join([_row("пометка", "что значит"), _row("---", "---")] + [_row(f"`{k}`", v) for k, v in B.BASES.items()])


def groups_table() -> str:
    return "\n".join([_row("группа", "что считаем"), _row("---", "---")] + [_row(f"**{k}**", v) for k, v in B.GROUPS.items()])


TABLES = {"stages_as_built": stages_as_built, "stages_target": stages_target, "scenarios": scenarios_table, "policy": policy_table,
          "floors": floors_table, "usb_floor": usb_floor_table, "chain": chain_table, "scene": scene_table, "pipe": pipe_table,
          "testpoints": testpoints_table, "bases": bases_table, "groups": groups_table}
_BLOCK = re.compile(r"(<!-- fp:(\w+) -->)\n?(.*?)\n?(<!-- /fp:\2 -->)", re.S)


def render_doc(text: str) -> str:
    return _BLOCK.sub(lambda m: m.group(1) + "\n" + TABLES[m.group(2)]() + "\n" + m.group(4), text)


def doc_is_current(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return render_doc(text) == text and set(m.group(2) for m in _BLOCK.finditer(text)) == set(TABLES)
