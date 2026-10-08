"""Markdown tables generated from the data, so docs/HARDWARE.md cannot drift from `spec`, `power`, `failsafe` and `indication`.

In the document a table sits between `<!-- hw:NAME -->` and `<!-- /hw:NAME -->`; `dataopen hw docs --write` refreshes them and a test checks
that they are current.
"""
from __future__ import annotations

import re
from pathlib import Path

from . import failsafe as F
from . import indication as I
from . import power as W
from . import spec as S


def _row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _usd(x: float) -> str:
    return f"${x:,.0f}"


def sku_table() -> str:
    out = [_row("SKU", "видео", "мышь", "кому", "розница (ориентир)", "BOM (ориентир, диапазон)"), _row(*["---"] * 6)]
    for s in S.SKUS:
        lo, hi = S.cost(s)
        out.append(_row(f"**{s.code} {s.name}**", s.video, s.mouse, s.audience, _usd(S.retail(s)), f"{_usd(lo)} - {_usd(hi)}"))
    return "\n".join(out)


def dnp_table() -> str:
    out = [_row("группа DNP", "что ставится", *[s.code for s in S.SKUS]), _row(*["---"] * (2 + len(S.SKUS)))]
    what = {S.G_HDMI: "HDMI IN/OUT, сплиттер, захват HDMI→MIPI, ВЧ-реле", S.G_DP: "DP IN/OUT, повторитель, захват DP→MIPI, ВЧ-реле",
            S.G_USBC_MOUSE: "гнездо USB-C для мыши, второй ключ VBUS", S.G_AUDIO: "зуммер", S.G_WIFI: "модуль Wi-Fi (SoftAP)"}
    out.append(_row(S.CORE, "SoM, MCU моста, K1/K2, безопасный домен, USB-A мыши, питание, джек, кнопки, светодиоды, secure element, корпус", *["да"] * len(S.SKUS)))
    for g in S.GROUPS:
        out.append(_row(g, what[g], *["да" if g in s.groups else "DNP" for s in S.SKUS]))
    return "\n".join(out)


def bom_table() -> str:
    out = [_row("ref", "позиция", "шт.", "цена за шт., USD", "DNP на"), _row(*["---"] * 5)]
    for p in S.PARTS:
        off = ", ".join(s.code for s in S.SKUS if not s.fits(p)) or "нигде"
        mark = " *" if p.candidate else ""
        out.append(_row(p.ref, p.name + mark, p.qty, f"{p.lo:g} - {p.hi:g}", off))
    out.append("")
    out.append("`*` семейство микросхем названо по памяти как кандидат и не проверялось по даташиту. Цены порядка 1000 штук, не котировки.")
    return "\n".join(out)


def bom_totals_table() -> str:
    out = [_row("SKU", "позиций", "BOM мин.", "BOM макс.", "BOM середина", "розница"), _row(*["---"] * 6)]
    for s in S.SKUS:
        lo, hi = S.cost(s)
        out.append(_row(s.code, len(S.bom(s)), _usd(lo), _usd(hi), _usd((lo + hi) / 2), _usd(S.retail(s))))
    return "\n".join(out)


def power_table() -> str:
    out = [_row("нагрузка", "домен", "мышь, Вт", "видео, Вт", "пик, Вт", "SKU"), _row(*["---"] * 6)]
    for l in W.LOADS:
        fitted = "все" if l.group == S.CORE else ", ".join(s.code for s in S.SKUS if l.group in s.groups)
        out.append(_row(l.name, l.domain, f"{l.mouse_only:g}", f"{l.video:g}", f"{l.peak:g}", fitted))
    return "\n".join(out)


def budget_table() -> str:
    out = [_row("SKU", "от USB-C: мышь, Вт", "видео, Вт", "пик, Вт", f"предел ({int(W.DERATE * 100)} % от {W.SUPPLY_V:g} В / {W.SUPPLY_A:g} А), Вт", "запас на пике, Вт",
                "перегрев корпуса: мышь, К", "видео, К"), _row(*["---"] * 8)]
    for s in S.SKUS:
        b, t = W.budget(s), W.thermal(s)
        out.append(_row(s.code, b["mouse_only"], b["video"], b["peak"], b["limit_w"], b["margin_peak_w"],
                        f"{t['mouse_only']['dt_k']:g} {'ок' if t['mouse_only']['ok'] else 'ВЫШЕ'}", f"{t['video']['dt_k']:g} {'ок' if t['video']['ok'] else 'ВЫШЕ'}"))
    out.append("")
    out.append(f"Корпус {'x'.join(str(v) for v in S.ENCLOSURE['size_mm'])} мм, площадь {W.case_area_m2() * 1e4:.0f} см², h = {W.H_W_M2K:g} Вт/(м²·К): "
               f"{W.r_th_k_per_w():.2f} К/Вт; предел касания {W.TOUCH_DT_K:g} К даёт **{W.sustained_cap_w():.1f} Вт** длительно.")
    return "\n".join(out)


def domains_table() -> str:
    out = [_row("домен", "что в нём", "когда питается"), _row(*["---"] * 3)]
    for k, (what, when) in W.DOMAINS.items():
        out.append(_row(k, what, when))
    return "\n".join(out)


def failsafe_table() -> str:
    out = [_row("событие", "что делает аппаратура", "до прямого соединения", "нужна ли живая прошивка", "что видит человек"), _row(*["---"] * 5)]
    for s in F.SCENARIOS:
        t = "контакты не двигаются" if s.to_bypass_ms is None else f"≈ {s.to_bypass_ms} мс"
        out.append(_row(s.event, s.hardware, t, "да" if s.needs_firmware else "**нет**", s.user_sees))
    return "\n".join(out)


def truth_table() -> str:
    """The cases that matter (the whole 128-row table is enumerated by the tests)."""
    I_ = F.Inputs
    rows = [("всё в норме, мост просит мышь и видео", I_(mcu_engage=True, video_en=True)), ("всё в норме, мост просит только мышь", I_(mcu_engage=True)),
            ("нет внешнего питания (всё остальное «хорошо»)", I_(vext_ok=False, mcu_engage=True, video_en=True)),
            ("Panic удержан ≥ 3 с или обрыв кабеля", I_(panic_loop_ok=False, mcu_engage=True, video_en=True)),
            ("зависание моста (watchdog)", I_(wdg_ok=False, mcu_engage=True, video_en=True)),
            ("ползунок MODE на BYPASS", I_(mode_assist=False, mcu_engage=True, video_en=True)),
            ("мост не просит ничего (старт, выключено)", I_()), ("D3 без питания, видео просят", I_(mcu_engage=True, video_en=True, video_pwr_ok=False))]
    out = [_row("ситуация", "мышь через мост (K1,K2)", "видео через сплиттер (KV)", "жёлтый светодиод BYPASS"), _row(*["---"] * 4)]
    for name, i in rows:
        o = F.evaluate(i)
        out.append(_row(name, "да" if o.mouse_engaged else "**напрямую**", "да" if o.video_engaged else "**напрямую**", "горит" if o.bypass_led else "нет"))
    return "\n".join(out)


def _pattern(s: I.State) -> str:
    segs = ", ".join(f"{'вкл' if on else 'выкл'} {ms}" for on, ms in s.pattern)
    return ("плавно: " if s.breathe else "") + segs + " мс"


def indication_table() -> str:
    out = [_row("состояние", "RGB-светодиод", "рисунок (цикл)", "зуммер", "что видит и слышит человек"), _row(*["---"] * 5)]
    for s in I.STATES:
        beep = "тишина" if not s.beep else " + ".join(f"{f} Гц {ms} мс" if f else f"пауза {ms}" for f, ms in I.BEEPS[s.beep])
        out.append(_row(f"**{s.key}**", s.color or "-", _pattern(s), beep, s.ru))
    return "\n".join(out)


def ports_table() -> str:
    out = [_row("грань", *[s.code for s in S.SKUS]), _row(*["---"] * (1 + len(S.SKUS)))]
    for f in S.FACES:
        out.append(_row(f"{f.name}: {f.where}, {f.length_mm:g} мм" + (f" x {f.rows} ряда" if f.rows > 1 else ""),
                        *[", ".join(S.ports(s)[f.name]) or "-" for s in S.SKUS]))
    return "\n".join(out)


def blocks_table() -> str:
    out = [_row("блок", "домен"), _row("---", "---")]
    out += [_row(k, v) for k, v in S.BLOCKS.items()]
    return "\n".join(out)


TABLES = {"sku": sku_table, "dnp": dnp_table, "bom": bom_table, "bom_totals": bom_totals_table, "power_loads": power_table, "power_budget": budget_table,
          "domains": domains_table, "blocks": blocks_table, "failsafe": failsafe_table, "truth": truth_table, "indication": indication_table, "ports": ports_table,
          "diagram": lambda: "```\n" + S.DIAGRAM + "\n```"}
_BLOCK = re.compile(r"(<!-- hw:(\w+) -->)\n?(.*?)\n?(<!-- /hw:\2 -->)", re.S)


def render_doc(text: str) -> str:
    def sub(m: re.Match) -> str:
        return m.group(1) + "\n" + TABLES[m.group(2)]() + "\n" + m.group(4)
    return _BLOCK.sub(sub, text)


def doc_is_current(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return render_doc(text) == text and set(m.group(2) for m in _BLOCK.finditer(text)) == set(TABLES)
