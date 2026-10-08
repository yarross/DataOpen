"""Tables for docs/PROVISIONING.md, generated from the data (markers `<!-- fp:NAME -->` ... `<!-- /fp:NAME -->`), so the document can not drift."""
from __future__ import annotations

import re
from pathlib import Path

from ..hw import indication as I
from . import recovery as RC
from . import records as R
from . import station as ST


def _row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def steps_table() -> str:
    out = [_row("№", "шаг", "что происходит"), _row("---", "---", "---")]
    out += [_row(i + 1, f"`{k}`", text) for i, (k, text) in enumerate(ST.STEPS)]
    return "\n".join(out)


def checks_table() -> str:
    out = [_row("проверка стенда", "что измеряется", "допуск"), _row("---", "---", "---")]
    out += [_row(f"`{k}`", d, lim) for k, (d, lim) in ST.CHECK_DOCS.items()]
    return "\n".join(out)


def identity_table() -> str:
    rows = [("серийный номер и запись", "пишется на заводе один раз; наклейка", "OTP / заблокированный слот secure element", "**неизменяема** никаким сбросом"),
            ("DAK, ключ подлинности", "рождается в secure element; подписывает карточку владельца и доказательство владения", "secure element, наружу не выходит", "**неизменяем** никаким сбросом"),
            ("аттестация производителя", "подпись производителя над (серийник, hw_id, DAK, плата)", "в записи устройства; копия в базе производителя", "**неизменяема**"),
            ("ключи владельца = Device Public ID", "привязка пакетов `DOBS`, подпись карточки", "secure element / хранилище ключей", "**меняются** заводским сбросом (L3) и возвратом (L4)"),
            ("DAK-сертификат карточки", "связывает серийник с текущей карточкой владельца", "в карточке; подписывается заново при смене ключей", "пересоздаётся при L3")]
    return "\n".join([_row("что", "для чего", "где хранится", "можно ли сбросить"), _row(*["---"] * 4)] + [_row(*r) for r in rows])


def lifecycle_table() -> str:
    names = {R.BLANK: "плата прошита, ничего не записано", R.DEV_TESTED: "стенд проверил плату", R.PROVISIONED: "запись устройства сделана, OTP ещё открыт",
             R.SHIPPED: "OTP и ключи заблокированы, готово к упаковке", R.IN_FIELD: "первое включение у владельца", R.RMA: "на сервисе: возврат к заводскому состоянию"}
    out = [_row("стадия", "смысл", "дальше"), _row("---", "---", "---")]
    out += [_row(f"`{s}`", names[s], ", ".join(f"`{t}`" for t in R.TRANSITIONS[s]) + (" (только по токену производителя)" if s == R.RMA else "")) for s in R.LIFECYCLE]
    return "\n".join(out)


def storage_table() -> str:
    out = [_row("что", "где хранится", "кто меняет", "L1", "L2", "L3", "L4"), _row(*["---"] * 7)]
    for it in R.ITEMS:
        out.append(_row(it.what, it.where, it.mutability, *it.levels))
    return "\n".join(out)


def levels_table() -> str:
    out = [_row("уровень", "название", "как вызывается", "что затрагивает"), _row(*["---"] * 4)]
    out += [_row(f"**{k}**", name, how, what) for k, name, how, what in R.LEVELS]
    return "\n".join(out)


def scenarios_table() -> str:
    out = [_row("событие", "что видит человек сразу", "как вернуть", "компьютер", "профили"), _row(*["---"] * 5)]
    for s in RC.SCENARIOS:
        o = RC.run(s)
        first = f"{o.first.stage}: мышь «{o.first.mouse}», экран «{o.first.video}», светодиод {o.first.led}"
        how = "; ".join(t for _, t in s.procedure)
        prof = "сохраняются" if o.profiles_kept else ("вернутся из резервной копии .dobundle" if o.backup_restores else "теряются (возврат производителю)")
        out.append(_row(s.event, first, how, "нужен" if o.computer else "не нужен", prof))
    return "\n".join(out)


def tools_table() -> str:
    rows = [("завод", "`dataopen prov provision`", "станция: прошивка, стенд, ключи, аттестация, запись, блокировка, наклейка, отчёт"),
            ("завод", "`dataopen prov verify`", "проверка цепочки производитель → DAK → карточка по каталогу устройства"),
            ("завод", "HSM / база серийников", "подпись аттестаций и токенов, уникальность серийник ↔ DAK ↔ плата, отзыв (в симуляции: ключ в памяти)"),
            ("поддержка", "`dataopen recover simulate`", "разбор сценария: что видит человек, что делать, нужен ли компьютер"),
            ("поддержка", "`dataopen recover full-return`", "L4: токен + присутствие, банки из золотого образа, личное стёрто"),
            ("человек", "кнопки SLOT + CONFIRM при подаче питания", "система восстановления без компьютера"),
            ("человек", "приложение (PWA)", "L1/L2/L3 (два шага + кнопка), серийный номер, резервные копии")]
    return "\n".join([_row("кто", "инструмент", "что делает"), _row("---", "---", "---")] + [_row(*r) for r in rows])


def indication_recovery_table() -> str:
    s = I.BY_KEY["RECOVERY"]
    return _row("состояние", "цвет", "рисунок", "что видит человек") + "\n" + _row(*["---"] * 4) + "\n" + _row(
        f"**{s.key}**", s.color, ", ".join(f"{'вкл' if on else 'выкл'} {ms}" for on, ms in s.pattern) + " мс", s.ru)


TABLES = {"steps": steps_table, "checks": checks_table, "identity": identity_table, "lifecycle": lifecycle_table, "storage": storage_table,
          "levels": levels_table, "scenarios": scenarios_table, "tools": tools_table, "recovery_led": indication_recovery_table}
_BLOCK = re.compile(r"(<!-- fp:(\w+) -->)\n?(.*?)\n?(<!-- /fp:\2 -->)", re.S)


def render_doc(text: str) -> str:
    return _BLOCK.sub(lambda m: m.group(1) + "\n" + TABLES[m.group(2)]() + "\n" + m.group(4), text)


def doc_is_current(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return render_doc(text) == text and set(m.group(2) for m in _BLOCK.finditer(text)) == set(TABLES)
