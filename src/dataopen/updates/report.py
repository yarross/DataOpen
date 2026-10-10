"""Tables for docs/UPDATES.md, generated from the channel descriptions, the formats and the table of reasons (markers `<!-- fp:NAME -->`)."""
from __future__ import annotations

import re
from pathlib import Path

from ..ctl import firmware as FW
from . import channels as C
from . import package as K


def _row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def channels_table() -> str:
    out = [_row("канал", "что входит", "кто подписывает", "как доставляется", "риск"), _row(*["---"] * 5)]
    for c in C.CHANNELS:
        out.append(_row(f"**{c.key}. {c.name}**", c.includes, c.signer, f"{c.delivery}; {c.audience}", c.risk))
    return "\n".join(out)


POLICY_ROWS = (("trust_root", "кому доверяет устройство"), ("binding", "к чему привязан файл"), ("confidentiality", "шифрование"),
               ("replay", "повтор и откат версии"), ("approval", "подтверждение человеком"), ("staging", "куда кладётся"),
               ("probation", "проба после применения"), ("rollback", "возврат"), ("transport", "по проводу"), ("blast_radius", "что может пойти не так"))


def policy_table() -> str:
    a, b = C.BY_KEY["A"], C.BY_KEY["B"]
    out = [_row("", f"A. {a.name}", f"B. {b.name}"), _row("---", "---", "---")]
    out += [_row(f"**{title}**", getattr(a, attr), getattr(b, attr)) for attr, title in POLICY_ROWS]
    return "\n".join(out)


def invariants_table() -> str:
    return "\n".join([_row("правило", "что гарантируется"), _row("---", "---")] + [_row(f"`{k}`", v) for k, v in C.INVARIANTS])


def format_b_table() -> str:
    out = [_row("поле", "байт", "смысл"), _row("---", "---", "---")]
    off = 0
    for name, n, what in K.FIELDS:
        out.append(_row(f"`{name}`", f"{n} ({off}…{off + n - 1})", what))
        off += n
    out.append(_row("`signature`", f"{K.SIG} ({K.HEAD}…{K.PREFIX - 1})", "Ed25519 отправителя над `'DOPK-v1-sig' ‖ заголовок`"))
    out.append(_row("`ciphertext`", f"{{ct_len}} (с {K.PREFIX})", f"части по `chunk_size` + {K.TAG} байт метки; метка части связана с её номером и признаком «последняя»"))
    return "\n".join(out)


def format_a_table() -> str:
    rows = (("magic", 4, "'DOFW'"), ("fmt", 1, "формат, 1"), ("(пусто)", 3, "нули, входят в подпись"), ("hw_id", 8, "тип железа"),
            ("version", 4, "версия образа"), ("min_version", 4, "порог анти-отката: ниже уже не принимается"), ("payload_len", 4, "длина образа"),
            ("sha256", 32, "SHA-256 образа"))
    assert sum(n for _, n, _ in rows) == FW.HEAD
    out = [_row("поле", "байт", "смысл"), _row("---", "---", "---")]
    off = 0
    for name, n, what in rows:
        out.append(_row(f"`{name}`", f"{n} ({off}…{off + n - 1})", what))
        off += n
    out.append(_row("`signature`", f"{FW.SIG} ({FW.HEAD}…{FW.HEAD + FW.SIG - 1})", "Ed25519 ПРОИЗВОДИТЕЛЯ над заголовком"))
    out.append(_row("`payload`", f"{{payload_len}} (с {FW.HEAD + FW.SIG})", "образ без шифрования: файл публичный"))
    return "\n".join(out)


def parts_table() -> str:
    rows = (("profile", "моторный профиль (то же кодирование, что в `DOBS`)", "в слоте, под ключом слота", "кнопка для нового отправителя; проба «Оставить / Вернуть»", "`profile.prev`, `ACT profile.restore`"),
            ("tuning", "два уровня: сила помощи и подавление дрожи", "в слоте", "то же", "проба"),
            ("ui_manifest", "раскладка экранов слота (проверяется тем же валидатором, что и встроенная)", "в слоте", "то же", "прежняя раскладка не хранится: слот можно очистить"),
            ("meta", "имя слота (до 24 символов)", "в слоте", "то же", "—"),
            ("model", "веса лёгкой модели интерфейсного детектора (ONNX) и её карточка", "в слоте: `model.bin`, `model.json` под ключом слота", "кнопка каждый раз", "`model.prev.*`, `ACT pkg.revert` (меняет местами)"))
    return "\n".join([_row("часть", "что это", "где лежит", "подтверждение", "возврат"), _row(*["---"] * 5)] + [_row(f"`{a}`", *r) for a, *r in rows])


def _reasons(channel: str) -> str:
    out = [_row("причина", "когда замечено", "что остаётся на устройстве", "что видит человек", "что делать"), _row(*["---"] * 5)]
    out += [_row(f"`{r.key}`", r.at, r.left, r.ru, r.do) for r in C.REASONS if r.channel == channel]
    return "\n".join(out)


def reasons_a() -> str:
    return _reasons("A")


def reasons_b() -> str:
    return _reasons("B")


TABLES = {"channels": channels_table, "policy": policy_table, "invariants": invariants_table, "format_a": format_a_table, "format_b": format_b_table,
          "parts": parts_table, "reasons_a": reasons_a, "reasons_b": reasons_b}
_BLOCK = re.compile(r"(<!-- fp:(\w+) -->)\n?(.*?)\n?(<!-- /fp:\2 -->)", re.S)


def render_doc(text: str) -> str:
    return _BLOCK.sub(lambda m: m.group(1) + "\n" + TABLES[m.group(2)]() + "\n" + m.group(4), text)


def doc_is_current(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return render_doc(text) == text and set(m.group(2) for m in _BLOCK.finditer(text)) == set(TABLES)
