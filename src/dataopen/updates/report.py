"""Tables for docs/UPDATES.md, generated from the channel descriptions, the formats and the table of reasons (markers `<!-- fp:NAME -->`)."""
from __future__ import annotations

import re
from pathlib import Path

from ..ctl import firmware as FW
from . import channels as C
from . import models as MD
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


def abi_table() -> str:
    a = MD.ABI
    rows = (("версия контракта", f"`ABI v{a['version']}`; файл ONNX, один, без внешних данных"),
            ("таксономия", f"`{a['taxonomy']}`: классы строго из {len(a['classes'])} интерфейсных (" + ", ".join(f"`{c}`" for c in a["classes"]) + "); ключевых точек нет"),
            ("вход", f"один, кадр {a['input_size']}×{a['input_size']}: " + " или ".join(f"`{c}`" for c in a["conventions"])),
            ("выходы", f"ровно {MD.N_OUT}, `p3`, `p4`, `p5`: [N, число_классов + 4, {MD.INPUT_SIZE}/s, {MD.INPUT_SIZE}/s] для s = " + ", ".join(str(x) for x in MD.STRIDES)),
            ("число классов", "= число классов в метаданных `ui_layout` = число в карточке пакета (метаданные не могут врать о голове)"),
            ("операторы", ", ".join(f"`{o}`" for o in a["ops"]) + "; только стандартный набор (`ai.onnx`), без подграфов"),
            ("opset", f"{a['opset'][0]}…{a['opset'][1]}"),
            ("размер", f"до {a['max_bytes'] // (1024 * 1024)} МиБ"),
            ("узлы", f"до {a['max_nodes']}"),
            ("вычисления", f"до {a['max_macs'] / 1e9:g} GMAC по статическим формам (допущение, не измерение; у UiNet около 1)"))
    return "\n".join([_row("что", "правило"), _row("---", "---")] + [_row(f"**{k}**", v) for k, v in rows])


MODEL_RULES = (("для этого устройства", "пакет запечатан под Device Public ID; чужой отвергается по заголовку, до данных", "`wrong_device`"),
               ("от известного отправителя", "подпись Ed25519; новый отправитель требует кнопки (`trust:<ID>`)", "`bad_signature`, `physical`"),
               ("каждая модель с кнопкой", "веса принимаются только после нажатия (`model:<ID>`), даже от доверенного", "`physical`"),
               ("только интерфейс", "классы из таксономии `ui-v1`; людей, позы, «игроков», ключевые точки нет", "`not_ui_model`"),
               ("только данные", "белый список операторов; нет внешних файлов, чужих доменов, подграфов", "`model_ops`, `model_files`"),
               ("контракт входа/выходов", "форма, тип и число выходов как у `UiNet`", "`model_io`"),
               ("по размеру и цене", "до 16 МиБ, до 2000 узлов, до 4 GMAC", "`too_large`, `model_cost`"),
               ("потом остаётся на устройстве", "модель лежит под ключом слота; наружу не выходит (карточка без весов)", "—"),
               ("не код", "исполняемого кода, скриптов, внешних данных и каталога моделей нет", "—"))


def model_rules_table() -> str:
    return "\n".join([_row("правило", "как проверяется", "отказ"), _row(*["---"] * 3)] + [_row(f"**{a}**", b, c) for a, b, c in MODEL_RULES])


TABLES = {"channels": channels_table, "policy": policy_table, "invariants": invariants_table, "format_a": format_a_table, "format_b": format_b_table,
          "parts": parts_table, "reasons_a": reasons_a, "reasons_b": reasons_b, "abi": abi_table, "model_rules": model_rules_table}
_BLOCK = re.compile(r"(<!-- fp:(\w+) -->)\n?(.*?)\n?(<!-- /fp:\2 -->)", re.S)


def render_doc(text: str) -> str:
    return _BLOCK.sub(lambda m: m.group(1) + "\n" + TABLES[m.group(2)]() + "\n" + m.group(4), text)


MODEL_TABLES = ("abi", "model_rules")            # these two live in docs/MODELS.md, the rest in docs/UPDATES.md


def expected_tables(path: Path) -> set:
    return set(MODEL_TABLES) if path.name == "MODELS.md" else set(TABLES) - set(MODEL_TABLES)


def doc_is_current(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return render_doc(text) == text and set(m.group(2) for m in _BLOCK.finditer(text)) == expected_tables(path)
