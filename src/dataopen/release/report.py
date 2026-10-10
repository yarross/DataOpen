"""Tables for docs/V1.md, generated from the registry (markers `<!-- fp:NAME -->`), so the document can not say more than the checks hold."""
from __future__ import annotations

import re
from pathlib import Path

from . import check as C
from . import registry as R


def _row(*cells) -> str:
    return "| " + " | ".join(str(c).replace("\n", " ") for c in cells) + " |"


def _files(refs: tuple) -> str:
    """Evidence as short file names (the whole refs are checked; the table only needs to show where to look)."""
    out: list[str] = []
    for r in refs:
        f = r.split("::")[0].rsplit("/", 1)[-1]
        if f not in out:
            out.append(f)
    return ", ".join(f"`{f}`" for f in out) if out else "—"


def _modules(status: str) -> str:
    out = [_row("модуль", "область", "что это", "чем подтверждено", "чего не проверено"), _row(*["---"] * 5)]
    for m in R.MODULES:
        if m.status != status:
            continue
        out.append(_row(f"**{m.name}**", R.AREA_TITLES[m.area], m.what, _files(m.evidence), "; ".join(m.gaps) if m.gaps else "—"))
    return "\n".join(out)


def modules_implemented() -> str:
    return _modules(R.IMPLEMENTED)


def modules_simulated() -> str:
    return _modules(R.SIMULATED)


def modules_architecture() -> str:
    return _modules(R.ARCHITECTURE)


def modules_deferred() -> str:
    out = [_row("что", "почему не в v1"), _row("---", "---")]
    out += [_row(f"**{m.name}**", m.what + ("; " + "; ".join(m.gaps) if m.gaps else "")) for m in R.MODULES if m.status == R.DEFERRED]
    return "\n".join(out)


def never() -> str:
    return "\n".join(f"* {x}" for x in R.NEVER)


def counts() -> str:
    v, c = C.verdict(), C.counts()
    st = v["modules"]
    rows = [("модулей", f"{sum(st.values())}: implemented {st['implemented']}, simulated {st['simulated']}, architecture-only {st['architecture']}, deferred {st['deferred']}"),
            ("автотестов", f"Python {c['python_tests']} функций в {c['python_files']} файлах (с параметрами больше), Node {c['node_tests']} в {c['node_files']} файлах; из них приёмочных Python {c['acceptance_tests']}"),
            ("критериев готовности v1", f"выполнено {v['met']} из {v['total']}"),
            ("проверено на железе", f"{v['hardware_verified']} критериев"), ("проверено с людьми", f"{v['human_verified']} критериев")]
    return "\n".join([_row("что", "сколько"), _row("---", "---")] + [_row(*r) for r in rows])


def scenarios_table() -> str:
    out = [_row("№", "сценарий", "приёмочный тест", "главное непроверенное"), _row(*["---"] * 4)]
    for s in R.SCENARIOS:
        out.append(_row(s.id, s.name, f"`test_acc{s.id[-2:]}_*` и «день пилота»", s.unverified[0]))
    return "\n".join(out)


def scenarios() -> str:
    out: list[str] = []
    for s in R.SCENARIOS:
        out.append(f"#### {s.id}. {s.name}\n")
        out.append("**Предусловия**\n" + "\n".join(f"* {x}" for x in s.pre) + "\n")
        out.append("**Шаги**\n" + "\n".join(f"{i}. {x}" for i, x in enumerate(s.steps, 1)) + "\n")
        out.append("**Ожидаемый результат**\n" + "\n".join(f"* {x}" for x in s.expect) + "\n")
        out.append(f"**Что покрыто** ({s.covered_text}): {_files(s.covered)}; приёмочная функция `{s.acceptance}`.\n")
        out.append("**Что не проверено**\n" + "\n".join(f"* {x}" for x in s.unverified) + "\n")
    return "\n".join(out).rstrip()


def scope() -> str:
    return "\n".join(f"* {x}" for x in R.V1_SCOPE)


def dod() -> str:
    out = [_row("№", "группа", "критерий", "что нужно", "статус"), _row(*["---"] * 5)]
    for d in R.DOD:
        status = "**выполнено** (" + (d.note or "проверено") + ")" if d.met else "открыто" + (f" ({d.note})" if d.note else "")
        out.append(_row(f"**{d.id}**", d.group, d.criterion, R.KIND_TITLES[d.kind], status))
    return "\n".join(out)


def verdict() -> str:
    v = C.verdict()
    k = v["open_by_kind"]
    line = (f"**Готовность к пилоту: {'ДА' if v['ready'] else 'НЕТ'}.** Выполнено {v['met']} из {v['total']}. Открыто: железо {len(k['hardware'])} ({', '.join(k['hardware'])}); "
            f"люди {len(k['human'])} ({', '.join(k['human'])}); код {len(k['software'])} ({', '.join(k['software'])}); процесс и решения {len(k['process'])} ({', '.join(k['process'])}). "
            f"`dataopen release status` печатает то же и завершается кодом 1, пока хоть один критерий открыт. Критерий про железо или людей закрывается только файлом-записью "
            f"о приёмке (поле `record`), а не правкой флага.")
    return line


def risks() -> str:
    names = {"high": "высокий", "medium": "средний", "low": "низкий"}
    out = [_row("№", "уровень", "риск", "почему", "чем закрывать"), _row(*["---"] * 5)]
    out += [_row(f"**{r.id}**", names[r.level], r.title, r.why, r.cure) for r in R.RISKS]
    return "\n".join(out)


def order() -> str:
    out = [_row("шаг", "что делаем", "закрывает", "после шагов"), _row(*["---"] * 4)]
    out += [_row(f"**{s.n}. {s.title}**", s.what, ", ".join(s.exit), ", ".join(map(str, s.depends)) or "—") for s in R.ORDER]
    return "\n".join(out)


TABLES = {"counts": counts, "modules_implemented": modules_implemented, "modules_simulated": modules_simulated, "modules_architecture": modules_architecture,
          "modules_deferred": modules_deferred, "never": never, "scenarios_table": scenarios_table, "scenarios": scenarios, "scope": scope, "dod": dod,
          "verdict": verdict, "risks": risks, "order": order}
_BLOCK = re.compile(r"(<!-- fp:(\w+) -->)\n?(.*?)\n?(<!-- /fp:\2 -->)", re.S)


def render_doc(text: str) -> str:
    return _BLOCK.sub(lambda m: m.group(1) + "\n" + TABLES[m.group(2)]() + "\n" + m.group(4), text)


def doc_is_current(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return render_doc(text) == text and set(m.group(2) for m in _BLOCK.finditer(text)) == set(TABLES)
