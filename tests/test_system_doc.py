"""docs/SYSTEM.md is the overview of the whole product: what it names must exist, so the overview can not quietly drift away from the code."""
import re
from pathlib import Path

from dataopen.release import registry as R

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "SYSTEM.md"
TEXT = DOC.read_text(encoding="utf-8")

SECTIONS = ["Название, статус, цель документа", "Оглавление", "Именование и карта подсистем", "Философия и принципы", "Аппаратная архитектура",
            "Программные подсистемы", "Межмодульные протоколы и данные", "Безопасность, идентичность, пакеты", "OTA и каналы обновлений",
            "Слоты и профили", "Latency и критический путь", "Provisioning, recovery, сбросы", "Тестирование и текущее покрытие",
            "Ограничения и то, что не реализовано", "Definition of Done и готовность к v1", "Дорожная карта"]


def test_the_sixteen_sections_of_the_brief_are_there_in_order():
    heads = re.findall(r"^## (\d+)\. (.+)$", TEXT, flags=re.M)
    assert [(int(n), t) for n, t in heads[:16]] == list(enumerate(SECTIONS, 1))
    for n, t in enumerate(SECTIONS, 1):                                                  # and the table of contents links to each of them
        assert f"{n}. [{t}](#" in TEXT, t


def test_every_file_the_overview_names_exists():
    paths = set(re.findall(r"`((?:src|tests|pwa|docs)/[A-Za-z0-9_./*{},-]+)`", TEXT))
    missing = []
    for p in paths:
        if "{" in p or "," in p:                                                         # a shorthand list: its directory must at least exist
            if not (ROOT / p.split("{")[0].split(",")[0].rsplit("/", 1)[0]).exists():
                missing.append(p)
        elif "*" in p:
            if not list(ROOT.glob(p)):
                missing.append(p)
        elif not (ROOT / p).exists():
            missing.append(p)
    assert not missing, missing


def test_every_linked_document_exists():
    links = set(re.findall(r"\]\(([A-Z0-9_]+\.md)\)", TEXT))
    assert links and all((ROOT / "docs" / name).exists() for name in links), links


def test_every_module_status_in_the_registry_is_told_the_same_way():
    """The overview must not call something working that the registry calls architecture-only, and the other way round."""
    for must in ("MOD", "LRN", "GAT", "BLD", "CAP", "BRD", "FAC"):
        assert re.search(rf"\| \*\*{must}\*\* \|", TEXT), must
    assert "**Ни одна цифра не получена на железе.**" in TEXT and "**НЕТ**" in TEXT
    for name in ("live-calibration", "module-daemon", "bootloader", "ble-gatt-server", "hid-bridge-board"):
        assert R.BY_ID[name].status == R.ARCHITECTURE, name                              # the seven [A] blocks of the overview are [A] in the registry


def test_the_numbers_quoted_from_the_registry_are_the_registry_numbers():
    total = len(R.DOD)
    met = sum(1 for d in R.DOD if d.met)
    assert f"Выполнено {met} из {total}" in TEXT or f"выполнено {met} из {total}" in TEXT
    assert len(R.SCENARIOS) == 9 and "9 сценариев" in TEXT
    for d in R.DOD:                                                                       # every criterion id appears in the DoD summary
        assert d.id in TEXT, d.id
    for r in R.RISKS:
        assert f"| {r.id} |" in TEXT, r.id
    for st in R.ORDER:
        assert f"| **{st.n}** |" in TEXT, st.n
