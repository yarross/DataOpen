"""The device manifest: which controls the phone shows, and what values the device will accept for them (docs/PWA.md section 4).

The client is deliberately dumb: it renders eight control types and knows no screen by name. The same object that is sent to the phone is
the one the gateway checks every SET / ACT against, so the device stays the only authority on what a value may be. The safety shell
(status, 'turn assistance off', 'hardware bypass') is NOT part of the manifest.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from . import protocol as P

SCHEMA = 1
TYPES = ("status", "toggle", "stepper", "action", "meter", "note", "file", "group")
CONFIRMS = ("none", "revert", "two-step")
FILE_OPS = ("bundle_put", "bundle_get", "bundle_for_card", "card_get")
MAX_PAGES, MAX_CONTROLS, MAX_DEPTH = 8, 64, 3
MAX_LABEL, MAX_HELP, MAX_STEPS = 80, 240, 20
LANGS = ("ru", "en")
_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,31}$")
_KEY = re.compile(r"^[a-z][a-z0-9_.]{0,39}$")


def L(ru: str, en: str) -> dict:
    return {"ru": ru, "en": en}


def default_manifest(rev: int = 1) -> dict:
    steps = {"0": L("Выкл", "Off"), "5": L("Как рекомендует профиль", "As the profile recommends"), "10": L("Максимум", "Maximum")}
    return {
        "schema": SCHEMA, "rev": rev, "title": L("Помощь при работе с мышью", "Mouse assistance"),
        "pages": [
            {"id": "main", "title": L("Помощь", "Assistance"), "controls": [
                {"id": "assist", "type": "toggle", "key": "assist.on", "confirm": "revert",
                 "label": L("Помощь", "Assistance"),
                 "help": L("Включает коррекцию движения мыши. Если станет хуже, нажмите «Вернуть».",
                           "Turns the pointer correction on. If it gets worse, press Undo.")},
                {"id": "strength", "type": "stepper", "key": "assist.strength", "min": 0, "max": 10, "marks": steps, "confirm": "revert",
                 "label": L("Сила помощи", "Assistance strength"),
                 "help": L("Насколько сильно помощь тормозит курсор у кнопок и значков.",
                           "How strongly the pointer is slowed near buttons and icons.")},
                {"id": "tremor", "type": "stepper", "key": "tremor.level", "min": 0, "max": 10, "marks": steps, "confirm": "revert",
                 "label": L("Подавление дрожи", "Tremor filter"),
                 "help": L("Насколько сильно убирается дрожь руки. Само движение не добавляется.",
                           "How much hand tremor is taken out. No motion is ever added.")},
            ]},
            {"id": "profile", "title": L("Профиль", "Profile"), "controls": [
                {"id": "fill", "type": "meter", "key": "profile.fill", "min": 0, "max": 100, "fmt": "{v} %",
                 "label": L("Профиль собран", "Profile collected"),
                 "help": L("Сколько наблюдений уже накоплено. Это не оценка качества.",
                           "How many observations were collected. Not a quality score.")},
                {"id": "layers", "type": "status", "key": "profile.layers",
                 "map": {"none": L("Данных пока мало", "Not enough data yet"),
                         "asc": L("Готова помощь при движении", "Movement help is ready"),
                         "tremor": L("Готово подавление дрожи", "Tremor filter is ready"),
                         "both": L("Готово всё", "Everything is ready")},
                 "label": L("Что уже готово", "What is ready")},
                {"id": "calib", "type": "toggle", "key": "calib.running", "confirm": "none",
                 "label": L("Калибровка", "Calibration"),
                 "help": L("Пока идёт калибровка, помощь выключена: устройство изучает ваши обычные движения. "
                           "Работайте с мышью как всегда.",
                           "While calibrating, assistance is off: the device learns your usual movements. Use the mouse as always.")},
                {"id": "import", "type": "file", "op": "bundle_put", "accept": ".dobundle", "max_bytes": 16000,
                 "label": L("Загрузить файл настроек", "Load a settings file"),
                 "help": L("Откроется только файл, сделанный для этого устройства.", "Only a file made for this device will open.")},
                {"id": "export", "type": "file", "op": "bundle_get", "accept": ".dobundle", "max_bytes": 16000,
                 "label": L("Сохранить копию для этого устройства", "Save a copy for this device"),
                 "help": L("Файл зашифрован и откроется только на этом устройстве.",
                           "The file is encrypted and opens only on this device.")},
                {"id": "export_other", "type": "file", "op": "bundle_for_card", "accept": ".docard", "max_bytes": 4096,
                 "label": L("Сохранить для другого устройства", "Save for another device"),
                 "help": L("Выберите карточку устройства-получателя и сверьте его номер с наклейкой. "
                           "Потребуется нажать кнопку на этом устройстве.",
                           "Choose the other device's card and check its number against its label. "
                           "You will need to press the button on this device.")},
                {"id": "restore", "type": "action", "key": "profile.restore", "confirm": "two-step",
                 "label": L("Вернуть прежний профиль", "Restore the previous profile")},
            ]},
            {"id": "more", "title": L("Ещё", "More"), "controls": [
                {"id": "device_id", "type": "status", "key": "device.id", "label": L("Номер устройства", "Device number")},
                {"id": "trusted", "type": "status", "key": "trusted.count",
                 "label": L("Откуда принимаются файлы (кроме своих копий)",
                            "Where files are accepted from (besides your own copies)")},
                {"id": "card", "type": "file", "op": "card_get", "accept": ".docard", "max_bytes": 4096,
                 "label": L("Сохранить карточку устройства", "Save the device card"),
                 "help": L("Карточку можно передавать: по ней нельзя прочитать ничего личного, "
                           "можно только подготовить файл для этого устройства.",
                           "The card is safe to share: it reveals nothing personal, "
                           "it only lets a file be prepared for this device.")},
                {"id": "forget", "type": "action", "key": "pairing.forget", "confirm": "two-step", "danger": True,
                 "label": L("Забыть все телефоны", "Forget all phones"),
                 "help": L("Потребуется нажать кнопку на самом устройстве.", "You will need to press the button on the device itself.")},
                {"id": "erase", "type": "action", "key": "erase.profile", "confirm": "two-step", "danger": True,
                 "label": L("Стереть личные данные", "Erase personal data"),
                 "help": L("Удаляет профиль, настройки и список доверенных. Помощь выключится. "
                           "Потребуется нажать кнопку на устройстве.",
                           "Deletes the profile, the settings and the trusted list. Assistance turns off. "
                           "You will need to press the button on the device.")},
                {"id": "factory", "type": "action", "key": "factory.reset", "confirm": "two-step", "danger": True,
                 "label": L("Заводской сброс", "Factory reset"),
                 "help": L("То же, и устройство получит новый номер: все файлы, сделанные для него, перестанут открываться. "
                           "Нужна кнопка на устройстве.",
                           "The same, and the device gets a new number: every file made for it stops opening. "
                           "You will need the button on the device.")},
            ]},
        ],
    }


def manifest_bytes(m: dict) -> bytes:
    return json.dumps(m, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def manifest_hash(m: dict) -> bytes:
    return P.crc32(manifest_bytes(m)).to_bytes(4, "big")


def _text(v: Any, limit: int, what: str, errs: list[str], required: bool = True) -> None:
    if v is None and not required:
        return
    if not isinstance(v, dict) or any(not isinstance(v.get(k), str) or not v.get(k) for k in LANGS):
        errs.append(f"{what}: needs non-empty text for {', '.join(LANGS)}")
        return
    for k in LANGS:
        if len(v[k]) > limit:
            errs.append(f"{what}.{k}: longer than {limit}")


def _items(v: Any, what: str, errs: list[str]):
    if v is None:
        return []
    if not isinstance(v, dict):
        errs.append(f"{what}: must be an object")
        return []
    return list(v.items())


def validate_manifest(m: Any) -> list[str]:
    """The empty list when the manifest is acceptable; otherwise what is wrong (used by the tests and by the gateway at start-up)."""
    errs: list[str] = []
    if not isinstance(m, dict):
        return ["manifest must be an object"]
    if m.get("schema") != SCHEMA:
        errs.append(f"schema must be {SCHEMA}")
    if not isinstance(m.get("rev"), int) or not 0 <= m["rev"] <= 0xFFFF:
        errs.append("rev must be an integer 0..65535")
    _text(m.get("title"), MAX_LABEL, "title", errs)
    pages = m.get("pages")
    if not isinstance(pages, list) or not 1 <= len(pages) <= MAX_PAGES:
        return errs + [f"pages: 1..{MAX_PAGES} entries"]
    ids: set[str] = set()
    count = [0]

    def control(c: Any, depth: int, where: str) -> None:
        count[0] += 1
        if count[0] > MAX_CONTROLS:
            errs.append(f"more than {MAX_CONTROLS} controls")
            return
        if not isinstance(c, dict):
            errs.append(f"{where}: must be an object")
            return
        cid, typ, key = c.get("id"), c.get("type"), c.get("key")
        if not isinstance(cid, str) or not _ID.match(cid):
            errs.append(f"{where}: bad id {cid!r}")
        elif cid in ids:
            errs.append(f"{where}: duplicate id {cid!r}")
        else:
            ids.add(cid)
        w = f"{where}/{cid}"
        if typ not in TYPES:
            errs.append(f"{w}: unknown type {typ!r}")
            return
        _text(c.get("label"), MAX_LABEL, f"{w}.label", errs, required=typ != "note")
        _text(c.get("help"), MAX_HELP, f"{w}.help", errs, required=False)
        needs_key = typ in ("status", "toggle", "stepper", "action", "meter")
        if needs_key and (not isinstance(key, str) or not _KEY.match(key)):
            errs.append(f"{w}: bad key {key!r}")
        if c.get("confirm", "none") not in CONFIRMS:
            errs.append(f"{w}: bad confirm")
        if typ == "stepper":
            lo, hi = c.get("min"), c.get("max")
            if not (isinstance(lo, int) and isinstance(hi, int) and not isinstance(lo, bool) and 0 < hi - lo <= MAX_STEPS):
                errs.append(f"{w}: min/max must be integers with 1..{MAX_STEPS} steps")
            else:
                for mk, mv in _items(c.get("marks"), f"{w}.marks", errs):
                    if not (mk.lstrip("-").isdigit() and lo <= int(mk) <= hi):
                        errs.append(f"{w}: mark {mk!r} outside the range")
                    _text(mv, MAX_LABEL, f"{w}.marks.{mk}", errs)
        elif typ == "meter":
            if not (isinstance(c.get("min"), int) and isinstance(c.get("max"), int) and c["min"] < c["max"]):
                errs.append(f"{w}: min/max must be integers")
            if "{v}" not in str(c.get("fmt", "{v}")) or len(str(c.get("fmt", ""))) > 16:
                errs.append(f"{w}: fmt must contain {{v}} and be short")
        elif typ == "status":
            for mk, mv in _items(c.get("map"), f"{w}.map", errs):
                _text(mv, MAX_LABEL, f"{w}.map.{mk}", errs)
        elif typ == "file":
            if c.get("op") not in FILE_OPS:
                errs.append(f"{w}: op must be one of {FILE_OPS}")
            if not isinstance(c.get("max_bytes"), int) or not 0 < c["max_bytes"] <= 65536:
                errs.append(f"{w}: max_bytes must be 1..65536")
        elif typ == "group":
            if depth >= MAX_DEPTH:
                errs.append(f"{w}: nested deeper than {MAX_DEPTH}")
            elif not isinstance(c.get("controls", []), list):
                errs.append(f"{w}: controls must be a list")
            else:
                for i, sub in enumerate(c.get("controls") or []):
                    control(sub, depth + 1, f"{w}[{i}]")

    for pi, page in enumerate(pages):
        if not isinstance(page, dict) or not isinstance(page.get("id"), str) or not _ID.match(page["id"]):
            errs.append(f"page {pi}: bad id")
            continue
        _text(page.get("title"), MAX_LABEL, f"page {page['id']}.title", errs)
        if not isinstance(page.get("controls"), list):
            errs.append(f"page {page['id']}: controls must be a list")
            continue
        for ci, c in enumerate(page["controls"]):
            control(c, 1, f"{page['id']}[{ci}]")
    if len(manifest_bytes(m)) > P.MAX_BODY:
        errs.append(f"manifest is larger than {P.MAX_BODY} bytes")
    return errs


@dataclass(frozen=True)
class Verdict:
    ok: bool
    code: int = 0
    detail: str = ""


class Manifest:
    """The device's own reading of its manifest: what every SET / ACT is checked against."""

    def __init__(self, m: Optional[dict] = None) -> None:
        self.m = m if m is not None else default_manifest()
        errs = validate_manifest(self.m)
        if errs:
            raise ValueError("invalid manifest: " + "; ".join(errs[:5]))
        self.by_key: dict[str, dict] = {}

        def walk(cs):
            for c in cs:
                if c.get("key"):
                    self.by_key[c["key"]] = c
                walk(c.get("controls") or [])
        for p in self.m["pages"]:
            walk(p["controls"])
        self.files = {c["op"]: c for p in self.m["pages"] for c in _flat(p["controls"]) if c["type"] == "file"}
        self.raw = manifest_bytes(self.m)
        self.hash = manifest_hash(self.m)
        self.rev = self.m["rev"]

    def check_set(self, key: Any, value: Any) -> Verdict:
        c = self.by_key.get(key) if isinstance(key, str) else None
        if c is None or c["type"] not in ("toggle", "stepper"):
            return Verdict(False, P.E.BAD_KEY, "no such control")
        if c["type"] == "toggle":
            return Verdict(True) if isinstance(value, bool) else Verdict(False, P.E.BAD_VALUE, "a boolean was expected")
        if isinstance(value, bool) or not isinstance(value, int) or not c["min"] <= value <= c["max"]:
            return Verdict(False, P.E.BAD_VALUE, f"{c['min']}..{c['max']} was expected")
        return Verdict(True)

    def check_act(self, key: Any, confirmed: bool) -> Verdict:
        c = self.by_key.get(key) if isinstance(key, str) else None
        if c is None or c["type"] != "action":
            return Verdict(False, P.E.BAD_KEY, "no such action")
        if c.get("confirm") == "two-step" and not confirmed:
            return Verdict(False, P.E.NOT_ALLOWED, "confirm")
        return Verdict(True)

    def max_file_bytes(self, op: str) -> int:
        c = self.files.get(op)
        return c["max_bytes"] if c else 0


def _flat(cs):
    for c in cs:
        yield c
        yield from _flat(c.get("controls") or [])
