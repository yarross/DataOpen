"""The device manifest: which controls the phone shows, and what values the device will accept for them (docs/PWA.md section 4).

The client is deliberately dumb: it renders nine control types and knows no screen by name. The same object that is sent to the phone is
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
TYPES = ("status", "toggle", "stepper", "action", "meter", "note", "file", "group", "text")
CONFIRMS = ("none", "revert", "two-step")
FILE_OPS = P.FILE_OPS             # files go IN, plus the public card: nothing else leaves the device (docs/RESIDENCY.md)
MAX_TEXT = 40
FW_MAX_BYTES = 64 * 1024 * 1024
PKG_MAX_BYTES = 21 * 1024 * 1024   # a package: up to 20 MiB of content plus the sealing
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
                 "help": L("Откроется только файл, сделанный для этого устройства. Из устройства профиль не выгружается никуда.",
                           "Only a file made for this device will open. The profile is never taken out of the device.")},
                {"id": "restore", "type": "action", "key": "profile.restore", "confirm": "two-step",
                 "label": L("Вернуть прежний профиль", "Restore the previous profile")},
                {"id": "slot_name", "type": "text", "key": "slot.name", "maxlen": 24,
                 "label": L("Название слота", "Slot name"),
                 "help": L("Например «Работа» или «Браузер». Название видно только на этом устройстве и в телефоне.",
                           "For example \"Work\" or \"Browser\". Shown only on this device and in the phone.")},
                {"id": "slot_clear", "type": "action", "key": "slot.clear", "confirm": "two-step", "danger": True,
                 "label": L("Очистить этот слот", "Clear this slot"),
                 "help": L("Стирает профиль, уровни и название только в этом слоте. Помощь выключится. Нужна кнопка на устройстве.",
                           "Erases the profile, levels and name of this slot only. Assistance turns off. Needs the button on the device.")},
            ]},
            {"id": "more", "title": L("Ещё", "More"), "controls": [
                {"id": "serial", "type": "status", "key": "device.serial", "label": L("Серийный номер (как на наклейке)",
                                                                             "Serial number (as on the label)")},
                {"id": "device_id", "type": "status", "key": "device.id", "label": L("Номер владельца (меняется заводским сбросом)",
                                                                                "Owner number (changes with a factory reset)")},
                {"id": "trusted", "type": "status", "key": "trusted.count",
                 "label": L("Откуда принимаются файлы (кроме своих копий)",
                            "Where files are accepted from (besides your own copies)")},
                {"id": "card", "type": "file", "op": "card_get", "accept": ".docard", "max_bytes": 4096,
                 "label": L("Сохранить карточку устройства", "Save the device card"),
                 "help": L("Карточку можно передавать: по ней нельзя прочитать ничего личного, "
                           "можно только подготовить файл для этого устройства.",
                           "The card is safe to share: it reveals nothing personal, "
                           "it only lets a file be prepared for this device.")},
                {"id": "fw_version", "type": "status", "key": "fw.version", "label": L("Версия прошивки", "Firmware version")},
                {"id": "fw_state", "type": "status", "key": "fw.state",
                 "map": {"unsupported": L("Обновление не поддерживается", "Updates are not supported"),
                         "current": L("Установлена, обновлений нет", "Installed, nothing pending"),
                         "staged": L("Загружено, ждёт применения", "Loaded, waiting to be applied"),
                         "trial": L("Новая версия проверяется", "The new version is being checked")},
                 "label": L("Обновление", "Update")},
                {"id": "fw_put", "type": "file", "op": "fw_put", "accept": ".dofw", "max_bytes": 4194304,
                 "label": L("Загрузить обновление прошивки", "Load a firmware update"),
                 "help": L("Файл должен быть подписан производителем. Он ложится во вторую область памяти, пока работает прежняя версия.",
                           "The file must be signed by the manufacturer. It goes into the second memory bank "
                           "while the current version keeps running.")},
                {"id": "fw_apply", "type": "action", "key": "fw.apply", "confirm": "two-step",
                 "label": L("Применить обновление", "Apply the update"),
                 "help": L("Устройство перезапустится; если новая версия не заработает, прежняя вернётся сама. "
                           "Нужна кнопка на устройстве.",
                           "The device restarts; if the new version does not work, the previous one comes back. "
                           "Needs the button on the device.")},
                {"id": "fw_rollback", "type": "action", "key": "fw.rollback", "confirm": "two-step",
                 "label": L("Вернуть прежнюю версию", "Go back to the previous version"),
                 "help": L("Потребуется нажать кнопку на устройстве.", "You will need to press the button on the device.")},
                {"id": "pkg_state", "type": "status", "key": "pkg.state",
                 "map": {"none": L("Пакета нет", "No package"), "receiving": L("Загружается", "Loading"),
                         "pending": L("Проверен, ждёт применения", "Checked, waiting")},
                 "label": L("Пакет для слотов", "Package for the slots")},
                {"id": "pkg_put", "type": "file", "op": "pkg_put", "accept": ".dopk", "max_bytes": PKG_MAX_BYTES,
                 "label": L("Загрузить пакет", "Load a package"),
                 "help": L("Только для этого устройства. Сначала проверка, применение отдельно.",
                           "For this device only. Checked first, applied separately.")},
                {"id": "pkg_apply", "type": "action", "key": "pkg.apply", "confirm": "two-step",
                 "label": L("Применить пакет", "Apply the package"),
                 "help": L("Может понадобиться кнопка на устройстве.", "The button on the device may be needed.")},
                {"id": "pkg_discard", "type": "action", "key": "pkg.discard", "confirm": "none",
                 "label": L("Отменить пакет", "Drop the package")},
                {"id": "model_state", "type": "status", "key": "model.state",
                 "map": {"none": L("Своей нет", "None of its own"), "ok": L("Установлена", "Installed"),
                         "needs_system": L("Нужна новая система", "Needs a newer system")},
                 "label": L("Модель значков в слоте", "Icon model in this slot")},
                {"id": "model_revert", "type": "action", "key": "pkg.revert", "confirm": "two-step",
                 "label": L("Вернуть прежнюю модель", "Previous model")},
                {"id": "model_from", "type": "status", "key": "model.from",
                 "label": L("Модель принесена отправителем (номер)", "Model brought by sender (number)")},
                {"id": "model_clear", "type": "action", "key": "model.clear", "confirm": "two-step",
                 "label": L("Убрать модель из слота", "Remove the model from this slot"),
                 "help": L("Профиль и настройки слота остаются. Кнопка на устройстве не нужна: помощь становится проще, а не опаснее.",
                           "The profile and settings of the slot stay. No button needed: it only takes capability away.")},
                {"id": "trust_clear", "type": "action", "key": "trust.clear", "confirm": "two-step", "danger": True,
                 "label": L("Забыть доверенных отправителей", "Forget trusted senders"),
                 "help": L("Следующий пакет от любого отправителя снова спросит кнопку. Модели и профили в слотах остаются. "
                           "Нужна кнопка на устройстве.",
                           "The next package from any sender asks for the button again. Models and profiles in the slots stay. "
                           "Needs the button on the device.")},
                {"id": "forget", "type": "action", "key": "pairing.forget", "confirm": "two-step", "danger": True,
                 "label": L("Забыть все телефоны", "Forget all phones"),
                 "help": L("Потребуется нажать кнопку на самом устройстве.", "You will need to press the button on the device itself.")},
                {"id": "erase", "type": "action", "key": "erase.profile", "confirm": "two-step", "danger": True,
                 "label": L("Стереть личные данные (все слоты)", "Erase personal data (all slots)"),
                 "help": L("Удаляет профили, настройки и доверенных во всех слотах; помощь выключится. "
                           "Нужна кнопка на устройстве или удержание 10 с.",
                           "Deletes profiles, settings and trusted senders in every slot; assistance turns off. "
                           "Needs the button, or holding it for 10 s.")},
                {"id": "factory", "type": "action", "key": "factory.reset", "confirm": "two-step", "danger": True,
                 "label": L("Заводской сброс", "Factory reset"),
                 "help": L("То же, и новый номер владельца (серийный остаётся): старые файлы не откроются, "
                           "телефоны забываются. Нужна кнопка или удержание 20 с.",
                           "The same, plus a new owner number (the serial stays): old files stop opening, "
                           "phones are forgotten. Needs the button, or holding it for 20 s.")},
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
        needs_key = typ in ("status", "toggle", "stepper", "action", "meter", "text")
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
            cap = FW_MAX_BYTES if c.get("op") == "fw_put" else PKG_MAX_BYTES if c.get("op") == "pkg_put" else 65536
            if not isinstance(c.get("max_bytes"), int) or isinstance(c.get("max_bytes"), bool) or not 0 < c["max_bytes"] <= cap:
                errs.append(f"{w}: max_bytes must be 1..{cap}")
            if "scope" in c:
                errs.append(f"{w}: a file control has no scope (files only come in)")
        elif typ == "text":
            if not isinstance(c.get("maxlen"), int) or isinstance(c.get("maxlen"), bool) or not 1 <= c["maxlen"] <= MAX_TEXT:
                errs.append(f"{w}: maxlen must be 1..{MAX_TEXT}")
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
        if c is None or c["type"] not in ("toggle", "stepper", "text"):
            return Verdict(False, P.E.BAD_KEY, "no such control")
        if c["type"] == "text":
            ok = isinstance(value, str) and len(value) <= c["maxlen"]
            return Verdict(True) if ok else Verdict(False, P.E.BAD_VALUE, f"text of at most {c['maxlen']} characters was expected")
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
