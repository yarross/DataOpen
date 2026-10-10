"""Model & Profile Residency: the profile of a person's hand and the weights of their model live on the device and nowhere else
(docs/RESIDENCY.md).

The rule is not a setting and not a thing the interface hides. It is made of three things that can be checked:

  1. THERE IS NO CODE THAT HANDS THEM OUT. The protocol has no 'export', no 'get profile', no 'get model'; the manifest has no file
     operation that returns device data (only the public device card); the gateway has no function that seals the profile for anybody,
     itself included. What is not there can not be asked for.
  2. THE SURFACE IS CLOSED. The kinds of thing the phone can ask the device to say (`GET_KINDS`), the keys it can set, the actions it can
     start, the message types the device can send, the operations a file control can have: all of them are lists here, and a test pins
     them. Adding one is a decision somebody has to write down, not a side effect.
  3. EVERYTHING THE DEVICE SAYS PASSES ONE DOOR. `scrub` checks every answer against a schema (default-deny: a key that is not in the
     schema is an error, a string is short, an answer is small), so a future change that puts a field of the profile into a reply is
     refused instead of sent.

This module is data and checks. It never reads the profile or the weights.

What it does NOT do (and says so in the document): protect against root on the module while the keys are a file (a secure element is
needed), or against whoever can take the flash and the keys out of the hardware.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from . import protocol as P

# ---------------------------------------------------------------------------------------------------------------- classes
SECRET, PERSONAL, DERIVED, METADATA, PUBLIC = "secret", "personal", "derived", "metadata", "public"
CLASSES = (SECRET, PERSONAL, DERIVED, METADATA, PUBLIC)
CLASS_TITLES = {
    SECRET: "секрет: живёт только на устройстве, наружу не выходит ни в каком виде",
    PERSONAL: "личное: видно владельцу в приложении, файлом наружу не выходит",
    DERIVED: "грубая производная: несколько значений из фиксированного списка",
    METADATA: "метаданные: серийник, версии, состояния",
    PUBLIC: "открытое: можно передавать любому",
}

KEPT, ERASED, REPLACED, NA = "kept", "erased", "replaced", "-"


@dataclass(frozen=True)
class Rule:
    id: str
    what: str                 # the object
    cls: str
    read: str                 # what the phone can learn about it through the protocol
    export: bool              # can it be turned into a file or a message that carries it out? (nothing secret or personal: never)
    where: str                # where it lives
    l1: str                   # clear a slot
    l2: str                   # erase all profiles
    l3: str                   # factory reset
    l4: str                   # full return (service)
    note: str = ""


RULES: tuple[Rule, ...] = (
    Rule("profile", "моторный профиль слота (BioProfileV1, 98 байт)", SECRET, "никак; только 3 грубых признака готовности (ниже)", False,
         "слот k, файлы profile.a/.b под ключом слота", ERASED, ERASED, ERASED, ERASED, "L1 стирает только профиль активного слота"),
    Rule("profile_prev", "прежний профиль слота (для «Вернуть»)", SECRET, "никак", False, "слот k, profile.prev под ключом слота",
         ERASED, ERASED, ERASED, ERASED),
    Rule("model", "веса модели слота (model.bin) и прежняя модель (model.prev.bin)", SECRET, "никак; только карточка и состояние", False,
         "слот k, под ключом слота", ERASED, ERASED, ERASED, ERASED, "L1 стирает модель активного слота вместе с профилем"),
    Rule("derived_params", "параметры моста, посчитанные из профиля (asc_params, tremor_params)", SECRET, "никак", False,
         "память модуля и моста; по линии BridgeLink внутри устройства", ERASED, ERASED, ERASED, ERASED,
         "после стирания мосту уходит нейтральный блоб; на телефон не выходят никогда"),
    Rule("keys", "ключи: диска, слотов, владельца (ed, x), DAK", SECRET, "никак; наружу только открытые части карточки", False,
         "хранилище ключей (файл в симуляции, secure element в изделии)", KEPT, REPLACED, REPLACED, REPLACED,
         "DAK не меняется никем; ключ диска меняют L2–L4; ключи владельца L3–L4; ключ слота L1 (эпоха)"),
    Rule("pending", "загруженный, но не применённый пакет (pkg/pending.dopk)", SECRET, "только сводка (от кого, что внутри)", False,
         "pkg/, запечатан ключами устройства", KEPT, ERASED, ERASED, ERASED),
    Rule("levels", "два уровня человека (сила помощи, подавление дрожи)", PERSONAL, "значения видны в приложении", False,
         "слот k, meta под ключом слота", ERASED, ERASED, ERASED, ERASED),
    Rule("slot_name", "имя слота", PERSONAL, "видно в приложении", False, "слот k, meta", ERASED, ERASED, ERASED, ERASED),
    Rule("layout", "раскладка экранов слота (манифест)", PERSONAL, "отдаётся телефону, чтобы он её нарисовал; файлом не выгружается", False,
         "слот k, manifest.json под ключом слота", ERASED, ERASED, ERASED, ERASED,
         "раскладка может содержать только допустимые типы контролов; операций, возвращающих данные устройства, в ней быть не может"),
    Rule("trust", "список доверенных отправителей, счётчики повтора", PERSONAL, "только число", False, "trust.a/.b под ключом диска",
         KEPT, ERASED, ERASED, ERASED),
    Rule("progress", "готовность профиля: заполненность 0..100, готовые слои, признак «дрожь слишком мала»", DERIVED,
         "три поля состояния с фиксированными значениями", False, "считается из профиля при запросе", "-", "-", "-", "-",
         "единственная производная профиля, которая выходит из устройства; список закреплён тестом"),
    Rule("serial", "серийный номер, hw_id, стадия жизненного цикла, порог анти-отката", METADATA, "да", False,
         "OTP / заблокированный слот secure element", KEPT, KEPT, KEPT, KEPT),
    Rule("versions", "версии прошивки, банки, состояние обновления, состояние модели (нет / ok / needs_system / over_budget), состояние пути сцены (none / warming / ok / off)", METADATA, "да", False,
         "fw/, слот k", KEPT, KEPT, KEPT, KEPT, "состояние модели меняется с её удалением"),
    Rule("model_card", "карточка модели: имя, версия, классы, размер, короткий id (8 байт хэша)", METADATA, "да", False,
         "слот k, model.json", ERASED, ERASED, ERASED, ERASED, "полный SHA-256 весов наружу не выходит"),
    Rule("card", "карточка устройства: открытые ключи, номер, цепочка производителя", PUBLIC, "да (card_get)", True,
         "вычисляется из ключей", KEPT, KEPT, REPLACED, REPLACED, "единственное, что устройство отдаёт файлом: по карточке нельзя ничего прочитать"),
)
RULES_BY_ID = {r.id: r for r in RULES}

# ---------------------------------------------------------------------------------------------------------------- the closed surface
SET_KEYS = ("assist.on", "assist.strength", "tremor.level", "calib.running", "slot.name", "slot.active")
ACT_KEYS = ("profile.restore", "pairing.forget", "erase.profile", "factory.reset", "slot.clear", "fw.apply", "fw.rollback",
            "pkg.apply", "pkg.discard", "pkg.revert", "model.clear", "trust.clear")
FILE_OPS_IN, FILE_OPS_OUT, FILE_OPS = P.FILE_OPS_IN, P.FILE_OPS_OUT, P.FILE_OPS     # files go IN; the device hands out only its public card
OUTGOING_TYPES = (P.T_HELLO_R, P.T_DATA, P.T_ACK, P.T_ERR, P.T_EVENT, P.T_PONG)
INCOMING_TYPES = (P.T_HELLO, P.T_GET, P.T_SET, P.T_ACT, P.T_CONFIRM, P.T_STOP, P.T_HARD_BYPASS, P.T_BUNDLE_PUT, P.T_FW_BEGIN, P.T_FW_CHUNK,
                  P.T_FW_END, P.T_PKG_BEGIN, P.T_PKG_CHUNK, P.T_PKG_END, P.T_PING)

# names a request can use to ask for the secret: all answered with RESIDENT, on purpose, so that nobody wonders whether it is a bug
ASK_FOR_SECRET = ("bundle", "profile", "profile_prev", "model", "weights", "export", "backup", "slot_data", "params", "keys", "dump",
                  "copy", "download")

MAX_ANSWER = 4096                 # every answer except the manifest
MAX_EVENT = 1024
MAX_ERR_DETAIL = 96
MAX_STR = 64


class ResidencyViolation(Exception):
    """Something that is not on the list was about to leave the device."""


# ---------------------------------------------------------------------------------------------------------------- schemas
_SLOT_KEY = re.compile(rf"^slot\.[0-{P.SLOT_COUNT - 1}]\.name$")
STATE_KEYS = ("assist.on", "assist.strength", "tremor.level", "calib.running", "profile.fill", "profile.layers", "profile.tremor",
              "device.id", "device.serial", "trusted.count", "trial.left_s", "pairing.open", "slot.active", "slot.name", "fw.version",
              "fw.state", "pkg.state", "pkg.from", "pkg.kinds", "model.state", "model.name", "model.version", "model.from", "scene.state")
PROFILE_LAYERS = ("none", "asc", "tremor", "both")            # the only values a derived profile field can have
PROFILE_TREMOR = ("collecting", "ready", "not_needed")
FW_STATES = ("unsupported", "current", "staged", "trial")
PKG_STATES = ("none", "receiving", "pending")
MODEL_STATES = ("none", "ok", "needs_system", "over_budget")
SCENE_STATES = ("none", "warming", "ok", "off")      # the scene path's health (ui/health.py), told to the gateway by the module

MODEL_CARD_PUBLIC = ("format", "taxonomy", "classes", "n_keypoints", "input_size", "opset", "name", "version", "size", "id")


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _str(v, where: str, limit: int = MAX_STR) -> None:
    if not isinstance(v, str) or len(v) > limit:
        raise ResidencyViolation(f"{where}: a short string was expected")


def _keys(obj, allowed: tuple, where: str, required: tuple = ()) -> dict:
    if not isinstance(obj, dict):
        raise ResidencyViolation(f"{where}: an object was expected")
    extra = set(obj) - set(allowed)
    if extra:
        raise ResidencyViolation(f"{where}: keys that are not on the list: {sorted(extra)}")
    missing = set(required) - set(obj)
    if missing:
        raise ResidencyViolation(f"{where}: missing {sorted(missing)}")
    return obj


def _scalar(v, where: str) -> None:
    if v is None or isinstance(v, bool) or _is_int(v):
        return
    _str(v, where)


def scrub_state(tree: dict, where: str = "state") -> dict:
    if not isinstance(tree, dict):
        raise ResidencyViolation(f"{where}: an object was expected")
    for k, v in tree.items():
        if k not in STATE_KEYS and not _SLOT_KEY.match(k):
            raise ResidencyViolation(f"{where}.{k}: not a state key")
        _scalar(v, f"{where}.{k}")
    for k, allowed in (("profile.layers", PROFILE_LAYERS), ("profile.tremor", PROFILE_TREMOR), ("fw.state", FW_STATES),
                       ("pkg.state", PKG_STATES), ("model.state", MODEL_STATES), ("scene.state", SCENE_STATES)):
        if k in tree and tree[k] not in allowed:
            raise ResidencyViolation(f"{where}.{k}: {tree[k]!r} is not one of {allowed}")
    if "profile.fill" in tree and not (_is_int(tree["profile.fill"]) and 0 <= tree["profile.fill"] <= 100):
        raise ResidencyViolation(f"{where}.profile.fill: 0..100 was expected")
    return tree


def scrub_model_card(card, where: str = "model") -> Any:
    if card is None:
        return None
    _keys(card, MODEL_CARD_PUBLIC, where)
    if "id" in card:
        _str(card["id"], f"{where}.id", 16)
    if "classes" in card:
        if not isinstance(card["classes"], list) or len(card["classes"]) > 32:
            raise ResidencyViolation(f"{where}.classes: a short list was expected")
        for c in card["classes"]:
            _str(c, f"{where}.classes[]", 32)
    for k in ("format", "taxonomy", "name"):
        if k in card:
            _str(card[k], f"{where}.{k}", 48)
    for k in ("n_keypoints", "input_size", "opset", "version", "size"):
        if k in card and not _is_int(card[k]):
            raise ResidencyViolation(f"{where}.{k}: an integer was expected")
    return card


def scrub_slots(j: dict) -> dict:
    _keys(j, ("active", "count", "slots"), "slots", ("active", "count", "slots"))
    if not isinstance(j["slots"], list) or len(j["slots"]) > P.SLOT_COUNT:
        raise ResidencyViolation("slots: at most the number of slots")
    for s in j["slots"]:
        _keys(s, ("n", "name", "has", "vetted", "manifest", "strength", "tremor"), "slots[]", ("n", "name", "has"))
        _str(s["name"], "slots[].name", 24)
        for k in ("has", "vetted", "manifest"):
            if k in s and not isinstance(s[k], bool):
                raise ResidencyViolation(f"slots[].{k}: a boolean was expected")
        for k in ("n", "strength", "tremor"):
            if k in s and not _is_int(s[k]):
                raise ResidencyViolation(f"slots[].{k}: an integer was expected")
    return j


def scrub_firmware(j: dict) -> dict:
    _keys(j, ("supported", "running", "active", "trial", "boots", "floor", "versions", "upload", "max_size"), "firmware", ("supported",))
    if "versions" in j:
        _keys(j["versions"], ("A", "B"), "firmware.versions")
        for v in j["versions"].values():
            if v is not None and not _is_int(v):
                raise ResidencyViolation("firmware.versions: an integer or none was expected")
    if j.get("upload") is not None:
        _keys(j["upload"], ("next", "size"), "firmware.upload")
    return j


def scrub_packages(j: dict) -> dict:
    _keys(j, ("supported", "schema", "max_size", "pending", "upload", "models"), "packages", ("supported",))
    p = j.get("pending")
    if p is not None:
        _keys(p, ("id", "kinds", "slot", "from", "self", "seq", "min_fw", "button", "model", "name"), "packages.pending")
        if p.get("model") is not None:
            _keys(p["model"], ("name", "version", "size"), "packages.pending.model")
        if "name" in p and p["name"] is not None:
            _str(p["name"], "packages.pending.name", 48)
        if not isinstance(p.get("kinds", []), list) or len(p.get("kinds", [])) > 8:
            raise ResidencyViolation("packages.pending.kinds: a short list was expected")
    if j.get("upload") is not None:
        _keys(j["upload"], ("next", "size"), "packages.upload")
    for m in j.get("models", []):
        _keys(m, ("n", "model", "state", "previous"), "packages.models[]", ("n", "state"))
        scrub_model_card(m.get("model"), "packages.models[].model")
        if m["state"] not in MODEL_STATES:
            raise ResidencyViolation("packages.models[].state: not a model state")
    return j


_CARD_KEYS = ("v", "ed", "x", "id", "label", "created", "sig", "device")


def scrub_identity(j: dict) -> dict:
    _keys(j, _CARD_KEYS, "identity", ("v", "ed", "x", "id", "sig"))
    for k in ("ed", "x", "sig"):
        _str(j[k], f"identity.{k}", 128)
    for k in ("id", "label", "created"):
        if k in j:
            _str(j[k], f"identity.{k}", 64)
    return j


def scrub_state_answer(j: dict) -> dict:
    _keys(j, ("rev", "state"), "state", ("rev", "state"))
    scrub_state(j["state"])
    return j


SCRUBBERS = {P.GET_STATE: scrub_state_answer, P.GET_SLOTS: scrub_slots, P.GET_FIRMWARE: scrub_firmware, P.GET_PACKAGES: scrub_packages,
             P.GET_IDENTITY: scrub_identity}


def public_card(card):
    """What the phone may see of a model's card: the description, and a short id instead of the full hash of the weights."""
    if not isinstance(card, dict):
        return None
    out = {k: card[k] for k in MODEL_CARD_PUBLIC if k in card and k != "id"}
    h = card.get("sha256")
    if isinstance(h, str) and h:
        out["id"] = h[:16]
    return out


def scrub(kind: str, obj):
    """The answer to a GET, checked against its schema. Returns it unchanged or raises `ResidencyViolation`. (The manifest is checked by
    `validate_manifest`, which has no operation that returns device data.)"""
    if kind not in P.GET_KINDS:
        raise ResidencyViolation(f"{kind!r} is not something the device says")
    if kind == P.GET_MANIFEST:
        return obj
    if kind not in SCRUBBERS:
        raise ResidencyViolation(f"no schema for {kind!r}")
    return SCRUBBERS[kind](obj)


def scrub_event(j: dict) -> dict:
    _keys(j, ("rev", "state"), "event", ("rev", "state"))
    scrub_state(j["state"], "event.state")
    return j


def scrub_err(j: dict) -> dict:
    _keys(j, ("code", "key", "detail"), "error", ("code", "key"))
    if "detail" in j:
        _str(j["detail"], "error.detail", MAX_ERR_DETAIL)
    return j


# ---------------------------------------------------------------------------------------------------------------- generated tables
def _row(*cells) -> str:
    return "| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |"


def rules_table() -> str:
    rows = [_row("объект", "класс", "что видит телефон", "можно ли выгрузить", "где живёт", "L1", "L2", "L3", "L4"), _row(*["---"] * 9)]
    for r in RULES:
        rows.append(_row(f"**{r.what}**", r.cls, r.read, "**нет**" if not r.export else "да, только открытое", r.where, r.l1, r.l2, r.l3, r.l4))
    return "\n".join(rows)


def classes_table() -> str:
    return "\n".join([_row("класс", "что значит"), _row("---", "---")] + [_row(f"`{c}`", CLASS_TITLES[c]) for c in CLASSES])


def surface_table() -> str:
    rows = [_row("что", "полный список"), _row("---", "---")]
    rows.append(_row("`GET` (что устройство может сказать телефону)", ", ".join(f"`{k}`" for k in P.GET_KINDS)))
    rows.append(_row("`SET` (что телефон может задать)", ", ".join(f"`{k}`" for k in SET_KEYS)))
    rows.append(_row("`ACT` (что телефон может запустить)", ", ".join(f"`{k}`" for k in ACT_KEYS)))
    rows.append(_row("файл: в устройство", ", ".join(f"`{k}`" for k in FILE_OPS_IN)))
    rows.append(_row("файл: из устройства", ", ".join(f"`{k}`" for k in FILE_OPS_OUT) + " (открытая карточка)"))
    names = {v: k for k, v in P.TYPES.items()}
    rows.append(_row("сообщения от устройства", ", ".join(f"`{names[t]}`" for t in OUTGOING_TYPES)))
    rows.append(_row("запрос секрета (`what` = …)", ", ".join(f"`{k}`" for k in ASK_FOR_SECRET) + f" → ошибка `RESIDENT` ({P.E.RESIDENT})"))
    rows.append(_row("размеры", f"ответ ≤ {MAX_ANSWER} байт (кроме раскладки), событие ≤ {MAX_EVENT}, `detail` ошибки ≤ {MAX_ERR_DETAIL} символов, "
                                f"строка ≤ {MAX_STR}"))
    return "\n".join(rows)


def state_table() -> str:
    rows = [_row("ключ состояния", "класс"), _row("---", "---")]
    derived = {"profile.fill", "profile.layers", "profile.tremor"}
    personal = {"assist.on", "assist.strength", "tremor.level", "slot.name", "slot.active", "calib.running", "trusted.count"}
    for k in STATE_KEYS + ("slot.<k>.name",):
        cls = DERIVED if k in derived else PERSONAL if (k in personal or k.startswith("slot.")) else METADATA
        rows.append(_row(f"`{k}`", cls))
    return "\n".join(rows)


TABLES = {"classes": classes_table, "rules": rules_table, "surface": surface_table, "state": state_table}
_BLOCK = re.compile(r"(<!-- fp:(\w+) -->\n)(.*?)(<!-- /fp:\2 -->)", re.S)


def render_doc(text: str) -> str:
    def sub(m):
        name = m.group(2)
        return m.group(1) + TABLES[name]() + "\n" + m.group(4) if name in TABLES else m.group(0)
    return _BLOCK.sub(sub, text)


def doc_is_current(path) -> bool:
    from pathlib import Path
    text = Path(path).read_text(encoding="utf-8")
    return render_doc(text) == text and {m.group(2) for m in _BLOCK.finditer(text)} == set(TABLES)
