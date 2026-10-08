"""The records a device is born with and the chain that makes them trustworthy.

    manufacturer key  ──signs──►  Attestation(serial, hw_id, DAK_pub, board_id)          (at the factory, once)
    DAK (inside the secure element, never leaves)  ──signs──►  DeviceCert(serial, owner card digest)     (on the device, after every owner-key change)
    owner keys (ed25519 + x25519, the 'Device Public ID' of docs/SECURITY.md)  ──►  Card.device = DeviceCert

A receiver that knows the manufacturer's public key can therefore tell a genuine device with serial S from a page or a script that merely
invents a card (the gap named in docs/SECURITY.md section 7).
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Optional

from ..ctl.identity import B32, Card, _lib

ATT_DOMAIN, DEV_DOMAIN, SVC_DOMAIN, POP_DOMAIN = b"DOATT1", b"DODEV1", b"DOSVC1", b"DOPOP1"
HW_ID_LEN = 8

# lifecycle: forward only; RMA -> PROVISIONED only through a signed service token
BLANK, DEV_TESTED, PROVISIONED, SHIPPED, IN_FIELD, RMA = "BLANK", "DEV_TESTED", "PROVISIONED", "SHIPPED", "IN_FIELD", "RMA"
LIFECYCLE = (BLANK, DEV_TESTED, PROVISIONED, SHIPPED, IN_FIELD, RMA)
TRANSITIONS = {BLANK: (DEV_TESTED,), DEV_TESTED: (PROVISIONED,), PROVISIONED: (SHIPPED,), SHIPPED: (IN_FIELD, RMA), IN_FIELD: (RMA,), RMA: (PROVISIONED,)}
SERVICE_ACTIONS = ("factory_return", "reprovision")


class RecordError(ValueError):
    """`key`: serial | format | signature | hardware | binding | lifecycle | token"""

    def __init__(self, key: str, msg: str = "") -> None:
        super().__init__(msg or key)
        self.key = key


def canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _unb64(s, n: int, what: str) -> bytes:
    try:
        b = base64.b64decode(s, validate=True)
    except (ValueError, TypeError):
        raise RecordError("format", f"{what}: not base64") from None
    if len(b) != n:
        raise RecordError("format", f"{what}: must be {n} bytes")
    return b


# ---------------------------------------------------------------------------------------------------------------------------- serial
_SERIAL = re.compile(r"^DO([1-9])-(\d{2})(\d{2})-(\d{5})-([0-9A-Z])$")


def _check_char(body: str) -> str:
    return B32[sum((i + 1) * ord(c) for i, c in enumerate(body)) % 31]            # 31 is prime: any single changed digit is caught


def make_serial(sku_n: int, year: int, week: int, seq: int) -> str:
    """DO<sku>-<yy><ww>-<seq>-<check>: the SKU digit, the production year and week, a five-digit sequence, one check character."""
    if not (1 <= sku_n <= 9 and 0 <= year <= 99 and 1 <= week <= 53 and 0 <= seq <= 99999):
        raise RecordError("serial", "out of range")
    body = f"DO{sku_n}{year:02d}{week:02d}{seq:05d}"
    return f"DO{sku_n}-{year:02d}{week:02d}-{seq:05d}-{_check_char(body)}"


def parse_serial(s: str) -> dict:
    m = _SERIAL.match(s) if isinstance(s, str) else None
    if not m:
        raise RecordError("serial", "not a serial number")
    sku_n, yy, ww, seq, chk = m.groups()
    if not 1 <= int(ww) <= 53 or _check_char(f"DO{sku_n}{yy}{ww}{seq}") != chk:
        raise RecordError("serial", "check character or week is wrong")
    return {"sku": int(sku_n), "year": int(yy), "week": int(ww), "seq": int(seq)}


def normalize_serial(s: str) -> str:
    """What a person typed from the label: case and look-alikes tolerated, the structure checked."""
    t = "".join(c for c in str(s).upper() if c.isalnum())
    if len(t) != 13 or t[0] != "D" or t[1] not in "O0":
        raise RecordError("serial", "a serial is DOn-yyww-sssss-c")
    t = "DO" + t[2:].translate(str.maketrans("ILOU", "1100"))          # after the fixed 'DO' the look-alikes are digits (the check char is tried as typed)
    out = f"DO{t[2]}-{t[3:7]}-{t[7:12]}-{t[12]}"
    parse_serial(out)
    return out


# ------------------------------------------------------------------------------------------------------------------------- attestation
def vendor_id(vendor_pub: bytes) -> str:
    return hashlib.sha256(b"DOVND1" + vendor_pub).hexdigest()[:16]


@dataclass(frozen=True)
class Attestation:
    serial: str
    hw_id: bytes
    dak: bytes
    board: bytes                           # SHA-256 over the unique ids of the chips on this board (MCU uid, SoM chip id, eMMC cid)
    date: str                              # YYYY-MM-DD of provisioning (a label, not a clock: the device has none)
    vendor: str                            # vendor_id(vendor_pub)
    sig: bytes = b""

    def body(self) -> bytes:
        return ATT_DOMAIN + canon({"v": 1, "serial": self.serial, "hw": self.hw_id.hex(), "dak": _b64(self.dak), "board": _b64(self.board),
                                   "date": self.date, "vendor": self.vendor})

    def verify(self, vendor_pub: bytes) -> bool:
        _, ed25519, _ = _lib()
        if vendor_id(vendor_pub) != self.vendor:
            return False
        try:
            ed25519.Ed25519PublicKey.from_public_bytes(vendor_pub).verify(self.sig, self.body())
            return True
        except Exception:
            return False

    def to_json(self) -> dict:
        return {"v": 1, "serial": self.serial, "hw": self.hw_id.hex(), "dak": _b64(self.dak), "board": _b64(self.board), "date": self.date,
                "vendor": self.vendor, "sig": _b64(self.sig)}

    @staticmethod
    def from_json(o) -> "Attestation":
        if not isinstance(o, dict) or o.get("v") != 1:
            raise RecordError("format", "not an attestation")
        for k in ("serial", "hw", "dak", "board", "date", "vendor", "sig"):
            if not isinstance(o.get(k), str):
                raise RecordError("format", f"missing {k}")
        parse_serial(o["serial"])
        try:
            hw = bytes.fromhex(o["hw"])
        except ValueError:
            raise RecordError("format", "hw") from None
        if len(hw) != HW_ID_LEN or len(o["date"]) != 10 or len(o["vendor"]) != 16:
            raise RecordError("format", "lengths")
        return Attestation(o["serial"], hw, _unb64(o["dak"], 32, "dak"), _unb64(o["board"], 32, "board"), o["date"], o["vendor"],
                           _unb64(o["sig"], 64, "sig"))


@dataclass(frozen=True)
class DeviceCert:
    """The DAK's signature over (serial, the owner card's digest), carried together with the attestation that vouches for the DAK."""
    att: Attestation
    sig: bytes

    @staticmethod
    def body(serial: str, owner_digest: bytes) -> bytes:
        return DEV_DOMAIN + serial.encode("ascii") + owner_digest

    def to_json(self) -> dict:
        return {"att": self.att.to_json(), "sig": _b64(self.sig)}

    @staticmethod
    def from_json(o) -> "DeviceCert":
        if not isinstance(o, dict) or not isinstance(o.get("sig"), str):
            raise RecordError("format", "not a device certificate")
        return DeviceCert(Attestation.from_json(o.get("att")), _unb64(o["sig"], 64, "sig"))


def verify_chain(card: Card, vendor_pub: bytes, hw_id: Optional[bytes] = None) -> str:
    """The serial this card belongs to, or a RecordError: manufacturer -> attestation -> DAK -> owner card. Nothing is trusted that is not checked."""
    _, ed25519, _ = _lib()
    dev = getattr(card, "device", None)
    if not dev:
        raise RecordError("binding", "the card carries no device certificate")
    cert = DeviceCert.from_json(dev)
    if not cert.att.verify(vendor_pub):
        raise RecordError("signature", "the attestation is not signed by this manufacturer")
    if hw_id is not None and cert.att.hw_id != hw_id.ljust(HW_ID_LEN, b"\0")[:HW_ID_LEN]:
        raise RecordError("hardware", "attested for other hardware")
    try:
        ed25519.Ed25519PublicKey.from_public_bytes(cert.att.dak).verify(cert.sig, DeviceCert.body(cert.att.serial, card.digest))
    except Exception:
        raise RecordError("binding", "the device key did not sign this card") from None
    return cert.att.serial


# --------------------------------------------------------------------------------------------------------------------------- the record
@dataclass
class DeviceRecord:
    serial: str
    sku: str
    hw_id: bytes
    att: Attestation
    factory_fw: int                        # version of the factory (golden) image
    report_hash: str = ""                  # SHA-256 of the signed factory test report
    lifecycle: str = BLANK

    def to_json(self) -> dict:
        return {"serial": self.serial, "sku": self.sku, "hw": self.hw_id.hex(), "att": self.att.to_json(), "factory_fw": self.factory_fw,
                "report": self.report_hash, "lifecycle": self.lifecycle}

    @staticmethod
    def from_json(o) -> "DeviceRecord":
        if not isinstance(o, dict):
            raise RecordError("format", "not a record")
        att = Attestation.from_json(o.get("att"))
        if o.get("serial") != att.serial or o.get("lifecycle") not in LIFECYCLE:
            raise RecordError("format", "serial or lifecycle")
        return DeviceRecord(att.serial, str(o.get("sku", "")), att.hw_id, att, int(o.get("factory_fw", 0)), str(o.get("report", "")), o["lifecycle"])


# ------------------------------------------------------------------------------------------------------------------------ service token
@dataclass(frozen=True)
class ServiceToken:
    """The manufacturer's permission for one service action on ONE device: serial + action + the nonce this device itself issued."""
    serial: str
    action: str
    nonce: bytes
    sig: bytes = b""

    def body(self) -> bytes:
        return SVC_DOMAIN + self.serial.encode("ascii") + b"|" + self.action.encode("ascii") + b"|" + self.nonce

    def to_json(self) -> dict:
        return {"serial": self.serial, "action": self.action, "nonce": self.nonce.hex(), "sig": _b64(self.sig)}

    @staticmethod
    def from_json(o) -> "ServiceToken":
        if not isinstance(o, dict) or not all(isinstance(o.get(k), str) for k in ("serial", "action", "nonce", "sig")):
            raise RecordError("token", "not a token")
        try:
            nonce = bytes.fromhex(o["nonce"])
        except ValueError:
            raise RecordError("token", "nonce") from None
        if len(nonce) != 16 or o["action"] not in SERVICE_ACTIONS:
            raise RecordError("token", "nonce or action")
        parse_serial(o["serial"])
        return ServiceToken(o["serial"], o["action"], nonce, _unb64(o["sig"], 64, "sig"))


# ------------------------------------------------------------------------------------------------------------------ storage and resets
# where each piece of state lives and who may change it (docs/PROVISIONING.md section 3)
W_ONCE, W_MONOTONIC, W_USER, W_VENDOR = "write-once", "monotonic", "user-resettable", "vendor-only"


@dataclass(frozen=True)
class Item:
    key: str
    what: str
    where: str
    mutability: str
    levels: tuple[str, str, str, str] = field(default=("kept", "kept", "kept", "kept"))      # what L1 L2 L3 L4 do to it


LEVELS = (
    ("L1", "Очистить слот", "приложение (два шага + кнопка) · `slot.clear`", "активный слот"),
    ("L2", "Очистить все профили", "приложение · `erase.profile`; удержание CONFIRM 10 с", "все слоты, список доверенных, ключ диска"),
    ("L3", "Заводской сброс", "приложение · `factory.reset`; удержание CONFIRM 20 с", "всё личное + новые ключи владельца + забыть телефоны"),
    ("L4", "Полный возврат к заводскому состоянию", "только режим восстановления: токен производителя + присутствие", "L3 + образы обоих банков SoM и MCU из золотого образа"),
)
ITEMS = (
    Item("serial", "серийный номер и запись устройства (SKU, hw_id, аттестация)", "OTP / заблокированный слот secure element", W_ONCE),
    Item("vendor_key", "хэш открытого ключа производителя (для проверки токенов, образов и аттестации)", "OTP / eFuse", W_ONCE),
    Item("rollback_floor", "порог анти-отката прошивки", "OTP / eFuse / RPMB (монотонный счётчик)", W_MONOTONIC),
    Item("dak", "DAK: ключ подлинности устройства", "secure element (рождается внутри, не выходит)", W_ONCE),
    Item("lifecycle", "стадия жизненного цикла", "OTP (только вперёд)", W_MONOTONIC),
    Item("golden", "золотой образ восстановления (подписан)", "eMMC boot-раздел с аппаратной защитой записи", W_VENDOR),
    Item("owner_keys", "ключи владельца (Device Public ID) и его DAK-сертификат", "secure element / хранилище ключей", W_USER, ("kept", "kept", "replaced", "replaced")),
    Item("storage_key", "ключ диска (шифрует профили, настройки)", "secure element / хранилище ключей", W_USER, ("kept", "replaced", "replaced", "replaced")),
    Item("slot_active", "профиль, уровни, имя, манифест и модель значков активного слота", "eMMC data, зашифровано ключом слота", W_USER, ("erased", "erased", "erased", "erased")),
    Item("slots_other", "остальные слоты (в том числе их модели)", "eMMC data, зашифровано ключами слотов", W_USER, ("kept", "erased", "erased", "erased")),
    Item("settings", "настройки (помощь вкл/выкл, активный слот)", "eMMC data", W_USER, ("assist off", "erased", "erased", "erased")),
    Item("trust", "список доверенных отправителей и счётчики повтора", "eMMC data", W_USER, ("kept", "erased", "erased", "erased")),
    Item("pkg_pending", "загруженный, но не применённый пакет (канал B, каталог pkg/)", "eMMC data", W_USER, ("kept", "erased", "erased", "erased")),
    Item("bonds", "сопряжённые телефоны (привязки BLE)", "стек BlueZ", W_USER, ("kept", "kept", "erased", "erased")),
    Item("fw_som", "образы SoM, банки A/B", "eMMC", W_VENDOR, ("kept", "kept", "kept", "reinstalled")),
    Item("fw_mcu", "образы MCU моста, банки A/B", "QSPI-флеш моста", W_VENDOR, ("kept", "kept", "kept", "reinstalled")),
)


def item(key: str) -> Item:
    return next(i for i in ITEMS if i.key == key)
