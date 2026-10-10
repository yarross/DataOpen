"""The two update channels of the device and the rules that keep them apart (docs/UPDATES.md).

    A  SYSTEM        the code that runs the device (SoM system, bridge MCU, BLE stack, bootloader, recovery image).
                     One file for every unit of a hardware type; signed by the MANUFACTURER; dual bank; anti-rollback; applied by a button.
    B  PACKAGES      what a person owns (profile, levels, UI layout, the weights of a light model).
                     One file for ONE device; sealed to its Device Public ID, signed by the SENDER (the person or someone they trust).

The point of the split is blast radius and trust. A can brick the device, so it is signed offline, staged in a bank, tried, and takes a
physical button. B can not brick anything (it is data: the safety shell and the bridge's `output <= input` clamp do not depend on it), so
it is cheap to receive, but it is personal, so it is encrypted, bound to one device and attributed to a sender. Nothing a B package carries
is ever executed, and nothing in A ever reads a sender's key. The two share NOTHING on the wire: different file magic, different message
types (`FW_*` / `PKG_*`), different error codes, different keys, different storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

MAGIC_A, MAGIC_B, MAGIC_B_SMALL = b"DOFW", b"DOPK", b"DOBS"
CHANNEL_OF_MAGIC = {MAGIC_A: "A", MAGIC_B: "B", MAGIC_B_SMALL: "B"}


def channel_of(prefix: bytes) -> Optional[str]:
    """Which channel a file belongs to, from its first four bytes (None: neither)."""
    return CHANNEL_OF_MAGIC.get(bytes(prefix[:4]))


@dataclass(frozen=True)
class Channel:
    key: str
    name: str
    includes: str
    signer: str
    trust_root: str
    audience: str
    delivery: str
    binding: str
    confidentiality: str
    replay: str
    approval: str
    staging: str
    probation: str
    rollback: str
    transport: str
    blast_radius: str
    risk: str


CHANNELS = (
    Channel(
        "A", "Система",
        "ядро (SoM), прошивка моста, BLE-стек, загрузчик, образ восстановления",
        "производитель: ключ вне устройства и вне репозитория, в HSM; на устройстве только открытый ключ",
        "открытый ключ производителя (OTP / secure element)",
        "все устройства одного типа (`hw_id`): один и тот же файл",
        "открыто: сайт, поддержка, магазин; через телефон или утилиту",
        "тип железа (`hw_id`), не устройство",
        "нет: файл публичный, нужны целостность и подлинность",
        "порог анти-отката (`min_version`) только растёт; та же версия отвергается",
        "кнопка на устройстве при применении; применяется не сама",
        "неработающий банк целиком; работающий не трогается",
        "пробная загрузка: `MAX_BOOTS` попыток, новая версия сама подтверждает себя; аппаратный обход не даёт подтвердить",
        "автоматически на прежний банк; вручную `fw.rollback` (кнопка); не ниже порога анти-отката",
        "`FW_BEGIN/CHUNK/END`, ошибка `FW_REJECTED`",
        "может сделать устройство неработоспособным (поэтому два банка, золотой образ и аппаратный обход мыши)",
        "подпись одного ключа даёт код на всех устройствах: ключ хранится вне сети; файл с известной дырой нельзя вернуть (анти-откат)"),
    Channel(
        "B", "Пакеты",
        "профиль, уровни помощи, UI-манифест, имя слота, веса лёгкой модели интерфейсного детектора",
        "отправитель: тот, кому человек доверяет (клиника, техник, другой человек); устройство пакетов само не делает; ключ Ed25519 отправителя",
        "карточка отправителя в списке доверенных (первый раз кнопкой); ключ производителя тут не принимается",
        "одно устройство: Device Public ID получателя внутри ключа шифрования и внутри подписанного заголовка",
        "через владельца: телефон (PWA), почта, облако; сам файл безопасно хранить где угодно",
        "ключи владельца (Device Public ID); после заводского сброса файл для старого номера не откроется",
        "всегда: X25519 + HKDF + ChaCha20-Poly1305 по частям, у каждого устройства свой ключ",
        "`seq` отправителя только растёт (повтор отвергается); пакет, сделанный самим этим устройством, отвергается (`own_file`)",
        "кнопка: первый раз от нового отправителя, каждый раз для весов модели; иначе проба «Оставить / Вернуть»",
        "каталог устройства `pkg/`, зашифрован ключом диска; один пакет за раз; слот не меняется до применения",
        "для профиля и уровней: проба 20 с; для модели: проверка при установке и проба, прежняя модель хранится",
        "прежнее поколение хранится в слоте (`profile.prev`, `model.prev`); `pkg.revert`; порога нет",
        "`PKG_BEGIN/CHUNK/END`, ошибка `PKG_REJECTED`",
        "не может сломать устройство: это данные; худший итог: помощь выключена, слот очищается (оболочка безопасности и обход от пакета не зависят)",
        "подмена получателя, повтор, чужая модель, подмена по частям, «усечённый» файл; веса модели не код, но влияют на то, что считается целью"),
)
BY_KEY = {c.key: c for c in CHANNELS}

# what each channel's verifier accepts as a SIGNER: keys are never shared between the channels
INVARIANTS = (
    ("signer_a", "файл канала A проверяется только по ключу производителя; подпись любого отправителя не годится"),
    ("signer_b", "пакет канала B проверяется по ключу его отправителя; подпись производителя не делает пакет доверенным"),
    ("magic", "каждый канал принимает только свою «магию» файла: DOFW в PKG_* и DOPK/DOBS в FW_* отвергаются в первом же куске"),
    ("transport", "у каналов разные типы сообщений и разные коды ошибок; один не может быть выдан за другой"),
    ("never_code", "из пакета канала B ничего не исполняется: ни скрипт, ни код модели, ни операторы вне списка"),
    ("storage", "пакет канала B не пишет в банки прошивки; обновление канала A не пишет в слоты и пакеты"),
    ("fw_trial", "пока новая прошивка на пробе, пакеты не применяются (она ещё не доказала, что читает их); загруженный пакет при этом ждёт"),
)

# ---------------------------------------------------------------------------------------------------------------- compatibility
PKG_SCHEMA = 1  # the package schema this firmware understands (grows with the formats inside)


@dataclass(frozen=True)
class Reason:
    """One way an update does not happen. `key` is the PKG_REJECTED / FW_REJECTED detail; `at`: when it is noticed; `left`: what the device
    holds afterwards; `ru`/`en`: what the phone says (the client has the same strings); `do`: what the person does next."""

    channel: str
    key: str
    at: str
    left: str
    ru: str
    en: str
    do: str


REASONS = (
    # --- channel A (the keys are those of FirmwareError)
    Reason("A", "signature", "в конце передачи, до укладки", "ничего не записано", "Файл не подписан производителем.", "The file is not signed by the manufacturer.",
           "взять файл с сайта производителя"),
    Reason("A", "hardware", "в конце передачи", "ничего не записано", "Это обновление сделано для другого устройства.", "This update was made for another device.",
           "взять файл для своей модели"),
    Reason("A", "rollback", "в конце передачи", "ничего не записано", "Эта версия старше допустимой.", "This version is older than what is allowed.",
           "ставить более новую"),
    Reason("A", "same", "в конце передачи", "ничего не записано", "Эта версия уже установлена.", "This version is already installed.", "ничего"),
    Reason("A", "too_large", "в начале или в ходе передачи", "ничего не записано", "Файл слишком большой для устройства.",
           "The file is too large for the device.", "проверить файл"),
    Reason("A", "damaged", "в конце передачи", "ничего не записано; передача сброшена", "Файл повреждён или передан с ошибкой.",
           "The file is damaged or was sent with an error.", "загрузить заново"),
    Reason("A", "wrong_channel", "в первом куске", "ничего не записано", "Это не обновление системы: это файл с личными данными.",
           "This is not a system update: it is a file with personal data.", "загрузить его как пакет"),
    Reason("A", "no_trial", "в начале", "прежнее обновление ждёт проверки", "Сначала дождитесь проверки предыдущего обновления.",
           "Wait for the previous update to be checked first.", "подождать или применить / вернуть прежнюю"),
    Reason("A", "sequence", "в ходе передачи", "часть принята, продолжение с места обрыва", "Передача сбилась.", "The transfer got out of step.",
           "повторить: телефон продолжит с места обрыва"),
    Reason("A", "interrupted", "обрыв питания при записи банка", "второй банк пуст; работающий не тронут",
           "(устройство запустится на прежней версии)", "(the device starts on the previous version)", "загрузить обновление заново"),
    Reason("A", "unsupported", "в начале", "ничего", "Это устройство не принимает обновления с телефона.",
           "This device does not take updates from a phone.", "обновить через утилиту поддержки"),
    Reason("A", "no_previous", "при возврате", "ничего", "Другой версии на устройстве нет.", "There is no other version on the device.", "ничего"),
    # --- channel B
    Reason("B", "wrong_channel", "в первом куске", "ничего не принято", "Это обновление системы, а не пакет.", "This is a system update, not a package.",
           "загрузить его как обновление системы"),
    Reason("B", "damaged", "в заголовке или в конце", "ничего не принято; передача сброшена", "Файл повреждён или передан с ошибкой.",
           "The file is damaged or was sent with an error.", "попросить отправителя прислать заново"),
    Reason("B", "bad_signature", "по заголовку, до загрузки остального", "ничего не принято", "Подпись не сходится: файл изменён или подделан.",
           "The signature does not match: the file was changed or forged.", "не доверять файлу; попросить прислать заново"),
    Reason("B", "wrong_device", "по заголовку, до загрузки остального", "ничего не принято", "Файл сделан для другого устройства.",
           "The file was made for another device.", "попросить отправителя сделать файл для этой карточки"),
    Reason("B", "own_file", "по заголовку, до загрузки остального", "ничего не принято", "Файл сделан самим этим устройством, а оно файлов не делает.",
           "The file was made by this very device, and the device makes no files.", "не доверять файлу; ключи могли быть украдены"),
    Reason("B", "replay", "по заголовку", "ничего не принято", "Такой файл уже был принят. Повтор отвергнут.",
           "This file was already accepted. The repeat was refused.", "ничего"),
    Reason("B", "needs_update", "по заголовку", "ничего не принято", "Пакету нужна более новая версия системы. Сначала обновите систему.",
           "The package needs a newer system version. Update the system first.", "обновить систему (канал A), затем повторить"),
    Reason("B", "unsupported", "по заголовку", "ничего не принято", "В пакете есть то, чего это устройство не понимает.",
           "The package holds something this device does not understand.", "обновить систему или попросить другой пакет"),
    Reason("B", "version", "при проверке частей", "ничего не принято", "Профиль внутри новее, чем понимает устройство.",
           "The profile inside is newer than the device understands.", "обновить систему"),
    Reason("B", "too_large", "по заголовку или в ходе", "ничего не принято", "Пакет слишком большой для устройства.",
           "The package is too large for the device.", "взять модель поменьше"),
    Reason("B", "sequence", "в ходе передачи", "часть принята, продолжение с места обрыва", "Передача сбилась.", "The transfer got out of step.",
           "повторить: телефон продолжит с места обрыва"),
    Reason("B", "tampered", "в ходе передачи", "ничего не принято; принятое стёрто", "Файл изменён по дороге: часть не прошла проверку.",
           "The file was changed on the way: a part failed its check.", "загрузить заново, лучше другим путём"),
    Reason("B", "truncated", "в конце передачи", "ничего не принято; принятое стёрто", "Файл оборван: конца нет.", "The file is cut short: its end is missing.",
           "загрузить заново"),
    Reason("B", "hash", "в конце передачи", "ничего не принято; принятое стёрто", "Содержимое не совпало с подписанным.",
           "The content does not match what was signed.", "загрузить заново"),
    Reason("B", "bad_part", "в конце передачи", "ничего не принято", "Часть пакета не годится: профиль, уровни или раскладка повреждены.",
           "A part of the package is not usable: the profile, levels or layout is damaged.", "попросить отправителя пересобрать пакет"),
    Reason("B", "not_ui_model", "в конце передачи", "ничего не принято", "Это не модель для кнопок и значков интерфейса. Устройство её не берёт.",
           "This is not a model for buttons and icons of the interface. The device does not take it.", "взять модель интерфейса"),
    Reason("B", "bad_model", "в конце передачи", "ничего не принято", "Модель не прошла проверку (формат, операторы, размер).",
           "The model did not pass the check (format, operators, size).", "взять другую сборку модели"),
    Reason("B", "no_pending", "при применении", "ничего", "Нет загруженного пакета для применения.", "There is no loaded package to apply.",
           "сначала загрузить пакет"),
    Reason("B", "no_previous", "при возврате", "ничего", "Прежней модели в этом слоте нет.", "There is no previous model in this slot.", "ничего"),
    Reason("B", "unsupported_device", "в начале", "ничего", "Это устройство не принимает пакеты.", "This device does not take packages.", "обновить систему"),
)
REASON_KEYS = {c: {r.key for r in REASONS if r.channel == c} for c in "AB"}


class UpdateError(ValueError):
    """`channel` + `key`: one row of REASONS."""

    def __init__(self, channel: str, key: str, msg: str = "") -> None:
        super().__init__(msg or key)
        self.channel, self.key = channel, key


def reason(channel: str, key: str) -> Reason:
    for r in REASONS:
        if r.channel == channel and r.key == key:
            return r
    raise KeyError((channel, key))


def check_compat(schema: int, min_fw: int, supported_schema: int, fw_version: int) -> None:
    """A package is refused up front (from its signed header) if it needs a newer package schema or a newer system than this device runs."""
    if schema > supported_schema or min_fw > fw_version:
        raise UpdateError("B", "needs_update", f"schema {schema} (have {supported_schema}), needs system {min_fw} (have {fw_version})")
