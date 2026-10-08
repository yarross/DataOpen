"""What the person sees and hears: one RGB status LED, four slot LEDs, a quiet buzzer, and a hardware-driven amber BYPASS LED
(docs/HARDWARE.md section 6).

A state is told apart by the SHAPE of its blinking, not by colour alone (colour-blind users, bright rooms, a mixed-up label). The slot LEDs
are the ones in `ctl/panel.py`; this module adds the status LED and the buzzer on top of them. Nothing here has been tried on a person.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..bridge.cbridge import REASONS
from ..ctl import protocol as P


@dataclass(frozen=True)
class State:
    key: str
    color: str                             # '' = dark
    pattern: tuple[tuple[int, int], ...]   # cycle of (on 0/1, milliseconds)
    breathe: bool = False                  # smooth ramp instead of hard steps
    beep: Optional[str] = None             # sound on ENTERING the state (never repeated while it lasts)
    ru: str = ""                           # what the person sees


def _blinks(n: int, on: int, gap: int, pause: int) -> tuple[tuple[int, int], ...]:
    seq: list[tuple[int, int]] = []
    for _ in range(n):
        seq += [(1, on), (0, gap)]
    seq[-1] = (0, pause)
    return tuple(seq)


STATES = (
    State("OFF", "", ((0, 1000),), ru="ничего не горит: питания нет, мышь и экран подключены напрямую (так и подписано на корпусе)"),
    State("BYPASS", "amber", ((1, 1800), (0, 200)), beep="bypass", ru="жёлтый, почти ровно (короткая отметка раз в 2 с): мышь напрямую; то же без прошивки показывает отдельный жёлтый светодиод, он горит ровно"),
    State("STARTING", "white", ((1, 800), (0, 800)), breathe=True, ru="белый, плавно «дышит»: устройство включается"),
    State("READY", "green", ((1, 100), (0, 1900)), ru="зелёный, короткая вспышка раз в 2 с: всё в порядке, помощь выключена"),
    State("ACTIVE", "green", ((1, 1000),), ru="зелёный, ровно: помощь работает"),
    State("NO_MOUSE", "amber", ((1, 1000), (0, 1000)), ru="жёлтый, медленно мигает: мышь не найдена"),
    State("DEGRADED", "amber", _blinks(2, 150, 150, 1550), ru="жёлтый, две вспышки и пауза: мышь работает, помощи нет (модуль не отвечает, или медленная мышь)"),
    State("CALIBRATING", "blue", ((1, 1500), (0, 1500)), breathe=True, ru="синий, плавно «дышит»: идёт калибровка"),
    State("PAIRING", "blue", _blinks(2, 100, 100, 700), beep="pair", ru="синий, две быстрые вспышки и пауза: можно подключить телефон"),
    State("OTA", "violet", _blinks(3, 120, 120, 800), beep="ota", ru="фиолетовый, три вспышки и пауза: идёт обновление"),
    State("TRIAL", "cyan", ((1, 250), (0, 250)), ru="голубой, быстро мигает: идёт проба, нажмите «Оставить» или «Вернуть»"),
    State("RECOVERY", "magenta", ((1, 600), (0, 150), (1, 100), (0, 150), (1, 100), (0, 900)), beep="recovery",
          ru="пурпурный, одна длинная и две короткие вспышки: система восстановления, мышь и экран напрямую, нажмите CONFIRM, чтобы восстановить прошивку"),
    State("ERROR", "red", ((1, 100), (0, 100)), beep="error", ru="красный, часто мигает и три сигнала: ошибка, мышь при этом напрямую или без помощи"),
)
BY_KEY = {s.key: s for s in STATES}

# sounds: (frequency Hz, milliseconds), 0 Hz = silence. All short and low in volume; none repeats on its own.
BEEPS = {
    "bypass": ((440, 400),),                                     # one long low tone: the device went to the direct path
    "pair": ((1760, 80),),                                       # one short high tone
    "ota": ((880, 80), (0, 60), (1320, 80)),                     # two rising tones: an update is being applied
    "error": ((1320, 90), (0, 90), (1320, 90), (0, 90), (1320, 90)),
    "recovery": ((660, 120), (0, 80), (660, 120), (0, 80), (990, 200)),      # the device entered the recovery system
    "tick": ((2400, 15),),                                       # a button press was accepted
}
MAX_BEEP_MS = 1000

# bridge reason -> what the status LED says. EVERY name in REASONS appears (a test checks it).
REASON_STATE = {
    "NONE": "ACTIVE", "POWER_ON": "STARTING", "NO_DEVICE": "NO_MOUSE", "SETTLING": "STARTING", "PROBING": "STARTING", "IMAGE": "STARTING",
    "PC_TIMEOUT": "ERROR", "USB_ERRORS": "ERROR", "PANIC": "READY", "PANIC_LONG": "BYPASS", "CHORD": "READY", "CMD_PASSTHRU": "READY",
    "CMD_BYPASS": "BYPASS", "STALE_PARAMS": "DEGRADED", "STALE_LINK": "DEGRADED", "INVARIANT": "ERROR", "OVERRUN": "ERROR",
    "SLOW_MOUSE": "DEGRADED", "CRASHLOOP": "ERROR", "FATAL": "ERROR", "ENGAGE_FAILED": "ERROR", "WAIT_PC": "STARTING", "IMAGE_TIMEOUT": "ERROR",
}
PRIORITY = ("OFF", "RECOVERY", "ERROR", "BYPASS", "OTA", "PAIRING", "CALIBRATING", "TRIAL", "NO_MOUSE", "DEGRADED", "STARTING", "ACTIVE", "READY")


def indicate(status: Optional[P.StatusSnapshot], *, powered: bool = True, upload: bool = False, fw_state: str = "current",
             pairing: bool = False, both_mice: bool = False, recovery: bool = False) -> str:
    """The state key for the status LED. `status` is the 20-byte snapshot the phone also sees; the rest is what the gateway knows."""
    if not powered:
        return "OFF"
    if recovery:                                                 # the recovery system is running: that is what the person needs to know
        return "RECOVERY"
    if both_mice:
        return "ERROR"
    if status is None or not status.bridge or status.mode == P.MODE_UNKNOWN:
        return "STARTING"
    base = REASON_STATE.get(status.reason_name, "ERROR")
    if base == "ERROR":
        return "ERROR"
    if status.mode == 0:                                         # HW_BYPASS whatever the reason: the direct path is what matters
        return "BYPASS" if base != "NO_MOUSE" else "NO_MOUSE"
    wants = [base]
    if upload or fw_state == "trial":
        wants.append("OTA")
    if pairing:
        wants.append("PAIRING")
    if status.flags & P.SF_CALIBRATING:
        wants.append("CALIBRATING")
    if status.flags & P.SF_TRIAL:
        wants.append("TRIAL")
    if base == "ACTIVE" and not (status.mode == 3 and status.ready & (P.RB_ASC | P.RB_TREMOR)):
        wants[0] = "READY"                                       # help is on, but there is nothing to correct yet / mode is not ASSIST
    return min(wants, key=PRIORITY.index)


def reasons_covered() -> bool:
    return set(REASON_STATE) == set(REASONS)
