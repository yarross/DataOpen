"""The fail-safe net of the adapter: when the mouse and the video are connected straight through, and how soon (docs/HARDWARE.md section 5).

The rule is the one in docs/BRIDGE.md (`bridge.sim.HwModel`), extended with the power input and the video relays:

    gate        = VEXT_OK and WDG_OK and PANIC_LOOP_OK and MODE_ASSIST            (hardware only: no processor in these four)
    mouse path  = gate and MCU_ENGAGE       -> K1, K2 coils energised = the mouse goes through the bridge MCU
    video path  = gate and VIDEO_EN and VIDEO_PWR_OK     -> KV coils energised = the monitor sees the card through the splitter
    anything else: coils OFF; the contacts are normally closed, so mouse and monitor are wired straight to the PC and the card.

The MCU and the SoM can only REQUEST (`MCU_ENGAGE`, `VIDEO_EN`); they can never energise a coil by themselves. This is a model of the
specified logic, not of a schematic: the schematic has to implement it (a test over `spec.LINKS` checks that no other path to a coil is
specified).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, fields

from . import spec as S

WDG_MS = 100                               # docs/BRIDGE.md: external window watchdog, fed only by a healthy bridge main loop
PANIC_RC_MS = 3000                         # docs/BRIDGE.md: NC loop through an RC; an open loop for this long drops PANIC_LOOP_OK
# ASSUMPTION: signal relay release time plus coil flyback decay (a datasheet figure to be verified)
RELEASE_MS = 8
LINK_TTL_MS = 500                          # docs/BRIDGE.md: no BridgeLink frame this long = the bridge stops assisting (STALE_LINK)
REENUM_MS = (300, 2000)                    # the PC re-enumerates the mouse after a bypass: 0.2 - 2 s in real hosts (docs/BRIDGE.md)
MONITOR_RELOCK_MS = (500, 3000)            # the monitor re-locks after the video path moves (an assumption, not measured)


@dataclass(frozen=True)
class Inputs:
    vext_ok: bool = True
    wdg_ok: bool = True
    panic_loop_ok: bool = True
    mode_assist: bool = True               # the MODE slide switch is on ASSIST
    mcu_engage: bool = False               # the MCU asks for the mouse path
    video_en: bool = False                 # the MCU (on the SoM's request) asks for the video path
    video_pwr_ok: bool = True              # D3 is powered and good


@dataclass(frozen=True)
class Outputs:
    mouse_engaged: bool                    # K1 + K2 coils energised
    video_engaged: bool                    # KV coils energised
    bypass_led: bool                       # the amber hardware LED: powered, and the mouse path is direct


def gate(i: Inputs) -> bool:
    return i.vext_ok and i.wdg_ok and i.panic_loop_ok and i.mode_assist


def evaluate(i: Inputs) -> Outputs:
    g = gate(i)
    mouse = g and i.mcu_engage
    return Outputs(mouse, g and i.video_en and i.video_pwr_ok, i.vext_ok and not mouse)


def all_inputs():
    names = [f.name for f in fields(Inputs)]
    for bits in itertools.product((False, True), repeat=len(names)):
        yield Inputs(**dict(zip(names, bits)))


# ---------------------------------------------------------------------------------------------------------------------------- scenarios
@dataclass(frozen=True)
class Scenario:
    key: str
    event: str
    hardware: str                          # what the net does
    to_bypass_ms: object                   # time from the event to released contacts (None: the contacts do not move)
    needs_firmware: bool                   # does the protection depend on any firmware being alive?
    user_sees: str


SCENARIOS = (
    Scenario("no_power", "внешнее питание пропало или просело ниже порога", "катушки обесточены, K1/K2/KV отпускаются", RELEASE_MS, False,
             "светодиоды гаснут; мышь и экран работают напрямую; мышь перенумеруется 0,3-2 с, экран перезахватывает 0,5-3 с"),
    Scenario("panic_short", "короткое нажатие Panic", "контакты не двигаются; мост сам выключает помощь (мягкая защёлка)", None, True,
             "указатель не пропадает ни на миг; помощи нет; светодиод показывает «выключено по вашему выбору»"),
    Scenario("panic_long", "удержание Panic около 3 с", "петля Panic размыкается, RC роняет PANIC_LOOP_OK, катушки обесточены", PANIC_RC_MS + RELEASE_MS, False,
             "через 3 с мышь и экран напрямую, янтарный светодиод; работает и при мёртвой прошивке"),
    Scenario("mcu_hang", "зависание прошивки моста", "перестают приходить кики, WDG_OK падает, катушки обесточены", WDG_MS + RELEASE_MS, False,
             "через десятую долю секунды мышь напрямую, янтарный светодиод"),
    Scenario("som_hang", "зависание SoM или потеря BridgeLink", "аппаратура не меняется: мост через 0,5 с перестаёт применять помощь и снимает VIDEO_EN", None, True,
             "мышь идёт через мост без помощи; видео возвращается в прямой bypass; светодиод «модуль не отвечает»"),
    Scenario("mode_switch", "ползунок MODE в положение BYPASS", "MODE_ASSIST падает, катушки обесточены", RELEASE_MS, False,
             "мгновенно напрямую, янтарный светодиод; возврат только рукой"),
    Scenario("cable_cut", "обрыв кабеля внешней кнопки (или вынут штекер)", "петля Panic разомкнута как при удержании", PANIC_RC_MS + RELEASE_MS, False,
             "через 3 с напрямую; безопасная сторона отказа"),
)


def scenario(key: str) -> Scenario:
    return next(s for s in SCENARIOS if s.key == key)


def reenumeration_note() -> str:
    return f"{REENUM_MS[0] / 1000:g}-{REENUM_MS[1] / 1000:g} с на стороне ПК; монитор {MONITOR_RELOCK_MS[0] / 1000:g}-{MONITOR_RELOCK_MS[1] / 1000:g} с"


# --------------------------------------------------------------------------------------------------------------------- structure checks
COILS = ("K1", "K2", "KV")


def coil_sources() -> dict[str, set[str]]:
    """For every block that touches a relay coil: who drives it, according to the specified links."""
    out: dict[str, set[str]] = {}
    for src, dst, what in S.LINKS:
        if dst == "COIL_DRV" or (dst in COILS and "coil" in what):          # signal links through the contacts are not drive links
            out.setdefault(dst, set()).add(src)
    return out


def firmware_blocks() -> set[str]:
    return {"MCU", "SOM", "BLE", "VIDEO_TAP", "PHONE", "BTN_SOFT"}
